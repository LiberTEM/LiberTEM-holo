"""Phase unwrapping by ADMM: weighted L1 (Huber) minimization with cosine transforms.

An alternative to :func:`~libertem_holo.base.unwrap.unwrap_phase_irls`
with the same inputs, results and stop rule: see :func:`unwrap_phase_admm`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numba
import numpy as np
import scipy.fft

from . import _admm_numba, _irls_numba
from .irls import (
    TAU,
    L1BranchesPrior,
    L1PhasePrior,
    L1Result,
    L1Round,
    L1Status,
    _Candidate,
    _check_options,
    _Edges,
    _Prepared,
    _restore_invalid,
    _run,
    _Solution,
    _start_and_rounds,
)

if TYPE_CHECKING:
    from libertem_holo.base.workspace import Workspace

    from .irls import IRLSOptions


@dataclass(frozen=True)
class ADMMOptions:
    """Settings of an ADMM unwrap; see :func:`unwrap_phase_admm`.

    Parameters
    ----------
    rounds
        Maximum number of rounds. Epsilon decays geometrically from
        :code:`epsilon_start` to :code:`epsilon_min` over them.
    iterations_per_round
        Number of ADMM iterations per round, an even number. After each round,
        the iterate is rounded onto the branches and scored.
    epsilon_start
        The Huber scale (in radians) of the first round: gradient residuals well
        below it are smoothed quadratically, larger ones count linearly.
    epsilon_min
        The Huber scale of the last round.
    rho_scale
        The ADMM penalty is :code:`rho_scale * median(C) / pi`, for the edge
        weights :math:`C`. Larger values stay closer to the smooth solution.
    stop_patience
        Stop once the cut cost has not improved for this many rounds.
        :code:`None` runs all :code:`rounds`.
    stop_tolerance
        A round counts as an improvement only if it lowers the cut cost by more
        than this fraction of it, and by more than half a cut through edges of
        full weight.
    cold_start
        Where a solve without a prior starts, as in
        :class:`~libertem_holo.base.unwrap.IRLSOptions`.

    """

    rounds: int = 30
    iterations_per_round: int = 6
    epsilon_start: float = 0.1
    epsilon_min: float = 1e-3
    rho_scale: float = 1.5
    stop_patience: int | None = 2
    stop_tolerance: float = 0.01
    cold_start: Literal["zero", "least_squares"] = "zero"

    method: ClassVar[str] = "admm"

    def __post_init__(self) -> None:
        """Check that the options describe a valid solve."""
        if self.rounds < 1:
            msg = f"rounds must be at least 1, got {self.rounds}"
            raise ValueError(msg)
        if self.iterations_per_round < 2 or self.iterations_per_round % 2:  # noqa: PLR2004
            msg = (
                "iterations_per_round must be even and at least 2, "
                f"got {self.iterations_per_round}"
            )
            raise ValueError(msg)
        if not self.epsilon_min > 0:
            msg = f"epsilon_min must be greater than zero, got {self.epsilon_min}"
            raise ValueError(msg)
        if not self.epsilon_start >= self.epsilon_min:
            msg = (
                f"epsilon_start ({self.epsilon_start}) must be at least "
                f"epsilon_min ({self.epsilon_min})"
            )
            raise ValueError(msg)
        if not self.rho_scale > 0:
            msg = f"rho_scale must be greater than zero, got {self.rho_scale}"
            raise ValueError(msg)
        if self.cold_start not in ("zero", "least_squares"):
            msg = (
                'cold_start must be "zero" or "least_squares", '
                f"got {self.cold_start!r}"
            )
            raise ValueError(msg)
        if not 0 <= self.stop_tolerance < 1:
            msg = f"stop_tolerance must be in [0, 1), got {self.stop_tolerance}"
            raise ValueError(msg)
        if self.stop_patience is not None and self.stop_patience < 1:
            msg = f"stop_patience must be at least 1 or None, got {self.stop_patience}"
            raise ValueError(msg)

    @property
    def decay(self) -> float:
        """The factor by which epsilon shrinks from one round to the next."""
        return (self.epsilon_min / self.epsilon_start) ** (1.0 / self.rounds)


def _rho(edges: np.ndarray | None, scale: float) -> float:
    """Return the ADMM penalty: `scale` times the median edge weight, over pi.

    The median of all `edges` (None: all ones) by a parallel selection, as the
    GPU takes it: interpolated halfway between the middle values, in float64.
    """
    if edges is None or not edges.size:
        return scale / math.pi
    n = edges.size
    a, b, _ = _irls_numba.select_pair(edges, (n - 1) // 2, n // 2)
    return scale * (b - (b - a) * 0.5) / math.pi


class _AdmmProblem(_Edges):
    """The state of an ADMM solve on the CPU, a sibling of `irls._IrlsProblem`.

    The u-step is the unweighted least-squares fit to the edge targets, which the
    exact (unpadded) Neumann Poisson solve by cosine transforms does in one go.
    """

    def __init__(
        self,
        wrapped: np.ndarray,
        workers: int | None,
        edge_weights: tuple[np.ndarray, np.ndarray] | None = None,
        options: ADMMOptions | None = None,
        *,
        invalid: np.ndarray | None = None,
    ) -> None:
        # The u-step must solve its least-squares problem exactly: on a padded
        # grid, the padding's edges would pull on the result.
        super().__init__(wrapped, workers, edge_weights, pad=False, invalid=invalid)
        options = options or ADMMOptions()
        all_ones = edge_weights is None and invalid is None
        self.rho = _rho(None if all_ones else self.edges, options.rho_scale)
        rows, cols = wrapped.shape
        self.duals = tuple(
            (np.zeros((rows, cols - 1), dtype=wrapped.dtype),
             np.zeros((rows - 1, cols), dtype=wrapped.dtype))
            for _ in range(2)
        )
        # The right-hand side of the u-step, which the cosine transforms turn
        # into the next u in place: two buffers, for the u that the next step
        # reads and the right-hand side that it writes.
        self._rhs = tuple(np.empty((rows, cols), dtype=wrapped.dtype) for _ in range(2))
        self._row_sin = np.empty(rows, dtype=np.float64)
        self._row_cos = np.empty(rows, dtype=np.float64)

    def congruent(
        self, phi: np.ndarray, wrapped: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Snap `phi` onto the branches of `wrapped`, around the circular mean.

        As `irls._congruent`, but the offset removed before rounding is the
        circular mean of wrap(phi - wrapped) instead of its median: one pass.
        """
        tau = phi.dtype.type(TAU)
        _admm_numba.circular_sums(phi, wrapped, tau, self._row_sin, self._row_cos)
        offset = phi.dtype.type(math.atan2(self._row_sin.sum(), self._row_cos.sum()))
        branches = np.empty(phi.shape, dtype=np.int32)
        phase = np.empty(phi.shape, dtype=np.float64)
        _irls_numba.branches_and_phase(phi, wrapped, offset, tau, branches, phase)
        return phase, branches

    def run_round(
        self,
        phi: np.ndarray,
        wrapped: np.ndarray,
        *,
        epsilon: float,
        at_end: bool,  # noqa: ARG002 (ADMM rounds always iterate)
        options: IRLSOptions | ADMMOptions,
        index: int,
    ) -> tuple[L1Round, _Candidate]:
        """Run one round of ADMM iterations on `phi` (in place).

        Returns the round's report and its candidate result.
        """
        if index == 0:  # a new solve: the duals start at zero
            for lam in self.duals:
                for x in lam:
                    x[...] = 0
        # From epsilon squared, as on the GPU (which gets it in `round_params`),
        # so that both use the same delta:
        delta = phi.dtype.type(math.sqrt(epsilon * epsilon))
        u = phi
        workers = self.precondition.workers
        blocks = 4 * numba.get_num_threads()
        for it in range(options.iterations_per_round):
            (lx, ly), (lx_new, ly_new) = self.duals[it % 2], self.duals[1 - it % 2]
            rhs = self._rhs[it % 2]
            _admm_numba.admm_update_rows(
                u, self.target_x, self.target_y, self.edge_x, self.edge_y,
                lx, ly, lx_new, ly_new, self.rho, delta, rhs, blocks,
            )
            # The u-step: the exact Neumann Poisson solve (see
            # `irls._PoissonPreconditioner`, unpadded). With `overwrite_x`, the
            # transforms work in `rhs`, which becomes the next u.
            spectrum = scipy.fft.dctn(
                rhs, type=2, norm="ortho", workers=workers, overwrite_x=True,
            )
            _admm_numba.scale_inplace(spectrum, self.precondition.inverse_symbol)
            u = scipy.fft.idctn(
                spectrum, type=2, norm="ortho", workers=workers, overwrite_x=True,
            )
        phi[...] = u
        phase, branches = self.congruent(phi, wrapped)
        turns, cost = self.cuts(branches)
        report = L1Round(
            iterations=options.iterations_per_round, residuals=[], converged=True,
        )
        # The weights only matter for the result: see `release`.
        candidate = _Candidate(
            phase=phase, branches=branches, cut_turns=turns, cut_cost=cost,
            weights_x=None, weights_y=None,
            weights_epsilon=math.nan, converged=True, round_index=index,
        )
        return report, candidate

    def release(self, candidate: _Candidate) -> _Candidate:
        """Return the result: a round's candidate gets its weights here.

        The edge weights times the Charbonnier weights of its phase at its
        epsilon, as the GPU computes them for the selected candidate.
        """
        if candidate.weights_x is None:
            candidate.weights_x = np.empty_like(self.weights_x)
            candidate.weights_y = np.empty_like(self.weights_y)
            _admm_numba.weights_from_phase(
                candidate.phase, self.target_x, self.target_y, self.edge_x,
                self.edge_y, candidate.weights_epsilon ** 2,
                candidate.weights_x, candidate.weights_y,
            )
        return candidate


def _admm_status(rounds: list[L1Round], options: ADMMOptions) -> L1Status:
    """Return the status of an ADMM solve: whether it stopped before the last round."""
    return "converged" if len(rounds) < options.rounds else "iteration_limit"


class _CpuAdmm:
    """ADMM on the CPU, on `workers` threads (see `irls._cpu_threads`)."""

    def __init__(self, workers: int) -> None:
        self.workers = workers

    def solve(
        self,
        prepared: _Prepared,
        prior: L1PhasePrior | L1BranchesPrior | None,
        options: ADMMOptions,
    ) -> _Solution:
        """See `irls._Solver.solve`."""
        problem = _AdmmProblem(
            prepared.data, self.workers, prepared.edge_weights, options,
            invalid=prepared.invalid,
        )
        rounds, best = _start_and_rounds(problem, prepared.data, prior, options)
        _restore_invalid(best, prepared.invalid)
        return _Solution.of(best, rounds, _admm_status(rounds, options))


def unwrap_phase_admm(
    wrapped: np.ndarray,
    *,
    prior: L1PhasePrior | L1BranchesPrior | L1Result | None = None,
    options: ADMMOptions | None = None,
    workers: int | None = None,
    edge_weights: tuple[np.ndarray, np.ndarray] | None = None,
    workspace: Workspace | None = None,
) -> L1Result:
    r"""Unwrap a 2D phase by ADMM on a weighted, smoothed L1 objective.

    The same inputs and result as
    :func:`~libertem_holo.base.unwrap.unwrap_phase_irls`, which describes them,
    and the same objective: the weighted sum :math:`\sum C |r|` of the gradient
    residuals :math:`r` of the result, smoothed by a Huber function of scale
    :math:`\varepsilon`, which decays over the rounds.

    Instead of reweighted least-squares solves, ADMM splits the residuals off
    into edge variables :math:`z = \nabla u - g`. An iteration is an unweighted
    least-squares fit of :math:`u` to the edge targets (one Neumann Poisson solve,
    a pair of cosine transforms), the proximal step of the Huber function for
    :math:`z` (elementwise; the edge weights only enter here), and a dual update.
    This follows B. Rouet-Leduc and C. Hulbert, "Phase Unwrapping in Seconds: A
    Spectral ADMM Algorithm for Large-Scale InSAR", Remote Sensing 18, 1801
    (2026), https://doi.org/10.3390/rs18111801. Added here are the Huber
    smoothing, and the zero start, rounds and stop rule of
    :func:`~libertem_holo.base.unwrap.unwrap_phase_irls`. Results are close to
    those of IRLS, at a fraction of its iterations.

    The rounds' reports carry the number of iterations, but no CG residuals.
    The status is :code:`"converged"` unless the solve ran all :code:`rounds`.

    Examples
    --------
    >>> import numpy as np
    >>> from libertem_holo.base.unwrap.admm import unwrap_phase_admm
    >>> y, x = np.mgrid[:64, :64]
    >>> truth = 0.3 * x + 0.2 * y
    >>> wrapped = np.angle(np.exp(1j * truth))
    >>> result = unwrap_phase_admm(wrapped)
    >>> np.allclose(result.phase - result.phase[0, 0], truth)
    True

    """
    return _run(
        wrapped, prior, _check_options(options, ADMMOptions), workers, edge_weights,
        workspace=workspace,
    )


__all__ = ["ADMMOptions", "unwrap_phase_admm"]

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
        Where a solve without a prior starts: :code:`"zero"` starts at zero phase,
        :code:`"least_squares"` at the unweighted least-squares (L2) solution.
        Either way, the rounded L2 solution is the first candidate result.

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

    The result is congruent with the input: it differs from it by an integer
    multiple of :math:`2\pi` at every valid pixel. The solve is global and places
    the discontinuities where they cost least.

    The solve runs in float32 for float32 and complex64 input, and in float64
    otherwise. The result phase is float64, computed from the input values, so it
    stays exactly congruent with them.

    This algorithm is recently added and still experimental, use at your own risk.

    Parameters
    ----------
    wrapped
        The wrapped phase, usually in :math:`[-\pi, \pi]`; any real values work,
        since only differences modulo :math:`2\pi` enter the solve. Or the complex
        wave: then its phase is unwrapped, with edge weights from its amplitude
        (:func:`~libertem_holo.base.unwrap.amplitude_edge_weights`) unless
        :code:`edge_weights` is given. For a cupy array, the solve runs on the GPU
        and the result holds cupy arrays. Pixels that are not finite (NaN or
        infinite) are invalid: the solve treats their phase as 0, gives the edges
        that touch them a weight of :code:`1e-3` (where the largest weight is 1),
        and leaves those edges out of the cut turns and cost. The result is NaN
        there, with branch 0.
    prior
        Start from an earlier solution instead of from scratch, and resume its
        epsilon schedule; see :class:`~libertem_holo.base.unwrap.L1PhasePrior` and
        :class:`~libertem_holo.base.unwrap.L1BranchesPrior`. Passing an
        :class:`~libertem_holo.base.unwrap.L1Result` is the same as passing
        :meth:`L1Result.as_prior() <libertem_holo.base.unwrap.L1Result.as_prior>`.
        A warm start from the previous frame of a series typically needs fewer
        rounds than a cold start.
    options
        Solver settings, :class:`ADMMOptions` by default.
    workers
        Number of threads. By default, the number of CPUs, but at most 16.
        The result does not depend on it. Not used on the GPU.
    edge_weights
        Weights :math:`C` of the horizontal and vertical edges between
        neighbouring pixels, with shapes :code:`(rows, cols - 1)` and
        :code:`(rows - 1, cols)`, finite and greater than zero. Edges with a low
        weight are cheap to cut, so the solve places discontinuities there. Only
        relative weights matter. Without edge weights, all edges weigh the same.
    workspace
        Where a solve on the GPU keeps its buffers, FFT plans and CUDA graphs for
        the next solve of the same shape (see
        :mod:`libertem_holo.base.workspace`). By default, the default workspace,
        which solves from several threads use one at a time. Not used on the CPU.

    Returns
    -------
    L1Result
        The unwrapped phase, the branches, the weights of the selected iterate
        (the edge weights times the Charbonnier weights
        :math:`\varepsilon^2 / \sqrt{r^2 + \varepsilon^2}` of its gradient residuals
        :math:`r` at its :math:`\varepsilon`), the cut turns and cut cost, and a
        report of each round. The reports give the number of ADMM iterations, but
        no CG residuals. The status is :code:`"converged"` if the solve stopped
        before running all :code:`rounds`, :code:`"iteration_limit"` otherwise.

    See Also
    --------
    libertem_holo.base.unwrap.unwrap_phase_irls : The same objective, minimized by
        iteratively reweighted least squares.

    Notes
    -----
    A plain least-squares (L2) unwrap, like
    :func:`~libertem_holo.base.unwrap.unwrap_phase_laplacian`, smears a genuine
    discontinuity or the step around a residue across the image. This solve
    minimizes the weighted sum :math:`\sum C |r|` of the gradient residuals
    :math:`r` of the result instead, where :math:`r` is the difference between the
    gradient of the result and the wrapped gradient of the input. The absolute
    value is smoothed by a Huber function of scale :math:`\varepsilon`, which
    decays geometrically from :code:`epsilon_start` to :code:`epsilon_min` over
    the rounds: the solve starts smooth and then sharpens.

    ADMM splits the residuals off into edge variables :math:`z = \nabla u - g`,
    for the wrapped gradient :math:`g`. An iteration is an unweighted
    least-squares fit of :math:`u` to the edge targets (one Neumann Poisson solve,
    a pair of cosine transforms), the proximal step of the Huber function for
    :math:`z` (elementwise; the edge weights only enter here), and a dual update.
    This follows B. Rouet-Leduc and C. Hulbert, "Phase Unwrapping in Seconds: A
    Spectral ADMM Algorithm for Large-Scale InSAR", Remote Sensing 18, 1801
    (2026), https://doi.org/10.3390/rs18111801. The Huber smoothing, the rounds
    and the stop rule below are additions to it. Results are close to those of
    :func:`~libertem_holo.base.unwrap.unwrap_phase_irls`, at a fraction of its
    iterations.

    By default (:code:`cold_start="zero"`), the solve starts at zero phase;
    with :code:`cold_start="least_squares"`, at the L2 solution, which smears
    discontinuities and residues. Either way, the rounded L2 solution is the first
    candidate result.

    After each round of :code:`iterations_per_round` iterations, the iterate is
    snapped onto the branches of the input and scored. Its *cut turns* are the
    number of whole turns by which its gradient differs from the wrapped gradient,
    summed over all edges between neighbouring pixels: the L1 objective of a
    congruent result. Its *cut cost* weights each edge by :math:`C`; without edge
    weights, the two are equal. The solve returns the latest rounded iterate with
    the lowest cut cost. The starting point (the rounded L2 solution, or the
    prior) is a candidate too, so the result is never worse than that. The solve
    stops once the cut cost has not improved for :code:`stop_patience` rounds,
    where an improvement must exceed both :code:`stop_tolerance` of the cost and
    half a cut through edges of full weight, so that tiny gains in nearly
    weightless regions do not keep it going. It also stops as soon as there are
    no cut turns, since no result can be better.

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

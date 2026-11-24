"""Phase unwrapping by iteratively reweighted least squares (IRLS).

See :func:`unwrap_phase_irls` for how it works.
"""
from __future__ import annotations

import contextlib
import functools
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeVar

import numba
import numpy as np
import scipy.fft
from libertem.common.threading import set_num_threads, set_numba_threads

from libertem_holo.base.align import _array_module
from libertem_holo.base.workspace import Workspace, default_workspace

from . import _irls_numba

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .admm import ADMMOptions

TAU = 2 * np.pi
_OptionsT = TypeVar("_OptionsT")
# The options of one method, and what takes them (`_RoundProblem`, `_Solver`):
_Options = TypeVar("_Options", bound="IRLSOptions | ADMMOptions")
_OptionsIn_contra = TypeVar(
    "_OptionsIn_contra", bound="IRLSOptions | ADMMOptions", contravariant=True,
)

# On a 72-core GH200, 16 threads made IRLS 8-22 % and ADMM 22-39 % faster than
# 8, from ~500^2 to 2048^2 images; 32 were slower again at ~500^2 (the
# per-iteration work is too small to spread wider). An earlier measurement on a
# 24-core machine found IRLS slower with more than 8.
_DEFAULT_MAX_THREADS = 16

L1Status = Literal["converged", "iteration_limit"]


@dataclass(frozen=True)
class IRLSOptions:
    """Settings of an IRLS unwrap.

    No default fits every phase image: a difficult one needs more rounds or more
    iterations. See :func:`unwrap_phase_irls` for what the options control.

    Parameters
    ----------
    rounds
        Maximum number of reweighting rounds. Epsilon decays geometrically from
        :code:`epsilon_start` to :code:`epsilon_min` over them.
    iterations_per_round
        Maximum number of conjugate gradient (CG) iterations per round.
    epsilon_start
        The Charbonnier scale (in radians) of the first round: differences between
        the current and the wrapped gradient well below it count as smooth, larger
        ones as discontinuities.
    epsilon_min
        The Charbonnier scale of the last round.
    cg_relative_tolerance
        A round's CG stops once the residual norm has fallen to this fraction of
        the norm of the right-hand side.
    stop_patience
        Stop once the cut cost (see :func:`unwrap_phase_irls`) of the rounded
        result has not improved for this many rounds. :code:`None` runs all
        :code:`rounds`.
    stop_tolerance
        A round counts as an improvement only if it lowers the cut cost by more
        than this fraction of it, and by more than half a cut through edges of
        full weight.
    cold_start
        Where a solve without a prior starts: :code:`"zero"` starts at zero phase,
        with weights derived from it, so that edges with a large wrapped gradient
        start out cheap to cut. :code:`"least_squares"` starts at the unweighted
        least-squares (L2) solution. Either way, the rounded L2 solution is the
        first candidate result.

    """

    rounds: int = 30
    iterations_per_round: int = 15
    epsilon_start: float = 0.1
    epsilon_min: float = 1e-3
    cg_relative_tolerance: float = 1e-5
    stop_patience: int | None = 3
    stop_tolerance: float = 0.01
    cold_start: Literal["zero", "least_squares"] = "zero"

    method: ClassVar[str] = "irls"

    def __post_init__(self) -> None:
        """Check that the options describe a valid solve."""
        if self.rounds < 1:
            msg = f"rounds must be at least 1, got {self.rounds}"
            raise ValueError(msg)
        if self.iterations_per_round < 1:
            msg = (
                "iterations_per_round must be at least 1, "
                f"got {self.iterations_per_round}"
            )
            raise ValueError(msg)
        # `not x > 0` instead of `x <= 0`, so that NaN is rejected too:
        if not self.cg_relative_tolerance > 0:
            msg = (
                "cg_relative_tolerance must be greater than zero, "
                f"got {self.cg_relative_tolerance}"
            )
            raise ValueError(msg)
        if not self.epsilon_min > 0:
            msg = f"epsilon_min must be greater than zero, got {self.epsilon_min}"
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
        if not self.epsilon_start >= self.epsilon_min:
            msg = (
                f"epsilon_start ({self.epsilon_start}) must be at least "
                f"epsilon_min ({self.epsilon_min})"
            )
            raise ValueError(msg)

    @property
    def decay(self) -> float:
        """The factor by which epsilon shrinks from one round to the next."""
        return (self.epsilon_min / self.epsilon_start) ** (1.0 / self.rounds)


@dataclass
class L1Round:
    """How one round of a solve ended.

    Attributes
    ----------
    iterations
        Number of iterations run: CG iterations for IRLS, ADMM iterations for ADMM.
    residuals
        The relative CG residual before the first iteration and after each one, so
        :code:`iterations + 1` entries; a single :code:`0.0` if the right-hand side
        was zero. Empty for ADMM.
    converged
        Whether CG reached :code:`cg_relative_tolerance` within
        :code:`iterations_per_round`. Always :code:`True` for ADMM.
    cut_turns, cut_cost
        The cut turns and cut cost of the round's iterate, rounded onto the
        branches of the input (see :func:`unwrap_phase_irls`).

    """

    iterations: int
    residuals: list[float] = field(default_factory=list)
    converged: bool = False
    cut_turns: int | None = None
    cut_cost: float | None = None


@dataclass(frozen=True)
class L1PhasePrior:
    r"""Start a solve from the unwrapped phase of a similar input.

    For example, the result phase of the previous frame of a series, or of the
    same field before more frames were averaged into it. Each pixel is moved to the
    branch of the new wrapped value that is nearest to the prior, so a pixel whose
    wrapped value crossed :math:`\pm\pi` between the two inputs ends up on the
    right side.

    Parameters
    ----------
    phase
        The unwrapped phase, with the same shape as the wrapped input.
    epsilon
        Where the prior's epsilon schedule stopped: its
        :attr:`L1Result.resume_epsilon`. The solve resumes the schedule there.

    """

    phase: np.ndarray
    epsilon: float


@dataclass(frozen=True)
class L1BranchesPrior:
    r"""Start a solve from the branches of an earlier solve of the same input.

    For example, to solve the same wrapped phase again with other options. Against a
    different wrapped input, the branches are off by one wherever a value crossed
    :math:`\pm\pi`, which plants :math:`2\pi` spikes in the starting phase and
    slows the solve down badly; use :class:`L1PhasePrior` there.

    Parameters
    ----------
    branches
        The integer branch field, :attr:`L1Result.branches` of the earlier solve.
    epsilon
        :attr:`L1Result.resume_epsilon` of the earlier solve.

    """

    branches: np.ndarray
    epsilon: float


@dataclass
class L1Result:
    r"""The result of :func:`unwrap_phase_irls` and :func:`unwrap_phase_admm`.

    Attributes
    ----------
    phase
        The unwrapped phase as float64. It equals
        :code:`wrapped + 2 * np.pi * branches` exactly, and is NaN at invalid
        pixels.
    branches
        The number of :math:`2\pi` turns added to each pixel of the input, as int32;
        0 at invalid pixels.
    weights_x, weights_y
        The weights of the selected iterate: the edge weights times the
        Charbonnier weights at its epsilon, with shapes :code:`(rows, cols - 1)`
        and :code:`(rows - 1, cols)`. Low weights mark where the solve placed a
        discontinuity. If the starting point is selected without a prior, these
        are the edge weights.
    rounds
        One :class:`L1Round` per round that ran. Empty if the starting point
        already had no cuts.
    selected_round
        Index into :attr:`rounds` of the round whose iterate is returned: the
        latest one with the lowest cut cost. -1 for the starting point: the rounded
        L2 solution, or the prior.
    cut_turns
        The number of whole turns by which the gradient of :attr:`phase` differs
        from the wrapped gradient, summed over all edges (see
        :func:`unwrap_phase_irls`).
    cut_cost
        The cut turns with each edge weighted by its edge weight; equal to
        :attr:`cut_turns` if all edge weights are equal.
    status
        For IRLS, :code:`"converged"` if every round's CG reached the tolerance,
        :code:`"iteration_limit"` if at least one round stopped at
        :code:`iterations_per_round` (for ADMM, see :func:`unwrap_phase_admm`).
        Either way, :attr:`phase` is the selected iterate, snapped onto the
        branches of the input.
    resume_epsilon
        Where a solve started from this result resumes the epsilon schedule (see
        :class:`L1PhasePrior`): the epsilon the returned weights were derived at, or
        one round coarser if the selected round did not converge.

    """

    phase: np.ndarray
    branches: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray
    rounds: list[L1Round]
    selected_round: int
    cut_turns: int
    cut_cost: float
    status: L1Status
    resume_epsilon: float

    @property
    def final_relative_residual(self) -> float:
        """The relative CG residual at the end of the last round, or NaN."""
        if not self.rounds or not self.rounds[-1].residuals:
            return math.nan
        return self.rounds[-1].residuals[-1]

    def as_prior(self) -> L1PhasePrior:
        """Return a :class:`L1PhasePrior` that starts another solve from this result."""
        return L1PhasePrior(phase=self.phase, epsilon=self.resume_epsilon)


def _wrap(value: np.ndarray) -> np.ndarray:
    """Wrap into [-π, π]."""
    return value - np.rint(value / TAU) * TAU


def _is_smooth(value: int, limit: int = 11) -> bool:
    for factor in range(2, limit + 1):
        while value % factor == 0:
            value //= factor
    return value == 1


def _padded_length(length: int) -> int:
    """Return the smallest even 11-smooth number that is at least `length`.

    Even 5-smooth lengths make single-threaded transforms a few percent faster,
    but the solve measured no faster end to end, since 7- or 11-smooth sizes then
    need more padding.
    """
    candidate = max(length, 2)
    while candidate % 2 != 0 or not _is_smooth(candidate):
        candidate += 1
    return candidate


class _PoissonPreconditioner:
    r"""Approximate inverse of the unweighted operator, by cosine transforms.

    Solves :math:`-\Delta z = r` with Neumann boundary conditions: a DCT-II,
    a division by the symbol :math:`4 - 2\cos(\pi u / N) - 2\cos(\pi v / M)`,
    and a DCT-III. With :code:`pad`, the image is zero-padded at the right and the
    bottom to even 11-smooth lengths, because transforms of prime lengths are
    several times slower. The padded solve is no longer the exact Neumann inverse
    of the image's operator, but it is still symmetric positive semi-definite,
    which is all a CG preconditioner has to be.

    The symbol is zero at DC: the operator only fixes the phase up to a constant,
    so the preconditioner leaves the mean at zero.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: np.dtype,
        workers: int | None = None,
        *,
        pad: bool = True,
    ) -> None:
        self.shape = shape
        self.padded_shape = (
            (_padded_length(shape[0]), _padded_length(shape[1])) if pad else shape
        )
        self.workers = workers
        self.inverse_symbol = _inverse_symbol(self.padded_shape, np.dtype(dtype).str)
        self._padded = np.zeros(self.padded_shape, dtype=dtype)

    def __call__(self, residual: np.ndarray) -> np.ndarray:
        rows, cols = self.shape
        # The area outside of the image stays zero:
        self._padded[:rows, :cols] = residual
        spectrum = scipy.fft.dctn(
            self._padded, type=2, norm="ortho", workers=self.workers,
        )
        spectrum *= self.inverse_symbol
        result = scipy.fft.idctn(
            spectrum, type=2, norm="ortho", workers=self.workers, overwrite_x=True,
        )
        return result[:rows, :cols]


@functools.lru_cache(maxsize=4)
def _inverse_symbol(shape: tuple[int, int], dtype: str) -> np.ndarray:
    """Return the inverse of the Neumann Laplacian's DCT symbol (0 at DC), read-only."""
    rows, cols = shape
    u = np.arange(rows)[:, np.newaxis]
    v = np.arange(cols)[np.newaxis, :]
    symbol = 4 - 2 * np.cos(np.pi * u / rows) - 2 * np.cos(np.pi * v / cols)
    inverse = np.zeros_like(symbol)
    np.divide(1.0, symbol, out=inverse, where=symbol != 0)
    inverse = inverse.astype(dtype)
    inverse.flags.writeable = False
    return inverse


def _negative_divergence(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    r"""Compute :math:`-\nabla \cdot` of edge values, into `out`.

    Each edge adds to the pixel at its end and subtracts from the pixel at its
    start, so the result sums to zero.
    """
    out[...] = 0
    out[:, 1:] += flux_x
    out[:, :-1] -= flux_x
    out[1:, :] += flux_y
    out[:-1, :] -= flux_y
    return out


class _Edges:
    """The core of a solve on the CPU, shared by `_IrlsProblem` and `admm._AdmmProblem`.

    The target gradients and their jumps, the edge weights :math:`C` (all ones
    if none are given) and the invalid pixels, the weights of the current
    iterate (:math:`C` times its Charbonnier weights), the unweighted
    least-squares solution, rounding onto the branches, and counting the cuts.
    """

    def __init__(
        self,
        wrapped: np.ndarray,
        workers: int | None,
        edge_weights: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        pad: bool = True,
        invalid: np.ndarray | None = None,
    ) -> None:
        rows, cols = wrapped.shape
        # The target gradients are the wrapped differences `_wrap(np.diff(wrapped))`:
        # a difference of two wrapped values is only known modulo 2π, and the
        # target is its smallest representative. The jumps are the whole turns
        # that wrapping removed from each difference: a congruent phase
        # `wrapped + 2π branches` differs from the target gradient on an edge by
        # `jumps + diff(branches)` turns. Both in one pass:
        self.target_x = np.empty((rows, cols - 1), dtype=wrapped.dtype)
        self.target_y = np.empty((rows - 1, cols), dtype=wrapped.dtype)
        self.jumps_x = np.empty((rows, cols - 1), dtype=np.int32)
        self.jumps_y = np.empty((rows - 1, cols), dtype=np.int32)
        _irls_numba.targets_and_jumps(
            wrapped, self.target_x, self.target_y, self.jumps_x, self.jumps_y,
        )
        # The edge weights, in one buffer (for a selection over all of them):
        n_x = rows * (cols - 1)
        self.edges = np.empty(n_x + (rows - 1) * cols, dtype=wrapped.dtype)
        self.edge_x = self.edges[:n_x].reshape(rows, cols - 1)
        self.edge_y = self.edges[n_x:].reshape(rows - 1, cols)
        if edge_weights is None:
            self.edges[...] = 1
        else:
            for source, target in zip(edge_weights, (self.edge_x, self.edge_y),
                                      strict=True):
                _irls_numba.scale_into(np.asarray(source), 1.0, target)
        # The invalid pixels (`invalid`, None if there are none) are 0 in
        # `wrapped`. The edges that touch them get a small weight, so
        # that the cuts this placeholder forces cost almost nothing, and they do
        # not count in the cuts (see `cuts`).
        if invalid is None:
            self.invalid = np.zeros((0, 0), dtype=np.bool_)
        else:
            self.invalid = invalid
            self.edge_x[invalid[:, 1:] | invalid[:, :-1]] = _INVALID_EDGE_WEIGHT
            self.edge_y[invalid[1:, :] | invalid[:-1, :]] = _INVALID_EDGE_WEIGHT
        self.weights_x = self.edge_x.copy()
        self.weights_y = self.edge_y.copy()
        self._iterations = 0  # of the last solve: see `reweight`
        self.precondition = _PoissonPreconditioner(
            (rows, cols), wrapped.dtype, workers=workers, pad=pad,
        )

    def cuts(self, branches: np.ndarray) -> tuple[int, float]:
        """Return the cut turns and the cut cost of `wrapped + 2π branches`.

        The cut turns are the sum over all edges of :code:`|k|`, where :code:`k` is the
        number of whole turns between the gradient of :code:`wrapped + 2π
        branches` and the wrapped gradient, and the same sum with each edge
        weighted by its edge weight. Edges that touch an invalid pixel do not
        count.
        """
        rows = branches.shape[0]
        turns = np.empty(rows, dtype=np.int64)
        cost = np.empty(rows, dtype=np.float64)
        _irls_numba.cuts_per_row(
            branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
            self.invalid, turns, cost,
        )
        return int(turns.sum()), float(cost.sum())

    def l2_solution(self) -> np.ndarray:
        """Return the unweighted least-squares (L2) solution, ignoring the edge weights.

        Exact up to the padding: the preconditioner is the inverse of the operator
        with unit weights.
        """
        b = np.empty(self.precondition.shape, dtype=self.target_x.dtype)
        _irls_numba.negative_divergence(self.target_x, self.target_y, b)
        return np.array(self.precondition(b))

    def reweight(
        self, phi: np.ndarray, epsilon: float, *, unless_settled: bool = False,
    ) -> None:
        """Set the weights from the iterate `phi` at `epsilon`.

        With `unless_settled`, only if the last solve ran an iteration (see
        `_IrlsProblem.solve_round`).
        """
        if unless_settled and self._iterations == 0:
            return
        _irls_numba.charbonnier_weights(
            phi, self.target_x, self.target_y, self.edge_x, self.edge_y,
            phi.dtype.type(epsilon * epsilon), self.weights_x, self.weights_y,
        )

    def congruent(
        self, phi: np.ndarray, wrapped: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Snap `phi` onto the branches of `wrapped`, see :func:`_congruent`."""
        return _congruent(phi, wrapped)

    def keep(self, candidate: _Candidate) -> _Candidate:
        """Return the candidate, to keep past the next round (it owns its arrays)."""
        return candidate

    def cold_start(
        self, wrapped: np.ndarray, options: IRLSOptions | ADMMOptions,
    ) -> tuple[np.ndarray, _Candidate, float]:
        """Start a solve without a prior, see :func:`_cold_start`."""
        return _cold_start(self, wrapped, options)

    def warm_start(
        self,
        wrapped: np.ndarray,
        prior: L1PhasePrior | L1BranchesPrior,
        options: IRLSOptions | ADMMOptions,
    ) -> tuple[np.ndarray, _Candidate, float]:
        """Start a solve from a prior, see :func:`_warm_start`."""
        return _warm_start(self, wrapped, prior, options)

    def release(self, candidate: _Candidate) -> _Candidate:
        """Return the result of the solve (it owns its arrays)."""
        return candidate


class _IrlsProblem(_Edges):
    r"""The weighted least-squares problem of one IRLS round.

    Find :math:`\phi` with :math:`A \phi = b`, where
    :math:`A \phi = -\nabla \cdot (w \nabla \phi)` and
    :math:`b = -\nabla \cdot (w g)` for the wrapped gradients :math:`g` and the
    weights :math:`w`. :math:`A` is positive semi-definite: for an interior
    pixel with unit weights, it is :code:`4 p - p[left] - p[right] - p[up] -
    p[down]`, the same sign convention as the preconditioner.
    """

    def __init__(
        self,
        wrapped: np.ndarray,
        workers: int | None,
        edge_weights: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        invalid: np.ndarray | None = None,
    ) -> None:
        super().__init__(wrapped, workers, edge_weights, invalid=invalid)
        self._row_sums = np.empty(wrapped.shape[0], dtype=np.float64)
        self._b = np.empty(wrapped.shape, dtype=wrapped.dtype)

    def apply(self, p: np.ndarray, out: np.ndarray) -> float:
        """Compute :math:`A p` into `out`, and return :code:`p · A p`."""
        _irls_numba.apply_operator(
            p, self.weights_x, self.weights_y, out, self._row_sums,
        )
        return float(self._row_sums.sum())

    def dot(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return the dot product :code:`a · b`, the same on any number of threads."""
        _irls_numba.dot_rows(a, b, self._row_sums)
        return float(self._row_sums.sum())

    def cg_update(
        self,
        phi: np.ndarray,
        direction: np.ndarray,
        residual: np.ndarray,
        operated: np.ndarray,
        alpha: float,
    ) -> float:
        """Step `phi` and `residual` along `direction`; return the new :code:`r · r`."""
        _irls_numba.cg_update(
            phi, direction, residual, operated, alpha, self._row_sums,
        )
        return float(self._row_sums.sum())

    def right_hand_side(self, out: np.ndarray) -> np.ndarray:
        """Compute :math:`b` into `out`, without its mean."""
        _irls_numba.negative_divergence_of_weighted(
            self.weights_x, self.target_x, self.weights_y, self.target_y, out,
        )
        # The operator annihilates constants, so its range excludes them:
        out -= out.mean()
        return out

    def solve_round(
        self, phi: np.ndarray, options: IRLSOptions,
    ) -> Callable[[], L1Round]:
        """Run one round's CG on `phi` (in place), and remove its mean if it changed.

        Returns a function that returns the round's report. (On the GPU, the
        report is only read back at the next synchronization.)
        """
        report, iterated = _conjugate_gradients(
            self, phi, self.right_hand_side(out=self._b), options,
        )
        # Nothing in the solve controls the mean of the iterate, so rounding
        # errors make it drift; remove it.
        if iterated:
            phi -= phi.mean()
        self._iterations = report.iterations
        return lambda: report

    def run_round(
        self,
        phi: np.ndarray,
        wrapped: np.ndarray,
        *,
        epsilon: float,
        at_end: bool,
        options: IRLSOptions,
        index: int,
    ) -> tuple[L1Round, _Candidate]:
        """Run one IRLS round on `phi` (in place).

        Solve the weighted problem, reweight at `epsilon` (unless the round
        settled: no CG iteration, `at_end` of the schedule), and round the result
        onto the branches. Returns the round's report and its candidate result,
        whose `weights_epsilon` and `converged` the caller sets.
        """
        pending = self.solve_round(phi, options)
        self.reweight(phi, epsilon, unless_settled=at_end)
        candidate = _Candidate.take(
            self, phi, wrapped,
            weights_epsilon=math.nan, converged=False, round_index=index,
        )
        return pending(), candidate


def _starting_phase(
    prior: L1PhasePrior | L1BranchesPrior,
    wrapped: np.ndarray,
) -> np.ndarray:
    """Turn the prior into a phase that is congruent with `wrapped`."""
    xp = _array_module(wrapped)
    if isinstance(prior, L1PhasePrior):
        raw = wrapped.astype(np.float64)
        branches = xp.rint((xp.asarray(prior.phase, dtype=np.float64) - raw) / TAU)
        # Where the prior is not finite (an invalid pixel of an earlier result),
        # the starting phase is the wrapped value:
        branches = xp.where(xp.isfinite(branches), branches, 0.0)
        return (raw + TAU * branches).astype(wrapped.dtype)
    branches = xp.asarray(prior.branches).astype(wrapped.dtype)
    return wrapped + wrapped.dtype.type(TAU) * branches


def _congruent(
    phi: np.ndarray,
    wrapped: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap the solution onto the branches the wrapped input allows.

    The solve produces a smooth field that is only approximately congruent with the
    input. Rounding to the nearest multiple of 2π makes it exactly so. Removing the
    median offset first keeps the rounding from being biased by the constant the
    solve cannot determine.
    """
    tau = phi.dtype.type(TAU)
    difference = np.empty_like(phi)
    _irls_numba.wrapped_difference(phi, wrapped, tau, difference)
    offset = np.median(difference, overwrite_input=True)
    branches = np.empty(phi.shape, dtype=np.int32)
    phase = np.empty(phi.shape, dtype=np.float64)
    _irls_numba.branches_and_phase(phi, wrapped, offset, tau, branches, phase)
    return phase, branches


def _conjugate_gradients(
    problem: _IrlsProblem,
    phi: np.ndarray,
    b: np.ndarray,
    options: IRLSOptions,
) -> tuple[L1Round, bool]:
    """Run preconditioned CG on `phi` (updated in place) against `b`.

    Returns the round's report, and whether `phi` was changed.
    """
    norm = math.sqrt(problem.dot(b, b))
    if not norm > 0:
        # Zero right-hand side: already solved, and the relative residual would
        # be 0/0.
        return L1Round(iterations=0, residuals=[0.0], converged=True), False

    report = L1Round(iterations=0)
    operated = np.empty_like(phi)
    problem.apply(phi, out=operated)
    residual = b - operated
    initial = math.sqrt(problem.dot(residual, residual)) / norm
    report.residuals.append(initial)
    if initial <= options.cg_relative_tolerance:
        # Already solved, for example by a warm start on a near-identical input.
        report.converged = True
        return report, False

    z = problem.precondition(residual)
    direction = z.copy()
    residual_dot_z = problem.dot(residual, z)
    tiny = np.finfo(phi.dtype).tiny

    for iteration in range(options.iterations_per_round):
        curvature = problem.apply(direction, out=operated)
        # Zero curvature: the direction lies in the nullspace (constants),
        # there is nothing left to descend.
        if abs(curvature) <= tiny:
            report.converged = True
            break
        alpha = phi.dtype.type(residual_dot_z / curvature)
        relative = math.sqrt(
            problem.cg_update(phi, direction, residual, operated, alpha),
        ) / norm
        report.iterations = iteration + 1
        report.residuals.append(relative)
        if relative <= options.cg_relative_tolerance:
            report.converged = True
            break

        z = problem.precondition(residual)
        next_residual_dot_z = problem.dot(residual, z)
        beta = phi.dtype.type(next_residual_dot_z / residual_dot_z)
        residual_dot_z = next_residual_dot_z
        _irls_numba.next_direction(direction, z, beta)
    return report, True


# A round counts as an improvement only if it lowers the cut cost by more than half
# a cut through edges of full weight. Without edge weights, the cost is an integer,
# so any decrease counts; with them, tiny gains in near-zero-weight regions don't.
_MIN_IMPROVEMENT = 0.5


@dataclass
class _Candidate:
    """A rounded iterate, the weights derived from it, and how it was obtained."""

    phase: np.ndarray
    branches: np.ndarray
    cut_turns: int
    cut_cost: float
    weights_x: np.ndarray
    weights_y: np.ndarray
    weights_epsilon: float
    converged: bool
    round_index: int

    @classmethod
    def take(
        cls,
        problem: _Edges,
        phi: np.ndarray,
        wrapped: np.ndarray,
        *,
        weights_epsilon: float,
        converged: bool,
        round_index: int,
    ) -> _Candidate:
        phase, branches = problem.congruent(phi, wrapped)
        cut_turns, cut_cost = problem.cuts(branches)
        return cls(
            phase=phase,
            branches=branches,
            cut_turns=cut_turns,
            cut_cost=cut_cost,
            weights_x=problem.weights_x.copy(),
            weights_y=problem.weights_y.copy(),
            weights_epsilon=weights_epsilon,
            converged=converged,
            round_index=round_index,
        )


class _RoundProblem(Protocol[_OptionsIn_contra]):
    """A problem that `_start_and_rounds` starts and `_irls_rounds` runs.

    `_IrlsProblem`, `_AdmmProblem`, and the GPU's IRLS problem.
    """

    def cold_start(
        self, wrapped: np.ndarray, options: _OptionsIn_contra,
    ) -> tuple[np.ndarray, _Candidate, float]:
        """Start a solve without a prior: the phase, best candidate, and epsilon."""
        ...

    def warm_start(
        self,
        wrapped: np.ndarray,
        prior: L1PhasePrior | L1BranchesPrior,
        options: _OptionsIn_contra,
    ) -> tuple[np.ndarray, _Candidate, float]:
        """Start a solve from a prior: the phase, best candidate, and epsilon."""
        ...

    def release(self, candidate: _Candidate) -> _Candidate:
        """Return the result, which owns its arrays."""
        ...

    def run_round(
        self,
        phi: np.ndarray,
        wrapped: np.ndarray,
        *,
        epsilon: float,
        at_end: bool,
        options: _OptionsIn_contra,
        index: int,
    ) -> tuple[L1Round, _Candidate]:
        """Run one round on `phi` (in place); return its report and candidate."""
        ...

    def keep(self, candidate: _Candidate) -> _Candidate:
        """Keep `candidate` past the next round."""
        ...


def _irls_rounds(
    problem: _RoundProblem[_Options],
    phi: np.ndarray,
    wrapped: np.ndarray,
    weights_epsilon: float,
    options: _Options,
    *,
    best: _Candidate,
) -> tuple[list[L1Round], _Candidate]:
    """Run the rounds (IRLS, or ADMM on the CPU) on `phi` (updated in place).

    `weights_epsilon` is the epsilon the current weights were derived at (infinity
    if they were not derived from an iterate), and `best` the candidate result to
    beat. Returns the report of each round, and the best rounded iterate: the
    latest one with the lowest cut cost.
    """
    epsilon = (
        options.epsilon_start
        if math.isinf(weights_epsilon)
        else max(options.epsilon_min, weights_epsilon * options.decay)
    )
    stale = 0
    rounds: list[L1Round] = []
    for index in range(options.rounds):
        # A fixed point: the iterate did not move and the schedule has reached
        # its end, so further rounds would repeat this one. Whether the iterate
        # moved is known only after the solve, so the problem decides itself
        # whether to reweight; on the GPU, a whole round is then one CUDA graph.
        at_end = epsilon <= options.epsilon_min
        report, candidate = problem.run_round(
            phi, wrapped, epsilon=epsilon, at_end=at_end, options=options, index=index,
        )
        rounds.append(report)
        settled = report.iterations == 0 and at_end
        if not settled:
            weights_epsilon = epsilon
            epsilon = max(options.epsilon_min, epsilon * options.decay)
        candidate.weights_epsilon = weights_epsilon
        candidate.converged = report.converged
        report.cut_turns = candidate.cut_turns
        report.cut_cost = candidate.cut_cost
        threshold = max(_MIN_IMPROVEMENT, options.stop_tolerance * best.cut_cost)
        improved = candidate.cut_cost < best.cut_cost - threshold
        stale = 0 if improved else stale + 1
        if candidate.cut_cost <= best.cut_cost:
            best = problem.keep(candidate)
        patience = options.stop_patience
        if settled or (patience is not None and stale >= patience):
            break
        # No cuts: the rounded result agrees with every wrapped gradient, which no
        # round can improve on.
        if best.cut_turns == 0:
            break
    return rounds, best


def _cold_start(
    problem: _Edges,
    wrapped: np.ndarray,
    options: IRLSOptions | ADMMOptions,
) -> tuple[np.ndarray, _Candidate, float]:
    """Start a solve without a prior: return the phase, best candidate, and epsilon.

    The epsilon is the one the current weights were derived at.
    """
    # The rounded L2 solution is the first candidate: where the field has no
    # residues, it is already the answer, and the result is never worse.
    phi = problem.l2_solution()
    weights_epsilon = math.inf
    best = _Candidate.take(
        problem, phi, wrapped,
        weights_epsilon=weights_epsilon, converged=True, round_index=-1,
    )
    if options.cold_start == "zero" and best.cut_turns > 0:
        # The L2 solution smears discontinuities and residues, which biases
        # the rounds towards the wrong ones of equally cheap cuts. From zero,
        # the first weights make edges with a large wrapped gradient cheap.
        phi = _array_module(wrapped).zeros_like(wrapped)
        weights_epsilon = options.epsilon_start
        problem.reweight(phi, weights_epsilon)
    return phi, best, weights_epsilon


def _warm_start(
    problem: _Edges,
    wrapped: np.ndarray,
    prior: L1PhasePrior | L1BranchesPrior,
    options: IRLSOptions | ADMMOptions,
) -> tuple[np.ndarray, _Candidate, float]:
    """Start a solve from a prior: return the phase, best candidate, and epsilon.

    Start on the prior, with the weights it implies where its schedule stopped.
    Restarting the schedule from unit weights would make the first round
    converge to the L2 solution and discard the prior.
    """
    phi = _starting_phase(prior, wrapped)
    weights_epsilon = min(
        max(prior.epsilon, options.epsilon_min), options.epsilon_start,
    )
    problem.reweight(phi, weights_epsilon)
    best = _Candidate.take(
        problem, phi, wrapped,
        weights_epsilon=weights_epsilon, converged=True, round_index=-1,
    )
    return phi, best, weights_epsilon


# ---- The interface between the driver and the solvers --------------------------------


@dataclass
class _Prepared:
    """The checked input of a solve, on its device (see `_check_input`).

    `data` is the wrapped phase, in the precision of the solve; on the GPU, it is
    the complex wave itself if no edge weights are given (the start graph
    computes its phase and amplitude weights). `edge_weights` are normalized to
    a maximum of 1, or None. `invalid` (on the CPU) are the pixels whose input is
    not finite, which are 0 in `data`, or None if there are none.
    """

    data: np.ndarray
    edge_weights: tuple[np.ndarray, np.ndarray] | None
    invalid: np.ndarray | None


@dataclass
class _Solution:
    """What a solver returns: the best rounded iterate, the rounds, the status."""

    phase: np.ndarray
    branches: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray
    rounds: list[L1Round]
    selected_round: int
    cut_turns: int
    cut_cost: float
    weights_epsilon: float
    converged: bool
    status: L1Status

    @classmethod
    def of(cls, best: _Candidate, rounds: list[L1Round], status: L1Status) -> _Solution:
        """Return the solution with the result `best`, after `rounds`."""
        return cls(
            phase=best.phase, branches=best.branches,
            weights_x=best.weights_x, weights_y=best.weights_y,
            rounds=rounds, selected_round=best.round_index,
            cut_turns=best.cut_turns, cut_cost=best.cut_cost,
            weights_epsilon=best.weights_epsilon, converged=best.converged,
            status=status,
        )


class _Solver(Protocol[_OptionsIn_contra]):
    """One method on one device: `_CpuIrls`, `_CpuAdmm`, `_GpuIrls`, `_GpuAdmm`."""

    def solve(
        self,
        prepared: _Prepared,
        prior: L1PhasePrior | L1BranchesPrior | None,
        options: _OptionsIn_contra,
    ) -> _Solution:
        """Solve `prepared`: start (from `prior`, if given), run the rounds, finish.

        The result has NaN at the invalid pixels, and branch 0 there.
        """
        ...


def _start_and_rounds(
    problem: _RoundProblem[_Options],
    wrapped: np.ndarray,
    prior: L1PhasePrior | L1BranchesPrior | None,
    options: _Options,
) -> tuple[list[L1Round], _Candidate]:
    """Start `problem` and run its rounds in Python; return them and the result.

    The result is the best candidate, released from the problem.
    """
    if prior is None:
        phi, best, weights_epsilon = problem.cold_start(wrapped, options)
    else:
        phi, best, weights_epsilon = problem.warm_start(wrapped, prior, options)
    rounds: list[L1Round] = []
    if best.cut_turns > 0:
        rounds, best = _irls_rounds(
            problem, phi, wrapped, weights_epsilon, options, best=best,
        )
    return rounds, problem.release(best)


def _irls_status(rounds: list[L1Round]) -> L1Status:
    """Return the status of an IRLS solve: whether the CG of every round converged."""
    return "converged" if all(r.converged for r in rounds) else "iteration_limit"


def _restore_invalid(best: _Candidate, invalid: np.ndarray | None) -> None:
    """Set the result at the invalid pixels: phase NaN, branch 0."""
    if invalid is not None:
        best.phase[invalid] = np.nan
        best.branches[invalid] = 0


class _CpuIrls:
    """IRLS on the CPU, on `workers` threads (see `_cpu_threads`)."""

    def __init__(self, workers: int) -> None:
        self.workers = workers

    def solve(
        self,
        prepared: _Prepared,
        prior: L1PhasePrior | L1BranchesPrior | None,
        options: IRLSOptions,
    ) -> _Solution:
        """See `_Solver.solve`."""
        problem = _IrlsProblem(
            prepared.data, self.workers, prepared.edge_weights,
            invalid=prepared.invalid,
        )
        rounds, best = _start_and_rounds(problem, prepared.data, prior, options)
        _restore_invalid(best, prepared.invalid)
        return _Solution.of(best, rounds, _irls_status(rounds))


def _check_edge_weights(
    edge_weights: tuple[np.ndarray, np.ndarray],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate edge weights and scale them to a maximum of one."""
    rows, cols = shape
    xp = _array_module(edge_weights[0]) if len(edge_weights) else np
    weights = tuple(xp.asarray(w, dtype=np.float64) for w in edge_weights)
    expected = ((rows, cols - 1), (rows - 1, cols))
    if len(weights) != 2 or any(  # noqa: PLR2004
        w.shape != e for w, e in zip(weights, expected, strict=False)
    ):
        msg = (
            "edge_weights must be two arrays with shapes "
            f"{expected[0]} and {expected[1]}, got {[w.shape for w in weights]}"
        )
        raise ValueError(msg)
    # All checks in one array, so that on the GPU, they cost one synchronization:
    checks = [xp.all(xp.isfinite(w) & (w > 0)) for w in weights]
    maxima = [w.max() for w in weights if w.size]
    values = xp.stack(
        [xp.asarray(v, dtype=np.float64) for v in (*checks, *maxima)],
    ).tolist()
    if not all(values[:len(checks)]):
        msg = "edge_weights must be finite and greater than zero"
        raise ValueError(msg)
    largest = max(values[len(checks):], default=1.0)
    return weights[0] / largest, weights[1] / largest


def _check_input(
    wrapped: np.ndarray,
    prior: L1PhasePrior | L1BranchesPrior | None,
    edge_weights: tuple[np.ndarray, np.ndarray] | None,
    *,
    keep_wave: bool = False,
    workers: int | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None, np.ndarray | None]:
    """Validate the input, and convert it to the precision of the solve.

    Returns the wrapped phase, the edge weights, which for a complex wave default
    to :func:`amplitude_edge_weights`, and the invalid pixels: a boolean array of
    the pixels whose input is not finite, which are 0 in the returned phase, or
    None if there are none. With `keep_wave`, a complex wave without edge weights
    is returned as it is, for the GPU to convert. For cupy input, the invalid
    pixels are always None: the GPU finds them itself.
    """
    xp = _array_module(wrapped)
    wrapped = xp.asarray(wrapped)
    if wrapped.ndim != 2 or wrapped.size == 0:  # noqa: PLR2004
        msg = f"wrapped must be a non-empty 2D array, got shape {wrapped.shape}"
        raise ValueError(msg)
    if wrapped.dtype.kind not in "iufc":
        msg = f"wrapped must be a real or complex array, got dtype {wrapped.dtype}"
        raise TypeError(msg)
    keep_wave = keep_wave and xp.iscomplexobj(wrapped) and edge_weights is None
    invalid = None
    if xp is np and np.iscomplexobj(wrapped) and edge_weights is None:
        # The same as below, on several threads:
        wrapped, edge_weights, invalid = _wave_setup(wrapped, workers or 1)
    elif xp.iscomplexobj(wrapped) and not keep_wave:
        if edge_weights is None:
            edge_weights = amplitude_edge_weights(xp.abs(wrapped))
        # A wave that is not finite can have a finite angle (an infinite one):
        wrapped = xp.where(xp.isfinite(wrapped), xp.angle(wrapped), xp.nan)
    if edge_weights is not None and not isinstance(edge_weights, _CheckedEdges):
        edge_weights = _check_edge_weights(edge_weights, wrapped.shape)
    _check_prior_shape(prior, wrapped.shape)
    if keep_wave:
        return wrapped, None, None
    dtype = np.float32 if wrapped.dtype == np.float32 else np.float64
    wrapped = xp.ascontiguousarray(wrapped, dtype=dtype)
    if xp is np and invalid is None:
        wrapped, invalid = _mask_invalid(wrapped)
    return wrapped, edge_weights, invalid


def _check_prior_shape(
    prior: L1PhasePrior | L1BranchesPrior | None, shape: tuple[int, ...],
) -> None:
    if prior is None:
        return
    prior_shape = np.shape(
        prior.phase if isinstance(prior, L1PhasePrior) else prior.branches,
    )
    if prior_shape != shape:
        msg = (
            f"the prior has shape {prior_shape}, "
            f"but wrapped has shape {shape}"
        )
        raise ValueError(msg)


def _mask_invalid(wrapped: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return `wrapped` with its invalid pixels set to 0, and the invalid pixels.

    If all pixels are finite, return `wrapped` itself and None.
    """
    if not _irls_numba.count_nonfinite(
        wrapped, np.empty(wrapped.shape[0], dtype=np.int64),
    ):
        return wrapped, None
    invalid = ~np.isfinite(wrapped)
    return np.where(invalid, 0, wrapped), invalid  # a copy: the caller's stays


_AMPLITUDE_FLOOR = 1e-3
_AMPLITUDE_PERCENTILE = 90.0
# The weight of edges that touch an invalid pixel, where the largest weight is 1:
_INVALID_EDGE_WEIGHT = _AMPLITUDE_FLOOR


class _CheckedEdges(tuple):
    """Edge weights that `_wave_setup` has already checked and normalized."""

    __slots__ = ()


_pools: dict[int, ThreadPoolExecutor] = {}
_pools_lock = threading.Lock()


def _by_rows(
    func: Callable[[np.ndarray, np.ndarray], object],
    x: np.ndarray,
    dtype: np.dtype,
    workers: int,
) -> np.ndarray:
    """Return `func(x)` into a new array of `dtype`, by blocks of rows on threads.

    `func(block, out)` must work row by row, as numpy's elementwise functions
    do, which also release the GIL: so the result is that of one call.
    """
    out = np.empty(x.shape, dtype=dtype)
    blocks = min(workers, x.shape[0])
    if blocks <= 1 or x.size < 1 << 16:  # not worth the threads
        func(x, out)
        return out
    with _pools_lock:
        pool = _pools.get(workers)
        if pool is None:
            pool = _pools[workers] = ThreadPoolExecutor(workers)
    bounds = np.linspace(0, x.shape[0], blocks + 1).astype(int)
    list(pool.map(
        lambda b: func(x[bounds[b]:bounds[b + 1]], out[bounds[b]:bounds[b + 1]]),
        range(blocks),
    ))
    return out


def _percentile(x: np.ndarray, q: float) -> float:
    """:code:`np.percentile(x, q)` of float64 values, by a parallel selection."""
    n = x.size
    virtual = (n - 1) * np.true_divide(q, 100)
    previous = int(np.floor(virtual))
    a, b, nan = _irls_numba.select_pair(x.ravel(), previous, min(previous + 1, n - 1))
    if nan:
        return math.nan
    # numpy's `_lerp`:
    t = virtual - previous
    difference = b - a
    return b - difference * (1 - t) if t >= 0.5 else a + difference * t  # noqa: PLR2004


def _wave_setup(
    wave: np.ndarray, workers: int,
) -> tuple[np.ndarray, _CheckedEdges, np.ndarray | None]:
    """Return the phase of `wave`, its edge weights, and its invalid pixels.

    The results are identical to :code:`np.angle(wave)` and to
    :func:`amplitude_edge_weights` of :code:`np.abs(wave)`, checked and normalized
    by :func:`_check_edge_weights`, but are computed on several threads: numpy's
    functions on blocks of rows, a parallel selection for the percentile, and one
    fused pass for the weights. The invalid pixels, where the wave is not finite,
    are 0 in the phase; they are None if there are none.
    """
    real = np.float32 if wave.dtype == np.complex64 else np.float64
    amplitude = _by_rows(lambda z, out: np.abs(z, out=out), wave, real, workers)
    phase = _by_rows(
        lambda z, out: np.arctan2(z.imag, z.real, out=out), wave, real, workers,
    )
    rows, cols = wave.shape
    invalid = None
    if _irls_numba.count_nonfinite(amplitude, np.empty(rows, dtype=np.int64)):
        invalid = ~np.isfinite(amplitude)
        phase[invalid] = 0
        valid = amplitude[~invalid]
        # As `amplitude_edge_weights`: invalid pixels have no signal, and do not
        # take part in the reference.
        amplitude[invalid] = 0
        reference = (
            _percentile(valid, _AMPLITUDE_PERCENTILE) if valid.size else math.nan
        )
    else:
        reference = _percentile(amplitude, _AMPLITUDE_PERCENTILE)
    if not reference > 0:
        # No signal anywhere: no edge is more reliable than another.
        return phase, _CheckedEdges((
            np.ones((rows, cols - 1)), np.ones((rows - 1, cols)),
        )), invalid
    edge_x = np.empty((rows, cols - 1))
    edge_y = np.empty((rows - 1, cols))
    row_max = np.empty(rows)
    row_ok = np.empty(rows, dtype=np.bool_)
    _irls_numba.amplitude_edges(
        amplitude, reference, _AMPLITUDE_FLOOR, edge_x, edge_y, row_max, row_ok,
    )
    if not row_ok.all():
        msg = "edge_weights must be finite and greater than zero"
        raise ValueError(msg)
    largest = float(row_max.max()) if edge_x.size + edge_y.size else 1.0
    for edges in (edge_x, edge_y):
        _irls_numba.scale_into(edges, largest, edges)
    return phase, _CheckedEdges((edge_x, edge_y)), invalid


def amplitude_edge_weights(
    amplitude: np.ndarray,
    floor: float = _AMPLITUDE_FLOOR,
    percentile: float = _AMPLITUDE_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Edge weights for the L1 unwrapping, from the amplitude of a complex wave.

    Where the amplitude is low, the phase is noisy or undefined, for example where
    the specimen blocks the beam. Cuts through such regions should cost little,
    so that :func:`unwrap_phase_irls` and :func:`unwrap_phase_admm` put the
    discontinuities there and do not spend iterations on reconciling their noise.

    The weight of an edge is the smaller amplitude of its two pixels, divided by
    the given percentile of the amplitude and clipped to :code:`[floor, 1]`.
    Clipping at one keeps bright features, such as Fresnel fringes, from
    attracting cuts. The floor keeps regions without signal weakly connected to
    the rest of the image. Pixels whose amplitude is not finite count as pixels
    without signal, and do not take part in the percentile.

    Parameters
    ----------
    amplitude
        Amplitude of the wave, for example :code:`np.abs(wave)`, as a numpy or
        cupy array.
    floor
        Smallest weight, relative to the largest.
    percentile
        The percentile of the amplitude that gets full weight. A high percentile
        rather than the median keeps the reference in the beam, even if most of
        the image is blocked.

    Returns
    -------
    edge_weights
        The edge weights of the horizontal and vertical edges, with shapes
        :code:`(rows, cols - 1)` and :code:`(rows - 1, cols)`, as arrays of the
        same kind as :code:`amplitude`. Pass them as :code:`edge_weights`.

    """
    xp = _array_module(amplitude)
    amplitude = xp.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2:  # noqa: PLR2004
        msg = f"amplitude must be a 2D array, got shape {amplitude.shape}"
        raise ValueError(msg)
    if not 0 < floor <= 1:
        msg = f"floor must be in (0, 1], got {floor}"
        raise ValueError(msg)
    finite = xp.isfinite(amplitude)
    if xp is np:
        if finite.all():
            reference = np.percentile(amplitude, percentile)
        else:
            reference = (
                np.percentile(amplitude[finite], percentile) if finite.any()
                else math.nan
            )
            amplitude = np.where(finite, amplitude, 0.0)
        if not reference > 0:
            # No signal anywhere: no edge is more reliable than another.
            return (
                np.ones((amplitude.shape[0], amplitude.shape[1] - 1)),
                np.ones((amplitude.shape[0] - 1, amplitude.shape[1])),
            )
        relative = np.clip(amplitude / reference, floor, 1.0)
    else:
        # The same, without the synchronization of a host-side check: the
        # percentile of the finite values, which sort before the others, by
        # numpy's "linear" method (`_lerp`), with the count on the device.
        valid = finite.sum()
        ordered = xp.sort(xp.where(finite, amplitude, xp.inf).ravel())
        virtual = (valid - 1) * np.true_divide(percentile, 100)
        low = xp.floor(virtual)
        t = virtual - low
        below = ordered[xp.maximum(low.astype(np.int64), 0)]
        above = ordered[xp.clip(low.astype(np.int64) + 1, 0, xp.maximum(valid - 1, 0))]
        difference = above - below
        reference = xp.where(
            t >= 0.5, above - difference * (1 - t), below + difference * t,  # noqa: PLR2004
        )
        amplitude = xp.where(finite, amplitude, 0.0)
        relative = xp.where(
            reference > 0, xp.clip(amplitude / reference, floor, 1.0), 1.0,
        )
    return (
        xp.minimum(relative[:, 1:], relative[:, :-1]),
        xp.minimum(relative[1:, :], relative[:-1, :]),
    )


def unwrap_phase_irls(
    wrapped: np.ndarray,
    *,
    prior: L1PhasePrior | L1BranchesPrior | L1Result | None = None,
    options: IRLSOptions | None = None,
    workers: int | None = None,
    edge_weights: tuple[np.ndarray, np.ndarray] | None = None,
    workspace: Workspace | None = None,
) -> L1Result:
    r"""Unwrap a 2D phase by iteratively reweighted least squares.

    The result is congruent with the input: it differs from it by an integer
    multiple of :math:`2\pi` at every valid pixel. The solve is global and places
    the discontinuities where they cost least.

    The solve runs in float32 for float32 and complex64 input, and in float64
    otherwise. The
    result phase is float64, computed from the input values, so it stays exactly
    congruent with them.

    This algorithm is recently added and still experimental, use at your own risk.

    Parameters
    ----------
    wrapped
        The wrapped phase, usually in :math:`[-\pi, \pi]`; any real values work,
        since only differences modulo :math:`2\pi` enter the solve. Or the complex
        wave: then its phase is unwrapped, with edge weights from its amplitude
        (:func:`amplitude_edge_weights`) unless :code:`edge_weights` is given.
        For a cupy array, the solve runs on the GPU and the result holds cupy
        arrays. Pixels that are not finite (NaN or infinite) are invalid: the solve
        treats their phase as 0, gives the edges that touch them a weight of
        :code:`1e-3` (where the largest weight is 1), and leaves those edges out
        of the cut turns and cost. The result is NaN there, with branch 0.
    prior
        Start from an earlier solution instead of from scratch, and resume its
        epsilon schedule; see :class:`L1PhasePrior` and :class:`L1BranchesPrior`.
        Passing an :class:`L1Result` is the same as passing
        :meth:`L1Result.as_prior`. A warm start from the previous frame of a
        series typically needs far fewer CG iterations than a cold start.
    options
        Solver settings, :class:`IRLSOptions` by default.
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
        which solves in several threads then share, one at a time. Not used on
        the CPU.

    Returns
    -------
    L1Result
        The unwrapped phase, the branches, the weights of the selected iterate,
        the cut turns and cut cost, and a report of each round.

    Notes
    -----
    A plain least-squares (L2) unwrap, like
    :func:`~libertem_holo.base.unwrap.unwrap_phase_laplacian`, minimizes the
    squared difference between the gradient of the solution and the wrapped
    gradient of the input. Where the phase is smooth, that is the right objective.
    At a genuine discontinuity or around a residue, it smears the step across the
    image instead of letting it stand. IRLS minimizes an approximately L1 objective
    instead: it solves a sequence of weighted least-squares problems, and after each
    one it gives every edge between neighbouring pixels the Charbonnier weight
    :math:`\varepsilon^2 / \sqrt{r^2 + \varepsilon^2}`, where :math:`r` is the
    difference between the edge's current and wrapped gradient. An edge with a large
    residual stops pulling on its surroundings. :math:`\varepsilon` decays
    geometrically from :code:`epsilon_start` to :code:`epsilon_min` over the
    rounds: the solve starts smooth and then sharpens.

    Each weighted problem is solved by conjugate gradients (CG), preconditioned by
    the unweighted Neumann Poisson solve, which is a pair of cosine transforms.
    By default (:code:`cold_start="zero"`), the solve starts at zero phase, with
    weights derived from it: edges whose wrapped gradient is large, and so
    ambiguous, start out cheap to cut. Starting from the L2 solution instead
    (:code:`cold_start="least_squares"`), which smears discontinuities and
    residues, tends to put more pixels on the wrong branch in noisy fields, with
    as many cut turns.

    With edge weights :math:`C`, the objective is the weighted sum
    :math:`\sum C |r|`: each round's weights are :math:`C` times the Charbonnier
    weights.

    After each round, the iterate is snapped onto the branches of the input and
    scored. Its *cut turns* are the number of whole turns by which its gradient
    differs from the wrapped gradient, summed over all edges between neighbouring
    pixels: the L1 objective of a congruent result. Its *cut cost* weights each
    edge by :math:`C`; without edge weights, the two are equal. The solve returns
    the latest rounded iterate with the lowest cut cost. The starting point (the
    rounded L2 solution, or the prior) is a candidate too, so the result is never
    worse than that. The solve stops once the cut cost has not improved for
    :code:`stop_patience` rounds, where an improvement must exceed both
    :code:`stop_tolerance` of the cost and half a cut through edges of full
    weight, so that tiny gains in nearly weightless regions do not keep it going.
    It also stops as soon as there are no cut turns, since no result can be better.

    The approach follows B. Dubois-Taine, R. Akiki and A. d'Aspremont,
    "Iteratively Reweighted Least Squares for Phase Unwrapping", arXiv:2401.09961
    (2024), with three differences: it reweights the gradient residuals of the
    phase directly, without the paper's slack variables and quadratic penalty; it
    decays :math:`\varepsilon` over the rounds instead of keeping the paper's
    :math:`\delta` fixed; and the edge weights scale the smoothed cost of an edge,
    instead of its residual inside the smoothing.

    Examples
    --------
    >>> import numpy as np
    >>> from libertem_holo.base.unwrap import unwrap_phase_irls
    >>> y, x = np.mgrid[:64, :64]
    >>> truth = 0.3 * x + 0.2 * y
    >>> wrapped = np.angle(np.exp(1j * truth))
    >>> result = unwrap_phase_irls(wrapped)
    >>> result.status
    'converged'
    >>> np.allclose(result.phase - result.phase[0, 0], truth)
    True

    """
    return _run(
        wrapped, prior, _check_options(options, IRLSOptions), workers, edge_weights,
        workspace=workspace,
    )


def _check_options(options: object, expected: type[_OptionsT]) -> _OptionsT:
    """Return `options`, or the defaults of `expected` if None; check the type."""
    if options is None:
        return expected()
    if not isinstance(options, expected):
        msg = (
            f"options must be {expected.__name__}, got {type(options).__name__}: "
            "unwrap_phase_irls takes IRLSOptions, unwrap_phase_admm ADMMOptions"
        )
        raise TypeError(msg)
    return options


def _check_arguments(
    wrapped: np.ndarray,
    prior: L1PhasePrior | L1BranchesPrior | L1Result | None,
    workers: int | None,
    edge_weights: tuple[np.ndarray, np.ndarray] | None,
    workspace: Workspace | None = None,
) -> None:
    """Check the arguments of a solve besides the arrays' shapes and values.

    The device is the one of `wrapped`: the edge weights and the prior must be
    cupy arrays for cupy input, and not for numpy input.
    """
    if workers is not None and (
        isinstance(workers, bool) or not isinstance(workers, (int, np.integer))
        or workers < 1
    ):
        msg = f"workers must be a positive integer or None, got {workers!r}"
        raise ValueError(msg)
    if workspace is not None and not isinstance(workspace, Workspace):
        msg = f"workspace must be a Workspace or None, got {type(workspace).__name__}"
        raise TypeError(msg)
    if prior is not None and not isinstance(
        prior, (L1PhasePrior, L1BranchesPrior, L1Result),
    ):
        msg = (
            "prior must be an L1PhasePrior, L1BranchesPrior or L1Result, "
            f"got {type(prior).__name__}"
        )
        raise TypeError(msg)
    arrays = []
    if edge_weights is not None:
        if not isinstance(edge_weights, (tuple, list)):
            msg = (
                "edge_weights must be a pair of arrays, "
                f"got {type(edge_weights).__name__}"
            )
            raise TypeError(msg)
        arrays += [("edge_weights", w) for w in edge_weights]
    if prior is not None:
        arrays.append(_check_prior(prior))
    _check_devices(wrapped, arrays)


def _check_prior(
    prior: L1PhasePrior | L1BranchesPrior | L1Result,
) -> tuple[str, object]:
    """Check the prior's epsilon; return the name and the array of the prior."""
    epsilon = prior.resume_epsilon if isinstance(prior, L1Result) else prior.epsilon
    if not (math.isfinite(epsilon) and epsilon >= 0):
        msg = f"the prior's epsilon must be finite and not negative, got {epsilon}"
        raise ValueError(msg)
    if isinstance(prior, L1BranchesPrior):
        return "the prior's branches", prior.branches
    return "the prior's phase", prior.phase


def _check_devices(wrapped: np.ndarray, arrays: list[tuple[str, object]]) -> None:
    """Check that the (named) `arrays` are on the device of `wrapped`."""
    on_gpu = _array_module(wrapped) is not np
    for name, array in arrays:
        if (_array_module(array) is not np) != on_gpu:
            msg = (
                f"{name} must be on the device of wrapped "
                f"({'cupy' if on_gpu else 'numpy'} arrays), got {type(array).__name__}"
            )
            raise TypeError(msg)


@contextlib.contextmanager
def _cpu_threads(workers: int) -> Iterator[None]:
    """Run numba's kernels and the cosine transforms on `workers` threads.

    BLAS only computes dot products here, on one thread, so that their summation
    order and the result do not depend on the thread count. Only the outermost
    numba setting takes effect, so it comes before `set_num_threads`, which sets
    numba too.
    """
    with (
        set_numba_threads(min(workers, numba.config.NUMBA_NUM_THREADS)),
        set_num_threads(1),
    ):
        yield


def _run(
    wrapped: np.ndarray,
    prior: L1PhasePrior | L1BranchesPrior | L1Result | None,
    options: IRLSOptions | ADMMOptions,
    workers: int | None,
    edge_weights: tuple[np.ndarray, np.ndarray] | None,
    *,
    workspace: Workspace | None = None,
) -> L1Result:
    """Check the arguments, set up the threads or the workspace, and solve."""
    _check_arguments(wrapped, prior, workers, edge_weights, workspace)
    if isinstance(prior, L1Result):
        prior = prior.as_prior()
    if _array_module(wrapped) is not np:
        # On the GPU, CPU threads play no part. The solve uses the workspace's
        # problem until it returns, on the device of the input:
        if workspace is None:
            workspace = default_workspace()
        with workspace.lock, wrapped.device:
            return _unwrap(
                wrapped, prior, options, workers or 1, edge_weights,
                workspace=workspace,
            )
    if workers is None:
        workers = min(os.cpu_count() or 1, _DEFAULT_MAX_THREADS)
    with _cpu_threads(workers):
        return _unwrap(wrapped, prior, options, workers, edge_weights)


def _unwrap(
    wrapped: np.ndarray,
    prior: L1PhasePrior | L1BranchesPrior | None,
    options: IRLSOptions | ADMMOptions,
    workers: int,
    edge_weights: tuple[np.ndarray, np.ndarray] | None,
    *,
    workspace: Workspace | None = None,
) -> L1Result:
    """Prepare the input, solve it with the solver for `options` and its device."""
    xp = _array_module(wrapped)
    data, edge_weights, invalid = _check_input(
        wrapped, prior, edge_weights, keep_wave=xp is not np, workers=workers,
    )
    solver = _solver_for(
        options, gpu=xp is not np, workers=workers, workspace=workspace,
    )
    solution = solver.solve(_Prepared(data, edge_weights, invalid), prior, options)
    return _result(solution, options)


def _solver_for(
    options: IRLSOptions | ADMMOptions,
    *,
    gpu: bool,
    workers: int,
    workspace: Workspace | None,
) -> _Solver[Any]:
    """Return the solver for the method of `options`, on the CPU or the GPU."""
    admm = options.method == "admm"
    if gpu:
        from ._irls_cupy import _GpuAdmm, _GpuIrls  # noqa: PLC0415
        return _GpuAdmm(workspace) if admm else _GpuIrls(workspace)
    if admm:
        from .admm import _CpuAdmm  # noqa: PLC0415
        return _CpuAdmm(workers)
    return _CpuIrls(workers)


def _result(solution: _Solution, options: IRLSOptions | ADMMOptions) -> L1Result:
    """Return the `L1Result` of `solution`."""
    # If the selected round did not converge, its iterate does not solve the
    # system at its epsilon, so a warm start resumes one round coarser:
    resume_epsilon = solution.weights_epsilon
    if not solution.converged:
        resume_epsilon /= options.decay
    resume_epsilon = min(
        max(resume_epsilon, options.epsilon_min), options.epsilon_start,
    )
    return L1Result(
        phase=solution.phase,
        branches=solution.branches,
        weights_x=solution.weights_x,
        weights_y=solution.weights_y,
        rounds=solution.rounds,
        selected_round=solution.selected_round,
        cut_turns=solution.cut_turns,
        cut_cost=solution.cut_cost,
        status=solution.status,
        resume_epsilon=resume_epsilon,
    )


__all__ = [
    "IRLSOptions",
    "L1BranchesPrior",
    "L1PhasePrior",
    "L1Result",
    "L1Round",
    "L1Status",
    "amplitude_edge_weights",
    "unwrap_phase_irls",
]

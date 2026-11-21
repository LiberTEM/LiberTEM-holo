"""Phase unwrapping by iteratively reweighted least squares (IRLS).

See :func:`unwrap_phase_irls` for how it works.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numba
import numpy as np
import scipy.fft
from libertem.common.threading import set_num_threads, set_numba_threads

from libertem_holo.base.align import _array_module

from . import _irls_numba

if TYPE_CHECKING:
    from collections.abc import Callable

TAU = 2 * np.pi

# More threads than this made the solve slower on a 24-core machine, even for
# 2048 x 2048 images: the per-iteration work is too small to spread wider.
_DEFAULT_MAX_THREADS = 8

IRLSStatus = Literal["converged", "iteration_limit"]


@dataclass(frozen=True)
class IRLSOptions:
    """Settings of an IRLS unwrap.

    No default fits every phase image: a difficult one needs more rounds or more
    iterations. See :func:`unwrap_phase_irls` for what the options control.

    Parameters
    ----------
    irls_rounds
        Maximum number of reweighting rounds. Epsilon reaches
        :code:`epsilon_min` in the last of them.
    cg_max_iterations
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
        :code:`irls_rounds`.
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

    irls_rounds: int = 30
    cg_max_iterations: int = 15
    epsilon_start: float = 0.1
    epsilon_min: float = 1e-3
    cg_relative_tolerance: float = 1e-5
    stop_patience: int | None = 3
    stop_tolerance: float = 0.01
    cold_start: Literal["zero", "least_squares"] = "zero"

    def __post_init__(self) -> None:
        """Check that the options describe a valid solve."""
        if self.irls_rounds < 1:
            msg = f"irls_rounds must be at least 1, got {self.irls_rounds}"
            raise ValueError(msg)
        if self.cg_max_iterations < 1:
            msg = f"cg_max_iterations must be at least 1, got {self.cg_max_iterations}"
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
        return (self.epsilon_min / self.epsilon_start) ** (1.0 / self.irls_rounds)


@dataclass
class CGRound:
    """How one IRLS round ended.

    Attributes
    ----------
    iterations
        Number of CG iterations run.
    residuals
        The relative residual before the first iteration and after each one, so
        :code:`iterations + 1` entries; a single :code:`0.0` if the right-hand side
        was zero.
    converged
        Whether CG reached :code:`cg_relative_tolerance` within
        :code:`cg_max_iterations`.
    cut_turns, cut_cost
        The cut count and cut cost of the round's iterate, rounded onto the
        branches of the input (see :func:`unwrap_phase_irls`).

    """

    iterations: int
    residuals: list[float] = field(default_factory=list)
    converged: bool = False
    cut_turns: int | None = None
    cut_cost: float | None = None


@dataclass(frozen=True)
class PhasePrior:
    r"""Start an IRLS solve from an unwrapped phase of a similar input.

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
        :attr:`IRLSResult.resume_epsilon`. The solve resumes the schedule there.

    """

    phase: np.ndarray
    epsilon: float


@dataclass(frozen=True)
class BranchesPrior:
    r"""Start an IRLS solve from the branch field of an earlier solve of the same input.

    For example, to solve the same wrapped phase again with other options. Against a
    different wrapped input, the branches are off by one wherever a value crossed
    :math:`\pm\pi`, which plants :math:`2\pi` spikes in the starting phase and
    slows the solve down badly; use :class:`PhasePrior` there.

    Parameters
    ----------
    branches
        Integer branch field, :attr:`IRLSResult.branches` of the earlier solve.
    epsilon
        :attr:`IRLSResult.resume_epsilon` of the earlier solve.

    """

    branches: np.ndarray
    epsilon: float


@dataclass
class IRLSResult:
    r"""The result of :func:`unwrap_phase_irls`.

    Attributes
    ----------
    phase
        The unwrapped phase as float64. It equals
        :code:`wrapped + 2 * np.pi * branches` exactly.
    branches
        The number of :math:`2\pi` turns added to each pixel of the input, as int32.
    weights_x, weights_y
        Edge weights derived from the selected iterate, with shapes
        :code:`(rows, cols - 1)` and :code:`(rows - 1, cols)`. Low weights mark
        where the solve placed a discontinuity. If the starting point is selected
        without a prior, these are the edge weights.
    rounds
        One :class:`CGRound` per IRLS round that ran. Empty if the starting point
        already had no cuts.
    selected_round
        Index into :attr:`rounds` of the round whose iterate is returned: the
        latest one with the lowest cut cost. -1 for the starting point: the rounded
        L2 solution, or the prior.
    cut_turns
        The cut count of the result (see :func:`unwrap_phase_irls`).
    cut_cost
        The cut count with each edge weighted by its edge weight; equal to
        :attr:`cut_turns` without edge weights.
    status
        :code:`"converged"` if every round's CG reached the tolerance,
        :code:`"iteration_limit"` if at least one round stopped at
        :code:`cg_max_iterations`. Either way, :attr:`phase` is the selected
        iterate, snapped onto the branches of the input.
    resume_epsilon
        Where a solve started from this result resumes the epsilon schedule (see
        :class:`PhasePrior`): the epsilon the returned weights were derived at, or
        one round coarser if the selected round did not converge.

    """

    phase: np.ndarray
    branches: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray
    rounds: list[CGRound]
    selected_round: int
    cut_turns: int
    cut_cost: float
    status: IRLSStatus
    resume_epsilon: float

    @property
    def final_relative_residual(self) -> float:
        """The relative CG residual at the end of the last round."""
        if not self.rounds or not self.rounds[-1].residuals:
            return math.nan
        return self.rounds[-1].residuals[-1]

    def as_prior(self) -> PhasePrior:
        """Return a :class:`PhasePrior` that starts another solve from this result."""
        return PhasePrior(phase=self.phase, epsilon=self.resume_epsilon)


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
    but measured no faster end to end, with more padding on 7- or 11-smooth sizes.
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
        rows, cols = self.padded_shape
        u = np.arange(rows)[:, np.newaxis]
        v = np.arange(cols)[np.newaxis, :]
        symbol = 4 - 2 * np.cos(np.pi * u / rows) - 2 * np.cos(np.pi * v / cols)
        inverse = np.zeros_like(symbol)
        np.divide(1.0, symbol, out=inverse, where=symbol != 0)
        self.inverse_symbol = inverse.astype(dtype)
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


class _WeightedProblem:
    r"""The weighted least-squares problem of one IRLS round.

    Find :math:`\phi` with :math:`A \phi = b`, where
    :math:`A \phi = -\nabla \cdot (w \nabla \phi)` and
    :math:`b = -\nabla \cdot (w g)` for the wrapped gradients :math:`g` and the
    edge weights :math:`w`. :math:`A` is positive semi-definite: for an interior
    pixel with unit weights, it is :code:`4 p - p[left] - p[right] - p[up] -
    p[down]`, the same sign convention as the preconditioner.

    The weights are the fixed edge weights :math:`C` (all ones if none are
    given) times the Charbonnier weights of the current round.
    """

    def __init__(
        self,
        wrapped: np.ndarray,
        workers: int | None,
        edge_weights: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        rows, cols = wrapped.shape
        # Wrapping the target gradients is what makes this an unwrapping problem:
        # a difference of two wrapped values is only known modulo 2π, and this
        # picks the smallest representative.
        self.target_x = _wrap(np.diff(wrapped, axis=1))
        self.target_y = _wrap(np.diff(wrapped, axis=0))
        # The whole turns the wrapping removed from each gradient: a congruent
        # phase `wrapped + 2π branches` disagrees with the target gradient on an
        # edge by `jumps + diff(branches)` turns.
        self.jumps_x = np.rint(
            (np.diff(wrapped, axis=1) - self.target_x) / TAU,
        ).astype(np.int32)
        self.jumps_y = np.rint(
            (np.diff(wrapped, axis=0) - self.target_y) / TAU,
        ).astype(np.int32)
        if edge_weights is None:
            self.edge_x = np.ones((rows, cols - 1), dtype=wrapped.dtype)
            self.edge_y = np.ones((rows - 1, cols), dtype=wrapped.dtype)
        else:
            self.edge_x, self.edge_y = (
                np.asarray(w, dtype=wrapped.dtype) for w in edge_weights
            )
        self.weights_x = self.edge_x.copy()
        self.weights_y = self.edge_y.copy()
        self._row_sums = np.empty(rows, dtype=np.float64)
        self._b = np.empty((rows, cols), dtype=wrapped.dtype)
        self._iterations = 0
        self.precondition = _PoissonPreconditioner(
            (rows, cols), wrapped.dtype, workers=workers,
        )

    def cuts(self, branches: np.ndarray) -> tuple[int, float]:
        """Count the turns by which a congruent phase cuts the wrapped gradients.

        Returns the sum over all edges of :code:`|k|`, where :code:`k` is the
        number of whole turns between the gradient of :code:`wrapped + 2π
        branches` and the wrapped gradient, and the same sum with each edge
        weighted by its edge weight.
        """
        rows = branches.shape[0]
        turns = np.empty(rows, dtype=np.int64)
        cost = np.empty(rows, dtype=np.float64)
        _irls_numba.cuts_per_row(
            branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
            turns, cost,
        )
        return int(turns.sum()), float(cost.sum())

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

    def l2_solution(self) -> np.ndarray:
        """Solve the unweighted problem, approximately.

        Exact up to the padding of the preconditioner, which with unit weights is
        the inverse of the operator. With edge weights, this is only the starting
        point of the weighted solve.
        """
        b = np.zeros(self.precondition.shape, dtype=self.target_x.dtype)
        _negative_divergence(self.target_x, self.target_y, b)
        return np.array(self.precondition(b))

    def reweight(
        self, phi: np.ndarray, epsilon: float, *, unless_settled: bool = False,
    ) -> None:
        """Set the weights from the iterate `phi` at `epsilon`.

        With `unless_settled`, only if the last :meth:`solve` ran an iteration.
        """
        if unless_settled and self._iterations == 0:
            return
        _irls_numba.charbonnier_weights(
            phi, self.target_x, self.target_y, self.edge_x, self.edge_y,
            phi.dtype.type(epsilon * epsilon), self.weights_x, self.weights_y,
        )

    def solve(
        self, phi: np.ndarray, options: IRLSOptions,
    ) -> Callable[[], CGRound]:
        """Run one round's CG on `phi` (in place), and remove its mean if it changed.

        Returns a function that returns the round's report. (On a GPU, the
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

    def congruent(
        self, phi: np.ndarray, wrapped: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Snap `phi` onto the branches of `wrapped`, see :func:`_congruent`."""
        return _congruent(phi, wrapped)

    def run_round(
        self,
        phi: np.ndarray,
        wrapped: np.ndarray,
        *,
        epsilon: float,
        at_end: bool,
        options: IRLSOptions,
        index: int,
    ) -> tuple[CGRound, _Candidate]:
        """Run one IRLS round on `phi` (in place).

        Solve the weighted problem, reweight at `epsilon` (unless the round
        settled: no CG iteration, `at_end` of the schedule), and round the result
        onto the branches. Returns the round's report and its candidate result,
        whose `weights_epsilon` and `converged` the caller sets.
        """
        pending = self.solve(phi, options)
        self.reweight(phi, epsilon, unless_settled=at_end)
        candidate = _Candidate.take(
            self, phi, wrapped,
            weights_epsilon=math.nan, converged=False, round_index=index,
        )
        return pending(), candidate

    def keep(self, candidate: _Candidate) -> _Candidate:
        """Return the candidate, to keep past the next round (it owns its arrays)."""
        return candidate

    def cold_start(
        self, wrapped: np.ndarray, options: IRLSOptions,
    ) -> tuple[np.ndarray, _Candidate, float]:
        """Start a solve without a prior, see :func:`_cold_start`."""
        return _cold_start(self, wrapped, options)

    def warm_start(
        self,
        wrapped: np.ndarray,
        prior: PhasePrior | BranchesPrior,
        options: IRLSOptions,
    ) -> tuple[np.ndarray, _Candidate, float]:
        """Start a solve from a prior, see :func:`_warm_start`."""
        return _warm_start(self, wrapped, prior, options)

    def release(self, candidate: _Candidate) -> _Candidate:
        """Return the result of the solve (it owns its arrays)."""
        return candidate


def _starting_phase(
    prior: PhasePrior | BranchesPrior,
    wrapped: np.ndarray,
) -> np.ndarray:
    """Turn the prior into a phase that is congruent with `wrapped`."""
    xp = _array_module(wrapped)
    if isinstance(prior, PhasePrior):
        raw = wrapped.astype(np.float64)
        branches = xp.rint((xp.asarray(prior.phase, dtype=np.float64) - raw) / TAU)
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
    problem: _WeightedProblem,
    phi: np.ndarray,
    b: np.ndarray,
    options: IRLSOptions,
) -> tuple[CGRound, bool]:
    """Run preconditioned CG on `phi` (updated in place) against `b`.

    Returns the round's report, and whether `phi` was changed.
    """
    norm = math.sqrt(problem.dot(b, b))
    if not norm > 0:
        # Zero right-hand side: already solved, and the relative residual would
        # be 0/0.
        return CGRound(iterations=0, residuals=[0.0], converged=True), False

    report = CGRound(iterations=0)
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

    for iteration in range(options.cg_max_iterations):
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
        problem: _WeightedProblem,
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


def _irls_rounds(
    problem: _WeightedProblem,
    phi: np.ndarray,
    wrapped: np.ndarray,
    weights_epsilon: float,
    options: IRLSOptions,
    *,
    best: _Candidate,
) -> tuple[list[CGRound], _Candidate]:
    """Run the IRLS rounds on `phi` (updated in place).

    `weights_epsilon` is the epsilon the current weights were derived at (infinity
    if they were not derived from an iterate), and `best` the candidate result to
    beat. Returns the report
    of each round, and the best rounded iterate: the latest one with the lowest
    cut cost.
    """
    epsilon = (
        options.epsilon_start
        if math.isinf(weights_epsilon)
        else max(options.epsilon_min, weights_epsilon * options.decay)
    )
    stale = 0
    rounds: list[CGRound] = []
    for index in range(options.irls_rounds):
        # A fixed point: the iterate did not move and the schedule has reached
        # its end, so further rounds would repeat this one. Whether the iterate
        # moved is known only after the solve, so the problem decides itself
        # whether to reweight; on a GPU, a whole round is then one graph.
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
    problem: _WeightedProblem,
    wrapped: np.ndarray,
    options: IRLSOptions,
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
    problem: _WeightedProblem,
    wrapped: np.ndarray,
    prior: PhasePrior | BranchesPrior,
    options: IRLSOptions,
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
    # All checks in one array, so that on a GPU, they cost one synchronization:
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
    prior: PhasePrior | BranchesPrior | None,
    edge_weights: tuple[np.ndarray, np.ndarray] | None,
    *,
    keep_wave: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """Validate the input, and convert it to the precision of the solve.

    Returns the wrapped phase and the edge weights, which for a complex wave
    default to :func:`amplitude_edge_weights`. With `keep_wave`, a complex wave
    without edge weights is returned as it is, for the GPU to convert.
    """
    xp = _array_module(wrapped)
    wrapped = xp.asarray(wrapped)
    if wrapped.ndim != 2 or wrapped.size == 0:  # noqa: PLR2004
        msg = f"wrapped must be a non-empty 2D array, got shape {wrapped.shape}"
        raise ValueError(msg)
    keep_wave = keep_wave and xp.iscomplexobj(wrapped) and edge_weights is None
    if xp.iscomplexobj(wrapped) and not keep_wave:
        if edge_weights is None:
            edge_weights = amplitude_edge_weights(xp.abs(wrapped))
        wrapped = xp.angle(wrapped)
    if edge_weights is not None:
        edge_weights = _check_edge_weights(edge_weights, wrapped.shape)
    if prior is not None:
        prior_shape = np.shape(
            prior.phase if isinstance(prior, PhasePrior) else prior.branches,
        )
        if prior_shape != wrapped.shape:
            msg = (
                f"the prior has shape {prior_shape}, "
                f"but the wrapped phase has shape {wrapped.shape}"
            )
            raise ValueError(msg)
    if keep_wave:
        return wrapped, None
    dtype = np.float32 if wrapped.dtype == np.float32 else np.float64
    return xp.ascontiguousarray(wrapped, dtype=dtype), edge_weights


_AMPLITUDE_FLOOR = 1e-3
_AMPLITUDE_PERCENTILE = 90.0


def amplitude_edge_weights(
    amplitude: np.ndarray,
    floor: float = _AMPLITUDE_FLOOR,
    percentile: float = _AMPLITUDE_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Edge weights for :func:`unwrap_phase_irls` from the amplitude of a wave.

    Where the amplitude is low, the phase is noisy or undefined, for example where
    the specimen blocks the beam. Cuts through such regions should cost little,
    so that the solve puts the discontinuities there and does not spend iterations
    on reconciling their noise.

    The weight of an edge is the smaller amplitude of its two pixels, divided by
    the given percentile of the amplitude and clipped to :code:`[floor, 1]`.
    Clipping at one keeps bright features, such as Fresnel fringes, from
    attracting cuts. The floor keeps regions without signal weakly connected to
    the rest of the image.

    Parameters
    ----------
    amplitude
        Amplitude of the wave, for example :code:`np.abs(wave)`.
    floor
        Smallest weight, relative to the largest.
    percentile
        The percentile of the amplitude that gets full weight. A high percentile
        rather than the median keeps the reference in the beam, even if most of
        the image is blocked.

    Returns
    -------
    weights_x, weights_y
        Weights of the horizontal and vertical edges, with shapes
        :code:`(rows, cols - 1)` and :code:`(rows - 1, cols)`.

    """
    xp = _array_module(amplitude)
    amplitude = xp.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2:  # noqa: PLR2004
        msg = f"amplitude must be a 2D array, got shape {amplitude.shape}"
        raise ValueError(msg)
    if not 0 < floor <= 1:
        msg = f"floor must be in (0, 1], got {floor}"
        raise ValueError(msg)
    reference = xp.percentile(amplitude, percentile)
    if xp is np:
        if not reference > 0:
            # No signal anywhere: no edge is more reliable than another.
            return (
                np.ones((amplitude.shape[0], amplitude.shape[1] - 1)),
                np.ones((amplitude.shape[0] - 1, amplitude.shape[1])),
            )
        relative = np.clip(amplitude / reference, floor, 1.0)
    else:
        # The same, without the synchronization of a host-side check:
        relative = xp.where(
            reference > 0, xp.clip(amplitude / reference, floor, 1.0), 1.0,
        )
    return (
        xp.minimum(relative[:, 1:], relative[:, :-1]),
        xp.minimum(relative[1:, :], relative[:-1, :]),
    )


def unwrap_phase_irls(
    wrapped: np.ndarray,
    prior: PhasePrior | BranchesPrior | IRLSResult | None = None,
    options: IRLSOptions | None = None,
    workers: int | None = None,
    edge_weights: tuple[np.ndarray, np.ndarray] | None = None,
) -> IRLSResult:
    r"""Unwrap a 2D phase by iteratively reweighted least squares.

    The result is congruent with the input: it differs from it by an integer
    multiple of :math:`2\pi` at every pixel. The solve is global and places the
    discontinuities where they cost least.

    The solve runs in float32 for float32 input and in float64 otherwise. The
    result phase is float64, computed from the input values, so it stays exactly
    congruent with them.

    Parameters
    ----------
    wrapped
        The wrapped phase, usually in :math:`[-\pi, \pi]`; any real values work,
        since only differences modulo :math:`2\pi` enter the solve. Or the complex
        wave: then its phase is unwrapped, with edge weights from its amplitude
        (:func:`amplitude_edge_weights`) unless :code:`edge_weights` is given.
    prior
        Start from an earlier solution instead of from scratch, and resume its
        epsilon schedule; see :class:`PhasePrior` and :class:`BranchesPrior`.
        Passing an :class:`IRLSResult` is the same as passing
        :meth:`IRLSResult.as_prior`. A warm start from the previous frame of a
        series typically needs far fewer CG iterations than a cold start.
    options
        Solver settings, :class:`IRLSOptions` by default.
    workers
        Number of threads. By default, the number of CPUs, but at most 8.
        The result does not depend on it.
    edge_weights
        Weights :math:`C` of the horizontal and vertical edges between
        neighbouring pixels, with shapes :code:`(rows, cols - 1)` and
        :code:`(rows - 1, cols)`, finite and greater than zero. Edges with a low
        weight are cheap to cut, so the solve places discontinuities there. Only
        relative weights matter. Without edge weights, all edges weigh the same.

    Returns
    -------
    IRLSResult
        The unwrapped phase, the branch field, the weights of the selected
        iterate, the cut count and cost, and a report of each round.

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
    ambiguous, start out cheap to cut. Starting from the L2 solution instead,
    which smears discontinuities and residues, puts noticeably more pixels on the
    wrong branch in noisy test fields, at the same cut count.

    With edge weights :math:`C`, the objective is the weighted sum
    :math:`\sum C |r|`: each round's weights are :math:`C` times the Charbonnier
    weights.

    After each round, the iterate is snapped onto the branches of the input and
    scored. Its *cut count* is the number of whole turns by which its gradient
    differs from the wrapped gradient, summed over all edges between neighbouring
    pixels: the L1 objective of a congruent result. Its *cut cost* weights each
    edge by :math:`C`; without edge weights, the two are equal. The solve returns
    the latest rounded iterate with the lowest cut cost. The starting point (the
    rounded L2 solution, or the prior) is a candidate too, so the result is never
    worse than that. The solve stops once the cut cost has not improved for
    :code:`stop_patience` rounds, where an improvement must exceed both
    :code:`stop_tolerance` of the cost and half a cut through edges of full
    weight, so that tiny gains in nearly weightless regions do not keep it going.
    It also stops as soon as the cut count is zero, since no result can be better.

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
    if options is None:
        options = IRLSOptions()
    if isinstance(prior, IRLSResult):
        prior = prior.as_prior()
    if _array_module(wrapped) is not np:
        # On the GPU, CPU threads play no part.
        return _unwrap(wrapped, prior, options, workers or 1, edge_weights)
    if workers is None:
        workers = min(os.cpu_count() or 1, _DEFAULT_MAX_THREADS)
    # BLAS only computes dot products here, on one thread, so that their summation
    # order and the result do not depend on the thread count. The kernels and the
    # cosine transforms use `workers` threads. Only the outermost numba setting
    # takes effect, so it comes before `set_num_threads`, which sets numba too.
    with (
        set_numba_threads(min(workers, numba.config.NUMBA_NUM_THREADS)),
        set_num_threads(1),
    ):
        return _unwrap(wrapped, prior, options, workers, edge_weights)


def _unwrap(
    wrapped: np.ndarray,
    prior: PhasePrior | BranchesPrior | None,
    options: IRLSOptions,
    workers: int,
    edge_weights: tuple[np.ndarray, np.ndarray] | None,
) -> IRLSResult:
    """Run :func:`unwrap_phase_irls` with the threads set up."""
    xp = _array_module(wrapped)
    wrapped, edge_weights = _check_input(
        wrapped, prior, edge_weights, keep_wave=xp is not np,
    )
    if xp is np:
        problem = _WeightedProblem(wrapped, workers=workers, edge_weights=edge_weights)
    elif xp.iscomplexobj(wrapped):
        # On the GPU, the phase and the amplitude weights of a wave come from a
        # few fused kernels:
        from ._irls_cupy import problem_for_wave  # noqa: PLC0415
        # The input is only staged; the start (setup, starting point, first
        # candidate) then runs as one CUDA graph:
        problem = problem_for_wave(
            wrapped, _AMPLITUDE_FLOOR, _AMPLITUDE_PERCENTILE, deferred=True,
        )
        wrapped = problem.wrapped
    else:
        from ._irls_cupy import problem_for  # noqa: PLC0415
        problem = problem_for(wrapped, edge_weights, deferred=True)
        wrapped = problem.wrapped

    if prior is None:
        phi, best, weights_epsilon = problem.cold_start(wrapped, options)
    else:
        phi, best, weights_epsilon = problem.warm_start(wrapped, prior, options)

    if best.cut_turns == 0:
        rounds: list[CGRound] = []
    else:
        rounds, best = _irls_rounds(
            problem, phi, wrapped, weights_epsilon, options, best=best,
        )
    # If the selected round did not converge, its iterate does not solve the
    # system at its epsilon, so a warm start resumes one round coarser:
    resume_epsilon = best.weights_epsilon
    if not best.converged:
        resume_epsilon /= options.decay
    resume_epsilon = min(
        max(resume_epsilon, options.epsilon_min), options.epsilon_start,
    )

    best = problem.release(best)
    return IRLSResult(
        phase=best.phase,
        branches=best.branches,
        weights_x=best.weights_x,
        weights_y=best.weights_y,
        rounds=rounds,
        selected_round=best.round_index,
        cut_turns=best.cut_turns,
        cut_cost=best.cut_cost,
        status="converged" if all(r.converged for r in rounds) else "iteration_limit",
        resume_epsilon=resume_epsilon,
    )


__all__ = [
    "BranchesPrior",
    "CGRound",
    "IRLSOptions",
    "IRLSResult",
    "IRLSStatus",
    "PhasePrior",
    "amplitude_edge_weights",
    "unwrap_phase_irls",
]

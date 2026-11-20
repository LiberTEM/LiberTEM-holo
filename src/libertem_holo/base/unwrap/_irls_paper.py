r"""IRLS phase unwrapping as described in the paper, for comparison.

A reference implementation of Algorithms 1 and 2 of B. Dubois-Taine, R. Akiki and
A. d'Aspremont, "Iteratively Reweighted Least Squares for Phase Unwrapping",
arXiv:2401.09961 (2024), with the CG iteration heuristic of its section 4.3. It
exists to compare :func:`~libertem_holo.base.unwrap.unwrap_phase_irls` against;
it is not part of the public API. Numbers in parentheses refer to equations of
the paper.

The paper minimizes the smoothed, weighted L1 norm of the gradient residuals,
with slack variables :math:`V` that are tied to the residuals by a quadratic
penalty:

.. math::

    F_\delta(U, V) = \sum \sqrt{C^2 V^2 + \delta^2}
        + \frac{1}{2\tau} \|\nabla U - G - V\|^2

Each IRLS iteration fixes the weights :math:`W = \sqrt{C^2 V^2 + \delta^2}` and
solves the linear system (12) for :math:`(U, V)` jointly by preconditioned
conjugate gradients, with the block-diagonal preconditioner (24). The paper
solves its :math:`U` block through eigendecompositions of :math:`S^\top S` and
:math:`T T^\top`; their eigenvectors are the DCT-II basis, so this uses cosine
transforms instead, which give the same result faster.

Where the paper leaves details open, this implementation chooses:

- The wrapped gradients are wrapped into :math:`[-\pi, \pi]`; the paper writes
  "mod :math:`2\pi`".
- Each CG solve starts from the previous iterate.
- The maximum number of IRLS iterations, a cap on the CG budget and a relative
  CG residual tolerance (see :class:`PaperOptions`) are not given in the paper.
- The CG budget is rounded up after scaling by :math:`\alpha`.
- The rounded ("congruent") result removes the median offset before rounding,
  as :func:`~libertem_holo.base.unwrap.unwrap_phase_irls` does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .irls import (
    TAU,
    _check_edge_weights,
    _congruent,
    _negative_divergence,
    _PoissonPreconditioner,
    _wrap,
    amplitude_edge_weights,
)


@dataclass(frozen=True)
class PaperOptions:
    r"""Parameters of the paper's algorithm.

    The defaults of :code:`tau`, :code:`delta`, :code:`cg_budget_start`,
    :code:`improvement_tolerance` and :code:`cg_budget_factor` are the values
    the paper uses in its experiments. The others are not given in the paper.

    Parameters
    ----------
    tau
        Weight :math:`\tau` of the quadratic penalty that ties the slack variables
        to the gradient residuals.
    delta
        Smoothing :math:`\delta` of the L1 norm.
    cg_budget_start
        CG iterations allowed in the first IRLS iteration, :math:`m_{CG}(0)`.
    improvement_tolerance
        :math:`\epsilon_{tol}`: relative improvements of the objective below this
        count as insignificant.
    cg_budget_factor
        :math:`\alpha`: the factor by which the CG budget grows.
    max_iterations
        Upper limit on the number of IRLS iterations.
    max_cg_budget
        Upper limit on the CG budget.
    cg_relative_tolerance
        CG also stops once the residual norm has fallen to this fraction of the
        norm of the right-hand side.
    pad_preconditioner
        Zero-pad the cosine transforms to fast lengths, as
        :func:`~libertem_holo.base.unwrap.unwrap_phase_irls` does. Off by
        default, so that the :math:`U` block is the exact inverse the paper uses.

    """

    tau: float = 1e-2
    delta: float = 1e-6
    cg_budget_start: int = 5
    improvement_tolerance: float = 1e-3
    cg_budget_factor: float = 1.7
    max_iterations: int = 200
    max_cg_budget: int = 1000
    cg_relative_tolerance: float = 1e-8
    pad_preconditioner: bool = False


@dataclass
class PaperIteration:
    r"""What one IRLS iteration of the paper's algorithm did.

    Attributes
    ----------
    cg_budget
        CG iterations allowed.
    cg_iterations
        CG iterations run.
    objective
        :math:`F_\delta` after the iteration.
    relative_improvement
        :math:`\Delta` of section 4.3: the relative decrease of the objective by
        the weight update that follows the iteration.

    """

    cg_budget: int
    cg_iterations: int
    objective: float
    relative_improvement: float


@dataclass
class PaperResult:
    """The result of :func:`unwrap_phase_irls_paper`.

    Attributes
    ----------
    unwrapped
        The solution :math:`U` with zero mean. The paper does not make it
        congruent with the input.
    phase, branches
        :attr:`unwrapped` rounded onto the branches of the input, like the
        result of :func:`~libertem_holo.base.unwrap.unwrap_phase_irls`.
    cut_turns, cut_cost
        Cut count and edge-weighted cut cost of :attr:`phase`, as in
        :class:`~libertem_holo.base.unwrap.IRLSResult`.
    slack_x, slack_y
        The slack variables :math:`V^h` and :math:`V^v`.
    iterations
        One :class:`PaperIteration` per IRLS iteration.
    stopped_by
        :code:`"heuristic"` if the CG heuristic of the paper ended the solve,
        :code:`"max_iterations"` if :attr:`PaperOptions.max_iterations` did.

    """

    unwrapped: np.ndarray
    phase: np.ndarray
    branches: np.ndarray
    cut_turns: int
    cut_cost: float
    slack_x: np.ndarray
    slack_y: np.ndarray
    iterations: list[PaperIteration] = field(default_factory=list)
    stopped_by: Literal["heuristic", "max_iterations"] = "max_iterations"


_Blocks = tuple[np.ndarray, np.ndarray, np.ndarray]


def _dot(a: _Blocks, b: _Blocks) -> float:
    return float(sum(np.vdot(x, y) for x, y in zip(a, b, strict=True)))


class _PaperSystem:
    """The linear system (12) of the paper, in the unknowns :math:`(U, V^h, V^v)`."""

    def __init__(
        self,
        wrapped: np.ndarray,
        edge_weights: tuple[np.ndarray, np.ndarray],
        options: PaperOptions,
        workers: int | None,
    ) -> None:
        dtype = wrapped.dtype
        self.tau = dtype.type(options.tau)
        self.delta_squared = dtype.type(options.delta**2)
        self.target_x = _wrap(np.diff(wrapped, axis=1))
        self.target_y = _wrap(np.diff(wrapped, axis=0))
        self.c_squared_x = np.square(edge_weights[0]).astype(dtype)
        self.c_squared_y = np.square(edge_weights[1]).astype(dtype)
        self.d_x = np.empty_like(self.target_x)
        self.d_y = np.empty_like(self.target_y)
        self.poisson = _PoissonPreconditioner(
            wrapped.shape, dtype, workers=workers, pad=options.pad_preconditioner,
        )

    def irls_weights(self, slack: _Blocks) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute :math:`W = \sqrt{C^2 V^2 + \delta^2}` for the slack variables."""
        _, slack_x, slack_y = slack
        return (
            np.sqrt(self.c_squared_x * slack_x**2 + self.delta_squared),
            np.sqrt(self.c_squared_y * slack_y**2 + self.delta_squared),
        )

    def set_weights(self, weights: tuple[np.ndarray, np.ndarray]) -> None:
        """Set :math:`D = C^2 / W` for the next solve."""
        np.divide(self.c_squared_x, weights[0], out=self.d_x)
        np.divide(self.c_squared_y, weights[1], out=self.d_y)

    def _misfit(self, x: _Blocks) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute :math:`\nabla U - V`, per axis."""
        u, slack_x, slack_y = x
        return np.diff(u, axis=1) - slack_x, np.diff(u, axis=0) - slack_y

    def apply(self, x: _Blocks) -> _Blocks:
        """Multiply by the matrix :math:`A` of (21)."""
        _, slack_x, slack_y = x
        misfit_x, misfit_y = self._misfit(x)
        out_u = _negative_divergence(misfit_x, misfit_y, np.empty_like(x[0]))
        out_u /= self.tau
        return (
            out_u,
            self.d_x * slack_x - misfit_x / self.tau,
            self.d_y * slack_y - misfit_y / self.tau,
        )

    def right_hand_side(self) -> _Blocks:
        """Compute the right-hand side :math:`b` of (12)."""
        rows, cols = self.target_y.shape[0] + 1, self.target_x.shape[1] + 1
        b_u = _negative_divergence(
            self.target_x, self.target_y, np.empty((rows, cols), self.target_x.dtype),
        )
        return (b_u / self.tau, -self.target_x / self.tau, -self.target_y / self.tau)

    def precondition(self, r: _Blocks) -> _Blocks:
        """Solve with the block-diagonal preconditioner (24)."""
        r_u, r_x, r_y = r
        return (
            self.tau * np.asarray(self.poisson(r_u)),
            r_x / (self.d_x + 1 / self.tau),
            r_y / (self.d_y + 1 / self.tau),
        )

    def objective(self, x: _Blocks, weights: tuple[np.ndarray, np.ndarray]) -> float:
        r"""Compute :math:`H_\delta(U, V, W)` of (10).

        It equals :math:`F_\delta` when :math:`W` are the IRLS weights of
        :math:`V`.
        """
        _, slack_x, slack_y = x
        misfit_x, misfit_y = self._misfit(x)
        total = 0.0
        for c_squared, slack, w in (
            (self.c_squared_x, slack_x, weights[0]),
            (self.c_squared_y, slack_y, weights[1]),
        ):
            smoothed = (c_squared * slack**2 + self.delta_squared) / w + w
            total += 0.5 * float(np.sum(smoothed))
        for misfit, target in ((misfit_x, self.target_x), (misfit_y, self.target_y)):
            total += float(np.sum((misfit - target) ** 2)) / (2 * float(self.tau))
        return total


def _conjugate_gradients(
    system: _PaperSystem,
    x: _Blocks,
    b: _Blocks,
    budget: int,
    tolerance: float,
) -> tuple[_Blocks, int]:
    """Algorithm 2 of the paper, starting from `x`, for at most `budget` iterations."""
    norm = math.sqrt(_dot(b, b))
    applied = system.apply(x)
    r = tuple(bi - ai for bi, ai in zip(b, applied, strict=True))
    if not norm > 0 or math.sqrt(_dot(r, r)) <= tolerance * norm:
        return x, 0
    z = system.precondition(r)
    p = z
    rho = _dot(r, z)
    for iteration in range(budget):
        q = system.apply(p)
        curvature = _dot(p, q)
        if not curvature > 0:
            return x, iteration
        alpha = rho / curvature
        x = tuple(xi + alpha * pi for xi, pi in zip(x, p, strict=True))
        r = tuple(ri - alpha * qi for ri, qi in zip(r, q, strict=True))
        if math.sqrt(_dot(r, r)) <= tolerance * norm:
            return x, iteration + 1
        z = system.precondition(r)
        rho_next = _dot(r, z)
        p = tuple(zi + (rho_next / rho) * pi for zi, pi in zip(z, p, strict=True))
        rho = rho_next
    return x, budget


def _cuts(
    branches: np.ndarray,
    wrapped: np.ndarray,
    edge_weights: tuple[np.ndarray, np.ndarray],
) -> tuple[int, float]:
    """Cut count and edge-weighted cut cost of `wrapped + 2π branches`."""
    turns = 0
    cost = 0.0
    for axis, weights in ((1, edge_weights[0]), (0, edge_weights[1])):
        difference = np.diff(wrapped, axis=axis)
        jumps = np.rint((difference - _wrap(difference)) / TAU).astype(np.int64)
        cuts = np.abs(jumps + np.diff(branches, axis=axis))
        turns += int(cuts.sum())
        cost += float(np.vdot(cuts, weights))
    return turns, cost


def unwrap_phase_irls_paper(
    wrapped: np.ndarray,
    options: PaperOptions | None = None,
    edge_weights: tuple[np.ndarray, np.ndarray] | None = None,
    workers: int | None = None,
) -> PaperResult:
    """Unwrap a 2D phase with the paper's algorithm.

    Parameters
    ----------
    wrapped
        The wrapped phase, or the complex wave.
    options
        Parameters, :class:`PaperOptions` by default.
    edge_weights
        The edge weights :math:`C^h` and :math:`C^v` of the paper, with shapes
        :code:`(rows, cols - 1)` and :code:`(rows - 1, cols)`. By default, all
        ones for a phase, and
        :func:`~libertem_holo.base.unwrap.amplitude_edge_weights` for a complex
        wave.
    workers
        Number of threads for the cosine transforms.

    Returns
    -------
    PaperResult

    """
    if options is None:
        options = PaperOptions()
    wrapped = np.asarray(wrapped)
    if wrapped.ndim != 2 or wrapped.size == 0:  # noqa: PLR2004
        msg = f"wrapped must be a non-empty 2D array, got shape {wrapped.shape}"
        raise ValueError(msg)
    if np.iscomplexobj(wrapped):
        if edge_weights is None:
            edge_weights = amplitude_edge_weights(np.abs(wrapped))
        wrapped = np.angle(wrapped)
    dtype = np.float32 if wrapped.dtype == np.float32 else np.float64
    wrapped = np.ascontiguousarray(wrapped, dtype=dtype)
    rows, cols = wrapped.shape
    if edge_weights is None:
        edge_weights = (np.ones((rows, cols - 1)), np.ones((rows - 1, cols)))
    edge_weights = _check_edge_weights(edge_weights, wrapped.shape)

    system = _PaperSystem(wrapped, edge_weights, options, workers)
    b = system.right_hand_side()
    # Algorithm 1, lines 1 and 2: U = 0, so V = ∇U - G = -G.
    x: _Blocks = (np.zeros_like(wrapped), -system.target_x, -system.target_y)
    weights = system.irls_weights(x)

    result = PaperResult(
        unwrapped=x[0], phase=x[0], branches=x[0], cut_turns=0, cut_cost=0.0,
        slack_x=x[1], slack_y=x[2],
    )
    budget = options.cg_budget_start
    increased = False
    for _ in range(options.max_iterations):
        system.set_weights(weights)
        x, cg_iterations = _conjugate_gradients(
            system, x, b, budget, options.cg_relative_tolerance,
        )
        x[0][...] -= x[0].mean()
        # Section 4.3: the relative improvement of the objective by the weight
        # update decides about the next CG budget, and about stopping.
        before = system.objective(x, weights)
        weights = system.irls_weights(x)
        after = system.objective(x, weights)
        improvement = (before - after) / before if before > 0 else 0.0
        result.iterations.append(PaperIteration(
            cg_budget=budget,
            cg_iterations=cg_iterations,
            objective=after,
            relative_improvement=improvement,
        ))
        if improvement > options.improvement_tolerance:
            increased = False
        elif increased:
            result.stopped_by = "heuristic"
            break
        else:
            new_budget = min(
                math.ceil(options.cg_budget_factor * budget), options.max_cg_budget,
            )
            increased = new_budget != budget
            budget = new_budget

    u, slack_x, slack_y = x
    phase, branches = _congruent(u, wrapped)
    result.cut_turns, result.cut_cost = _cuts(branches, wrapped, edge_weights)
    result.unwrapped = u.astype(np.float64)
    result.phase = phase
    result.branches = branches
    result.slack_x = slack_x
    result.slack_y = slack_y
    return result

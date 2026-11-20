"""Fused, row-parallel kernels for the IRLS phase unwrapping.

Each kernel replaces a sequence of numpy passes over whole images, and their
temporaries, with one pass. Elementwise, they do the same floating-point operations
in the same order as those numpy passes, so those results are identical.

Dot products are summed per row, sequentially and in float64, into `row_sums`; the
caller adds up the rows with :code:`row_sums.sum()`. The result is the same on any
number of threads.

All arrays are 2D and C-contiguous. Horizontal edges have shape
:code:`(rows, cols - 1)`, vertical edges :code:`(rows - 1, cols)`.
"""
# Kernels take their arrays as positional arguments:
# ruff: noqa: PLR0917
from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True, parallel=True)
def apply_operator(
    p: np.ndarray,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    out: np.ndarray,
    row_sums: np.ndarray,
) -> None:
    """Compute :code:`-div(w grad(p))` into `out`, and the rows of :code:`p · out`.

    Each edge adds its flux :code:`w (p[end] - p[start])` to the pixel at its end
    and subtracts it from the pixel at its start, in the order of
    :func:`~libertem_holo.base.unwrap.irls._negative_divergence`.
    """
    rows, cols = p.shape
    for i in numba.prange(rows):
        row_sum = 0.0
        for j in range(cols):
            # Accumulating in `out` keeps every step in its dtype, like numpy.
            out[i, j] = 0
            if j > 0:
                out[i, j] += weights_x[i, j - 1] * (p[i, j] - p[i, j - 1])
            if j < cols - 1:
                out[i, j] -= weights_x[i, j] * (p[i, j + 1] - p[i, j])
            if i > 0:
                out[i, j] += weights_y[i - 1, j] * (p[i, j] - p[i - 1, j])
            if i < rows - 1:
                out[i, j] -= weights_y[i, j] * (p[i + 1, j] - p[i, j])
            row_sum += np.float64(p[i, j]) * np.float64(out[i, j])
        row_sums[i] = row_sum


@numba.njit(cache=True, parallel=True)
def negative_divergence_of_weighted(
    weights_x: np.ndarray,
    target_x: np.ndarray,
    weights_y: np.ndarray,
    target_y: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute :code:`-div(w g)` of the weighted target gradients into `out`."""
    rows, cols = out.shape
    for i in numba.prange(rows):
        for j in range(cols):
            out[i, j] = 0
            if j > 0:
                out[i, j] += weights_x[i, j - 1] * target_x[i, j - 1]
            if j < cols - 1:
                out[i, j] -= weights_x[i, j] * target_x[i, j]
            if i > 0:
                out[i, j] += weights_y[i - 1, j] * target_y[i - 1, j]
            if i < rows - 1:
                out[i, j] -= weights_y[i, j] * target_y[i, j]


@numba.njit(cache=True, parallel=True)
def cg_update(
    phi: np.ndarray,
    direction: np.ndarray,
    residual: np.ndarray,
    operated: np.ndarray,
    alpha: float,
    row_sums: np.ndarray,
) -> None:
    """Step the iterate and the residual, :code:`phi += alpha d; r -= alpha A d`.

    Also sums the rows of :code:`r · r` for the new residual.
    """
    rows, cols = phi.shape
    for i in numba.prange(rows):
        row_sum = 0.0
        for j in range(cols):
            phi[i, j] += alpha * direction[i, j]
            residual[i, j] -= alpha * operated[i, j]
            value = np.float64(residual[i, j])
            row_sum += value * value
        row_sums[i] = row_sum


@numba.njit(cache=True, parallel=True)
def dot_rows(a: np.ndarray, b: np.ndarray, row_sums: np.ndarray) -> None:
    """Sum the rows of :code:`a · b`."""
    rows, cols = a.shape
    for i in numba.prange(rows):
        row_sum = 0.0
        for j in range(cols):
            row_sum += np.float64(a[i, j]) * np.float64(b[i, j])
        row_sums[i] = row_sum


@numba.njit(cache=True, parallel=True)
def next_direction(direction: np.ndarray, z: np.ndarray, beta: float) -> None:
    """Compute the next search direction :code:`d = beta d + z` in place."""
    rows, cols = direction.shape
    for i in numba.prange(rows):
        for j in range(cols):
            direction[i, j] = direction[i, j] * beta + z[i, j]


@numba.njit(cache=True, parallel=True)
def charbonnier_weights(
    phi: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    squared: float,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
) -> None:
    """Set :code:`w = C eps^2 / sqrt((grad phi - g)^2 + eps^2)` per edge."""
    rows, cols = phi.shape
    for i in numba.prange(rows):
        for j in range(cols - 1):
            r = (phi[i, j + 1] - phi[i, j]) - target_x[i, j]
            weights_x[i, j] = (squared / np.sqrt(r * r + squared)) * edge_x[i, j]
        if i < rows - 1:
            for j in range(cols):
                r = (phi[i + 1, j] - phi[i, j]) - target_y[i, j]
                weights_y[i, j] = (squared / np.sqrt(r * r + squared)) * edge_y[i, j]


@numba.njit(cache=True, parallel=True)
def wrapped_difference(
    phi: np.ndarray, wrapped: np.ndarray, tau: float, out: np.ndarray,
) -> None:
    """Compute :code:`phi - wrapped`, wrapped into [-pi, pi], into `out`."""
    rows, cols = phi.shape
    for i in numba.prange(rows):
        for j in range(cols):
            value = phi[i, j] - wrapped[i, j]
            out[i, j] = value - np.rint(value / tau) * tau


@numba.njit(cache=True, parallel=True)
def branches_and_phase(
    phi: np.ndarray,
    wrapped: np.ndarray,
    offset: float,
    tau: float,
    branches: np.ndarray,
    phase: np.ndarray,
) -> None:
    """Round :code:`phi - offset` onto the branches of `wrapped`.

    `branches` is int32, `phase` float64: :code:`wrapped + 2 pi branches`.
    """
    rows, cols = phi.shape
    for i in numba.prange(rows):
        for j in range(cols):
            branch = np.rint((phi[i, j] - offset - wrapped[i, j]) / tau)
            branches[i, j] = np.int32(branch)
            phase[i, j] = np.float64(wrapped[i, j]) + 2 * np.pi * np.float64(
                branches[i, j],
            )


@numba.njit(cache=True, parallel=True)
def cuts_per_row(
    branches: np.ndarray,
    jumps_x: np.ndarray,
    jumps_y: np.ndarray,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    turns: np.ndarray,
    cost: np.ndarray,
) -> None:
    """Per row, sum the cut turns :code:`|jumps + diff(branches)|` and their cost.

    The cost weights each edge's turns by its edge weight. A row owns its
    horizontal edges and the vertical edges below it.
    """
    rows, cols = branches.shape
    for i in numba.prange(rows):
        row_turns = 0
        row_cost = 0.0
        for j in range(cols - 1):
            k = abs(jumps_x[i, j] + (branches[i, j + 1] - branches[i, j]))
            row_turns += k
            row_cost += k * np.float64(edge_x[i, j])
        if i < rows - 1:
            for j in range(cols):
                k = abs(jumps_y[i, j] + (branches[i + 1, j] - branches[i, j]))
                row_turns += k
                row_cost += k * np.float64(edge_y[i, j])
        turns[i] = row_turns
        cost[i] = row_cost

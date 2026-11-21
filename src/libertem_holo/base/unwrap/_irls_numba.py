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

import math

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


# ---- Setup: targets, order statistics, amplitude edge weights --------------------


@numba.njit(cache=True, parallel=True)
def targets_and_jumps(
    wrapped: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    jumps_x: np.ndarray,
    jumps_y: np.ndarray,
) -> None:
    """Compute the wrapped target gradients and their whole-turn jumps, in one pass.

    As :code:`_wrap(np.diff(wrapped))` and :code:`np.rint((np.diff(wrapped) -
    target) / TAU)`: the same operations, in the dtype of `wrapped`.
    """
    rows, cols = wrapped.shape
    tau = wrapped.dtype.type(2 * np.pi)
    for i in numba.prange(rows):
        for j in range(cols):
            if j < cols - 1:
                d = wrapped[i, j + 1] - wrapped[i, j]
                g = d - np.rint(d / tau) * tau
                target_x[i, j] = g
                jumps_x[i, j] = np.int32(np.rint((d - g) / tau))
            if i < rows - 1:
                d = wrapped[i + 1, j] - wrapped[i, j]
                g = d - np.rint(d / tau) * tau
                target_y[i, j] = g
                jumps_y[i, j] = np.int32(np.rint((d - g) / tau))


_SELECT_BINS = 4096
_SELECT_SMALL = 1 << 15  # sort what is left at this size


@numba.njit(cache=True, parallel=True)
def _min_max_nan(x: np.ndarray, chunks: int) -> tuple[float, float, bool]:
    n = x.size
    chunks = max(1, min(chunks, n))
    per = (n + chunks - 1) // chunks
    lows = np.full(chunks, np.inf)
    highs = np.full(chunks, -np.inf)
    nans = np.zeros(chunks, dtype=np.bool_)
    for c in numba.prange(chunks):
        lo, hi, nan = np.inf, -np.inf, False
        for k in range(c * per, min(n, (c + 1) * per)):
            v = np.float64(x[k])
            if np.isnan(v):
                nan = True
            else:
                lo = min(lo, v)
                hi = max(hi, v)
        lows[c], highs[c], nans[c] = lo, hi, nan
    return lows.min(), highs.max(), nans.any()


@numba.njit(cache=True, parallel=True)
def _bucket(
    x: np.ndarray, lo: float, hi: float, k: int, chunks: int,
) -> tuple[np.ndarray, int]:
    """Return the values of the histogram bucket over [lo, hi] that holds rank k.

    Returns them and the number of values in lower buckets. The bucket of a
    value is monotonic in it, so ranks carry over.
    """
    n = x.size
    chunks = max(1, min(chunks, n))
    per = (n + chunks - 1) // chunks
    scale = _SELECT_BINS / (hi - lo)
    hist = np.zeros((chunks, _SELECT_BINS), dtype=np.int64)
    for c in numba.prange(chunks):
        for i in range(c * per, min(n, (c + 1) * per)):
            hist[c, min(int((np.float64(x[i]) - lo) * scale), _SELECT_BINS - 1)] += 1
    totals = hist.sum(axis=0)
    before = 0
    b = 0
    while before + totals[b] <= k:
        before += totals[b]
        b += 1
    offsets = np.zeros(chunks + 1, dtype=np.int64)
    for c in range(chunks):
        offsets[c + 1] = offsets[c] + hist[c, b]
    values = np.empty(offsets[chunks], dtype=np.float64)
    for c in numba.prange(chunks):
        o = offsets[c]
        for i in range(c * per, min(n, (c + 1) * per)):
            v = np.float64(x[i])
            if min(int((v - lo) * scale), _SELECT_BINS - 1) == b:
                values[o] = v
                o += 1
    return values, before


def _select(x: np.ndarray, k: int) -> float:
    """Return the value of rank k of finite values `x`, exactly (as np.partition)."""
    chunks = numba.get_num_threads() * 4
    while x.size > _SELECT_SMALL:
        lo, hi, _ = _min_max_nan(x, chunks)
        if lo == hi:  # all values equal
            return float(lo)
        x, before = _bucket(x, lo, hi, k, chunks)
        k -= before
    return float(np.partition(np.asarray(x, dtype=np.float64), k)[k])


def select_pair(x: np.ndarray, k0: int, k1: int) -> tuple[float, float, bool]:
    """Return the values of ranks k0 <= k1 in the 1D array `x`, and if it has NaN.

    Exact, like :code:`np.partition`: parallel histograms over the value range
    narrow down the bucket of each rank (the bucket of a value is monotonic in
    it), until few values are left to partition.
    """
    lo, hi, nan = _min_max_nan(x, numba.get_num_threads() * 4)
    if nan:
        return math.nan, math.nan, True
    if not (np.isfinite(lo) and np.isfinite(hi)):
        s = np.partition(np.asarray(x, dtype=np.float64), (k0, k1))
        return float(s[k0]), float(s[k1]), False
    return _select(x, k0), _select(x, k1), False


@numba.njit(cache=True, parallel=True)
def amplitude_edges(
    amplitude: np.ndarray,
    reference: float,
    floor: float,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    row_max: np.ndarray,
    row_ok: np.ndarray,
) -> None:
    """Edge weights from the amplitude, as `irls.amplitude_edge_weights`, in float64.

    :code:`np.minimum` of :code:`np.clip(amplitude / reference, floor, 1)` of the
    edge's pixels (NaN propagates through both); per row also the largest weight,
    and whether all are finite and greater than zero (as `_check_edge_weights`).
    """
    rows, cols = amplitude.shape
    for i in numba.prange(rows):
        largest = -np.inf
        ok = True
        for j in range(cols):
            a = _clip(amplitude[i, j] / reference, floor)
            if j < cols - 1:
                w = _nan_min(a, _clip(amplitude[i, j + 1] / reference, floor))
                edge_x[i, j] = w
                ok = ok and np.isfinite(w) and w > 0
                largest = max(largest, w)
            if i < rows - 1:
                w = _nan_min(a, _clip(amplitude[i + 1, j] / reference, floor))
                edge_y[i, j] = w
                ok = ok and np.isfinite(w) and w > 0
                largest = max(largest, w)
        row_max[i] = largest
        row_ok[i] = ok


@numba.njit(cache=True, inline="always")
def _clip(v: float, floor: float) -> float:
    """:code:`np.clip(v, floor, 1.0)`, NaN propagating."""
    if np.isnan(v):
        return v
    return min(max(v, floor), 1.0)


@numba.njit(cache=True, inline="always")
def _nan_min(a: float, b: float) -> float:
    """:code:`np.minimum(a, b)`, NaN propagating."""
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return min(a, b)


@numba.njit(cache=True, parallel=True)
def negative_divergence(
    flux_x: np.ndarray, flux_y: np.ndarray, out: np.ndarray,
) -> None:
    """Compute :code:`-div` of edge values into `out` (`irls._negative_divergence`)."""
    rows, cols = out.shape
    for i in numba.prange(rows):
        for j in range(cols):
            out[i, j] = 0
            if j > 0:
                out[i, j] += flux_x[i, j - 1]
            if j < cols - 1:
                out[i, j] -= flux_x[i, j]
            if i > 0:
                out[i, j] += flux_y[i - 1, j]
            if i < rows - 1:
                out[i, j] -= flux_y[i, j]


@numba.njit(cache=True, parallel=True)
def scale_into(src: np.ndarray, divisor: float, out: np.ndarray) -> None:
    """Compute :code:`out[...] = src / divisor` (float64, then cast to out's dtype)."""
    rows, cols = src.shape
    for i in numba.prange(rows):
        for j in range(cols):
            out[i, j] = src[i, j] / divisor

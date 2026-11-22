"""Fused, row-parallel CPU kernels for the ADMM phase unwrapping.

The same arithmetic, in the same order, as the numba CUDA kernels in
`_irls_numba_cuda`: `_edge` as `admm_edge`, `admm_update` and `admm_update_rows`
as `admm_update`, `circular_sums` as `circular_offset`, `weights_from_phase` as
`admm_weights` (products are not fused into multiply-adds on either side). So
the CPU and the GPU solver differ only by the rounding of their cosine transforms
and the summation order of the offset's sums.

All arrays are 2D and C-contiguous. Horizontal edges have shape
:code:`(rows, cols - 1)`, vertical edges :code:`(rows - 1, cols)`.
"""
# Kernels take their arrays as positional arguments:
# ruff: noqa: PLR0917, C901
from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(cache=True, inline="always")
def _edge(
    grad: float, g: float, c: float, lam: float, rho: float, inv_rho: float,
    delta: float,
) -> tuple[float, float]:
    """One edge: z = prox of (c / rho) huber_delta at t; returns (lam', v).

    t = grad u - g + lam / rho; lam' = lam + rho (grad u - g - z); v = g + z -
    lam' / rho, the edge's target of the next u-solve.
    """
    tau = c * inv_rho
    t = grad - g + lam * inv_rho
    if abs(t) <= delta + tau:
        z = t * delta / (delta + tau)
    elif t > 0:
        z = t - tau
    else:
        z = t + tau
    lam_new = lam + rho * (grad - g - z)
    return lam_new, g + z - lam_new * inv_rho


@numba.njit(cache=True, parallel=True)
def admm_update(
    u: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    lx_new: np.ndarray,
    ly_new: np.ndarray,
    rho: float,
    delta: float,
    rhs: np.ndarray,
) -> None:
    """One ADMM step for min sum C huber_delta(grad u - g), from u.

    The z- and dual updates on the 4 edges of each pixel (each pixel writes the
    duals of its right and lower edge, into the other buffer: rows run in
    parallel), and the right-hand side G^T v of the next u-solve into `rhs`.
    `delta` is epsilon, the Huber scale.
    """
    rows, cols = u.shape
    t = u.dtype.type
    rho_t = t(rho)
    inv_rho = t(1.0 / rho)
    delta_t = t(delta)
    for i in numba.prange(rows):
        for j in range(cols):
            c = u[i, j]
            acc = t(0)
            if j < cols - 1:  # right edge (own): contributes -v
                lam, val = _edge(u[i, j + 1] - c, gx[i, j], cx[i, j], lx[i, j],
                                 rho_t, inv_rho, delta_t)
                lx_new[i, j] = lam
                acc -= val
            if j > 0:  # left edge: contributes +v
                _, val = _edge(c - u[i, j - 1], gx[i, j - 1], cx[i, j - 1],
                               lx[i, j - 1], rho_t, inv_rho, delta_t)
                acc += val
            if i < rows - 1:  # lower edge (own)
                lam, val = _edge(u[i + 1, j] - c, gy[i, j], cy[i, j], ly[i, j],
                                 rho_t, inv_rho, delta_t)
                ly_new[i, j] = lam
                acc -= val
            if i > 0:  # upper edge
                _, val = _edge(c - u[i - 1, j], gy[i - 1, j], cy[i - 1, j],
                               ly[i - 1, j], rho_t, inv_rho, delta_t)
                acc += val
            rhs[i, j] = acc


@numba.njit(cache=True, parallel=True)
def circular_sums(
    phi: np.ndarray,
    wrapped: np.ndarray,
    tau: float,
    row_sin: np.ndarray,
    row_cos: np.ndarray,
) -> None:
    """Sum sin and cos of wrap(phi - wrapped) per row, in float64.

    Into `row_sin` and `row_cos`.
    """
    rows, cols = phi.shape
    tau_t = phi.dtype.type(tau)
    for i in numba.prange(rows):
        s = 0.0
        c = 0.0
        for j in range(cols):
            v = phi[i, j] - wrapped[i, j]
            v = v - np.rint(v / tau_t) * tau_t
            s += math.sin(np.float64(v))
            c += math.cos(np.float64(v))
        row_sin[i] = s
        row_cos[i] = c


@numba.njit(cache=True, parallel=True)
def weights_from_phase(
    phase: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    squared: float,
    wx: np.ndarray,
    wy: np.ndarray,
) -> None:
    """Charbonnier weights C eps^2 / sqrt(r^2 + eps^2) of a float64 phase.

    r = grad phase - g, in float64; eps^2 = `squared`.
    """
    rows, cols = phase.shape
    t = wx.dtype.type
    for i in numba.prange(rows):
        for j in range(cols):
            if j < cols - 1:
                r = (phase[i, j + 1] - phase[i, j]) - np.float64(tx[i, j])
                wx[i, j] = t(squared / math.sqrt(r * r + squared)) * cx[i, j]
            if i < rows - 1:
                r = (phase[i + 1, j] - phase[i, j]) - np.float64(ty[i, j])
                wy[i, j] = t(squared / math.sqrt(r * r + squared)) * cy[i, j]


@numba.njit(cache=True, parallel=True)
def scale_inplace(x: np.ndarray, factor: np.ndarray) -> None:
    """Compute :code:`x *= factor` elementwise (as numpy does), on all threads."""
    rows, cols = x.shape
    for i in numba.prange(rows):
        for j in range(cols):
            x[i, j] *= factor[i, j]


@numba.njit(cache=True, parallel=True)
def admm_update_rows(
    u: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    lx_new: np.ndarray,
    ly_new: np.ndarray,
    rho: float,
    delta: float,
    rhs: np.ndarray,
    blocks: int,
) -> None:
    """:func:`admm_update`, with each edge evaluated once (identical results).

    The rows are split into `blocks` blocks, which run in parallel. Each keeps the
    values v of the previous row's lower edges in a buffer, so only the edges
    above a block's first row are evaluated twice.
    """
    rows, cols = u.shape
    t = u.dtype.type
    rho_t = t(rho)
    inv_rho = t(1.0 / rho)
    delta_t = t(delta)
    blocks = max(1, min(blocks, rows))
    per = (rows + blocks - 1) // blocks
    for block in numba.prange(blocks):
        i0 = block * per
        i1 = min(rows, i0 + per)
        vx = np.empty(max(cols - 1, 0), dtype=u.dtype)
        above = np.zeros(cols, dtype=u.dtype)
        below = np.zeros(cols, dtype=u.dtype)
        if 0 < i0 < rows:
            for j in range(cols):
                _, above[j] = _edge(u[i0, j] - u[i0 - 1, j], gy[i0 - 1, j],
                                    cy[i0 - 1, j], ly[i0 - 1, j], rho_t, inv_rho,
                                    delta_t)
        for i in range(i0, i1):
            for j in range(cols - 1):
                lam, vx[j] = _edge(u[i, j + 1] - u[i, j], gx[i, j], cx[i, j],
                                   lx[i, j], rho_t, inv_rho, delta_t)
                lx_new[i, j] = lam
            if i < rows - 1:
                for j in range(cols):
                    lam, below[j] = _edge(u[i + 1, j] - u[i, j], gy[i, j], cy[i, j],
                                          ly[i, j], rho_t, inv_rho, delta_t)
                    ly_new[i, j] = lam
            for j in range(cols):
                # in the order of `admm_update`: right, left, lower, upper
                acc = t(0)
                if j < cols - 1:
                    acc -= vx[j]
                if j > 0:
                    acc += vx[j - 1]
                if i < rows - 1:
                    acc -= below[j]
                if i > 0:
                    acc += above[j]
                rhs[i, j] = acc
            above, below = below, above

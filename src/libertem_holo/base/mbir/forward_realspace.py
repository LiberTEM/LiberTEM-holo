"""Real-space finite-cell analytic forward model.

This module provides a direct-summation alternative to the FFT-based RDFC
forward model in :mod:`.kernel` and :mod:`.forward`.

The mathematical operator is **identical** to the ``geometry="slab"`` variant
of :func:`.kernel.build_rdfc_kernel`: each voxel column contributes an exact
finite-area integral of the :math:`y/r^2` Biot-Savart kernel, summed over all
source cells.  The only difference is that the convolution is evaluated by
direct summation in real space rather than by FFT multiplication of
zero-padded arrays.

Use this module when:

* You want an FFT-free reference implementation to rule out wraparound /
  Gibbs artifacts (there are none for the zero-padded RDFC slab, but this
  lets you verify that empirically).
* You need to evaluate the phase on a detector grid whose coordinates are
  *not* the voxel grid (e.g. arbitrary sample points, masked regions).

For large dense grids the FFT path (``forward_model_2d`` with
``geometry="slab"``) will be orders of magnitude faster; see the
``notebooks/MBIR/forward_model_benchmark.ipynb`` notebook for timing data.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import unxt as u

from .forward import apply_ramp
from .units import (
    KERNEL_COEFF,
    _as_dimensionless_quantity,
    _as_length_quantity,
    _as_ramp_coeffs,
    _assert_quantity_compatible,
)

_EPS = 1e-30


def _F_a(n: jax.Array, m: jax.Array) -> jax.Array:
    r"""Primitive whose mixed derivative is :math:`2\,m/(n^2+m^2)`.

    Identical to the primitive used in
    :func:`.kernel._rdfc_elementary_phase` for the ``"slab"`` geometry;
    uses ``atan(n/m)`` with an ``eps`` regulariser to avoid the
    ``atan2`` branch cut at the origin.  Corner differences of this
    primitive give **twice** the exact cell integral of
    :math:`m/(n^2+m^2)`.
    """
    A = jnp.log(n * n + m * m + _EPS)
    B = jnp.arctan(n / (m + _EPS))
    return n * A - 2.0 * n + 2.0 * m * B


def _cell_integral_y_over_r2(ox: jax.Array, oy: jax.Array) -> jax.Array:
    r"""Exact integral of :math:`v/(u^2+v^2)` over the unit cell at offset ``(ox, oy)``.

    Using ``n = u, m = v`` and the primitive :func:`_F_a` (whose mixed
    derivative is :math:`2m/(n^2+m^2)`), the corner sum divided by two
    gives the exact cell integral.
    """
    return 0.5 * (
        _F_a(ox - 0.5, oy - 0.5)
        - _F_a(ox + 0.5, oy - 0.5)
        - _F_a(ox - 0.5, oy + 0.5)
        + _F_a(ox + 0.5, oy + 0.5)
    )


def _cell_integral_x_over_r2(ox: jax.Array, oy: jax.Array) -> jax.Array:
    r"""Exact integral of :math:`u/(u^2+v^2)` over the unit cell at offset ``(ox, oy)``.

    By symmetry this is the same as :func:`_cell_integral_y_over_r2` with
    ``ox`` and ``oy`` swapped.
    """
    return 0.5 * (
        _F_a(oy - 0.5, ox - 0.5)
        - _F_a(oy + 0.5, ox - 0.5)
        - _F_a(oy - 0.5, ox + 0.5)
        + _F_a(oy + 0.5, ox + 0.5)
    )


def _kernel_offsets(height: int, width: int, dtype) -> tuple[jax.Array, jax.Array]:
    """Offset grid (ox, oy) in pixel units covering (-(W-1)..W-1, -(H-1)..H-1)."""
    di = jnp.arange(-(height - 1), height, dtype=dtype)
    dj = jnp.arange(-(width - 1), width, dtype=dtype)
    return jnp.meshgrid(dj, di, indexing="xy")  # (ox along cols, oy along rows)


def build_realspace_kernels(
    dim_uv: tuple[int, int],
    dtype=jnp.float64,
) -> tuple[u.Quantity, u.Quantity]:
    """Build the real-space finite-cell phase kernels.

    Returns two arrays ``K_u`` and ``K_v`` of shape ``(2H-1, 2W-1)`` where
    ``K_u[i, j]`` is the phase contribution at the detector pixel from a unit
    ``m_x`` source voxel offset by ``(i - (H-1), j - (W-1))`` pixels.  Units
    are :math:`1/\\text{nm}^2`; multiply by ``pixel_size**2`` and the voxel
    magnetization value to get radians.

    Parameters
    ----------
    dim_uv
        Detector shape ``(height, width)``.
    dtype
        Floating dtype.

    Returns
    -------
    K_u, K_v : Quantity["1/nm^2"]
        Real-space kernels for the ``m_u`` and ``m_v`` components.
    """
    height, width = dim_uv
    ox, oy = _kernel_offsets(height, width, dtype)
    # Sign convention matches RDFC: phi = m_u * K_u + m_v * K_v.
    # The y/r^2 Biot-Savart contribution multiplies m_u; the -x/r^2
    # contribution multiplies m_v.
    k_u = _cell_integral_y_over_r2(ox, oy)
    k_v = -_cell_integral_x_over_r2(ox, oy)

    unit = str(KERNEL_COEFF.unit)
    coeff = KERNEL_COEFF.value
    return (
        u.Quantity(jnp.asarray(coeff * k_u, dtype=dtype), unit),
        u.Quantity(jnp.asarray(coeff * k_v, dtype=dtype), unit),
    )


def _direct_convolve_valid(
    m_field: jax.Array,
    kernel: jax.Array,
) -> jax.Array:
    """Direct (non-FFT) 'valid' 2D convolution with a kernel of size (2H-1, 2W-1).

    ``m_field`` has shape ``(H, W)``; ``kernel`` has shape ``(2H-1, 2W-1)``.
    The result has shape ``(H, W)`` and equals

        out[i, j] = sum_{i', j'} K[i - i' + (H-1), j - j' + (W-1)] * m[i', j']
    """
    # jax.scipy.signal.convolve2d with mode='valid' flips the second argument
    # internally, so passing m_field as the second argument gives the
    # correlation structure above.  The kernel (first arg) must be larger
    # than the signal for 'valid' mode.
    return jax.scipy.signal.convolve2d(
        kernel,
        m_field,
        mode="valid",
    )


def phase_mapper_realspace(
    u_field: u.Quantity,
    v_field: u.Quantity,
    k_u: u.Quantity,
    k_v: u.Quantity,
) -> u.Quantity:
    """Map (u, v) magnetization to a phase image by direct real-space convolution.

    Parameters
    ----------
    u_field, v_field
        Projected in-plane magnetization components, shape ``(H, W)``.
    k_u, k_v
        Real-space kernels returned by :func:`build_realspace_kernels`.

    Returns
    -------
    Quantity["1/nm^2"]
        Phase-density image of shape ``(H, W)`` (multiply by ``pixel_size**2``
        to get radians).
    """
    u_q = cast(u.Quantity, _as_dimensionless_quantity(u_field))
    v_q = cast(u.Quantity, _as_dimensionless_quantity(v_field))

    phase = _direct_convolve_valid(u_q.value, k_u.value) + _direct_convolve_valid(
        v_q.value, k_v.value,
    )
    unit = str(k_u.unit)
    return u.Quantity(phase, unit)


def forward_model_realspace_2d(
    magnetization,
    pixel_size,
    ramp_coeffs=None,
    kernels: tuple[u.Quantity, u.Quantity] | None = None,
) -> u.Quantity:
    """Real-space finite-cell analytic forward model.

    Drop-in alternative to :func:`.forward.forward_model_2d` with
    ``geometry="slab"``.  Mathematically identical in the continuum limit;
    numerically identical (to floating-point tolerance) to the RDFC-slab FFT
    path for all grid sizes, because RDFC uses 2x zero-padding.

    Parameters
    ----------
    magnetization : Quantity["dimensionless"]
        In-plane magnetization of shape ``(H, W, 2)``.
    pixel_size : Quantity["length"]
        Pixel size.
    ramp_coeffs : RampCoeffs, optional
        Background phase-ramp coefficients.
    kernels : tuple, optional
        Pre-built ``(k_u, k_v)`` from :func:`build_realspace_kernels`.

    Returns
    -------
    Quantity["angle"]
        Predicted phase image of shape ``(H, W)`` in radians.
    """
    pixel_size_q = _as_length_quantity(pixel_size)
    mag_q = _as_dimensionless_quantity(magnetization)

    height, width = mag_q.shape[:2]
    if kernels is None:
        k_u, k_v = build_realspace_kernels(
            (height, width),
            dtype=mag_q.value.dtype,
        )
    else:
        k_u, k_v = kernels

    ramp_q = _as_ramp_coeffs(ramp_coeffs, dtype=mag_q.value.dtype)

    u_field = mag_q[..., 0]
    v_field = mag_q[..., 1]

    phase_density = phase_mapper_realspace(u_field, v_field, k_u, k_v)
    phase = u.Quantity(
        pixel_size_q.value ** 2 * phase_density.value,
        "rad",
    )
    ramp = apply_ramp(ramp_q, height, width, pixel_size_q)

    phase_total = phase + ramp
    _assert_quantity_compatible(phase_total, "rad", "phase_total")
    return phase_total


__all__ = [
    "build_realspace_kernels",
    "forward_model_realspace_2d",
    "phase_mapper_realspace",
]

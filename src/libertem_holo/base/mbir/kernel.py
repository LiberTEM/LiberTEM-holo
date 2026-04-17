"""RDFC Fourier-space phase-mapping kernel.

Implements the Real-space Decomposition of the Fourier-space Convolution
kernel that maps a 2D projected magnetization field to its magnetic phase
shift. Exposes :func:`build_rdfc_kernel`, :func:`phase_mapper_rdfc` and
the FFT frequency-grid helper :func:`get_freq_grid`.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import unxt as u

from .units import (
    KERNEL_COEFF,
    _as_dimensionless_quantity,
    _as_length_quantity,
    _assert_quantity_compatible,
)

def get_freq_grid(
    height: int,
    width: int,
    pixel_size: u.Quantity,
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """Build frequency grids for FFT-based phase propagation.

    Parameters
    ----------
    height
        Number of pixels along the y-axis.
    width
        Number of pixels along the x-axis.
    pixel_size
        Pixel size in nanometres, used as the FFT sampling interval.

    Returns
    -------
    f_y : Quantity["1/length"]
        2D array of y-frequency values (units of ``1/nm``).
    f_x : Quantity["1/length"]
        2D array of x-frequency values (real-FFT half-spectrum).
    denom : Quantity["1/length^2"]
        ``f_x**2 + f_y**2`` with the zero-frequency bin set to 1
        to avoid division by zero.
    """
    pixel_size_q = _as_length_quantity(pixel_size)
    fy = jnp.fft.fftfreq(height, d=pixel_size_q.value)
    fx = jnp.fft.rfftfreq(width, d=pixel_size_q.value)
    f_y_val, f_x_val = jnp.meshgrid(fy, fx, indexing="ij")
    f_y = u.Quantity(f_y_val, "1/nm")
    f_x = u.Quantity(f_x_val, "1/nm")
    denom_val = f_x_val**2 + f_y_val**2
    denom = u.Quantity(jnp.where(denom_val == 0, 1.0, denom_val), "1/nm2")
    return f_y, f_x, denom


def _rdfc_elementary_phase(
    geometry: str,
    n: jax.Array,
    m: jax.Array,
) -> jax.Array:
    """Compute the elementary kernel phase for the RDFC mapper.

    Parameters
    ----------
    geometry
        Voxel geometry, either ``'disc'`` or ``'slab'``.
    n
        Row coordinate array.
    m
        Column coordinate array.
    Returns
    -------
    jax.Array
        Elementary phase kernel evaluated on the coordinate grid.
    """
    if geometry == "disc":
        in_or_out = jnp.logical_not(jnp.logical_and(n == 0, m == 0))
        return m / (n**2 + m**2 + 1e-30) * in_or_out
    if geometry == "slab":
        def _F_a(n_val, m_val):
            radius2 = n_val**2 + m_val**2 + 1e-30
            A = jnp.log(radius2)
            B = jnp.arctan(n_val / (m_val + 1e-30))
            return n_val * A - 2 * n_val + 2 * m_val * B

        return 0.5 * (
            _F_a(n - 0.5, m - 0.5)
            - _F_a(n + 0.5, m - 0.5)
            - _F_a(n - 0.5, m + 0.5)
            + _F_a(n + 0.5, m + 0.5)
        )
    raise ValueError("Unknown geometry (use 'disc' or 'slab')")


def build_rdfc_kernel(
    dim_uv: tuple[int, int],
    geometry: str = "disc",
    prw_vec: jax.Array | None = None,
    dtype: type = jnp.float64,
) -> dict[str, Any]:
    """Build an RDFC phase-mapping kernel in Fourier space (JIT-compiled).

    The kernel coefficient is ``KERNEL_COEFF`` = :math:`B_{\\text{ref}} / (2 \\Phi_0)`
    with units of :math:`1/\\text{nm}^2`.  When multiplied by voxel-area
    sums in the elementary-phase functions and by ``pixel_size²``
    in the forward model, the result is dimensionless (radians).

    Parameters
    ----------
    dim_uv : tuple[int, int]
        Image dimensions ``(height, width)``.
    geometry : str, optional
        ``"disc"`` or ``"slab"``, default ``"disc"``.
    prw_vec : jax.Array or None, optional
        Projected reference wave vector ``(v, u)``.
    dtype : type, optional
        JAX float dtype, default ``jnp.float64``.

    Returns
    -------
    dict
        Dictionary with keys ``"u_fft"``, ``"v_fft"``, ``"dim_uv"``,
        and ``"dim_pad"``.
    """
    height, width = dim_uv
    dim_kern = (2 * height - 1, 2 * width - 1)
    dim_pad = (2 * height, 2 * width)

    u_coords = jnp.linspace(-(width - 1), width - 1, num=dim_kern[1]).astype(dtype)
    v_coords = jnp.linspace(-(height - 1), height - 1, num=dim_kern[0]).astype(dtype)
    uu, vv = jnp.meshgrid(u_coords, v_coords, indexing="xy")

    coeff = KERNEL_COEFF
    
    # Compute elementary phases once
    elem_uv = _rdfc_elementary_phase(geometry, uu, vv)
    elem_vu = _rdfc_elementary_phase(geometry, vv, uu)
    
    u_kernel = coeff * elem_uv
    v_kernel = -coeff * elem_vu

    if prw_vec is not None:
        uu_prw = uu + prw_vec[1]
        vv_prw = vv + prw_vec[0]
        elem_prw_uv = _rdfc_elementary_phase(geometry, uu_prw, vv_prw)
        elem_prw_vu = _rdfc_elementary_phase(geometry, vv_prw, uu_prw)
        u_kernel = u_kernel - coeff * elem_prw_uv
        v_kernel = v_kernel + coeff * elem_prw_vu

    kernel_unit = str(coeff.unit)
    u_pad = u.Quantity(
        jax.lax.dynamic_update_slice(
            jnp.zeros(dim_pad, dtype=dtype),
            u_kernel.value,
            (0, 0),
        ),
        kernel_unit,
    )
    v_pad = u.Quantity(
        jax.lax.dynamic_update_slice(
            jnp.zeros(dim_pad, dtype=dtype),
            v_kernel.value,
            (0, 0),
        ),
        kernel_unit,
    )

    result = {
        "u_fft": u.Quantity(jnp.fft.rfft2(u_pad.value), kernel_unit),
        "v_fft": u.Quantity(jnp.fft.rfft2(v_pad.value), kernel_unit),
        "dim_uv": dim_uv,
        "dim_pad": dim_pad,
    }
    _assert_quantity_compatible(cast(u.Quantity, result["u_fft"]), "1 / nm2", "kernel['u_fft']")
    _assert_quantity_compatible(cast(u.Quantity, result["v_fft"]), "1 / nm2", "kernel['v_fft']")
    return result


def phase_mapper_rdfc(
    u_field: u.Quantity,
    v_field: u.Quantity,
    rdfc_kernel: dict[str, Any],
) -> u.Quantity:
    """Map (u, v) magnetization components to a phase image via RDFC.

    Uses a precomputed Fourier-space kernel from
    :func:`build_rdfc_kernel`.

    Parameters
    ----------
    u_field
        In-plane magnetization component along x, shape ``(H, W)``.
    v_field
        In-plane magnetization component along y, shape ``(H, W)``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.

    Returns
    -------
    Quantity
        Fourier-space phase-shift contribution of shape ``(H, W)``.
        Carries the kernel's ``1/nm^2`` units; the forward model
        multiplies by ``pixel_size**2`` to obtain radians.
    """
    u_field_q = cast(u.Quantity, _as_dimensionless_quantity(u_field))
    v_field_q = cast(u.Quantity, _as_dimensionless_quantity(v_field))
    height, width = u_field_q.shape
    dim_pad = (2 * height, 2 * width)

    u_pad = u.Quantity(
        jax.lax.dynamic_update_slice(
            jnp.zeros(dim_pad, dtype=u_field_q.value.dtype),
            u_field_q.value,
            (0, 0),
        ),
        str(u_field_q.unit),
    )
    v_pad = u.Quantity(
        jax.lax.dynamic_update_slice(
            jnp.zeros(dim_pad, dtype=v_field_q.value.dtype),
            v_field_q.value,
            (0, 0),
        ),
        str(v_field_q.unit),
    )

    u_fft = u.Quantity(jnp.fft.rfft2(u_pad.value), str(u_field_q.unit))
    v_fft = u.Quantity(jnp.fft.rfft2(v_pad.value), str(v_field_q.unit))

    phase_fft = u_fft * rdfc_kernel["u_fft"] + v_fft * rdfc_kernel["v_fft"]
    phase_pad = u.Quantity(jnp.fft.irfft2(phase_fft.value, s=dim_pad), str(phase_fft.unit))

    phase = u.Quantity(
        jax.lax.dynamic_slice(
            phase_pad.value,
            (height - 1, width - 1),
            (height, width),
        ),
        str(phase_pad.unit),
    )
    return phase


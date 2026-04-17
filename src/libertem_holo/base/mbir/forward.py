"""Forward models that map magnetization to phase images.

Provides :func:`apply_ramp`, the low-level
:func:`forward_model_single_rdfc_2d`, the user-facing
:func:`forward_model_2d`, and the 3D volume variants
:func:`project_3d` and :func:`forward_model_3d`.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import quaxed.numpy as qnp
import unxt as u

from .units import (
    RampCoeffs,
    _as_angle_quantity,
    _as_dimensionless_quantity,
    _as_length_quantity,
    _as_ramp_coeffs,
    _assert_quantity_compatible,
    _validate_positive,
)
from .kernel import build_rdfc_kernel, phase_mapper_rdfc

def apply_ramp(ramp_coeffs, height, width, pixel_size):
    """Generate a first-order 2D polynomial background ramp.

    Requires :class:`RampCoeffs` with ``unxt.Quantity`` fields and a
    ``Quantity["length"]`` pixel size.

    Parameters
    ----------
    ramp_coeffs : RampCoeffs
        Background phase-ramp coefficients with Quantity fields.
    height : int
        Number of pixels along the y-axis.
    width : int
        Number of pixels along the x-axis.
    pixel_size : Quantity["length"]
        Pixel size converted to nm internally.

    Returns
    -------
    Quantity["angle"]
        Ramp image of shape ``(height, width)``.
    """
    pixel_size_q = _as_length_quantity(pixel_size)
    ramp_coeffs_q = _as_ramp_coeffs(ramp_coeffs)

    y, x = qnp.meshgrid(qnp.arange(height), qnp.arange(width), indexing="ij")
    ramp = ramp_coeffs_q.offset
    ramp = ramp + ramp_coeffs_q.slope_y * (y * pixel_size_q)
    ramp = ramp + ramp_coeffs_q.slope_x * (x * pixel_size_q)

    ramp = _as_angle_quantity(ramp)
    _assert_quantity_compatible(ramp, "rad", "ramp")
    return ramp


def forward_model_single_rdfc_2d(
    magnetization: u.Quantity,
    ramp_coeffs: RampCoeffs,
    rdfc_kernel: dict[str, Any],
    pixel_size: u.Quantity,
) -> u.Quantity:
    """RDFC forward model mapping projected magnetization to phase.

    Computes the magnetic phase shift from a 2D projected
    magnetization field and adds a polynomial background ramp.

    Parameters
    ----------
    magnetization
        In-plane magnetization of shape ``(N, M, 2)`` where the
        last axis holds the (u, v) components.
    ramp_coeffs
        Background ramp coefficients ``[offset, slope_y, slope_x]``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    pixel_size
        Pixel size in nanometres.

    Returns
    -------
    jax.Array
        Predicted phase image of shape ``(N, M)``.
    """
    magnetization_q = _as_dimensionless_quantity(magnetization)
    ramp_coeffs_q = _as_ramp_coeffs(
        ramp_coeffs,
        dtype=magnetization_q.value.dtype,
    )
    pixel_size_q = _as_length_quantity(pixel_size)

    height, width = magnetization_q.shape[:2]

    u_field = magnetization_q[..., 0]
    v_field = magnetization_q[..., 1]

    # Kernel is defined in pixel coordinates; convert to physical phase scale.
    phase_density = pixel_size_q**2 * phase_mapper_rdfc(u_field, v_field, rdfc_kernel)
    phase = _as_angle_quantity(phase_density)
    ramp = apply_ramp(ramp_coeffs_q, height, width, pixel_size_q)

    phase_total = phase + ramp
    _assert_quantity_compatible(phase_total, "rad", "phase_total")
    return phase_total


def forward_model_2d(
    magnetization,
    pixel_size,
    ramp_coeffs=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
) -> u.Quantity:
    """Convenience forward model for 2D projected magnetization.

    Computes the magnetic phase shift from a magnetization field,
    automatically building the RDFC kernel when not provided.

    Parameters
    ----------
    magnetization : array_like
        In-plane magnetization of shape ``(N, M, 2)`` where the
        last axis holds the (u, v) components. The input is assumed
        to be the projected normalized magnetization returned by
        :func:`reconstruct_2d`, i.e. a dimensionless ``Quantity``.
        Use :func:`to_projected_induction_integral` if you want the same field
        expressed as a projected induction line integral.
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
    ramp_coeffs : array_like, optional
        Background ramp coefficients ``[offset, slope_y, slope_x]``
        in units of **[rad, rad/nm, rad/nm]**.
        Defaults to zeros (no ramp).
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.

    Returns
    -------
    Quantity["angle"]
        Predicted phase image of shape ``(N, M)`` in **radians**.
    """
    pixel_size = _as_length_quantity(pixel_size)
    _validate_positive(pixel_size, "pixel_size")

    magnetization = _as_dimensionless_quantity(magnetization)
    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            magnetization.shape[:2],
            geometry=geometry,
            prw_vec=prw_vec,
        )
    ramp_coeffs = _as_ramp_coeffs(
        ramp_coeffs,
        dtype=magnetization.value.dtype,
    )

    result = forward_model_single_rdfc_2d(
        magnetization, ramp_coeffs, rdfc_kernel, pixel_size,
    )
    _assert_quantity_compatible(result, "rad", "forward_model_2d result")
    return result


_SIMPLE_PROJ = {
    "z": {
        "sum_axis": 0,
        "coeff": [[1, 0, 0], [0, 1, 0]],   # u=mx, v=my
        "transpose": False,                  # (Y, X) is already (V, U)
    },
    "y": {
        "sum_axis": 1,
        "coeff": [[1, 0, 0], [0, 0, 1]],   # u=mx, v=mz
        "transpose": False,                  # (Z, X) is already (V, U)
    },
    "x": {
        "sum_axis": 2,
        "coeff": [[0, 0, 1], [0, 1, 0]],   # u=mz, v=my
        "transpose": True,                   # (Z, Y) -> (Y, Z) = (V, U)
    },
}


def project_3d(
    magnetization_3d,
    axis="z",
) -> u.Quantity:
    """Project a 3D magnetization field along a major axis.

    Implements the simple-projector case from pyramid: sum along
    the projection axis and mix (mx, my, mz) into (u, v) components.

    Parameters
    ----------
    magnetization_3d : array_like
        3D magnetization of shape ``(Z, Y, X, 3)`` where the last
        axis holds ``(mx, my, mz)`` components.
    axis : {'z', 'y', 'x'}, optional
        Projection axis, default ``'z'``.

    Returns
    -------
    Quantity["dimensionless"]
        Projected 2D magnetization of shape ``(V, U, 2)`` where
        the last axis holds ``(u, v)`` components suitable for
        :func:`phase_mapper_rdfc`.
    """
    axis = axis.lower()
    if axis not in _SIMPLE_PROJ:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")

    cfg = _SIMPLE_PROJ[axis]
    magnetization_3d = _as_dimensionless_quantity(magnetization_3d)

    # Sum along projection direction: (Z, Y, X, 3) -> (*, *, 3)
    summed = cast(u.Quantity, qnp.sum(magnetization_3d, axis=cfg["sum_axis"]))

    # Mix (mx, my, mz) -> (u, v) via coefficient matrix
    coeff = jnp.array(cfg["coeff"], dtype=summed.value.dtype)  # (2, 3)
    projected = cast(u.Quantity, qnp.einsum("...c,oc->...o", summed, coeff))  # (*, *, 2)

    if cfg["transpose"]:
        projected = u.Quantity(
            jnp.transpose(projected.value, (1, 0, 2)),
            str(projected.unit),
        )

    return projected


def forward_model_3d(
    magnetization_3d,
    pixel_size,
    axis="z",
    ramp_coeffs=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
) -> u.Quantity:
    """Convenience forward model for a 3D magnetization volume.

    Projects a 3D magnetization field along a major axis using
    :func:`project_3d` (simple projector), then computes the
    magnetic phase shift via RDFC.

    Parameters
    ----------
    magnetization_3d : array_like
        3D magnetization of shape ``(Z, Y, X, 3)`` where the last
        axis holds ``(mx, my, mz)`` components.
    pixel_size : Quantity["length"]
        Voxel size as a ``unxt.Quantity`` with length units.
    axis : {'z', 'y', 'x'}, optional
        Projection axis, default ``'z'``.
    ramp_coeffs : array_like, optional
        Background ramp coefficients ``[offset, slope_y, slope_x]``.
        Defaults to zeros (no ramp).
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.

    Returns
    -------
    Quantity["angle"]
        Predicted phase image of shape ``(V, U)``.
    """
    projected = project_3d(magnetization_3d, axis=axis)

    return forward_model_2d(
        projected,
        pixel_size,
        ramp_coeffs=ramp_coeffs,
        geometry=geometry,
        prw_vec=prw_vec,
        rdfc_kernel=rdfc_kernel,
    )


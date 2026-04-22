"""Forward models that map magnetization to phase images.

Provides :func:`apply_ramp`, the low-level
:func:`forward_model_single_rdfc_2d`, the user-facing
:func:`forward_model_2d`, and the 3D volume variants
:func:`project_3d`, :func:`project_3d_tilted`,
:func:`forward_model_3d`, and :func:`forward_model_3d_tilted`.
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jsnd
import numpy as np
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
    magnetization: Quantity["dimensionless"]
        In-plane magnetization of shape ``(N, M, 2)`` where the
        last axis holds the (u, v) components.
    ramp_coeffs: RampCoeffs
        Background phase-ramp coefficients with explicit units
        (offset in rad, slopes in rad/nm).
    rdfc_kernel: dict
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    pixel_size: Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units.

    Returns
    -------
    Quantity["angle"]
        Predicted phase image of shape ``(N, M)`` in **radians**.
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
    magnetization : Quantity["dimensionless"]
        In-plane magnetization of shape ``(N, M, 2)`` where the
        last axis holds the (u, v) components. The input is assumed
        to be the projected normalized magnetization returned by
        :func:`reconstruct_2d`, i.e. a dimensionless ``Quantity``.
        Use :func:`to_projected_induction_integral` if you want the same field
        expressed as a projected induction line integral.
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
    ramp_coeffs : RampCoeffs, optional
        Background phase-ramp coefficients with ``unxt.Quantity`` fields
        (offset in rad, slopes in rad/nm). Defaults to zeros (no ramp).
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


def _rotation_matrix_z(angle, dtype):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


def _rotation_matrix_x(angle, dtype):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=dtype,
    )


def _build_tilt_rotation_matrix(rotation, tilt):
    rotation = jnp.asarray(rotation)
    tilt = jnp.asarray(tilt, dtype=rotation.dtype)
    dtype = jnp.result_type(rotation.dtype, tilt.dtype, jnp.float64)
    rotation = rotation.astype(dtype)
    tilt = tilt.astype(dtype)
    return (
        _rotation_matrix_z(-rotation, dtype)
        @ _rotation_matrix_x(tilt, dtype)
        @ _rotation_matrix_z(rotation, dtype)
    )


def _projection_coefficients(rotation, tilt, dtype):
    return _build_tilt_rotation_matrix(rotation, tilt).astype(dtype)[:2, :]


def _compute_tilted_projection_geometry(dim_zyx, rotation, tilt, dim_uv=None):
    dim_z, dim_y, dim_x = (int(dim) for dim in dim_zyx)
    rot = np.asarray(_build_tilt_rotation_matrix(float(rotation), float(tilt)))

    half_extents = np.array([dim_x, dim_y, dim_z], dtype=np.float64) / 2.0
    corner_signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=np.float64)
    corners_xyz = corner_signs * half_extents
    rotated = corners_xyz @ rot.T

    span_u = rotated[:, 0].max() - rotated[:, 0].min()
    span_v = rotated[:, 1].max() - rotated[:, 1].min()
    span_t = rotated[:, 2].max() - rotated[:, 2].min()

    if dim_uv is None:
        dim_uv = (
            max(1, int(np.ceil(span_v))),
            max(1, int(np.ceil(span_u))),
        )
    else:
        dim_uv = tuple(int(dim) for dim in dim_uv)
        if len(dim_uv) != 2 or dim_uv[0] <= 0 or dim_uv[1] <= 0:
            raise ValueError(f"dim_uv must be a pair of positive ints, got {dim_uv!r}")

    dim_t = max(1, int(np.ceil(span_t)))
    return cast(tuple[int, int], dim_uv), dim_t


def _tilted_detector_grid(dim_t, dim_uv, dtype):
    dim_v, dim_u = dim_uv
    t = jnp.arange(dim_t, dtype=dtype) + 0.5 - dim_t / 2.0
    v = jnp.arange(dim_v, dtype=dtype) + 0.5 - dim_v / 2.0
    u = jnp.arange(dim_u, dtype=dtype) + 0.5 - dim_u / 2.0
    tt, vv, uu = jnp.meshgrid(t, v, u, indexing="ij")
    return jnp.stack([uu, vv, tt], axis=0)


def _sample_tilted_volume(volume, coords_zyx, out_shape):
    components_first = jnp.moveaxis(volume, -1, 0)

    def sample_component(field):
        sampled = jsnd.map_coordinates(
            field,
            coords_zyx,
            order=1,
            mode="constant",
            cval=0.0,
        )
        return sampled.reshape(out_shape)

    sampled = jax.vmap(sample_component, in_axes=0, out_axes=0)(components_first)
    return jnp.moveaxis(sampled, 0, -1)


def project_3d(
    magnetization_3d,
    axis="z",
) -> u.Quantity:
    """Project a 3D magnetization field along a major axis.

    Implements the simple-projector case from pyramid: sum along
    the projection axis and mix (mx, my, mz) into (u, v) components.

    Parameters
    ----------
    magnetization_3d : Quantity["dimensionless"]
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


def project_3d_tilted(
    magnetization_3d,
    rotation,
    tilt,
    dim_uv=None,
) -> u.Quantity:
    """Project a 3D magnetization field along a tilted beam direction.

    This implements a cleaned-up, differentiable analogue of pyramid's
    ``RotTiltProjector``. The 3D volume is sampled along the detector beam
    direction using trilinear interpolation, then the projected vector field
    is mixed into detector-plane ``(u, v)`` components.

    Parameters
    ----------
    magnetization_3d : Quantity["dimensionless"]
        3D magnetization of shape ``(Z, Y, X, 3)`` where the last axis holds
        ``(mx, my, mz)`` components.
    rotation : float
        In-plane rotation angle in radians.
    tilt : float
        Tilt angle in radians.
    dim_uv : tuple[int, int], optional
        Explicit detector shape ``(V, U)``. If omitted, uses the rotated
        bounding box of the volume.

    Returns
    -------
    Quantity["dimensionless"]
        Projected 2D magnetization of shape ``(V, U, 2)`` where the last axis
        holds ``(u, v)`` components suitable for :func:`phase_mapper_rdfc`.
    """
    magnetization_q = _as_dimensionless_quantity(magnetization_3d)
    dtype = magnetization_q.value.dtype
    rotation_f = float(rotation)
    tilt_f = float(tilt)

    if rotation_f == 0.0 and tilt_f == 0.0 and dim_uv is None:
        return project_3d(magnetization_q, axis="z")

    dim_z, dim_y, dim_x, comp = magnetization_q.shape
    if comp != 3:
        raise ValueError(
            f"magnetization_3d must have shape (Z, Y, X, 3); got {magnetization_q.shape!r}",
        )

    dim_uv_resolved, dim_t = _compute_tilted_projection_geometry(
        magnetization_q.shape[:3],
        rotation_f,
        tilt_f,
        dim_uv=dim_uv,
    )

    detector_xyz = _tilted_detector_grid(dim_t, dim_uv_resolved, dtype)
    detector_xyz_flat = detector_xyz.reshape(3, -1)

    rotation_matrix = _build_tilt_rotation_matrix(rotation_f, tilt_f).astype(dtype)
    volume_xyz = rotation_matrix.T @ detector_xyz_flat

    x_idx = volume_xyz[0] + dim_x / 2.0 - 0.5
    y_idx = volume_xyz[1] + dim_y / 2.0 - 0.5
    z_idx = volume_xyz[2] + dim_z / 2.0 - 0.5
    coords_zyx = jnp.stack([z_idx, y_idx, x_idx], axis=0)

    sampled = _sample_tilted_volume(
        magnetization_q.value,
        coords_zyx,
        (dim_t, dim_uv_resolved[0], dim_uv_resolved[1]),
    )
    summed = jnp.sum(sampled, axis=0)

    coeff = _projection_coefficients(rotation_f, tilt_f, dtype)
    projected = jnp.einsum("...c,oc->...o", summed, coeff)
    return u.Quantity(projected, str(magnetization_q.unit))


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
    magnetization_3d : Quantity["dimensionless"]
        3D magnetization of shape ``(Z, Y, X, 3)`` where the last
        axis holds ``(mx, my, mz)`` components.
    pixel_size : Quantity["length"]
        Voxel size as a ``unxt.Quantity`` with length units.
    axis : {'z', 'y', 'x'}, optional
        Projection axis, default ``'z'``.
    ramp_coeffs : RampCoeffs, optional
        Background phase-ramp coefficients with ``unxt.Quantity`` fields
        (offset in rad, slopes in rad/nm). Defaults to zeros (no ramp).
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


def forward_model_3d_tilted(
    magnetization_3d,
    pixel_size,
    rotation,
    tilt,
    dim_uv=None,
    ramp_coeffs=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
) -> u.Quantity:
    """Convenience forward model for a tilted 3D magnetization volume.

    Projects a 3D magnetization field along a rotated and tilted beam using
    :func:`project_3d_tilted`, then computes the magnetic phase shift via RDFC.
    """
    projected = project_3d_tilted(
        magnetization_3d,
        rotation=rotation,
        tilt=tilt,
        dim_uv=dim_uv,
    )

    return forward_model_2d(
        projected,
        pixel_size,
        ramp_coeffs=ramp_coeffs,
        geometry=geometry,
        prw_vec=prw_vec,
        rdfc_kernel=rdfc_kernel,
    )


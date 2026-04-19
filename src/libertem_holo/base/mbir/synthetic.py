"""Synthetic differentiable shape and magnetization primitives for MBIR.

All spatial arrays use LiberTEM ordering ``(Z, Y, X)``.
Magnetization fields append a component axis and use ``(mx, my, mz)``,
i.e. shape ``(Z, Y, X, 3)``.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

_EPS = 1e-12


def _as_center_yx(shape_zyx: tuple[int, int, int], center_yx: tuple[float, float] | None) -> tuple[float, float]:
    if center_yx is not None:
        return center_yx
    _, ny, nx = shape_zyx
    return ((ny - 1) / 2.0, (nx - 1) / 2.0)


def _spatial_grid(shape_zyx: tuple[int, int, int], dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    _, ny, nx = shape_zyx
    yy, xx = jnp.meshgrid(
        jnp.arange(ny, dtype=dtype),
        jnp.arange(nx, dtype=dtype),
        indexing="ij",
    )
    return yy, xx


def soft_disc_support(
    shape_zyx: tuple[int, int, int],
    radius: float,
    *,
    center_yx: tuple[float, float] | None = None,
    edge_width: float = 1.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Return a differentiable soft disc support field with shape ``(Z, Y, X)``."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if edge_width <= 0:
        raise ValueError(f"edge_width must be positive, got {edge_width}")

    yy, xx = _spatial_grid(shape_zyx, dtype=dtype)
    cy, cx = _as_center_yx(shape_zyx, center_yx)
    r = jnp.hypot(yy - cy, xx - cx)
    soft_2d = jax.nn.sigmoid((radius - r) / edge_width)
    return jnp.broadcast_to(soft_2d[None, ...], shape_zyx)


def uniform_magnetization(
    shape_zyx: tuple[int, int, int],
    *,
    support_zyx: jnp.ndarray | None = None,
    direction_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0),
    magnitude: float = 1.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Return a uniform magnetization field of shape ``(Z, Y, X, 3)``."""
    if magnitude < 0:
        raise ValueError(f"magnitude must be non-negative, got {magnitude}")

    direction = jnp.asarray(direction_xyz, dtype=dtype)
    norm = jnp.linalg.norm(direction)
    if float(norm) <= 0.0:
        raise ValueError("direction_xyz must have non-zero norm")
    unit_direction = direction / norm

    support = jnp.ones(shape_zyx, dtype=dtype) if support_zyx is None else jnp.asarray(support_zyx, dtype=dtype)
    support = jnp.broadcast_to(support, shape_zyx)
    return cast(jnp.ndarray, magnitude * support[..., None] * unit_direction)


def vortex_magnetization(
    shape_zyx: tuple[int, int, int],
    *,
    support_zyx: jnp.ndarray | None = None,
    center_yx: tuple[float, float] | None = None,
    core_radius: float = 1.0,
    chirality: float = 1.0,
    magnitude: float = 1.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Return a vortex-like field with in-plane curl and out-of-plane core."""
    if core_radius <= 0:
        raise ValueError(f"core_radius must be positive, got {core_radius}")
    if magnitude < 0:
        raise ValueError(f"magnitude must be non-negative, got {magnitude}")

    yy, xx = _spatial_grid(shape_zyx, dtype=dtype)
    cy, cx = _as_center_yx(shape_zyx, center_yx)
    dy = yy - cy
    dx = xx - cx
    r = jnp.hypot(dy, dx)
    rinv = 1.0 / jnp.maximum(r, _EPS)

    tangent_x = -chirality * dy * rinv
    tangent_y = chirality * dx * rinv

    core = jnp.exp(-((r / core_radius) ** 2))
    in_plane_scale = jnp.sqrt(jnp.maximum(0.0, 1.0 - core**2))

    mx = tangent_x * in_plane_scale
    my = tangent_y * in_plane_scale
    mz = core
    base = jnp.stack([mx, my, mz], axis=-1)
    base = jnp.broadcast_to(base[None, ...], shape_zyx + (3,))

    support = jnp.ones(shape_zyx, dtype=dtype) if support_zyx is None else jnp.asarray(support_zyx, dtype=dtype)
    support = jnp.broadcast_to(support, shape_zyx)
    return cast(jnp.ndarray, magnitude * support[..., None] * base)


def domain_wall_magnetization(
    shape_zyx: tuple[int, int, int],
    *,
    support_zyx: jnp.ndarray | None = None,
    wall_x: float | None = None,
    wall_width: float = 2.0,
    magnitude: float = 1.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Return a simple analytic Néel-type domain wall in the x-z plane."""
    if wall_width <= 0:
        raise ValueError(f"wall_width must be positive, got {wall_width}")
    if magnitude < 0:
        raise ValueError(f"magnitude must be non-negative, got {magnitude}")

    _, _, nx = shape_zyx
    x = jnp.arange(nx, dtype=dtype)
    center_x = (nx - 1) / 2.0 if wall_x is None else wall_x
    xi = (x - center_x) / wall_width

    mx_line = jnp.tanh(xi)
    mz_line = 1.0 / jnp.cosh(xi)
    my_line = jnp.zeros_like(mx_line)

    base_line = jnp.stack([mx_line, my_line, mz_line], axis=-1)  # (X, 3)
    base = jnp.broadcast_to(base_line[None, None, :, :], shape_zyx + (3,))

    support = jnp.ones(shape_zyx, dtype=dtype) if support_zyx is None else jnp.asarray(support_zyx, dtype=dtype)
    support = jnp.broadcast_to(support, shape_zyx)
    return cast(jnp.ndarray, magnitude * support[..., None] * base)


__all__ = [
    "domain_wall_magnetization",
    "soft_disc_support",
    "uniform_magnetization",
    "vortex_magnetization",
]

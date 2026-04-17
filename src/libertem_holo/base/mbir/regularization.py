"""Regularization terms used by the MBIR loss."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import unxt as u

from .units import _as_dimensionless_quantity

def exchange_loss_fn(
    mag: jax.Array,
    reg_mask: jax.Array,
) -> jax.Array:
    r"""First-order regularization via masked finite differences.

    Computes the squared L2 norm of the spatial gradient of the
    magnetization field inside a mask:

    .. math::

        E = \sum_{i,j \in \text{mask}} \left\lVert
            \frac{\partial \mathbf{m}}{\partial y}\bigg|_{i,j}
            \right\rVert^2
          + \left\lVert
            \frac{\partial \mathbf{m}}{\partial x}\bigg|_{i,j}
            \right\rVert^2

    where each norm is over both magnetization components (u, v).
    The result is a scalar smoothness penalty: large when
    neighboring magnetization values differ, zero when the field
    is spatially uniform.  It enters the total MBIR loss as
    ``lambda_exchange * E``.

    Algorithm
    ---------
    1. **Neighbor detection**.  For each masked pixel the four
       cardinal neighbors (up, down, left, right) are checked
       against the mask.  ``neighbor_count`` records how many
       valid neighbors each pixel has (0--4).

    2. **Adaptive difference stencil**.  For each spatial axis the
       best available finite-difference is selected per pixel:

       * Both neighbors present → **central difference**
         (e.g. ``m[i+1,j] − m[i−1,j]``), accuracy *O(h²)*.
       * Only one neighbor → **forward or backward difference**
         (e.g. ``m[i+1,j] − m[i,j]``), accuracy *O(h)*.
       * No neighbor in that direction → zero contribution.

    3. **Neighbor-count normalization**.  Each difference is divided
       by the total neighbor count at that pixel.  Interior pixels
       (4 neighbors) are scaled by 1/4, edge pixels (2 neighbors)
       by 1/2, corner pixels (1 neighbor) by 1.  This keeps the
       per-pixel regularization contribution roughly uniform
       regardless of connectivity.

    4. **Squared L2 summation**.  The final loss is
       ``sum(diff_y²) + sum(diff_x²)`` over all masked pixels and
       both magnetization components.

    Differences from Pyramid's ``FirstOrderRegularisator``
    ------------------------------------------------------
    * **Dimensionality**: 2D (y, x) only, matching the projected
      magnetization geometry.  Pyramid regularizes all 3 spatial
      axes of a 3D volume.
    * **Stencil**: Adaptive central/one-sided selection at mask
      boundaries.  Pyramid uses forward differences everywhere via
      a sparse operator ``D`` built by ``jutil.diff``.
    * **Normalization**: Division by neighbor count.  Pyramid's
      sparse ``D`` matrix applies unit-weight forward differences
      with no per-pixel normalization.
    * **Implementation**: Pure JAX array operations; all
      derivatives obtained via autodiff.  Pyramid provides
      analytic ``jac``, ``hess_dot``, and ``hess_diag`` methods
      through its ``Regularisator`` class hierarchy.
    * **Lambda scale**: Because of the neighbor-count
      normalization and 2D-vs-3D geometry, a ``lambda_exchange``
      value here is not numerically identical to Pyramid's ``lam``
      for the same problem, even though both weight the same
      physical quantity (spatial roughness).

    Parameters
    ----------
    mag
        Magnetization array of shape ``(N, M, 2)``.
    reg_mask
        Boolean or binary mask of shape ``(N, M)`` defining the
        regularization region.

    Returns
    -------
    jax.Array
        Scalar exchange loss (L2 norm squared of finite differences).
    """
    mag_q = cast(u.Quantity, _as_dimensionless_quantity(mag))
    mag_val = mag_q.value
    reg_mask = jnp.asarray(reg_mask)
    if reg_mask.shape != mag_q.shape[:2]:
        raise ValueError(
            f"reg_mask must have shape {mag_q.shape[:2]}; got {reg_mask.shape}."
        )

    mask = reg_mask.astype(bool)
    mag_dtype = mag_val.dtype

    has_up = jnp.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    has_down = jnp.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    has_left = jnp.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    has_right = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)

    neighbor_count = (
        has_up.astype(mag_dtype)
        + has_down.astype(mag_dtype)
        + has_left.astype(mag_dtype)
        + has_right.astype(mag_dtype)
    )
    neighbor_count = neighbor_count * mask.astype(mag_dtype)
    denom = jnp.where(neighbor_count > 0, neighbor_count, jnp.ones_like(neighbor_count))

    mag_up = jnp.pad(mag_val[:-1, :, :], ((1, 0), (0, 0), (0, 0)), constant_values=0)
    mag_down = jnp.pad(mag_val[1:, :, :], ((0, 1), (0, 0), (0, 0)), constant_values=0)
    mag_left = jnp.pad(mag_val[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=0)
    mag_right = jnp.pad(mag_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)), constant_values=0)

    both_y = (has_up & has_down)[..., None]
    only_down = (has_down & ~has_up)[..., None]
    only_up = (has_up & ~has_down)[..., None]
    diff_y_fallback = cast(jax.Array, jnp.where(only_up, mag_val - mag_up, 0.0))
    diff_y_fallback = cast(
        jax.Array,
        jnp.where(only_down, mag_down - mag_val, diff_y_fallback),
    )
    diff_y = cast(jax.Array, jnp.where(both_y, mag_down - mag_up, diff_y_fallback))

    both_x = (has_left & has_right)[..., None]
    only_right = (has_right & ~has_left)[..., None]
    only_left = (has_left & ~has_right)[..., None]
    diff_x_fallback = cast(jax.Array, jnp.where(only_left, mag_val - mag_left, 0.0))
    diff_x_fallback = cast(
        jax.Array,
        jnp.where(only_right, mag_right - mag_val, diff_x_fallback),
    )
    diff_x = cast(jax.Array, jnp.where(both_x, mag_right - mag_left, diff_x_fallback))

    diff_y = cast(jax.Array, diff_y * mask[..., None] / denom[..., None])
    diff_x = cast(jax.Array, diff_x * mask[..., None] / denom[..., None])

    loss = jnp.sum(diff_y * diff_y) + jnp.sum(diff_x * diff_x)

    return u.Quantity(loss, "")

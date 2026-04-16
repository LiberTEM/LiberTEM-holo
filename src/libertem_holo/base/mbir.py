"""Model-based iterative reconstruction (MBIR) for 2D projected magnetization.

Unit conventions
----------------
All public functions in this module require ``unxt.Quantity`` inputs for
physical parameters and return ``Quantity``-annotated outputs.

* ``pixel_size`` — pixel side length as a ``unxt.Quantity`` with length units
  (e.g. ``Quantity(0.58, "nm")``).  Converted to nanometres internally.
* ``phase`` — measured holographic phase in **radians** (rad).
* ``ramp_coeffs`` — background phase-ramp parameters as a :class:`RampCoeffs`
  named tuple with ``unxt.Quantity`` fields (offset in rad, slopes in rad/nm).
* ``PHI_0`` — magnetic flux quantum :math:`h/(2e)` expressed as
  ``Quantity(2067.83, "T nm2")``.
* ``B_REF`` — reference magnetic induction :math:`B_0 = 1\\,\\text{T}`,
  baked into ``KERNEL_COEFF``.
* ``KERNEL_COEFF`` — :math:`B_{\\text{ref}} / (2 \\Phi_0)` with units
  :math:`1/\\text{nm}^2`.

The reconstructed magnetization is **dimensionless** (normalised
:math:`M / M_s`).  Phase outputs carry ``Quantity["rad"]``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple, Union, cast

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import numpy as np
import quaxed.numpy as qnp
import unxt as u

PHI_0 = u.Quantity(2067.83, "T nm2")  # magnetic flux quantum h/(2e)
B_REF = u.Quantity(1.0, "T")  # reference magnetic induction
KERNEL_COEFF = B_REF / (2 * PHI_0)  # Quantity["1/nm2"]
# Plain float for JIT internals — computed directly to avoid Quantity
# arithmetic round-off in the hot path.
_KERNEL_COEFF_FLOAT = 1.0 / (2 * 2067.83)  # == B_REF / (2 * PHI_0) in nm⁻²


# ---------------------------------------------------------------------------
# RampCoeffs — typed container for background ramp parameters
# ---------------------------------------------------------------------------

class RampCoeffs(NamedTuple):
    """Background phase-ramp coefficients with explicit units.

    Attributes
    ----------
    offset : Quantity["angle"]
        Constant phase offset in radians.
    slope_y : Quantity["angle / length"]
        Phase gradient along the y-axis in rad/nm.
    slope_x : Quantity["angle / length"]
        Phase gradient along the x-axis in rad/nm.
    """

    offset: u.Quantity
    slope_y: u.Quantity
    slope_x: u.Quantity

    @classmethod
    def zeros(cls, dtype=jnp.float64):
        """Create a zero-valued RampCoeffs."""
        return cls(
            offset=u.Quantity(jnp.zeros((), dtype=dtype), "rad"),
            slope_y=u.Quantity(jnp.zeros((), dtype=dtype), "rad/nm"),
            slope_x=u.Quantity(jnp.zeros((), dtype=dtype), "rad/nm"),
        )


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _to_nm(q: u.Quantity) -> u.Quantity:
    """Convert a length Quantity to nanometres."""
    return u.uconvert("nm", q)


def _to_rad(q: u.Quantity) -> u.Quantity:
    """Convert an angle Quantity to radians."""
    return u.uconvert("rad", q)


def _is_unit_convertible(value: u.Quantity, unit: str) -> bool:
    """Return True when *value* can be converted to *unit*."""
    try:
        u.uconvert(unit, value)
    except Exception:
        return False
    return True


def _assert_quantity_compatible(value: Any, unit: str, name: str) -> None:
    """Assert that *value* is a Quantity compatible with *unit*."""
    assert isinstance(value, u.Quantity), f"{name} must be a Quantity, got {type(value)}"
    assert _is_unit_convertible(value, unit), (
        f"{name} must be convertible to {unit!r}, got unit {value.unit!s}"
    )


def _assert_ramp_coeffs_units(ramp_coeffs: RampCoeffs, name: str = "ramp_coeffs") -> None:
    """Assert the canonical units for RampCoeffs."""
    assert isinstance(ramp_coeffs, RampCoeffs), (
        f"{name} must be a RampCoeffs instance, got {type(ramp_coeffs)}"
    )
    _assert_quantity_compatible(ramp_coeffs.offset, "rad", f"{name}.offset")
    _assert_quantity_compatible(ramp_coeffs.slope_y, "rad / nm", f"{name}.slope_y")
    _assert_quantity_compatible(ramp_coeffs.slope_x, "rad / nm", f"{name}.slope_x")


def _assert_solver_result_units(result: SolverResult) -> None:
    """Assert the public unit contract of SolverResult."""
    _assert_quantity_compatible(result.magnetization, "", "result.magnetization")
    _assert_ramp_coeffs_units(result.ramp_coeffs, "result.ramp_coeffs")
    _assert_quantity_compatible(result.loss_history, "rad2", "result.loss_history")


def _assert_lcurve_result_units(result: LCurveResult) -> None:
    """Assert the public unit contract of LCurveResult."""
    _assert_quantity_compatible(result.lambdas, "rad2", "result.lambdas")
    _assert_quantity_compatible(result.data_misfits, "rad2", "result.data_misfits")
    _assert_quantity_compatible(result.reg_norms, "", "result.reg_norms")
    _assert_quantity_compatible(result.magnetizations, "", "result.magnetizations")
    _assert_ramp_coeffs_units(result.ramp_coeffs, "result.ramp_coeffs")


def _ramp_coeffs_to_array(ramp_coeffs: RampCoeffs) -> jax.Array:
    """Convert typed ramp coefficients to a plain JAX array in [rad, rad/nm, rad/nm]."""
    offset = _to_rad(ramp_coeffs.offset)
    slope_y = cast(u.Quantity, u.uconvert("rad / nm", ramp_coeffs.slope_y))
    slope_x = cast(u.Quantity, u.uconvert("rad / nm", ramp_coeffs.slope_x))
    return jnp.array([
        offset.value,
        slope_y.value,
        slope_x.value,
    ])


def _ramp_coeffs_from_array(values: jax.Array) -> RampCoeffs:
    """Convert a plain ramp vector [offset, slope_y, slope_x] to typed coefficients."""
    values = jnp.asarray(values)
    return RampCoeffs(
        offset=u.Quantity(values[0], "rad"),
        slope_y=u.Quantity(values[1], "rad/nm"),
        slope_x=u.Quantity(values[2], "rad/nm"),
    )


def _as_length_quantity(value) -> u.Quantity:
    """Normalize a length-like input to a Quantity in nm."""
    if isinstance(value, u.Quantity):
        _assert_quantity_compatible(value, "nm", "pixel_size")
        result = _to_nm(value)
    else:
        result = u.Quantity(jnp.asarray(value), "nm")
    _assert_quantity_compatible(result, "nm", "pixel_size")
    return result


def _as_angle_quantity(value) -> u.Quantity:
    """Normalize an angle-like input to a Quantity in rad.

    Dimensionless intermediate phase values are re-labelled as radians by
    convention after the physical nm² cancellation in the forward model.
    """
    if isinstance(value, u.Quantity):
        if str(value.unit) == "":
            result = u.Quantity(value.value, "rad")
        else:
            _assert_quantity_compatible(value, "rad", "phase")
            result = _to_rad(value)
    else:
        result = u.Quantity(jnp.asarray(value), "rad")
    _assert_quantity_compatible(result, "rad", "phase")
    return result


def _as_dimensionless_quantity(value) -> u.Quantity:
    """Normalize a dimensionless input to a Quantity with unit ''."""
    if isinstance(value, u.Quantity):
        _assert_quantity_compatible(value, "", "magnetization")
        result = cast(u.Quantity, u.uconvert("", value))
    else:
        result = u.Quantity(jnp.asarray(value), "")
    _assert_quantity_compatible(result, "", "magnetization")
    return result


def _as_ramp_coeffs(value, *, dtype=jnp.float64) -> RampCoeffs:
    """Normalize ramp coefficients to the typed Quantity container."""
    if value is None:
        result = RampCoeffs.zeros(dtype=dtype)
        _assert_ramp_coeffs_units(result)
        return result
    if isinstance(value, RampCoeffs):
        result = RampCoeffs(
            offset=_to_rad(value.offset),
            slope_y=cast(u.Quantity, u.uconvert("rad / nm", value.slope_y)),
            slope_x=cast(u.Quantity, u.uconvert("rad / nm", value.slope_x)),
        )
        _assert_ramp_coeffs_units(result)
        return result
    result = _ramp_coeffs_from_array(jnp.asarray(value))
    _assert_ramp_coeffs_units(result)
    return result


def _contains_quantity(value: Any) -> bool:
    """Return True when the input already carries explicit unit information."""
    if isinstance(value, u.Quantity):
        return True
    if isinstance(value, RampCoeffs):
        return True
    if isinstance(value, dict):
        return any(_contains_quantity(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_quantity(item) for item in value)
    return False


def _maybe_strip_quantity(value, *inputs):
    """Return raw JAX values when all inputs were unitless/raw arrays."""
    if any(_contains_quantity(item) for item in inputs):
        return value
    if isinstance(value, u.Quantity):
        return value.value
    return value


def _lambda_exchange_quantity(value) -> u.Quantity:
    """Normalize the exchange weight to the loss unit rad².

    The regularization norm is dimensionless for dimensionless magnetization,
    so lambda_exchange must carry rad² for the total objective to be unit
    consistent.
    """
    if isinstance(value, u.Quantity):
        _assert_quantity_compatible(value, "rad2", "lambda_exchange")
        result = cast(u.Quantity, u.uconvert("rad2", value))
    else:
        result = u.Quantity(jnp.asarray(value), "rad2")
    _assert_quantity_compatible(result, "rad2", "lambda_exchange")
    return result


def _rfft2_keep_unit(arr):
    """Discrete FFT that preserves the input unit instead of inverting it.

    unxt/quax currently labels FFT outputs with reciprocal units. For the
    discrete, pixel-sum convention used here the transform should preserve the
    array's unit. We therefore apply FFT to the raw value and reattach the
    original unit explicitly.
    """
    if isinstance(arr, u.Quantity):
        return u.Quantity(jnp.fft.rfft2(arr.value), str(arr.unit))
    return jnp.fft.rfft2(arr)


def _irfft2_keep_unit(arr, *, s):
    """Inverse discrete FFT that preserves the input unit."""
    if isinstance(arr, u.Quantity):
        return u.Quantity(jnp.fft.irfft2(arr.value, s=s), str(arr.unit))
    return jnp.fft.irfft2(arr, s=s)


# ---------------------------------------------------------------------------
# Runtime validation helpers
# ---------------------------------------------------------------------------

def _validate_positive(value, name):
    """Raise ValueError if *value* is not positive.

    Handles both plain scalars and ``unxt.Quantity`` instances.
    """
    if isinstance(value, u.Quantity):
        v = float(value.value) if np.ndim(value.value) == 0 else float(np.min(np.asarray(value.value)))
    else:
        v = float(value) if np.ndim(value) == 0 else np.min(value)
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


class SolverResult(NamedTuple):
    """Result returned by :func:`solve_mbir_2d`.

    Attributes
    ----------
    magnetization : Quantity["dimensionless"]
        Reconstructed **projected** (thickness-integrated) in-plane
        magnetization of shape ``(N, M, 2)``.  Dimensionless
        (:math:`M / M_s`).
    ramp_coeffs : RampCoeffs
        Background phase-ramp coefficients with explicit units.
    loss_history : Quantity["rad2"]
        Per-step loss values.
    """
    magnetization: u.Quantity
    ramp_coeffs: RampCoeffs
    loss_history: u.Quantity


class LCurveResult(NamedTuple):
    """Result returned by :func:`lcurve_sweep` and :func:`lcurve_sweep_vmap`.

    Attributes
    ----------
    lambdas : Quantity["rad2"]
        Regularization weights used in the sweep.
    data_misfits : Quantity["rad2"]
        Data-fidelity term for each lambda.
    reg_norms : Quantity["dimensionless"]
        Unweighted regularization norm for each lambda.
    magnetizations : Quantity["dimensionless"]
        Reconstructed projected magnetizations, shape
        ``(n_lambdas, N, M, 2)``.
    ramp_coeffs : RampCoeffs
        Background ramp coefficients per lambda (batched
        ``Quantity`` fields with leading lambda dimension).
    corner_index : int
        Index of the detected L-curve corner.
    """
    lambdas: u.Quantity
    data_misfits: u.Quantity
    reg_norms: u.Quantity
    magnetizations: u.Quantity
    ramp_coeffs: RampCoeffs
    corner_index: int


class BootstrapThresholdResult(NamedTuple):
    """Result returned by :func:`bootstrap_threshold_uncertainty_2d`."""
    threshold: u.Quantity
    threshold_low: u.Quantity
    threshold_high: u.Quantity
    threshold_draws: u.Quantity
    magnetizations: u.Quantity  # Quantity["dimensionless"]
    mean_magnetization: u.Quantity  # Quantity["dimensionless"]
    mean_norm: u.Quantity  # Quantity["dimensionless"]
    norm_low: u.Quantity  # Quantity["dimensionless"]
    norm_high: u.Quantity  # Quantity["dimensionless"]
    norm_ci95: u.Quantity  # Quantity["dimensionless"]
    relative_ci95: u.Quantity  # Quantity["dimensionless"]
    mask_frequency: np.ndarray


@dataclasses.dataclass(frozen=True)
class NewtonCGConfig:
    """Configuration for the Newton-CG solver.

    Parameters
    ----------
    cg_maxiter : int
        Maximum number of conjugate-gradient iterations used to
        solve the Newton system ``H @ delta = -g``.
    cg_tol : float
        CG convergence tolerance (relative residual norm).
    """
    cg_maxiter: int = 10000
    cg_tol: float = 1e-16


@dataclasses.dataclass(frozen=True)
class AdamConfig:
    """Configuration for the Adam solver."""
    num_steps: int = 2000
    learning_rate: float = 1e-2
    patience: int = 50
    min_delta: float = 1e-6


@dataclasses.dataclass(frozen=True)
class LBFGSConfig:
    """Configuration for the L-BFGS solver."""
    num_steps: int = 500
    patience: int = 50
    min_delta: float = 1e-6


SolverConfig = Union[NewtonCGConfig, AdamConfig, LBFGSConfig]

_SOLVER_DEFAULTS = {
    "newton_cg": NewtonCGConfig,
    "adam": AdamConfig,
    "lbfgs": LBFGSConfig,
}


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
    original_mag = mag
    mag_q = cast(u.Quantity, _as_dimensionless_quantity(original_mag))
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

    if _contains_quantity(original_mag):
        return u.Quantity(loss, "")
    return loss


def forward_diff_norm(
    mag: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    r"""Forward-difference regularization norm matching Pyramid's convention.

    Computes ``\|D x\|^2`` using forward differences, with no
    per-pixel neighbor-count normalization.  This matches the
    ``FirstOrderRegularisator`` from Pyramid (which uses
    ``WeightedL2Square(D)`` with a sparse forward-diff operator),
    but is implemented in pure JAX for autodiff compatibility.

    Use this when comparing L-curve values with Pyramid's
    ``LCurve`` output, since :func:`exchange_loss_fn` applies
    adaptive central/one-sided differences with neighbor-count
    normalization that produce numerically different values.

    Parameters
    ----------
    mag
        Magnetization array of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` defining the active region.

    Returns
    -------
    jax.Array
        Scalar ``\|Dx\|^2`` (sum of squared forward differences
        over y and x, both magnetization components).
    """
    original_mag = mag
    mag_q = cast(u.Quantity, _as_dimensionless_quantity(original_mag))
    mag_val = mag_q.value
    mask = jnp.asarray(mask, dtype=bool)

    valid_y = mask[:-1, :] & mask[1:, :]
    dy = (mag_val[1:, :, :] - mag_val[:-1, :, :]) * valid_y[..., None]
    valid_x = mask[:, :-1] & mask[:, 1:]
    dx = (mag_val[:, 1:, :] - mag_val[:, :-1, :]) * valid_x[..., None]
    loss = jnp.sum(dy ** 2) + jnp.sum(dx ** 2)

    if _contains_quantity(original_mag):
        return u.Quantity(loss, "")
    return loss


def get_freq_grid(
    height: int,
    width: int,
    pixel_size: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
    f_y : jax.Array
        2D array of y-frequency values.
    f_x : jax.Array
        2D array of x-frequency values (real-FFT half-spectrum).
    denom : jax.Array
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
    return (
        _maybe_strip_quantity(f_y, pixel_size),
        _maybe_strip_quantity(f_x, pixel_size),
        _maybe_strip_quantity(denom, pixel_size),
    )


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
        "u_fft": _rfft2_keep_unit(u_pad),
        "v_fft": _rfft2_keep_unit(v_pad),
        "dim_uv": dim_uv,
        "dim_pad": dim_pad,
    }
    _assert_quantity_compatible(cast(u.Quantity, result["u_fft"]), "1 / nm2", "kernel['u_fft']")
    _assert_quantity_compatible(cast(u.Quantity, result["v_fft"]), "1 / nm2", "kernel['v_fft']")
    return result


def phase_mapper_rdfc(
    u_field: u.Quantity | jax.Array,
    v_field: u.Quantity | jax.Array,
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
    jax.Array
        Magnetic phase-shift image of shape ``(H, W)``.
    """
    original_u_field = u_field
    original_v_field = v_field
    return_quantity = _contains_quantity(original_u_field) or _contains_quantity(original_v_field)
    u_field_q = cast(u.Quantity, _as_dimensionless_quantity(original_u_field))
    v_field_q = cast(u.Quantity, _as_dimensionless_quantity(original_v_field))
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

    u_fft = _rfft2_keep_unit(u_pad)
    v_fft = _rfft2_keep_unit(v_pad)

    phase_fft = u_fft * rdfc_kernel["u_fft"] + v_fft * rdfc_kernel["v_fft"]
    phase_pad = cast(u.Quantity, _irfft2_keep_unit(phase_fft, s=dim_pad))

    phase = u.Quantity(
        jax.lax.dynamic_slice(
            phase_pad.value,
            (height - 1, width - 1),
            (height, width),
        ),
        str(phase_pad.unit),
    )
    if return_quantity:
        return phase
    return phase.value


def apply_ramp(ramp_coeffs, height, width, pixel_size):
    """Generate a first-order 2D polynomial background ramp.

    Accepts either :class:`RampCoeffs` with ``unxt.Quantity`` fields
    (returns a ``Quantity["angle"]``) or a plain JAX array of
    ``[offset, slope_y, slope_x]`` with a float *pixel_size* in nm
    (returns a plain JAX array).

    Parameters
    ----------
    ramp_coeffs : RampCoeffs or jax.Array
        Background phase-ramp coefficients.  When a :class:`RampCoeffs`,
        units are converted automatically; when a plain array, values
        must already be in ``[rad, rad/nm, rad/nm]``.
    height : int
        Number of pixels along the y-axis.
    width : int
        Number of pixels along the x-axis.
    pixel_size : Quantity["length"] or float
        Pixel size.  ``Quantity`` is converted to nm; a plain float
        is assumed to be in nanometres.

    Returns
    -------
    Quantity["angle"] or jax.Array
        Ramp image of shape ``(height, width)``.
    """
    original_ramp_coeffs = ramp_coeffs
    original_pixel_size = pixel_size
    return_quantity = _contains_quantity(original_ramp_coeffs) or _contains_quantity(original_pixel_size)
    pixel_size_q = _as_length_quantity(original_pixel_size)
    ramp_coeffs_q = _as_ramp_coeffs(original_ramp_coeffs)

    y, x = qnp.meshgrid(qnp.arange(height), qnp.arange(width), indexing="ij")
    ramp = ramp_coeffs_q.offset
    ramp = ramp + ramp_coeffs_q.slope_y * (y * pixel_size_q)
    ramp = ramp + ramp_coeffs_q.slope_x * (x * pixel_size_q)

    ramp = _as_angle_quantity(ramp)
    _assert_quantity_compatible(ramp, "rad", "ramp")
    if return_quantity:
        return ramp
    return ramp.value


def forward_model_single_rdfc_2d(
    magnetization: u.Quantity | jax.Array,
    ramp_coeffs: RampCoeffs | jax.Array,
    rdfc_kernel: dict[str, Any],
    pixel_size: u.Quantity | float,
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
    original_magnetization = magnetization
    original_ramp_coeffs = ramp_coeffs
    original_pixel_size = pixel_size
    return_quantity = any(
        _contains_quantity(item)
        for item in (original_magnetization, original_ramp_coeffs, original_pixel_size)
    )
    magnetization_q = _as_dimensionless_quantity(original_magnetization)
    ramp_coeffs_q = _as_ramp_coeffs(
        original_ramp_coeffs,
        dtype=magnetization_q.value.dtype,
    )
    pixel_size_q = _as_length_quantity(original_pixel_size)

    height, width = magnetization_q.shape[:2]

    u_field = magnetization_q[..., 0]
    v_field = magnetization_q[..., 1]

    # Kernel is defined in pixel coordinates; convert to physical phase scale.
    phase_density = pixel_size_q**2 * phase_mapper_rdfc(u_field, v_field, rdfc_kernel)
    phase = _as_angle_quantity(phase_density)
    ramp = apply_ramp(ramp_coeffs_q, height, width, pixel_size_q)

    phase_total = phase + ramp
    _assert_quantity_compatible(phase_total, "rad", "phase_total")
    if return_quantity:
        return phase_total
    return phase_total.value


def mbir_loss_2d(
    params: tuple[u.Quantity | jax.Array, RampCoeffs | jax.Array],
    mask: jax.Array,
    phase: u.Quantity | jax.Array,
    rdfc_kernel: dict[str, Any],
    pixel_size: u.Quantity | float,
    reg_config: dict[str, Any],
    reg_mask: jax.Array | None = None,
) -> u.Quantity:
    """Compute the MBIR loss for 2D projected magnetization.

    The total loss is the sum of a least-squares data-fidelity term
    and optional exchange-energy regularization.

    Parameters
    ----------
    params
        Tuple of ``(magnetization, ramp_coeffs)`` where
        *magnetization* has shape ``(N, M, 2)`` and *ramp_coeffs*
        has shape ``(3,)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization before the forward model.
    phase
        Observed phase image of shape ``(H, W)``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    pixel_size
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary.  Recognised key:
        ``'lambda_exchange'`` (float, default 0.0).
    reg_mask
        Optional regularization mask of shape ``(N, M)`` passed to
        :func:`exchange_loss_fn`.  Defaults to *mask* when not
        provided.

    Returns
    -------
    jax.Array
        Scalar loss value.
    """
    if reg_mask is None:
        reg_mask = mask
    magnetization, ramp_coeffs = params
    return_quantity = any(
        _contains_quantity(item)
        for item in (params, phase, pixel_size, reg_config)
    )
    mask = jnp.asarray(mask)
    reg_mask = jnp.asarray(reg_mask)
    phase_q = _as_angle_quantity(phase)
    magnetization_q = _as_dimensionless_quantity(magnetization)
    ramp_coeffs_q = _as_ramp_coeffs(
        ramp_coeffs,
        dtype=magnetization_q.value.dtype,
    )
    pixel_size_q = _as_length_quantity(pixel_size)

    magnetization_q = qnp.stack([
        magnetization_q[..., 0] * mask,
        magnetization_q[..., 1] * mask,
    ], axis=-1)

    predictions = forward_model_single_rdfc_2d(
        magnetization_q,
        ramp_coeffs_q,
        rdfc_kernel,
        pixel_size_q,
    )

    residuals = predictions - phase_q
    loss = 0.5 * qnp.sum(residuals ** 2)

    lambda_exchange = _lambda_exchange_quantity(reg_config.get("lambda_exchange", 0.0))

    loss += lambda_exchange * exchange_loss_fn(magnetization_q, reg_mask)
    _assert_quantity_compatible(loss, "rad2", "loss")

    if return_quantity:
        return loss
    return loss.value


def _run_newton_cg_solver_2d(
    phase: jax.Array,
    init_mag: jax.Array,
    mask: jax.Array,
    pixel_size: float,
    reg_config: dict[str, Any] | None = None,
    rdfc_kernel: dict[str, Any] | None = None,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 10000,
    init_ramp_coeffs: jax.Array | None = None,
    reg_mask: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Minimize :func:`mbir_loss_2d` using a single Newton-CG solve.

    The MBIR objective is quadratic in the reconstruction
    parameters, so Newton-CG reduces to a single linear solve of
    ``H @ delta = -g``.  The solver accuracy is therefore fully
    controlled by the inner CG tolerance and iteration budget.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(H, W)``.
    init_mag
        Initial magnetization of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary (see
        :func:`mbir_loss_2d`), default ``{}``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    cg_tol
        Tolerance for the CG solver, default 1e-8.
    cg_maxiter
        Maximum number of CG iterations for the Newton solve,
        default 10000.
    init_ramp_coeffs
        Initial ramp coefficients of shape ``(3,)``.  Defaults to
        zeros.
    reg_mask
        Optional regularization mask of shape ``(N, M)``.
        Defaults to *mask*.

    Returns
    -------
    (magnetization, ramp_coeffs) : tuple[jax.Array, jax.Array]
        Optimized magnetization ``(N, M, 2)`` and ramp ``(3,)``.
    loss_history : jax.Array
        Length-1 array containing the loss after the Newton update.
    """
    phase_q = _as_angle_quantity(phase)
    init_mag_q = _as_dimensionless_quantity(init_mag)
    pixel_size_q = _as_length_quantity(pixel_size)
    init_ramp_coeffs_q = _as_ramp_coeffs(
        init_ramp_coeffs,
        dtype=init_mag_q.value.dtype,
    )
    if reg_config is None:
        reg_config = {}
    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(phase_q.shape)
    kernel = cast(dict[str, Any], rdfc_kernel)

    x0_tree = (init_mag_q, init_ramp_coeffs_q)
    x0_flat, unravel = ravel_pytree(x0_tree)

    def objective_flat(x_flat):
        mag, ramp = unravel(x_flat)
        return mbir_loss_2d(
            (mag, ramp),
            mask,
            phase_q,
            kernel,
            pixel_size_q,
            reg_config,
            reg_mask=reg_mask,
        ).value

    loss_grad = jax.grad(objective_flat)

    grad_at_x0 = loss_grad(x0_flat)

    def matvec_hvp(v):
        return jax.jvp(loss_grad, (x0_flat,), (v,))[1]

    delta, _info = jax.scipy.sparse.linalg.cg(
        matvec_hvp, -grad_at_x0, tol=cg_tol, maxiter=cg_maxiter,
    )
    final_flat = x0_flat + delta
    history = u.Quantity(jnp.expand_dims(objective_flat(final_flat), axis=0), "rad2")

    final_mag, final_ramp = unravel(final_flat)

    return (final_mag, final_ramp), history


def _run_adam_solver_2d(
    phase: jax.Array,
    init_mag: jax.Array,
    mask: jax.Array,
    pixel_size: float,
    reg_config: dict[str, Any] | None = None,
    num_steps: int = 2000,
    learning_rate: float = 1e-2,
    rdfc_kernel: dict[str, Any] | None = None,
    init_ramp_coeffs: jax.Array | None = None,
    patience: int = 50,
    min_delta: float = 1e-6,
    reg_mask: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Minimize :func:`mbir_loss_2d` using the Adam optimizer.

    Includes early stopping: optimisation halts when the loss has
    not improved by more than *min_delta* for *patience* consecutive
    steps.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(N, M)``.
    init_mag
        Initial magnetization of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary (see
        :func:`mbir_loss_2d`), default ``{}``.
    num_steps
        Maximum number of optimisation steps, default 2000.
    learning_rate
        Adam learning rate, default 1e-2.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    init_ramp_coeffs
        Initial ramp coefficients of shape ``(3,)``.  Defaults to
        zeros.
    patience
        Number of steps without sufficient improvement before
        stopping, default 50.
    min_delta
        Minimum loss decrease to qualify as an improvement,
        default 1e-6.
    reg_mask
        Optional regularization mask of shape ``(N, M)``.
        Defaults to *mask*.

    Returns
    -------
    (magnetization, ramp_coeffs) : tuple[jax.Array, jax.Array]
        Optimized magnetization ``(N, M, 2)`` and ramp ``(3,)``.
    loss_history : jax.Array
        Per-step loss values (truncated at the step where early
        stopping triggered, if applicable).
    """
    phase_q = _as_angle_quantity(phase)
    init_mag_q = _as_dimensionless_quantity(init_mag)
    pixel_size_q = _as_length_quantity(pixel_size)
    init_ramp_coeffs_q = _as_ramp_coeffs(
        init_ramp_coeffs,
        dtype=init_mag_q.value.dtype,
    )
    if reg_config is None:
        reg_config = {}
    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(phase_q.shape)
    kernel = cast(dict[str, Any], rdfc_kernel)

    params = (init_mag_q, init_ramp_coeffs_q)
    x0_flat, unravel = ravel_pytree(params)

    def objective_flat(x_flat):
        mag, ramp = unravel(x_flat)
        return mbir_loss_2d(
            (mag, ramp),
            mask,
            phase_q,
            kernel,
            pixel_size_q,
            reg_config,
            reg_mask=reg_mask,
        ).value

    # Initialize optax optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(x0_flat)
    loss_and_grad = jax.value_and_grad(objective_flat)

    # initial_loss needed for state initialization
    init_loss = objective_flat(x0_flat)

    # State: (params, opt_state, step_idx, loss_history_arr, best_loss, patience_counter, stop_flag)
    # We allocate a fixed-size array for history because shapes must be static, but handle early exit via while_loop
    loss_history = jnp.zeros(num_steps, dtype=jnp.asarray(init_loss).dtype)
    loss_history = loss_history.at[0].set(init_loss)

    init_state = (
        x0_flat,
        opt_state,
        0,  # step index
        loss_history,
        init_loss,  # best_loss
        0,  # patience counter
        False,  # stop flag
    )

    def cond_fn(state):
        _, _, step_i, _, _, _, stop_flag = state
        return jnp.logical_and(step_i < num_steps, jnp.logical_not(stop_flag))

    def body_fn(state):
        curr_flat, curr_opt, i, history, best_loss, pat_count, _ = state

        loss_val, grad_flat = loss_and_grad(curr_flat)

        updates, next_opt = optimizer.update(grad_flat, curr_opt, curr_flat)
        next_flat = optax.apply_updates(curr_flat, updates)

        # Check early stopping
        # Using a simple check: if current loss is not improving best_loss by min_delta
        improved = loss_val < (best_loss - min_delta)
        next_best_loss = jnp.where(improved, loss_val, best_loss)
        next_pat_count = cast(jax.Array, jnp.where(improved, 0, pat_count + 1))
        next_stop = next_pat_count >= patience

        next_history = history.at[i].set(loss_val)

        return (
            next_flat,
            next_opt,
            i + 1,
            next_history,
            next_best_loss,
            next_pat_count,
            next_stop,
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    final_flat, _, _steps_taken, final_history, _, _, _ = final_state
    final_params = unravel(cast(jax.Array, final_flat))

    # Return full history (may contain trailing zeros after early stopping).
    return final_params, u.Quantity(final_history, "rad2")


@jax.jit(static_argnames=("num_steps", "patience", "min_delta"))
def _run_lbfgs_solver_2d(
    phase: jax.Array,
    init_mag: jax.Array,
    mask: jax.Array,
    pixel_size: float,
    reg_config: dict[str, Any] | None = None,
    num_steps: int = 500,
    rdfc_kernel: dict[str, Any] | None = None,
    init_ramp_coeffs: jax.Array | None = None,
    patience: int = 50,
    min_delta: float = 1e-6,
    reg_mask: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Minimize :func:`mbir_loss_2d` using L-BFGS with zoom line-search.

    Uses the optax L-BFGS implementation with early stopping.
    This function is JIT-compiled; *num_steps*, *patience*, and
    *min_delta* are static arguments.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(N, M)``.
    init_mag
        Initial magnetization of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary (see
        :func:`mbir_loss_2d`), default ``{}``.
    num_steps
        Maximum number of optimisation steps, default 500.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    init_ramp_coeffs
        Initial ramp coefficients of shape ``(3,)``.  Defaults to
        zeros.
    patience
        Number of steps without sufficient improvement before
        stopping, default 50.
    min_delta
        Minimum loss decrease to qualify as an improvement,
        default 1e-6.
    reg_mask
        Optional regularization mask of shape ``(N, M)``.
        Defaults to *mask*.

    Returns
    -------
    (magnetization, ramp_coeffs) : tuple[jax.Array, jax.Array]
        Optimized magnetization ``(N, M, 2)`` and ramp ``(3,)``.
    loss_history : jax.Array
        Per-step loss values (zero-filled after early stopping,
        if applicable).
    """
    phase_q = _as_angle_quantity(phase)
    init_mag_q = _as_dimensionless_quantity(init_mag)
    pixel_size_q = _as_length_quantity(pixel_size)
    init_ramp_coeffs_q = _as_ramp_coeffs(
        init_ramp_coeffs,
        dtype=init_mag_q.value.dtype,
    )
    if reg_config is None:
        reg_config = {}
    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(phase_q.shape)
    kernel = cast(dict[str, Any], rdfc_kernel)

    params = (init_mag_q, init_ramp_coeffs_q)
    x0_flat, unravel = ravel_pytree(params)

    def objective_flat(x_flat):
        mag, ramp = unravel(x_flat)
        return mbir_loss_2d(
            (mag, ramp),
            mask,
            phase_q,
            kernel,
            pixel_size_q,
            reg_config,
            reg_mask=reg_mask,
        ).value

    def value_fn(x_flat):
        return objective_flat(x_flat)

    ls_inst = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)
    solver = optax.lbfgs(linesearch=ls_inst)
    opt_state = solver.init(x0_flat)

    # Optax helper: lets us reuse value/grad stored in state (esp. with line-search)
    value_and_grad = optax.value_and_grad_from_state(value_fn)

    # Initial value for history dtype
    init_value, _init_grad = value_and_grad(x0_flat, state=opt_state)
    loss_history = jnp.zeros((num_steps,), dtype=jnp.asarray(init_value).dtype)

    # State: (params, opt_state, step_idx, history, best_loss, patience_counter, stop_flag)
    init_state = (
        x0_flat,
        opt_state,
        0,
        loss_history,
        init_value,
        0,
        False,
    )

    def cond_fn(state):
        _, _, i, _, _, _, stop_flag = state
        return jnp.logical_and(i < num_steps, jnp.logical_not(stop_flag))

    def body_fn(state):
        curr_flat, curr_state, i, history, best_loss, pat_count, _ = state

        value, grad = value_and_grad(curr_flat, state=curr_state)

        updates, next_state = solver.update(
            grad,                 # positional grads
            curr_state,
            curr_flat,
            value=value,          # current value at curr_params
            grad=grad,            # current grad at curr_params
            value_fn=value_fn,    # scalar objective for line-search
        )

        next_flat = optax.apply_updates(curr_flat, updates)
        history = history.at[i].set(value)

        improved = value < (best_loss - min_delta)
        next_best_loss = jnp.where(improved, value, best_loss)
        next_pat_count = cast(jax.Array, jnp.where(improved, 0, pat_count + 1))
        next_stop = next_pat_count >= patience

        return (
            next_flat,
            next_state,
            i + 1,
            history,
            next_best_loss,
            next_pat_count,
            next_stop,
        )

    final_flat, _final_opt_state, _, final_history, _, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return unravel(cast(jax.Array, final_flat)), u.Quantity(final_history, "rad2")


def solve_mbir_2d(
    phase,
    init_mag,
    mask,
    pixel_size,
    solver: str | SolverConfig = "newton_cg",
    reg_config=None,
    rdfc_kernel=None,
    init_ramp_coeffs=None,
    reg_mask=None,
):
    """
    Unified MBIR solver for 2D projected magnetization reconstruction.

    Parameters
    ----------
    phase : array_like
        Measured phase image in **radians**.
    init_mag : array_like
        Initial magnetization estimate, shape ``(N, M, 2)``.
    mask : array_like
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size : Quantity["length"] or float
        Pixel size as a ``unxt.Quantity`` with length units, or a float in nm.
    solver : str or SolverConfig, optional
        Which solver to use.  Pass a string (``"newton_cg"``, ``"adam"``,
        ``"lbfgs"``) for default parameters, or a config object
        (:class:`NewtonCGConfig`, :class:`AdamConfig`, :class:`LBFGSConfig`)
        for full control.  Default is ``"newton_cg"``.
    reg_config : dict, optional
        Regularization configuration (e.g. ``{"lambda_exchange": 1.0}``).
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
    init_ramp_coeffs : array_like, optional
        Initial ramp coefficients ``[offset, slope_y, slope_x]``
        in units of **[rad, rad/nm, rad/nm]**.
    reg_mask : array_like, optional
        Regularization mask of shape ``(N, M)``.  Defaults to *mask*.

    Returns
    -------
    SolverResult
        Named tuple with fields ``magnetization``, ``ramp_coeffs``, and
        ``loss_history``.

        .. note::

           For iterative solvers (Adam, L-BFGS) the ``loss_history`` array
           may contain trailing zeros if the solver stopped early.
    """
    phase = _as_angle_quantity(phase)
    init_mag = _as_dimensionless_quantity(init_mag)
    pixel_size = _as_length_quantity(pixel_size)
    if init_ramp_coeffs is not None:
        init_ramp_coeffs = _as_ramp_coeffs(
            init_ramp_coeffs,
            dtype=init_mag.value.dtype,
        )
    if isinstance(solver, str):
        solver_name = solver.lower()
        if solver_name not in _SOLVER_DEFAULTS:
            raise ValueError(
                f"Unknown solver {solver!r}. "
                f"Choose from {list(_SOLVER_DEFAULTS)}"
            )
        config = _SOLVER_DEFAULTS[solver_name]()
    elif isinstance(solver, (NewtonCGConfig, AdamConfig, LBFGSConfig)):
        config = solver
    else:
        raise TypeError(
            f"solver must be a string or a SolverConfig instance, got {type(solver)}"
        )

    shared = dict(
        phase=phase,
        init_mag=init_mag,
        mask=mask,
        pixel_size=pixel_size,
        reg_config=reg_config,
        rdfc_kernel=rdfc_kernel,
        init_ramp_coeffs=init_ramp_coeffs,
        reg_mask=reg_mask,
    )

    if isinstance(config, NewtonCGConfig):
        (mag, ramp), loss_history = _run_newton_cg_solver_2d(
            **shared,
            cg_tol=config.cg_tol,
            cg_maxiter=config.cg_maxiter,
        )
    elif isinstance(config, AdamConfig):
        (mag, ramp), loss_history = _run_adam_solver_2d(
            **shared,
            num_steps=config.num_steps,
            learning_rate=config.learning_rate,
            patience=config.patience,
            min_delta=config.min_delta,
        )
    elif isinstance(config, LBFGSConfig):
        (mag, ramp), loss_history = _run_lbfgs_solver_2d(
            **shared,
            num_steps=config.num_steps,
            patience=config.patience,
            min_delta=config.min_delta,
        )
    else:
        raise AssertionError(f"Unhandled solver config type: {type(config)}")

    result = SolverResult(
        magnetization=mag,
        ramp_coeffs=cast(RampCoeffs, ramp),
        loss_history=loss_history,
    )
    _assert_solver_result_units(result)
    return result


def reconstruct_2d(
    phase,
    pixel_size,
    mask,
    lam=1e-3,
    solver: str | SolverConfig = "newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config: SolverConfig | None = None,
):
    """Convenience wrapper for 2D MBIR magnetization reconstruction.

    Provides a simple interface similar to pyramid's
    ``reconstruction_2d_from_phasemap``.  Builds the RDFC kernel,
    initial magnetization guess, and mask automatically.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)`` in **radians**.
    pixel_size : Quantity["length"] or float
        Pixel size as a ``unxt.Quantity`` with length units, or a float in nm.  Must be positive.
    mask : array_like, optional
        Binary mask of shape ``(N, M)``.  Defaults to all ones.
    lam : float, optional
        Regularization weight (``lambda_exchange``), default 1e-3.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``, ``"adam"``,
        ``"lbfgs"``) or a :class:`SolverConfig` instance.
        Ignored when *solver_config* is provided.
        Default is ``"newton_cg"``.
    reg_mask : array_like, optional
        Separate regularization mask of shape ``(N, M)``.
        Defaults to *mask*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.
    solver_config : SolverConfig, optional
        Explicit solver configuration object.  When provided,
        the *solver* string argument is ignored.

    Returns
    -------
    SolverResult
        Named tuple with fields ``magnetization``, ``ramp_coeffs``,
        and ``loss_history``.
    """
    pixel_size = _as_length_quantity(pixel_size)
    _validate_positive(pixel_size, "pixel_size")

    phase = _as_angle_quantity(phase)
    if mask is None:
        mask = jnp.ones(phase.shape, dtype=bool)
    else:
        mask = jnp.asarray(mask, dtype=bool)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    init_mag = u.Quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")
    lam_q = _lambda_exchange_quantity(lam)
    reg_config = {"lambda_exchange": lam_q.value}

    if solver_config is not None:
        solver = solver_config

    result = solve_mbir_2d(
        phase=phase,
        init_mag=init_mag,
        mask=mask,
        pixel_size=pixel_size,
        solver=solver,
        reg_config=reg_config,
        rdfc_kernel=rdfc_kernel,
        reg_mask=reg_mask,
    )
    _assert_solver_result_units(result)
    return result


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
        to be the projected (thickness-integrated) magnetization
        in units of **Tesla**.
    pixel_size : Quantity["length"] or float
        Pixel size as a ``unxt.Quantity`` with length units, or a float in nm.  Must be positive.
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


def reconstruct_2d_ensemble(
    phase,
    masks,
    pixel_size,
    lam=1e-3,
    solver: str | SolverConfig = "newton_cg",
    reg_masks=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
) -> u.Quantity:
    """Batched MBIR reconstruction over an ensemble of bootstrap masks.

    Runs :func:`reconstruct_2d` for each mask in the ensemble using
    ``jax.vmap`` for efficient parallel execution on GPU.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(H, W)`` in **radians**.
    masks : array_like
        Bootstrap mask ensemble of shape ``(N_boot, H, W)``.
    pixel_size : Quantity["length"] or float
        Pixel size as a ``unxt.Quantity`` with length units, or a float in nm.  Must be positive.
    lam : float, optional
        Regularization weight (``lambda_exchange``), default 1e-3.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``, ``"adam"``,
        ``"lbfgs"``) or a :class:`SolverConfig` instance.
        Ignored when *solver_config* is provided.
        Default is ``"newton_cg"``.
    reg_masks : array_like, optional
        Separate regularization masks of shape ``(N_boot, H, W)``.
        Defaults to *masks*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.
    solver_config : SolverConfig, optional
        Explicit solver configuration object.  When provided,
        the *solver* string argument is ignored.

    Returns
    -------
    Quantity["dimensionless"]
        Reconstructed magnetization ensemble of shape
        ``(N_boot, H, W, 2)``.

    Notes
    -----
    For iterative solvers (Adam, L-BFGS) early stopping is
    effectively disabled under ``vmap``; all bootstrap samples
    run for the maximum number of steps.
    """
    pixel_size = _as_length_quantity(pixel_size)
    _validate_positive(pixel_size, "pixel_size")

    phase = _as_angle_quantity(phase)
    masks = jnp.asarray(masks)
    if reg_masks is None:
        reg_masks = masks
    else:
        reg_masks = jnp.asarray(reg_masks)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    # Resolve solver config once (Python-level dispatch, outside vmap)
    if solver_config is not None:
        config = solver_config
    elif isinstance(solver, str):
        solver_name = solver.lower()
        if solver_name not in _SOLVER_DEFAULTS:
            raise ValueError(
                f"Unknown solver {solver!r}. "
                f"Choose from {list(_SOLVER_DEFAULTS)}"
            )
        config = _SOLVER_DEFAULTS[solver_name]()
    elif isinstance(solver, (NewtonCGConfig, AdamConfig, LBFGSConfig)):
        config = solver
    else:
        raise TypeError(
            f"solver must be a string or a SolverConfig instance, "
            f"got {type(solver)}"
        )

    init_mag = u.Quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")
    lam_q = _lambda_exchange_quantity(lam)
    reg_config = {"lambda_exchange": lam_q.value}

    # Build a vmappable function for the chosen solver
    if isinstance(config, NewtonCGConfig):
        def _solve_single_newton(mask, reg_mask):
            (mag, _ramp), _loss = _run_newton_cg_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size=pixel_size,
                reg_config=reg_config,
                rdfc_kernel=rdfc_kernel,
                cg_tol=config.cg_tol,
                cg_maxiter=config.cg_maxiter,
                reg_mask=reg_mask,
            )
            return cast(u.Quantity, mag).value
        solve_single = _solve_single_newton
    elif isinstance(config, AdamConfig):
        def _solve_single_adam(mask, reg_mask):
            (mag, _ramp), _loss = _run_adam_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size=pixel_size,
                reg_config=reg_config,
                num_steps=config.num_steps,
                learning_rate=config.learning_rate,
                rdfc_kernel=rdfc_kernel,
                patience=config.patience,
                min_delta=config.min_delta,
                reg_mask=reg_mask,
            )
            return cast(u.Quantity, mag).value
        solve_single = _solve_single_adam
    elif isinstance(config, LBFGSConfig):
        def _solve_single_lbfgs(mask, reg_mask):
            (mag, _ramp), _loss = _run_lbfgs_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size=pixel_size,
                reg_config=reg_config,
                num_steps=config.num_steps,
                rdfc_kernel=rdfc_kernel,
                patience=config.patience,
                min_delta=config.min_delta,
                reg_mask=reg_mask,
            )
            return mag.value
        solve_single = _solve_single_lbfgs
    else:
        raise AssertionError(f"Unhandled solver config type: {type(config)}")

    solve_batch = jax.jit(jax.vmap(solve_single, in_axes=(0, 0)))
    return u.Quantity(solve_batch(masks, reg_masks), "")


# Mapping from projection axis to (sum_axis, coeff_matrix, need_transpose).
# coeff maps (mx, my, mz) -> (u, v) following pyramid's SimpleProjector.
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
    original_magnetization_3d = magnetization_3d
    return_quantity = _contains_quantity(original_magnetization_3d)
    magnetization_3d = _as_dimensionless_quantity(original_magnetization_3d)

    # Sum along projection direction: (Z, Y, X, 3) -> (*, *, 3)
    summed = cast(u.Quantity, qnp.sum(magnetization_3d, axis=cfg["sum_axis"]))

    # Mix (mx, my, mz) -> (u, v) via coefficient matrix
    coeff = jnp.array(cfg["coeff"], dtype=summed.value.dtype)  # (2, 3)
    projected = qnp.einsum("...c,oc->...o", summed, coeff)  # (*, *, 2)

    if cfg["transpose"]:
        projected = qnp.transpose(projected, (1, 0, 2))

    if return_quantity:
        return projected
    return cast(u.Quantity, projected).value


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
    pixel_size : Quantity["length"] or float
        Voxel size in nanometres.
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


def decompose_loss(
    magnetization,
    ramp_coeffs,
    phase,
    mask,
    reg_mask,
    rdfc_kernel,
    pixel_size,
    *,
    pyramid_compat=False,
) -> tuple[u.Quantity, u.Quantity]:
    """Decompose the MBIR loss into data-fidelity and regularization terms.

    Evaluates the two components of the loss **without** the
    ``lambda_exchange`` multiplier on the regularization term, so
    that they can be compared on an L-curve plot.

    Parameters
    ----------
    magnetization : array_like
        Reconstructed magnetization of shape ``(N, M, 2)``.
    ramp_coeffs : array_like
        Background ramp coefficients ``[offset, slope_y, slope_x]``.
    phase : array_like
        Observed phase image of shape ``(N, M)``.
    mask : array_like
        Binary mask of shape ``(N, M)`` applied to the
        magnetization before the forward model.
    reg_mask : array_like
        Regularization mask of shape ``(N, M)`` passed to
        :func:`exchange_loss_fn`.
    rdfc_kernel : dict
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
    pixel_size : Quantity["length"] or float
        Pixel size in nanometres.
    pyramid_compat : bool, optional
        If *True*, compute the regularization norm using
        :func:`forward_diff_norm` (simple forward differences, no
        per-pixel normalization), which matches Pyramid's
        ``FirstOrderRegularisator`` convention.  Default *False*
        uses :func:`exchange_loss_fn` (adaptive stencil with
        neighbor-count normalization).

    Returns
    -------
    data_misfit : Quantity["rad2"]
        ``sum((predicted - observed)**2)`` — the squared-residual
        norm, matching Pyramid's ``chisq_m`` convention (no 1/2
        factor).
    exchange_norm : Quantity["dimensionless"]
        Unweighted exchange regularization norm (no lambda
        multiplier).  When *pyramid_compat=True* this is
        ``||Dx||²`` (forward differences); otherwise it uses the
        adaptive stencil from :func:`exchange_loss_fn`.
    """
    pixel_size = _as_length_quantity(pixel_size)
    magnetization = _as_dimensionless_quantity(magnetization)
    ramp_coeffs = _as_ramp_coeffs(
        ramp_coeffs,
        dtype=magnetization.value.dtype,
    )
    phase = _as_angle_quantity(phase)
    mask = jnp.asarray(mask)
    reg_mask = jnp.asarray(reg_mask)

    masked_mag = qnp.stack([
        magnetization[..., 0] * mask,
        magnetization[..., 1] * mask,
    ], axis=-1)

    predicted = forward_model_single_rdfc_2d(
        masked_mag, ramp_coeffs, rdfc_kernel, pixel_size,
    )
    residuals = predicted - phase
    data_misfit = qnp.sum(residuals ** 2)
    if pyramid_compat:
        exchange_norm = forward_diff_norm(masked_mag, reg_mask)
    else:
        exchange_norm = exchange_loss_fn(masked_mag, reg_mask)

    _assert_quantity_compatible(cast(u.Quantity, data_misfit), "rad2", "data_misfit")
    _assert_quantity_compatible(cast(u.Quantity, exchange_norm), "", "exchange_norm")

    return data_misfit, exchange_norm

def bootstrap_threshold_uncertainty_2d(
    phase,
    mip_phase,
    threshold,
    pixel_size,
    lam=1e-3,
    solver: str | SolverConfig = "newton_cg",
    n_boot=50,
    threshold_low=None,
    threshold_high=None,
    rng_seed=0,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
) -> BootstrapThresholdResult:
    """Bootstrap a thresholded mask ensemble and summarize the uncertainty.

    Threshold draws are sampled uniformly from ``[threshold_low, threshold_high]``.
    Each draw defines a binary mask from ``abs(mip_phase) > threshold_draw``.
    The masks are passed through :func:`reconstruct_2d_ensemble` and the
    resulting magnetization ensemble is summarized with percentile maps for
    ``|M|``.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(H, W)``.
    mip_phase
        MIP phase image used for thresholding, shape ``(H, W)``.
    threshold
        Central threshold value around which the bootstrap draws are sampled.
    pixel_size
        Pixel size in nanometres.
    lam : float, optional
        Regularization weight (``lambda_exchange``), default 1e-3.
    solver : str or SolverConfig, optional
        Solver selection string or config object.  Ignored when
        *solver_config* is provided.  Default ``"newton_cg"``.
    n_boot : int, optional
        Number of threshold draws, default 50.
    threshold_low : float, optional
        Lower bound for the threshold draws.  Defaults to
        ``threshold - 0.25``.
    threshold_high : float, optional
        Upper bound for the threshold draws.  Defaults to
        ``threshold + 0.25``.
    rng_seed : int, optional
        Seed for the pseudo-random number generator, default 0.
    geometry : str, optional
        Voxel geometry for the RDFC kernel, default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when ``None``.
    solver_config : SolverConfig, optional
        Explicit solver configuration object.

    Returns
    -------
    BootstrapThresholdResult
        Summary object containing the threshold draws, reconstructed
        magnetizations, 2.5th and 97.5th percentile maps, their 95% width,
        the relative 95% width, and the mask inclusion frequency.
    """
    phase = _as_angle_quantity(phase)
    mip_phase = _as_angle_quantity(mip_phase)
    pixel_size = _as_length_quantity(pixel_size)
    _validate_positive(pixel_size, "pixel_size")
    threshold = _as_angle_quantity(threshold)
    if phase.shape != mip_phase.shape:
        raise ValueError(
            f"phase and mip_phase must have the same shape; got {phase.shape} and {mip_phase.shape}."
        )

    if threshold_low is None:
        threshold_low = threshold - u.Quantity(0.25, "rad")
    else:
        threshold_low = _as_angle_quantity(threshold_low)
    if threshold_high is None:
        threshold_high = threshold + u.Quantity(0.25, "rad")
    else:
        threshold_high = _as_angle_quantity(threshold_high)

    threshold_low_q = cast(u.Quantity, u.uconvert("rad", threshold_low))
    threshold_high_q = cast(u.Quantity, u.uconvert("rad", threshold_high))
    threshold_q = cast(u.Quantity, u.uconvert("rad", threshold))
    threshold_low_value = float(np.asarray(threshold_low_q.value))
    threshold_high_value = float(np.asarray(threshold_high_q.value))
    threshold_value = float(np.asarray(threshold_q.value))
    if threshold_high_value <= threshold_low_value:
        raise ValueError(
            f"threshold_high must be greater than threshold_low; got {threshold_low} and {threshold_high}."
        )

    rng = np.random.default_rng(rng_seed)
    threshold_draws = u.Quantity(
        rng.uniform(
            low=threshold_low_value,
            high=threshold_high_value,
            size=n_boot,
        ),
        "rad",
    )

    mip_phase_q = cast(u.Quantity, u.uconvert("rad", mip_phase))
    mip_abs = np.abs(np.asarray(mip_phase_q.value))
    reference_mask = (mip_abs > threshold_value).astype(np.float64)
    bootstrap_masks = (
        mip_abs[None, ...] > np.asarray(threshold_draws.value)[:, None, None]
    ).astype(np.float64)

    for draw_index in range(n_boot):
        if bootstrap_masks[draw_index].sum() == 0:
            bootstrap_masks[draw_index] = reference_mask

    bootstrap_mag = reconstruct_2d_ensemble(
        phase=phase,
        masks=bootstrap_masks,
        pixel_size=pixel_size,
        lam=lam,
        solver=solver,
        reg_masks=bootstrap_masks,
        geometry=geometry,
        prw_vec=prw_vec,
        rdfc_kernel=rdfc_kernel,
        solver_config=solver_config,
    )

    bootstrap_mag = _as_dimensionless_quantity(bootstrap_mag)
    mean_magnetization = _as_dimensionless_quantity(qnp.mean(bootstrap_mag, axis=0))
    mean_norm = _as_dimensionless_quantity(
        qnp.sqrt(qnp.sum(mean_magnetization ** 2, axis=-1))
    )

    norm_samples = _as_dimensionless_quantity(
        qnp.sqrt(qnp.sum(bootstrap_mag ** 2, axis=-1))
    )
    norm_low = u.Quantity(
        np.percentile(np.asarray(norm_samples.value), 2.5, axis=0),
        str(norm_samples.unit),
    )
    norm_high = u.Quantity(
        np.percentile(np.asarray(norm_samples.value), 97.5, axis=0),
        str(norm_samples.unit),
    )
    norm_ci95 = norm_high - norm_low
    relative_ci95 = _as_dimensionless_quantity(
        norm_ci95 / (mean_norm + u.Quantity(1e-12, ""))
    )
    mask_frequency = bootstrap_masks.mean(axis=0)

    return BootstrapThresholdResult(
        threshold=threshold,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        threshold_draws=threshold_draws,
        magnetizations=bootstrap_mag,
        mean_magnetization=mean_magnetization,
        mean_norm=mean_norm,
        norm_low=norm_low,
        norm_high=norm_high,
        norm_ci95=norm_ci95,
        relative_ci95=relative_ci95,
        mask_frequency=mask_frequency,
    )


def plot_bootstrap_threshold_uncertainty(result: BootstrapThresholdResult):
    """Plot the summary produced by :func:`bootstrap_threshold_uncertainty_2d`."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)

    def as_array(value):
        if isinstance(value, u.Quantity):
            return np.asarray(value.value)
        return np.asarray(value)

    def set_panel_title(ax, title, subtitle):
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    mean_norm = as_array(result.mean_norm)
    norm_low = as_array(result.norm_low)
    norm_high = as_array(result.norm_high)
    norm_ci95 = as_array(result.norm_ci95)
    relative_ci95 = as_array(result.relative_ci95)
    mask_frequency = as_array(result.mask_frequency)

    display_mask = mask_frequency > 0.5
    if not np.any(display_mask):
        display_mask = np.ones_like(mean_norm, dtype=bool)

    mean_region = mean_norm[display_mask]
    low_region = norm_low[display_mask]
    high_region = norm_high[display_mask]
    ci_region = norm_ci95[display_mask]
    rel_region = 100.0 * relative_ci95[display_mask]

    vmax_mean = float(np.percentile(mean_region, 99))
    vmax_low = float(np.percentile(low_region, 99))
    vmax_high = float(np.percentile(high_region, 99))
    vmax_ci = float(np.percentile(ci_region, 99))
    vmax_rel = float(np.percentile(rel_region, 99))

    im = axs[0, 0].imshow(
        mean_norm, cmap="viridis", origin="lower", vmin=0, vmax=vmax_mean
    )
    set_panel_title(
        axs[0, 0],
        "Bootstrap mean |M|",
        "Typical magnitude across draws.",
    )
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046)

    im = axs[0, 1].imshow(
        norm_low, cmap="viridis", origin="lower", vmin=0, vmax=vmax_low
    )
    set_panel_title(
        axs[0, 1],
        "2.5% percentile of |M|",
        "Lower 95% interval bound.",
    )
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046)

    im = axs[0, 2].imshow(
        norm_high, cmap="viridis", origin="lower", vmin=0, vmax=vmax_high
    )
    set_panel_title(
        axs[0, 2],
        "97.5% percentile of |M|",
        "Upper 95% interval bound.",
    )
    plt.colorbar(im, ax=axs[0, 2], fraction=0.046)

    im = axs[1, 0].imshow(
        norm_ci95, cmap="magma", origin="lower", vmin=0, vmax=vmax_ci
    )
    set_panel_title(
        axs[1, 0],
        "95% CI width of |M|",
        "Larger values mean more spread.",
    )
    plt.colorbar(im, ax=axs[1, 0], fraction=0.046)

    im = axs[1, 1].imshow(
        100.0 * relative_ci95,
        cmap="cividis",
        origin="lower",
        vmin=0,
        vmax=vmax_rel,
    )
    set_panel_title(
        axs[1, 1],
        "Relative 95% CI width of |M| (%)",
        "Width divided by the local mean.",
    )
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046)

    im = axs[1, 2].imshow(
        mask_frequency, cmap="gray", origin="lower", vmin=0, vmax=1
    )
    set_panel_title(
        axs[1, 2],
        "Threshold inclusion frequency",
        "Near 1 is stable; near 0 is rare.",
    )
    plt.colorbar(im, ax=axs[1, 2], fraction=0.046)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axs


def kneedle_corner(data_misfits, reg_norms):
    """Find the L-curve corner using the Kneedle algorithm.

    Projects points in log-log space onto the line connecting the
    two endpoints and returns the index of the point with the
    largest perpendicular distance (the "elbow").

    Parameters
    ----------
    data_misfits : array_like
        1D array of data-fidelity values, one per lambda.
    reg_norms : array_like
        1D array of regularization norms (unweighted), one per
        lambda.

    Returns
    -------
    corner_index : int
        Index into the input arrays of the detected corner point.
        Returns ``-1`` when fewer than 3 points are provided.
    score : float
        Maximum normalized perpendicular distance.  Larger values
        indicate a more pronounced corner.
    """
    x = np.log10(np.asarray(reg_norms, dtype=np.float64))
    y = np.log10(np.asarray(data_misfits, dtype=np.float64))

    if len(x) < 3:
        return -1, 0.0

    order = np.argsort(x)
    x, y = x[order], y[order]

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    if x_range == 0 or y_range == 0:
        return -1, 0.0

    x_n = (x - x.min()) / (x_range + 1e-30)
    y_n = (y - y.min()) / (y_range + 1e-30)

    # Perpendicular distance from the line joining first and last point
    x1, y1 = x_n[0], y_n[0]
    x2, y2 = x_n[-1], y_n[-1]
    line_len = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-30
    d = np.abs(
        (y2 - y1) * x_n - (x2 - x1) * y_n + x2 * y1 - y2 * x1
    ) / line_len

    best_sorted = int(np.argmax(d))
    return int(order[best_sorted]), float(d[best_sorted])


def lcurve_sweep(
    phase,
    mask,
    pixel_size,
    lambdas,
    solver: str | SolverConfig = "newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
    pyramid_compat=False,
) -> LCurveResult:
    """Sequential L-curve sweep over regularization weights.

    Runs :func:`solve_mbir_2d` for each value in *lambdas*,
    collects the data-fidelity and regularization norms, and
    detects the L-curve corner via :func:`kneedle_corner`.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)`` in **radians**.
    mask : array_like
        Binary mask of shape ``(N, M)``.
    pixel_size : Quantity["length"] or float
        Pixel size as a ``unxt.Quantity`` with length units, or a float in nm.  Must be positive.
    lambdas : array_like
        1D array of ``lambda_exchange`` values to sweep.
    solver : str or SolverConfig, optional
        Solver selection string or config object.  Ignored when
        *solver_config* is provided.  Default ``"newton_cg"``.
    reg_mask : array_like, optional
        Regularization mask of shape ``(N, M)``.  Defaults to
        *mask*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel, default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel.  Built automatically when ``None``.
    solver_config : SolverConfig, optional
        Explicit solver configuration.
    pyramid_compat : bool, optional
        If *True*, compute the regularization norm using
        :func:`forward_diff_norm` (simple forward differences)
        instead of :func:`exchange_loss_fn`, to match Pyramid's
        ``FirstOrderRegularisator`` convention.  Default *False*.

    Returns
    -------
    LCurveResult
        Named tuple with ``lambdas``, ``data_misfits``,
        ``reg_norms``, ``magnetizations``, ``ramp_coeffs``, and
        ``corner_index``.
    """
    phase = _as_angle_quantity(phase)
    pixel_size = _as_length_quantity(pixel_size)
    _validate_positive(pixel_size, "pixel_size")

    mask = jnp.asarray(mask, dtype=bool)
    if reg_mask is None:
        reg_mask = mask
    else:
        reg_mask = jnp.asarray(reg_mask)

    lambdas_q = _lambda_exchange_quantity(lambdas)
    lambdas = np.atleast_1d(np.asarray(lambdas_q.value, dtype=np.float64))

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    actual_solver = solver_config if solver_config is not None else solver

    data_misfits = []
    reg_norms = []
    mag_list = []
    ramp_list = []

    init_mag = u.Quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")

    for lam in lambdas:
        reg_config = {"lambda_exchange": lam}

        result = solve_mbir_2d(
            phase=phase,
            init_mag=init_mag,
            mask=mask,
            pixel_size=pixel_size,
            solver=actual_solver,
            reg_config=reg_config,
            rdfc_kernel=rdfc_kernel,
            reg_mask=reg_mask,
        )

        dm, rn = decompose_loss(
            result.magnetization, result.ramp_coeffs,
            phase, mask, reg_mask, rdfc_kernel, pixel_size,
            pyramid_compat=pyramid_compat,
        )
        data_misfits.append(float(dm.value))
        reg_norms.append(float(rn.value))
        mag_list.append(result.magnetization.value)
        ramp_list.append(jnp.array([
            result.ramp_coeffs.offset.value,
            result.ramp_coeffs.slope_y.value,
            result.ramp_coeffs.slope_x.value,
        ]))

    data_misfits = np.array(data_misfits)
    reg_norms = np.array(reg_norms)
    corner_idx, _ = kneedle_corner(data_misfits, reg_norms)

    all_mag = jnp.stack(mag_list)
    all_ramp = jnp.stack(ramp_list)

    result = LCurveResult(
        lambdas=u.Quantity(lambdas, "rad2"),
        data_misfits=u.Quantity(data_misfits, "rad2"),
        reg_norms=u.Quantity(reg_norms, ""),
        magnetizations=u.Quantity(all_mag, ""),
        ramp_coeffs=RampCoeffs(
            offset=u.Quantity(all_ramp[:, 0], "rad"),
            slope_y=u.Quantity(all_ramp[:, 1], "rad/nm"),
            slope_x=u.Quantity(all_ramp[:, 2], "rad/nm"),
        ),
        corner_index=corner_idx,
    )
    _assert_lcurve_result_units(result)
    return result


def lcurve_sweep_vmap(
    phase,
    mask,
    pixel_size,
    lambdas,
    solver: str | SolverConfig = "newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
    pyramid_compat=False,
) -> LCurveResult:
    """Parallel L-curve sweep using ``jax.vmap`` over lambda values.

    Runs all reconstructions in parallel (no warm-starting).
    This is faster on GPU when many lambda values are evaluated,
    but uses more memory than :func:`lcurve_sweep`.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)`` in **radians**.
    mask : array_like
        Binary mask of shape ``(N, M)``.
    pixel_size : Quantity["length"] or float
        Pixel size as a ``unxt.Quantity`` with length units, or a float in nm.  Must be positive.
    lambdas : array_like
        1D array of ``lambda_exchange`` values to sweep.
    solver : str or SolverConfig, optional
        Solver selection string or config object.  Ignored when
        *solver_config* is provided.  Default ``"newton_cg"``.
    reg_mask : array_like, optional
        Regularization mask of shape ``(N, M)``.  Defaults to
        *mask*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel, default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel.  Built automatically when ``None``.
    solver_config : SolverConfig, optional
        Explicit solver configuration.
    pyramid_compat : bool, optional
        If *True*, compute the regularization norm using
        :func:`forward_diff_norm` (simple forward differences)
        instead of :func:`exchange_loss_fn`, to match Pyramid's
        ``FirstOrderRegularisator`` convention.  Default *False*.

    Returns
    -------
    LCurveResult
        Named tuple with ``lambdas``, ``data_misfits``,
        ``reg_norms``, ``magnetizations``, ``ramp_coeffs``, and
        ``corner_index``.

    Notes
    -----
    For iterative solvers (Adam, L-BFGS) early stopping is
    effectively disabled under ``vmap``; all lambda values run for
    the maximum number of steps.
    """
    phase = _as_angle_quantity(phase)
    pixel_size = _as_length_quantity(pixel_size)
    _validate_positive(pixel_size, "pixel_size")

    mask = jnp.asarray(mask, dtype=bool)
    if reg_mask is None:
        reg_mask = mask
    else:
        reg_mask = jnp.asarray(reg_mask)

    lambdas_q = _lambda_exchange_quantity(lambdas)
    lambdas_np = np.atleast_1d(np.asarray(lambdas_q.value, dtype=np.float64))
    lambdas_jax = jnp.asarray(lambdas_np)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    if solver_config is not None:
        config = solver_config
    elif isinstance(solver, str):
        solver_name = solver.lower()
        if solver_name not in _SOLVER_DEFAULTS:
            raise ValueError(
                f"Unknown solver {solver!r}. "
                f"Choose from {list(_SOLVER_DEFAULTS)}"
            )
        config = _SOLVER_DEFAULTS[solver_name]()
    elif isinstance(solver, (NewtonCGConfig, AdamConfig, LBFGSConfig)):
        config = solver
    else:
        raise TypeError(
            f"solver must be a string or a SolverConfig instance, "
            f"got {type(solver)}"
        )

    init_mag = u.Quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")

    # Build a vmappable function for the chosen solver.
    if isinstance(config, NewtonCGConfig):
        def _solve_for_lam_newton(lam):
            reg_config = {"lambda_exchange": lam}
            (mag, ramp), _loss = _run_newton_cg_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size=pixel_size,
                reg_config=reg_config,
                rdfc_kernel=rdfc_kernel,
                cg_tol=config.cg_tol,
                cg_maxiter=config.cg_maxiter,
                reg_mask=reg_mask,
            )
            return cast(u.Quantity, mag).value, _ramp_coeffs_to_array(cast(RampCoeffs, ramp))
        solve_for_lam = _solve_for_lam_newton
    elif isinstance(config, AdamConfig):
        def _solve_for_lam_adam(lam):
            reg_config = {"lambda_exchange": lam}
            (mag, ramp), _loss = _run_adam_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size=pixel_size,
                reg_config=reg_config,
                num_steps=config.num_steps,
                learning_rate=config.learning_rate,
                rdfc_kernel=rdfc_kernel,
                patience=config.patience,
                min_delta=config.min_delta,
                reg_mask=reg_mask,
            )
            return cast(u.Quantity, mag).value, _ramp_coeffs_to_array(cast(RampCoeffs, ramp))
        solve_for_lam = _solve_for_lam_adam
    elif isinstance(config, LBFGSConfig):
        def _solve_for_lam_lbfgs(lam):
            reg_config = {"lambda_exchange": lam}
            (mag, ramp), _loss = _run_lbfgs_solver_2d(
                phase=phase,
                init_mag=init_mag,
                mask=mask,
                pixel_size=pixel_size,
                reg_config=reg_config,
                num_steps=config.num_steps,
                rdfc_kernel=rdfc_kernel,
                patience=config.patience,
                min_delta=config.min_delta,
                reg_mask=reg_mask,
            )
            return cast(u.Quantity, mag).value, _ramp_coeffs_to_array(cast(RampCoeffs, ramp))
        solve_for_lam = _solve_for_lam_lbfgs
    else:
        raise AssertionError(f"Unhandled solver config type: {type(config)}")

    solve_batch = jax.jit(jax.vmap(solve_for_lam))
    all_mag, all_ramp = solve_batch(lambdas_jax)

    # Decompose losses (vmapped)
    _norm_fn = forward_diff_norm if pyramid_compat else exchange_loss_fn

    def _decompose_single(mag, ramp):
        masked_mag = _as_dimensionless_quantity(qnp.stack([
            mag[..., 0] * mask,
            mag[..., 1] * mask,
        ], axis=-1))
        predicted = forward_model_single_rdfc_2d(
            masked_mag, _ramp_coeffs_from_array(ramp), rdfc_kernel, pixel_size,
        )
        residuals = predicted - phase
        dm = cast(u.Quantity, qnp.sum(residuals ** 2)).value
        rn = cast(u.Quantity, _norm_fn(masked_mag, reg_mask)).value
        return dm, rn

    decompose_batch = jax.jit(jax.vmap(_decompose_single))
    all_dm, all_rn = decompose_batch(all_mag, all_ramp)

    data_misfits = np.asarray(all_dm)
    reg_norms = np.asarray(all_rn)
    corner_idx, _ = kneedle_corner(data_misfits, reg_norms)

    result = LCurveResult(
        lambdas=u.Quantity(lambdas_np, "rad2"),
        data_misfits=u.Quantity(data_misfits, "rad2"),
        reg_norms=u.Quantity(reg_norms, ""),
        magnetizations=u.Quantity(all_mag, ""),
        ramp_coeffs=RampCoeffs(
            offset=u.Quantity(all_ramp[:, 0], "rad"),
            slope_y=u.Quantity(all_ramp[:, 1], "rad/nm"),
            slope_x=u.Quantity(all_ramp[:, 2], "rad/nm"),
        ),
        corner_index=corner_idx,
    )
    _assert_lcurve_result_units(result)
    return result


def plot_lcurve(
    lcurve_result,
    pyramid_style=False,
    ax=None,
    cmap="nipy_spectral",
    colorbar=True,
    **kwargs,
):
    """
    Plot an L-curve from an LCurveResult, with optional Pyramid-style axes.

    Parameters
    ----------
    lcurve_result : LCurveResult
        Result from lcurve_sweep or lcurve_sweep_vmap.
    pyramid_style : bool, optional
        If True, plot y = reg_norm / lambda vs x = data_misfit (Pyramid style).
        If False (default), plot y = data_misfit vs x = reg_norm (standard style).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.
    cmap : str, optional
        Colormap for lambda values.
    colorbar : bool, optional
        Whether to show a colorbar. Default True.
    **kwargs :
        Additional arguments passed to scatter/plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if isinstance(lcurve_result.lambdas, u.Quantity):
        lambdas = np.asarray(lcurve_result.lambdas.value)
    else:
        lambdas = np.asarray(lcurve_result.lambdas)

    if isinstance(lcurve_result.reg_norms, u.Quantity):
        reg_norms = np.asarray(lcurve_result.reg_norms.value)
    else:
        reg_norms = np.asarray(lcurve_result.reg_norms)

    if isinstance(lcurve_result.data_misfits, u.Quantity):
        data_misfits = np.asarray(lcurve_result.data_misfits.value)
    else:
        data_misfits = np.asarray(lcurve_result.data_misfits)

    if pyramid_style:
        x = data_misfits
        y = reg_norms
        xlabel = r"$\Vert\mathbf{F}(\mathbf{x})-\mathbf{y}\Vert^2$"
        ylabel = r"$\Vert\mathbf{D}\mathbf{x}\Vert^2$"
    else:
        x = reg_norms
        y = data_misfits
        xlabel = "Regularisation norm (exchange)"
        ylabel = "Data misfit"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x, y, c=lambdas, cmap=cmap, norm=LogNorm(), s=80, zorder=2, **kwargs)
    ax.plot(x, y, "k-", lw=1.5, zorder=1)
    if hasattr(lcurve_result, "corner_index") and lcurve_result.corner_index >= 0:
        ax.plot(
            x[lcurve_result.corner_index],
            y[lcurve_result.corner_index],
            "r*", ms=18, zorder=3,
            label=f"corner λ={lambdas[lcurve_result.corner_index]:.2e}",
        )
        ax.legend(frameon=False, fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title("MBIR L-curve" + (" (Pyramid style)" if pyramid_style else ""))
    ax.grid(alpha=0.25)
    if colorbar:
        plt.colorbar(sc, ax=ax, label=r"$\lambda$")
    return ax

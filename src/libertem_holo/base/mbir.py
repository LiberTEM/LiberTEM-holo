"""Model-based iterative reconstruction (MBIR) for 2D projected magnetization.

Unit conventions
----------------
Public functions in this module generally require ``unxt.Quantity`` inputs for
physical parameters and return ``Quantity``-annotated outputs, unless a
specific API explicitly documents scalar convenience inputs.

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
import warnings
from typing import Any, Literal, NamedTuple, Union, cast

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import quaxed.numpy as qnp
import unxt as u

PHI_0 = u.Quantity(2067.83, "T nm2")  # magnetic flux quantum h/(2e)
B_REF = u.Quantity(1.0, "T")  # reference magnetic induction
KERNEL_COEFF = B_REF / (2 * PHI_0)  # Quantity["1/nm2"]

MU_0 = u.Quantity(4e-7 * np.pi, "T m / A")  # vacuum permeability
ELECTRON_INTERACTION_CONSTANT_300KV = u.Quantity(6.53e6, "rad / (V m)")
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
            offset=make_quantity(jnp.zeros((), dtype=dtype), "rad"),
            slope_y=make_quantity(jnp.zeros((), dtype=dtype), "rad/nm"),
            slope_x=make_quantity(jnp.zeros((), dtype=dtype), "rad/nm"),
        )


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def make_quantity(value: Any, unit: str) -> u.Quantity:
    """Construct a Quantity from scalar or array-like input.

    This is a convenience constructor only. Semantic normalization and
    validation still belong in the specialized ``_as_*`` helpers below.
    """
    return u.Quantity(jnp.asarray(value), unit)


def add_units_to_inputs(
    *,
    phase: Any | None = None,
    mag_phase: Any | None = None,
    mip_phase: Any | None = None,
    phase_unit: str = "rad",
    pixel_size: Any | None = None,
    pixel_size_unit: str = "nm",
    thickness: Any | None = None,
    thickness_unit: str = "nm",
) -> dict[str, u.Quantity]:
    """Build a dictionary of common MBIR input quantities in one call.

    This is a user-facing convenience helper for notebook and scripting
    workflows where several scalar-or-array inputs need units attached at
    once. Structured inputs such as :class:`RampCoeffs` remain separate.

    ``phase_unit`` is shared across all phase-like inputs. Use either the
    generic ``phase`` key or the more explicit ``mag_phase`` / ``mip_phase``
    keys, but do not mix ``phase`` with the specific phase variants in the
    same call.

    Only values that are provided are included in the returned dictionary.
    Existing ``unxt.Quantity`` inputs are passed through unit conversion to
    the requested target unit.
    """
    result: dict[str, u.Quantity] = {}

    if phase is not None and (mag_phase is not None or mip_phase is not None):
        raise ValueError(
            "Use either 'phase' or the explicit 'mag_phase'/'mip_phase' inputs, not both."
        )

    for key, value, unit in (
        ("phase", phase, phase_unit),
        ("mag_phase", mag_phase, phase_unit),
        ("mip_phase", mip_phase, phase_unit),
        ("pixel_size", pixel_size, pixel_size_unit),
        ("thickness", thickness, thickness_unit),
    ):
        if value is None:
            continue
        if isinstance(value, u.Quantity):
            result[key] = cast(u.Quantity, u.uconvert(unit, value))
        else:
            result[key] = make_quantity(value, unit)

    return result

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
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be a unxt unitful Quantity, got {type(value)}")
    if not _is_unit_convertible(value, unit):
        raise ValueError(
            f"{name} must be convertible to {unit!r}, got unit {value.unit!s}"
        )


def _assert_ramp_coeffs_units(ramp_coeffs: RampCoeffs, name: str = "ramp_coeffs") -> None:
    """Assert the canonical units for RampCoeffs."""
    if not isinstance(ramp_coeffs, RampCoeffs):
        raise TypeError(f"{name} must be a RampCoeffs instance, got {type(ramp_coeffs)}")
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
        offset=make_quantity(values[0], "rad"),
        slope_y=make_quantity(values[1], "rad/nm"),
        slope_x=make_quantity(values[2], "rad/nm"),
    )


def _as_quantity(value, unit: str, name: str) -> u.Quantity:
    """Validate *value* is a Quantity convertible to *unit* and return it in *unit*.

    This is the canonical normalizer. The specialized ``_as_*_quantity``
    helpers below are thin wrappers that fix *unit* and *name* for
    commonly-used physical quantities in the MBIR module.
    """
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f"{name} must be a unxt.Quantity compatible with {unit!r}, got {type(value)}"
        )
    _assert_quantity_compatible(value, unit, name)
    return cast(u.Quantity, u.uconvert(unit, value))


def _as_length_quantity(value, name: str = "pixel_size") -> u.Quantity:
    """Normalize a length Quantity to nanometres."""
    return _as_quantity(value, "nm", name)


def _as_angle_quantity(value) -> u.Quantity:
    """Normalize an angle Quantity to radians.

    Dimensionless intermediate phase values are re-labelled as radians by
    convention after the physical nm² cancellation in the forward model.
    """
    if not isinstance(value, u.Quantity):
        raise TypeError(f"phase must be a unxt.Quantity, got {type(value)}")
    if str(value.unit) == "":
        return make_quantity(value.value, "rad")
    return _as_quantity(value, "rad", "phase")


def _as_dimensionless_quantity(value) -> u.Quantity:
    """Normalize a dimensionless Quantity to unit ''."""
    return _as_quantity(value, "", "magnetization")


def _as_induction_quantity(value, name: str = "reference_induction") -> u.Quantity:
    """Normalize a magnetic induction Quantity to tesla."""
    return _as_quantity(value, "T", name)


def _as_voltage_quantity(value, name: str = "mean_inner_potential") -> u.Quantity:
    """Normalize an electric potential Quantity to volts."""
    return _as_quantity(value, "V", name)


def _as_interaction_constant_quantity(
    value,
    name: str = "interaction_constant",
) -> u.Quantity:
    """Normalize an electron interaction constant to rad / (V m)."""
    return _as_quantity(value, "rad / (V m)", name)


def _as_physical_magnetization_quantity(
    value,
    name: str = "projected_magnetization",
) -> u.Quantity:
    """Normalize a physical magnetization Quantity to A/m."""
    return _as_quantity(value, "A / m", name)


def _as_projected_induction_integral_quantity(
    value,
    name: str = "projected_induction_integral",
) -> u.Quantity:
    """Normalize a projected induction line integral to T nm."""
    return _as_quantity(value, "T nm", name)


def _as_projected_magnetization_integral_quantity(
    value,
    name: str = "projected_magnetization_integral",
) -> u.Quantity:
    """Normalize a projected magnetization line integral to ampere."""
    return _as_quantity(value, "A", name)


def _as_ramp_coeffs(value, *, dtype=jnp.float64) -> RampCoeffs:
    """Normalize ramp coefficients to the typed Quantity container."""
    if value is None:
        result = RampCoeffs.zeros(dtype=dtype)
        _assert_ramp_coeffs_units(result)
        return result
    if not isinstance(value, RampCoeffs):
        raise TypeError(f"ramp_coeffs must be a RampCoeffs instance, got {type(value)}")
    result = RampCoeffs(
        offset=_to_rad(value.offset),
        slope_y=cast(u.Quantity, u.uconvert("rad / nm", value.slope_y)),
        slope_x=cast(u.Quantity, u.uconvert("rad / nm", value.slope_x)),
    )
    _assert_ramp_coeffs_units(result)
    return result


def _as_threshold_scalar(value, name: str = "threshold") -> float:
    """Normalize a dimensionless threshold input to a plain scalar."""
    if isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be a plain scalar without units, got {type(value)}")
    array = np.asarray(value)
    if array.ndim != 0:
        raise TypeError(f"{name} must be a scalar, got shape {array.shape}")
    return float(array)


def _to_lambda_exchange(value) -> u.Quantity:
    """Convert a scalar-or-Quantity exchange weight to ``Quantity['rad2']``."""
    if isinstance(value, u.Quantity):
        _assert_quantity_compatible(value, "rad2", "lambda_exchange")
        return cast(u.Quantity, u.uconvert("rad2", value))
    return make_quantity(value, "rad2")


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


def _validate_all_positive_quantity(value: u.Quantity, name: str) -> None:
    """Raise ValueError if any element of *value* is not positive."""
    values = np.asarray(value.value)
    if np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive everywhere, got {value}")


def _broadcast_thickness_like(projected: u.Quantity, thickness: u.Quantity) -> u.Quantity:
    """Append singleton axes so thickness maps broadcast against projected fields."""
    projected_shape = projected.shape
    thickness_shape = thickness.shape
    if len(thickness_shape) >= len(projected_shape):
        return thickness
    if projected_shape[: len(thickness_shape)] != thickness_shape:
        return thickness
    extra_axes = (1,) * (len(projected_shape) - len(thickness_shape))
    return u.Quantity(jnp.reshape(thickness.value, thickness_shape + extra_axes), str(thickness.unit))


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
    converged : bool
        Whether the CG solver converged within the iteration budget.
    """
    magnetization: u.Quantity
    ramp_coeffs: RampCoeffs
    loss_history: u.Quantity
    converged: bool


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
    threshold: float
    threshold_low: float
    threshold_high: float
    threshold_draws: np.ndarray
    magnetizations: u.Quantity  # Quantity["dimensionless"]
    mean_magnetization: u.Quantity  # Quantity["dimensionless"]
    mean_norm: u.Quantity  # Quantity["dimensionless"]
    norm_low: u.Quantity  # Quantity["dimensionless"]
    norm_high: u.Quantity  # Quantity["dimensionless"]
    norm_ci95: u.Quantity  # Quantity["dimensionless"]
    relative_ci95: u.Quantity  # Quantity["dimensionless"]
    mask_frequency: np.ndarray
    local_induction_mean_samples: u.Quantity | None  # Quantity["T"]
    local_induction_mean: u.Quantity | None  # Quantity["T"]
    local_induction_mean_low: u.Quantity | None  # Quantity["T"]
    local_induction_mean_high: u.Quantity | None  # Quantity["T"]
    local_induction_mean_ci95: u.Quantity | None  # Quantity["T"]
    local_induction_roi_pixels: np.ndarray | None


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
    preconditioner : {None, "block_jacobi"}
        Experimental inverse preconditioner for the Newton-CG linear
        solve. ``"block_jacobi"`` uses one curvature scale for the
        flattened magnetization block and individual curvature scales
        for the three ramp parameters.
    """
    cg_maxiter: int = 10000
    cg_tol: float = 1e-9
    preconditioner: Literal["block_jacobi"] | None = None


SolverConfig = Union[NewtonCGConfig]

_SOLVER_DEFAULTS = {
    "newton_cg": NewtonCGConfig,
}


def _resolve_solver_config(
    solver: str | SolverConfig,
    solver_config: SolverConfig | None = None,
) -> SolverConfig:
    """Resolve a solver string or config object (and optional override) to a SolverConfig.

    When *solver_config* is provided it takes precedence over *solver*.
    A string selects the default config for the named solver.
    """
    if solver_config is not None:
        return solver_config
    if isinstance(solver, str):
        solver_name = solver.lower()
        if solver_name not in _SOLVER_DEFAULTS:
            raise ValueError(
                f"Unknown solver {solver!r}. "
                f"Choose from {list(_SOLVER_DEFAULTS)}"
            )
        return _SOLVER_DEFAULTS[solver_name]()
    if isinstance(solver, NewtonCGConfig):
        return solver
    raise TypeError(
        f"solver must be a string or a SolverConfig instance, got {type(solver)}"
    )


_NEWTON_CG_PRECONDITIONERS = {None, "block_jacobi"}


def _validate_newton_cg_preconditioner(preconditioner: str | None) -> str | None:
    """Validate the requested experimental Newton-CG preconditioner."""
    if preconditioner not in _NEWTON_CG_PRECONDITIONERS:
        raise ValueError(
            f"Unknown Newton-CG preconditioner {preconditioner!r}. "
            f"Choose from {sorted(name for name in _NEWTON_CG_PRECONDITIONERS if name is not None)} "
            "or None."
        )
    return preconditioner


def _make_block_jacobi_preconditioner(
    x0_flat: jax.Array,
    matvec_hvp,
    magnetization_size: int,
):
    """Build a cheap block-diagonal inverse preconditioner in flat space."""
    dtype = x0_flat.dtype
    eps = jnp.sqrt(jnp.finfo(dtype).eps)
    total_size = x0_flat.size
    ramp_size = total_size - magnetization_size

    if ramp_size != 3:
        raise ValueError(
            "Newton-CG block Jacobi preconditioner expects three ramp coefficients, "
            f"got {ramp_size}."
        )

    mag_probe = jnp.concatenate([
        jnp.ones((magnetization_size,), dtype=dtype),
        jnp.zeros((ramp_size,), dtype=dtype),
    ])
    mag_curvature = jnp.vdot(mag_probe, matvec_hvp(mag_probe)).real / magnetization_size

    ramp_curvatures = []
    for ramp_index in range(ramp_size):
        basis = jnp.zeros_like(x0_flat)
        basis = basis.at[magnetization_size + ramp_index].set(1)
        ramp_curvatures.append(matvec_hvp(basis)[magnetization_size + ramp_index].real)

    diag = jnp.concatenate([
        jnp.full((magnetization_size,), mag_curvature, dtype=dtype),
        jnp.asarray(ramp_curvatures, dtype=dtype),
    ])
    diag = jnp.where(jnp.isfinite(diag) & (diag > eps), diag, jnp.ones_like(diag))

    def inverse_preconditioner(v):
        return v / diag

    return inverse_preconditioner


def _make_newton_cg_preconditioner(
    preconditioner: str | None,
    x0_flat: jax.Array,
    matvec_hvp,
    magnetization_size: int,
):
    """Create an experimental inverse preconditioner for JAX CG."""
    preconditioner = _validate_newton_cg_preconditioner(preconditioner)
    if preconditioner is None:
        return None
    if preconditioner == "block_jacobi":
        return _make_block_jacobi_preconditioner(
            x0_flat,
            matvec_hvp,
            magnetization_size,
        )
    raise AssertionError(f"Unhandled Newton-CG preconditioner: {preconditioner}")


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
    jax.Array
        Magnetic phase-shift image of shape ``(H, W)``.
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


def mbir_loss_2d(
    params: tuple[u.Quantity, RampCoeffs],
    mask: jax.Array,
    phase: u.Quantity,
    rdfc_kernel: dict[str, Any],
    pixel_size: u.Quantity,
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

    lambda_exchange = _to_lambda_exchange(
        reg_config.get("lambda_exchange", 0.0)
    )

    loss += lambda_exchange * exchange_loss_fn(magnetization_q, reg_mask)
    _assert_quantity_compatible(loss, "rad2", "loss")

    return loss


def _run_newton_cg_solver_2d(
    phase: u.Quantity,
    init_mag: u.Quantity,
    mask: jax.Array,
    pixel_size: u.Quantity,
    reg_config: dict[str, Any] | None = None,
    rdfc_kernel: dict[str, Any] | None = None,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 10000,
    preconditioner: str | None = None,
    init_ramp_coeffs: RampCoeffs | None = None,
    reg_mask: jax.Array | None = None,
) -> tuple[tuple[u.Quantity, RampCoeffs], u.Quantity, bool]:
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
    preconditioner
        Experimental inverse preconditioner for the CG solve.
    init_ramp_coeffs
        Initial ramp coefficients of shape ``(3,)``.  Defaults to
        zeros.
    reg_mask
        Optional regularization mask of shape ``(N, M)``.
        Defaults to *mask*.

    Returns
    -------
    (magnetization, ramp_coeffs) : tuple[Quantity[""], RampCoeffs]
        Optimized magnetization ``(N, M, 2)`` and typed ramp coefficients.
    loss_history : Quantity["rad2"]
        Length-1 array containing the loss after the Newton update.
    converged : bool
        Whether the CG solver converged within the iteration budget.
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

    preconditioner_fn = _make_newton_cg_preconditioner(
        preconditioner,
        x0_flat,
        matvec_hvp,
        init_mag_q.value.size,
    )

    delta, cg_info = jax.scipy.sparse.linalg.cg(
        matvec_hvp,
        -grad_at_x0,
        tol=cg_tol,
        maxiter=cg_maxiter,
        M=preconditioner_fn,
    )
    # JAX's cg returns None for info; compute convergence from the residual.
    # The system is: H @ delta = b where b = -grad_at_x0.
    b = -grad_at_x0
    residual = matvec_hvp(delta) - b
    b_norm = jnp.linalg.norm(b)
    residual_norm = jnp.linalg.norm(residual)
    # CG convention: converged when ||r|| / ||b|| < tol (or b == 0).
    converged = (b_norm == 0) | (residual_norm / (b_norm + 1e-30) < cg_tol)
    final_flat = x0_flat + delta
    history = make_quantity(jnp.expand_dims(objective_flat(final_flat), axis=0), "rad2")

    final_mag, final_ramp = unravel(final_flat)

    return (final_mag, final_ramp), history, converged


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
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units.
    solver : str or SolverConfig, optional
        Solver selection. Pass ``"newton_cg"`` or a
        :class:`NewtonCGConfig` for full control. Default is ``"newton_cg"``.
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
        config = _resolve_solver_config(solver)
    elif isinstance(solver, NewtonCGConfig):
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
        (mag, ramp), loss_history, converged_jax = _run_newton_cg_solver_2d(
            **shared,
            cg_tol=config.cg_tol,
            cg_maxiter=config.cg_maxiter,
            preconditioner=config.preconditioner,
        )
        converged = bool(converged_jax)
    else:
        raise AssertionError(f"Unhandled solver config type: {type(config)}")

    result = SolverResult(
        magnetization=mag,
        ramp_coeffs=cast(RampCoeffs, ramp),
        loss_history=loss_history,
        converged=converged,
    )
    _assert_solver_result_units(result)
    if not result.converged:
        warnings.warn(
            f"CG solver did not converge within {config.cg_maxiter} iterations "
            f"(tol={config.cg_tol}). Consider increasing cg_maxiter or "
            f"relaxing cg_tol in NewtonCGConfig.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def reconstruct_2d(
    phase,
    pixel_size,
    mask=None,
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
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
    mask : array_like, optional
        Binary mask of shape ``(N, M)``.  Defaults to all ones.
    lam : Quantity["rad2"], optional
        Regularization weight (``lambda_exchange``), default ``Quantity(1e-3, "rad2")``.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``) or a :class:`SolverConfig` instance.
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
    lam = _to_lambda_exchange(lam)
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

    init_mag = make_quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")
    reg_config = {"lambda_exchange": lam}

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


def to_projected_induction_integral(
    magnetization,
    pixel_size: u.Quantity,
    reference_induction: u.Quantity = B_REF,
) -> u.Quantity:
    """Convert normalized projected magnetization to a projected induction integral.

    Parameters
    ----------
    magnetization : Quantity["dimensionless"]
        Normalized projected line-sum magnetization, typically the
        ``result.magnetization`` output from :func:`reconstruct_2d`.
    pixel_size : Quantity["length"]
        Pixel size along the beam direction, equal to the slice thickness of
        the implicit projection discretization.
    reference_induction : Quantity["magnetic induction"], optional
        Induction scale corresponding to unit normalized magnetization.
        The default is the 1 T reference baked into the MBIR kernel.

    Returns
    -------
    Quantity["magnetic induction * length"]
        Projected induction line integral with the same shape as
        ``magnetization`` in T nm.
    """
    magnetization = _as_dimensionless_quantity(magnetization)
    pixel_size = _as_length_quantity(pixel_size)
    reference_induction = _as_induction_quantity(reference_induction)
    result = pixel_size * reference_induction * magnetization
    _assert_quantity_compatible(result, "T nm", "projected_induction_integral")
    return cast(u.Quantity, u.uconvert("T nm", result))


def to_projected_magnetization_integral(
    magnetization,
    pixel_size: u.Quantity,
    reference_induction: u.Quantity = B_REF,
) -> u.Quantity:
    r"""Convert normalized projected magnetization to a projected magnetization integral.

    This applies the induction scale from
    :func:`to_projected_induction_integral` and then uses
    :math:`M = B / \mu_0` to express the projected line integral in amperes.

    Parameters
    ----------
    magnetization : Quantity["dimensionless"]
        Normalized projected line-sum magnetization, typically the
        ``result.magnetization`` output from :func:`reconstruct_2d`.
    pixel_size : Quantity["length"]
        Pixel size along the beam direction, equal to the slice thickness of
        the implicit projection discretization.
    reference_induction : Quantity["magnetic induction"], optional
        Induction scale corresponding to unit normalized magnetization.
        Pass the appropriate physical induction scale if it differs from
        the 1 T kernel reference.

    Returns
    -------
    Quantity["current"]
        Projected magnetization line integral in A.
    """
    induction = to_projected_induction_integral(
        magnetization,
        pixel_size,
        reference_induction=reference_induction,
    )
    result = induction / MU_0
    _assert_quantity_compatible(result, "A", "projected_magnetization_integral")
    return cast(u.Quantity, u.uconvert("A", result))


def estimate_thickness_from_mip_phase(
    mip_phase,
    mean_inner_potential: u.Quantity,
    interaction_constant: u.Quantity = ELECTRON_INTERACTION_CONSTANT_300KV,
    *,
    use_abs: bool = True,
) -> u.Quantity:
    r"""Estimate a thickness map from a mean-inner-potential phase image.

    Uses the standard relation

    .. math::

        \phi_E = C_E V_0 t

    so that the thickness is

    .. math::

        t = \frac{\phi_E}{C_E V_0}.

    Parameters
    ----------
    mip_phase : Quantity["angle"]
        Mean-inner-potential phase image.
    mean_inner_potential : Quantity["voltage"]
        Mean inner potential :math:`V_0` of the material.
    interaction_constant : Quantity["angle / (voltage * length)"], optional
        Electron interaction constant :math:`C_E`. Defaults to the
        300 kV value ``6.53e6 rad / (V m)``.
    use_abs : bool, optional
        If ``True`` (default), use ``abs(mip_phase)`` when estimating the
        thickness map.

    Returns
    -------
    Quantity["length"]
        Estimated thickness map in nanometres.
    """
    mip_phase_q = _as_angle_quantity(mip_phase)
    mean_inner_potential_q = _as_voltage_quantity(mean_inner_potential)
    interaction_constant_q = _as_interaction_constant_quantity(interaction_constant)
    _validate_positive(mean_inner_potential_q, "mean_inner_potential")
    _validate_positive(interaction_constant_q, "interaction_constant")

    phase_for_estimate = mip_phase_q
    if use_abs:
        phase_for_estimate = make_quantity(jnp.abs(mip_phase_q.value), str(mip_phase_q.unit))

    result = cast(
        u.Quantity,
        phase_for_estimate / (interaction_constant_q * mean_inner_potential_q),
    )
    result = cast(u.Quantity, u.uconvert("nm", result))

    _assert_quantity_compatible(result, "nm", "thickness")
    return result


def to_local_induction(
    projected_induction_integral,
    thickness,
    *,
    min_effective_thickness: u.Quantity | None = None,
    invalid_to_nan: bool = False,
) -> u.Quantity:
    r"""Convert a projected induction line integral to local induction.

    This divides a projected induction line integral by a physical thickness,
    producing a thickness-averaged local induction in tesla.

    Parameters
    ----------
    projected_induction_integral : Quantity["magnetic induction * length"]
        Projected induction line integral, typically returned by
        :func:`to_projected_induction_integral`.
    thickness : Quantity["length"]
        Physical thickness in length units. May be a scalar or an image that
        broadcasts against *projected_induction_integral*.
    min_effective_thickness : Quantity["length"], optional
        Positive minimum thickness used to stabilise the division. When
        provided, pixels below this thickness are either clamped for the
        division or marked invalid, depending on ``invalid_to_nan``.
    invalid_to_nan : bool, optional
        If ``True`` and ``min_effective_thickness`` is provided, pixels with
        thickness below the threshold are returned as ``NaN`` instead of being
        clamped.

    Returns
    -------
    Quantity["magnetic induction"]
        Thickness-averaged local induction in T.
    """
    projected_induction_integral = _as_projected_induction_integral_quantity(
        projected_induction_integral,
        name="projected_induction_integral",
    )
    thickness = _as_length_quantity(thickness, name="thickness")
    thickness = _broadcast_thickness_like(projected_induction_integral, thickness)
    divisor = thickness
    invalid_mask = None
    if min_effective_thickness is None:
        _validate_all_positive_quantity(thickness, "thickness")
    else:
        min_effective_thickness_q = _as_length_quantity(
            min_effective_thickness,
            name="min_effective_thickness",
        )
        _validate_positive(min_effective_thickness_q, "min_effective_thickness")
        invalid_mask = np.asarray(thickness.value) < float(np.asarray(min_effective_thickness_q.value))
        divisor = make_quantity(
            jnp.maximum(thickness.value, min_effective_thickness_q.value),
            str(thickness.unit),
        )

    result = cast(u.Quantity, projected_induction_integral / divisor)
    result = cast(u.Quantity, u.uconvert("T", result))
    if invalid_mask is not None and invalid_to_nan:
        result = make_quantity(
            jnp.where(invalid_mask, jnp.nan, result.value),
            "T",
        )
    _assert_quantity_compatible(result, "T", "local_induction")
    return result


def to_local_magnetization(
    projected_magnetization_integral,
    thickness,
    *,
    min_effective_thickness: u.Quantity | None = None,
    invalid_to_nan: bool = False,
) -> u.Quantity:
    r"""Convert a projected magnetization line integral to local magnetization.

    This divides a projected magnetization line integral by a physical
    thickness, producing a thickness-averaged local magnetization in A/m.

    Parameters
    ----------
    projected_magnetization_integral : Quantity["current"]
        Projected magnetization line integral, typically returned by
        :func:`to_projected_magnetization_integral`.
    thickness : Quantity["length"]
        Physical thickness in length units. May be a scalar or an image that
        broadcasts against *projected_magnetization_integral*.
    min_effective_thickness : Quantity["length"], optional
        Positive minimum thickness used to stabilise the division. When
        provided, pixels below this thickness are either clamped for the
        division or marked invalid, depending on ``invalid_to_nan``.
    invalid_to_nan : bool, optional
        If ``True`` and ``min_effective_thickness`` is provided, pixels with
        thickness below the threshold are returned as ``NaN`` instead of being
        clamped.

    Returns
    -------
    Quantity["magnetization"]
        Thickness-averaged local magnetization in A/m.
    """
    projected_magnetization_integral = _as_projected_magnetization_integral_quantity(
        projected_magnetization_integral,
        name="projected_magnetization_integral",
    )
    thickness = _as_length_quantity(thickness, name="thickness")
    thickness = _broadcast_thickness_like(projected_magnetization_integral, thickness)
    divisor = thickness
    invalid_mask = None
    if min_effective_thickness is None:
        _validate_all_positive_quantity(thickness, "thickness")
    else:
        min_effective_thickness_q = _as_length_quantity(
            min_effective_thickness,
            name="min_effective_thickness",
        )
        _validate_positive(min_effective_thickness_q, "min_effective_thickness")
        invalid_mask = np.asarray(thickness.value) < float(np.asarray(min_effective_thickness_q.value))
        divisor = make_quantity(
            jnp.maximum(thickness.value, min_effective_thickness_q.value),
            str(thickness.unit),
        )

    result = cast(u.Quantity, projected_magnetization_integral / divisor)
    result = cast(u.Quantity, u.uconvert("A / m", result))
    if invalid_mask is not None and invalid_to_nan:
        result = make_quantity(
            jnp.where(invalid_mask, jnp.nan, result.value),
            "A / m",
        )
    _assert_quantity_compatible(result, "A / m", "local_magnetization")
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
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
    lam : float, optional
        Regularization weight (``lambda_exchange``), default 1e-3.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``) or a
        :class:`NewtonCGConfig` instance.
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
    config = _resolve_solver_config(solver, solver_config)

    init_mag = make_quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")
    reg_config = {"lambda_exchange": _to_lambda_exchange(lam)}

    # Build a vmappable function for the chosen solver
    def _solve_single_newton(mask, reg_mask):
        (mag, _ramp), _loss, converged = _run_newton_cg_solver_2d(
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
        return cast(u.Quantity, mag).value, converged

    solve_single = _solve_single_newton

    solve_batch = jax.jit(jax.vmap(solve_single, in_axes=(0, 0)))
    all_mag, all_converged = solve_batch(masks, reg_masks)

    converged_arr = np.asarray(all_converged)
    if not np.all(converged_arr):
        n_failed = int(np.sum(~converged_arr))
        warnings.warn(
            f"CG solver did not converge for {n_failed} of {len(converged_arr)} "
            f"ensemble member(s). Consider increasing cg_maxiter or "
            f"relaxing cg_tol in NewtonCGConfig.",
            RuntimeWarning,
            stacklevel=2,
        )

    return make_quantity(all_mag, "")


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


def decompose_loss(
    magnetization,
    ramp_coeffs,
    phase,
    mask,
    reg_mask,
    rdfc_kernel,
    pixel_size,
) -> tuple[u.Quantity, u.Quantity]:
    """Decompose the MBIR loss into data-fidelity and regularization terms.

    Evaluates the two components of the loss **without** the
    ``lambda_exchange`` multiplier on the regularization term, so
    that they can be compared on an L-curve plot.

    Parameters
    ----------
    magnetization : Quantity["dimensionless"]
        Reconstructed magnetization of shape ``(N, M, 2)``.
    ramp_coeffs : RampCoeffs
        Background ramp coefficients ``[offset, slope_y, slope_x]``.
    phase : Quantity["angle"]
        Observed phase image of shape ``(N, M)``.
    mask : array_like
        Binary mask of shape ``(N, M)`` applied to the
        magnetization before the forward model.
    reg_mask : array_like
        Regularization mask of shape ``(N, M)`` passed to
        :func:`exchange_loss_fn`.
    rdfc_kernel : dict
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
    pixel_size : Quantity["length"]
        Pixel size converted to nanometres internally.

    Returns
    -------
    data_misfit : Quantity["rad2"]
        ``sum((predicted - observed)**2)`` — the squared-residual
        norm, matching Pyramid's ``chisq_m`` convention (no 1/2
        factor).
    exchange_norm : Quantity["dimensionless"]
        Unweighted exchange regularization norm (no lambda
        multiplier) computed with :func:`exchange_loss_fn`.
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
    thickness: u.Quantity | None = None,
    reference_induction: u.Quantity = B_REF,
    min_effective_thickness: u.Quantity | None = None,
    invalid_to_nan: bool = False,
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
    threshold : float
        Central threshold value around which the bootstrap draws are sampled.
        This is a plain scalar applied to ``abs(mip_phase.value)`` after the
        MIP phase has been converted to radians.
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units.
    lam : float, optional
        Regularization weight (``lambda_exchange``), default 1e-3.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``) or a
        :class:`NewtonCGConfig` instance. Ignored when
        *solver_config* is provided. Default ``"newton_cg"``.
    n_boot : int, optional
        Number of threshold draws, default 50.
    threshold_low : float, optional
        Lower bound for the threshold draws. Defaults to ``threshold - 0.25``.
    threshold_high : float, optional
        Upper bound for the threshold draws. Defaults to ``threshold + 0.25``.
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
    thickness : Quantity["length"], optional
        Physical thickness map or scalar thickness used to convert the
        reconstructed projected induction to local induction. When provided,
        the result also includes per-draw mean ``|B_local|`` values inside
        each draw's bootstrap mask and their 95% bootstrap interval.
    reference_induction : Quantity["magnetic induction"], optional
        Physical induction scale corresponding to unit normalized
        magnetization. Default is the module-level 1 T reference.
    min_effective_thickness : Quantity["length"], optional
        Positive lower bound passed to :func:`to_local_induction` to stabilize
        the local-induction conversion.
    invalid_to_nan : bool, optional
        Forwarded to :func:`to_local_induction`. When ``True``, pixels below
        ``min_effective_thickness`` are excluded from the per-draw mean.

    Returns
    -------
    BootstrapThresholdResult
        Summary object containing the threshold draws, reconstructed
        magnetizations, 2.5th and 97.5th percentile maps, their 95% width,
        the relative 95% width, the mask inclusion frequency, and optionally
        the mean local-induction magnitude inside each draw's segmented
        object plus its 95% bootstrap interval.
    """
    phase = _as_angle_quantity(phase)
    mip_phase = _as_angle_quantity(mip_phase)
    pixel_size = _as_length_quantity(pixel_size)
    threshold_value = _as_threshold_scalar(threshold, "threshold")
    _validate_positive(pixel_size, "pixel_size")
    if phase.shape != mip_phase.shape:
        raise ValueError(
            f"phase and mip_phase must have the same shape; got {phase.shape} and {mip_phase.shape}."
        )

    if threshold_low is None:
        threshold_low_value = threshold_value - 0.25
    else:
        threshold_low_value = _as_threshold_scalar(threshold_low, "threshold_low")
    if threshold_high is None:
        threshold_high_value = threshold_value + 0.25
    else:
        threshold_high_value = _as_threshold_scalar(threshold_high, "threshold_high")

    if threshold_high_value <= threshold_low_value:
        raise ValueError(
            "threshold_high must be greater than threshold_low; "
            f"got {threshold_high_value} and {threshold_low_value}."
        )

    rng = np.random.default_rng(rng_seed)
    threshold_draws = rng.uniform(
        low=threshold_low_value,
        high=threshold_high_value,
        size=n_boot,
    )

    mip_phase_q = cast(u.Quantity, u.uconvert("rad", mip_phase))
    mip_abs = np.abs(np.asarray(mip_phase_q.value))
    reference_mask = (mip_abs > threshold_value).astype(np.float64)
    bootstrap_masks = (
        mip_abs[None, ...] > threshold_draws[:, None, None]
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
    norm_low = make_quantity(
        np.percentile(np.asarray(norm_samples.value), 2.5, axis=0),
        str(norm_samples.unit),
    )
    norm_high = make_quantity(
        np.percentile(np.asarray(norm_samples.value), 97.5, axis=0),
        str(norm_samples.unit),
    )
    norm_ci95 = norm_high - norm_low
    relative_ci95 = _as_dimensionless_quantity(
        norm_ci95 / (mean_norm + make_quantity(1e-12, ""))
    )
    mask_frequency = bootstrap_masks.mean(axis=0)

    local_induction_mean_samples = None
    local_induction_mean = None
    local_induction_mean_low = None
    local_induction_mean_high = None
    local_induction_mean_ci95 = None
    local_induction_roi_pixels = None

    if thickness is not None:
        thickness_q = _as_length_quantity(thickness, name="thickness")
        reference_induction_q = _as_induction_quantity(reference_induction)
        draw_means = np.empty(n_boot, dtype=np.float64)
        draw_pixels = np.zeros(n_boot, dtype=np.int64)
        bootstrap_mag_values = np.asarray(bootstrap_mag.value)
        bootstrap_mag_unit = str(bootstrap_mag.unit)

        for draw_index in range(n_boot):
            mag_draw = make_quantity(bootstrap_mag_values[draw_index], bootstrap_mag_unit)
            projected_draw = to_projected_induction_integral(
                mag_draw,
                pixel_size,
                reference_induction=reference_induction_q,
            )
            local_draw = to_local_induction(
                projected_draw,
                thickness_q,
                min_effective_thickness=min_effective_thickness,
                invalid_to_nan=invalid_to_nan,
            )
            local_draw_norm = np.linalg.norm(np.asarray(local_draw.value), axis=-1)
            roi_values = local_draw_norm[np.asarray(bootstrap_masks[draw_index], dtype=bool)]
            finite_roi_values = roi_values[np.isfinite(roi_values)]
            draw_pixels[draw_index] = finite_roi_values.size
            draw_means[draw_index] = (
                float(finite_roi_values.mean()) if finite_roi_values.size > 0 else np.nan
            )

        local_induction_mean_samples = make_quantity(draw_means, "T")
        local_induction_mean = make_quantity(np.nanmean(draw_means), "T")
        local_induction_mean_low = make_quantity(np.nanpercentile(draw_means, 2.5), "T")
        local_induction_mean_high = make_quantity(np.nanpercentile(draw_means, 97.5), "T")
        local_induction_mean_ci95 = local_induction_mean_high - local_induction_mean_low
        local_induction_roi_pixels = draw_pixels

    return BootstrapThresholdResult(
        threshold=threshold_value,
        threshold_low=threshold_low_value,
        threshold_high=threshold_high_value,
        threshold_draws=threshold_draws,
        magnetizations=bootstrap_mag,
        mean_magnetization=mean_magnetization,
        mean_norm=mean_norm,
        norm_low=norm_low,
        norm_high=norm_high,
        norm_ci95=norm_ci95,
        relative_ci95=relative_ci95,
        mask_frequency=mask_frequency,
        local_induction_mean_samples=local_induction_mean_samples,
        local_induction_mean=local_induction_mean,
        local_induction_mean_low=local_induction_mean_low,
        local_induction_mean_high=local_induction_mean_high,
        local_induction_mean_ci95=local_induction_mean_ci95,
        local_induction_roi_pixels=local_induction_roi_pixels,
    )


def plot_bootstrap_mask_summary(
    result: BootstrapThresholdResult,
    *,
    ax=None,
    stable_frequency: float = 0.5,
    cmap: str = "gray",
):
    """Plot the aggregate bootstrap support mask frequency.

    This is a compact view of how often each pixel was included in the
    threshold-defined support across all bootstrap draws. The red contour
    marks the default ``stable_frequency`` boundary.

    Parameters
    ----------
    result
        Bootstrap summary returned by :func:`bootstrap_threshold_uncertainty_2d`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. When omitted, a new figure and axes are created.
    stable_frequency : float, optional
        Contour level used to mark the stable-support boundary. Must lie in
        the closed interval ``[0, 1]``. Default is ``0.5``.
    cmap : str, optional
        Matplotlib colormap for the inclusion-frequency image.

    Returns
    -------
    fig, ax, info
        Figure, axes, and a dictionary with the rendered mask-frequency array,
        stable-support mask, draw count, and threshold metadata.
    """
    import matplotlib.pyplot as plt

    if not 0.0 <= stable_frequency <= 1.0:
        raise ValueError(
            f"stable_frequency must lie in [0, 1], got {stable_frequency}."
        )

    mask_frequency = np.asarray(result.mask_frequency)
    if mask_frequency.ndim != 2:
        raise ValueError(
            "result.mask_frequency must be a 2D image; "
            f"got shape {mask_frequency.shape}."
        )

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    else:
        fig = ax.figure

    image = ax.imshow(mask_frequency, cmap=cmap, origin="lower", vmin=0, vmax=1)
    stable_support = mask_frequency >= stable_frequency
    if np.any(mask_frequency > 0) and np.any(mask_frequency < 1):
        ax.contour(
            mask_frequency,
            levels=[stable_frequency],
            colors="tab:red",
            linewidths=1.0,
        )

    n_draws = len(result.threshold_draws)
    ax.set_title(
        "Bootstrap mask inclusion frequency\n"
        f"{n_draws} draws in [{result.threshold_low:.3f}, {result.threshold_high:.3f}]",
        fontsize=10,
    )
    ax.set_xlabel(
        f"Stable support at frequency >= {stable_frequency:.2f}: "
        f"{int(np.count_nonzero(stable_support))} pixels"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, label="Inclusion frequency")

    info = {
        "mask_frequency": mask_frequency,
        "stable_support": stable_support,
        "stable_frequency": stable_frequency,
        "n_draws": n_draws,
        "threshold": result.threshold,
        "threshold_low": result.threshold_low,
        "threshold_high": result.threshold_high,
    }
    return fig, ax, info


def plot_physical_bootstrap_uncertainty(
    result: BootstrapThresholdResult,
    magnetization,
    pixel_size,
    thickness,
    *,
    support_mask=None,
    min_effective_thickness: u.Quantity | None = None,
    invalid_to_nan: bool = False,
    max_relative_ci_for_display: float = 1.0,
    min_mask_frequency_for_display: float = 0.5,
):
    """Plot physical-unit maps weighted by bootstrap stability.

    The reconstruction is converted to projected and local physical units.
    Bootstrap uncertainty is then used only as a display weight,

    ``alpha = clip(1 - relative_ci / max_relative_ci_for_display, 0, 1) * mask_frequency``

    so pixels with wide intervals or unstable threshold support fade out.

    Parameters
    ----------
    result
        Bootstrap summary returned by :func:`bootstrap_threshold_uncertainty_2d`.
    magnetization
        Final reconstructed magnetization field of shape ``(H, W, 2)``.
    pixel_size : Quantity["length"]
        Pixel size used for the physical conversion.
    thickness : Quantity["length"]
        Thickness map or scalar thickness for the local-unit conversion.
    support_mask : array_like, optional
        Boolean mask that defines the displayed reconstruction support.
        Defaults to all pixels.
    min_effective_thickness : Quantity["length"], optional
        Lower bound passed to :func:`to_local_induction` and
        :func:`to_local_magnetization`.
    invalid_to_nan : bool, optional
        Forwarded to the local conversion helpers.
    max_relative_ci_for_display : float, optional
        Relative 95% CI width where display certainty reaches zero.
    min_mask_frequency_for_display : float, optional
        Minimum threshold-inclusion frequency for the ``reliable_support``
        summary mask returned in the info dictionary.

    Returns
    -------
    fig, axs, info
        Matplotlib figure and axes plus a dictionary containing the derived
        physical quantities, display masks, and summary scalars.
    """
    import matplotlib.pyplot as plt

    def as_array(value):
        if isinstance(value, u.Quantity):
            return np.asarray(value.value)
        return np.asarray(value)

    magnetization = _as_dimensionless_quantity(magnetization)
    pixel_size = _as_length_quantity(pixel_size)
    thickness = _as_length_quantity(thickness, name="thickness")

    image_shape = magnetization.shape[:-1]
    if support_mask is None:
        support_mask_arr = np.ones(image_shape, dtype=bool)
    else:
        support_mask_arr = np.asarray(support_mask, dtype=bool)
        if support_mask_arr.shape != image_shape:
            raise ValueError(
                "support_mask must match the spatial shape of magnetization; "
                f"got {support_mask_arr.shape} and {image_shape}."
            )

    relative_ci_map = as_array(result.relative_ci95)
    mask_frequency = as_array(result.mask_frequency)
    if relative_ci_map.shape != image_shape:
        raise ValueError(
            "result.relative_ci95 must match the spatial shape of magnetization; "
            f"got {relative_ci_map.shape} and {image_shape}."
        )
    if mask_frequency.shape != image_shape:
        raise ValueError(
            "result.mask_frequency must match the spatial shape of magnetization; "
            f"got {mask_frequency.shape} and {image_shape}."
        )

    projected_induction_integral = to_projected_induction_integral(
        magnetization,
        pixel_size,
    )
    projected_magnetization_integral = to_projected_magnetization_integral(
        magnetization,
        pixel_size,
    )
    local_induction = to_local_induction(
        projected_induction_integral,
        thickness,
        min_effective_thickness=min_effective_thickness,
        invalid_to_nan=invalid_to_nan,
    )
    local_magnetization = to_local_magnetization(
        projected_magnetization_integral,
        thickness,
        min_effective_thickness=min_effective_thickness,
        invalid_to_nan=invalid_to_nan,
    )

    projected_induction_integral_norm = np.linalg.norm(
        as_array(projected_induction_integral),
        axis=-1,
    )
    projected_magnetization_integral_norm = np.linalg.norm(
        as_array(projected_magnetization_integral),
        axis=-1,
    )
    local_induction_norm = np.linalg.norm(as_array(local_induction), axis=-1)
    local_magnetization_norm = np.linalg.norm(as_array(local_magnetization), axis=-1)

    masked_projected_induction = np.where(
        support_mask_arr,
        projected_induction_integral_norm,
        np.nan,
    )
    masked_projected_magnetization = np.where(
        support_mask_arr,
        projected_magnetization_integral_norm,
        np.nan,
    )

    certainty_alpha = np.clip(
        1.0 - relative_ci_map / max_relative_ci_for_display,
        0.0,
        1.0,
    )
    certainty_alpha *= mask_frequency
    certainty_alpha = np.where(support_mask_arr, certainty_alpha, 0.0)

    display_local_induction = np.where(
        support_mask_arr & np.isfinite(local_induction_norm),
        local_induction_norm * certainty_alpha,
        np.nan,
    )
    display_local_magnetization = np.where(
        support_mask_arr & np.isfinite(local_magnetization_norm),
        local_magnetization_norm * certainty_alpha,
        np.nan,
    )

    reliable_support = (
        support_mask_arr
        & np.isfinite(local_induction_norm)
        & np.isfinite(local_magnetization_norm)
        & (relative_ci_map <= max_relative_ci_for_display)
        & (mask_frequency >= min_mask_frequency_for_display)
    )
    masked_relative_ci_percent = np.where(
        support_mask_arr,
        100.0 * relative_ci_map,
        np.nan,
    )
    masked_certainty = np.where(support_mask_arr, certainty_alpha, np.nan)

    weighted_support = (
        support_mask_arr
        & np.isfinite(local_induction_norm)
        & np.isfinite(local_magnetization_norm)
    )
    certainty_weights = np.where(weighted_support, certainty_alpha, 0.0)
    weight_sum = float(np.sum(certainty_weights))
    if weight_sum > 0:
        normalized_weights = certainty_weights / weight_sum
        certainty_weighted_mean_local_induction = float(
            np.sum(
                normalized_weights
                * np.where(weighted_support, local_induction_norm, 0.0)
            )
        )
        certainty_weighted_mean_local_magnetization = float(
            np.sum(
                normalized_weights
                * np.where(weighted_support, local_magnetization_norm, 0.0)
            )
        )
        certainty_weighted_mean_thickness = float(
            np.sum(
                normalized_weights
                * np.where(weighted_support, as_array(thickness), 0.0)
            )
        )
    else:
        normalized_weights = np.zeros_like(certainty_weights)
        certainty_weighted_mean_local_induction = np.nan
        certainty_weighted_mean_local_magnetization = np.nan
        certainty_weighted_mean_thickness = np.nan

    mean_certainty_inside_support = (
        float(np.mean(certainty_alpha[support_mask_arr]))
        if np.any(support_mask_arr)
        else np.nan
    )

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    im = axs[0, 0].imshow(masked_projected_induction, cmap="magma", origin="lower")
    axs[0, 0].set_title(rf"Projected $|B|$ integral ({projected_induction_integral.unit})")
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046, label=projected_induction_integral.unit)

    im = axs[0, 1].imshow(masked_projected_magnetization, cmap="cividis", origin="lower")
    axs[0, 1].set_title(rf"Projected $|M|$ integral ({projected_magnetization_integral.unit})")
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046, label=projected_magnetization_integral.unit)

    im = axs[0, 2].imshow(masked_relative_ci_percent, cmap="magma_r", origin="lower")
    axs[0, 2].set_title("Relative 95% CI width of |M| (%)")
    plt.colorbar(im, ax=axs[0, 2], fraction=0.046, label="%")

    im = axs[1, 0].imshow(display_local_induction, cmap="magma", origin="lower")
    axs[1, 0].set_title(rf"Certainty-weighted local $|B|$ ({local_induction.unit})")
    plt.colorbar(im, ax=axs[1, 0], fraction=0.046, label=local_induction.unit)

    im = axs[1, 1].imshow(display_local_magnetization, cmap="cividis", origin="lower")
    axs[1, 1].set_title(rf"Certainty-weighted local $|M|$ ({local_magnetization.unit})")
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046, label=local_magnetization.unit)

    im = axs[1, 2].imshow(masked_certainty, cmap="gray", origin="lower", vmin=0, vmax=1)
    axs[1, 2].set_title("Display certainty / alpha")
    plt.colorbar(im, ax=axs[1, 2], fraction=0.046, label="alpha")

    fig.suptitle(
        "Physical-unit maps with bootstrap uncertainty used to suppress unstable edge pixels",
        fontsize=13,
    )

    info = {
        "projected_induction_integral": projected_induction_integral,
        "projected_magnetization_integral": projected_magnetization_integral,
        "local_induction": local_induction,
        "local_magnetization": local_magnetization,
        "masked_projected_induction": masked_projected_induction,
        "masked_projected_magnetization": masked_projected_magnetization,
        "masked_relative_ci_percent": masked_relative_ci_percent,
        "masked_certainty": masked_certainty,
        "certainty_alpha": certainty_alpha,
        "reliable_support": reliable_support,
        "normalized_weights": normalized_weights,
        "certainty_weighted_mean_local_induction": certainty_weighted_mean_local_induction,
        "certainty_weighted_mean_local_magnetization": certainty_weighted_mean_local_magnetization,
        "certainty_weighted_mean_thickness": certainty_weighted_mean_thickness,
        "mean_certainty_inside_support": mean_certainty_inside_support,
        "equivalent_fully_certain_pixels": weight_sum,
        "max_relative_ci_for_display": max_relative_ci_for_display,
        "min_mask_frequency_for_display": min_mask_frequency_for_display,
    }
    return fig, axs, info


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
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
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
    Returns
    -------
    LCurveResult
        Named tuple with ``lambdas``, ``data_misfits``,
        ``reg_norms``, ``magnetizations``, ``ramp_coeffs``, and
        ``corner_index``.
    """
    phase = _as_angle_quantity(phase)
    pixel_size = _as_length_quantity(pixel_size)
    lambdas_q = _to_lambda_exchange(lambdas)
    lambdas = np.atleast_1d(np.asarray(lambdas_q.value, dtype=np.float64))

    _validate_positive(pixel_size, "pixel_size")

    mask = jnp.asarray(mask, dtype=bool)
    if reg_mask is None:
        reg_mask = mask
    else:
        reg_mask = jnp.asarray(reg_mask)

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

    init_mag = make_quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")

    for lam in lambdas:
        reg_config = {"lambda_exchange": make_quantity(lam, "rad2")}

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
        lambdas=make_quantity(lambdas, "rad2"),
        data_misfits=make_quantity(data_misfits, "rad2"),
        reg_norms=make_quantity(reg_norms, ""),
        magnetizations=make_quantity(all_mag, ""),
        ramp_coeffs=RampCoeffs(
            offset=make_quantity(all_ramp[:, 0], "rad"),
            slope_y=make_quantity(all_ramp[:, 1], "rad/nm"),
            slope_x=make_quantity(all_ramp[:, 2], "rad/nm"),
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
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
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
    Returns
    -------
    LCurveResult
        Named tuple with ``lambdas``, ``data_misfits``,
        ``reg_norms``, ``magnetizations``, ``ramp_coeffs``, and
        ``corner_index``.

    """
    phase = _as_angle_quantity(phase)
    pixel_size = _as_length_quantity(pixel_size)
    lambdas_q = _to_lambda_exchange(lambdas)
    lambdas_np = np.atleast_1d(np.asarray(lambdas_q.value, dtype=np.float64))
    lambdas_jax = jnp.asarray(lambdas_np)
    _validate_positive(pixel_size, "pixel_size")

    mask = jnp.asarray(mask, dtype=bool)
    if reg_mask is None:
        reg_mask = mask
    else:
        reg_mask = jnp.asarray(reg_mask)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    config = _resolve_solver_config(solver, solver_config)

    init_mag = make_quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")

    def _solve_for_lam_newton(lam):
        reg_config = {"lambda_exchange": make_quantity(lam, "rad2")}
        (mag, ramp), _loss, converged = _run_newton_cg_solver_2d(
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
        return cast(u.Quantity, mag).value, _ramp_coeffs_to_array(cast(RampCoeffs, ramp)), converged

    solve_for_lam = _solve_for_lam_newton

    solve_batch = jax.jit(jax.vmap(solve_for_lam))
    all_mag, all_ramp, all_converged = solve_batch(lambdas_jax)

    converged_arr = np.asarray(all_converged)
    if not np.all(converged_arr):
        failed = np.where(~converged_arr)[0]
        failed_lams = lambdas_np[failed]
        warnings.warn(
            f"CG solver did not converge for {len(failed)} of {len(lambdas_np)} "
            f"lambda value(s): {failed_lams.tolist()}. Consider increasing "
            f"cg_maxiter or relaxing cg_tol in NewtonCGConfig.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _decompose_single(mag, ramp):
        masked_mag = u.Quantity(
            jnp.stack([
                mag[..., 0] * mask,
                mag[..., 1] * mask,
            ], axis=-1),
            "",
        )
        predicted = forward_model_single_rdfc_2d(
            masked_mag, _ramp_coeffs_from_array(ramp), rdfc_kernel, pixel_size,
        )
        residuals = predicted - phase
        dm = cast(u.Quantity, qnp.sum(residuals ** 2)).value
        rn = cast(u.Quantity, exchange_loss_fn(masked_mag, reg_mask)).value
        return dm, rn

    decompose_batch = jax.jit(jax.vmap(_decompose_single))
    all_dm, all_rn = decompose_batch(all_mag, all_ramp)

    data_misfits = np.asarray(all_dm)
    reg_norms = np.asarray(all_rn)
    corner_idx, _ = kneedle_corner(data_misfits, reg_norms)

    result = LCurveResult(
        lambdas=make_quantity(lambdas_np, "rad2"),
        data_misfits=make_quantity(data_misfits, "rad2"),
        reg_norms=make_quantity(reg_norms, ""),
        magnetizations=make_quantity(all_mag, ""),
        ramp_coeffs=RampCoeffs(
            offset=make_quantity(all_ramp[:, 0], "rad"),
            slope_y=make_quantity(all_ramp[:, 1], "rad/nm"),
            slope_x=make_quantity(all_ramp[:, 2], "rad/nm"),
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

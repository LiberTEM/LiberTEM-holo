"""Unit handling, validators and physical constants for MBIR.

This module contains:

* Physical constants used across the MBIR pipeline (``PHI_0``, ``B_REF``,
  ``KERNEL_COEFF``, ``MU_0``, ``ELECTRON_INTERACTION_CONSTANT_300KV``).
* :class:`RampCoeffs` — the typed container for background ramp parameters.
* ``make_quantity`` / ``add_units_to_inputs`` — user-facing Quantity
  constructors.
* Internal ``_as_*``/``_assert_*``/``_validate_*`` helpers that normalize
  inputs to the canonical units used by the rest of the package.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
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

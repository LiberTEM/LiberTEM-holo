"""Physical-unit post-processing for reconstructed magnetization.

Converts the dimensionless projected magnetization returned by the solver
into projected and local magnetic induction / magnetization, and
provides a helper to estimate thickness from an MIP phase image.
"""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import numpy as np
import unxt as u

from .units import (
    B_REF,
    ELECTRON_INTERACTION_CONSTANT_300KV,
    MU_0,
    _to_nm,
    _as_angle_quantity,
    _as_dimensionless_quantity,
    _as_induction_quantity,
    _as_interaction_constant_quantity,
    _as_length_quantity,
    _as_projected_induction_integral_quantity,
    _as_projected_magnetization_integral_quantity,
    _as_voltage_quantity,
    _assert_quantity_compatible,
    _broadcast_thickness_like,
    _validate_all_positive_quantity,
    _validate_positive,
    make_quantity,
)


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
    result = _to_nm(result)

    _assert_quantity_compatible(result, "nm", "thickness")
    return result


def estimate_mip_phase_from_thickness(
    thickness,
    mean_inner_potential: u.Quantity,
    interaction_constant: u.Quantity = ELECTRON_INTERACTION_CONSTANT_300KV,
) -> u.Quantity:
    r"""Estimate the MIP phase corresponding to a given thickness.

    Uses the standard relation

    .. math::

        \phi_E = C_E V_0 t

    to convert a physical thickness back to the equivalent mean-inner-
    potential phase shift.

    Parameters
    ----------
    thickness : Quantity["length"]
        Physical thickness map or scalar thickness.
    mean_inner_potential : Quantity["voltage"]
        Mean inner potential :math:`V_0` of the material.
    interaction_constant : Quantity["angle / (voltage * length)"], optional
        Electron interaction constant :math:`C_E`. Defaults to the
        300 kV value ``6.53e6 rad / (V m)``.

    Returns
    -------
    Quantity["angle"]
        Mean-inner-potential phase in radians.
    """
    thickness_q = _as_length_quantity(thickness, name="thickness")
    mean_inner_potential_q = _as_voltage_quantity(mean_inner_potential)
    interaction_constant_q = _as_interaction_constant_quantity(interaction_constant)
    _validate_positive(mean_inner_potential_q, "mean_inner_potential")
    _validate_positive(interaction_constant_q, "interaction_constant")

    result = cast(
        u.Quantity,
        interaction_constant_q * mean_inner_potential_q * thickness_q,
    )
    result = cast(u.Quantity, u.uconvert("rad", result))

    _assert_quantity_compatible(result, "rad", "mip_phase")
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


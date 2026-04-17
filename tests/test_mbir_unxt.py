"""Tests for unxt Quantity support in libertem_holo.base.mbir.

Verifies that every public function accepting physical parameters
works correctly when called with ``unxt.Quantity`` arguments,
and that unit conversion (e.g. µm → nm) produces identical results.
"""

import os
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp  # noqa: E402
import unxt as u  # noqa: E402

from libertem_holo.base.mbir import (  # noqa: E402
    RampCoeffs,
    RegConfig,
    SolverResult,
    add_units_to_inputs,
    apply_ramp,
    bootstrap_threshold_uncertainty_2d,
    build_rdfc_kernel,
    decompose_loss,
    estimate_mip_phase_from_thickness,
    estimate_thickness_from_mip_phase,
    forward_model_2d,
    forward_model_3d,
    forward_model_single_rdfc_2d,
    lcurve_sweep,
    make_quantity,
    mbir_loss_2d,
    plot_bootstrap_mask_summary,
    reconstruct_2d,
    solve_mbir_2d,
    NewtonCGConfig,
    to_local_induction,
    to_local_magnetization,
    to_projected_induction_integral,
    to_projected_magnetization_integral,
)


def array_value(value):
    if isinstance(value, u.Quantity):
        return np.asarray(value.value)
    return np.asarray(value)


def assert_quantity_compatible(value, unit: str):
    assert isinstance(value, u.Quantity)
    u.uconvert(unit, value)


def test_make_quantity_constructs_scalar_and_array_values():
    scalar = make_quantity(0.58, "nm")
    vector = make_quantity([1.0, 2.0, 3.0], "rad2")

    assert_quantity_compatible(scalar, "nm")
    assert_quantity_compatible(vector, "rad2")
    assert_allclose(np.asarray(scalar.value), 0.58)
    assert_allclose(np.asarray(vector.value), np.array([1.0, 2.0, 3.0]))


def test_add_units_to_inputs_builds_common_quantity_dict():
    result = add_units_to_inputs(
        phase=np.array([[1.0, 2.0]]),
        pixel_size=0.58,
        thickness=125.0,
    )

    assert set(result) == {
        "phase",
        "pixel_size",
        "thickness",
    }
    assert_quantity_compatible(result["phase"], "rad")
    assert_quantity_compatible(result["pixel_size"], "nm")
    assert_quantity_compatible(result["thickness"], "nm")


def test_add_units_to_inputs_converts_existing_quantities():
    result = add_units_to_inputs(
        phase=u.Quantity([180.0], "deg"),
        pixel_size=u.Quantity(0.007423, "um"),
        thickness=u.Quantity(0.125, "um"),
    )

    assert_allclose(array_value(result["phase"]), np.array([np.pi]), rtol=1e-12, atol=1e-12)
    assert_allclose(array_value(result["pixel_size"]), 7.423, rtol=1e-12, atol=1e-12)
    assert_allclose(array_value(result["thickness"]), 125.0, rtol=1e-12, atol=1e-12)


def test_add_units_to_inputs_accepts_explicit_phase_variants_with_shared_unit():
    result = add_units_to_inputs(
        mag_phase=u.Quantity([180.0], "deg"),
        mip_phase=np.array([500.0]),
        phase_unit="mrad",
    )

    assert set(result) == {"mag_phase", "mip_phase"}
    assert_quantity_compatible(result["mag_phase"], "mrad")
    assert_quantity_compatible(result["mip_phase"], "mrad")
    assert_allclose(array_value(result["mag_phase"]), np.array([np.pi * 1000.0]), rtol=1e-12, atol=1e-12)
    assert_allclose(array_value(result["mip_phase"]), np.array([500.0]), rtol=1e-12, atol=1e-12)


def test_add_units_to_inputs_rejects_mixed_generic_and_explicit_phase_keys():
    with pytest.raises(ValueError, match="Use either 'phase' or the explicit 'mag_phase'/'mip_phase' inputs"):
        add_units_to_inputs(
            phase=np.array([1.0]),
            mip_phase=np.array([2.0]),
        )


def test_estimate_thickness_from_mip_phase_matches_reference_formula():
    mip_phase = make_quantity(np.array([[10.0, -20.0], [40.0, 80.0]]), "rad")
    mean_inner_potential = make_quantity(17.0, "V")

    result = estimate_thickness_from_mip_phase(
        mip_phase,
        mean_inner_potential=mean_inner_potential,
    )

    expected = np.abs(np.array([[10.0, -20.0], [40.0, 80.0]])) / (17.0 * 6.53e6) * 1e9
    assert_quantity_compatible(result, "nm")
    assert_allclose(array_value(result), expected, rtol=1e-12, atol=1e-12)


def test_estimate_mip_phase_from_thickness_matches_inverse_formula():
    thickness = make_quantity(np.array([[10.0, 20.0], [40.0, 80.0]]), "nm")
    mean_inner_potential = make_quantity(17.0, "V")

    result = estimate_mip_phase_from_thickness(
        thickness,
        mean_inner_potential=mean_inner_potential,
    )

    expected = np.array([[10.0, 20.0], [40.0, 80.0]]) * 1e-9 * 17.0 * 6.53e6
    assert_quantity_compatible(result, "rad")
    assert_allclose(array_value(result), expected, rtol=1e-12, atol=1e-12)


def test_phase_thickness_estimates_round_trip_for_positive_inputs():
    mip_phase = make_quantity(np.array([[10.0, 20.0], [40.0, 80.0]]), "rad")
    mean_inner_potential = make_quantity(17.0, "V")

    thickness = estimate_thickness_from_mip_phase(
        mip_phase,
        mean_inner_potential=mean_inner_potential,
    )
    reconstructed_phase = estimate_mip_phase_from_thickness(
        thickness,
        mean_inner_potential=mean_inner_potential,
    )

    assert_allclose(
        array_value(reconstructed_phase),
        array_value(mip_phase),
        rtol=1e-12,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_problem():
    """8×8 reconstruction problem (same as test_mbir.py)."""
    H, W = 8, 8
    voxel_size = 10.0
    pixel_size = u.Quantity(voxel_size, "nm")

    mask = np.zeros((H, W), dtype=bool)
    mask[2:-2, 2:-2] = True

    gt_mag = np.zeros((H, W, 2), dtype=np.float64)
    gt_mag[..., 0] = mask.astype(np.float64)
    gt_mag[..., 1] = 0.5 * mask.astype(np.float64)
    gt_mag_q = u.Quantity(jnp.array(gt_mag), "")

    kernel = build_rdfc_kernel((H, W), geometry="disc")
    ramp = RampCoeffs.zeros(dtype=jnp.float64)
    phase = forward_model_single_rdfc_2d(
        gt_mag_q, cast(Any, ramp), kernel, pixel_size,
    )

    return {
        "H": H,
        "W": W,
        "voxel_size": voxel_size,
        "pixel_size": pixel_size,
        "mask": mask,
        "gt_mag": gt_mag,
        "gt_mag_q": gt_mag_q,
        "phase": phase,
        "kernel": kernel,
        "ramp": ramp,
    }


# ===================================================================
# build_rdfc_kernel
# ===================================================================


# ===================================================================
# apply_ramp
# ===================================================================

class TestApplyRampQuantity:
    """apply_ramp with Quantity arguments."""

    def test_ramp_coeffs_quantity(self):
        rc = RampCoeffs(
            offset=u.Quantity(1.0, "rad"),
            slope_y=u.Quantity(0.0, "rad/nm"),
            slope_x=u.Quantity(0.0, "rad/nm"),
        )
        result = apply_ramp(rc, 4, 4, pixel_size=u.Quantity(10.0, "nm"))
        assert isinstance(result, u.Quantity)
        assert_allclose(np.asarray(result), 1.0, atol=1e-15)

    def test_ramp_with_micrometer_pixel_size(self):
        """Pixel size in µm converts to nm internally."""
        rc = RampCoeffs(
            offset=u.Quantity(0.0, "rad"),
            slope_y=u.Quantity(0.1, "rad/nm"),
            slope_x=u.Quantity(0.0, "rad/nm"),
        )
        ref = apply_ramp(rc, 4, 4, pixel_size=u.Quantity(10.0, "nm"))
        qty = apply_ramp(rc, 4, 4, pixel_size=u.Quantity(0.01, "um"))
        assert_allclose(np.asarray(qty), np.asarray(ref), atol=1e-12)

    def test_pixel_size_in_angstrom(self):
        """100 Å == 10 nm."""
        rc = RampCoeffs(
            offset=u.Quantity(1.0, "rad"),
            slope_y=u.Quantity(0.0, "rad/nm"),
            slope_x=u.Quantity(0.0, "rad/nm"),
        )
        ref = apply_ramp(rc, 4, 4, pixel_size=u.Quantity(10.0, "nm"))
        qty = apply_ramp(rc, 4, 4, pixel_size=u.Quantity(100.0, "Angstrom"))
        assert_allclose(np.asarray(qty), np.asarray(ref), atol=1e-12)

    def test_returns_quantity_rad(self):
        rc = RampCoeffs(
            offset=u.Quantity(0.0, "rad"),
            slope_y=u.Quantity(0.0, "rad/nm"),
            slope_x=u.Quantity(0.0, "rad/nm"),
        )
        result = apply_ramp(rc, 4, 4, pixel_size=u.Quantity(10.0, "nm"))
        assert isinstance(result, u.Quantity)
        assert_quantity_compatible(result, "rad")

    def test_rejects_non_length_pixel_size(self):
        rc = RampCoeffs.zeros(dtype=jnp.float64)
        with pytest.raises(ValueError, match="pixel_size"):
            apply_ramp(rc, 4, 4, pixel_size=u.Quantity(1.0, "rad"))


# ===================================================================
# forward_model_2d
# ===================================================================

class TestForwardModel2DQuantity:
    """forward_model_2d with Quantity physical params."""

    def test_basic_quantity_call(self, small_problem):
        sp = small_problem
        result = forward_model_2d(
            sp["gt_mag_q"],
            pixel_size=sp["pixel_size"],
            geometry="disc",
        )
        assert isinstance(result, u.Quantity)
        assert_quantity_compatible(result, "rad")
        assert result.shape == sp["phase"].shape

    def test_pixel_size_um_matches_nm(self, small_problem):
        """10 nm == 0.01 µm → identical phase."""
        sp = small_problem
        ref = forward_model_2d(
            sp["gt_mag_q"],
            pixel_size=sp["pixel_size"],
            geometry="disc",
        )
        qty = forward_model_2d(
            sp["gt_mag_q"],
            pixel_size=u.Quantity(sp["voxel_size"] / 1000.0, "um"),
            geometry="disc",
        )
        assert_allclose(array_value(qty), array_value(ref), atol=1e-12)

    def test_rejects_non_dimensionless_magnetization(self, small_problem):
        sp = small_problem
        with pytest.raises(ValueError, match="magnetization"):
            forward_model_2d(
                u.Quantity(sp["gt_mag"], "T"),
                pixel_size=sp["pixel_size"],
                geometry="disc",
            )


# ===================================================================
# forward_model_single_rdfc_2d
# ===================================================================

class TestForwardModelSingleRDFCQuantity:
    """forward_model_single_rdfc_2d with Quantity params."""

    def test_full_quantity_call(self, small_problem):
        sp = small_problem
        result = forward_model_single_rdfc_2d(
            sp["gt_mag_q"],
            cast(Any, sp["ramp"]),
            sp["kernel"],
            sp["pixel_size"],
        )
        assert isinstance(result, u.Quantity)
        assert result.shape == sp["phase"].shape
        assert_allclose(array_value(result), array_value(sp["phase"]), atol=1e-12)


# ===================================================================
# forward_model_3d
# ===================================================================

class TestForwardModel3DQuantity:
    """forward_model_3d with Quantity physical params."""

    def test_basic_quantity_call(self):
        mag_3d = np.zeros((4, 4, 4, 3), dtype=np.float64)
        mag_3d[1:3, 1:3, 1:3, 0] = 1.0
        result = forward_model_3d(
            u.Quantity(mag_3d, ""),
            pixel_size=u.Quantity(10.0, "nm"),
            axis="z",
        )
        assert isinstance(result, u.Quantity)
        assert result.shape == (4, 4)

    def test_pixel_size_um_matches_nm(self):
        mag_3d = np.zeros((4, 4, 4, 3), dtype=np.float64)
        mag_3d[1:3, 1:3, 1:3, 0] = 1.0
        ref = forward_model_3d(
            u.Quantity(mag_3d, ""),
            pixel_size=u.Quantity(10.0, "nm"),
            axis="z",
        )
        qty = forward_model_3d(
            u.Quantity(mag_3d, ""),
            pixel_size=u.Quantity(0.01, "um"),
            axis="z",
        )
        assert_allclose(np.asarray(qty), np.asarray(ref), atol=1e-12)


# ===================================================================
# solve_mbir_2d
# ===================================================================

class TestSolveMBIR2DQuantity:
    """solve_mbir_2d with Quantity pixel_size."""

    def test_quantity_pixel_size(self, small_problem):
        sp = small_problem
        result = solve_mbir_2d(
            sp["phase"],
            u.Quantity(jnp.zeros_like(jnp.array(sp["gt_mag"])), ""),
            sp["mask"],
            pixel_size=sp["pixel_size"],
            solver=NewtonCGConfig(cg_maxiter=10),
            reg_config=RegConfig(lambda_exchange=u.Quantity(1e-3, "rad2")),
            rdfc_kernel=sp["kernel"],
        )
        assert isinstance(result, SolverResult)
        assert isinstance(result.magnetization, u.Quantity)
        assert isinstance(result.ramp_coeffs, RampCoeffs)
        assert isinstance(result.loss_history, u.Quantity)
        assert_quantity_compatible(result.magnetization, "")
        assert_quantity_compatible(result.ramp_coeffs.offset, "rad")
        assert_quantity_compatible(result.ramp_coeffs.slope_y, "rad / nm")
        assert_quantity_compatible(result.ramp_coeffs.slope_x, "rad / nm")
        assert_quantity_compatible(result.loss_history, "rad2")
        assert result.magnetization.shape == sp["gt_mag"].shape

    def test_block_jacobi_preconditioner_preserves_quantity_contract(self, small_problem):
        sp = small_problem
        result = solve_mbir_2d(
            sp["phase"],
            u.Quantity(jnp.zeros_like(jnp.array(sp["gt_mag"])), ""),
            sp["mask"],
            pixel_size=sp["pixel_size"],
            solver=NewtonCGConfig(cg_maxiter=10, preconditioner="block_jacobi"),
            reg_config={"lambda_exchange": u.Quantity(1e-3, "rad2")},  # dict for backward compat
            rdfc_kernel=sp["kernel"],
        )
        assert isinstance(result, SolverResult)
        assert isinstance(result.magnetization, u.Quantity)
        assert isinstance(result.ramp_coeffs, RampCoeffs)
        assert isinstance(result.loss_history, u.Quantity)
        assert_quantity_compatible(result.magnetization, "")
        assert_quantity_compatible(result.ramp_coeffs.offset, "rad")
        assert_quantity_compatible(result.ramp_coeffs.slope_y, "rad / nm")
        assert_quantity_compatible(result.ramp_coeffs.slope_x, "rad / nm")
        assert_quantity_compatible(result.loss_history, "rad2")
        assert result.magnetization.shape == sp["gt_mag"].shape


# ===================================================================
# reconstruct_2d
# ===================================================================

class TestReconstruct2DQuantity:
    """reconstruct_2d with Quantity parameters."""

    def test_all_quantity_params(self, small_problem):
        sp = small_problem
        result = reconstruct_2d(
            sp["phase"],
            pixel_size=sp["pixel_size"],
            mask=sp["mask"],
            lam=u.Quantity(1e-3, "rad2"),
            solver=NewtonCGConfig(cg_maxiter=10),
        )
        assert isinstance(result, SolverResult)
        assert isinstance(result.magnetization, u.Quantity)
        assert isinstance(result.ramp_coeffs, RampCoeffs)
        assert isinstance(result.loss_history, u.Quantity)
        assert_quantity_compatible(result.magnetization, "")
        assert_quantity_compatible(result.loss_history, "rad2")
        assert result.magnetization.shape == (*sp["phase"].shape, 2)

    def test_um_pixel_size_matches_nm(self, small_problem):
        """Quantity in µm and nm give same result."""
        sp = small_problem
        cfg = NewtonCGConfig(cg_maxiter=5)
        ref = reconstruct_2d(
            sp["phase"],
            pixel_size=sp["pixel_size"],
            mask=sp["mask"],
            lam=u.Quantity(1e-3, "rad2"),
            solver=cfg,
        )
        qty = reconstruct_2d(
            sp["phase"],
            pixel_size=u.Quantity(sp["voxel_size"] / 1000.0, "um"),
            mask=sp["mask"],
            lam=u.Quantity(1e-3, "rad2"),
            solver=cfg,
        )
        assert_allclose(
            np.asarray(qty.magnetization.value),
            np.asarray(ref.magnetization.value),
            atol=1e-6,
        )

    def test_rejects_non_angle_phase(self, small_problem):
        sp = small_problem
        with pytest.raises(ValueError, match="phase"):
            reconstruct_2d(
                u.Quantity(array_value(sp["phase"]), "nm"),
                pixel_size=sp["pixel_size"],
                mask=sp["mask"],
                lam=u.Quantity(1e-3, "rad2"),
                solver=NewtonCGConfig(cg_maxiter=10),
            )


# ===================================================================
# magnetization conversion helpers
# ===================================================================

class TestMagnetizationConversionHelpers:
    """Public helpers for projected integrals and local physical fields."""

    def test_to_projected_induction_integral_scales_dimensionless_result(self, small_problem):
        magnetization = small_problem["gt_mag_q"]
        result = to_projected_induction_integral(
            magnetization,
            pixel_size=small_problem["pixel_size"],
            reference_induction=u.Quantity(0.6, "T"),
        )
        assert isinstance(result, u.Quantity)
        assert_quantity_compatible(result, "T nm")
        assert_allclose(
            array_value(result),
            0.6 * small_problem["voxel_size"] * array_value(magnetization),
            atol=1e-12,
        )

    def test_to_projected_magnetization_integral_uses_mu0(self, small_problem):
        magnetization = small_problem["gt_mag_q"]
        result = to_projected_magnetization_integral(
            magnetization,
            pixel_size=small_problem["pixel_size"],
            reference_induction=u.Quantity(0.6, "T"),
        )
        assert isinstance(result, u.Quantity)
        assert_quantity_compatible(result, "A")
        expected = 0.6 * (small_problem["voxel_size"] * 1e-9) * array_value(magnetization) / (4e-7 * np.pi)
        assert_allclose(array_value(result), expected, rtol=1e-12, atol=1e-12)

    def test_to_projected_induction_integral_rejects_non_dimensionless_input(self, small_problem):
        with pytest.raises(ValueError, match="magnetization"):
            to_projected_induction_integral(
                u.Quantity(array_value(small_problem["gt_mag_q"]), "T"),
                pixel_size=small_problem["pixel_size"],
            )


    def test_to_local_induction_scales_by_scalar_thickness(self, small_problem):
        magnetization = small_problem["gt_mag_q"]
        projected = to_projected_induction_integral(
            magnetization,
            pixel_size=small_problem["pixel_size"],
            reference_induction=u.Quantity(0.6, "T"),
        )
        thickness = u.Quantity(12.5, "nm")

        result = to_local_induction(projected, thickness)

        assert isinstance(result, u.Quantity)
        assert_quantity_compatible(result, "T")
        expected = 0.6 * small_problem["voxel_size"] * array_value(magnetization) / 12.5
        assert_allclose(array_value(result), expected, rtol=1e-12, atol=1e-12)

    def test_to_local_induction_broadcasts_thickness_image(self, small_problem):
        magnetization = small_problem["gt_mag_q"]
        projected = to_projected_induction_integral(
            magnetization,
            pixel_size=small_problem["pixel_size"],
        )
        thickness_map = u.Quantity(
            jnp.linspace(5.0, 20.0, small_problem["H"] * small_problem["W"]).reshape(
                small_problem["H"], small_problem["W"]
            ),
            "nm",
        )

        result = to_local_induction(projected, thickness_map)

        assert isinstance(result, u.Quantity)
        assert_quantity_compatible(result, "T")
        expected = array_value(projected) / array_value(thickness_map)[..., None]
        assert_allclose(array_value(result), expected, rtol=1e-12, atol=1e-12)

    def test_to_local_magnetization_scales_by_scalar_thickness(self, small_problem):
        projected = to_projected_magnetization_integral(
            small_problem["gt_mag_q"],
            pixel_size=small_problem["pixel_size"],
            reference_induction=u.Quantity(0.6, "T"),
        )
        thickness = u.Quantity(12.5, "nm")

        result = to_local_magnetization(projected, thickness)

        assert isinstance(result, u.Quantity)
        assert_quantity_compatible(result, "A / m")
        expected = array_value(projected) / (12.5e-9)
        assert_allclose(array_value(result), expected, rtol=1e-12, atol=1e-12)

    def test_to_local_magnetization_broadcasts_thickness_image(self, small_problem):
        projected = to_projected_magnetization_integral(
            small_problem["gt_mag_q"],
            pixel_size=small_problem["pixel_size"],
        )
        thickness_map = u.Quantity(
            jnp.linspace(5.0, 20.0, small_problem["H"] * small_problem["W"]).reshape(
                small_problem["H"], small_problem["W"]
            ),
            "nm",
        )

        result = to_local_magnetization(projected, thickness_map)

        assert isinstance(result, u.Quantity)
        assert_quantity_compatible(result, "A / m")
        expected = array_value(projected) / (array_value(thickness_map)[..., None] * 1e-9)
        assert_allclose(array_value(result), expected, rtol=1e-12, atol=1e-12)

    def test_local_scaling_rejects_non_positive_thickness(self, small_problem):
        projected = to_projected_induction_integral(
            small_problem["gt_mag_q"],
            pixel_size=small_problem["pixel_size"],
        )
        thickness_map = u.Quantity(jnp.ones((small_problem["H"], small_problem["W"])), "nm")
        thickness_map = u.Quantity(thickness_map.value.at[0, 0].set(0.0), "nm")

        with pytest.raises(ValueError, match="thickness"):
            to_local_induction(projected, thickness_map)

    def test_to_local_induction_clamps_small_thickness_when_requested(self, small_problem):
        projected = to_projected_induction_integral(
            small_problem["gt_mag_q"],
            pixel_size=small_problem["pixel_size"],
        )
        thickness_map = u.Quantity(jnp.array([[0.0, 100.0]]), "nm")
        projected = projected[:1, :2, :]

        result = to_local_induction(
            projected,
            thickness_map,
            min_effective_thickness=u.Quantity(50.0, "nm"),
        )

        expected_divisor = np.array([[50.0, 100.0]])[..., None]
        expected = array_value(projected) / expected_divisor
        assert_allclose(array_value(result), expected, rtol=1e-12, atol=1e-12)

    def test_to_local_induction_can_mark_small_thickness_invalid(self, small_problem):
        projected = to_projected_induction_integral(
            small_problem["gt_mag_q"],
            pixel_size=small_problem["pixel_size"],
        )
        thickness_map = u.Quantity(jnp.array([[10.0, 100.0]]), "nm")
        projected = projected[:1, :2, :]

        result = to_local_induction(
            projected,
            thickness_map,
            min_effective_thickness=u.Quantity(50.0, "nm"),
            invalid_to_nan=True,
        )

        values = array_value(result)
        assert np.isnan(values[0, 0]).all()
        assert np.isfinite(values[0, 1]).all()

    def test_to_local_induction_rejects_non_length_thickness(self, small_problem):
        projected = to_projected_induction_integral(
            small_problem["gt_mag_q"],
            pixel_size=small_problem["pixel_size"],
        )
        with pytest.raises(ValueError, match="thickness"):
            to_local_induction(projected, u.Quantity(3.0, "rad"))

    def test_to_local_magnetization_rejects_non_magnetization_input(self, small_problem):
        projected = to_projected_induction_integral(
            small_problem["gt_mag_q"],
            pixel_size=small_problem["pixel_size"],
        )
        with pytest.raises(ValueError, match="projected_magnetization_integral"):
            to_local_magnetization(projected, u.Quantity(10.0, "nm"))

    def test_to_local_magnetization_can_mark_small_thickness_invalid(self, small_problem):
        projected = to_projected_magnetization_integral(
            small_problem["gt_mag_q"],
            pixel_size=small_problem["pixel_size"],
        )
        thickness_map = u.Quantity(jnp.array([[10.0, 100.0]]), "nm")
        projected = projected[:1, :2, :]

        result = to_local_magnetization(
            projected,
            thickness_map,
            min_effective_thickness=u.Quantity(50.0, "nm"),
            invalid_to_nan=True,
        )

        values = array_value(result)
        assert np.isnan(values[0, 0]).all()
        assert np.isfinite(values[0, 1]).all()

# ===================================================================
# mbir_loss_2d
# ===================================================================

class TestMBIRLossQuantity:
    """mbir_loss_2d keeps units through the objective."""

    def test_quantity_loss_and_lambda(self, small_problem):
        sp = small_problem
        loss = mbir_loss_2d(
            (sp["gt_mag_q"], cast(Any, sp["ramp"])),
            sp["mask"],
            sp["phase"],
            sp["kernel"],
            sp["pixel_size"],
            {"lambda_exchange": u.Quantity(1e-3, "rad2")},
            sp["mask"],
        )
        assert isinstance(loss, u.Quantity)
        assert_quantity_compatible(loss, "rad2")
        assert float(array_value(loss)) >= 0.0

    def test_rejects_non_angle_lambda(self, small_problem):
        sp = small_problem
        with pytest.raises(ValueError, match="lambda_exchange"):
            mbir_loss_2d(
                (sp["gt_mag_q"], cast(Any, sp["ramp"])),
                sp["mask"],
                sp["phase"],
                sp["kernel"],
                sp["pixel_size"],
                {"lambda_exchange": u.Quantity(1.0, "nm")},
                sp["mask"],
            )


# ===================================================================
# decompose_loss
# ===================================================================

class TestDecomposeLossQuantity:
    """decompose_loss with Quantity pixel_size."""

    def test_basic_quantity_call(self, small_problem):
        sp = small_problem
        result = decompose_loss(
            sp["gt_mag_q"], sp["ramp"], sp["phase"],
            sp["mask"], sp["mask"], sp["kernel"],
            pixel_size=sp["pixel_size"],
        )
        assert len(result) >= 2
        assert isinstance(result[0], u.Quantity)
        assert isinstance(result[1], u.Quantity)
        assert_quantity_compatible(result[0], "rad2")
        assert_quantity_compatible(result[1], "")

    def test_um_matches_nm(self, small_problem):
        sp = small_problem
        ref = decompose_loss(
            sp["gt_mag_q"], sp["ramp"], sp["phase"],
            sp["mask"], sp["mask"], sp["kernel"],
            pixel_size=sp["pixel_size"],
        )
        qty = decompose_loss(
            sp["gt_mag_q"], sp["ramp"], sp["phase"],
            sp["mask"], sp["mask"], sp["kernel"],
            pixel_size=u.Quantity(sp["voxel_size"] / 1000.0, "um"),
        )
        assert_allclose(array_value(qty[0]), array_value(ref[0]), atol=1e-12)
        assert_allclose(array_value(qty[1]), array_value(ref[1]), atol=1e-12)


# ===================================================================
# lcurve_sweep
# ===================================================================

class TestLCurveQuantity:
    """lcurve_sweep with Quantity lambdas and outputs."""

    def test_quantity_lambdas(self, small_problem):
        sp = small_problem
        result = lcurve_sweep(
            phase=sp["phase"],
            mask=sp["mask"],
            pixel_size=sp["pixel_size"],
            lambdas=u.Quantity(np.array([1e-4, 1e-3]), "rad2"),
            solver_config=NewtonCGConfig(cg_maxiter=5),
            rdfc_kernel=sp["kernel"],
        )
        assert isinstance(result.lambdas, u.Quantity)
        assert isinstance(result.data_misfits, u.Quantity)
        assert isinstance(result.reg_norms, u.Quantity)
        assert_quantity_compatible(result.lambdas, "rad2")
        assert_quantity_compatible(result.data_misfits, "rad2")
        assert_quantity_compatible(result.reg_norms, "")
        assert_quantity_compatible(result.magnetizations, "")
        assert result.magnetizations.shape[0] == 2

    def test_rejects_non_rad2_lambdas(self, small_problem):
        sp = small_problem
        with pytest.raises(ValueError, match="lambda_exchange"):
            lcurve_sweep(
                phase=sp["phase"],
                mask=sp["mask"],
                pixel_size=sp["pixel_size"],
                lambdas=u.Quantity(np.array([1.0, 2.0]), "nm"),
                solver_config=NewtonCGConfig(cg_maxiter=5),
                rdfc_kernel=sp["kernel"],
            )


# ===================================================================
# bootstrap_threshold_uncertainty_2d
# ===================================================================

class TestBootstrapThresholdQuantity:
    """bootstrap_threshold_uncertainty_2d uses plain scalar thresholds."""

    def test_scalar_thresholds_and_outputs(self, small_problem):
        sp = small_problem
        mip_abs = np.abs(np.asarray(sp["phase"].value))
        threshold_value = 0.5 * float(mip_abs.max())
        result = bootstrap_threshold_uncertainty_2d(
            phase=sp["phase"],
            mip_phase=sp["phase"],
            threshold=threshold_value,
            threshold_low=0.8 * threshold_value,
            threshold_high=1.2 * threshold_value,
            pixel_size=sp["pixel_size"],
            lam=u.Quantity(1e-3, "rad2"),
            solver_config=NewtonCGConfig(cg_maxiter=5),
            n_boot=3,
            rdfc_kernel=sp["kernel"],
        )
        assert isinstance(result.threshold, float)
        assert isinstance(result.threshold_low, float)
        assert isinstance(result.threshold_high, float)
        assert isinstance(result.threshold_draws, np.ndarray)
        assert isinstance(result.mean_norm, u.Quantity)
        assert_allclose(result.threshold, threshold_value, atol=1e-12)
        assert_allclose(result.threshold_low, 0.8 * threshold_value, atol=1e-12)
        assert_allclose(result.threshold_high, 1.2 * threshold_value, atol=1e-12)
        assert result.magnetizations.shape[0] == 3
        assert result.local_induction_mean_samples is None
        assert result.local_induction_mean is None
        assert result.local_induction_mean_low is None
        assert result.local_induction_mean_high is None
        assert result.local_induction_mean_ci95 is None
        assert result.local_induction_roi_pixels is None

    def test_returns_local_induction_mean_summary_inside_each_bootstrap_mask(self, small_problem):
        sp = small_problem
        mip_abs = np.abs(np.asarray(sp["phase"].value))
        threshold_value = 0.5 * float(mip_abs.max())
        thickness = u.Quantity(np.full(sp["phase"].shape, 50.0), "nm")
        result = bootstrap_threshold_uncertainty_2d(
            phase=sp["phase"],
            mip_phase=sp["phase"],
            threshold=threshold_value,
            threshold_low=0.8 * threshold_value,
            threshold_high=1.2 * threshold_value,
            pixel_size=sp["pixel_size"],
            lam=u.Quantity(1e-3, "rad2"),
            solver_config=NewtonCGConfig(cg_maxiter=5),
            n_boot=4,
            rng_seed=3,
            rdfc_kernel=sp["kernel"],
            thickness=thickness,
            min_effective_thickness=u.Quantity(10.0, "nm"),
            invalid_to_nan=True,
        )

        assert_quantity_compatible(cast(u.Quantity, result.local_induction_mean_samples), "T")
        assert_quantity_compatible(cast(u.Quantity, result.local_induction_mean), "T")
        assert_quantity_compatible(cast(u.Quantity, result.local_induction_mean_low), "T")
        assert_quantity_compatible(cast(u.Quantity, result.local_induction_mean_high), "T")
        assert_quantity_compatible(cast(u.Quantity, result.local_induction_mean_ci95), "T")
        assert result.local_induction_roi_pixels is not None
        assert result.local_induction_roi_pixels.shape == (4,)

        manual_samples = []
        manual_pixels = []
        for draw_index, draw_threshold in enumerate(result.threshold_draws):
            draw_mask = mip_abs > draw_threshold
            projected_draw = to_projected_induction_integral(
                result.magnetizations[draw_index],
                sp["pixel_size"],
            )
            local_draw = to_local_induction(
                projected_draw,
                thickness,
                min_effective_thickness=u.Quantity(10.0, "nm"),
                invalid_to_nan=True,
            )
            local_norm = np.linalg.norm(np.asarray(local_draw.value), axis=-1)
            roi_values = local_norm[draw_mask]
            finite_roi_values = roi_values[np.isfinite(roi_values)]
            manual_pixels.append(finite_roi_values.size)
            manual_samples.append(finite_roi_values.mean())

        manual_samples = np.asarray(manual_samples)
        manual_pixels = np.asarray(manual_pixels)

        assert_allclose(
            np.asarray(cast(u.Quantity, result.local_induction_mean_samples).value),
            manual_samples,
            rtol=1e-12,
            atol=1e-12,
        )
        assert_allclose(result.local_induction_roi_pixels, manual_pixels)
        assert_allclose(
            np.asarray(cast(u.Quantity, result.local_induction_mean).value),
            np.nanmean(manual_samples),
            rtol=1e-12,
            atol=1e-12,
        )
        assert_allclose(
            np.asarray(cast(u.Quantity, result.local_induction_mean_low).value),
            np.nanpercentile(manual_samples, 2.5),
            rtol=1e-12,
            atol=1e-12,
        )
        assert_allclose(
            np.asarray(cast(u.Quantity, result.local_induction_mean_high).value),
            np.nanpercentile(manual_samples, 97.5),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_local_induction_summary_respects_invalid_thickness_pixels(self, small_problem):
        sp = small_problem
        mip_abs = np.abs(np.asarray(sp["phase"].value))
        threshold_value = 0.5 * float(mip_abs.max())
        thickness = u.Quantity(np.full(sp["phase"].shape, 50.0), "nm")
        thickness_values = np.asarray(thickness.value).copy()
        thickness_values[2:4, 2:4] = 5.0
        thickness = u.Quantity(thickness_values, "nm")

        result = bootstrap_threshold_uncertainty_2d(
            phase=sp["phase"],
            mip_phase=sp["phase"],
            threshold=threshold_value,
            threshold_low=0.9 * threshold_value,
            threshold_high=1.1 * threshold_value,
            pixel_size=sp["pixel_size"],
            lam=u.Quantity(1e-3, "rad2"),
            solver_config=NewtonCGConfig(cg_maxiter=5),
            n_boot=3,
            rng_seed=1,
            rdfc_kernel=sp["kernel"],
            thickness=thickness,
            min_effective_thickness=u.Quantity(10.0, "nm"),
            invalid_to_nan=True,
        )

        assert result.local_induction_roi_pixels is not None
        assert np.all(result.local_induction_roi_pixels >= 0)
        raw_bootstrap_pixels = np.array([
            np.count_nonzero(mip_abs > draw_threshold)
            for draw_threshold in result.threshold_draws
        ])
        assert np.all(result.local_induction_roi_pixels <= raw_bootstrap_pixels)
        assert np.any(result.local_induction_roi_pixels < raw_bootstrap_pixels)
        assert np.isfinite(np.asarray(cast(u.Quantity, result.local_induction_mean).value))

    def test_unitful_thresholds_are_rejected(self, small_problem):
        sp = small_problem
        mip_abs = np.abs(np.asarray(sp["phase"].value))
        threshold_value = 0.5 * float(mip_abs.max())
        with pytest.raises(TypeError, match="threshold must be a plain scalar without units"):
            bootstrap_threshold_uncertainty_2d(
                phase=sp["phase"],
                mip_phase=sp["phase"],
                threshold=u.Quantity(threshold_value, "rad"),
                threshold_low=0.8 * threshold_value,
                threshold_high=1.2 * threshold_value,
                pixel_size=sp["pixel_size"],
                lam=u.Quantity(1e-3, "rad2"),
                solver_config=NewtonCGConfig(cg_maxiter=5),
                n_boot=3,
                rdfc_kernel=sp["kernel"],
            )


class TestBootstrapMaskSummaryPlot:
    """Plotting helper for aggregate bootstrap mask stability."""

    def test_plot_bootstrap_mask_summary_returns_figure_axes_and_info(self, small_problem):
        import matplotlib.pyplot as plt

        sp = small_problem
        mip_abs = np.abs(np.asarray(sp["phase"].value))
        threshold_value = 0.5 * float(mip_abs.max())
        result = bootstrap_threshold_uncertainty_2d(
            phase=sp["phase"],
            mip_phase=sp["phase"],
            threshold=threshold_value,
            threshold_low=0.8 * threshold_value,
            threshold_high=1.2 * threshold_value,
            pixel_size=sp["pixel_size"],
            lam=u.Quantity(1e-3, "rad2"),
            solver_config=NewtonCGConfig(cg_maxiter=5),
            n_boot=4,
            rng_seed=2,
            rdfc_kernel=sp["kernel"],
        )

        fig, ax, info = plot_bootstrap_mask_summary(result)

        assert fig is ax.figure
        assert info["n_draws"] == 4
        assert info["stable_frequency"] == 0.5
        assert info["mask_frequency"].shape == sp["phase"].shape
        assert info["stable_support"].dtype == bool
        assert np.array_equal(info["stable_support"], info["mask_frequency"] >= 0.5)
        assert "Bootstrap mask inclusion frequency" in ax.get_title()

        plt.close(fig)

    def test_plot_bootstrap_mask_summary_rejects_invalid_frequency(self, small_problem):
        sp = small_problem
        mip_abs = np.abs(np.asarray(sp["phase"].value))
        threshold_value = 0.5 * float(mip_abs.max())
        result = bootstrap_threshold_uncertainty_2d(
            phase=sp["phase"],
            mip_phase=sp["phase"],
            threshold=threshold_value,
            pixel_size=sp["pixel_size"],
            lam=u.Quantity(1e-3, "rad2"),
            solver_config=NewtonCGConfig(cg_maxiter=5),
            n_boot=3,
            rdfc_kernel=sp["kernel"],
        )

        with pytest.raises(ValueError, match="stable_frequency"):
            plot_bootstrap_mask_summary(result, stable_frequency=1.5)


# ===================================================================
# Unit conversion edge cases
# ===================================================================

class TestUnitConversionEdgeCases:
    """Verify correct unit conversion from non-default units."""

    def test_pixel_size_in_angstrom(self):
        """100 Å == 10 nm."""
        rc = RampCoeffs(
            offset=u.Quantity(1.0, "rad"),
            slope_y=u.Quantity(0.0, "rad/nm"),
            slope_x=u.Quantity(0.0, "rad/nm"),
        )
        ref = apply_ramp(rc, 4, 4, pixel_size=u.Quantity(10.0, "nm"))
        qty = apply_ramp(rc, 4, 4, pixel_size=u.Quantity(100.0, "Angstrom"))
        assert_allclose(np.asarray(qty), np.asarray(ref), atol=1e-12)

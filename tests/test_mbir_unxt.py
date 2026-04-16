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
    SolverResult,
    apply_ramp,
    bootstrap_threshold_uncertainty_2d,
    build_rdfc_kernel,
    decompose_loss,
    forward_model_2d,
    forward_model_3d,
    forward_model_single_rdfc_2d,
    lcurve_sweep,
    mbir_loss_2d,
    reconstruct_2d,
    solve_mbir_2d,
    NewtonCGConfig,
)


def array_value(value):
    if isinstance(value, u.Quantity):
        return np.asarray(value.value)
    return np.asarray(value)


def assert_quantity_compatible(value, unit: str):
    assert isinstance(value, u.Quantity)
    u.uconvert(unit, value)


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
        with pytest.raises(AssertionError, match="pixel_size"):
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
        with pytest.raises(AssertionError, match="magnetization"):
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
            mag_3d,
            pixel_size=u.Quantity(10.0, "nm"),
            axis="z",
        )
        assert isinstance(result, u.Quantity)
        assert result.shape == (4, 4)

    def test_pixel_size_um_matches_nm(self):
        mag_3d = np.zeros((4, 4, 4, 3), dtype=np.float64)
        mag_3d[1:3, 1:3, 1:3, 0] = 1.0
        ref = forward_model_3d(
            mag_3d,
            pixel_size=u.Quantity(10.0, "nm"),
            axis="z",
        )
        qty = forward_model_3d(
            mag_3d,
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
            reg_config={"lambda_exchange": u.Quantity(1e-3, "rad2")},
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
            lam=1e-3,
            solver=cfg,
        )
        qty = reconstruct_2d(
            sp["phase"],
            pixel_size=u.Quantity(sp["voxel_size"] / 1000.0, "um"),
            mask=sp["mask"],
            lam=1e-3,
            solver=cfg,
        )
        assert_allclose(
            np.asarray(qty.magnetization.value),
            np.asarray(ref.magnetization.value),
            atol=1e-6,
        )

    def test_rejects_non_angle_phase(self, small_problem):
        sp = small_problem
        with pytest.raises(AssertionError, match="phase"):
            reconstruct_2d(
                u.Quantity(array_value(sp["phase"]), "nm"),
                pixel_size=sp["pixel_size"],
                mask=sp["mask"],
                lam=1e-3,
                solver=NewtonCGConfig(cg_maxiter=10),
            )


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
        with pytest.raises(AssertionError, match="lambda_exchange"):
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
        with pytest.raises(AssertionError, match="lambda_exchange"):
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
    """bootstrap_threshold_uncertainty_2d keeps threshold/norm units."""

    def test_quantity_thresholds_and_outputs(self, small_problem):
        sp = small_problem
        mip_abs = np.abs(np.asarray(sp["phase"].value))
        threshold_value = 0.5 * float(mip_abs.max())
        result = bootstrap_threshold_uncertainty_2d(
            phase=sp["phase"],
            mip_phase=sp["phase"],
            threshold=u.Quantity(threshold_value, "rad"),
            threshold_low=u.Quantity(0.8 * threshold_value, "rad"),
            threshold_high=u.Quantity(1.2 * threshold_value, "rad"),
            pixel_size=sp["pixel_size"],
            lam=u.Quantity(1e-3, "rad2"),
            solver_config=NewtonCGConfig(cg_maxiter=5),
            n_boot=3,
            rdfc_kernel=sp["kernel"],
        )
        assert isinstance(result.threshold, u.Quantity)
        assert isinstance(result.threshold_draws, u.Quantity)
        assert isinstance(result.mean_norm, u.Quantity)
        assert result.magnetizations.shape[0] == 3


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

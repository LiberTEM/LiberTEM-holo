"""Tests for unxt Quantity support in libertem_holo.base.mbir.

Verifies that every public function accepting physical parameters
works correctly when called with ``unxt.Quantity`` arguments,
and that unit conversion (e.g. µm → nm) produces identical results
to plain-float calls.
"""

import os

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
    build_rdfc_kernel,
    decompose_loss,
    forward_model_2d,
    forward_model_3d,
    forward_model_single_rdfc_2d,
    reconstruct_2d,
    solve_mbir_2d,
    NewtonCGConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_problem():
    """8×8 reconstruction problem (same as test_mbir.py)."""
    H, W = 8, 8
    voxel_size = 10.0
    b0 = 1.0

    mask = np.zeros((H, W), dtype=bool)
    mask[2:-2, 2:-2] = True

    gt_mag = np.zeros((H, W, 2), dtype=np.float64)
    gt_mag[..., 0] = mask.astype(np.float64)
    gt_mag[..., 1] = 0.5 * mask.astype(np.float64)

    kernel = build_rdfc_kernel((H, W), b0_tesla=b0, geometry="disc")
    ramp = jnp.zeros(3, dtype=jnp.float64)
    phase = forward_model_single_rdfc_2d(
        jnp.array(gt_mag), ramp, kernel, voxel_size,
    )
    phase = np.asarray(phase)

    return {
        "H": H,
        "W": W,
        "voxel_size": voxel_size,
        "b0": b0,
        "mask": mask,
        "gt_mag": gt_mag,
        "phase": phase,
        "kernel": kernel,
        "ramp": ramp,
    }


# ===================================================================
# build_rdfc_kernel
# ===================================================================

class TestBuildRDFCKernelQuantity:
    """build_rdfc_kernel with Quantity b0."""

    def test_quantity_tesla_matches_float(self):
        ref = build_rdfc_kernel((4, 4), b0_tesla=1.5, geometry="disc")
        qty = build_rdfc_kernel((4, 4), b0=u.Quantity(1.5, "T"), geometry="disc")
        assert_allclose(np.asarray(qty["u_fft"]), np.asarray(ref["u_fft"]), atol=1e-15)
        assert_allclose(np.asarray(qty["v_fft"]), np.asarray(ref["v_fft"]), atol=1e-15)

    def test_quantity_millitesla_converted(self):
        """1500 mT == 1.5 T → identical kernel."""
        ref = build_rdfc_kernel((4, 4), b0_tesla=1.5, geometry="disc")
        qty = build_rdfc_kernel((4, 4), b0=u.Quantity(1500.0, "mT"), geometry="disc")
        assert_allclose(np.asarray(qty["u_fft"]), np.asarray(ref["u_fft"]), atol=1e-12)
        assert_allclose(np.asarray(qty["v_fft"]), np.asarray(ref["v_fft"]), atol=1e-12)

    def test_coeff_is_quantity(self):
        kernel = build_rdfc_kernel((4, 4), b0=u.Quantity(1.0, "T"))
        assert isinstance(kernel["coeff"], u.Quantity)

    def test_plain_float_still_works(self):
        """Bare float b0 is assumed Tesla."""
        ref = build_rdfc_kernel((4, 4), b0_tesla=2.0)
        qty = build_rdfc_kernel((4, 4), b0=2.0)
        assert_allclose(np.asarray(qty["u_fft"]), np.asarray(ref["u_fft"]), atol=1e-15)


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
        assert_allclose(np.asarray(result.value), 1.0, atol=1e-15)

    def test_ramp_with_micrometer_pixel_size(self):
        """Pixel size in µm converts to nm internally."""
        ref = apply_ramp(
            jnp.array([0.0, 0.1, 0.0]), 4, 4, pixel_size_nm=10.0,
        )
        qty = apply_ramp(
            jnp.array([0.0, 0.1, 0.0]), 4, 4, pixel_size=u.Quantity(0.01, "um"),
        )
        assert_allclose(np.asarray(qty.value), np.asarray(ref.value), atol=1e-12)

    def test_returns_quantity_rad(self):
        result = apply_ramp(jnp.zeros(3), 4, 4, pixel_size_nm=10.0)
        assert isinstance(result, u.Quantity)


# ===================================================================
# forward_model_2d
# ===================================================================

class TestForwardModel2DQuantity:
    """forward_model_2d with Quantity physical params."""

    def test_pixel_size_nm_matches_quantity_nm(self, small_problem):
        sp = small_problem
        ref = forward_model_2d(
            sp["gt_mag"], pixel_size_nm=sp["voxel_size"],
            b0_tesla=sp["b0"], geometry="disc",
        )
        qty = forward_model_2d(
            sp["gt_mag"],
            pixel_size=u.Quantity(sp["voxel_size"], "nm"),
            b0=u.Quantity(sp["b0"], "T"),
            geometry="disc",
        )
        assert_allclose(np.asarray(qty), np.asarray(ref), atol=1e-14)

    def test_pixel_size_um_converts(self, small_problem):
        """10 nm == 0.01 µm → identical phase."""
        sp = small_problem
        ref = forward_model_2d(
            sp["gt_mag"], pixel_size_nm=sp["voxel_size"],
            b0_tesla=sp["b0"], geometry="disc",
        )
        qty = forward_model_2d(
            sp["gt_mag"],
            pixel_size=u.Quantity(sp["voxel_size"] / 1000.0, "um"),
            b0=u.Quantity(sp["b0"], "T"),
            geometry="disc",
        )
        assert_allclose(np.asarray(qty), np.asarray(ref), atol=1e-12)

    def test_with_thickness_quantity(self, small_problem):
        sp = small_problem
        ref = forward_model_2d(
            sp["gt_mag"], pixel_size_nm=sp["voxel_size"],
            b0_tesla=sp["b0"], thickness=20.0, geometry="disc",
        )
        qty = forward_model_2d(
            sp["gt_mag"],
            pixel_size=u.Quantity(sp["voxel_size"], "nm"),
            b0=u.Quantity(sp["b0"], "T"),
            thickness=u.Quantity(20.0, "nm"),
            geometry="disc",
        )
        assert_allclose(np.asarray(qty), np.asarray(ref), atol=1e-14)


# ===================================================================
# forward_model_3d
# ===================================================================

class TestForwardModel3DQuantity:
    """forward_model_3d with Quantity physical params."""

    def test_quantity_matches_float(self):
        mag_3d = np.zeros((4, 4, 4, 3), dtype=np.float64)
        mag_3d[1:3, 1:3, 1:3, 0] = 1.0  # uniform mx
        ref = forward_model_3d(
            mag_3d, pixel_size_nm=10.0, b0_tesla=1.0, axis="z",
        )
        qty = forward_model_3d(
            mag_3d,
            pixel_size=u.Quantity(10.0, "nm"),
            b0=u.Quantity(1.0, "T"),
            axis="z",
        )
        assert_allclose(np.asarray(qty), np.asarray(ref), atol=1e-14)


# ===================================================================
# solve_mbir_2d
# ===================================================================

class TestSolveMBIR2DQuantity:
    """solve_mbir_2d with Quantity pixel_size."""

    def test_quantity_pixel_size(self, small_problem):
        sp = small_problem
        result = solve_mbir_2d(
            sp["phase"],
            jnp.zeros_like(jnp.array(sp["gt_mag"])),
            sp["mask"],
            pixel_size=u.Quantity(sp["voxel_size"], "nm"),
            solver=NewtonCGConfig(cg_maxiter=10),
            reg_config={"lambda_exchange": 1e-3},
            rdfc_kernel=sp["kernel"],
        )
        assert isinstance(result, SolverResult)
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
            pixel_size=u.Quantity(sp["voxel_size"], "nm"),
            b0=u.Quantity(sp["b0"], "T"),
            thickness=u.Quantity(sp["voxel_size"], "nm"),
            mask=sp["mask"],
            lam=1e-3,
            solver=NewtonCGConfig(cg_maxiter=10),
        )
        assert isinstance(result, SolverResult)
        assert result.magnetization.shape == (*sp["phase"].shape, 2)

    def test_quantity_matches_float(self, small_problem):
        """Quantity call and legacy float call produce identical results."""
        sp = small_problem
        cfg = NewtonCGConfig(cg_maxiter=5)
        ref = reconstruct_2d(
            sp["phase"],
            pixel_size_nm=sp["voxel_size"],
            b0_tesla=sp["b0"],
            thickness=sp["voxel_size"],
            mask=sp["mask"],
            lam=1e-3,
            solver=cfg,
        )
        qty = reconstruct_2d(
            sp["phase"],
            pixel_size=u.Quantity(sp["voxel_size"], "nm"),
            b0=u.Quantity(sp["b0"], "T"),
            thickness=u.Quantity(sp["voxel_size"], "nm"),
            mask=sp["mask"],
            lam=1e-3,
            solver=cfg,
        )
        assert_allclose(
            np.asarray(qty.magnetization),
            np.asarray(ref.magnetization),
            atol=1e-14,
        )
        assert_allclose(
            np.asarray(qty.ramp_coeffs),
            np.asarray(ref.ramp_coeffs),
            atol=1e-14,
        )


# ===================================================================
# decompose_loss
# ===================================================================

class TestDecomposeLossQuantity:
    """decompose_loss with Quantity pixel_size."""

    def test_quantity_matches_float(self, small_problem):
        sp = small_problem
        ref = decompose_loss(
            sp["gt_mag"], sp["ramp"], sp["phase"],
            sp["mask"], sp["mask"], sp["kernel"],
            pixel_size_nm=sp["voxel_size"],
        )
        qty = decompose_loss(
            sp["gt_mag"], sp["ramp"], sp["phase"],
            sp["mask"], sp["mask"], sp["kernel"],
            pixel_size=u.Quantity(sp["voxel_size"], "nm"),
        )
        assert_allclose(qty[0], ref[0], atol=1e-14)
        assert_allclose(qty[1], ref[1], atol=1e-14)


# ===================================================================
# Unit conversion edge cases
# ===================================================================

class TestUnitConversionEdgeCases:
    """Verify correct unit conversion from non-default units."""

    def test_b0_in_gauss(self):
        """10000 G == 1 T → same kernel."""
        ref = build_rdfc_kernel((4, 4), b0_tesla=1.0)
        qty = build_rdfc_kernel((4, 4), b0=u.Quantity(10000.0, "G"))
        assert_allclose(np.asarray(qty["u_fft"]), np.asarray(ref["u_fft"]), atol=1e-10)

    def test_pixel_size_in_angstrom(self):
        """100 Å == 10 nm."""
        ref = apply_ramp(jnp.array([1.0, 0.0, 0.0]), 4, 4, pixel_size_nm=10.0)
        qty = apply_ramp(
            jnp.array([1.0, 0.0, 0.0]), 4, 4,
            pixel_size=u.Quantity(100.0, "Angstrom"),
        )
        assert_allclose(np.asarray(qty.value), np.asarray(ref.value), atol=1e-12)

    def test_thickness_in_um(self):
        """0.02 µm == 20 nm → same forward model."""
        mag = np.zeros((4, 4, 2), dtype=np.float64)
        mag[1:3, 1:3, 0] = 1.0
        ref = forward_model_2d(mag, pixel_size_nm=10.0, thickness=20.0)
        qty = forward_model_2d(
            mag,
            pixel_size=u.Quantity(10.0, "nm"),
            thickness=u.Quantity(0.02, "um"),
        )
        assert_allclose(np.asarray(qty), np.asarray(ref), atol=1e-12)

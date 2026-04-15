"""Test suite for libertem_holo.base.mbir.

Ported from the empyre/pyramid test suite and adapted for the JAX-based
MBIR implementation.  Tests cover kernel construction, RDFC phase mapping,
3D→2D projection, forward models, regularization, cost function, solvers,
convenience wrappers, and L-curve utilities.

Reference data files in ``tests/test_mbir_data/`` were extracted from the
original pyramid HDF5 test fixtures.
"""

import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

# JAX must see x64 before any other import, so we set it at import time.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from libertem_holo.base.mbir import (  # noqa: E402
    PHI_0_T_NM2,
    AdamConfig,
    LBFGSConfig,
    LCurveResult,
    NewtonCGConfig,
    SolverResult,
    _rdfc_elementary_phase,
    apply_ramp,
    build_rdfc_kernel,
    decompose_loss,
    exchange_loss_fn,
    forward_model_2d,
    forward_model_3d,
    forward_model_single_rdfc_2d,
    get_freq_grid,
    kneedle_corner,
    lcurve_sweep,
    mbir_loss_2d,
    phase_mapper_rdfc,
    project_3d,
    reconstruct_2d,
    solve_mbir_2d,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_mbir_data")
UPSTREAM_DIR = os.path.join(os.path.dirname(__file__), "test_mbir_data", "upstream")
PYRAMID_PHASEMAPPER_DIR = os.path.join(UPSTREAM_DIR, "phasemapper")
PYRAMID_PROJECTOR_DIR = os.path.join(UPSTREAM_DIR, "projector")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kernel_ref():
    """Load pyramid kernel reference data (dim_uv=4×4, a=1, b0=1, disc)."""
    return np.load(os.path.join(DATA_DIR, "kernel_ref.npz"))


@pytest.fixture(scope="module")
def phasemapper_ref():
    """Load pyramid RDFC phasemapper reference (a=10, 4×4, b0=1, disc)."""
    return np.load(os.path.join(DATA_DIR, "phasemapper_ref.npz"))


@pytest.fixture(scope="module")
def projector_ref():
    """Load pyramid SimpleProjector reference (3D magdata + projections)."""
    return np.load(os.path.join(DATA_DIR, "projector_ref.npz"))


@pytest.fixture(scope="module")
def small_problem():
    """Return a small 8×8 reconstruction problem with a known ground truth.

    Creates a disc-shaped magnetization on a small grid, computes
    its phase via the forward model, and returns all components needed
    to run/test the solvers and loss functions.
    """
    H, W = 8, 8
    voxel_size = 10.0
    b0 = 1.0

    mask = np.zeros((H, W), dtype=bool)
    mask[2:-2, 2:-2] = True

    # Ground-truth magnetization: uniform inside the mask
    gt_mag = np.zeros((H, W, 2), dtype=np.float64)
    gt_mag[..., 0] = mask.astype(np.float64)  # u-component
    gt_mag[..., 1] = 0.5 * mask.astype(np.float64)  # v-component

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
# 1. Kernel construction  (cf. pyramid test_kernel.py)
# ===================================================================


class TestBuildRDFCKernel:
    """Verify build_rdfc_kernel against pyramid's Kernel class."""

    def test_kernel_u_fft_matches_pyramid(self, kernel_ref):
        kernel = build_rdfc_kernel((4, 4), b0_tesla=1.0, geometry="disc")
        assert_allclose(
            np.asarray(kernel["u_fft"]),
            kernel_ref["ref_u_fft"],
            atol=1e-7,
            err_msg="u_fft does not match pyramid reference",
        )

    def test_kernel_v_fft_matches_pyramid(self, kernel_ref):
        kernel = build_rdfc_kernel((4, 4), b0_tesla=1.0, geometry="disc")
        assert_allclose(
            np.asarray(kernel["v_fft"]),
            kernel_ref["ref_v_fft"],
            atol=1e-7,
            err_msg="v_fft does not match pyramid reference",
        )

    def test_raw_u_kernel_matches_pyramid(self, kernel_ref):
        """Manually compute the raw 7×7 u kernel and compare."""
        dim_uv = (4, 4)
        u_coords = jnp.linspace(-3, 3, num=7, dtype=jnp.float64)
        v_coords = jnp.linspace(-3, 3, num=7, dtype=jnp.float64)
        uu, vv = jnp.meshgrid(u_coords, v_coords, indexing="xy")
        coeff = 1.0 / (2 * PHI_0_T_NM2)
        u_kernel = coeff * _rdfc_elementary_phase("disc", uu, vv)
        assert_allclose(
            np.asarray(u_kernel),
            kernel_ref["ref_u"],
            atol=1e-7,
            err_msg="Raw u kernel does not match pyramid reference",
        )

    def test_raw_v_kernel_matches_pyramid(self, kernel_ref):
        """Manually compute the raw 7×7 v kernel and compare."""
        u_coords = jnp.linspace(-3, 3, num=7, dtype=jnp.float64)
        v_coords = jnp.linspace(-3, 3, num=7, dtype=jnp.float64)
        uu, vv = jnp.meshgrid(u_coords, v_coords, indexing="xy")
        coeff = 1.0 / (2 * PHI_0_T_NM2)
        v_kernel = -coeff * _rdfc_elementary_phase("disc", vv, uu)
        assert_allclose(
            np.asarray(v_kernel),
            kernel_ref["ref_v"],
            atol=1e-7,
            err_msg="Raw v kernel does not match pyramid reference",
        )

    def test_kernel_dim_metadata(self):
        kernel = build_rdfc_kernel((10, 12))
        assert kernel["dim_uv"] == (10, 12)
        assert kernel["dim_pad"] == (20, 24)
        assert kernel["u_fft"].shape == (20, 13)  # rfft2
        assert kernel["v_fft"].shape == (20, 13)

    def test_kernel_slab_geometry_runs(self):
        kernel = build_rdfc_kernel((4, 4), geometry="slab")
        assert kernel["u_fft"].shape == (8, 5)

    def test_kernel_with_prw_vec(self):
        kernel_no_prw = build_rdfc_kernel((4, 4))
        kernel_prw = build_rdfc_kernel((4, 4), prw_vec=jnp.array([1.0, 0.0]))
        # Should differ
        assert not np.allclose(
            np.asarray(kernel_no_prw["u_fft"]),
            np.asarray(kernel_prw["u_fft"]),
        )

    def test_kernel_invalid_geometry_raises(self):
        with pytest.raises(ValueError, match="Unknown geometry"):
            build_rdfc_kernel((4, 4), geometry="cube")

    def test_kernel_b0_scaling(self):
        k1 = build_rdfc_kernel((4, 4), b0_tesla=1.0)
        k2 = build_rdfc_kernel((4, 4), b0_tesla=2.0)
        assert_allclose(
            np.asarray(k2["u_fft"]),
            2.0 * np.asarray(k1["u_fft"]),
            atol=1e-12,
            err_msg="Kernel should scale linearly with b0",
        )


# ===================================================================
# 2. Phase mapper RDFC  (cf. pyramid test_phasemapper.py TestCasePhaseMapperRDFC)
# ===================================================================


class TestPhaseMapperRDFC:
    """Verify phase_mapper_rdfc against pyramid's PhaseMapperRDFC."""

    def test_phase_matches_pyramid(self, phasemapper_ref):
        """phase_mapper_rdfc(u, v) should reproduce pyramid's phasemap."""
        u_field = jnp.array(phasemapper_ref["u_field"], dtype=jnp.float64)
        v_field = jnp.array(phasemapper_ref["v_field"], dtype=jnp.float64)
        phase_ref = phasemapper_ref["phase_ref"]
        voxel_size = float(phasemapper_ref["voxel_size"])

        kernel = build_rdfc_kernel((4, 4), b0_tesla=1.0, geometry="disc")
        phase = voxel_size**2 * phase_mapper_rdfc(u_field, v_field, kernel)

        assert_allclose(
            np.asarray(phase),
            phase_ref,
            atol=1e-7,
            err_msg="Phase mapper output does not match pyramid reference",
        )

    def test_linearity(self, phasemapper_ref):
        """RDFC mapper should be linear: f(a*u) = a*f(u)."""
        u = jnp.array(phasemapper_ref["u_field"], dtype=jnp.float64)
        v = jnp.array(phasemapper_ref["v_field"], dtype=jnp.float64)
        kernel = build_rdfc_kernel(u.shape, b0_tesla=1.0)

        phase1 = phase_mapper_rdfc(u, v, kernel)
        phase2 = phase_mapper_rdfc(2.0 * u, 2.0 * v, kernel)

        assert_allclose(
            np.asarray(phase2),
            2.0 * np.asarray(phase1),
            atol=1e-12,
            err_msg="Phase mapper is not linear",
        )

    def test_zero_input_gives_zero_phase(self):
        kernel = build_rdfc_kernel((4, 4))
        zero = jnp.zeros((4, 4), dtype=jnp.float64)
        phase = phase_mapper_rdfc(zero, zero, kernel)
        assert_allclose(
            np.asarray(phase),
            0.0,
            atol=1e-15,
            err_msg="Zero magnetization should give zero phase",
        )

    def test_jac_consistency(self, phasemapper_ref):
        """Forward model via matrix-vector product should match direct call.

        This tests the Jacobian of the RDFC mapper implicitly:
        applying the mapper to e_i basis vectors and stacking gives the
        Jacobian matrix, and Jac @ mag_vec should equal the phase.
        """
        u = jnp.array(phasemapper_ref["u_field"], dtype=jnp.float64)
        v = jnp.array(phasemapper_ref["v_field"], dtype=jnp.float64)
        voxel_size = float(phasemapper_ref["voxel_size"])
        kernel = build_rdfc_kernel(u.shape, b0_tesla=1.0)

        # Direct phase
        phase_direct = np.asarray(voxel_size**2 * phase_mapper_rdfc(u, v, kernel)).ravel()

        H, W = u.shape
        n = 2 * H * W  # number of magnetization DOF
        m = H * W  # number of phase pixels

        # Build Jacobian column-by-column
        jac = np.zeros((m, n))
        for i in range(H * W):
            # u-component basis vector
            u_basis = np.zeros((H, W), dtype=np.float64)
            u_basis.ravel()[i] = 1.0
            col = np.asarray(
                voxel_size**2 * phase_mapper_rdfc(
                    jnp.array(u_basis), jnp.zeros((H, W)), kernel,
                )
            ).ravel()
            jac[:, i] = col

        for i in range(H * W):
            # v-component basis vector
            v_basis = np.zeros((H, W), dtype=np.float64)
            v_basis.ravel()[i] = 1.0
            col = np.asarray(
                voxel_size**2 * phase_mapper_rdfc(
                    jnp.zeros((H, W)), jnp.array(v_basis), kernel,
                )
            ).ravel()
            jac[:, H * W + i] = col

        # Jac @ mag_vec should give the same phase
        mag_vec = np.concatenate([u.ravel(), v.ravel()])
        phase_jac = jac @ mag_vec
        assert_allclose(
            phase_jac,
            phase_direct,
            atol=1e-7,
            err_msg="Jacobian-based phase does not match direct phase",
        )

    def test_jac_matches_pyramid(self, phasemapper_ref):
        """Our numerically-computed Jacobian should match pyramid's jac.npy."""
        jac_data = np.load(os.path.join(DATA_DIR, "phasemapper_jac_ref.npz"))
        jac_ref = jac_data["jac_ref"]  # (m, n) = (16, 32) for 4×4

        u_shape = (4, 4)
        voxel_size = float(phasemapper_ref["voxel_size"])
        kernel = build_rdfc_kernel(u_shape, b0_tesla=1.0)

        H, W = u_shape
        n = 2 * H * W
        m = H * W

        jac = np.zeros((m, n))
        for i in range(H * W):
            u_basis = np.zeros((H, W), dtype=np.float64)
            u_basis.ravel()[i] = 1.0
            col = np.asarray(
                voxel_size**2 * phase_mapper_rdfc(
                    jnp.array(u_basis), jnp.zeros((H, W)), kernel,
                )
            ).ravel()
            jac[:, i] = col

        for i in range(H * W):
            v_basis = np.zeros((H, W), dtype=np.float64)
            v_basis.ravel()[i] = 1.0
            col = np.asarray(
                voxel_size**2 * phase_mapper_rdfc(
                    jnp.zeros((H, W)), jnp.array(v_basis), kernel,
                )
            ).ravel()
            jac[:, H * W + i] = col

        assert_allclose(
            jac, jac_ref, atol=1e-7,
            err_msg="Jacobian does not match pyramid reference",
        )

    def test_jac_matches_upstream_npy_fixture(self, phasemapper_ref):
        """Computed Jacobian should match upstream pyramid jac.npy directly."""
        jac_ref = np.load(os.path.join(PYRAMID_PHASEMAPPER_DIR, "jac.npy"))
        voxel_size = float(phasemapper_ref["voxel_size"])
        H, W = 4, 4
        n = 2 * H * W

        kernel = build_rdfc_kernel((H, W), b0_tesla=1.0)

        def phase_from_mag_vec(mag_vec):
            u_vec = mag_vec[: H * W].reshape(H, W)
            v_vec = mag_vec[H * W :].reshape(H, W)
            return (voxel_size**2 * phase_mapper_rdfc(u_vec, v_vec, kernel)).reshape(-1)

        jac = np.asarray(jax.jacfwd(phase_from_mag_vec)(jnp.zeros((n,), dtype=jnp.float64)))

        assert_allclose(
            jac,
            jac_ref,
            atol=1e-7,
            err_msg="RDFC Jacobian does not match upstream test_phasemapper/jac.npy",
        )


# ===================================================================
# 3. 3D Projection  (cf. pyramid test_projector.py SimpleProjector)
# ===================================================================


class TestProject3D:
    """Verify project_3d against pyramid's SimpleProjector."""

    @pytest.mark.parametrize("axis", ["z", "y", "x"])
    def test_projection_matches_pyramid(self, projector_ref, axis):
        # Pyramid format: (comp, Z, Y, X) → our format: (Z, Y, X, comp)
        mag_3d = jnp.array(
            np.transpose(projector_ref["magdata"], (1, 2, 3, 0)),
            dtype=jnp.float64,
        )
        result = project_3d(mag_3d, axis=axis)
        result_np = np.asarray(result)

        ref = projector_ref[f"proj_{axis}"]  # (3, 1, H, W)
        ref_u = ref[0, 0]
        ref_v = ref[1, 0]

        assert_allclose(
            result_np[..., 0], ref_u, atol=1e-5,
            err_msg=f"u-component projection mismatch for axis={axis}",
        )
        assert_allclose(
            result_np[..., 1], ref_v, atol=1e-5,
            err_msg=f"v-component projection mismatch for axis={axis}",
        )

    def test_projection_shape_z(self):
        mag_3d = jnp.zeros((6, 5, 4, 3))
        result = project_3d(mag_3d, axis="z")
        assert result.shape == (5, 4, 2)

    def test_projection_shape_y(self):
        mag_3d = jnp.zeros((6, 5, 4, 3))
        result = project_3d(mag_3d, axis="y")
        assert result.shape == (6, 4, 2)

    def test_projection_shape_x(self):
        mag_3d = jnp.zeros((6, 5, 4, 3))
        result = project_3d(mag_3d, axis="x")
        # x-axis: sum over X (axis=2) -> (Z, Y, 3) -> coeff -> (Z, Y, 2)
        # then transpose -> (Y, Z, 2) = (5, 6, 2)
        assert result.shape == (5, 6, 2)

    def test_projection_invalid_axis(self):
        mag_3d = jnp.zeros((4, 4, 4, 3))
        with pytest.raises(ValueError, match="axis must be"):
            project_3d(mag_3d, axis="w")

    def test_uniform_field_projection(self):
        """Uniform 3D field: projection sums over depth."""
        mag_3d = jnp.ones((4, 5, 6, 3), dtype=jnp.float64)
        result = project_3d(mag_3d, axis="z")
        # z-axis: sum over Z (4 slices of 1.0) → 4.0
        # coeff for z: u=mx, v=my so (u=4, v=4)
        assert_allclose(np.asarray(result[..., 0]), 4.0, atol=1e-12)
        assert_allclose(np.asarray(result[..., 1]), 4.0, atol=1e-12)

    @pytest.mark.parametrize(
        "axis,jac_name",
        [("z", "jac_z.npy"), ("y", "jac_y.npy"), ("x", "jac_x.npy")],
    )
    def test_projector_jac_matches_upstream_npy_fixture(self, axis, jac_name):
        """project_3d Jacobian should match upstream pyramid Jacobians."""
        jac_ref = np.load(os.path.join(PYRAMID_PROJECTOR_DIR, jac_name))

        # Pyramid ordering is (comp, Z, Y, X), flattened C-order.
        Z, Y, X = 6, 5, 4
        n = 3 * Z * Y * X

        def project_from_vec(vec_flat):
            mag_comp_first = vec_flat.reshape(3, Z, Y, X)
            mag_ours = jnp.transpose(mag_comp_first, (1, 2, 3, 0))
            proj = project_3d(mag_ours, axis=axis)
            return jnp.transpose(proj, (2, 0, 1)).reshape(-1)

        jac = np.asarray(jax.jacfwd(project_from_vec)(jnp.zeros((n,), dtype=jnp.float64)))

        assert_allclose(
            jac,
            jac_ref,
            atol=1e-7,
            err_msg=f"Projector Jacobian mismatch for axis={axis} against {jac_name}",
        )


# ===================================================================
# 4. Forward models  (cf. pyramid test_forwardmodel.py)
# ===================================================================


class TestForwardModels:
    """Test forward_model_single_rdfc_2d, forward_model_2d, forward_model_3d."""

    def test_single_rdfc_matches_phase_mapper(self, phasemapper_ref):
        """forward_model_single_rdfc_2d with no ramp should match phase_mapper_rdfc."""
        u = jnp.array(phasemapper_ref["u_field"], dtype=jnp.float64)
        v = jnp.array(phasemapper_ref["v_field"], dtype=jnp.float64)
        voxel_size = float(phasemapper_ref["voxel_size"])
        kernel = build_rdfc_kernel(u.shape, b0_tesla=1.0)

        phase_mapper = voxel_size**2 * phase_mapper_rdfc(u, v, kernel)

        mag = jnp.stack([u, v], axis=-1)
        ramp = jnp.zeros(3, dtype=jnp.float64)
        phase_fwd = forward_model_single_rdfc_2d(mag, ramp, kernel, voxel_size)

        assert_allclose(
            np.asarray(phase_fwd),
            np.asarray(phase_mapper),
            atol=1e-12,
        )

    def test_single_rdfc_with_ramp(self, phasemapper_ref):
        """Ramp should add a linear background to the phase."""
        u = jnp.array(phasemapper_ref["u_field"], dtype=jnp.float64)
        v = jnp.array(phasemapper_ref["v_field"], dtype=jnp.float64)
        voxel_size = float(phasemapper_ref["voxel_size"])
        kernel = build_rdfc_kernel(u.shape, b0_tesla=1.0)

        mag = jnp.stack([u, v], axis=-1)
        ramp_zero = jnp.zeros(3, dtype=jnp.float64)
        ramp_off = jnp.array([0.5, 0.0, 0.0], dtype=jnp.float64)

        phase_zero = forward_model_single_rdfc_2d(mag, ramp_zero, kernel, voxel_size)
        phase_off = forward_model_single_rdfc_2d(mag, ramp_off, kernel, voxel_size)

        assert_allclose(
            np.asarray(phase_off),
            np.asarray(phase_zero) + 0.5,
            atol=1e-12,
            err_msg="Offset ramp should shift phase uniformly",
        )

    def test_forward_model_2d_matches_pyramid(self, phasemapper_ref):
        """forward_model_2d convenience wrapper should match pyramid phase."""
        u = phasemapper_ref["u_field"]
        v = phasemapper_ref["v_field"]
        phase_ref = phasemapper_ref["phase_ref"]
        voxel_size = float(phasemapper_ref["voxel_size"])

        mag = jnp.stack([jnp.array(u), jnp.array(v)], axis=-1)
        phase = forward_model_2d(mag, voxel_size, b0_tesla=1.0, geometry="disc")

        assert_allclose(
            np.asarray(phase),
            phase_ref,
            atol=1e-7,
            err_msg="forward_model_2d does not match pyramid phasemap",
        )

    def test_forward_model_3d_z_axis(self, projector_ref, phasemapper_ref):
        """forward_model_3d along z should project then map phase."""
        magdata = projector_ref["magdata"]  # (3, Z, Y, X)
        mag_3d = jnp.array(np.transpose(magdata, (1, 2, 3, 0)), dtype=jnp.float64)
        voxel_size = float(phasemapper_ref["voxel_size"])

        phase = forward_model_3d(mag_3d, voxel_size, b0_tesla=1.0, axis="z")
        # Should produce a (5, 4) phase image
        assert phase.shape == (5, 4)
        # Not all zeros (the magdata is non-trivial)
        assert np.any(np.asarray(phase) != 0)

    def test_forward_model_3d_consistency(self, projector_ref, phasemapper_ref):
        """forward_model_3d should equal project_3d + forward_model_2d."""
        magdata = projector_ref["magdata"]
        mag_3d = jnp.array(np.transpose(magdata, (1, 2, 3, 0)), dtype=jnp.float64)
        voxel_size = float(phasemapper_ref["voxel_size"])

        for axis in ["z", "y", "x"]:
            projected = project_3d(mag_3d, axis=axis)
            phase_2step = forward_model_2d(projected, voxel_size, b0_tesla=1.0)
            phase_direct = forward_model_3d(mag_3d, voxel_size, b0_tesla=1.0, axis=axis)

            assert_allclose(
                np.asarray(phase_direct),
                np.asarray(phase_2step),
                atol=1e-10,
                err_msg=f"forward_model_3d inconsistent with project_3d+forward_model_2d (axis={axis})",
            )


# ===================================================================
# 5. Auxiliary functions
# ===================================================================


class TestGetFreqGrid:
    def test_shapes(self):
        f_y, f_x, denom = get_freq_grid(8, 10, 1.0)
        assert f_y.shape == (8, 6)  # rfft half-spectrum
        assert f_x.shape == (8, 6)
        assert denom.shape == (8, 6)

    def test_dc_bin_is_one(self):
        _, _, denom = get_freq_grid(8, 8, 1.0)
        assert float(denom[0, 0]) == 1.0


class TestApplyRamp:
    def test_offset_only(self):
        ramp = apply_ramp(jnp.array([3.0, 0.0, 0.0]), 4, 4, 1.0)
        assert_allclose(np.asarray(ramp), 3.0, atol=1e-12)

    def test_slope(self):
        coeffs = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
        ramp = apply_ramp(coeffs, 4, 4, 1.0)
        # slope_y * (y * voxel_size): y=0..3, voxel_size=1
        expected_col = np.array([0.0, 1.0, 2.0, 3.0])
        for col in range(4):
            assert_allclose(np.asarray(ramp[:, col]), expected_col, atol=1e-12)


# ===================================================================
# 6. Exchange regularization  (cf. pyramid test_regularisator.py)
# ===================================================================


class TestExchangeLossFn:
    """Verify exchange_loss_fn regularization behaviour.

    Note: the 2D implementation in mbir.py differs from pyramid's 3D
    FirstOrderRegularisator (different stencil, neighbor-count
    normalization).  We therefore test properties rather than exact
    numerical match to pyramid reference values.
    """

    def test_uniform_field_zero_loss(self):
        """A spatially uniform field inside the mask should have zero exchange loss."""
        mask = jnp.ones((6, 6), dtype=bool)
        mag = jnp.ones((6, 6, 2), dtype=jnp.float64) * 3.14
        loss = exchange_loss_fn(mag, mask)
        assert_allclose(float(loss), 0.0, atol=1e-12)

    def test_zero_field_zero_loss(self):
        mask = jnp.ones((6, 6), dtype=bool)
        mag = jnp.zeros((6, 6, 2), dtype=jnp.float64)
        loss = exchange_loss_fn(mag, mask)
        assert_allclose(float(loss), 0.0, atol=1e-15)

    def test_non_uniform_positive_loss(self):
        """A non-uniform field should produce positive exchange loss."""
        mask = jnp.ones((6, 6), dtype=bool)
        rng = np.random.default_rng(42)
        mag = jnp.array(rng.standard_normal((6, 6, 2)))
        loss = exchange_loss_fn(mag, mask)
        assert float(loss) > 0

    def test_mask_limits_region(self):
        """Variation outside the mask should not contribute to the loss."""
        rng = np.random.default_rng(42)
        full_mag = jnp.array(rng.standard_normal((8, 8, 2)))

        # Mask only center
        mask_center = jnp.zeros((8, 8), dtype=bool)
        mask_center = mask_center.at[3:5, 3:5].set(True)

        # Mask everything
        mask_full = jnp.ones((8, 8), dtype=bool)

        loss_center = float(exchange_loss_fn(full_mag, mask_center))
        loss_full = float(exchange_loss_fn(full_mag, mask_full))

        # Center loss should be less than full loss
        assert loss_center < loss_full

    def test_differentiable(self):
        """exchange_loss_fn should be differentiable via JAX autodiff."""
        mask = jnp.ones((4, 4), dtype=bool)
        mag = jnp.ones((4, 4, 2), dtype=jnp.float64)

        grad_fn = jax.grad(lambda m: exchange_loss_fn(m, mask))
        grad = grad_fn(mag)
        # Uniform field: gradient should be zero
        assert_allclose(np.asarray(grad), 0.0, atol=1e-12)

    def test_shape_mismatch_raises(self):
        mask = jnp.ones((4, 5), dtype=bool)
        mag = jnp.ones((4, 4, 2), dtype=jnp.float64)
        with pytest.raises(ValueError, match="reg_mask must have shape"):
            exchange_loss_fn(mag, mask)


# ===================================================================
# 7. MBIR loss function  (cf. pyramid test_costfunction.py)
# ===================================================================


class TestMBIRLoss:
    """Test mbir_loss_2d combining data-fidelity and regularization."""

    def test_zero_loss_at_ground_truth(self, small_problem):
        """Loss should be near zero when params match ground truth exactly."""
        sp = small_problem
        params = (jnp.array(sp["gt_mag"]), sp["ramp"])
        mask = jnp.array(sp["mask"])
        phase = jnp.array(sp["phase"])

        loss = mbir_loss_2d(
            params, mask, phase, sp["kernel"], sp["voxel_size"],
            reg_config={"lambda_exchange": 0.0},
        )
        # With lambda=0 and perfect forward model, loss should be ~0
        assert float(loss) < 1e-10

    def test_positive_loss_with_wrong_params(self, small_problem):
        """Loss should be positive when params don't match observed phase."""
        sp = small_problem
        wrong_mag = jnp.zeros_like(jnp.array(sp["gt_mag"]))
        params = (wrong_mag, sp["ramp"])
        mask = jnp.array(sp["mask"])
        phase = jnp.array(sp["phase"])

        loss = mbir_loss_2d(
            params, mask, phase, sp["kernel"], sp["voxel_size"],
            reg_config={"lambda_exchange": 0.0},
        )
        assert float(loss) > 0

    def test_regularization_increases_loss(self, small_problem):
        """Adding regularization should increase (or not decrease) loss."""
        sp = small_problem
        rng = np.random.default_rng(123)
        noisy_mag = jnp.array(rng.standard_normal(sp["gt_mag"].shape))
        params = (noisy_mag, sp["ramp"])
        mask = jnp.array(sp["mask"])
        phase = jnp.array(sp["phase"])

        loss_noreg = float(mbir_loss_2d(
            params, mask, phase, sp["kernel"], sp["voxel_size"],
            reg_config={"lambda_exchange": 0.0},
        ))
        loss_reg = float(mbir_loss_2d(
            params, mask, phase, sp["kernel"], sp["voxel_size"],
            reg_config={"lambda_exchange": 1.0},
        ))
        assert loss_reg >= loss_noreg

    def test_loss_is_differentiable(self, small_problem):
        """Gradient of loss w.r.t. params should be computable."""
        sp = small_problem
        params = (jnp.array(sp["gt_mag"]), sp["ramp"])
        mask = jnp.array(sp["mask"])
        phase = jnp.array(sp["phase"])

        grad_fn = jax.grad(
            lambda p: mbir_loss_2d(
                p, mask, phase, sp["kernel"], sp["voxel_size"],
                reg_config={"lambda_exchange": 1e-3},
            ),
        )
        grad = grad_fn(params)
        assert grad[0].shape == sp["gt_mag"].shape
        assert grad[1].shape == (3,)


# ===================================================================
# 8. Solvers  (cf. pyramid reconstruction pipeline)
# ===================================================================


class TestSolvers:
    """Test solve_mbir_2d with different solver backends."""

    def test_newton_cg_reduces_loss(self, small_problem):
        sp = small_problem
        mask = jnp.array(sp["mask"])
        phase = jnp.array(sp["phase"])
        init_mag = jnp.zeros_like(jnp.array(sp["gt_mag"]))
        kernel = sp["kernel"]

        result = solve_mbir_2d(
            phase=phase,
            init_mag=init_mag,
            mask=mask,
            pixel_size_nm=sp["voxel_size"],
            solver=NewtonCGConfig(cg_maxiter=200, cg_tol=1e-8),
            reg_config={"lambda_exchange": 1e-4},
            rdfc_kernel=kernel,
        )

        assert isinstance(result, SolverResult)
        assert result.magnetization.shape == sp["gt_mag"].shape
        assert result.ramp_coeffs.shape == (3,)
        assert result.loss_history.shape == (1,)

        # Final loss should be significantly less than initial
        final_loss = float(mbir_loss_2d(
            (result.magnetization, result.ramp_coeffs),
            mask, phase, kernel, sp["voxel_size"],
            reg_config={"lambda_exchange": 1e-4},
        ))
        init_loss = float(mbir_loss_2d(
            (init_mag, jnp.zeros(3)), mask, phase, kernel, sp["voxel_size"],
            reg_config={"lambda_exchange": 1e-4},
        ))
        assert final_loss < init_loss

    @pytest.mark.slow
    def test_adam_reduces_loss(self, small_problem):
        sp = small_problem
        mask = jnp.array(sp["mask"])
        phase = jnp.array(sp["phase"])
        init_mag = jnp.zeros_like(jnp.array(sp["gt_mag"]))
        kernel = sp["kernel"]

        result = solve_mbir_2d(
            phase=phase,
            init_mag=init_mag,
            mask=mask,
            pixel_size_nm=sp["voxel_size"],
            solver=AdamConfig(num_steps=200, learning_rate=1e-2, patience=50),
            reg_config={"lambda_exchange": 1e-4},
            rdfc_kernel=kernel,
        )

        assert isinstance(result, SolverResult)
        final_loss = float(mbir_loss_2d(
            (result.magnetization, result.ramp_coeffs),
            mask, phase, kernel, sp["voxel_size"],
            reg_config={"lambda_exchange": 1e-4},
        ))
        init_loss = float(mbir_loss_2d(
            (init_mag, jnp.zeros(3)), mask, phase, kernel, sp["voxel_size"],
            reg_config={"lambda_exchange": 1e-4},
        ))
        assert final_loss < init_loss

    @pytest.mark.slow
    def test_lbfgs_reduces_loss(self, small_problem):
        sp = small_problem
        mask = jnp.array(sp["mask"])
        phase = jnp.array(sp["phase"])
        init_mag = jnp.zeros_like(jnp.array(sp["gt_mag"]))
        kernel = sp["kernel"]

        result = solve_mbir_2d(
            phase=phase,
            init_mag=init_mag,
            mask=mask,
            pixel_size_nm=sp["voxel_size"],
            solver=LBFGSConfig(num_steps=100, patience=30),
            reg_config={"lambda_exchange": 1e-4},
            rdfc_kernel=kernel,
        )

        assert isinstance(result, SolverResult)
        final_loss = float(mbir_loss_2d(
            (result.magnetization, result.ramp_coeffs),
            mask, phase, kernel, sp["voxel_size"],
            reg_config={"lambda_exchange": 1e-4},
        ))
        init_loss = float(mbir_loss_2d(
            (init_mag, jnp.zeros(3)), mask, phase, kernel, sp["voxel_size"],
            reg_config={"lambda_exchange": 1e-4},
        ))
        assert final_loss < init_loss

    def test_solver_string_dispatch(self, small_problem):
        """String solver names should work and select the right backend."""
        sp = small_problem
        mask = jnp.array(sp["mask"])
        phase = jnp.array(sp["phase"])
        init_mag = jnp.zeros_like(jnp.array(sp["gt_mag"]))

        result = solve_mbir_2d(
            phase=phase,
            init_mag=init_mag,
            mask=mask,
            pixel_size_nm=sp["voxel_size"],
            solver="newton_cg",
            reg_config={"lambda_exchange": 1e-4},
            rdfc_kernel=sp["kernel"],
        )
        assert isinstance(result, SolverResult)

    def test_invalid_solver_string_raises(self, small_problem):
        sp = small_problem
        with pytest.raises(ValueError, match="Unknown solver"):
            solve_mbir_2d(
                phase=jnp.array(sp["phase"]),
                init_mag=jnp.zeros_like(jnp.array(sp["gt_mag"])),
                mask=jnp.array(sp["mask"]),
                pixel_size_nm=sp["voxel_size"],
                solver="bogus",
                rdfc_kernel=sp["kernel"],
            )

    def test_invalid_solver_type_raises(self, small_problem):
        sp = small_problem
        with pytest.raises(TypeError, match="solver must be a string"):
            solve_mbir_2d(
                phase=jnp.array(sp["phase"]),
                init_mag=jnp.zeros_like(jnp.array(sp["gt_mag"])),
                mask=jnp.array(sp["mask"]),
                pixel_size_nm=sp["voxel_size"],
                solver=42,
                rdfc_kernel=sp["kernel"],
            )


# ===================================================================
# 9. Convenience wrappers
# ===================================================================


class TestReconstruct2D:
    """Test reconstruct_2d convenience wrapper."""

    def test_basic_call(self, small_problem):
        sp = small_problem
        result = reconstruct_2d(
            phase=jnp.array(sp["phase"]),
            pixel_size_nm=sp["voxel_size"],
            b0_tesla=sp["b0"],
            mask=jnp.array(sp["mask"]),
            lam=1e-3,
            solver="newton_cg",
        )
        assert isinstance(result, SolverResult)
        assert result.magnetization.shape == sp["gt_mag"].shape

    def test_default_mask_is_ones(self, small_problem):
        sp = small_problem
        result = reconstruct_2d(
            phase=jnp.array(sp["phase"]),
            pixel_size_nm=sp["voxel_size"],
            b0_tesla=sp["b0"],
            mask=None,
        )
        assert isinstance(result, SolverResult)

    def test_solver_config_overrides_string(self, small_problem):
        sp = small_problem
        result = reconstruct_2d(
            phase=jnp.array(sp["phase"]),
            pixel_size_nm=sp["voxel_size"],
            b0_tesla=sp["b0"],
            mask=jnp.array(sp["mask"]),
            solver="adam",  # this should be overridden
            solver_config=NewtonCGConfig(cg_maxiter=200),
        )
        assert isinstance(result, SolverResult)


# ===================================================================
# 10. Decompose loss & L-curve utilities
# ===================================================================


class TestDecomposeLoss:
    """Test decompose_loss utility."""

    def test_perfect_reconstruction_zero_misfit(self, small_problem):
        sp = small_problem
        mag = jnp.array(sp["gt_mag"])
        ramp = sp["ramp"]
        phase = jnp.array(sp["phase"])
        mask = jnp.array(sp["mask"])
        kernel = sp["kernel"]

        dm, rn = decompose_loss(
            mag, ramp, phase, mask, mask, kernel, sp["voxel_size"],
        )
        assert dm < 1e-10, f"Data misfit should be ~0, got {dm}"

    def test_exchange_norm_zero_for_uniform(self, small_problem):
        sp = small_problem
        mask = jnp.array(sp["mask"])
        # Uniform magnetization
        uniform_mag = jnp.ones((sp["H"], sp["W"], 2), dtype=jnp.float64) * mask[..., None]
        ramp = sp["ramp"]
        phase = jnp.array(sp["phase"])
        kernel = sp["kernel"]

        _, rn = decompose_loss(
            uniform_mag, ramp, phase, mask, mask, kernel, sp["voxel_size"],
        )
        assert_allclose(rn, 0.0, atol=1e-12)

    def test_returns_two_floats(self, small_problem):
        sp = small_problem
        dm, rn = decompose_loss(
            jnp.array(sp["gt_mag"]),
            sp["ramp"],
            jnp.array(sp["phase"]),
            jnp.array(sp["mask"]),
            jnp.array(sp["mask"]),
            sp["kernel"],
            sp["voxel_size"],
        )
        assert isinstance(dm, float)
        assert isinstance(rn, float)


class TestKneedleCorner:
    """Test kneedle_corner L-curve elbow detection."""

    def test_detects_clear_elbow(self):
        """L-curve with a clear corner should be detected."""
        dm = np.array([1000, 100, 10, 5, 4.5, 4.4, 4.39])
        rn = np.array([0.001, 0.01, 0.1, 1.0, 10, 50, 100])
        idx, score = kneedle_corner(dm, rn)
        assert 0 <= idx < len(dm)
        assert score > 0

    def test_fewer_than_3_points(self):
        dm = np.array([10, 5])
        rn = np.array([0.1, 1.0])
        idx, score = kneedle_corner(dm, rn)
        assert idx == -1
        assert score == 0.0

    def test_single_point(self):
        idx, score = kneedle_corner(np.array([1.0]), np.array([1.0]))
        assert idx == -1

    def test_constant_values(self):
        """All same values → no corner."""
        dm = np.array([5.0, 5.0, 5.0, 5.0])
        rn = np.array([1.0, 1.0, 1.0, 1.0])
        idx, score = kneedle_corner(dm, rn)
        assert idx == -1

    def test_index_within_range(self):
        rng = np.random.default_rng(42)
        dm = np.sort(rng.uniform(1, 100, 10))[::-1]
        rn = np.sort(rng.uniform(0.01, 10, 10))
        idx, score = kneedle_corner(dm, rn)
        assert -1 <= idx < 10


class TestLCurveSweep:
    """Test lcurve_sweep sequential sweep."""

    @pytest.mark.slow
    def test_basic_sweep(self, small_problem):
        sp = small_problem
        lambdas = np.array([1e-5, 1e-3, 1e-1])
        result = lcurve_sweep(
            phase=jnp.array(sp["phase"]),
            mask=jnp.array(sp["mask"]),
            pixel_size_nm=sp["voxel_size"],
            lambdas=lambdas,
            b0_tesla=sp["b0"],
            solver="newton_cg",
            rdfc_kernel=sp["kernel"],
        )
        assert isinstance(result, LCurveResult)
        assert len(result.data_misfits) == 3
        assert len(result.reg_norms) == 3
        assert result.magnetizations.shape[0] == 3
        assert result.ramp_coeffs.shape[0] == 3

    @pytest.mark.slow
    def test_warm_start_sorts_lambdas(self, small_problem):
        sp = small_problem
        lambdas = np.array([1e-1, 1e-5, 1e-3])  # unsorted
        result = lcurve_sweep(
            phase=jnp.array(sp["phase"]),
            mask=jnp.array(sp["mask"]),
            pixel_size_nm=sp["voxel_size"],
            lambdas=lambdas,
            b0_tesla=sp["b0"],
            rdfc_kernel=sp["kernel"],
        )
        # Should be sorted
        assert np.all(np.diff(result.lambdas) >= 0)

    @pytest.mark.slow
    def test_all_misfits_are_finite(self, small_problem):
        """All data-misfit and reg-norm values should be finite and non-negative."""
        sp = small_problem
        lambdas = np.array([1e-6, 1e-2, 1.0])
        result = lcurve_sweep(
            phase=jnp.array(sp["phase"]),
            mask=jnp.array(sp["mask"]),
            pixel_size_nm=sp["voxel_size"],
            lambdas=lambdas,
            b0_tesla=sp["b0"],
            rdfc_kernel=sp["kernel"],
        )
        assert np.all(np.isfinite(result.data_misfits))
        assert np.all(np.isfinite(result.reg_norms))
        assert np.all(result.data_misfits >= 0)
        assert np.all(result.reg_norms >= 0)


# ===================================================================
# 11. Solver configuration dataclasses
# ===================================================================


class TestSolverConfigs:
    def test_newton_cg_defaults(self):
        cfg = NewtonCGConfig()
        assert cfg.cg_maxiter == 1000
        assert cfg.cg_tol == 1e-10

    def test_adam_defaults(self):
        cfg = AdamConfig()
        assert cfg.num_steps == 2000
        assert cfg.learning_rate == 1e-2
        assert cfg.patience == 50

    def test_lbfgs_defaults(self):
        cfg = LBFGSConfig()
        assert cfg.num_steps == 500
        assert cfg.patience == 50

    def test_configs_are_frozen(self):
        cfg = NewtonCGConfig()
        with pytest.raises(Exception):
            cfg.cg_maxiter = 100


# ===================================================================
# 12. Elementary phase kernel
# ===================================================================


class TestElementaryPhase:
    def test_disc_zero_at_origin(self):
        """disc kernel should be zero at (0, 0)."""
        n = jnp.array([[0.0]])
        m = jnp.array([[0.0]])
        val = _rdfc_elementary_phase("disc", n, m)
        assert_allclose(np.asarray(val).item(), 0.0, atol=1e-15)

    def test_disc_antisymmetric(self):
        """disc kernel m/(n²+m²) is odd in m."""
        n = jnp.array([[1.0]])
        m_pos = jnp.array([[2.0]])
        m_neg = jnp.array([[-2.0]])
        val_pos = np.asarray(_rdfc_elementary_phase("disc", n, m_pos)).item()
        val_neg = np.asarray(_rdfc_elementary_phase("disc", n, m_neg)).item()
        assert_allclose(val_pos, -val_neg, atol=1e-15)

    def test_slab_runs(self):
        n = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        m = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        val = _rdfc_elementary_phase("slab", n, m)
        assert val.shape == (2, 2)

    def test_invalid_geometry_raises(self):
        with pytest.raises(ValueError, match="Unknown geometry"):
            _rdfc_elementary_phase("cube", jnp.array([[0.0]]), jnp.array([[0.0]]))


# ===================================================================
# 13. Result tuple types
# ===================================================================


class TestResultTypes:
    def test_solver_result_fields(self):
        r = SolverResult(
            magnetization=jnp.zeros((4, 4, 2)),
            ramp_coeffs=jnp.zeros(3),
            loss_history=jnp.zeros(10),
        )
        assert r.magnetization.shape == (4, 4, 2)
        assert r.ramp_coeffs.shape == (3,)
        assert r.loss_history.shape == (10,)

    def test_lcurve_result_fields(self):
        r = LCurveResult(
            lambdas=np.zeros(3),
            data_misfits=np.zeros(3),
            reg_norms=np.zeros(3),
            magnetizations=jnp.zeros((3, 4, 4, 2)),
            ramp_coeffs=jnp.zeros((3, 3)),
            corner_index=1,
        )
        assert r.corner_index == 1
        assert len(r.lambdas) == 3


# ===================================================================
# 14. Upstream fixture parity — direct tests against empyre/pyramid
#     HDF5 and .npy files (not intermediate .npz extracts)
# ===================================================================

# Helpers to load HDF5 array data from pyramid fixtures
try:
    import h5py as _h5py

    def _load_hdf5_field(path):
        """Load the main data array from a pyramid HDF5 file."""
        with _h5py.File(path, "r") as h:
            return h["Experiments/__unnamed__/data"][:]

    def _load_hdf5_scale(path, axis_idx=2):
        """Load the voxel-size scale from a pyramid HDF5 axis attribute."""
        with _h5py.File(path, "r") as h:
            return float(
                h[f"Experiments/__unnamed__/axis-{axis_idx}"].attrs["scale"]
            )

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

_skip_no_h5py = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")

PYRAMID_KERNEL_DIR = os.path.join(UPSTREAM_DIR, "kernel")
PYRAMID_FORWARDMODEL_DIR = os.path.join(UPSTREAM_DIR, "forwardmodel")


class TestUpstreamKernelNpy:
    """Verify build_rdfc_kernel directly against upstream kernel .npy files."""

    def test_u_fft_matches_upstream_npy(self):
        ref = np.load(os.path.join(PYRAMID_KERNEL_DIR, "ref_u_fft.npy"))
        kernel = build_rdfc_kernel((4, 4), b0_tesla=1.0, geometry="disc")
        assert_allclose(
            np.asarray(kernel["u_fft"]), ref, atol=1e-7,
            err_msg="u_fft does not match upstream test_kernel/ref_u_fft.npy",
        )

    def test_v_fft_matches_upstream_npy(self):
        ref = np.load(os.path.join(PYRAMID_KERNEL_DIR, "ref_v_fft.npy"))
        kernel = build_rdfc_kernel((4, 4), b0_tesla=1.0, geometry="disc")
        assert_allclose(
            np.asarray(kernel["v_fft"]), ref, atol=1e-7,
            err_msg="v_fft does not match upstream test_kernel/ref_v_fft.npy",
        )

    def test_raw_u_matches_upstream_npy(self):
        ref = np.load(os.path.join(PYRAMID_KERNEL_DIR, "ref_u.npy"))
        u_coords = jnp.linspace(-3, 3, num=7, dtype=jnp.float64)
        v_coords = jnp.linspace(-3, 3, num=7, dtype=jnp.float64)
        uu, vv = jnp.meshgrid(u_coords, v_coords, indexing="xy")
        coeff = 1.0 / (2 * PHI_0_T_NM2)
        u_kernel = coeff * _rdfc_elementary_phase("disc", uu, vv)
        assert_allclose(
            np.asarray(u_kernel), ref, atol=1e-7,
            err_msg="Raw u kernel does not match upstream test_kernel/ref_u.npy",
        )

    def test_raw_v_matches_upstream_npy(self):
        ref = np.load(os.path.join(PYRAMID_KERNEL_DIR, "ref_v.npy"))
        u_coords = jnp.linspace(-3, 3, num=7, dtype=jnp.float64)
        v_coords = jnp.linspace(-3, 3, num=7, dtype=jnp.float64)
        uu, vv = jnp.meshgrid(u_coords, v_coords, indexing="xy")
        coeff = 1.0 / (2 * PHI_0_T_NM2)
        v_kernel = -coeff * _rdfc_elementary_phase("disc", vv, uu)
        assert_allclose(
            np.asarray(v_kernel), ref, atol=1e-7,
            err_msg="Raw v kernel does not match upstream test_kernel/ref_v.npy",
        )


class TestUpstreamPhaseMapperHDF5:
    """End-to-end test: load mag_proj and phasemap from HDF5 and verify."""

    @_skip_no_h5py
    def test_phasemap_from_hdf5_mag_proj(self):
        """Load magnetisation from mag_proj.hdf5, compute phase, compare to phasemap.hdf5."""
        mag_field = _load_hdf5_field(
            os.path.join(PYRAMID_PHASEMAPPER_DIR, "mag_proj.hdf5")
        )  # (3, 1, 4, 4)
        phase_ref = _load_hdf5_field(
            os.path.join(PYRAMID_PHASEMAPPER_DIR, "phasemap.hdf5")
        )  # (4, 4)
        voxel_size = _load_hdf5_scale(
            os.path.join(PYRAMID_PHASEMAPPER_DIR, "mag_proj.hdf5")
        )

        u_field = jnp.array(mag_field[0, 0], dtype=jnp.float64)
        v_field = jnp.array(mag_field[1, 0], dtype=jnp.float64)
        kernel = build_rdfc_kernel(u_field.shape, b0_tesla=1.0, geometry="disc")
        phase = voxel_size**2 * phase_mapper_rdfc(u_field, v_field, kernel)

        assert_allclose(
            np.asarray(phase), phase_ref, atol=1e-7,
            err_msg="Phase from HDF5 mag_proj does not match phasemap.hdf5",
        )

    @_skip_no_h5py
    def test_forward_model_2d_from_hdf5(self):
        """forward_model_2d with HDF5 inputs should match phasemap.hdf5."""
        mag_field = _load_hdf5_field(
            os.path.join(PYRAMID_PHASEMAPPER_DIR, "mag_proj.hdf5")
        )
        phase_ref = _load_hdf5_field(
            os.path.join(PYRAMID_PHASEMAPPER_DIR, "phasemap.hdf5")
        )
        voxel_size = _load_hdf5_scale(
            os.path.join(PYRAMID_PHASEMAPPER_DIR, "mag_proj.hdf5")
        )

        u = jnp.array(mag_field[0, 0], dtype=jnp.float64)
        v = jnp.array(mag_field[1, 0], dtype=jnp.float64)
        mag = jnp.stack([u, v], axis=-1)
        phase = forward_model_2d(mag, voxel_size, b0_tesla=1.0, geometry="disc")

        assert_allclose(
            np.asarray(phase), phase_ref, atol=1e-7,
            err_msg="forward_model_2d does not match phasemap.hdf5",
        )

    @_skip_no_h5py
    def test_jac_transpose_matches_upstream(self):
        """Transposed Jacobian should equal upstream jac.npy.T (RDFC jac_T_dot test)."""
        jac_ref = np.load(os.path.join(PYRAMID_PHASEMAPPER_DIR, "jac.npy"))
        voxel_size = _load_hdf5_scale(
            os.path.join(PYRAMID_PHASEMAPPER_DIR, "mag_proj.hdf5")
        )
        H, W = 4, 4
        n = 2 * H * W
        kernel = build_rdfc_kernel((H, W), b0_tesla=1.0)

        def phase_from_mag_vec(mag_vec):
            u = mag_vec[:H * W].reshape(H, W)
            v = mag_vec[H * W:].reshape(H, W)
            return (voxel_size**2 * phase_mapper_rdfc(u, v, kernel)).reshape(-1)

        jac = np.asarray(
            jax.jacfwd(phase_from_mag_vec)(jnp.zeros(n, dtype=jnp.float64))
        )
        # Transpose parity (mirrors pyramid's test_PhaseMapperRDFC_jac_T_dot)
        assert_allclose(
            jac.T, jac_ref.T, atol=1e-7,
            err_msg="Transposed Jacobian does not match upstream jac.npy.T",
        )


class TestUpstreamProjectorHDF5:
    """Load ref_magdata.hdf5, project, compare against ref_mag_proj_*.hdf5."""

    @_skip_no_h5py
    @pytest.mark.parametrize("axis", ["z", "y", "x"])
    def test_projection_matches_hdf5(self, axis):
        magdata = _load_hdf5_field(
            os.path.join(PYRAMID_PROJECTOR_DIR, "ref_magdata.hdf5")
        )  # (3, 6, 5, 4)
        proj_ref = _load_hdf5_field(
            os.path.join(PYRAMID_PROJECTOR_DIR, f"ref_mag_proj_{axis}.hdf5")
        )  # (3, 1, V, U)

        mag_3d = jnp.array(
            np.transpose(magdata, (1, 2, 3, 0)), dtype=jnp.float64
        )
        result = project_3d(mag_3d, axis=axis)
        result_np = np.asarray(result)

        # proj_ref layout: (comp, 1, V, U) where comp 0=u, 1=v
        assert_allclose(
            result_np[..., 0], proj_ref[0, 0], atol=1e-5,
            err_msg=f"u-component projection mismatch for axis={axis} vs HDF5",
        )
        assert_allclose(
            result_np[..., 1], proj_ref[1, 0], atol=1e-5,
            err_msg=f"v-component projection mismatch for axis={axis} vs HDF5",
        )

    @pytest.mark.parametrize(
        "axis,jac_name",
        [("z", "jac_z.npy"), ("y", "jac_y.npy"), ("x", "jac_x.npy")],
    )
    def test_projector_jac_T_matches_upstream(self, axis, jac_name):
        """Transposed projector Jacobian should match jac_*.npy.T."""
        jac_ref = np.load(os.path.join(PYRAMID_PROJECTOR_DIR, jac_name))
        Z, Y, X = 6, 5, 4
        n = 3 * Z * Y * X

        def project_from_vec(vec_flat):
            mag_comp_first = vec_flat.reshape(3, Z, Y, X)
            mag_ours = jnp.transpose(mag_comp_first, (1, 2, 3, 0))
            proj = project_3d(mag_ours, axis=axis)
            return jnp.transpose(proj, (2, 0, 1)).reshape(-1)

        jac = np.asarray(
            jax.jacfwd(project_from_vec)(jnp.zeros(n, dtype=jnp.float64))
        )
        assert_allclose(
            jac.T, jac_ref.T, atol=1e-7,
            err_msg=f"Transposed projector Jacobian mismatch for axis={axis}",
        )


class TestUpstreamForwardModelChain:
    """Full 3D forward-model chain: mask → project → phasemap.

    Pyramid's ForwardModel composes SimpleProjector(z) + PhaseMapperRDFC
    on masked 3D magnetization. We replicate this chain and compare
    the Jacobian against upstream test_forwardmodel/jac.npy.

    Setup (matching pyramid test_forwardmodel.py):
      a = 10, dim = (4,5,6), mask[1:-1,1:-1,1:-1] = True
      n = 3 * mask.sum() = 72   (masked magnetization DOF)
      m = 2 * 30 = 60           (two appended views, each 5×6=30 pixels)
      Top 30 rows == bottom 30 rows (same projector+phasemapper twice)
    """

    @_skip_no_h5py
    def test_forward_chain_jac_matches_upstream(self):
        """Single-view Jacobian of (mask→project_z→phase_mapper) vs jac.npy[:30,:]."""
        jac_ref_full = np.load(os.path.join(PYRAMID_FORWARDMODEL_DIR, "jac.npy"))
        # Both views are identical; use only the first 30 rows
        jac_ref = jac_ref_full[:30, :]  # (30, 72)

        Z, Y, X = 4, 5, 6
        mask_3d = np.zeros((Z, Y, X), dtype=bool)
        mask_3d[1:-1, 1:-1, 1:-1] = True
        n_masked = int(mask_3d.sum())  # 24
        assert n_masked == 24
        n = 3 * n_masked  # 72

        voxel_size = 10.0
        kernel = build_rdfc_kernel((Y, X), b0_tesla=1.0, geometry="disc")

        # Indices of masked voxels for scatter
        mask_flat = mask_3d.ravel()
        masked_indices = np.where(mask_flat)[0]

        def forward_chain(mag_vec_masked):
            """Replicate pyramid's ForwardModel.__call__ for a single view.

            mag_vec_masked has shape (72,) = (3 * 24):
              first 24 = mx at masked voxels
              next 24  = my at masked voxels
              last 24  = mz at masked voxels
            """
            mx_masked = mag_vec_masked[:n_masked]
            my_masked = mag_vec_masked[n_masked : 2 * n_masked]
            mz_masked = mag_vec_masked[2 * n_masked :]

            # Scatter into full 3D volumes
            total = Z * Y * X
            mx_full = jnp.zeros(total, dtype=jnp.float64).at[masked_indices].set(mx_masked)
            my_full = jnp.zeros(total, dtype=jnp.float64).at[masked_indices].set(my_masked)
            mz_full = jnp.zeros(total, dtype=jnp.float64).at[masked_indices].set(mz_masked)

            mx_3d = mx_full.reshape(Z, Y, X)
            my_3d = my_full.reshape(Z, Y, X)
            mz_3d = mz_full.reshape(Z, Y, X)

            mag_3d = jnp.stack([mx_3d, my_3d, mz_3d], axis=-1)  # (Z,Y,X,3)

            # Project along z (SimpleProjector default)
            proj = project_3d(mag_3d, axis="z")  # (Y, X, 2)
            u_proj = proj[..., 0]
            v_proj = proj[..., 1]

            # Phase mapper
            phase = voxel_size**2 * phase_mapper_rdfc(u_proj, v_proj, kernel)
            return phase.reshape(-1)

        jac = np.asarray(
            jax.jacfwd(forward_chain)(jnp.zeros(n, dtype=jnp.float64))
        )
        assert jac.shape == (30, 72)

        assert_allclose(
            jac, jac_ref, atol=1e-7,
            err_msg=(
                "Full chain Jacobian (mask→project_z→phase_mapper) "
                "does not match upstream test_forwardmodel/jac.npy"
            ),
        )

    @_skip_no_h5py
    def test_forward_chain_call_matches_phasemap_ref(self):
        """ForwardModel(ones_vector) should reproduce the reference phasemap."""
        phase_ref = _load_hdf5_field(
            os.path.join(PYRAMID_FORWARDMODEL_DIR, "phasemap_ref.hdf5")
        )  # (5, 6)

        Z, Y, X = 4, 5, 6
        mask_3d = np.zeros((Z, Y, X), dtype=bool)
        mask_3d[1:-1, 1:-1, 1:-1] = True
        n_masked = int(mask_3d.sum())

        voxel_size = 10.0
        kernel = build_rdfc_kernel((Y, X), b0_tesla=1.0, geometry="disc")
        masked_indices = np.where(mask_3d.ravel())[0]

        # ones vector (like costfunction test: self.cost(np.ones(self.cost.n)) ≈ 0)
        mag_vec = jnp.ones(3 * n_masked, dtype=jnp.float64)

        mx_masked = mag_vec[:n_masked]
        my_masked = mag_vec[n_masked : 2 * n_masked]
        mz_masked = mag_vec[2 * n_masked :]

        total = Z * Y * X
        mx_full = jnp.zeros(total, dtype=jnp.float64).at[masked_indices].set(mx_masked)
        my_full = jnp.zeros(total, dtype=jnp.float64).at[masked_indices].set(my_masked)
        mz_full = jnp.zeros(total, dtype=jnp.float64).at[masked_indices].set(mz_masked)

        mag_3d = jnp.stack(
            [mx_full.reshape(Z, Y, X), my_full.reshape(Z, Y, X), mz_full.reshape(Z, Y, X)],
            axis=-1,
        )

        proj = project_3d(mag_3d, axis="z")
        phase = voxel_size**2 * phase_mapper_rdfc(proj[..., 0], proj[..., 1], kernel)

        assert_allclose(
            np.asarray(phase), phase_ref, atol=1e-7,
            err_msg="Forward chain on ones vector does not match phasemap_ref.hdf5",
        )


# ===================================================================
# 15. Analytic phase formulas  (cf. pyramid analytic.py / test_analytic.py)
#
#     Pure-numpy reimplementations of the four analytic phase functions
#     from pyramid.analytic, verified against the upstream .npy reference
#     files.  These serve as ground-truth phase images for validating
#     the numerical forward model (RDFC) on simple geometries.
# ===================================================================

PYRAMID_ANALYTIC_DIR = os.path.join(UPSTREAM_DIR, "analytic")

# Magnetic flux quantum in T·nm² (same constant as pyramid.analytic.PHI_0)
_PHI_0_ANALYTIC = 2067.83


def _analytic_phase_slab(dim, a, phi, center, width, b_0=1.0):
    """Analytic magnetic phase for a homogeneously magnetized slab."""
    z_dim, y_dim, x_dim = dim
    y0, x0 = a * center[1], a * center[2]
    Lz, Ly, Lx = a * width[0], a * width[1], a * width[2]
    coeff = -b_0 / (4 * _PHI_0_ANALYTIC)

    def _F_0(x, y):
        A = np.log(x**2 + y**2 + 1e-30)
        B = np.arctan(x / (y + 1e-30))
        return x * A - 2 * x + 2 * y * B

    x = np.linspace(a / 2, x_dim * a - a / 2, num=x_dim)
    y = np.linspace(a / 2, y_dim * a - a / 2, num=y_dim)
    xx, yy = np.meshgrid(x, y)

    phase = coeff * Lz * (
        -np.cos(phi) * (
            _F_0(xx - x0 - Lx / 2, yy - y0 - Ly / 2)
            - _F_0(xx - x0 + Lx / 2, yy - y0 - Ly / 2)
            - _F_0(xx - x0 - Lx / 2, yy - y0 + Ly / 2)
            + _F_0(xx - x0 + Lx / 2, yy - y0 + Ly / 2)
        )
        + np.sin(phi) * (
            _F_0(yy - y0 - Ly / 2, xx - x0 - Lx / 2)
            - _F_0(yy - y0 + Ly / 2, xx - x0 - Lx / 2)
            - _F_0(yy - y0 - Ly / 2, xx - x0 + Lx / 2)
            + _F_0(yy - y0 + Ly / 2, xx - x0 + Lx / 2)
        )
    )
    return phase


def _analytic_phase_disc(dim, a, phi, center, radius, height, b_0=1.0):
    """Analytic magnetic phase for a homogeneously magnetized disc."""
    z_dim, y_dim, x_dim = dim
    y0, x0 = a * center[1], a * center[2]
    Lz = a * height
    R = a * radius
    coeff = np.pi * b_0 / (2 * _PHI_0_ANALYTIC)

    x = np.linspace(a / 2, x_dim * a - a / 2, num=x_dim)
    y = np.linspace(a / 2, y_dim * a - a / 2, num=y_dim)
    xx, yy = np.meshgrid(x, y)

    r = np.hypot(xx - x0, yy - y0)
    result = coeff * Lz * ((yy - y0) * np.cos(phi) - (xx - x0) * np.sin(phi))
    result *= np.where(r <= R, 1, (R / (r + 1e-30)) ** 2)
    return result


def _analytic_phase_sphere(dim, a, phi, center, radius, b_0=1.0):
    """Analytic magnetic phase for a homogeneously magnetized sphere."""
    z_dim, y_dim, x_dim = dim
    y0, x0 = a * center[1], a * center[2]
    R = a * radius
    coeff = 2.0 / 3.0 * np.pi * b_0 / _PHI_0_ANALYTIC

    x = np.linspace(a / 2, x_dim * a - a / 2, num=x_dim)
    y = np.linspace(a / 2, y_dim * a - a / 2, num=y_dim)
    xx, yy = np.meshgrid(x, y)

    r = np.hypot(xx - x0, yy - y0)
    result = coeff * R**3 / (r + 1e-30) ** 2 * (
        (yy - y0) * np.cos(phi) - (xx - x0) * np.sin(phi)
    )
    result *= 1 - np.clip(1 - (r / R) ** 2, 0, 1) ** (3.0 / 2.0)
    return result


def _analytic_phase_vortex(dim, a, center, radius, height, b_0=1.0):
    """Analytic magnetic phase for a vortex state disc."""
    z_dim, y_dim, x_dim = dim
    y0, x0 = a * center[1], a * center[2]
    Lz = a * height
    R = a * radius
    coeff = -np.pi * b_0 * Lz / _PHI_0_ANALYTIC

    x = np.linspace(a / 2, x_dim * a - a / 2, num=x_dim)
    y = np.linspace(a / 2, y_dim * a - a / 2, num=y_dim)
    xx, yy = np.meshgrid(x, y)

    r = np.hypot(xx - x0, yy - y0)
    return coeff * np.where(r <= R, r - R, 0)


class TestAnalyticPhase:
    """Verify analytic phase formulas against upstream pyramid references.

    Parameters match pyramid's test_analytic.py:
      dim=(4,4,4), a=10.0, phi=pi/4, center=(2,2,2), radius=1
    """

    dim = (4, 4, 4)
    a = 10.0
    phi = np.pi / 4
    center = (2, 2, 2)
    radius = 1

    def test_slab_matches_upstream(self):
        width = (self.dim[0] / 2, self.dim[1] / 2, self.dim[2] / 2)
        phase = _analytic_phase_slab(self.dim, self.a, self.phi, self.center, width)
        ref = np.load(os.path.join(PYRAMID_ANALYTIC_DIR, "ref_phase_slab.npy"))
        assert_allclose(
            phase, ref, atol=1e-10,
            err_msg="Analytic slab phase does not match upstream reference",
        )

    def test_disc_matches_upstream(self):
        height = self.dim[2] / 2
        phase = _analytic_phase_disc(
            self.dim, self.a, self.phi, self.center, self.radius, height,
        )
        ref = np.load(os.path.join(PYRAMID_ANALYTIC_DIR, "ref_phase_disc.npy"))
        assert_allclose(
            phase, ref, atol=1e-10,
            err_msg="Analytic disc phase does not match upstream reference",
        )

    def test_sphere_matches_upstream(self):
        phase = _analytic_phase_sphere(
            self.dim, self.a, self.phi, self.center, self.radius,
        )
        ref = np.load(os.path.join(PYRAMID_ANALYTIC_DIR, "ref_phase_sphere.npy"))
        assert_allclose(
            phase, ref, atol=1e-10,
            err_msg="Analytic sphere phase does not match upstream reference",
        )

    def test_vortex_matches_upstream(self):
        height = self.dim[2] / 2
        phase = _analytic_phase_vortex(
            self.dim, self.a, self.center, self.radius, height,
        )
        ref = np.load(os.path.join(PYRAMID_ANALYTIC_DIR, "ref_phase_vort.npy"))
        assert_allclose(
            phase, ref, atol=1e-10,
            err_msg="Analytic vortex phase does not match upstream reference",
        )

    def test_slab_shape(self):
        width = (2, 2, 2)
        phase = _analytic_phase_slab(self.dim, self.a, self.phi, self.center, width)
        assert phase.shape == (self.dim[1], self.dim[2])

    def test_disc_symmetry(self):
        """Disc phase should be antisymmetric about the center."""
        dim = (8, 8, 8)
        center = (4, 4, 4)
        phase = _analytic_phase_disc(dim, 1.0, 0.0, center, 2, 4)
        # phi=0 → cos(0)=1, sin(0)=0 → phase ∝ (y - y0)
        # Should be antisymmetric in y about center
        assert_allclose(
            phase[3, :], -phase[4, :], atol=1e-12,
            err_msg="Disc phase should be antisymmetric about center in y",
        )

    def test_vortex_zero_outside(self):
        """Vortex phase should be exactly zero outside the disc radius."""
        dim = (16, 16, 16)
        center = (8, 8, 8)
        phase = _analytic_phase_vortex(dim, 1.0, center, 3, 4)
        # Pixels far from center (r > radius) should be zero
        y0, x0 = center[1], center[2]
        ys = np.arange(dim[1]) + 0.5
        xs = np.arange(dim[2]) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        r = np.hypot(xx - x0, yy - y0)
        outside = r > 3
        assert_allclose(
            phase[outside], 0.0, atol=1e-15,
            err_msg="Vortex phase should be zero outside the disc",
        )


# ===================================================================
# 16. Analytic round-trip: magnetization recovery via MBIR
#
#     End-to-end tests that verify reconstruct_2d accurately recovers
#     a known projected magnetization.  Any scaling-factor error in the
#     RDFC kernel coefficient, the pixel_size_nm**2 factor, or the b0
#     parameter will cause these tests to fail.
# ===================================================================


@pytest.fixture(scope="module")
def slab_roundtrip():
    """Ground-truth slab projected magnetization, mask, kernel, and phases.

    16×16 grid, centered 8×8 slab of depth 4 in z, magnetized at phi=pi/3.
    Provides both the RDFC forward-model phase and the independent analytic
    slab phase for cross-validation.
    """
    dim = (4, 16, 16)
    a = 10.0
    phi = np.pi / 3
    center = (2, 8, 8)
    width = (4, 8, 8)
    b_0 = 1.0

    z_dim, y_dim, x_dim = dim
    y_lo = int(center[1] - width[1] / 2)
    y_hi = int(center[1] + width[1] / 2)
    x_lo = int(center[2] - width[2] / 2)
    x_hi = int(center[2] + width[2] / 2)

    mask_2d = np.zeros((y_dim, x_dim), dtype=bool)
    mask_2d[y_lo:y_hi, x_lo:x_hi] = True

    depth = width[0]
    gt_proj_mag = np.zeros((y_dim, x_dim, 2), dtype=np.float64)
    gt_proj_mag[..., 0] = mask_2d * depth * np.cos(phi)
    gt_proj_mag[..., 1] = mask_2d * depth * np.sin(phi)

    kernel = build_rdfc_kernel(
        (y_dim, x_dim), b0_tesla=b_0, geometry="slab",
    )

    analytic_phase = _analytic_phase_slab(
        dim, a, phi, center, width, b_0=b_0,
    )

    fwd_phase = np.asarray(forward_model_2d(
        jnp.array(gt_proj_mag), a,
        b0_tesla=b_0, geometry="slab", rdfc_kernel=kernel,
    ))

    return {
        "dim": dim,
        "a": a,
        "phi": phi,
        "center": center,
        "width": width,
        "b_0": b_0,
        "mask_2d": mask_2d,
        "gt_proj_mag": gt_proj_mag,
        "kernel": kernel,
        "analytic_phase": analytic_phase,
        "fwd_phase": fwd_phase,
        "depth": depth,
    }


class TestMBIRAnalyticRoundTrip:
    """End-to-end MBIR recovery tests against analytic slab magnetization.

    A uniformly magnetized rectangular slab has a closed-form phase
    solution (``_analytic_phase_slab``).  These tests verify:

    1. The RDFC forward model reproduces the analytic phase.
    2. ``reconstruct_2d`` inverts its own forward model exactly.
    3. ``reconstruct_2d`` recovers the correct magnetization magnitude
       and direction from the independent analytic phase.
    4. ``forward_model_3d`` (project + RDFC chain) matches the analytic
       phase.

    Failure in any of these indicates a scaling-factor or sign error
    in the reconstruction pipeline.
    """

    def test_forward_model_matches_analytic_slab_phase(self, slab_roundtrip):
        """RDFC slab forward model should reproduce the analytic slab phase.

        Compares on interior pixels (4-pixel margin) to avoid grid-edge
        discretization artefacts.
        """
        fwd = slab_roundtrip["fwd_phase"]
        ana = slab_roundtrip["analytic_phase"]

        y_dim, x_dim = fwd.shape
        interior = np.zeros((y_dim, x_dim), dtype=bool)
        interior[4:-4, 4:-4] = True

        assert_allclose(
            fwd[interior], ana[interior], rtol=0.02,
            err_msg=(
                "RDFC forward-model phase does not match analytic slab phase; "
                "possible scaling error in kernel coefficient or voxel_size factor"
            ),
        )

    def test_mbir_round_trip_from_forward_model(self, slab_roundtrip):
        """MBIR should perfectly invert its own forward model.

        Phase is generated by the same RDFC forward model used in the
        reconstruction, so with near-zero regularisation the recovered
        magnetization should match the ground truth to <1 %.
        """
        s = slab_roundtrip
        result = reconstruct_2d(
            phase=jnp.array(s["fwd_phase"]),
            pixel_size_nm=s["a"],
            b0_tesla=s["b_0"],
            mask=jnp.array(s["mask_2d"]),
            lam=1e-10,
            solver_config=NewtonCGConfig(cg_maxiter=5000, cg_tol=1e-12),
            geometry="slab",
            rdfc_kernel=s["kernel"],
        )

        rec = np.asarray(result.magnetization)
        gt = s["gt_proj_mag"]
        mask = s["mask_2d"]

        # --- scaling check: mean components inside mask ---
        rec_u_mean = rec[mask, 0].mean()
        rec_v_mean = rec[mask, 1].mean()
        gt_u_mean = gt[mask, 0].mean()
        gt_v_mean = gt[mask, 1].mean()

        assert_allclose(
            rec_u_mean, gt_u_mean, rtol=1e-3,
            err_msg=(
                f"Mean Mu inside mask: recovered {rec_u_mean:.6f}, "
                f"expected {gt_u_mean:.6f}"
            ),
        )
        assert_allclose(
            rec_v_mean, gt_v_mean, rtol=1e-3,
            err_msg=(
                f"Mean Mv inside mask: recovered {rec_v_mean:.6f}, "
                f"expected {gt_v_mean:.6f}"
            ),
        )

        # --- pixel-wise inside mask ---
        assert_allclose(
            rec[mask], gt[mask], rtol=0.01,
            err_msg=(
                "MBIR round-trip: pixel-wise magnetization inside mask "
                "deviates >1 % from ground truth"
            ),
        )

        # --- magnitude check ---
        rec_norm = np.sqrt(rec[mask, 0] ** 2 + rec[mask, 1] ** 2)
        gt_norm = np.sqrt(gt[mask, 0] ** 2 + gt[mask, 1] ** 2)
        assert_allclose(
            rec_norm.mean(), gt_norm.mean(), rtol=1e-3,
            err_msg="Mean |M| inside mask does not match ground truth",
        )

    def test_mbir_recovers_magnetization_from_analytic_phase(self, slab_roundtrip):
        """MBIR should recover the correct magnetization from analytic phase.

        The analytic phase is computed by a completely independent code
        path.  Any scaling discrepancy between the RDFC kernel and the
        analytic integral will appear as a systematic bias in the
        recovered magnetization components.
        """
        s = slab_roundtrip
        result = reconstruct_2d(
            phase=jnp.array(s["analytic_phase"]),
            pixel_size_nm=s["a"],
            b0_tesla=s["b_0"],
            mask=jnp.array(s["mask_2d"]),
            lam=1e-10,
            solver_config=NewtonCGConfig(cg_maxiter=5000, cg_tol=1e-12),
            geometry="slab",
            rdfc_kernel=s["kernel"],
        )

        rec = np.asarray(result.magnetization)
        mask = s["mask_2d"]
        depth = s["depth"]
        phi = s["phi"]

        expected_mu = depth * np.cos(phi)
        expected_mv = depth * np.sin(phi)
        expected_norm = depth  # |M| = depth * 1.0

        rec_u_mean = rec[mask, 0].mean()
        rec_v_mean = rec[mask, 1].mean()
        rec_norm_mean = np.sqrt(rec[mask, 0] ** 2 + rec[mask, 1] ** 2).mean()

        assert_allclose(
            rec_u_mean, expected_mu, rtol=0.05,
            err_msg=(
                f"Mean recovered Mu = {rec_u_mean:.6f}, "
                f"expected {expected_mu:.6f} — likely a scaling factor error"
            ),
        )
        assert_allclose(
            rec_v_mean, expected_mv, rtol=0.05,
            err_msg=(
                f"Mean recovered Mv = {rec_v_mean:.6f}, "
                f"expected {expected_mv:.6f} — likely a scaling factor error"
            ),
        )
        assert_allclose(
            rec_norm_mean, expected_norm, rtol=0.05,
            err_msg=(
                f"Mean recovered |M| = {rec_norm_mean:.6f}, "
                f"expected {expected_norm:.6f} — likely a scaling factor error"
            ),
        )

        # Direction should be preserved: recovered angle ≈ phi
        rec_angle = np.arctan2(rec_v_mean, rec_u_mean)
        assert abs(rec_angle - phi) < 0.05, (
            f"Recovered magnetization angle {rec_angle:.4f} rad "
            f"deviates from expected {phi:.4f} rad"
        )

    def test_forward_model_3d_matches_analytic_slab(self, slab_roundtrip):
        """forward_model_3d (project + RDFC) should match the analytic phase.

        Builds the full 3D magnetization array for the slab, passes it
        through ``forward_model_3d`` (which internally calls
        ``project_3d`` then ``forward_model_2d``), and compares the
        result to the closed-form phase.
        """
        s = slab_roundtrip
        dim = s["dim"]
        z_dim, y_dim, x_dim = dim
        center = s["center"]
        width = s["width"]

        z_lo = int(center[0] - width[0] / 2)
        z_hi = int(center[0] + width[0] / 2)
        y_lo = int(center[1] - width[1] / 2)
        y_hi = int(center[1] + width[1] / 2)
        x_lo = int(center[2] - width[2] / 2)
        x_hi = int(center[2] + width[2] / 2)

        mag_3d = np.zeros((z_dim, y_dim, x_dim, 3), dtype=np.float64)
        mag_3d[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi, 0] = np.cos(s["phi"])
        mag_3d[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi, 1] = np.sin(s["phi"])

        phase_3d = np.asarray(forward_model_3d(
            jnp.array(mag_3d), s["a"],
            b0_tesla=s["b_0"], axis="z", geometry="slab",
        ))

        ana = s["analytic_phase"]
        interior = np.zeros((y_dim, x_dim), dtype=bool)
        interior[4:-4, 4:-4] = True

        assert_allclose(
            phase_3d[interior], ana[interior], rtol=0.02,
            err_msg=(
                "3D forward model (project_3d + RDFC) does not match "
                "analytic slab phase"
            ),
        )

    @pytest.mark.parametrize("b0", [0.5, 0.6, 1.5, 2.0])
    def test_mbir_recovers_magnetization_nonunity_b0(self, b0):
        """MBIR recovery must be correct for non-unity b0 values.

        A wrong b0 scaling in either the kernel or the forward model
        would show up as a systematic factor in the recovered
        magnetization.  This test is parametrized over several b0
        values to catch any hidden ``b0 == 1`` special-casing.
        """
        dim = (4, 16, 16)
        a = 10.0
        phi = np.pi / 3
        center = (2, 8, 8)
        width = (4, 8, 8)

        z_dim, y_dim, x_dim = dim
        y_lo = int(center[1] - width[1] / 2)
        y_hi = int(center[1] + width[1] / 2)
        x_lo = int(center[2] - width[2] / 2)
        x_hi = int(center[2] + width[2] / 2)

        mask_2d = np.zeros((y_dim, x_dim), dtype=bool)
        mask_2d[y_lo:y_hi, x_lo:x_hi] = True

        depth = width[0]
        gt_proj_mag = np.zeros((y_dim, x_dim, 2), dtype=np.float64)
        gt_proj_mag[..., 0] = mask_2d * depth * np.cos(phi)
        gt_proj_mag[..., 1] = mask_2d * depth * np.sin(phi)

        kernel = build_rdfc_kernel(
            (y_dim, x_dim), b0_tesla=b0, geometry="slab",
        )

        # Use analytic phase (independent code path) at this b0
        analytic_phase = _analytic_phase_slab(
            dim, a, phi, center, width, b_0=b0,
        )

        result = reconstruct_2d(
            phase=jnp.array(analytic_phase),
            pixel_size_nm=a,
            b0_tesla=b0,
            mask=jnp.array(mask_2d),
            lam=1e-10,
            solver_config=NewtonCGConfig(cg_maxiter=5000, cg_tol=1e-12),
            geometry="slab",
            rdfc_kernel=kernel,
        )

        rec = np.asarray(result.magnetization)
        expected_mu = depth * np.cos(phi)
        expected_mv = depth * np.sin(phi)

        rec_u_mean = rec[mask_2d, 0].mean()
        rec_v_mean = rec[mask_2d, 1].mean()
        rec_norm_mean = np.sqrt(rec[mask_2d, 0] ** 2 + rec[mask_2d, 1] ** 2).mean()

        assert_allclose(
            rec_u_mean, expected_mu, rtol=0.05,
            err_msg=(
                f"b0={b0}: Mean recovered Mu = {rec_u_mean:.6f}, "
                f"expected {expected_mu:.6f}"
            ),
        )
        assert_allclose(
            rec_v_mean, expected_mv, rtol=0.05,
            err_msg=(
                f"b0={b0}: Mean recovered Mv = {rec_v_mean:.6f}, "
                f"expected {expected_mv:.6f}"
            ),
        )
        assert_allclose(
            rec_norm_mean, float(depth), rtol=0.05,
            err_msg=(
                f"b0={b0}: Mean recovered |M| = {rec_norm_mean:.6f}, "
                f"expected {float(depth):.6f}"
            ),
        )


# ===================================================================
# 17. Synthetic round-trip at notebook parameters (a=0.58, b0=0.6)
#
#     Isolates internal consistency of the forward model + reconstruction
#     pipeline at the exact parameters used in the holography notebook.
#     If these tests pass, any discrepancy with pyramid is due to a
#     convention difference, not a bug in the code.
# ===================================================================


class TestMBIRRoundTripNotebookParams:
    """Synthetic round-trip at notebook parameters: a=0.58 nm, b0=0.6 T.

    Uses disc geometry (matching the notebook workflow) and generates
    phase via forward_model_2d, then reconstructs.  If the code has a
    scaling bug that cancels at a=10/b0=1 but not at a=0.58/b0=0.6,
    these tests will catch it.
    """

    @pytest.fixture(scope="class")
    def notebook_problem(self):
        """16x16 disc-geometry problem at a=0.58, b0=0.6."""
        H, W = 16, 16
        a = 0.58        # voxel size in nm (from notebook)
        b0 = 0.6        # Tesla (from notebook)
        phi = np.pi / 4
        gt_norm = 30.0   # known ground-truth |M|

        mask = np.zeros((H, W), dtype=bool)
        mask[4:12, 4:12] = True

        gt_mag = np.zeros((H, W, 2), dtype=np.float64)
        gt_mag[..., 0] = mask * gt_norm * np.cos(phi)
        gt_mag[..., 1] = mask * gt_norm * np.sin(phi)

        kernel = build_rdfc_kernel((H, W), b0_tesla=b0, geometry="disc")

        # Generate phase from the forward model
        fwd_phase = np.asarray(forward_model_2d(
            jnp.array(gt_mag), a,
            b0_tesla=b0, geometry="disc", rdfc_kernel=kernel,
        ))

        return {
            "H": H, "W": W, "a": a, "b0": b0, "phi": phi,
            "gt_norm": gt_norm, "mask": mask, "gt_mag": gt_mag,
            "kernel": kernel, "fwd_phase": fwd_phase,
        }

    def test_round_trip_recovers_magnetization(self, notebook_problem):
        """MBIR should perfectly invert its own forward model at a=0.58, b0=0.6."""
        p = notebook_problem
        result = reconstruct_2d(
            phase=jnp.array(p["fwd_phase"]),
            pixel_size_nm=p["a"],
            b0_tesla=p["b0"],
            mask=jnp.array(p["mask"]),
            lam=1e-10,
            solver_config=NewtonCGConfig(cg_maxiter=5000, cg_tol=1e-12),
            geometry="disc",
            rdfc_kernel=p["kernel"],
        )

        rec = np.asarray(result.magnetization)
        gt = p["gt_mag"]
        mask = p["mask"]

        rec_u_mean = rec[mask, 0].mean()
        rec_v_mean = rec[mask, 1].mean()
        gt_u_mean = gt[mask, 0].mean()
        gt_v_mean = gt[mask, 1].mean()
        rec_norm_mean = np.sqrt(rec[mask, 0] ** 2 + rec[mask, 1] ** 2).mean()

        assert_allclose(
            rec_u_mean, gt_u_mean, rtol=1e-3,
            err_msg=(
                f"Mean Mu: recovered {rec_u_mean:.6f}, expected {gt_u_mean:.6f}"
            ),
        )
        assert_allclose(
            rec_v_mean, gt_v_mean, rtol=1e-3,
            err_msg=(
                f"Mean Mv: recovered {rec_v_mean:.6f}, expected {gt_v_mean:.6f}"
            ),
        )
        assert_allclose(
            rec_norm_mean, p["gt_norm"], rtol=1e-3,
            err_msg=(
                f"Mean |M|: recovered {rec_norm_mean:.6f}, "
                f"expected {p['gt_norm']:.6f}"
            ),
        )

    def test_round_trip_direction_preserved(self, notebook_problem):
        """Recovered magnetization angle should match ground truth."""
        p = notebook_problem
        result = reconstruct_2d(
            phase=jnp.array(p["fwd_phase"]),
            pixel_size_nm=p["a"],
            b0_tesla=p["b0"],
            mask=jnp.array(p["mask"]),
            lam=1e-10,
            solver_config=NewtonCGConfig(cg_maxiter=5000, cg_tol=1e-12),
            geometry="disc",
            rdfc_kernel=p["kernel"],
        )

        rec = np.asarray(result.magnetization)
        mask = p["mask"]
        rec_angle = np.arctan2(rec[mask, 1].mean(), rec[mask, 0].mean())

        assert abs(rec_angle - p["phi"]) < 0.01, (
            f"Recovered angle {rec_angle:.4f} rad, expected {p['phi']:.4f} rad"
        )

    def test_round_trip_pixel_wise(self, notebook_problem):
        """Pixel-wise magnetization inside mask should match ground truth."""
        p = notebook_problem
        result = reconstruct_2d(
            phase=jnp.array(p["fwd_phase"]),
            pixel_size_nm=p["a"],
            b0_tesla=p["b0"],
            mask=jnp.array(p["mask"]),
            lam=1e-10,
            solver_config=NewtonCGConfig(cg_maxiter=5000, cg_tol=1e-12),
            geometry="disc",
            rdfc_kernel=p["kernel"],
        )

        rec = np.asarray(result.magnetization)
        mask = p["mask"]

        assert_allclose(
            rec[mask], p["gt_mag"][mask], rtol=0.01,
            err_msg="Pixel-wise magnetization deviates >1% from ground truth",
        )


# ---------------------------------------------------------------------------
# 18. Cross-validation against pyramid (requires ``pyramid`` package)
# ---------------------------------------------------------------------------
pyramid = pytest.importorskip("pyramid")


class TestPyramidCrossValidation:
    """Compare forward model and reconstruction with pyramid (live)."""

    A = 0.58          # nm
    B0 = 0.6          # T
    NY, NX = 64, 64
    M_AMP = 30.0
    LAM = 1e-3

    @pytest.fixture()
    def disc_problem(self):
        """Synthetic disc with known |M|, phase from both codes."""
        from pyramid.fielddata import VectorData
        from pyramid.kernel import Kernel
        from pyramid.phasemapper import PhaseMapperRDFC

        yy, xx = np.mgrid[: self.NY, : self.NX]
        r = np.sqrt((xx - self.NX // 2) ** 2 + (yy - self.NY // 2) ** 2)
        mask = (r < 15).astype(float)
        Mx = np.where(mask > 0.5, self.M_AMP, 0.0)
        My = np.zeros_like(Mx)

        # pyramid forward model
        kern_pyr = Kernel(self.A, (self.NY, self.NX), b_0=self.B0, geometry="disc")
        mapper_pyr = PhaseMapperRDFC(kern_pyr)
        field = np.zeros((3, 1, self.NY, self.NX))
        field[0, 0] = Mx
        field[1, 0] = My
        phase_pyr = mapper_pyr(VectorData(self.A, field)).phase

        # libertem_holo forward model
        kern_lt = build_rdfc_kernel(
            (self.NY, self.NX), b0_tesla=self.B0, geometry="disc",
        )
        mag_lt = jnp.stack([jnp.array(Mx), jnp.array(My)], axis=-1)
        phase_lt = np.asarray(
            forward_model_single_rdfc_2d(mag_lt, jnp.zeros(3), kern_lt, self.A)
        )

        return {
            "mask": mask,
            "Mx": Mx,
            "phase_pyr": phase_pyr,
            "phase_lt": phase_lt,
            "kern_lt": kern_lt,
        }

    def test_forward_models_match(self, disc_problem):
        """Pyramid and libertem_holo forward models give identical phase."""
        assert_allclose(
            disc_problem["phase_lt"],
            disc_problem["phase_pyr"],
            atol=1e-12,
            err_msg="Forward model phases differ between pyramid and libertem_holo",
        )

    def test_reconstruction_matches_pyramid(self, disc_problem):
        """Both codes recover the same |M| from the same phase."""
        from pyramid.phasemap import PhaseMap
        from pyramid.utils.convenience import reconstruction_2d_from_phasemap

        p = disc_problem
        phase = p["phase_pyr"]
        mask = p["mask"]

        # libertem_holo reconstruction
        result_lt = reconstruct_2d(
            phase=jnp.array(phase),
            pixel_size_nm=self.A,
            b0_tesla=self.B0,
            mask=jnp.array(mask),
            lam=self.LAM,
            solver="newton_cg",
            rdfc_kernel=p["kern_lt"],
        )
        mag_lt = np.asarray(result_lt.magnetization)
        abs_lt = np.sqrt(mag_lt[..., 0] ** 2 + mag_lt[..., 1] ** 2)

        # pyramid reconstruction
        pm = PhaseMap(a=self.A, phase=phase, mask=mask.astype(bool))
        md_rec, _ = reconstruction_2d_from_phasemap(
            pm, b_0=self.B0, lam=self.LAM, max_iter=100, verbose=False,
        )
        abs_pyr = np.sqrt(md_rec.field[0, 0] ** 2 + md_rec.field[1, 0] ** 2)

        inside = mask > 0.5
        assert_allclose(
            abs_lt[inside].mean(),
            abs_pyr[inside].mean(),
            rtol=1e-3,
            err_msg="|M| mean inside mask differs between codes",
        )
        assert_allclose(
            abs_lt[inside].mean(),
            self.M_AMP,
            rtol=1e-3,
            err_msg="Recovered |M| does not match ground truth",
        )


# ---------------------------------------------------------------------------
# 19. Projected-magnetization normalization (3D → 2D round-trip)
# ---------------------------------------------------------------------------
pyramid = pytest.importorskip("pyramid")


class TestProjectedMagnetizationNormalization:
    """Verify that MBIR recovers the projected magnetization,
    and that dividing by the number of voxels along z
    gives back the per-voxel magnetization.

    The synthetic workflow is:
    1. Build a 3D disc with per-voxel |M| = 1.0 and height *H* voxels.
    2. Project along z → projected |M| inside the disc = H.
    3. Compute phase via Pyramid's PhaseMapperRDFC (consistent with MBIR).
    4. Reconstruct via MBIR ``reconstruct_2d``.
    5. Assert that the recovered magnetization matches *H*, not 1.0.

    A separate test checks that the notebook's use of PhaseMapperFDFC
    is the source of any apparent normalization error, because FDFC
    (periodic/circular convolution) and RDFC (zero-padded linear
    convolution) give different phase maps on a finite grid.
    """

    N = 32
    R = 10
    H = 10          # disc height in voxels
    PX = 1.0        # voxel size in nm
    B0 = 1.25       # Tesla
    PHI = np.pi / 2  # magnetization along y

    @pytest.fixture(scope="class")
    def disc_3d(self):
        """Create 3D disc, project, and compute phases."""
        from pyramid.magcreator.shapes import disc as pyr_disc
        from pyramid.magcreator.magcreator import create_mag_dist_homog
        from pyramid.fielddata import VectorData
        from pyramid.projector import SimpleProjector
        from pyramid.phasemapper import PhaseMapperFDFC, PhaseMapperRDFC
        from pyramid.kernel import Kernel

        N, R, H = self.N, self.R, self.H
        dim3 = (N, N, N)
        center = (N // 2, N // 2, N // 2)

        mag_shape = pyr_disc(dim=dim3, center=center, radius=R, height=H)
        field_3d = create_mag_dist_homog(mag_shape=mag_shape, phi=self.PHI)
        mag_distrib = VectorData(a=self.PX, field=field_3d)

        projector = SimpleProjector(mag_distrib.dim, axis="z")
        field_proj = projector(mag_distrib)
        projected_u = field_proj.field[0, 0]
        projected_v = field_proj.field[1, 0]
        mask = (field_proj.get_mask()[0, ...] > 0.5).astype(np.float64)

        # Phase via RDFC (consistent with MBIR forward model)
        kern_pyr = Kernel(self.PX, (N, N), b_0=self.B0, geometry="disc")
        mapper_rdfc = PhaseMapperRDFC(kern_pyr)
        field_2d = np.zeros((3, 1, N, N))
        field_2d[0, 0] = projected_u
        field_2d[1, 0] = projected_v
        phase_rdfc = mapper_rdfc(VectorData(self.PX, field_2d)).phase

        # Phase via FDFC (what the notebook uses — periodic convolution)
        mapper_fdfc = PhaseMapperFDFC(self.PX, projector.dim_uv, b_0=self.B0)
        phase_fdfc = mapper_fdfc(field_proj).phase

        # Phase via MBIR forward model
        kern_lt = build_rdfc_kernel(
            (N, N), b0_tesla=self.B0, geometry="disc",
        )
        mag_2d = np.zeros((N, N, 2))
        mag_2d[..., 0] = projected_u
        mag_2d[..., 1] = projected_v
        phase_mbir = np.asarray(forward_model_2d(
            jnp.array(mag_2d), self.PX,
            b0_tesla=self.B0, rdfc_kernel=kern_lt,
        ))

        return {
            "projected_u": projected_u,
            "projected_v": projected_v,
            "mask": mask,
            "phase_rdfc": phase_rdfc,
            "phase_fdfc": phase_fdfc,
            "phase_mbir": phase_mbir,
            "kern_lt": kern_lt,
        }

    def test_mbir_forward_matches_pyramid_rdfc(self, disc_3d):
        """MBIR and Pyramid RDFC forward models give identical phase."""
        assert_allclose(
            disc_3d["phase_mbir"],
            disc_3d["phase_rdfc"],
            atol=1e-12,
            err_msg="MBIR forward model differs from Pyramid RDFC",
        )

    def test_fdfc_and_rdfc_differ(self, disc_3d):
        """FDFC and RDFC give different phases on a finite grid.

        This documents the known discrepancy: FDFC uses periodic
        (circular) convolution, RDFC uses zero-padded (linear)
        convolution.  On a small grid (N=32, R=10) the difference
        can be very large.
        """
        rel_diff = (
            np.abs(disc_3d["phase_fdfc"] - disc_3d["phase_rdfc"]).max()
            / np.abs(disc_3d["phase_rdfc"]).max()
        )
        # They should differ noticeably — if they don't, this test
        # becomes informational ("good, they've converged").
        assert rel_diff > 0.05, (
            f"FDFC/RDFC phases are unexpectedly close (rel diff {rel_diff:.4f})"
        )

    def test_rdfc_phase_recovers_projected_magnetization(self, disc_3d):
        """MBIR from RDFC-generated phase recovers the projected M = H."""
        p = disc_3d
        result = reconstruct_2d(
            phase=jnp.array(p["phase_rdfc"]),
            pixel_size_nm=self.PX,
            b0_tesla=self.B0,
            mask=jnp.array(p["mask"]),
            lam=1e-10,
            solver_config=NewtonCGConfig(cg_maxiter=5000, cg_tol=1e-12),
            geometry="disc",
            rdfc_kernel=p["kern_lt"],
        )
        rec = np.asarray(result.magnetization)
        inside = p["mask"] > 0.5

        rec_u_mean = rec[inside, 0].mean()
        rec_v_mean = rec[inside, 1].mean()
        rec_norm = np.sqrt(rec[inside, 0] ** 2 + rec[inside, 1] ** 2).mean()

        # True projected values: Mx ≈ 0, My ≈ H = 10
        assert_allclose(rec_u_mean, 0.0, atol=1e-3,
                        err_msg="Projected Mx should be ~0")
        assert_allclose(rec_v_mean, float(self.H), rtol=1e-3,
                        err_msg=f"Projected My should be ~{self.H}")
        assert_allclose(rec_norm, float(self.H), rtol=1e-3,
                        err_msg=f"|M| should match projected value {self.H}")

    def test_projected_magnetization_divided_by_height_recovers_per_voxel(self, disc_3d):
        """Dividing the projected M by the disc height recovers per-voxel M = 1."""
        p = disc_3d
        result = reconstruct_2d(
            phase=jnp.array(p["phase_rdfc"]),
            pixel_size_nm=self.PX,
            b0_tesla=self.B0,
            mask=jnp.array(p["mask"]),
            lam=1e-10,
            solver_config=NewtonCGConfig(cg_maxiter=5000, cg_tol=1e-12),
            geometry="disc",
            rdfc_kernel=p["kern_lt"],
        )
        rec = np.asarray(result.magnetization)
        inside = p["mask"] > 0.5
        rec_norm = np.sqrt(rec[inside, 0] ** 2 + rec[inside, 1] ** 2).mean()

        per_voxel = rec_norm / self.H
        assert_allclose(per_voxel, 1.0, rtol=1e-3,
                        err_msg="projected |M| / H should give per-voxel M = 1")

    def test_fdfc_phase_gives_wrong_magnetization(self, disc_3d):
        """Using FDFC-generated phase with RDFC reconstruction is wrong.

        This test documents the forward-model mismatch you get when the
        phase was generated with Pyramid's PhaseMapperFDFC (periodic
        convolution) but the reconstruction uses the RDFC-based MBIR
        forward model (linear convolution).
        """
        p = disc_3d
        result = reconstruct_2d(
            phase=jnp.array(p["phase_fdfc"]),
            pixel_size_nm=self.PX,
            b0_tesla=self.B0,
            mask=jnp.array(p["mask"]),
            lam=1e-10,
            solver_config=NewtonCGConfig(cg_maxiter=5000, cg_tol=1e-12),
            geometry="disc",
            rdfc_kernel=p["kern_lt"],
        )
        rec = np.asarray(result.magnetization)
        inside = p["mask"] > 0.5
        rec_norm = np.sqrt(rec[inside, 0] ** 2 + rec[inside, 1] ** 2).mean()

        # The mismatch should be large — the recovered M will NOT be H
        rel_error = abs(rec_norm - self.H) / self.H
        assert rel_error > 0.05, (
            f"Expected significant error from FDFC/RDFC mismatch, "
            f"but got only {rel_error*100:.1f}%"
        )

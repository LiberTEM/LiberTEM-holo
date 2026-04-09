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
PYRAMID_DIR = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "empyre-pyramid-master-tests",
    "tests",
)
PYRAMID_PHASEMAPPER_DIR = os.path.join(PYRAMID_DIR, "test_phasemapper")
PYRAMID_PROJECTOR_DIR = os.path.join(PYRAMID_DIR, "test_projector")


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
            voxel_size_nm=sp["voxel_size"],
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
            voxel_size_nm=sp["voxel_size"],
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
            voxel_size_nm=sp["voxel_size"],
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
            voxel_size_nm=sp["voxel_size"],
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
                voxel_size_nm=sp["voxel_size"],
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
                voxel_size_nm=sp["voxel_size"],
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
            voxel_size_nm=sp["voxel_size"],
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
            voxel_size_nm=sp["voxel_size"],
        )
        assert isinstance(result, SolverResult)

    def test_solver_config_overrides_string(self, small_problem):
        sp = small_problem
        result = reconstruct_2d(
            phase=jnp.array(sp["phase"]),
            voxel_size_nm=sp["voxel_size"],
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
            voxel_size_nm=sp["voxel_size"],
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
            voxel_size_nm=sp["voxel_size"],
            lambdas=lambdas,
            rdfc_kernel=sp["kernel"],
            warm_start=True,
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
            voxel_size_nm=sp["voxel_size"],
            lambdas=lambdas,
            rdfc_kernel=sp["kernel"],
            warm_start=True,
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
        assert cfg.cg_maxiter == 10000
        assert cfg.cg_tol == 1e-8

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

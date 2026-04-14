"""Tests for libertem_holo.base.mbir – MBIR magnetic-field reconstruction."""
import numpy as np
import pytest
import unxt

from libertem_holo.base.mbir import (
    ELEMENTARY_CHARGE,
    FLUX_QUANTUM,
    HBAR,
    LCurveResult,
    MBIRResult,
    _find_lcurve_corner,
    b_field_to_phase,
    compute_lcurve,
    phase_scale_factor,
    phase_to_b_field,
    reconstruct_b_field_pyramid,
    reconstruct_b_field_tikhonov,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_phase(ny: int = 32, nx: int = 32, seed: int = 0) -> unxt.Quantity:
    rng = np.random.default_rng(seed)
    return unxt.Quantity(rng.standard_normal((ny, nx)) * 0.5, "rad")


def _default_params():
    return {
        "voxel_size": unxt.Quantity(5.0, "nm"),
        "thickness": unxt.Quantity(50.0, "nm"),
    }


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------


def test_physical_constants_units():
    assert str(ELEMENTARY_CHARGE.unit) == "C"
    assert str(HBAR.unit) == "J s"
    # Flux quantum is T m² (unit string may vary in component order)
    flux_unit_str = str(FLUX_QUANTUM.unit)
    assert "T" in flux_unit_str
    assert "m" in flux_unit_str


def test_physical_constants_values():
    assert pytest.approx(float(ELEMENTARY_CHARGE.value), rel=1e-5) == 1.602176634e-19
    assert pytest.approx(float(HBAR.value), rel=1e-5) == 1.054571817e-34
    assert pytest.approx(float(FLUX_QUANTUM.value), rel=1e-5) == 2.067833848e-15


# ---------------------------------------------------------------------------
# phase_scale_factor
# ---------------------------------------------------------------------------


def test_phase_scale_factor_unit():
    p = _default_params()
    C = phase_scale_factor(**p)
    assert str(C.unit) == "rad / T"


def test_phase_scale_factor_value():
    # C = e * t * pix / hbar
    voxel_m = 5e-9
    thickness_m = 50e-9
    e = 1.602176634e-19
    hbar = 1.054571817e-34
    expected = e * thickness_m * voxel_m / hbar
    p = _default_params()
    C = phase_scale_factor(**p)
    assert pytest.approx(float(C.value), rel=1e-4) == expected


def test_phase_scale_factor_scales_with_thickness():
    voxel = unxt.Quantity(5.0, "nm")
    C1 = phase_scale_factor(voxel_size=voxel, thickness=unxt.Quantity(50.0, "nm"))
    C2 = phase_scale_factor(voxel_size=voxel, thickness=unxt.Quantity(100.0, "nm"))
    assert pytest.approx(float(C2.value), rel=1e-5) == 2 * float(C1.value)


def test_phase_scale_factor_unit_consistency():
    # Check that nm and m inputs give the same result
    voxel_nm = unxt.Quantity(5.0, "nm")
    voxel_m = unxt.Quantity(5e-9, "m")
    thickness_nm = unxt.Quantity(50.0, "nm")
    thickness_m = unxt.Quantity(50e-9, "m")
    C_nm = phase_scale_factor(voxel_size=voxel_nm, thickness=thickness_nm)
    C_m = phase_scale_factor(voxel_size=voxel_m, thickness=thickness_m)
    assert pytest.approx(float(C_nm.value), rel=1e-4) == pytest.approx(
        float(C_m.value), rel=1e-4
    )


# ---------------------------------------------------------------------------
# phase_to_b_field
# ---------------------------------------------------------------------------


def test_phase_to_b_field_returns_quantities():
    phase = _make_phase()
    bx, by = phase_to_b_field(phase, **_default_params())
    assert isinstance(bx, unxt.Quantity)
    assert isinstance(by, unxt.Quantity)


def test_phase_to_b_field_default_unit():
    phase = _make_phase()
    bx, by = phase_to_b_field(phase, **_default_params())
    assert str(bx.unit) == "T"
    assert str(by.unit) == "T"


def test_phase_to_b_field_output_unit_mT():
    phase = _make_phase()
    bx, by = phase_to_b_field(phase, **_default_params(), output_unit="mT")
    assert str(bx.unit) == "mT"
    assert str(by.unit) == "mT"


def test_phase_to_b_field_shape():
    ny, nx = 16, 24
    phase = _make_phase(ny, nx)
    bx, by = phase_to_b_field(phase, **_default_params())
    assert np.asarray(bx.value).shape == (ny, nx)
    assert np.asarray(by.value).shape == (ny, nx)


def test_phase_to_b_field_zero_input():
    ny, nx = 32, 32
    phase = unxt.Quantity(np.zeros((ny, nx)), "rad")
    bx, by = phase_to_b_field(phase, **_default_params())
    assert np.allclose(np.asarray(bx.value), 0, atol=1e-20)
    assert np.allclose(np.asarray(by.value), 0, atol=1e-20)


def test_phase_to_b_field_magnitude():
    """A uniform phase gradient should give a uniform field."""
    ny, nx = 32, 32
    # Linear phase ramp in x → constant By
    slope = 0.1  # rad/pixel
    phase_arr = np.tile(np.arange(nx) * slope, (ny, 1)).astype(float)
    phase = unxt.Quantity(phase_arr, "rad")
    p = _default_params()
    bx, by = phase_to_b_field(phase, **p)
    # Interior should be approximately constant
    bx_inner = np.asarray(bx.value)[2:-2, 2:-2]
    by_inner = np.asarray(by.value)[2:-2, 2:-2]
    assert np.std(bx_inner) < 1e-6 * np.abs(np.mean(bx_inner)) + 1e-10
    assert np.std(by_inner) < 1e-6 * np.abs(np.mean(by_inner)) + 1e-10


# ---------------------------------------------------------------------------
# b_field_to_phase
# ---------------------------------------------------------------------------


def test_b_field_to_phase_returns_quantity():
    ny, nx = 32, 32
    bx = unxt.Quantity(np.zeros((ny, nx)), "T")
    by = unxt.Quantity(np.zeros((ny, nx)), "T")
    phi = b_field_to_phase(bx, by, **_default_params())
    assert isinstance(phi, unxt.Quantity)


def test_b_field_to_phase_zero_input():
    ny, nx = 32, 32
    bx = unxt.Quantity(np.zeros((ny, nx)), "T")
    by = unxt.Quantity(np.zeros((ny, nx)), "T")
    phi = b_field_to_phase(bx, by, **_default_params())
    assert np.allclose(np.asarray(phi.value), 0, atol=1e-20)


def test_b_field_to_phase_unit():
    ny, nx = 32, 32
    bx = unxt.Quantity(np.zeros((ny, nx)), "T")
    by = unxt.Quantity(np.zeros((ny, nx)), "T")
    phi = b_field_to_phase(bx, by, **_default_params())
    assert str(phi.unit) == "rad"


def test_b_field_to_phase_shape():
    ny, nx = 16, 24
    bx = unxt.Quantity(np.zeros((ny, nx)), "T")
    by = unxt.Quantity(np.zeros((ny, nx)), "T")
    phi = b_field_to_phase(bx, by, **_default_params())
    assert np.asarray(phi.value).shape == (ny, nx)


def test_b_field_to_phase_accepts_mT():
    """b_field_to_phase should accept mT inputs."""
    ny, nx = 32, 32
    bx = unxt.Quantity(np.zeros((ny, nx)), "mT")
    by = unxt.Quantity(np.zeros((ny, nx)), "mT")
    phi = b_field_to_phase(bx, by, **_default_params())
    assert isinstance(phi, unxt.Quantity)


# ---------------------------------------------------------------------------
# Forward-inverse consistency
# ---------------------------------------------------------------------------


def test_forward_inverse_roundtrip():
    """b_field → phase → b_field should recover the original field approximately.

    A divergence-free 2D magnetic field (B = curl(Az ẑ)) is guaranteed to be
    fully representable by the scalar phase, so the roundtrip should give high
    correlation.
    """
    ny, nx = 64, 64
    y, x = np.mgrid[-ny // 2 : ny // 2, -nx // 2 : nx // 2]
    sigma = 8.0
    # Divergence-free B: B = curl(Az z-hat) → Bx = dAz/dy, By = -dAz/dx
    az = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    dy_az, dx_az = np.gradient(az)
    bx_true = dy_az
    by_true = -dx_az

    bx_q = unxt.Quantity(bx_true, "T")
    by_q = unxt.Quantity(by_true, "T")
    params = _default_params()

    # Forward
    phi = b_field_to_phase(bx_q, by_q, **params)

    # Inverse (direct gradient method)
    bx_rec, by_rec = phase_to_b_field(phi, **params)

    # High correlation expected for divergence-free field
    corr = np.corrcoef(bx_true.ravel(), np.asarray(bx_rec.value).ravel())[0, 1]
    assert corr > 0.99, f"Forward-inverse correlation too low: {corr}"


# ---------------------------------------------------------------------------
# reconstruct_b_field_tikhonov
# ---------------------------------------------------------------------------


def test_tikhonov_returns_mbir_result():
    phase = _make_phase()
    result = reconstruct_b_field_tikhonov(phase, **_default_params())
    assert isinstance(result, MBIRResult)


def test_tikhonov_result_units():
    phase = _make_phase()
    result = reconstruct_b_field_tikhonov(phase, **_default_params())
    assert str(result.b_x.unit) == "T"
    assert str(result.b_y.unit) == "T"
    assert str(result.b_magnitude.unit) == "T"


def test_tikhonov_output_unit_mT():
    phase = _make_phase()
    result = reconstruct_b_field_tikhonov(
        phase, **_default_params(), output_unit="mT"
    )
    assert str(result.b_x.unit) == "mT"
    assert str(result.b_y.unit) == "mT"
    assert str(result.b_magnitude.unit) == "mT"


def test_tikhonov_result_shape():
    ny, nx = 16, 24
    phase = _make_phase(ny, nx)
    result = reconstruct_b_field_tikhonov(phase, **_default_params())
    assert np.asarray(result.b_x.value).shape == (ny, nx)
    assert np.asarray(result.b_y.value).shape == (ny, nx)
    assert np.asarray(result.b_magnitude.value).shape == (ny, nx)


def test_tikhonov_result_voxel_size():
    phase = _make_phase()
    params = _default_params()
    result = reconstruct_b_field_tikhonov(phase, **params)
    assert result.voxel_size is params["voxel_size"]


def test_tikhonov_result_lambda():
    phase = _make_phase()
    lam = 1.23e-2
    result = reconstruct_b_field_tikhonov(
        phase, **_default_params(), regularization_parameter=lam
    )
    assert result.regularization_parameter == lam


def test_tikhonov_magnitude_non_negative():
    phase = _make_phase()
    result = reconstruct_b_field_tikhonov(phase, **_default_params())
    assert np.all(np.asarray(result.b_magnitude.value) >= 0)


def test_tikhonov_magnitude_equals_norm_of_components():
    phase = _make_phase()
    result = reconstruct_b_field_tikhonov(phase, **_default_params())
    bx = np.asarray(result.b_x.value)
    by = np.asarray(result.b_y.value)
    bmag = np.asarray(result.b_magnitude.value)
    expected = np.sqrt(bx**2 + by**2)
    np.testing.assert_allclose(bmag, expected, rtol=1e-5)


def test_tikhonov_larger_lambda_smoother():
    """Larger regularization should yield smaller solution norm."""
    phase = _make_phase()
    p = _default_params()
    r1 = reconstruct_b_field_tikhonov(phase, **p, regularization_parameter=1e-6)
    r2 = reconstruct_b_field_tikhonov(phase, **p, regularization_parameter=1e0)
    norm1 = np.linalg.norm(np.asarray(r1.b_magnitude.value))
    norm2 = np.linalg.norm(np.asarray(r2.b_magnitude.value))
    assert norm1 > norm2


def test_tikhonov_zero_phase():
    ny, nx = 32, 32
    phase = unxt.Quantity(np.zeros((ny, nx)), "rad")
    result = reconstruct_b_field_tikhonov(phase, **_default_params())
    assert np.allclose(np.asarray(result.b_x.value), 0, atol=1e-20)
    assert np.allclose(np.asarray(result.b_y.value), 0, atol=1e-20)


# ---------------------------------------------------------------------------
# compute_lcurve
# ---------------------------------------------------------------------------


def test_lcurve_returns_result():
    phase = _make_phase()
    lc = compute_lcurve(phase, **_default_params(), n_lambdas=5)
    assert isinstance(lc, LCurveResult)


def test_lcurve_shapes():
    n = 8
    phase = _make_phase()
    lc = compute_lcurve(phase, **_default_params(), n_lambdas=n)
    assert lc.lambdas.shape == (n,)
    assert lc.residual_norms.shape == (n,)
    assert lc.solution_norms.shape == (n,)


def test_lcurve_custom_lambdas():
    phase = _make_phase()
    lambdas = np.array([1e-4, 1e-2, 1e0, 1e2])
    lc = compute_lcurve(phase, **_default_params(), lambdas=lambdas)
    np.testing.assert_array_equal(lc.lambdas, lambdas)
    assert lc.lambdas.shape == (4,)


def test_lcurve_residual_increases_with_lambda():
    """Larger λ → larger residual (less data fidelity)."""
    phase = _make_phase()
    lc = compute_lcurve(
        phase, **_default_params(), n_lambdas=20, lambda_min=1e-6, lambda_max=1e2
    )
    # residuals should be monotonically non-decreasing
    assert np.all(np.diff(lc.residual_norms) >= -1e-6 * lc.residual_norms[:-1])


def test_lcurve_solution_decreases_with_lambda():
    """Larger λ → smaller solution norm."""
    phase = _make_phase()
    lc = compute_lcurve(
        phase, **_default_params(), n_lambdas=20, lambda_min=1e-6, lambda_max=1e2
    )
    assert np.all(np.diff(lc.solution_norms) <= 1e-6 * lc.solution_norms[:-1])


def test_lcurve_optimal_lambda_in_range():
    phase = _make_phase()
    lc = compute_lcurve(phase, **_default_params(), n_lambdas=10)
    assert lc.lambdas[0] <= lc.optimal_lambda <= lc.lambdas[-1]


# ---------------------------------------------------------------------------
# _find_lcurve_corner
# ---------------------------------------------------------------------------


def test_find_lcurve_corner_returns_int():
    r = np.logspace(0, -2, 20)
    s = np.logspace(-2, 0, 20)
    idx = _find_lcurve_corner(r, s)
    assert isinstance(idx, int)
    assert 0 <= idx < 20


def test_find_lcurve_corner_middle():
    """For a perfect L-shape, the corner should be near the middle."""
    n = 20
    mid = n // 2
    r = np.concatenate([np.ones(mid) * 10, np.logspace(1, -2, n - mid)])
    s = np.concatenate([np.logspace(-2, 1, mid), np.ones(n - mid) * 10])
    idx = _find_lcurve_corner(r, s)
    assert abs(idx - mid) <= 3


# ---------------------------------------------------------------------------
# reconstruct_b_field_pyramid
# ---------------------------------------------------------------------------


def test_pyramid_returns_mbir_result():
    phase = _make_phase(64, 64)
    result = reconstruct_b_field_pyramid(phase, **_default_params(), n_levels=2)
    assert isinstance(result, MBIRResult)


def test_pyramid_result_units():
    phase = _make_phase(64, 64)
    result = reconstruct_b_field_pyramid(phase, **_default_params(), n_levels=2)
    assert str(result.b_x.unit) == "T"
    assert str(result.b_y.unit) == "T"
    assert str(result.b_magnitude.unit) == "T"


def test_pyramid_output_unit_mT():
    phase = _make_phase(64, 64)
    result = reconstruct_b_field_pyramid(
        phase, **_default_params(), n_levels=2, output_unit="mT"
    )
    assert str(result.b_x.unit) == "mT"


def test_pyramid_result_shape():
    ny, nx = 64, 64
    phase = _make_phase(ny, nx)
    result = reconstruct_b_field_pyramid(phase, **_default_params(), n_levels=2)
    assert np.asarray(result.b_x.value).shape == (ny, nx)


def test_pyramid_magnitude_non_negative():
    phase = _make_phase(64, 64)
    result = reconstruct_b_field_pyramid(phase, **_default_params(), n_levels=2)
    assert np.all(np.asarray(result.b_magnitude.value) >= 0)


def test_pyramid_single_level_equals_tikhonov():
    """1-level pyramid should equal direct Tikhonov reconstruction."""
    phase = _make_phase(32, 32)
    params = _default_params()
    lam = 1e-3

    r_tik = reconstruct_b_field_tikhonov(phase, **params, regularization_parameter=lam)
    r_pyr = reconstruct_b_field_pyramid(
        phase, **params, n_levels=1, regularization_parameter=lam
    )

    np.testing.assert_allclose(
        np.asarray(r_tik.b_x.value),
        np.asarray(r_pyr.b_x.value),
        rtol=1e-5,
        atol=1e-12,
    )

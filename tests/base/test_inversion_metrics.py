import numpy as np
import pytest

from libertem_holo.base.mbir.inversion import (
    depth_correlation,
    equilibrium_residual,
    iterations_to_threshold,
    mz_rmse,
    phase_residual,
    projected_m_error,
    SmoothnessBackend,
    vortex_core_z_error,
)


def test_phase_residual_zero_for_equal_inputs():
    phi = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert float(phase_residual(phi, phi)) == pytest.approx(0.0)


def test_projected_m_error_zero_for_equal_fields():
    m = np.zeros((4, 3, 2, 3), dtype=np.float32)
    m[..., 0] = 1.0
    assert float(projected_m_error(m, m)) == pytest.approx(0.0)


def test_mz_rmse_matches_expected_value():
    m_true = np.zeros((2, 2, 2, 3), dtype=np.float32)
    m_recon = m_true.copy()
    m_recon[..., 2] = 2.0
    assert float(mz_rmse(m_recon, m_true)) == pytest.approx(2.0)


def test_depth_correlation_is_one_for_identical_profiles():
    m = np.zeros((5, 3, 3, 3), dtype=np.float32)
    m[:, 1, 1, 0] = np.linspace(-1.0, 1.0, 5)
    assert float(depth_correlation(m, m, (1, 1))) == pytest.approx(1.0, abs=1e-6)


def test_vortex_core_z_error_detects_shift():
    m_true = np.zeros((6, 3, 3, 3), dtype=np.float32)
    m_recon = np.zeros_like(m_true)
    m_true[2, 1, 1, 2] = 1.0
    m_recon[4, 1, 1, 2] = 1.0
    assert float(vortex_core_z_error(m_recon, m_true, (1, 1))) == pytest.approx(2.0)


def test_iterations_to_threshold_returns_first_hit_or_length():
    history = np.array([4.0, 2.0, 0.5, 0.25], dtype=np.float32)
    assert iterations_to_threshold(history, 1.0) == 3
    assert iterations_to_threshold(history, 0.1) == len(history)


def test_equilibrium_residual_is_zero_for_uniform_field_under_smoothness_backend():
    rho = np.ones((4, 4, 4), dtype=np.float32)
    m = np.zeros((4, 4, 4, 3), dtype=np.float32)
    m[..., 0] = 1.0

    residual = equilibrium_residual(m, SmoothnessBackend(), rho=rho)

    assert float(residual) == pytest.approx(0.0, abs=1e-6)


def test_equilibrium_residual_is_positive_for_non_equilibrium_field():
    rho = np.ones((4, 4, 4), dtype=np.float32)
    m = np.zeros((4, 4, 4, 3), dtype=np.float32)
    m[1:3, :, :, 0] = 1.0
    m[:1, :, :, 2] = 1.0
    m[3:, :, :, 2] = -1.0

    residual = equilibrium_residual(m, SmoothnessBackend(), rho=rho)

    assert float(residual) > 0.0
from dataclasses import dataclass

import numpy as np
import pytest
import unxt as u

from libertem_holo.base.mbir.inversion import (
    analytic_vortex_init,
    run_with_scaled_rho,
    support_center_yx,
)


@dataclass(frozen=True)
class _DummyReconstruction:
    m_recon: np.ndarray


def test_support_center_yx_finds_disc_center():
    rho = np.zeros((4, 9, 9), dtype=np.float32)
    rho[:, 3:6, 2:7] = 1.0
    assert support_center_yx(rho) == (4, 4)


def test_analytic_vortex_init_matches_rho_shape():
    rho = np.ones((6, 8, 10), dtype=np.float32)
    m0 = np.asarray(analytic_vortex_init(rho))
    assert m0.shape == rho.shape + (3,)
    assert np.all(np.isfinite(m0))


def test_run_with_scaled_rho_returns_histogram_and_mean_abs_m():
    rho = np.ones((4, 4, 4), dtype=np.float32)
    m = np.zeros((4, 4, 4, 3), dtype=np.float32)
    m[..., 0] = 1.0

    def pipeline(phi_scaled, rho_scaled):
        del phi_scaled, rho_scaled
        recon = np.zeros_like(m)
        recon[..., 0] = 0.75
        return _DummyReconstruction(m_recon=recon)

    result = run_with_scaled_rho(
        pipeline,
        rho,
        m,
        u.Quantity(5.0, "nm"),
        scale=1.5,
        histogram_bins=8,
    )

    assert result.rho_scaled.shape == rho.shape
    assert result.m_scaled_truth.shape == m.shape
    assert result.phi_scaled.shape == (4, 4)
    assert result.hist_counts.shape == (8,)
    assert result.hist_edges.shape == (9,)
    assert result.mean_abs_m == pytest.approx(0.75, abs=1e-6)
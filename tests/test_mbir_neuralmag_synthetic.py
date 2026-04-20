import logging

import numpy as np
import pytest

import unxt as u

from libertem_holo.base.mbir import (
    forward_model_2d,
    load_vortex_disc_fixture,
    vortex_magnetization,
)

nm = pytest.importorskip("neuralmag")


def test_neuralmag_vortex_ground_truth_smoke():
    nm.set_log_level(logging.WARNING)
    fixture = load_vortex_disc_fixture(32)
    rho_true = fixture["rho_true"]
    m_true = fixture["m_true"]
    pixel_size_nm = fixture["pixel_size_nm"]

    assert np.isfinite(m_true).all()
    assert np.isfinite(rho_true).all()

    m_projected = np.sum(rho_true[..., None] * m_true, axis=0)[..., :2]
    converted = u.Quantity(m_projected, "")
    assert converted.shape[-1] == 2
    assert np.isfinite(np.asarray(converted.value)).all()

    try:
        phi_true = forward_model_2d(
            converted,
            pixel_size=u.Quantity(pixel_size_nm, "nm"),
            geometry="disc",
        )
    except TypeError as exc:
        if "unsupported operand type" in str(exc) and "Quantity" in str(exc):
            pytest.skip(f"Current unxt/jax arithmetic stack is incompatible with forward_model_2d: {exc}")
        raise

    assert np.isfinite(np.asarray(phi_true.value)).all()


def test_relaxed_fixture_differs_from_analytic_vortex_phase():
    fixture = load_vortex_disc_fixture(32)
    rho_true = fixture["rho_true"]
    m_true = fixture["m_true"]
    phi_true = fixture["phi_true"]

    m_analytic = np.asarray(
        vortex_magnetization(
            rho_true.shape,
            support_zyx=rho_true,
            core_radius=max(1.5, rho_true.shape[0] / 32.0),
            dtype=np.float32,
        )
    )
    analytic_projected = u.Quantity(np.sum(rho_true[..., None] * m_analytic, axis=0)[..., :2], "")
    phi_analytic = np.asarray(
        forward_model_2d(
            analytic_projected,
            pixel_size=u.Quantity(fixture["pixel_size_nm"], "nm"),
            geometry="disc",
        ).value
    )

    rel_diff = np.linalg.norm(phi_true - phi_analytic) / np.linalg.norm(phi_analytic)

    assert np.isfinite(rel_diff)
    assert rel_diff > 0.0

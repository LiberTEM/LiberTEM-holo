import numpy as np
import pytest

import libertem_holo.base.mbir.neuralmag_phase_recovery as neuralmag_phase_recovery
from libertem_holo.base.mbir import (
    NeuralMagPhaseRecoveryConfig,
    make_initial_m_cell,
    prepare_neuralmag_phase_target,
    prepare_neuralmag_phase_target_from_phase_image,
    run_neuralmag_phase_recovery,
    select_anisotropy_orientation_from_phase,
)


def _cube_support(shape=(4, 4, 4)):
    rho = np.zeros(shape, dtype=np.float32)
    rho[1:-1, 1:-1, 1:-1] = 1.0
    m = np.zeros(shape + (3,), dtype=np.float32)
    m[..., 1] = rho
    return rho, m


def test_prepare_neuralmag_phase_target_pads_phase_image_once():
    rho, m = _cube_support((4, 5, 6))
    config = NeuralMagPhaseRecoveryConfig(
        phase_pad=2,
    )

    target = prepare_neuralmag_phase_target(
        rho,
        m,
        cellsize_nm=2.0,
        config=config,
    )

    assert target.rho_xyz.shape == rho.shape
    assert target.m_target_xyz.shape == m.shape
    assert target.rho_zyx_view.shape == (6, 9, 8)
    assert target.phase_target.shape == (9, 8)
    assert target.phase_mask.shape == target.phase_target.shape
    assert np.isfinite(np.asarray(target.phase_target)).all()
    assert np.all(np.asarray(target.phase_mask) == 1.0)


def test_prepare_neuralmag_phase_target_from_phase_image_accepts_phase_only_input():
    rho, m = _cube_support((4, 5, 6))
    config = NeuralMagPhaseRecoveryConfig(
        phase_pad=2,
    )
    reference_target = prepare_neuralmag_phase_target(
        rho,
        m,
        cellsize_nm=2.0,
        config=config,
    )

    target = prepare_neuralmag_phase_target_from_phase_image(
        rho,
        np.asarray(reference_target.phase_target),
        cellsize_nm=2.0,
        config=config,
    )

    assert target.has_reference_magnetization is False
    assert target.rho_xyz.shape == rho.shape
    assert target.phase_target.shape == reference_target.phase_target.shape
    np.testing.assert_allclose(np.asarray(target.phase_target), np.asarray(reference_target.phase_target))
    assert np.count_nonzero(target.m_target_xyz) == 0


def test_make_initial_m_cell_enforces_support_contract():
    rho, m = _cube_support()

    initialized = make_initial_m_cell(
        rho,
        m,
        mode="random",
        rng_seed=1,
    )

    norms = np.linalg.norm(initialized, axis=-1)
    assert np.allclose(norms[rho > 0.5], 1.0, atol=1e-6)
    assert np.count_nonzero(initialized[rho <= 0.5]) == 0


def test_run_neuralmag_phase_recovery_smoke_enforces_support_constraints():
    pytest.importorskip("neuralmag")
    rho, m = _cube_support()
    m0 = make_initial_m_cell(
        rho,
        m,
        mode="random",
        rng_seed=5,
    )
    config = NeuralMagPhaseRecoveryConfig(
        phase_weight_schedule=(1e-4,),
        phase_pad=1,
        phase_energy_scale=1.0,
        minimizer_max_iter=1,
        demag_p=1,
        init_mode="uniform_y",
    )

    result = run_neuralmag_phase_recovery(
        rho,
        m,
        cellsize_nm=2.0,
        config=config,
        m0_cell_xyz=m0,
    )

    assert result.n_iter == 1
    assert len(result.history) == 2
    assert result.history[0]["event"] == "start"
    assert result.history[1]["event"] == "end"
    assert np.isfinite(result.history[0]["phase_rms"])
    assert np.isfinite(result.history[-1]["phase_rms"])
    assert np.isfinite(result.max_g)

    norms = np.linalg.norm(result.m_recovered_xyz, axis=-1)
    assert np.allclose(norms[rho > 0.5], 1.0, atol=1e-6)
    assert np.count_nonzero(result.m_recovered_xyz[rho <= 0.5]) == 0


def test_select_anisotropy_orientation_from_phase_ranks_candidates(monkeypatch):
    rho, m = _cube_support((4, 4, 4))
    config = NeuralMagPhaseRecoveryConfig(
        phase_pad=1,
        init_mode="uniform_y",
        rng_seed=3,
    )
    target = prepare_neuralmag_phase_target(
        rho,
        m,
        cellsize_nm=2.0,
        config=config,
    )

    calls = []
    candidate_m0 = {
        "0, 0, 1 Axis": np.pad(
            np.ones((2, 2, 2, 3), dtype=np.float32) * np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ((1, 1), (1, 1), (1, 1), (0, 0)),
        ),
        "1, 1, 1 Axis": np.pad(
            np.ones((2, 2, 2, 3), dtype=np.float32) * np.array([0.0, 0.0, 1.0], dtype=np.float32),
            ((1, 1), (1, 1), (1, 1), (0, 0)),
        ),
    }

    def fake_run(target_arg, *, config, m0_cell_xyz=None, **kwargs):
        axis1 = np.asarray(config.anisotropy_axis1, dtype=np.float32)
        is_best = bool(np.allclose(axis1, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
        calls.append((axis1.copy(), np.asarray(m0_cell_xyz, dtype=np.float32).copy()))
        recovered = np.asarray(target_arg.m_target_xyz if is_best else np.zeros_like(target_arg.m_target_xyz), dtype=np.float32)
        last = {
            "event": "end",
            "phase_rms": 0.1 if is_best else 0.4,
            "raw_phase_loss": 0.01 if is_best else 0.2,
            "E_total": 1.0 if is_best else 2.0,
        }
        return neuralmag_phase_recovery.NeuralMagPhaseRecoveryResult(
            state=None,
            target=target_arg,
            config=config,
            phase_terms={},
            history=[{"event": "start", **last}, last],
            phase_energy_scale=1.0,
            m_recovered_xyz=recovered,
            phase_recovered=np.asarray(target_arg.phase_target, dtype=np.float32),
            n_iter=1,
            max_g=1.0,
        )

    monkeypatch.setattr(neuralmag_phase_recovery, "run_neuralmag_phase_recovery", fake_run)

    selection = select_anisotropy_orientation_from_phase(
        target,
        candidates={
            "0, 0, 1 Axis": (
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
            "1, 1, 1 Axis": (
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                np.array([1.0, -1.0, 0.0], dtype=np.float32),
            ),
        },
        config=config,
        candidate_m0_cell_xyz=candidate_m0,
    )

    assert selection.best_name == "0, 0, 1 Axis"
    assert tuple(fit.name for fit in selection.fits) == ("0, 0, 1 Axis", "1, 1, 1 Axis")
    np.testing.assert_allclose(selection.best_fit.component_rmse_xyz, 0.0, atol=1e-6)
    assert selection.fits[1].component_rmse_xyz.mean() > 0.0
    assert len(calls) == 2
    assert not np.allclose(calls[0][1], calls[1][1])


def test_select_anisotropy_orientation_from_phase_can_rank_by_initial_phase(monkeypatch):
    rho, m = _cube_support((4, 4, 4))
    config = NeuralMagPhaseRecoveryConfig(
        phase_pad=1,
        init_mode="uniform_y",
        rng_seed=3,
    )
    target = prepare_neuralmag_phase_target(
        rho,
        m,
        cellsize_nm=2.0,
        config=config,
    )
    candidate_m0 = {
        "0, 0, 1 Axis": np.asarray(target.m_target_xyz, dtype=np.float32),
        "1, 1, 1 Axis": np.zeros_like(target.m_target_xyz, dtype=np.float32),
    }

    def fake_run(target_arg, *, config, m0_cell_xyz=None, **kwargs):
        axis1 = np.asarray(config.anisotropy_axis1, dtype=np.float32)
        is_wrong_but_better_final = bool(np.allclose(axis1, np.array([1.0, 1.0, 1.0], dtype=np.float32)))
        last = {
            "event": "end",
            "phase_rms": 0.1 if is_wrong_but_better_final else 0.4,
            "raw_phase_loss": 0.01 if is_wrong_but_better_final else 0.2,
            "E_total": 1.0 if is_wrong_but_better_final else 2.0,
        }
        return neuralmag_phase_recovery.NeuralMagPhaseRecoveryResult(
            state=None,
            target=target_arg,
            config=config,
            phase_terms={},
            history=[{"event": "start", **last}, last],
            phase_energy_scale=1.0,
            m_recovered_xyz=np.asarray(m0_cell_xyz, dtype=np.float32),
            phase_recovered=np.asarray(target_arg.phase_target, dtype=np.float32),
            n_iter=1,
            max_g=1.0,
        )

    monkeypatch.setattr(neuralmag_phase_recovery, "run_neuralmag_phase_recovery", fake_run)

    selection = select_anisotropy_orientation_from_phase(
        target,
        candidates={
            "0, 0, 1 Axis": (
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
            "1, 1, 1 Axis": (
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                np.array([1.0, -1.0, 0.0], dtype=np.float32),
            ),
        },
        config=config,
        candidate_m0_cell_xyz=candidate_m0,
        selection_metric="initial_phase_rms",
    )

    assert selection.best_name == "0, 0, 1 Axis"
    assert selection.best_fit.initial_phase_rms == 0.0

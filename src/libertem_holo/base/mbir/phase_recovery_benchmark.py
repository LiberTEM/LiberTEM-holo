"""Reusable benchmark helpers for phase-constrained NeuralMag recovery."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .fixtures import load_vortex_disc_fixture
from .neuralmag_phase_recovery import (
    calibrate_phase_energy_scale,
    make_initial_m_cell,
    NeuralMagPhaseRecoveryConfig,
    NeuralMagPhaseRecoveryResult,
    NeuralMagPhaseTarget,
    prepare_neuralmag_phase_target,
    run_neuralmag_phase_recovery,
)


@dataclass(frozen=True)
class PhaseRecoveryBenchmarkVariant:
    """One solver configuration to benchmark on a shared phase-recovery case."""

    name: str
    solver_family: str = "neuralmag"
    minimizer_method: str | None = None
    minimizer_update: str | None = None
    minimizer_tau_max: float | None = None
    optax_lbfgs_memory_size: int | None = None
    optax_lbfgs_max_linesearch_steps: int | None = None


@dataclass(frozen=True)
class PhaseRecoveryBenchmarkCase:
    """Prepared target data and baseline config shared across solver variants."""

    target: NeuralMagPhaseTarget
    base_config: NeuralMagPhaseRecoveryConfig
    m0_cell_xyz: np.ndarray


PHASE_RECOVERY_BENCHMARK_VARIANTS = (
    PhaseRecoveryBenchmarkVariant(
        name="alternating_cayley",
        solver_family="neuralmag",
        minimizer_method="alternating",
        minimizer_update="cayley",
        minimizer_tau_max=1e-4,
    ),
    PhaseRecoveryBenchmarkVariant(
        name="bb1_cayley",
        solver_family="neuralmag",
        minimizer_method="bb1",
        minimizer_update="cayley",
        minimizer_tau_max=1e-4,
    ),
    PhaseRecoveryBenchmarkVariant(
        name="bb2_cayley",
        solver_family="neuralmag",
        minimizer_method="bb2",
        minimizer_update="cayley",
        minimizer_tau_max=1e-4,
    ),
    PhaseRecoveryBenchmarkVariant(
        name="alternating_projected",
        solver_family="neuralmag",
        minimizer_method="alternating",
        minimizer_update="projected",
        minimizer_tau_max=1e-4,
    ),
    PhaseRecoveryBenchmarkVariant(
        name="optax_lbfgs",
        solver_family="optax_lbfgs",
        optax_lbfgs_memory_size=10,
        optax_lbfgs_max_linesearch_steps=20,
    ),
)


def configure_neuralmag_benchmark_logging() -> Any:
    """Reduce NeuralMag logging noise during scripted benchmark runs."""
    try:
        import neuralmag as nm
    except ImportError:
        return None

    if hasattr(nm, "set_log_level"):
        nm.set_log_level(logging.WARNING)
    return nm


def make_phase_recovery_benchmark_case(
    *,
    crop_shape: tuple[int, int, int] = (12, 12, 12),
    minimizer_max_iter: int = 4,
) -> PhaseRecoveryBenchmarkCase:
    """Build the shared benchmark case used for solver comparisons."""
    fixture = load_vortex_disc_fixture(32)
    rho_true = np.asarray(fixture["rho_true"], dtype=np.float32)
    m_true = np.asarray(fixture["m_true"], dtype=np.float32)
    pixel_size_nm = float(fixture["pixel_size_nm"])

    base_config = NeuralMagPhaseRecoveryConfig(
        phase_weight_schedule=(1e-4, 1e-3),
        phase_pad=2,
        phase_energy_scale=None,
        init_mode="random",
        rng_seed=5,
        demag_p=1,
        minimizer_max_iter=minimizer_max_iter,
    )
    target = prepare_neuralmag_phase_target(
        rho_true,
        m_true,
        cellsize_nm=pixel_size_nm,
        config=base_config,
        crop_shape=crop_shape,
    )
    phase_energy_scale = calibrate_phase_energy_scale(
        target,
        base_config,
        lambda_ref=base_config.phase_weight_schedule[0],
    )
    base_config = replace(base_config, phase_energy_scale=phase_energy_scale)
    m0_cell_xyz = make_initial_m_cell(
        target.rho_xyz,
        target.m_target_xyz,
        mode=base_config.init_mode,
        rng_seed=base_config.rng_seed,
        support_threshold=base_config.support_threshold,
    )
    return PhaseRecoveryBenchmarkCase(
        target=target,
        base_config=base_config,
        m0_cell_xyz=np.asarray(m0_cell_xyz, dtype=np.float32),
    )


def config_for_phase_recovery_benchmark_variant(
    base_config: NeuralMagPhaseRecoveryConfig,
    variant: PhaseRecoveryBenchmarkVariant,
) -> NeuralMagPhaseRecoveryConfig:
    """Overlay one benchmark variant onto the shared baseline config."""
    kwargs = {
        "solver_family": variant.solver_family,
    }
    if variant.minimizer_method is not None:
        kwargs["minimizer_method"] = variant.minimizer_method
    if variant.minimizer_update is not None:
        kwargs["minimizer_update"] = variant.minimizer_update
    if variant.minimizer_tau_max is not None:
        kwargs["minimizer_tau_max"] = variant.minimizer_tau_max
    if variant.optax_lbfgs_memory_size is not None:
        kwargs["optax_lbfgs_memory_size"] = variant.optax_lbfgs_memory_size
    if variant.optax_lbfgs_max_linesearch_steps is not None:
        kwargs["optax_lbfgs_max_linesearch_steps"] = variant.optax_lbfgs_max_linesearch_steps
    return replace(base_config, **kwargs)


def run_phase_recovery_benchmark_variant(
    case: PhaseRecoveryBenchmarkCase,
    variant: PhaseRecoveryBenchmarkVariant,
) -> NeuralMagPhaseRecoveryResult:
    """Run one solver variant on a fresh copy of the shared initial state."""
    config = config_for_phase_recovery_benchmark_variant(case.base_config, variant)
    return run_neuralmag_phase_recovery(
        case.target,
        config=config,
        m0_cell_xyz=np.array(case.m0_cell_xyz, copy=True),
    )


def final_phase_recovery_metrics(result: NeuralMagPhaseRecoveryResult) -> dict[str, float | int | str]:
    """Return the terminal metrics row for one phase-recovery result."""
    return next(row for row in reversed(result.history) if row["event"] == "end")


__all__ = [
    "PHASE_RECOVERY_BENCHMARK_VARIANTS",
    "PhaseRecoveryBenchmarkCase",
    "PhaseRecoveryBenchmarkVariant",
    "config_for_phase_recovery_benchmark_variant",
    "configure_neuralmag_benchmark_logging",
    "final_phase_recovery_metrics",
    "make_phase_recovery_benchmark_case",
    "run_phase_recovery_benchmark_variant",
]
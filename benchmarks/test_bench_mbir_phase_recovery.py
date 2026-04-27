from __future__ import annotations

import numpy as np
import pytest

from libertem_holo.base.mbir.phase_recovery_benchmark import (
    PHASE_RECOVERY_BENCHMARK_VARIANTS,
    configure_neuralmag_benchmark_logging,
    final_phase_recovery_metrics,
    make_phase_recovery_benchmark_case,
    run_phase_recovery_benchmark_variant,
)


nm = pytest.importorskip("neuralmag")


@pytest.fixture(scope="module")
def phase_recovery_benchmark_case():
    configure_neuralmag_benchmark_logging()
    return make_phase_recovery_benchmark_case()


@pytest.mark.benchmark(group="mbir-phase-recovery")
@pytest.mark.parametrize(
    "variant",
    PHASE_RECOVERY_BENCHMARK_VARIANTS,
    ids=lambda variant: variant.name,
)
def test_phase_recovery_solver_variants(variant, benchmark, phase_recovery_benchmark_case, request):
    case = phase_recovery_benchmark_case
    benchmark.extra_info["solver_family"] = variant.solver_family
    if variant.minimizer_method is not None:
        benchmark.extra_info["minimizer_method"] = variant.minimizer_method
    if variant.minimizer_update is not None:
        benchmark.extra_info["minimizer_update"] = variant.minimizer_update
    if variant.minimizer_tau_max is not None:
        benchmark.extra_info["minimizer_tau_max"] = variant.minimizer_tau_max
    if variant.optax_lbfgs_memory_size is not None:
        benchmark.extra_info["optax_lbfgs_memory_size"] = variant.optax_lbfgs_memory_size
    if variant.optax_lbfgs_max_linesearch_steps is not None:
        benchmark.extra_info["optax_lbfgs_max_linesearch_steps"] = variant.optax_lbfgs_max_linesearch_steps
    benchmark.extra_info["phase_weight_schedule"] = tuple(case.base_config.phase_weight_schedule)
    benchmark.extra_info["minimizer_max_iter"] = case.base_config.minimizer_max_iter
    benchmark.extra_info["crop_shape"] = tuple(case.target.rho_xyz.shape)

    if not request.config.getoption("benchmark_disable", default=False):
        # Warm JIT and state-resolution caches outside the measurement loop.
        warmup_result = run_phase_recovery_benchmark_variant(case, variant)
        assert np.isfinite(warmup_result.max_g)

    result = benchmark(lambda: run_phase_recovery_benchmark_variant(case, variant))
    final_metrics = final_phase_recovery_metrics(result)
    benchmark.extra_info["result_n_iter"] = result.n_iter
    benchmark.extra_info["result_max_g"] = result.max_g
    benchmark.extra_info["final_phase_rms"] = final_metrics["phase_rms"]
    benchmark.extra_info["final_raw_phase_loss"] = final_metrics["raw_phase_loss"]
    benchmark.extra_info["final_E_total"] = final_metrics["E_total"]

    assert result.n_iter >= 0
    assert result.m_recovered_xyz.shape == case.target.rho_xyz.shape + (3,)
    assert result.phase_recovered.shape == np.asarray(case.target.phase_target).shape
    assert np.isfinite(result.max_g)
    assert np.isfinite(final_metrics["phase_rms"])
    assert np.isfinite(final_metrics["raw_phase_loss"])
    assert np.isfinite(np.asarray(result.phase_recovered)).all()

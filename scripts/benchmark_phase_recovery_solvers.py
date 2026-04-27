#!/usr/bin/env python3
"""Run repeatable phase-recovery solver benchmarks and print a compact report."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from libertem_holo.base.mbir.phase_recovery_benchmark import (
    PHASE_RECOVERY_BENCHMARK_VARIANTS,
    configure_neuralmag_benchmark_logging,
    final_phase_recovery_metrics,
    make_phase_recovery_benchmark_case,
    run_phase_recovery_benchmark_variant,
)


@dataclass(frozen=True)
class SolverBenchmarkSummary:
    name: str
    solver_family: str
    repeats: int
    warmups: int
    elapsed_mean_s: float
    elapsed_std_s: float
    elapsed_samples_s: tuple[float, ...]
    n_iter: int
    max_g: float
    phase_rms: float
    raw_phase_loss: float
    E_total: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2, help="number of measured repeats per variant")
    parser.add_argument("--warmups", type=int, default=1, help="number of untimed warmup runs per variant")
    parser.add_argument(
        "--crop-size",
        type=int,
        default=12,
        help="cubic crop size to benchmark, used as (N, N, N)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=4,
        help="per-stage iteration budget for each solver",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path to write the raw JSON report",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="optional subset of variant names to run",
    )
    return parser.parse_args()


def _selected_variants(names: list[str] | None):
    if not names:
        return PHASE_RECOVERY_BENCHMARK_VARIANTS
    selected = [variant for variant in PHASE_RECOVERY_BENCHMARK_VARIANTS if variant.name in set(names)]
    missing = sorted(set(names) - {variant.name for variant in selected})
    if missing:
        raise SystemExit(f"Unknown variants: {', '.join(missing)}")
    return tuple(selected)


def _run_variant(variant, case, *, warmups: int, repeats: int) -> SolverBenchmarkSummary:
    for _ in range(warmups):
        run_phase_recovery_benchmark_variant(case, variant)

    elapsed_samples: list[float] = []
    last_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = run_phase_recovery_benchmark_variant(case, variant)
        elapsed_samples.append(time.perf_counter() - start)

    if last_result is None:
        raise RuntimeError("Benchmark run produced no results.")

    final_metrics = final_phase_recovery_metrics(last_result)
    elapsed_mean = statistics.fmean(elapsed_samples)
    elapsed_std = statistics.stdev(elapsed_samples) if len(elapsed_samples) > 1 else 0.0
    return SolverBenchmarkSummary(
        name=variant.name,
        solver_family=variant.solver_family,
        repeats=repeats,
        warmups=warmups,
        elapsed_mean_s=elapsed_mean,
        elapsed_std_s=elapsed_std,
        elapsed_samples_s=tuple(elapsed_samples),
        n_iter=int(last_result.n_iter),
        max_g=float(last_result.max_g),
        phase_rms=float(final_metrics["phase_rms"]),
        raw_phase_loss=float(final_metrics["raw_phase_loss"]),
        E_total=float(final_metrics["E_total"]),
    )


def _print_report(summaries: list[SolverBenchmarkSummary]) -> None:
    print(json.dumps([asdict(summary) for summary in summaries], indent=2))
    print("\nSORTED_BY_MEAN_ELAPSED")
    for summary in sorted(summaries, key=lambda item: item.elapsed_mean_s):
        print(
            f"{summary.name}: mean={summary.elapsed_mean_s:.3f}s, std={summary.elapsed_std_s:.3f}s, "
            f"n_iter={summary.n_iter}, phase_rms={summary.phase_rms:.6g}, "
            f"raw_phase_loss={summary.raw_phase_loss:.6g}, E_total={summary.E_total:.6g}, "
            f"max_g={summary.max_g:.6g}"
        )


def main() -> None:
    args = _parse_args()
    configure_neuralmag_benchmark_logging()
    case = make_phase_recovery_benchmark_case(
        crop_shape=(args.crop_size, args.crop_size, args.crop_size),
        minimizer_max_iter=args.max_iter,
    )
    summaries = [
        _run_variant(variant, case, warmups=args.warmups, repeats=args.repeats)
        for variant in _selected_variants(args.variants)
    ]
    _print_report(summaries)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps([asdict(summary) for summary in summaries], indent=2))


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import unxt as u


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")


REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT / "src", REPO_ROOT / "vendor" / "neuralmag"):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))


from libertem_holo.base.mbir.differentiable_anisotropy import unit_vector_to_axis_angles  # noqa: E402
from libertem_holo.base.mbir.equilibrium_orientation_fit import (  # noqa: E402
    EquilibriumOrientationFitConfig,
    broadcast_anisotropy_basis,
    build_equilibrium_orientation_problem,
    ensure_neuralmag_jax_backend,
    phase_from_relaxed_m,
    prepare_equilibrium_fit_target,
    prepare_equilibrium_fit_target_from_npz,
    relax_magnetization_native,
)
from libertem_holo.base.mbir.forward import forward_phase_from_density_and_magnetization  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "notebooks" / "MBIR" / "neuralmag_cube_tower_outputs"
FINE_TARGET_NPZ = OUTPUT_DIR / "cube_tower_relaxed_comparison.npz"
COARSE_TARGET_NPZ = OUTPUT_DIR / "cube_tower_coarse_llg_true_axis_targets.npz"
SUMMARY_JSON = OUTPUT_DIR / "cube_tower_solver_controls_summary.json"
LLG_FINAL_SUMMARY_JSON = OUTPUT_DIR / "cube_tower_llg_final_analysis.json"
LLG_FINAL_SUMMARY_MD = OUTPUT_DIR / "cube_tower_llg_final_analysis.md"

TRUE_AXIS = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)
TRUE_AXIS2 = np.array([1.0, -1.0, 0.0], dtype=np.float32)
TRUE_AXIS_ANGLES = jnp.asarray(unit_vector_to_axis_angles(TRUE_AXIS), dtype=jnp.float32)

COARSE_CONFIG = EquilibriumOrientationFitConfig(
    coarse_grain_factor=2,
    support_threshold=0.5,
    demag_p=2,
    phase_pad=16,
    minimizer_tol=1e3,
    minimizer_relative_tol=1e-2,
    minimizer_min_iter=32,
    minimizer_stall_patience=512,
    minimizer_stall_relative_improvement=1e-5,
    minimizer_max_iter=4096,
    minimizer_tau_min=1e-18,
    minimizer_tau_max=1e-4,
)
COARSE_PHASE_CONFIG = replace(COARSE_CONFIG, coarse_grain_factor=1)

MATERIAL = {
    "Msat_A_per_m": 480e3,
    "Aex_J_per_m": 1e-11,
    "Kc1_J_per_m3": -1.35e4,
}

LLG_ALPHA = 0.1
LLG_MAX_STEPS = 300
LLG_TOL = 5e8
LLG_DT_S = 3e-12
FINE_DEMAG_P = 3
def normalize_on_support(m_xyz: np.ndarray, rho_xyz: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    m_xyz = np.asarray(m_xyz, dtype=np.float32)
    rho_xyz = np.asarray(rho_xyz, dtype=np.float32)
    norms = np.linalg.norm(m_xyz, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return np.where(rho_xyz[..., None] > threshold, m_xyz / safe_norms, 0.0).astype(np.float32)


def uniform_y_on_support(rho_xyz: np.ndarray) -> np.ndarray:
    m_xyz = np.zeros(rho_xyz.shape + (3,), dtype=np.float32)
    m_xyz[..., 1] = 1.0
    return normalize_on_support(m_xyz, rho_xyz)


def phase_metrics(phase_a: np.ndarray, phase_b: np.ndarray) -> dict[str, float]:
    residual = np.asarray(phase_a, dtype=np.float64) - np.asarray(phase_b, dtype=np.float64)
    mse = float(np.mean(residual * residual))
    return {
        "loss": 0.5 * mse,
        "phase_rms": float(np.sqrt(mse)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def magnetization_metrics(m_a: np.ndarray, m_b: np.ndarray, rho_xyz: np.ndarray) -> dict[str, float]:
    support = np.asarray(rho_xyz) > 0.5
    diff = np.asarray(m_a, dtype=np.float64) - np.asarray(m_b, dtype=np.float64)
    diff_support = diff[support]
    dot = np.sum(np.asarray(m_a, dtype=np.float64)[support] * np.asarray(m_b, dtype=np.float64)[support], axis=-1)
    return {
        "vector_rms": float(np.sqrt(np.mean(diff_support * diff_support))),
        "max_abs": float(np.max(np.abs(diff_support))),
        "mean_dot": float(np.mean(dot)),
    }


def orthonormalize_anisotropy_axes(axis1: np.ndarray, axis2: np.ndarray):
    axis1 = np.asarray(axis1, dtype=np.float32)
    axis2 = np.asarray(axis2, dtype=np.float32)
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 - np.dot(axis2, axis1) * axis1
    axis2 = axis2 / np.linalg.norm(axis2)
    axis3 = np.cross(axis1, axis2).astype(np.float32)
    axis3 = axis3 / np.linalg.norm(axis3)
    return axis1.astype(np.float32), axis2.astype(np.float32), axis3.astype(np.float32)


def cell_vectors_to_normalized_node_function(state, m_cell_xyz: np.ndarray, rho_cell_xyz: np.ndarray, nm):
    rho_cf = nm.CellFunction(state, tensor=state.tensor(np.asarray(rho_cell_xyz, dtype=np.float32)))
    m_cf = nm.VectorCellFunction(state, tensor=state.tensor(np.asarray(m_cell_xyz, dtype=np.float32)))
    rho_node = np.asarray(rho_cf.to_node().tensor)
    m_node = np.asarray(m_cf.to_node().tensor)
    norms = np.linalg.norm(m_node, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    m_node = np.where(rho_node[..., None] > 0.5, m_node / safe_norms, 0.0).astype(np.float32)
    return nm.VectorFunction(state, tensor=state.tensor(m_node))


def build_state(
    nm,
    *,
    rho_xyz: np.ndarray,
    m0_xyz: np.ndarray,
    cellsize_nm: float,
    axis1: np.ndarray,
    axis2: np.ndarray,
    alpha: float,
    demag_p: int,
):
    nx, ny, nz = rho_xyz.shape
    state = nm.State(nm.Mesh((nx, ny, nz), (cellsize_nm * 1e-9,) * 3))
    rho_min = float(getattr(state, "eps", 1e-12))
    rho_safe = np.clip(np.asarray(rho_xyz, dtype=np.float32), rho_min, 1.0)
    state.rho = nm.CellFunction(state, tensor=state.tensor(rho_safe))

    state.material.Ms = nm.CellFunction(state).fill(MATERIAL["Msat_A_per_m"])
    state.material.A = nm.CellFunction(state).fill(MATERIAL["Aex_J_per_m"])
    state.material.alpha = nm.CellFunction(state).fill(alpha)
    state.material.Kc = nm.CellFunction(state).fill(MATERIAL["Kc1_J_per_m3"])

    axis1_vec, axis2_vec, axis3_vec = orthonormalize_anisotropy_axes(axis1, axis2)
    axis_shape = rho_safe.shape + (3,)
    state.material.Kc_axis1 = nm.VectorCellFunction(
        state,
        tensor=state.tensor(np.broadcast_to(axis1_vec, axis_shape).copy()),
    )
    state.material.Kc_axis2 = nm.VectorCellFunction(
        state,
        tensor=state.tensor(np.broadcast_to(axis2_vec, axis_shape).copy()),
    )
    state.material.Kc_axis3 = nm.VectorCellFunction(
        state,
        tensor=state.tensor(np.broadcast_to(axis3_vec, axis_shape).copy()),
    )
    state.m = cell_vectors_to_normalized_node_function(state, m0_xyz, rho_safe, nm)

    nm.ExchangeField().register(state, "exchange")
    nm.DemagField(p=demag_p).register(state, "demag")
    nm.CubicAnisotropyField().register(state, "cubic")
    nm.TotalField("exchange", "demag", "cubic").register(state)
    return state


def run_llg(nm, state) -> tuple[np.ndarray, dict[str, float | int]]:
    started = time.perf_counter()
    llg = nm.LLGSolver(state, max_steps=LLG_MAX_STEPS)
    llg.relax(tol=LLG_TOL, dt=LLG_DT_S)
    elapsed_s = time.perf_counter() - started
    m_relaxed = normalize_on_support(np.asarray(state.m.to_cell().tensor), np.asarray(state.rho.tensor))
    return m_relaxed, {
        "elapsed_s": elapsed_s,
        "max_steps": int(LLG_MAX_STEPS),
        "tol": float(LLG_TOL),
        "dt": float(LLG_DT_S),
    }


def run_coarse_llg(problem, m0_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    axis1_field, axis2_field, axis3_field = broadcast_anisotropy_basis(problem, TRUE_AXIS_ANGLES)
    problem.state.material.Kc_axis1.tensor = problem.state.tensor(np.asarray(axis1_field, dtype=np.float32))
    problem.state.material.Kc_axis2.tensor = problem.state.tensor(np.asarray(axis2_field, dtype=np.float32))
    problem.state.material.Kc_axis3.tensor = problem.state.tensor(np.asarray(axis3_field, dtype=np.float32))
    problem.state.material.alpha = problem.nm.CellFunction(problem.state).fill(LLG_ALPHA)
    problem.state.m.tensor = problem.state.tensor(np.asarray(m0_xyz, dtype=np.float32))

    started = time.perf_counter()
    llg = problem.nm.LLGSolver(problem.state, max_steps=LLG_MAX_STEPS)
    llg.relax(tol=LLG_TOL, dt=LLG_DT_S)
    elapsed_s = time.perf_counter() - started

    m_relaxed = normalize_on_support(np.asarray(problem.state.m.tensor), problem.target.rho_xyz)
    phase_relaxed = np.asarray(phase_from_relaxed_m(problem, jnp.asarray(m_relaxed, dtype=jnp.float32)), dtype=np.float32)
    return m_relaxed, phase_relaxed, {
        "elapsed_s": elapsed_s,
        "max_steps": int(LLG_MAX_STEPS),
        "tol": float(LLG_TOL),
        "dt": float(LLG_DT_S),
    }


def run_native_bb(nm, state) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    rho_node = jnp.asarray(state.rho.to_node().tensor, dtype=jnp.float32)

    def projection(values):
        values = jnp.asarray(values, dtype=jnp.float32)
        norms = jnp.linalg.norm(values, axis=-1, keepdims=True)
        safe_norms = jnp.where(norms > 0.0, norms, 1.0)
        return jnp.where(rho_node[..., None] > 0.5, values / safe_norms, 0.0)

    solver_kwargs = dict(
        method="alternating",
        update="cayley",
        tol=COARSE_CONFIG.minimizer_tol,
        max_iter=COARSE_CONFIG.minimizer_max_iter,
        tau_min=COARSE_CONFIG.minimizer_tau_min,
        tau_max=COARSE_CONFIG.minimizer_tau_max,
    )
    uses_projection = True
    try:
        solver = nm.EnergyMinimizer(state, projection=projection, **solver_kwargs)
    except TypeError as exc:
        if "projection" not in str(exc):
            raise
        solver = nm.EnergyMinimizer(state, **solver_kwargs)
        uses_projection = False

    started = time.perf_counter()
    try:
        max_g, info = solver.minimize(return_info=True)
        n_iter = int(np.asarray(info["n_iter"]))
    except TypeError:
        max_g = solver.minimize()
        n_iter = int(getattr(solver, "n_iter", COARSE_CONFIG.minimizer_max_iter))
    elapsed_s = time.perf_counter() - started

    if not uses_projection:
        state.m.tensor = state.tensor(projection(np.asarray(state.m.tensor)))

    m_relaxed = normalize_on_support(np.asarray(state.m.to_cell().tensor), np.asarray(state.rho.tensor))
    return m_relaxed, {
        "elapsed_s": elapsed_s,
        "n_iter": n_iter,
        "max_g": float(np.asarray(max_g)),
        "uses_projection": uses_projection,
    }


def phase_from_xyz(rho_xyz: np.ndarray, m_xyz: np.ndarray, *, cellsize_nm: float) -> np.ndarray:
    return np.asarray(
        forward_phase_from_density_and_magnetization(
            rho=np.asarray(rho_xyz, dtype=np.float32),
            magnetization_3d=np.asarray(m_xyz, dtype=np.float32),
            pixel_size=u.Quantity(cellsize_nm, "nm"),
            axis="z",
            geometry="disc",
        ),
        dtype=np.float32,
    )


def coarse_initial_m(target, mode: str, *, seed: int = 11) -> np.ndarray:
    if mode == "target":
        return np.asarray(target.m_xyz, dtype=np.float32)
    if mode == "random":
        rng = np.random.default_rng(seed)
        m0_xyz = rng.normal(size=target.m_xyz.shape).astype(np.float32)
        return normalize_on_support(m0_xyz, target.rho_xyz, threshold=COARSE_CONFIG.support_threshold)
    raise ValueError(f"Unsupported coarse initial mode: {mode!r}")


def run_coarse_true_axis_control(nm, *, modes: tuple[str, ...] = ("target", "random")) -> dict[str, object]:
    imported_target = prepare_equilibrium_fit_target_from_npz(
        FINE_TARGET_NPZ,
        "1, 1, 1 Axis",
        config=COARSE_CONFIG,
    )
    reference_problem = build_equilibrium_orientation_problem(imported_target, config=COARSE_CONFIG, **MATERIAL)
    imported_phase = np.asarray(reference_problem.target.phase_target, dtype=np.float32)

    coarse_payload: dict[str, np.ndarray] = {
        "cellsize_nm": np.asarray(imported_target.cellsize_nm, dtype=np.float32),
        "rho_xyz": np.asarray(imported_target.rho_xyz, dtype=np.float32),
        "imported_phase_target": imported_phase,
    }
    summary: dict[str, object] = {}

    for mode in modes:
        m0_xyz = coarse_initial_m(imported_target, mode, seed=11)

        print(f"[coarse] mode={mode}: running LLG", flush=True)
        llg_problem = build_equilibrium_orientation_problem(imported_target, config=COARSE_CONFIG, **MATERIAL)
        llg_m, llg_phase, llg_info = run_coarse_llg(llg_problem, m0_xyz)

        print(f"[coarse] mode={mode}: running BB", flush=True)
        bb_problem = build_equilibrium_orientation_problem(imported_target, config=COARSE_CONFIG, **MATERIAL)
        bb_m, bb_info = relax_magnetization_native(
            bb_problem,
            TRUE_AXIS_ANGLES,
            jnp.asarray(m0_xyz, dtype=jnp.float32),
            method="alternating",
            update="cayley",
        )
        bb_phase = np.asarray(
            phase_from_relaxed_m(bb_problem, jnp.asarray(bb_m, dtype=jnp.float32)),
            dtype=np.float32,
        )

        coarse_payload[f"m_{mode}_start"] = np.asarray(llg_m, dtype=np.float32)
        coarse_payload[f"phase_{mode}_start"] = llg_phase

        summary[mode] = {
            "llg_info": llg_info,
            "bb_info": bb_info,
            "llg_vs_imported_fine_target": phase_metrics(llg_phase, imported_phase),
            "bb_vs_imported_fine_target": phase_metrics(bb_phase, imported_phase),
            "bb_vs_coarse_llg_target": phase_metrics(bb_phase, llg_phase),
            "bb_vs_coarse_llg_m": magnetization_metrics(bb_m, llg_m, imported_target.rho_xyz),
        }

    np.savez_compressed(COARSE_TARGET_NPZ, **coarse_payload)
    return {
        "coarse_cellsize_nm": float(imported_target.cellsize_nm),
        "coarse_shape_xyz": list(imported_target.rho_xyz.shape),
        "coarse_modes_run": list(modes),
        "target_npz": str(COARSE_TARGET_NPZ.relative_to(REPO_ROOT)),
        "runs": summary,
    }


def run_fine_true_axis_control(nm) -> dict[str, object]:
    with np.load(FINE_TARGET_NPZ, allow_pickle=True) as data:
        rho_fine = np.asarray(data["rho_1,_1,_1_axis"], dtype=np.float32)
        m_saved_target = np.asarray(data["m_1,_1,_1_axis"], dtype=np.float32)
        cellsize_nm = float(np.asarray(data["cellsize_nm"]))

    ovf_path = REPO_ROOT / "vortex_cube_128_00+1.ovf"
    m0_fine = uniform_y_on_support(rho_fine)
    init_label = "uniform_y"
    if ovf_path.exists():
        init_label = f"uniform_y (ovf present but not used: {ovf_path.name})"

    saved_phase = phase_from_xyz(rho_fine, m_saved_target, cellsize_nm=cellsize_nm)

    llg_state = build_state(
        nm,
        rho_xyz=rho_fine,
        m0_xyz=m0_fine,
        cellsize_nm=cellsize_nm,
        axis1=TRUE_AXIS,
        axis2=TRUE_AXIS2,
        alpha=LLG_ALPHA,
        demag_p=FINE_DEMAG_P,
    )
    llg_m, llg_info = run_llg(nm, llg_state)
    llg_phase = phase_from_xyz(rho_fine, llg_m, cellsize_nm=cellsize_nm)

    bb_state = build_state(
        nm,
        rho_xyz=rho_fine,
        m0_xyz=m0_fine,
        cellsize_nm=cellsize_nm,
        axis1=TRUE_AXIS,
        axis2=TRUE_AXIS2,
        alpha=LLG_ALPHA,
        demag_p=FINE_DEMAG_P,
    )
    bb_m, bb_info = run_native_bb(nm, bb_state)
    bb_phase = phase_from_xyz(rho_fine, bb_m, cellsize_nm=cellsize_nm)

    return {
        "fine_cellsize_nm": float(cellsize_nm),
        "fine_shape_xyz": list(rho_fine.shape),
        "initialization_used": init_label,
        "ovf_present": ovf_path.exists(),
        "llg_info": llg_info,
        "bb_info": bb_info,
        "llg_vs_saved_target": phase_metrics(llg_phase, saved_phase),
        "bb_vs_saved_target": phase_metrics(bb_phase, saved_phase),
        "bb_vs_llg": phase_metrics(bb_phase, llg_phase),
        "saved_target_vs_llg_m": magnetization_metrics(m_saved_target, llg_m, rho_fine),
        "saved_target_vs_bb_m": magnetization_metrics(m_saved_target, bb_m, rho_fine),
        "bb_vs_llg_m": magnetization_metrics(bb_m, llg_m, rho_fine),
    }


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: object, digits: int = 6) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}g}"


def build_llg_final_analysis(summary: dict[str, object]) -> dict[str, object]:
    coarse = summary.get("coarse_true_axis_control")
    fine = summary.get("fine_true_axis_control")
    if not isinstance(coarse, dict) or not isinstance(fine, dict):
        return {
            "status": "incomplete",
            "reason": (
                "Both coarse_true_axis_control and fine_true_axis_control are required "
                "to finalize the cube analysis on the LLG basis."
            ),
        }

    runs = coarse.get("runs")
    target_run = runs.get("target") if isinstance(runs, dict) else None
    coarse_llg = target_run.get("llg_vs_imported_fine_target") if isinstance(target_run, dict) else None
    fine_llg_phase = fine.get("llg_vs_saved_target")
    fine_llg_m = fine.get("saved_target_vs_llg_m")
    if not isinstance(coarse_llg, dict) or not isinstance(fine_llg_phase, dict) or not isinstance(fine_llg_m, dict):
        return {
            "status": "incomplete",
            "reason": "Missing LLG metrics in the existing cube summary.",
        }

    coarse_phase_rms = _as_float(coarse_llg.get("phase_rms"))
    fine_phase_rms = _as_float(fine_llg_phase.get("phase_rms"))
    coarse_to_fine_ratio = None
    if coarse_phase_rms is not None and fine_phase_rms not in (None, 0.0):
        coarse_to_fine_ratio = coarse_phase_rms / fine_phase_rms

    return {
        "status": "finalized_on_fine_llg",
        "trusted_llg_target_npz": str(FINE_TARGET_NPZ.relative_to(REPO_ROOT)),
        "derived_from_summary_json": str(SUMMARY_JSON.relative_to(REPO_ROOT)),
        "neuralmag_backend": summary.get("neuralmag_backend"),
        "coarse_model_check": {
            "shape_xyz": coarse.get("coarse_shape_xyz"),
            "cellsize_nm": coarse.get("coarse_cellsize_nm"),
            "llg_info": target_run.get("llg_info"),
            "llg_vs_imported_fine_target": coarse_llg,
        },
        "fine_llg_reproduction_check": {
            "shape_xyz": fine.get("fine_shape_xyz"),
            "cellsize_nm": fine.get("fine_cellsize_nm"),
            "initialization_used": fine.get("initialization_used"),
            "ovf_present": fine.get("ovf_present"),
            "llg_info": fine.get("llg_info"),
            "llg_vs_saved_target": fine_llg_phase,
            "saved_target_vs_llg_m": fine_llg_m,
        },
        "coarse_to_fine_phase_rms_ratio": coarse_to_fine_ratio,
        "recommended_basis": (
            "Use the fine-grid LLG target stored in cube_tower_relaxed_comparison.npz "
            "as the trusted basis for follow-on cube analysis."
        ),
        "conclusion": (
            "The coarse-grid LLG model does not reproduce the imported fine-grid target, "
            "so that discrepancy is a model/target mismatch. Fine-grid LLG reproduces the "
            "saved fine-grid target closely in both phase and magnetization, so the cube "
            "analysis should be finalized on the fine-grid LLG result."
        ),
    }


def write_llg_final_report(llg_analysis: dict[str, object]) -> None:
    if llg_analysis.get("status") != "finalized_on_fine_llg":
        report = "# Cube-Tower LLG Finalization\n\n"
        report += "LLG finalization is incomplete.\n\n"
        report += f"Reason: {llg_analysis.get('reason', 'missing data')}\n"
        LLG_FINAL_SUMMARY_MD.write_text(report, encoding="utf-8")
        return

    coarse_check = llg_analysis.get("coarse_model_check", {})
    fine_check = llg_analysis.get("fine_llg_reproduction_check", {})
    coarse_metrics = coarse_check.get("llg_vs_imported_fine_target", {})
    fine_phase = fine_check.get("llg_vs_saved_target", {})
    fine_m = fine_check.get("saved_target_vs_llg_m", {})

    lines = [
        "# Cube-Tower LLG Finalization",
        "",
        "## Basis",
        "",
        f"- Status: {llg_analysis['status']}",
        f"- Backend: {llg_analysis.get('neuralmag_backend', 'n/a')}",
        f"- Trusted LLG target: {llg_analysis.get('trusted_llg_target_npz', 'n/a')}",
        "",
        "## Coarse Model Check",
        "",
        f"- Coarse shape XYZ: {coarse_check.get('shape_xyz', 'n/a')}",
        f"- Coarse cell size (nm): {coarse_check.get('cellsize_nm', 'n/a')}",
        f"- Coarse LLG phase RMS vs imported fine target: {_format_float(coarse_metrics.get('phase_rms'))}",
        f"- Coarse LLG max abs phase residual vs imported fine target: {_format_float(coarse_metrics.get('max_abs'))}",
        "",
        "## Fine LLG Reproduction Check",
        "",
        f"- Fine shape XYZ: {fine_check.get('shape_xyz', 'n/a')}",
        f"- Fine cell size (nm): {fine_check.get('cellsize_nm', 'n/a')}",
        f"- Initialization used: {fine_check.get('initialization_used', 'n/a')}",
        f"- Fine LLG phase RMS vs saved target: {_format_float(fine_phase.get('phase_rms'))}",
        f"- Fine LLG max abs phase residual vs saved target: {_format_float(fine_phase.get('max_abs'))}",
        f"- Fine saved-target vs LLG mean dot: {_format_float(fine_m.get('mean_dot'))}",
        f"- Fine saved-target vs LLG vector RMS: {_format_float(fine_m.get('vector_rms'))}",
        f"- Coarse/fine phase-RMS ratio: {_format_float(llg_analysis.get('coarse_to_fine_phase_rms_ratio'))}",
        "",
        "## Conclusion",
        "",
        llg_analysis["conclusion"],
        "",
        llg_analysis["recommended_basis"],
        "",
    ]
    LLG_FINAL_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-coarse", action="store_true", help="Skip the coarse-model control.")
    parser.add_argument("--skip-fine", action="store_true", help="Skip the fine-model control.")
    parser.add_argument(
        "--llg-report-only",
        action="store_true",
        help="Write the derived LLG-only final analysis from the existing summary without rerunning controls.",
    )
    parser.add_argument(
        "--coarse-mode",
        choices=("all", "target", "random"),
        default="all",
        help="Limit the coarse-model control to one initialization mode.",
    )
    args = parser.parse_args()

    if args.skip_coarse and args.skip_fine and not args.llg_report_only:
        raise SystemExit("Nothing to do: both controls were skipped.")
    if args.llg_report_only and not SUMMARY_JSON.exists():
        raise FileNotFoundError(f"Missing existing cube summary: {SUMMARY_JSON}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {}
    if SUMMARY_JSON.exists():
        try:
            summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    needs_solver_run = (not args.skip_coarse) or (not args.skip_fine)
    if needs_solver_run:
        if not FINE_TARGET_NPZ.exists():
            raise FileNotFoundError(f"Missing saved cube-tower target: {FINE_TARGET_NPZ}")
        nm = ensure_neuralmag_jax_backend()
        backend_name = getattr(getattr(nm, "config", None), "backend", None)
        backend_name = getattr(backend_name, "name", backend_name)
        summary["neuralmag_backend"] = backend_name

    summary["fine_target_npz"] = str(FINE_TARGET_NPZ.relative_to(REPO_ROOT))

    if not args.skip_coarse:
        coarse_modes = ("target", "random") if args.coarse_mode == "all" else (args.coarse_mode,)
        summary["coarse_true_axis_control"] = run_coarse_true_axis_control(nm, modes=coarse_modes)
    if not args.skip_fine:
        summary["fine_true_axis_control"] = run_fine_true_axis_control(nm)

    llg_final_analysis = build_llg_final_analysis(summary)
    summary["llg_final_analysis"] = llg_final_analysis

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LLG_FINAL_SUMMARY_JSON.write_text(
        json.dumps(llg_final_analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_llg_final_report(llg_final_analysis)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote summary to {SUMMARY_JSON}")
    print(f"Wrote LLG-only analysis to {LLG_FINAL_SUMMARY_JSON}")
    print(f"Wrote LLG-only report to {LLG_FINAL_SUMMARY_MD}")


if __name__ == "__main__":
    main()
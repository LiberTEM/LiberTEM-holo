"""Differentiable BB2/Cayley equilibrium fitting for anisotropy orientation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import unxt as u

from .differentiable_anisotropy import (
    angle_params_to_anisotropy_axes,
    pad_phase_view_zyx_jax,
    unit_vector_to_axis_angles,
)
from .forward import forward_phase_from_density_and_magnetization
from .kernel import build_rdfc_kernel


def _default_axis2_hint() -> np.ndarray:
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _xyz_to_zyx(values: np.ndarray | jax.Array) -> jax.Array:
    array = jnp.asarray(values)
    axes = (2, 1, 0) if array.ndim == 3 else (2, 1, 0, 3)
    return jnp.transpose(array, axes)


def _safe_norm(
    values: jax.Array,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    eps: float = 1e-8,
) -> jax.Array:
    values = jnp.asarray(values)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    sq_norm = jnp.sum(values * values, axis=axis, keepdims=keepdims)
    return jnp.sqrt(sq_norm + eps_arr * eps_arr)


def _safe_normalize(values: jax.Array, *, axis: int = -1, eps: float = 1e-8) -> jax.Array:
    values = jnp.asarray(values)
    return values / _safe_norm(values, axis=axis, keepdims=True, eps=eps)


def _normalize_on_support_np(
    m_xyz: np.ndarray,
    rho_xyz: np.ndarray,
    *,
    support_threshold: float,
) -> np.ndarray:
    m_xyz = np.asarray(m_xyz, dtype=np.float32)
    rho_xyz = np.asarray(rho_xyz, dtype=np.float32)
    norms = np.linalg.norm(m_xyz, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normalized = m_xyz / safe_norms
    return np.where(rho_xyz[..., None] > support_threshold, normalized, 0.0).astype(np.float32)


def coarse_grain_volume_xyz(
    rho_xyz: np.ndarray,
    m_xyz: np.ndarray,
    factor: int,
    *,
    support_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-average a cell-centred volume before differentiable fitting."""
    if factor <= 1:
        return np.asarray(rho_xyz, dtype=np.float32), _normalize_on_support_np(
            m_xyz,
            rho_xyz,
            support_threshold=support_threshold,
        )

    nx0, ny0, nz0 = rho_xyz.shape
    nx_t = (nx0 // factor) * factor
    ny_t = (ny0 // factor) * factor
    nz_t = (nz0 // factor) * factor
    rho_t = np.asarray(rho_xyz[:nx_t, :ny_t, :nz_t], dtype=np.float32)
    m_t = np.asarray(m_xyz[:nx_t, :ny_t, :nz_t], dtype=np.float32)
    block_shape = (nx_t // factor, factor, ny_t // factor, factor, nz_t // factor, factor)

    rho_cg = rho_t.reshape(block_shape).mean(axis=(1, 3, 5))
    weighted_m = (rho_t[..., None] * m_t).reshape(*block_shape, 3).sum(axis=(1, 3, 5))
    rho_sum = rho_t.reshape(block_shape).sum(axis=(1, 3, 5))
    m_avg = weighted_m / np.maximum(rho_sum[..., None], 1e-9)
    m_cg = _normalize_on_support_np(m_avg, rho_cg, support_threshold=support_threshold)
    return rho_cg.astype(np.float32), m_cg.astype(np.float32)


def angular_distance_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    axis_a = np.asarray(axis_a, dtype=np.float32)
    axis_b = np.asarray(axis_b, dtype=np.float32)
    axis_a = axis_a / np.linalg.norm(axis_a)
    axis_b = axis_b / np.linalg.norm(axis_b)
    dot = abs(float(np.clip(np.dot(axis_a, axis_b), -1.0, 1.0)))
    return float(np.degrees(np.arccos(dot)))


@dataclass(frozen=True)
class EquilibriumOrientationFitConfig:
    """Configuration for differentiable BB2/Cayley equilibrium fitting."""

    coarse_grain_factor: int = 4
    support_threshold: float = 0.5
    demag_p: int = 2
    phase_pad: int = 16
    projection_axis: str = "z"
    geometry: str = "disc"
    minimizer_tol: float = 1e3
    minimizer_max_iter: int = 2000
    minimizer_relative_tol: float = 1e-2
    minimizer_min_iter: int = 16
    minimizer_stall_patience: int = 32
    minimizer_stall_relative_improvement: float = 1e-4
    minimizer_tau_min: float = 1e-18
    minimizer_tau_max: float = 1e-4
    outer_steps: int = 8
    axis_learning_rate: float = 4e-2
    axis2_hint: np.ndarray = field(default_factory=_default_axis2_hint)


@dataclass(frozen=True)
class EquilibriumOrientationFitTarget:
    """Prepared coarse-grained target for differentiable equilibrium fitting."""

    name: str
    cellsize_nm: float
    rho_xyz: np.ndarray
    m_xyz: np.ndarray
    rho_zyx: jax.Array
    rho_zyx_view: jax.Array
    phase_target: jax.Array
    rdfc_kernel: dict[str, Any]


@dataclass
class EquilibriumOrientationProblem:
    """Resolved NeuralMag field operators for a differentiable fit target."""

    target: EquilibriumOrientationFitTarget
    config: EquilibriumOrientationFitConfig
    state: Any
    nm: Any
    h_with_axes: Any
    support_mask: jax.Array
    axis_field_shape: tuple[int, int, int, int]


def ensure_neuralmag_jax_backend():
    """Return the NeuralMag module after ensuring the JAX backend is active."""
    try:
        import neuralmag as nm
    except ImportError as exc:
        raise ImportError("Differentiable equilibrium fitting requires neuralmag.") from exc

    try:
        backend_name = nm.config.backend.name
    except AttributeError:
        backend_name = getattr(nm.config, "backend_name", None)

    if backend_name != "jax":
        try:
            nm.config.set_backend("jax")
        except Exception as exc:
            raise RuntimeError(
                "Differentiable equilibrium fitting requires NeuralMag's JAX backend. "
                "Configure a fresh kernel with the JAX backend before running this workflow."
            ) from exc
    return nm


def prepare_equilibrium_fit_target(
    name: str,
    rho_xyz: np.ndarray,
    m_xyz: np.ndarray,
    *,
    cellsize_nm: float,
    config: EquilibriumOrientationFitConfig | None = None,
) -> EquilibriumOrientationFitTarget:
    """Prepare a differentiable phase-fit target from cell-centred XYZ arrays."""
    config = config or EquilibriumOrientationFitConfig()
    rho_cg, m_cg = coarse_grain_volume_xyz(
        rho_xyz,
        m_xyz,
        config.coarse_grain_factor,
        support_threshold=config.support_threshold,
    )
    fit_cellsize_nm = float(cellsize_nm) * config.coarse_grain_factor
    rho_zyx = _xyz_to_zyx(rho_cg)
    m_zyx = _xyz_to_zyx(m_cg)
    rho_zyx_view, m_zyx_view = pad_phase_view_zyx_jax(rho_zyx, m_zyx, config.phase_pad)
    phase_target = jnp.asarray(
        forward_phase_from_density_and_magnetization(
            rho=rho_zyx_view,
            magnetization_3d=m_zyx_view,
            pixel_size=u.Quantity(fit_cellsize_nm, "nm"),
            axis=config.projection_axis,
            geometry=config.geometry,
        ),
        dtype=jnp.float32,
    )
    rdfc_kernel = build_rdfc_kernel(tuple(phase_target.shape), geometry=config.geometry)
    return EquilibriumOrientationFitTarget(
        name=name,
        cellsize_nm=fit_cellsize_nm,
        rho_xyz=rho_cg,
        m_xyz=m_cg,
        rho_zyx=jnp.asarray(rho_zyx, dtype=jnp.float32),
        rho_zyx_view=jnp.asarray(rho_zyx_view, dtype=jnp.float32),
        phase_target=phase_target,
        rdfc_kernel=rdfc_kernel,
    )


def prepare_equilibrium_fit_target_from_npz(
    npz_path: str | bytes | Any,
    name: str,
    *,
    config: EquilibriumOrientationFitConfig | None = None,
) -> EquilibriumOrientationFitTarget:
    """Load one saved cube-tower orientation from the notebook export NPZ."""
    key = name.lower().replace(" ", "_")
    with np.load(npz_path, allow_pickle=True) as data:
        cellsize_nm = float(np.asarray(data["cellsize_nm"]))
        rho_xyz = np.asarray(data[f"rho_{key}"], dtype=np.float32)
        m_xyz = np.asarray(data[f"m_{key}"], dtype=np.float32)
    return prepare_equilibrium_fit_target(
        name,
        rho_xyz,
        m_xyz,
        cellsize_nm=cellsize_nm,
        config=config,
    )


def build_equilibrium_orientation_problem(
    target: EquilibriumOrientationFitTarget,
    *,
    Msat_A_per_m: float,
    Aex_J_per_m: float,
    Kc1_J_per_m3: float,
    config: EquilibriumOrientationFitConfig | None = None,
) -> EquilibriumOrientationProblem:
    """Resolve the effective-field callable used inside differentiable fitting."""
    config = config or EquilibriumOrientationFitConfig()
    nm = ensure_neuralmag_jax_backend()
    from neuralmag.backends.jax.energy_minimizer_jax import effective_field_fn

    nx_fit, ny_fit, nz_fit = target.rho_xyz.shape
    state = nm.State(nm.Mesh((nx_fit, ny_fit, nz_fit), (target.cellsize_nm * 1e-9,) * 3))
    rho_min = float(getattr(state, "eps", 1e-12))
    rho_safe = np.clip(target.rho_xyz.astype(np.float32), rho_min, 1.0)
    state.rho = nm.CellFunction(state, tensor=state.tensor(rho_safe))

    state.material.Ms = nm.CellFunction(state).fill(Msat_A_per_m)
    state.material.A = nm.CellFunction(state).fill(Aex_J_per_m)
    state.material.Kc = nm.CellFunction(state).fill(Kc1_J_per_m3)

    axis1_ref, axis2_ref, axis3_ref = angle_params_to_anisotropy_axes(
        jnp.array(unit_vector_to_axis_angles(np.array([0.0, 0.0, 1.0], dtype=np.float32)), dtype=jnp.float32),
        axis2_hint=jnp.asarray(config.axis2_hint, dtype=jnp.float32),
    )
    axis_shape = target.rho_xyz.shape + (3,)
    state.material.Kc_axis1 = nm.VectorCellFunction(
        state,
        tensor=state.tensor(np.broadcast_to(np.asarray(axis1_ref, dtype=np.float32), axis_shape)),
    )
    state.material.Kc_axis2 = nm.VectorCellFunction(
        state,
        tensor=state.tensor(np.broadcast_to(np.asarray(axis2_ref, dtype=np.float32), axis_shape)),
    )
    state.material.Kc_axis3 = nm.VectorCellFunction(
        state,
        tensor=state.tensor(np.broadcast_to(np.asarray(axis3_ref, dtype=np.float32), axis_shape)),
    )
    state.m = nm.VectorCellFunction(state, tensor=state.tensor(target.m_xyz.astype(np.float32)))

    nm.ExchangeField().register(state, "exchange")
    nm.DemagField(p=config.demag_p).register(state, "demag")
    nm.CubicAnisotropyField().register(state, "cubic")
    nm.TotalField("exchange", "demag", "cubic").register(state)

    h_with_axes = jax.jit(
        state.resolve(
            effective_field_fn,
            ["m", "material__Kc_axis1", "material__Kc_axis2", "material__Kc_axis3"],
        )
    )
    support_mask = jnp.asarray(target.rho_xyz > config.support_threshold)
    axis_field_shape = target.rho_xyz.shape + (3,)
    return EquilibriumOrientationProblem(
        target=target,
        config=config,
        state=state,
        nm=nm,
        h_with_axes=h_with_axes,
        support_mask=support_mask,
        axis_field_shape=axis_field_shape,
    )


def project_m_to_support(problem: EquilibriumOrientationProblem, m_xyz: jax.Array) -> jax.Array:
    """Project a magnetization field to unit norm on support and zero off support."""
    m_xyz = jnp.asarray(m_xyz, dtype=jnp.float32)
    support = problem.support_mask.astype(m_xyz.dtype)[..., None]
    safe_norms = _safe_norm(m_xyz, axis=-1, keepdims=True, eps=float(jnp.finfo(m_xyz.dtype).eps))
    return support * (m_xyz / safe_norms)


def broadcast_anisotropy_basis(
    problem: EquilibriumOrientationProblem,
    axis_angles: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Broadcast one anisotropy basis over the full fit volume."""
    axis1, axis2, axis3 = angle_params_to_anisotropy_axes(
        jnp.asarray(axis_angles, dtype=jnp.float32),
        axis2_hint=jnp.asarray(problem.config.axis2_hint, dtype=jnp.float32),
    )
    return (
        jnp.broadcast_to(axis1, problem.axis_field_shape),
        jnp.broadcast_to(axis2, problem.axis_field_shape),
        jnp.broadcast_to(axis3, problem.axis_field_shape),
    )


def phase_from_relaxed_m(problem: EquilibriumOrientationProblem, m_xyz: jax.Array) -> jax.Array:
    """Project a relaxed XYZ magnetization volume into the phase image domain."""
    m_zyx = _xyz_to_zyx(m_xyz)
    _, m_zyx_view = pad_phase_view_zyx_jax(problem.target.rho_zyx, m_zyx, problem.config.phase_pad)
    return forward_phase_from_density_and_magnetization(
        rho=problem.target.rho_zyx_view,
        magnetization_3d=m_zyx_view,
        pixel_size=u.Quantity(problem.target.cellsize_nm, "nm"),
        axis=problem.config.projection_axis,
        geometry=problem.config.geometry,
        rdfc_kernel=problem.target.rdfc_kernel,
    )


def _descent_direction(m_xyz: jax.Array, h_xyz: jax.Array) -> jax.Array:
    m_dot_h = jnp.sum(m_xyz * h_xyz, axis=-1, keepdims=True)
    m_dot_m = jnp.sum(m_xyz * m_xyz, axis=-1, keepdims=True)
    return m_xyz * m_dot_h - h_xyz * m_dot_m


def _flatten_trailing_dims(values: jax.Array, trailing_ndim: int) -> jax.Array:
    values = jnp.asarray(values)
    if values.ndim <= trailing_ndim:
        return values.reshape((-1,))
    return values.reshape(values.shape[:-trailing_ndim] + (-1,))


def _field_max_norm(field_xyz: jax.Array) -> jax.Array:
    norm_sq = jnp.sum(field_xyz * field_xyz, axis=-1)
    flat_norm_sq = _flatten_trailing_dims(norm_sq, trailing_ndim=3)
    return jnp.sqrt(jnp.max(flat_norm_sq, axis=-1))


def _flatten_dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    lhs_flat = _flatten_trailing_dims(lhs, trailing_ndim=4)
    rhs_flat = _flatten_trailing_dims(rhs, trailing_ndim=4)
    return jnp.sum(lhs_flat * rhs_flat, axis=-1)


def _mean_squared_residual(residual: jax.Array) -> jax.Array:
    residual = jnp.asarray(residual)
    residual_sq = residual * residual
    flat_residual_sq = _flatten_trailing_dims(residual_sq, trailing_ndim=2)
    return jnp.mean(flat_residual_sq, axis=-1)


def _initial_tau(problem: EquilibriumOrientationProblem, h_xyz: jax.Array) -> jax.Array:
    h_max = _field_max_norm(h_xyz)
    tau = 1.0 / jnp.maximum(h_max, problem.config.minimizer_tau_min)
    return jnp.clip(tau, problem.config.minimizer_tau_min, problem.config.minimizer_tau_max)


def _use_bb1(method: str, iteration: jax.Array) -> jax.Array:
    if method == "bb1":
        return jnp.ones_like(jnp.asarray(iteration), dtype=bool)
    if method == "alternating":
        return jnp.asarray(iteration) % 2 == 1
    return jnp.zeros_like(jnp.asarray(iteration), dtype=bool)


def _bb_tau(
    problem: EquilibriumOrientationProblem,
    m,
    g,
    m_prev,
    g_prev,
    tau_prev,
    iteration,
    has_history,
    *,
    method: str,
):
    s_vec = m - m_prev
    y_vec = g - g_prev
    sy = _flatten_dot(s_vec, y_vec)
    yy = _flatten_dot(y_vec, y_vec)
    ss = _flatten_dot(s_vec, s_vec)
    use_bb1 = _use_bb1(method, iteration)
    numer = jnp.where(use_bb1, ss, sy)
    denom = jnp.where(use_bb1, sy, yy)
    valid = has_history & (denom > 0.0) & (numer > 0.0)
    safe_denom = jnp.maximum(denom, jnp.asarray(problem.config.minimizer_tau_min, dtype=denom.dtype))
    tau = jnp.where(valid, numer / safe_denom, tau_prev)
    return jnp.clip(tau, problem.config.minimizer_tau_min, problem.config.minimizer_tau_max)


def _cayley_update(m_xyz: jax.Array, h_xyz: jax.Array, g_xyz: jax.Array, tau: jax.Array) -> jax.Array:
    tau = jnp.asarray(tau, dtype=m_xyz.dtype)
    tau = jnp.reshape(tau, tau.shape + (1,) * max(m_xyz.ndim - tau.ndim, 0))
    m_cross_h = jnp.cross(m_xyz, h_xyz)
    a2 = tau * tau * jnp.sum(m_cross_h * m_cross_h, axis=-1, keepdims=True)
    return ((1.0 - 0.25 * a2) * m_xyz - tau * g_xyz) / (1.0 + 0.25 * a2)


def relax_magnetization(
    problem: EquilibriumOrientationProblem,
    axis_angles: jax.Array,
    m0_xyz: jax.Array,
    *,
    max_iter: int | None = None,
    tol: float | None = None,
    method: str = "bb2",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Relax one magnetization field with adaptive BB2/Cayley convergence checks."""
    valid_methods = {"bb1", "bb2", "alternating"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported BB method {method!r}. Expected one of {sorted(valid_methods)}.")

    max_iter = problem.config.minimizer_max_iter if max_iter is None else int(max_iter)
    tol = problem.config.minimizer_tol if tol is None else float(tol)
    min_iter = min(max(int(problem.config.minimizer_min_iter), 0), max_iter)
    stall_patience = max(int(problem.config.minimizer_stall_patience), 0)
    stall_rel_improvement = float(problem.config.minimizer_stall_relative_improvement)
    axis1_field, axis2_field, axis3_field = broadcast_anisotropy_basis(problem, axis_angles)

    m_initial = project_m_to_support(problem, m0_xyz)
    h_initial = problem.h_with_axes(m_initial, axis1_field, axis2_field, axis3_field)
    g_initial = _descent_direction(m_initial, h_initial)
    max_g_initial = _field_max_norm(g_initial)
    threshold_enabled = jnp.asarray(tol >= 0.0)
    target_max_g = jnp.where(
        threshold_enabled,
        jnp.maximum(
            jnp.asarray(tol, dtype=max_g_initial.dtype),
            max_g_initial * jnp.asarray(problem.config.minimizer_relative_tol, dtype=max_g_initial.dtype),
        ),
        jnp.asarray(tol, dtype=max_g_initial.dtype),
    )
    tau_initial = _initial_tau(problem, h_initial)

    carry = {
        "m": m_initial,
        "h": h_initial,
        "g": g_initial,
        "m_prev": m_initial,
        "g_prev": g_initial,
        "tau": tau_initial,
        "has_history": jnp.asarray(False),
        "n_iter": jnp.asarray(0, dtype=jnp.int32),
        "max_g": max_g_initial,
        "initial_max_g": max_g_initial,
        "target_max_g": target_max_g,
        "best_max_g": max_g_initial,
        "stall_count": jnp.asarray(0, dtype=jnp.int32),
    }

    def cond_fn(inner_carry):
        finite_max_g = jnp.isfinite(inner_carry["max_g"])
        converged = threshold_enabled & (inner_carry["max_g"] <= inner_carry["target_max_g"])
        stalled = (
            jnp.asarray(stall_patience > 0)
            & (inner_carry["n_iter"] >= jnp.asarray(min_iter, dtype=jnp.int32))
            & (inner_carry["stall_count"] >= jnp.asarray(stall_patience, dtype=jnp.int32))
        )
        return (inner_carry["n_iter"] < max_iter) & finite_max_g & ~converged & ~stalled

    def body_fn(inner_carry):
        m = inner_carry["m"]
        h = inner_carry["h"]
        g = inner_carry["g"]
        tau = _bb_tau(
            problem,
            m,
            g,
            inner_carry["m_prev"],
            inner_carry["g_prev"],
            inner_carry["tau"],
            inner_carry["n_iter"],
            inner_carry["has_history"],
            method=method,
        )
        m_candidate = project_m_to_support(problem, _cayley_update(m, h, g, tau))
        h_candidate = problem.h_with_axes(m_candidate, axis1_field, axis2_field, axis3_field)
        g_candidate = _descent_direction(m_candidate, h_candidate)
        max_g_candidate = _field_max_norm(g_candidate)
        improved = jnp.isfinite(max_g_candidate) & (
            max_g_candidate
            < inner_carry["best_max_g"] * jnp.asarray(1.0 - stall_rel_improvement, dtype=max_g_candidate.dtype)
        )
        best_max_g_candidate = jnp.where(
            improved,
            max_g_candidate,
            inner_carry["best_max_g"],
        )
        stall_count_candidate = jnp.where(
            improved,
            jnp.asarray(0, dtype=jnp.int32),
            inner_carry["stall_count"] + jnp.asarray(1, dtype=jnp.int32),
        )
        return {
            "m": m_candidate,
            "h": h_candidate,
            "g": g_candidate,
            "m_prev": m,
            "g_prev": g,
            "tau": tau,
            "has_history": jnp.asarray(True),
            "n_iter": inner_carry["n_iter"] + jnp.asarray(1, dtype=jnp.int32),
            "max_g": max_g_candidate,
            "initial_max_g": inner_carry["initial_max_g"],
            "target_max_g": inner_carry["target_max_g"],
            "best_max_g": best_max_g_candidate,
            "stall_count": stall_count_candidate,
        }

    result = jax.lax.while_loop(cond_fn, body_fn, carry)
    converged = threshold_enabled & jnp.isfinite(result["max_g"]) & (result["max_g"] <= result["target_max_g"])
    stalled = (
        jnp.asarray(stall_patience > 0)
        & (result["n_iter"] >= jnp.asarray(min_iter, dtype=jnp.int32))
        & (result["stall_count"] >= jnp.asarray(stall_patience, dtype=jnp.int32))
    )
    info = {
        "n_iter": result["n_iter"],
        "max_g": result["max_g"],
        "initial_max_g": result["initial_max_g"],
        "target_max_g": result["target_max_g"],
        "best_max_g": result["best_max_g"],
        "converged": converged,
        "stalled": stalled & ~converged,
        "max_iter_limited": (result["n_iter"] >= max_iter) & ~converged & ~stalled,
    }
    return result["m"], info


def phase_loss_after_relax(
    problem: EquilibriumOrientationProblem,
    axis_angles: jax.Array,
    m0_xyz: jax.Array,
    *,
    max_iter: int | None = None,
    tol: float | None = None,
    method: str = "bb2",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return phase loss after BB2/Cayley relaxation plus auxiliary diagnostics."""
    m_relaxed, relax_info = relax_magnetization(
        problem,
        axis_angles,
        m0_xyz,
        max_iter=max_iter,
        tol=tol,
        method=method,
    )
    phase_pred = phase_from_relaxed_m(problem, m_relaxed)
    residual = phase_pred - problem.target.phase_target
    residual_mse = _mean_squared_residual(residual)
    loss = 0.5 * residual_mse
    aux = {
        "phase_rms": jnp.sqrt(residual_mse),
        "phase_pred": phase_pred,
        "m_relaxed": m_relaxed,
        **relax_info,
    }
    return loss, aux


def phase_loss_and_axis_grad(
    problem: EquilibriumOrientationProblem,
    axis_angles: jax.Array,
    m0_xyz: jax.Array,
    *,
    max_iter: int | None = None,
    tol: float | None = None,
    method: str = "bb2",
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Differentiate the post-relaxation phase loss with respect to axis angles."""
    def scalar_loss_for_angles(angles):
        loss, _ = phase_loss_after_relax(
            problem,
            angles,
            m0_xyz,
            max_iter=max_iter,
            tol=tol,
            method=method,
        )
        return loss

    loss, aux = phase_loss_after_relax(
        problem,
        axis_angles,
        m0_xyz,
        max_iter=max_iter,
        tol=tol,
        method=method,
    )
    grad = jax.jacfwd(scalar_loss_for_angles)(jnp.asarray(axis_angles, dtype=jnp.float32))
    return loss, grad, aux


def finite_difference_axis_gradient_check(
    problem: EquilibriumOrientationProblem,
    axis_angles: jax.Array,
    m0_xyz: jax.Array,
    *,
    eps: float = 1e-2,
    max_iter: int = 5,
) -> dict[str, Any]:
    """Compare jacfwd axis gradients against finite differences."""
    axis_angles = jnp.asarray(axis_angles, dtype=jnp.float32)
    _, jac_grad, aux = phase_loss_and_axis_grad(
        problem,
        axis_angles,
        m0_xyz,
        max_iter=max_iter,
        tol=-1.0,
    )
    jac_grad_np = np.asarray(jac_grad, dtype=np.float32)
    fd_grad = []
    for idx in range(2):
        step = jnp.zeros_like(axis_angles).at[idx].set(eps)
        f_plus, _ = phase_loss_after_relax(problem, axis_angles + step, m0_xyz, max_iter=max_iter, tol=-1.0)
        f_minus, _ = phase_loss_after_relax(problem, axis_angles - step, m0_xyz, max_iter=max_iter, tol=-1.0)
        fd_grad.append(float((f_plus - f_minus) / (2.0 * eps)))
    fd_grad_np = np.asarray(fd_grad, dtype=np.float32)
    return {
        "jacfwd_grad": jac_grad_np,
        "finite_difference_grad": fd_grad_np,
        "abs_error": np.abs(jac_grad_np - fd_grad_np),
        "phase_rms": float(np.asarray(aux["phase_rms"])),
        "n_iter": int(np.asarray(aux["n_iter"])),
        "converged": bool(np.asarray(aux["converged"])),
        "max_iter": int(max_iter),
    }


def one_step_match_check(
    problem: EquilibriumOrientationProblem,
    axis_angles: jax.Array,
    m0_xyz: jax.Array,
) -> dict[str, float | int]:
    """Compare one custom BB2/Cayley step against NeuralMag's solver step."""
    axis1_field, axis2_field, axis3_field = broadcast_anisotropy_basis(problem, axis_angles)
    problem.state.material.Kc_axis1.tensor = problem.state.tensor(np.asarray(axis1_field, dtype=np.float32))
    problem.state.material.Kc_axis2.tensor = problem.state.tensor(np.asarray(axis2_field, dtype=np.float32))
    problem.state.material.Kc_axis3.tensor = problem.state.tensor(np.asarray(axis3_field, dtype=np.float32))
    problem.state.m.tensor = project_m_to_support(problem, jnp.asarray(m0_xyz, dtype=jnp.float32))

    solver_kwargs = dict(
        method="bb2",
        update="cayley",
        tol=problem.config.minimizer_tol,
        max_iter=1,
        tau_min=problem.config.minimizer_tau_min,
        tau_max=problem.config.minimizer_tau_max,
    )
    try:
        solver = problem.nm.EnergyMinimizer(problem.state, projection=lambda m: project_m_to_support(problem, m), **solver_kwargs)
    except TypeError as exc:
        if "projection" not in str(exc):
            raise
        solver = problem.nm.EnergyMinimizer(problem.state, **solver_kwargs)
    solver.step()
    neuralmag_one_step = np.asarray(problem.state.m.tensor, dtype=np.float32)

    jax_one_step, jax_info = relax_magnetization(problem, axis_angles, m0_xyz, max_iter=1, tol=-1.0)
    jax_one_step = np.asarray(jax_one_step, dtype=np.float32)
    return {
        "max_abs_m_diff": float(np.max(np.abs(neuralmag_one_step - jax_one_step))),
        "jax_n_iter": int(np.asarray(jax_info["n_iter"])),
        "jax_max_g": float(np.asarray(jax_info["max_g"])),
    }


def relax_magnetization_native(
    problem: EquilibriumOrientationProblem,
    axis_angles: jax.Array,
    m0_xyz: jax.Array,
    *,
    max_iter: int | None = None,
    tol: float | None = None,
    method: str = "alternating",
    update: str = "cayley",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Relax one magnetization field with NeuralMag's public minimizer.

    This uses the same BB2/Cayley family as :func:`relax_magnetization`, but
    delegates the actual solve to NeuralMag's native `EnergyMinimizer` so the
    notebook can validate custom differentiable relaxations against a trusted
    equilibrium solve.
    """
    max_iter = problem.config.minimizer_max_iter if max_iter is None else int(max_iter)
    tol = problem.config.minimizer_tol if tol is None else float(tol)

    axis1_field, axis2_field, axis3_field = broadcast_anisotropy_basis(problem, axis_angles)
    m_initial = project_m_to_support(problem, m0_xyz)
    h_initial = problem.h_with_axes(m_initial, axis1_field, axis2_field, axis3_field)
    g_initial = _descent_direction(m_initial, h_initial)
    max_g_initial = _field_max_norm(g_initial)
    threshold_enabled = tol >= 0.0
    target_max_g = (
        jnp.maximum(
            jnp.asarray(tol, dtype=max_g_initial.dtype),
            max_g_initial * jnp.asarray(problem.config.minimizer_relative_tol, dtype=max_g_initial.dtype),
        )
        if threshold_enabled
        else jnp.asarray(tol, dtype=max_g_initial.dtype)
    )

    problem.state.material.Kc_axis1.tensor = problem.state.tensor(np.asarray(axis1_field, dtype=np.float32))
    problem.state.material.Kc_axis2.tensor = problem.state.tensor(np.asarray(axis2_field, dtype=np.float32))
    problem.state.material.Kc_axis3.tensor = problem.state.tensor(np.asarray(axis3_field, dtype=np.float32))
    problem.state.m.tensor = problem.state.tensor(np.asarray(m_initial, dtype=np.float32))

    solver_kwargs = dict(
        method=method,
        update=update,
        tol=float(np.asarray(target_max_g)),
        max_iter=max_iter,
        tau_min=problem.config.minimizer_tau_min,
        tau_max=problem.config.minimizer_tau_max,
    )
    minimizer_uses_projection = True
    try:
        solver = problem.nm.EnergyMinimizer(
            problem.state,
            projection=lambda m: project_m_to_support(problem, m),
            **solver_kwargs,
        )
    except TypeError as exc:
        if "projection" not in str(exc):
            raise
        solver = problem.nm.EnergyMinimizer(problem.state, **solver_kwargs)
        minimizer_uses_projection = False

    try:
        max_g_value, native_info = solver.minimize(return_info=True)
        n_iter = int(np.asarray(native_info["n_iter"]))
    except TypeError:
        max_g_value = solver.minimize()
        n_iter = int(getattr(solver, "n_iter", max_iter))

    m_projected = project_m_to_support(problem, problem.state.m.tensor)
    if not minimizer_uses_projection:
        problem.state.m.tensor = problem.state.tensor(np.asarray(m_projected, dtype=np.float32))

    max_g_value = float(np.asarray(max_g_value))
    target_max_g_value = float(np.asarray(target_max_g))
    converged = bool(threshold_enabled and np.isfinite(max_g_value) and (max_g_value <= target_max_g_value))
    info = {
        "method": method,
        "update": update,
        "n_iter": n_iter,
        "max_g": max_g_value,
        "initial_max_g": float(np.asarray(max_g_initial)),
        "target_max_g": target_max_g_value,
        "best_max_g": max_g_value,
        "converged": converged,
        "stalled": False,
        "max_iter_limited": bool((n_iter >= max_iter) and (not converged)),
    }
    return np.asarray(m_projected, dtype=np.float32), info


def phase_loss_after_native_relax(
    problem: EquilibriumOrientationProblem,
    axis_angles: jax.Array,
    m0_xyz: jax.Array,
    *,
    max_iter: int | None = None,
    tol: float | None = None,
    method: str = "alternating",
    update: str = "cayley",
) -> tuple[jax.Array, dict[str, jax.Array | float | int | bool]]:
    """Return phase loss after relaxation with NeuralMag's public minimizer."""
    m_relaxed, relax_info = relax_magnetization_native(
        problem,
        axis_angles,
        m0_xyz,
        max_iter=max_iter,
        tol=tol,
        method=method,
        update=update,
    )
    m_relaxed_jax = jnp.asarray(m_relaxed, dtype=jnp.float32)
    phase_pred = phase_from_relaxed_m(problem, m_relaxed_jax)
    residual = phase_pred - problem.target.phase_target
    residual_mse = _mean_squared_residual(residual)
    loss = 0.5 * residual_mse
    aux = {
        "phase_rms": jnp.sqrt(residual_mse),
        "phase_pred": phase_pred,
        "m_relaxed": m_relaxed_jax,
        **relax_info,
    }
    return loss, aux


def fit_axis_from_phase(
    problem: EquilibriumOrientationProblem,
    initial_axis_angles: np.ndarray,
    m0_xyz: np.ndarray,
    *,
    outer_steps: int | None = None,
    axis_learning_rate: float | None = None,
) -> dict[str, Any]:
    """Fit axis angles from phase by differentiating through BB2/Cayley relaxation."""
    outer_steps = problem.config.outer_steps if outer_steps is None else int(outer_steps)
    axis_learning_rate = problem.config.axis_learning_rate if axis_learning_rate is None else float(axis_learning_rate)

    # JIT-compile the full step (jacfwd + while_loop + phase FFT) once, closing
    # over problem and m0_xyz as compile-time constants so JAX can fuse the
    # entire computation into a single kernel instead of re-tracing each outer step.
    _m0 = jnp.asarray(m0_xyz, dtype=jnp.float32)

    @jax.jit
    def _jitted_loss_and_grad(angles: jax.Array):
        return phase_loss_and_axis_grad(problem, angles, _m0)

    optimizer = optax.adam(axis_learning_rate)
    axis_angles = jnp.asarray(initial_axis_angles, dtype=jnp.float32)
    opt_state = optimizer.init(axis_angles)
    history: list[dict[str, Any]] = []
    best_loss = np.inf
    best_axis_angles = np.asarray(axis_angles, dtype=np.float32)
    best_row: dict[str, Any] | None = None

    for outer_step in range(outer_steps + 1):
        loss, grad, aux = _jitted_loss_and_grad(axis_angles)
        axis1, axis2, axis3 = angle_params_to_anisotropy_axes(
            axis_angles,
            axis2_hint=jnp.asarray(problem.config.axis2_hint, dtype=jnp.float32),
        )
        grad_norm = float(np.asarray(_safe_norm(grad, eps=1e-12)))
        row = {
            "outer_step": int(outer_step),
            "loss": float(np.asarray(loss)),
            "phase_rms": float(np.asarray(aux["phase_rms"])),
            "axis_angles": np.asarray(axis_angles, dtype=np.float32),
            "axis1": np.asarray(axis1, dtype=np.float32),
            "axis2": np.asarray(axis2, dtype=np.float32),
            "axis3": np.asarray(axis3, dtype=np.float32),
            "grad": np.asarray(grad, dtype=np.float32),
            "grad_norm": grad_norm,
            "n_iter": int(np.asarray(aux["n_iter"])),
            "max_g": float(np.asarray(aux["max_g"])),
            "initial_max_g": float(np.asarray(aux["initial_max_g"])),
            "target_max_g": float(np.asarray(aux["target_max_g"])),
            "best_max_g": float(np.asarray(aux["best_max_g"])),
            "converged": bool(np.asarray(aux["converged"])),
            "stalled": bool(np.asarray(aux["stalled"])),
            "max_iter_limited": bool(np.asarray(aux["max_iter_limited"])),
        }
        history.append(row)

        row_is_finite = (
            np.isfinite(row["loss"])
            and np.isfinite(row["phase_rms"])
            and np.all(np.isfinite(row["axis_angles"]))
            and np.all(np.isfinite(row["axis1"]))
        )
        if row_is_finite and row["loss"] < best_loss:
            best_loss = row["loss"]
            best_axis_angles = row["axis_angles"]
            best_row = dict(row)

        if outer_step == outer_steps:
            break
        if not row_is_finite or not np.all(np.isfinite(row["grad"])):
            break
        updates, opt_state = optimizer.update(grad, opt_state, axis_angles)
        axis_angles = optax.apply_updates(axis_angles, updates)
        if not np.all(np.isfinite(np.asarray(axis_angles))):
            break

    best_axis_angles_jax = jnp.asarray(best_axis_angles, dtype=jnp.float32)
    best_axis1, best_axis2, best_axis3 = angle_params_to_anisotropy_axes(
        best_axis_angles_jax,
        axis2_hint=jnp.asarray(problem.config.axis2_hint, dtype=jnp.float32),
    )
    final = dict(best_row or history[-1])
    final["axis_angles"] = np.asarray(best_axis_angles_jax, dtype=np.float32)
    final["axis1"] = np.asarray(best_axis1, dtype=np.float32)
    final["axis2"] = np.asarray(best_axis2, dtype=np.float32)
    final["axis3"] = np.asarray(best_axis3, dtype=np.float32)
    final_loss, final_aux = phase_loss_after_relax(problem, best_axis_angles_jax, m0_xyz)
    final["loss"] = float(np.asarray(final_loss))
    final["phase_rms"] = float(np.asarray(final_aux["phase_rms"]))
    final["phase_pred"] = np.asarray(final_aux["phase_pred"], dtype=np.float32)
    final["m_relaxed"] = np.asarray(final_aux["m_relaxed"], dtype=np.float32)
    final["n_iter"] = int(np.asarray(final_aux["n_iter"]))
    final["max_g"] = float(np.asarray(final_aux["max_g"]))
    final["initial_max_g"] = float(np.asarray(final_aux["initial_max_g"]))
    final["target_max_g"] = float(np.asarray(final_aux["target_max_g"]))
    final["best_max_g"] = float(np.asarray(final_aux["best_max_g"]))
    final["converged"] = bool(np.asarray(final_aux["converged"]))
    final["stalled"] = bool(np.asarray(final_aux["stalled"]))
    final["max_iter_limited"] = bool(np.asarray(final_aux["max_iter_limited"]))
    return {"history": history, "final": final}


def make_vmapped_multi_start(
    problem: EquilibriumOrientationProblem,
    m0_xyz: np.ndarray | jax.Array,
) -> Any:
    """Build a JIT+vmap compiled function for parallel multi-start axis fitting.

    Uses :func:`jax.lax.scan` for the outer Adam loop and :func:`jax.vmap`
    over starting angle parameters, so all N starts execute in a single
    compiled kernel rather than a Python for-loop.

    Parameters
    ----------
    problem:
        Resolved orientation problem (from
        :func:`build_equilibrium_orientation_problem`).
    m0_xyz:
        Initial magnetization array ``(nx, ny, nz, 3)``.

    Returns
    -------
    callable
        ``run(init_angles_batch)`` where ``init_angles_batch`` has shape
        ``(N, 2)``.  Returns
        ``(final_angles, loss_history, rms_history, axis1_history)`` with
        shapes ``(N, 2)``, ``(N, S)``, ``(N, S)``, ``(N, S, 3)`` where
        ``S = config.outer_steps``.  Pick the best start via
        ``jnp.argmin(loss_history[:, -1])`` then call
        :func:`phase_loss_after_relax` on the winning angles for full
        diagnostics (``phase_pred``, convergence info, etc.).
    """
    _m0 = jnp.asarray(m0_xyz, dtype=jnp.float32)
    outer_steps = problem.config.outer_steps
    _axis2_hint = jnp.asarray(problem.config.axis2_hint, dtype=jnp.float32)
    optimizer = optax.adam(problem.config.axis_learning_rate)

    def run_one_start(init_angles: jax.Array) -> tuple:
        opt_state = optimizer.init(init_angles)

        def scan_body(carry: tuple, _: Any) -> tuple:
            angles, opt_state_ = carry
            loss, grad, aux = phase_loss_and_axis_grad(problem, angles, _m0)
            # Cast to float32 so the carry dtype is stable under lax.scan when
            # jax_enable_x64=True causes intermediate computations to promote.
            grad = grad.astype(jnp.float32)
            loss = loss.astype(jnp.float32)
            phase_rms = aux["phase_rms"].astype(jnp.float32)
            axis1, _, _ = angle_params_to_anisotropy_axes(angles, axis2_hint=_axis2_hint)
            updates, new_opt_state = optimizer.update(grad, opt_state_, angles)
            new_angles = optax.apply_updates(angles, updates).astype(jnp.float32)
            return (new_angles, new_opt_state), (loss, phase_rms, axis1)

        (final_angles, _), (loss_history, rms_history, axis1_history) = jax.lax.scan(
            scan_body, (init_angles, opt_state), None, length=outer_steps
        )
        return final_angles, loss_history, rms_history, axis1_history

    return jax.jit(jax.vmap(run_one_start))


def make_vmapped_steepest_descent_multi_start(
    problem: EquilibriumOrientationProblem,
    m0_xyz: np.ndarray | jax.Array,
    *,
    learning_rate: float | None = None,
    outer_steps: int | None = None,
    normalize_gradients: bool = True,
) -> Any:
    """Build a JIT+vmap steepest-descent runner for parallel multi-start axis fitting.

    This is a minimal outer optimizer intended for notebook workflows that want
    to try a fixed batch of crystallographic candidate starts without carrying a
    notebook-local optimization loop. It reuses the same differentiable phase
    loss and BB2/Cayley inner relaxation as :func:`fit_axis_from_phase`, but the
    outer update is plain steepest descent on the axis-angle parameters.

    Parameters
    ----------
    problem:
        Resolved orientation problem.
    m0_xyz:
        Initial magnetization array ``(nx, ny, nz, 3)``.
    learning_rate:
        Outer steepest-descent step size. Defaults to
        ``problem.config.axis_learning_rate``.
    outer_steps:
        Number of steepest-descent steps. Defaults to
        ``problem.config.outer_steps``.
    normalize_gradients:
        If true, step along the unit steepest-descent direction to keep the
        outer update scale predictable across starts.

    Returns
    -------
    callable
        ``run(init_angles_batch)`` where ``init_angles_batch`` has shape
        ``(N, 2)``. Returns
        ``(final_angles, final_losses, final_rms, loss_history, rms_history, axis1_history)``
        with shapes ``(N, 2)``, ``(N,)``, ``(N,)``, ``(N, S)``, ``(N, S)``,
        ``(N, S, 3)`` where ``S = outer_steps``.
    """
    _m0 = jnp.asarray(m0_xyz, dtype=jnp.float32)
    lr = jnp.asarray(
        problem.config.axis_learning_rate if learning_rate is None else learning_rate,
        dtype=jnp.float32,
    )
    n_steps = problem.config.outer_steps if outer_steps is None else int(outer_steps)
    _axis2_hint = jnp.asarray(problem.config.axis2_hint, dtype=jnp.float32)
    eps = jnp.asarray(1e-12, dtype=jnp.float32)

    def run_one_start(init_angles: jax.Array) -> tuple:
        def scan_body(angles: jax.Array, _: Any) -> tuple:
            loss, grad, aux = phase_loss_and_axis_grad(problem, angles, _m0)
            grad = grad.astype(jnp.float32)
            if normalize_gradients:
                grad = grad / jnp.maximum(_safe_norm(grad, eps=float(eps)), eps)
            new_angles = (angles - lr * grad).astype(jnp.float32)
            axis1, _, _ = angle_params_to_anisotropy_axes(angles, axis2_hint=_axis2_hint)
            outputs = (
                loss.astype(jnp.float32),
                aux["phase_rms"].astype(jnp.float32),
                axis1.astype(jnp.float32),
            )
            return new_angles, outputs

        final_angles, (loss_history, rms_history, axis1_history) = jax.lax.scan(
            scan_body,
            jnp.asarray(init_angles, dtype=jnp.float32),
            None,
            length=n_steps,
        )
        final_loss, final_aux = phase_loss_after_relax(problem, final_angles, _m0)
        return (
            final_angles.astype(jnp.float32),
            final_loss.astype(jnp.float32),
            final_aux["phase_rms"].astype(jnp.float32),
            loss_history,
            rms_history,
            axis1_history,
        )

    return jax.jit(jax.vmap(run_one_start))


__all__ = [
    "EquilibriumOrientationFitConfig",
    "EquilibriumOrientationFitTarget",
    "EquilibriumOrientationProblem",
    "angular_distance_deg",
    "build_equilibrium_orientation_problem",
    "coarse_grain_volume_xyz",
    "ensure_neuralmag_jax_backend",
    "finite_difference_axis_gradient_check",
    "fit_axis_from_phase",
    "make_vmapped_multi_start",
    "make_vmapped_steepest_descent_multi_start",
    "one_step_match_check",
    "phase_from_relaxed_m",
    "phase_loss_after_native_relax",
    "phase_loss_after_relax",
    "phase_loss_and_axis_grad",
    "prepare_equilibrium_fit_target",
    "prepare_equilibrium_fit_target_from_npz",
    "project_m_to_support",
    "relax_magnetization_native",
    "relax_magnetization",
    "unit_vector_to_axis_angles",
]

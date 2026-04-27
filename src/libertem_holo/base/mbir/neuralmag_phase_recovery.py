"""Phase-constrained NeuralMag recovery helpers for MBIR workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import unxt as u

from .forward import forward_phase_from_density_and_magnetization
from .kernel import build_rdfc_kernel

MU0 = 4.0e-7 * np.pi

InitMode = Literal["random", "uniform_y", "target_warm_start"]
SolverFamily = Literal["neuralmag", "optax_lbfgs"]


def _default_phase_schedule() -> tuple[float, ...]:
    return (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def _default_axis1() -> np.ndarray:
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _default_axis2() -> np.ndarray:
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class NeuralMagPhaseRecoveryConfig:
    """Configuration for phase-constrained NeuralMag minimization."""

    phase_weight_schedule: Sequence[float] = field(default_factory=_default_phase_schedule)
    phase_pad: int = 32
    phase_field_fraction: float = 0.10
    phase_energy_scale: float | None = None

    projection_axis: str = "z"
    geometry: str = "disc"
    init_mode: InitMode = "random"
    rng_seed: int = 8
    support_threshold: float = 0.5

    Msat_A_per_m: float = 480e3
    Aex_J_per_m: float = 1e-11
    Kc1_J_per_m3: float = -1.35e4
    demag_p: int = 3
    anisotropy_axis1: np.ndarray = field(default_factory=_default_axis1)
    anisotropy_axis2: np.ndarray = field(default_factory=_default_axis2)

    solver_family: SolverFamily = "neuralmag"
    minimizer_method: str = "alternating"
    minimizer_update: str = "cayley"
    minimizer_tol: float = 1e-3
    minimizer_max_iter: int = 75
    minimizer_tau_min: float = 1e-18
    minimizer_tau_max: float = 1e-4
    optax_lbfgs_memory_size: int = 10
    optax_lbfgs_max_linesearch_steps: int = 20


@dataclass(frozen=True)
class NeuralMagPhaseTarget:
    """Prepared target fields and phase data for NeuralMag recovery."""

    cellsize_nm: float
    pixel_size: u.Quantity
    rho_xyz: np.ndarray
    m_target_xyz: np.ndarray
    rho_zyx: np.ndarray
    m_target_zyx: np.ndarray
    rho_zyx_view: np.ndarray
    m_target_zyx_view: np.ndarray
    phase_target: Any
    phase_mask: Any
    phase_mask_core: np.ndarray
    rdfc_kernel: Mapping[str, Any]
    has_reference_magnetization: bool = True


@dataclass(frozen=True)
class NeuralMagPhaseRecoveryResult:
    """Result of a scheduled phase-constrained NeuralMag recovery."""

    state: Any
    target: NeuralMagPhaseTarget
    config: NeuralMagPhaseRecoveryConfig
    phase_terms: Mapping[str, Any]
    history: list[dict[str, float | int | str]]
    phase_energy_scale: float
    m_recovered_xyz: np.ndarray
    phase_recovered: np.ndarray
    n_iter: int
    max_g: float


@dataclass(frozen=True)
class NeuralMagAnisotropyCandidateFit:
    """Fit summary for one anisotropy candidate."""

    name: str
    axis1: np.ndarray
    axis2: np.ndarray
    config: NeuralMagPhaseRecoveryConfig
    result: NeuralMagPhaseRecoveryResult
    initial_phase_rms: float
    initial_raw_phase_loss: float
    initial_component_rmse_xyz: np.ndarray
    final_phase_rms: float
    final_raw_phase_loss: float
    component_rmse_xyz: np.ndarray
    mean_component_error_xyz: np.ndarray


@dataclass(frozen=True)
class NeuralMagAnisotropySelectionResult:
    """Ranked anisotropy-candidate fits for a phase target."""

    best_name: str
    best_fit: NeuralMagAnisotropyCandidateFit
    fits: tuple[NeuralMagAnisotropyCandidateFit, ...]


def center_crop(values: np.ndarray, crop_shape: tuple[int, int, int] | None) -> np.ndarray:
    """Center-crop an array over its first three axes."""
    if crop_shape is None:
        return values
    starts = [(s - min(c, s)) // 2 for s, c in zip(values.shape[:3], crop_shape)]
    stops = [start + min(c, s) for start, s, c in zip(starts, values.shape[:3], crop_shape)]
    slices = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    return values[slices] if values.ndim == 3 else values[slices + (slice(None),)]


def xyz_to_zyx(values: np.ndarray) -> np.ndarray:
    """Convert NeuralMag-style XYZ arrays to MBIR ZYX arrays."""
    axes = (2, 1, 0) if values.ndim == 3 else (2, 1, 0, 3)
    return np.transpose(values, axes)


def normalize_on_support(
    m_cell_xyz: np.ndarray,
    rho_cell_xyz: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Normalize magnetization to unit length on support and zero elsewhere."""
    m = np.asarray(m_cell_xyz, dtype=np.float32)
    rho = np.asarray(rho_cell_xyz, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    normalized = m / np.where(norms > 0.0, norms, 1.0)
    return np.where(rho[..., None] > threshold, normalized, 0.0).astype(np.float32)


def pad_for_phase_view(
    rho_zyx: np.ndarray,
    m_zyx: np.ndarray,
    pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad ZYX rho and magnetization arrays in the projected Y/X plane."""
    if pad <= 0:
        return rho_zyx, m_zyx
    rho_view = np.pad(rho_zyx, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    m_view = np.pad(m_zyx, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    return rho_view, m_view


def orthonormalize_anisotropy_axes(
    axis1: np.ndarray,
    axis2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormalize two anisotropy axes with Gram-Schmidt."""
    axis1 = np.asarray(axis1, dtype=np.float32)
    axis2 = np.asarray(axis2, dtype=np.float32)
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 - np.dot(axis2, axis1) * axis1
    axis2 = axis2 / np.linalg.norm(axis2)
    axis3 = np.cross(axis1, axis2).astype(np.float32)
    axis3 = axis3 / np.linalg.norm(axis3)
    return axis1, axis2, axis3


def make_initial_m_cell(
    rho_cell_xyz: np.ndarray,
    m_target_xyz: np.ndarray,
    *,
    mode: InitMode,
    rng_seed: int = 0,
    support_threshold: float = 0.5,
) -> np.ndarray:
    """Create a cell-centered initial magnetization field."""
    if mode == "target_warm_start":
        return normalize_on_support(m_target_xyz, rho_cell_xyz, support_threshold)

    shape = rho_cell_xyz.shape + (3,)
    if mode == "random":
        rng = np.random.default_rng(rng_seed)
        m0 = rng.normal(size=shape).astype(np.float32)
    elif mode == "uniform_y":
        m0 = np.zeros(shape, dtype=np.float32)
        m0[..., 1] = 1.0
    else:
        raise ValueError(f"Unsupported init_mode {mode!r}.")

    return normalize_on_support(m0, rho_cell_xyz, support_threshold)


def make_support_projection(
    rho_cell_xyz: np.ndarray,
    *,
    threshold: float = 0.5,
):
    """Build a JAX-compatible projection onto unit magnetization support."""
    support = jnp.asarray(np.asarray(rho_cell_xyz) > threshold)

    def projection(m):
        support_cast = support.astype(m.dtype)
        eps = jnp.asarray(jnp.finfo(m.dtype).eps, dtype=m.dtype)
        sq_norms = jnp.sum(m * m, axis=-1, keepdims=True)
        safe_norms = jnp.sqrt(jnp.maximum(sq_norms, eps * eps))
        normalized = m / safe_norms
        return jnp.where(support_cast[..., None] > 0, normalized, 0.0)

    return projection


def prepare_neuralmag_phase_target(
    rho_xyz: np.ndarray,
    m_target_xyz: np.ndarray,
    *,
    cellsize_nm: float,
    config: NeuralMagPhaseRecoveryConfig | None = None,
    crop_shape: tuple[int, int, int] | None = None,
) -> NeuralMagPhaseTarget:
    """Prepare target arrays, phase image, mask, and RDFC kernel."""
    config = config or NeuralMagPhaseRecoveryConfig()
    rho_cropped = np.asarray(center_crop(rho_xyz, crop_shape), dtype=np.float32)
    m_cropped = np.asarray(center_crop(m_target_xyz, crop_shape), dtype=np.float32)
    m_target = normalize_on_support(
        m_cropped,
        rho_cropped,
        threshold=config.support_threshold,
    )
    rho_zyx = xyz_to_zyx(rho_cropped)
    m_target_zyx = xyz_to_zyx(m_target)
    rho_zyx_view, m_target_zyx_view = pad_for_phase_view(
        rho_zyx,
        m_target_zyx,
        config.phase_pad,
    )
    pixel_size = u.Quantity(cellsize_nm, "nm")
    phase_target = jnp.asarray(
        forward_phase_from_density_and_magnetization(
            rho=rho_zyx_view,
            magnetization_3d=m_target_zyx_view,
            pixel_size=pixel_size,
            axis=config.projection_axis,
            geometry=config.geometry,
        ),
        dtype=jnp.float32,
    )
    phase_mask_core = (rho_zyx_view.max(axis=0) > config.support_threshold).astype(np.float32)
    phase_mask = jnp.ones_like(phase_target, dtype=jnp.float32)
    rdfc_kernel = build_rdfc_kernel(phase_target.shape, geometry=config.geometry)

    return NeuralMagPhaseTarget(
        cellsize_nm=float(cellsize_nm),
        pixel_size=pixel_size,
        rho_xyz=rho_cropped,
        m_target_xyz=m_target,
        rho_zyx=rho_zyx,
        m_target_zyx=m_target_zyx,
        rho_zyx_view=rho_zyx_view,
        m_target_zyx_view=m_target_zyx_view,
        phase_target=phase_target,
        phase_mask=phase_mask,
        phase_mask_core=phase_mask_core,
        rdfc_kernel=rdfc_kernel,
        has_reference_magnetization=True,
    )


def prepare_neuralmag_phase_target_from_phase_image(
    rho_xyz: np.ndarray,
    phase_target_yx: np.ndarray,
    *,
    cellsize_nm: float,
    config: NeuralMagPhaseRecoveryConfig | None = None,
    crop_shape: tuple[int, int, int] | None = None,
    m_reference_xyz: np.ndarray | None = None,
) -> NeuralMagPhaseTarget:
    """Prepare a NeuralMag phase target from a measured phase image."""
    config = config or NeuralMagPhaseRecoveryConfig()
    rho_cropped = np.asarray(center_crop(rho_xyz, crop_shape), dtype=np.float32)
    rho_zyx = xyz_to_zyx(rho_cropped)

    if m_reference_xyz is None:
        m_target = np.zeros(rho_cropped.shape + (3,), dtype=np.float32)
        has_reference_magnetization = False
    else:
        m_target = normalize_on_support(
            np.asarray(center_crop(m_reference_xyz, crop_shape), dtype=np.float32),
            rho_cropped,
            threshold=config.support_threshold,
        )
        has_reference_magnetization = True

    m_target_zyx = xyz_to_zyx(m_target)
    rho_zyx_view, m_target_zyx_view = pad_for_phase_view(
        rho_zyx,
        m_target_zyx,
        config.phase_pad,
    )
    phase_target = jnp.asarray(phase_target_yx, dtype=jnp.float32)
    expected_shape = (rho_zyx_view.shape[1], rho_zyx_view.shape[2])
    if tuple(phase_target.shape) != expected_shape:
        raise ValueError(
            "phase_target_yx shape does not match rho_xyz with configured phase_pad: "
            f"expected {expected_shape}, got {tuple(phase_target.shape)}."
        )

    pixel_size = u.Quantity(cellsize_nm, "nm")
    phase_mask_core = (rho_zyx_view.max(axis=0) > config.support_threshold).astype(np.float32)
    phase_mask = jnp.ones_like(phase_target, dtype=jnp.float32)
    rdfc_kernel = build_rdfc_kernel(phase_target.shape, geometry=config.geometry)

    return NeuralMagPhaseTarget(
        cellsize_nm=float(cellsize_nm),
        pixel_size=pixel_size,
        rho_xyz=rho_cropped,
        m_target_xyz=m_target,
        rho_zyx=rho_zyx,
        m_target_zyx=m_target_zyx,
        rho_zyx_view=rho_zyx_view,
        m_target_zyx_view=m_target_zyx_view,
        phase_target=phase_target,
        phase_mask=phase_mask,
        phase_mask_core=phase_mask_core,
        rdfc_kernel=rdfc_kernel,
        has_reference_magnetization=has_reference_magnetization,
    )


def _predict_phase_from_m(
    m_cell_xyz,
    target: NeuralMagPhaseTarget,
    config: NeuralMagPhaseRecoveryConfig,
):
    m_zyx = jnp.transpose(m_cell_xyz, (2, 1, 0, 3))
    pad = config.phase_pad
    if pad > 0:
        m_zyx = jnp.pad(m_zyx, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    return forward_phase_from_density_and_magnetization(
        rho=jnp.asarray(target.rho_zyx_view, dtype=m_cell_xyz.dtype),
        magnetization_3d=m_zyx,
        pixel_size=target.pixel_size,
        axis=config.projection_axis,
        geometry=config.geometry,
        rdfc_kernel=target.rdfc_kernel,
    )


def predict_neuralmag_phase(
    m_cell_xyz,
    target: NeuralMagPhaseTarget,
    config: NeuralMagPhaseRecoveryConfig,
):
    """Predict the MBIR phase image for a NeuralMag cell magnetization."""
    return _predict_phase_from_m(m_cell_xyz, target, config)


def _make_phase_field_callables(
    target: NeuralMagPhaseTarget,
    config: NeuralMagPhaseRecoveryConfig,
    *,
    phase_energy_scale: float,
) -> dict[str, Any]:
    target_const = jnp.asarray(target.phase_target, dtype=jnp.float32)
    mask_const = jnp.asarray(target.phase_mask, dtype=jnp.float32)
    denom = jnp.maximum(jnp.sum(mask_const), 1.0)
    field_scale = jnp.asarray(MU0 * config.Msat_A_per_m, dtype=jnp.float32)

    def raw_phase_loss(m):
        pred = _predict_phase_from_m(m, target, config)
        residual = (pred - target_const) * mask_const
        return 0.5 * jnp.sum(residual * residual) / denom

    grad_raw_phase_loss = jax.grad(raw_phase_loss)

    def E_phase(m, phase_scale):
        scale = jnp.asarray(phase_scale, dtype=m.dtype)
        return scale * raw_phase_loss(m)

    def h_phase(m, phase_scale):
        scale = jnp.asarray(phase_scale, dtype=m.dtype)
        grad = grad_raw_phase_loss(m)
        return -scale * grad / jnp.maximum(field_scale, jnp.finfo(m.dtype).eps)

    def e_phase(m):
        return jnp.zeros(m.shape[:-1], dtype=m.dtype)

    return {
        "phase_energy_scale": float(phase_energy_scale),
        "raw_phase_loss": jax.jit(raw_phase_loss),
        "E_phase": jax.jit(E_phase),
        "h_phase": jax.jit(h_phase),
        "e_phase": e_phase,
    }


def _set_phase_scale(
    state,
    phase_terms: Mapping[str, Any],
    *,
    lambda_phase: float,
    phase_energy_scale: float,
) -> None:
    """Update the runtime phase-energy scale on a reused NeuralMag state."""
    state.phase_scale = state.tensor(float(lambda_phase * phase_energy_scale))
    phase_terms["lambda_phase"] = float(lambda_phase)


def _build_neuralmag_state(
    target: NeuralMagPhaseTarget,
    config: NeuralMagPhaseRecoveryConfig,
    *,
    lambda_phase: float,
    phase_energy_scale: float,
    m0_cell_xyz: np.ndarray | None,
) -> tuple[Any, Mapping[str, Any]]:
    try:
        import neuralmag as nm
    except ImportError as exc:
        raise ImportError("run_neuralmag_phase_recovery requires neuralmag.") from exc
    _ = nm.config.backend
    if nm.config.backend_name != "jax":
        raise ValueError("NeuralMag phase recovery currently requires the JAX backend.")

    nx, ny, nz = target.rho_xyz.shape
    state = nm.State(nm.Mesh((nx, ny, nz), (target.cellsize_nm * 1e-9,) * 3))
    rho_min = float(getattr(state, "eps", 1e-12))
    rho_safe = np.clip(target.rho_xyz, rho_min, 1.0)
    state.rho = nm.CellFunction(state, tensor=state.tensor(rho_safe))

    state.material.Ms = nm.CellFunction(state).fill(config.Msat_A_per_m)
    state.material.A = nm.CellFunction(state).fill(config.Aex_J_per_m)
    state.material.Kc = nm.CellFunction(state).fill(config.Kc1_J_per_m3)

    axis1_vec, axis2_vec, axis3_vec = orthonormalize_anisotropy_axes(
        config.anisotropy_axis1,
        config.anisotropy_axis2,
    )
    axis_shape = target.rho_xyz.shape + (3,)
    axis1 = np.broadcast_to(axis1_vec, axis_shape).astype(np.float32)
    axis2 = np.broadcast_to(axis2_vec, axis_shape).astype(np.float32)
    axis3 = np.broadcast_to(axis3_vec, axis_shape).astype(np.float32)
    state.material.Kc_axis1 = nm.VectorCellFunction(state, tensor=state.tensor(axis1))
    state.material.Kc_axis2 = nm.VectorCellFunction(state, tensor=state.tensor(axis2))
    state.material.Kc_axis3 = nm.VectorCellFunction(state, tensor=state.tensor(axis3))

    if m0_cell_xyz is None:
        if config.init_mode == "target_warm_start" and not target.has_reference_magnetization:
            raise ValueError(
                "target_warm_start requires target reference magnetization; "
                "pass m0_cell_xyz explicitly or use a different init_mode."
            )
        m0_cell_xyz = make_initial_m_cell(
            target.rho_xyz,
            target.m_target_xyz,
            mode=config.init_mode,
            rng_seed=config.rng_seed,
            support_threshold=config.support_threshold,
        )
    else:
        m0_cell_xyz = normalize_on_support(
            m0_cell_xyz,
            target.rho_xyz,
            threshold=config.support_threshold,
        )
    state.m = nm.VectorCellFunction(state, tensor=state.tensor(m0_cell_xyz.astype(np.float32)))

    nm.ExchangeField().register(state, "exchange")
    nm.DemagField(p=config.demag_p).register(state, "demag")
    nm.CubicAnisotropyField().register(state, "cubic")
    nm.TotalField("exchange", "demag", "cubic").register(state, "micromagnetic")

    phase_terms = _make_phase_field_callables(
        target,
        config,
        phase_energy_scale=phase_energy_scale,
    )
    _set_phase_scale(
        state,
        phase_terms,
        lambda_phase=lambda_phase,
        phase_energy_scale=phase_energy_scale,
    )
    state.h_phase = nm.VectorCellFunction(state, tensor=phase_terms["h_phase"])
    state.e_phase = nm.CellFunction(state, tensor=phase_terms["e_phase"])
    state.E_phase = phase_terms["E_phase"]
    nm.TotalField("exchange", "demag", "cubic", "phase").register(state)

    return state, phase_terms


def _current_max_gradient(state) -> float:
    h = state.h.tensor
    m = state.m.tensor
    m_dot_h = jnp.sum(m * h, axis=-1, keepdims=True)
    m_dot_m = jnp.sum(m * m, axis=-1, keepdims=True)
    g = m * m_dot_h - h * m_dot_m
    return float(np.asarray(jnp.sqrt(jnp.sum(g * g, axis=-1).max())))


def _make_projected_energy_objective(state, projection):
    """Build the projected total-energy objective used by Optax solvers."""
    energy_fn = jax.jit(state.resolve("E", ["m"]))

    def objective(m):
        return energy_fn(projection(m))

    return objective


def _run_neuralmag_minimizer_stage(state, config: NeuralMagPhaseRecoveryConfig, projection) -> tuple[int, float]:
    """Run one scheduled stage using NeuralMag's built-in BB minimizer."""
    import neuralmag as nm

    minimizer_kwargs = {
        "method": config.minimizer_method,
        "update": config.minimizer_update,
        "tol": config.minimizer_tol,
        "max_iter": config.minimizer_max_iter,
        "tau_min": config.minimizer_tau_min,
        "tau_max": config.minimizer_tau_max,
    }
    minimizer_uses_projection = True
    try:
        minimizer = nm.EnergyMinimizer(
            state,
            projection=projection,
            **minimizer_kwargs,
        )
    except TypeError:
        minimizer = nm.EnergyMinimizer(
            state,
            **minimizer_kwargs,
        )
        minimizer_uses_projection = False

    try:
        max_g_value, info = minimizer.minimize(return_info=True)
        n_iter = int(np.asarray(info["n_iter"]))
    except TypeError:
        max_g_value = minimizer.minimize()
        n_iter = int(minimizer.n_iter)
    if not minimizer_uses_projection:
        state.m.tensor = projection(state.m.tensor)
    return n_iter, float(np.asarray(max_g_value))


def _run_optax_lbfgs_stage(state, config: NeuralMagPhaseRecoveryConfig, projection) -> tuple[int, float]:
    """Run one scheduled stage using Optax L-BFGS on the projected energy."""
    try:
        import optax
    except ImportError as exc:
        raise ImportError("solver_family='optax_lbfgs' requires optax.") from exc

    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=config.optax_lbfgs_max_linesearch_steps,
        initial_guess_strategy="one",
    )
    solver = optax.lbfgs(
        memory_size=config.optax_lbfgs_memory_size,
        linesearch=linesearch,
    )
    objective = _make_projected_energy_objective(state, projection)

    value_and_grad = optax.value_and_grad_from_state(objective)
    params = projection(state.m.tensor)
    state.m.tensor = params
    opt_state = solver.init(params)
    max_g = _current_max_gradient(state)
    n_iter = 0

    for iteration in range(config.minimizer_max_iter):
        if not np.isfinite(max_g) or max_g <= config.minimizer_tol:
            break
        value, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad,
            opt_state,
            params,
            value=value,
            grad=grad,
            value_fn=objective,
        )
        params = projection(optax.apply_updates(params, updates))
        state.m.tensor = params
        max_g = _current_max_gradient(state)
        n_iter = iteration + 1

    return n_iter, max_g


def _run_solver_stage(state, config: NeuralMagPhaseRecoveryConfig, projection) -> tuple[int, float]:
    """Dispatch one scheduled stage to the configured solver family."""
    if config.solver_family == "neuralmag":
        return _run_neuralmag_minimizer_stage(state, config, projection)
    if config.solver_family == "optax_lbfgs":
        return _run_optax_lbfgs_stage(state, config, projection)
    raise ValueError(f"Unsupported solver_family {config.solver_family!r}.")


def _resolve_phase_recovery_target(
    rho_xyz: np.ndarray | NeuralMagPhaseTarget,
    m_target_xyz: np.ndarray | None,
    *,
    cellsize_nm: float | None,
    config: NeuralMagPhaseRecoveryConfig,
    crop_shape: tuple[int, int, int] | None,
) -> NeuralMagPhaseTarget:
    """Normalize raw arrays or a pre-built target to one target object."""
    if isinstance(rho_xyz, NeuralMagPhaseTarget):
        return rho_xyz
    if m_target_xyz is None or cellsize_nm is None:
        raise ValueError("m_target_xyz and cellsize_nm are required when passing raw arrays.")
    return prepare_neuralmag_phase_target(
        rho_xyz,
        m_target_xyz,
        cellsize_nm=cellsize_nm,
        config=config,
        crop_shape=crop_shape,
    )


def _resolve_phase_energy_scale(
    target: NeuralMagPhaseTarget,
    config: NeuralMagPhaseRecoveryConfig,
    phase_schedule: tuple[float, ...],
) -> float:
    """Resolve the runtime phase-energy scale from config or calibration."""
    if config.phase_energy_scale is not None:
        return float(config.phase_energy_scale)
    return calibrate_phase_energy_scale(target, config, lambda_ref=phase_schedule[0])


def _metrics_for_state(
    state,
    phase_terms: Mapping[str, Any],
    target: NeuralMagPhaseTarget,
    config: NeuralMagPhaseRecoveryConfig,
    *,
    stage: int,
    global_step: int,
    event: str,
    max_g: float | None,
) -> dict[str, float | int | str]:
    m_cell_xyz = np.asarray(state.m.tensor, dtype=np.float32)
    support = target.rho_xyz > config.support_threshold

    pred_phase = np.asarray(_predict_phase_from_m(state.m.tensor, target, config), dtype=np.float32)
    mask = np.asarray(target.phase_mask, dtype=np.float32)
    residual = (pred_phase - np.asarray(target.phase_target, dtype=np.float32)) * mask
    phase_rms = float(np.sqrt(np.sum(residual * residual) / max(float(mask.sum()), 1.0)))

    raw_phase_loss = float(phase_terms["raw_phase_loss"](state.m.tensor))
    e_phase = float(phase_terms["E_phase"](state.m.tensor, state.phase_scale))
    e_micro = float(np.asarray(state.E_micromagnetic)) if hasattr(state, "E_micromagnetic") else np.nan
    e_total = float(np.asarray(state.E)) if hasattr(state, "E") else e_micro + e_phase

    if target.has_reference_magnetization:
        diff = m_cell_xyz[support] - target.m_target_xyz[support]
        rmse_3d = float(np.sqrt(np.mean(diff * diff))) if diff.size else np.nan

        m_zyx = xyz_to_zyx((target.rho_xyz[..., None] > config.support_threshold) * m_cell_xyz)
        projected_rec = np.sum(m_zyx, axis=0)[..., :2]
        projected_true = np.sum(
            (target.rho_zyx[..., None] > config.support_threshold) * target.m_target_zyx,
            axis=0,
        )[..., :2]
        proj_err = float(
            np.linalg.norm(projected_rec - projected_true)
            / max(np.linalg.norm(projected_true), np.finfo(np.float32).eps)
        )
    else:
        rmse_3d = np.nan
        proj_err = np.nan

    norms = np.linalg.norm(m_cell_xyz, axis=-1)
    unit_norm_error = float(np.mean(np.abs(norms[support] - 1.0))) if np.any(support) else np.nan
    off_support_leak = (
        float(np.max(np.linalg.norm(m_cell_xyz[~support], axis=-1))) if np.any(~support) else 0.0
    )

    row: dict[str, float | int | str] = {
        "stage": int(stage),
        "global_step": int(global_step),
        "event": event,
        "lambda_phase": float(phase_terms["lambda_phase"]),
        "phase_rms": phase_rms,
        "raw_phase_loss": raw_phase_loss,
        "E_phase": e_phase,
        "E_micromagnetic": e_micro,
        "E_total": e_total,
        "rmse_3d": rmse_3d,
        "projected_error": proj_err,
        "unit_norm_error": unit_norm_error,
        "off_support_leak": off_support_leak,
    }
    if max_g is not None:
        row["max_g"] = float(max_g)
    return row


def calibrate_phase_energy_scale(
    target: NeuralMagPhaseTarget,
    config: NeuralMagPhaseRecoveryConfig,
    *,
    lambda_ref: float,
) -> float:
    """Scale phase loss so its field is a fraction of micromagnetic field."""
    state, phase_terms = _build_neuralmag_state(
        target,
        config,
        lambda_phase=lambda_ref,
        phase_energy_scale=1.0,
        m0_cell_xyz=None,
    )
    m_ref = state.m.tensor
    h_phase_ref = phase_terms["h_phase"](m_ref, state.phase_scale)
    h_micro_ref = state.h_micromagnetic.tensor
    phase_max = float(jnp.max(jnp.abs(h_phase_ref)))
    micro_max = float(jnp.max(jnp.abs(h_micro_ref)))
    target_phase_max = config.phase_field_fraction * micro_max
    return float(target_phase_max / max(phase_max, np.finfo(np.float32).tiny))


def run_neuralmag_phase_recovery(
    rho_xyz: np.ndarray | NeuralMagPhaseTarget,
    m_target_xyz: np.ndarray | None = None,
    *,
    cellsize_nm: float | None = None,
    config: NeuralMagPhaseRecoveryConfig | None = None,
    crop_shape: tuple[int, int, int] | None = None,
    m0_cell_xyz: np.ndarray | None = None,
) -> NeuralMagPhaseRecoveryResult:
    """Run scheduled NeuralMag recovery constrained by an MBIR phase image."""
    config = config or NeuralMagPhaseRecoveryConfig()
    target = _resolve_phase_recovery_target(
        rho_xyz,
        m_target_xyz,
        cellsize_nm=cellsize_nm,
        config=config,
        crop_shape=crop_shape,
    )

    phase_schedule = tuple(float(v) for v in config.phase_weight_schedule)
    if not phase_schedule:
        raise ValueError("phase_weight_schedule must contain at least one value.")

    phase_energy_scale = _resolve_phase_energy_scale(target, config, phase_schedule)
    projection = make_support_projection(
        target.rho_xyz,
        threshold=config.support_threshold,
    )

    history: list[dict[str, float | int | str]] = []
    total_iter = 0
    state, phase_terms = _build_neuralmag_state(
        target,
        config,
        lambda_phase=phase_schedule[0],
        phase_energy_scale=phase_energy_scale,
        m0_cell_xyz=m0_cell_xyz,
    )
    max_g = np.nan

    for stage, lambda_phase in enumerate(phase_schedule):
        _set_phase_scale(
            state,
            phase_terms,
            lambda_phase=lambda_phase,
            phase_energy_scale=phase_energy_scale,
        )
        state.m.tensor = projection(state.m.tensor)
        initial_max_g = _current_max_gradient(state)
        history.append(
            _metrics_for_state(
                state,
                phase_terms,
                target,
                config,
                stage=stage,
                global_step=total_iter,
                event="start",
                max_g=initial_max_g,
            )
        )
        n_iter, max_g = _run_solver_stage(state, config, projection)
        total_iter += n_iter
        history.append(
            _metrics_for_state(
                state,
                phase_terms,
                target,
                config,
                stage=stage,
                global_step=total_iter,
                event="end",
                max_g=max_g,
            )
        )
    m_recovered = np.asarray(state.m.tensor, dtype=np.float32)
    phase_recovered = np.asarray(_predict_phase_from_m(state.m.tensor, target, config), dtype=np.float32)
    return NeuralMagPhaseRecoveryResult(
        state=state,
        target=target,
        config=config,
        phase_terms=phase_terms,
        history=history,
        phase_energy_scale=phase_energy_scale,
        m_recovered_xyz=m_recovered,
        phase_recovered=phase_recovered,
        n_iter=total_iter,
        max_g=max_g,
    )


def _component_error_statistics(
    recovered_xyz: np.ndarray,
    target: NeuralMagPhaseTarget,
    *,
    support_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not target.has_reference_magnetization:
        nan3 = np.full(3, np.nan, dtype=np.float32)
        return nan3, nan3

    support = np.asarray(target.rho_xyz > support_threshold)
    recovered_support = np.asarray(recovered_xyz, dtype=np.float32)[support]
    target_support = np.asarray(target.m_target_xyz, dtype=np.float32)[support]
    if recovered_support.size == 0:
        nan3 = np.full(3, np.nan, dtype=np.float32)
        return nan3, nan3

    diff = recovered_support - target_support
    component_rmse_xyz = np.sqrt(np.mean(diff * diff, axis=0)).astype(np.float32)
    mean_component_error_xyz = (
        recovered_support.mean(axis=0) - target_support.mean(axis=0)
    ).astype(np.float32)
    return component_rmse_xyz, mean_component_error_xyz


def _phase_fit_statistics(
    m_cell_xyz: np.ndarray,
    target: NeuralMagPhaseTarget,
    config: NeuralMagPhaseRecoveryConfig,
) -> tuple[float, float]:
    pred_phase = np.asarray(
        _predict_phase_from_m(jnp.asarray(m_cell_xyz, dtype=jnp.float32), target, config),
        dtype=np.float32,
    )
    mask = np.asarray(target.phase_mask, dtype=np.float32)
    residual = (pred_phase - np.asarray(target.phase_target, dtype=np.float32)) * mask
    denom = max(float(mask.sum()), 1.0)
    phase_rms = float(np.sqrt(np.sum(residual * residual) / denom))
    raw_phase_loss = float(0.5 * np.sum(residual * residual) / denom)
    return phase_rms, raw_phase_loss


def select_anisotropy_orientation_from_phase(
    rho_xyz: np.ndarray | NeuralMagPhaseTarget,
    phase_target_yx: np.ndarray | None = None,
    *,
    cellsize_nm: float | None = None,
    candidates: Mapping[str, tuple[np.ndarray, np.ndarray]],
    config: NeuralMagPhaseRecoveryConfig | None = None,
    crop_shape: tuple[int, int, int] | None = None,
    m_reference_xyz: np.ndarray | None = None,
    m0_cell_xyz: np.ndarray | None = None,
    candidate_m0_cell_xyz: Mapping[str, np.ndarray] | None = None,
    warmup_steps: int = 0,
    selection_metric: Literal["final_phase_rms", "initial_phase_rms"] = "final_phase_rms",
) -> NeuralMagAnisotropySelectionResult:
    """Fit and rank discrete anisotropy candidates against one phase target."""
    from dataclasses import replace

    config = config or NeuralMagPhaseRecoveryConfig()
    if not candidates:
        raise ValueError("candidates must contain at least one anisotropy orientation.")

    if isinstance(rho_xyz, NeuralMagPhaseTarget):
        target = rho_xyz
    else:
        if phase_target_yx is None or cellsize_nm is None:
            raise ValueError(
                "phase_target_yx and cellsize_nm are required when passing raw rho_xyz arrays."
            )
        target = prepare_neuralmag_phase_target_from_phase_image(
            rho_xyz,
            phase_target_yx,
            cellsize_nm=cellsize_nm,
            config=config,
            crop_shape=crop_shape,
            m_reference_xyz=m_reference_xyz,
        )

    if m0_cell_xyz is None:
        shared_initial_m = make_initial_m_cell(
            target.rho_xyz,
            target.m_target_xyz,
            mode=config.init_mode,
            rng_seed=config.rng_seed,
            support_threshold=config.support_threshold,
        )
    else:
        shared_initial_m = normalize_on_support(
            m0_cell_xyz,
            target.rho_xyz,
            threshold=config.support_threshold,
        )

    fits: list[NeuralMagAnisotropyCandidateFit] = []
    phase_steps = max(config.minimizer_max_iter - max(warmup_steps, 0), 1)

    for name, (axis1, axis2) in candidates.items():
        candidate_config = replace(
            config,
            anisotropy_axis1=np.asarray(axis1, dtype=np.float32),
            anisotropy_axis2=np.asarray(axis2, dtype=np.float32),
        )
        if candidate_m0_cell_xyz is not None and name in candidate_m0_cell_xyz:
            candidate_initial_m = normalize_on_support(
                np.asarray(candidate_m0_cell_xyz[name], dtype=np.float32),
                target.rho_xyz,
                threshold=config.support_threshold,
            )
        else:
            candidate_initial_m = shared_initial_m

        initial_phase_rms, initial_raw_phase_loss = _phase_fit_statistics(
            candidate_initial_m,
            target,
            candidate_config,
        )
        initial_component_rmse_xyz, _ = _component_error_statistics(
            candidate_initial_m,
            target,
            support_threshold=candidate_config.support_threshold,
        )

        if warmup_steps > 0:
            warmup_config = replace(
                candidate_config,
                phase_weight_schedule=(0.0,),
                phase_energy_scale=1.0,
                minimizer_max_iter=warmup_steps,
            )
            warmup_result = run_neuralmag_phase_recovery(
                target,
                config=warmup_config,
                m0_cell_xyz=candidate_initial_m,
            )
            phase_config = replace(
                candidate_config,
                minimizer_max_iter=phase_steps,
            )
            result = run_neuralmag_phase_recovery(
                target,
                config=phase_config,
                m0_cell_xyz=warmup_result.m_recovered_xyz,
            )
        else:
            result = run_neuralmag_phase_recovery(
                target,
                config=candidate_config,
                m0_cell_xyz=candidate_initial_m,
            )

        last = result.history[-1]
        component_rmse_xyz, mean_component_error_xyz = _component_error_statistics(
            result.m_recovered_xyz,
            target,
            support_threshold=candidate_config.support_threshold,
        )
        fits.append(
            NeuralMagAnisotropyCandidateFit(
                name=name,
                axis1=np.asarray(axis1, dtype=np.float32),
                axis2=np.asarray(axis2, dtype=np.float32),
                config=candidate_config,
                result=result,
                initial_phase_rms=initial_phase_rms,
                initial_raw_phase_loss=initial_raw_phase_loss,
                initial_component_rmse_xyz=initial_component_rmse_xyz,
                final_phase_rms=float(last["phase_rms"]),
                final_raw_phase_loss=float(last["raw_phase_loss"]),
                component_rmse_xyz=component_rmse_xyz,
                mean_component_error_xyz=mean_component_error_xyz,
            )
        )

    if selection_metric == "final_phase_rms":
        sort_key = lambda fit: (fit.final_phase_rms, fit.final_raw_phase_loss)
    elif selection_metric == "initial_phase_rms":
        sort_key = lambda fit: (fit.initial_phase_rms, fit.initial_raw_phase_loss)
    else:
        raise ValueError(f"Unsupported selection_metric {selection_metric!r}.")

    fits_sorted = tuple(sorted(fits, key=sort_key))
    return NeuralMagAnisotropySelectionResult(
        best_name=fits_sorted[0].name,
        best_fit=fits_sorted[0],
        fits=fits_sorted,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import unxt as u

from .energy_backend import NeuralMagEnergyBackend
from .forward import forward_phase_from_density_and_magnetization
from .synthetic import vortex_magnetization
from .units import RampCoeffs, make_quantity


def _as_jax_array(value, *, dtype=jnp.float32):
    return jnp.asarray(value, dtype=dtype)


def _support_mask(rho, *, threshold: float = 0.5):
    return _as_jax_array(rho) > threshold


def project_unit_norm(m, rho, *, threshold: float = 0.5, eps: float = 1e-12):
    m = _as_jax_array(m)
    support = _support_mask(rho, threshold=threshold)
    norm2 = jnp.sum(m * m, axis=-1, keepdims=True)
    safe_norm = jnp.sqrt(norm2 + eps)
    fallback = jnp.zeros_like(m)
    fallback = fallback.at[..., 0].set(1.0)
    normalized = m / safe_norm
    normalized = jnp.where(norm2 > eps, normalized, fallback)
    return jnp.where(support[..., None], normalized, 0.0)


@dataclass(frozen=True)
class FieldState:
    rho: jax.Array
    m: jax.Array


@dataclass(frozen=True)
class InversionResult:
    m_recon: np.ndarray
    phi_pred: np.ndarray
    loss_history: np.ndarray
    ramp_coeffs: RampCoeffs


@dataclass(frozen=True)
class ScaledRhoExperimentResult:
    rho_scaled: np.ndarray
    m_scaled_truth: np.ndarray
    phi_scaled: np.ndarray
    hist_counts: np.ndarray
    hist_edges: np.ndarray
    mean_abs_m: float


class PhysicsBackend:
    supports_jax = False

    def prepare(self, rho, m) -> FieldState:
        return FieldState(rho=_as_jax_array(rho), m=_as_jax_array(m))

    def energies(self, field: FieldState) -> dict[str, Any]:
        raise NotImplementedError


class IdentityBackend(PhysicsBackend):
    supports_jax = True

    def energies(self, field: FieldState) -> dict[str, Any]:
        del field
        return {}


def _smoothness_energy(field: FieldState) -> jax.Array:
    m = field.m
    support = field.rho > 0.5
    energy = jnp.asarray(0.0, dtype=m.dtype)
    for axis in range(3):
        head = [slice(None)] * 3
        tail = [slice(None)] * 3
        head[axis] = slice(1, None)
        tail[axis] = slice(None, -1)
        pair_mask = support[tuple(head)] & support[tuple(tail)]
        diff = m[tuple(head)] - m[tuple(tail)]
        energy = energy + jnp.sum(diff * diff * pair_mask[..., None])
    return energy


class SmoothnessBackend(PhysicsBackend):
    supports_jax = True

    def energies(self, field: FieldState) -> dict[str, Any]:
        return {"smoothness": _smoothness_energy(field)}


class CombinedBackend(PhysicsBackend):
    def __init__(self, *backends: Any) -> None:
        self.backends = tuple(backends)
        self.supports_jax = all(getattr(backend, "supports_jax", False) for backend in self.backends)

    def energies(self, field: FieldState) -> dict[str, Any]:
        terms: dict[str, Any] = {}
        for backend in self.backends:
            for name, value in backend.energies(field).items():
                if name in terms:
                    raise ValueError("duplicate energy term names")
                terms[name] = value
        return terms


class WeightedBackend(PhysicsBackend):
    def __init__(self, backend: Any, *, weight: float) -> None:
        self.backend = backend
        self.weight = float(weight)
        self.supports_jax = getattr(backend, "supports_jax", False)

    def energies(self, field: FieldState) -> dict[str, Any]:
        return {
            name: self.weight * value
            for name, value in self.backend.energies(field).items()
        }


def _backend_total_energy(m, rho, backend: Any) -> jax.Array:
    field = backend.prepare(rho, m)
    terms = backend.energies(field)
    total = jnp.asarray(0.0, dtype=_as_jax_array(m).dtype)
    for value in terms.values():
        total = total + jnp.asarray(value, dtype=total.dtype)
    return total


def _equilibrium_residual_jax(m, rho, backend: Any) -> jax.Array:
    if not getattr(backend, "supports_jax", False):
        return jnp.asarray(0.0, dtype=_as_jax_array(m).dtype)

    m = _as_jax_array(m)
    rho = _as_jax_array(rho)
    grad = jax.grad(lambda mm: _backend_total_energy(mm, rho, backend))(m)
    torque = jnp.cross(m, grad)
    support = (rho > 0.5)[..., None]
    torque_sq = jnp.sum((torque * support) ** 2, axis=-1)
    denom = jnp.maximum(jnp.sum(support[..., 0]), 1.0)
    return jnp.sqrt(jnp.sum(torque_sq) / denom)


class EquilibriumTorqueBackend(PhysicsBackend):
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.supports_jax = getattr(backend, "supports_jax", False)

    def energies(self, field: FieldState) -> dict[str, Any]:
        return {
            "equilibrium_torque": _equilibrium_residual_jax(field.m, field.rho, self.backend),
        }


class NeuralMagCritic(PhysicsBackend):
    supports_jax = False

    def __init__(self, backend: NeuralMagEnergyBackend) -> None:
        self.backend = backend

    @classmethod
    def from_state(cls, state, terms: tuple[str, ...] | list[str]):
        return cls(NeuralMagEnergyBackend.from_state(state, terms))

    def energies(self, field: FieldState) -> dict[str, Any]:
        return {
            name: jnp.asarray(value, dtype=field.m.dtype)
            for name, value in self.backend.energy_terms(
                np.asarray(field.rho),
                np.asarray(field.m),
            ).items()
        }


def phase_residual(phi_pred, phi_true) -> float:
    phi_pred = np.asarray(phi_pred, dtype=np.float32)
    phi_true = np.asarray(phi_true, dtype=np.float32)
    denom = max(float(np.linalg.norm(phi_true)), np.finfo(np.float32).eps)
    return float(np.linalg.norm(phi_pred - phi_true) / denom)


def projected_m_error(m_recon, m_true, rho=None) -> float:
    m_recon = np.asarray(m_recon, dtype=np.float32)
    m_true = np.asarray(m_true, dtype=np.float32)
    if rho is None:
        weight = 1.0
    else:
        weight = (np.asarray(rho, dtype=np.float32) > 0.5)[..., None]
    proj_recon = np.sum(weight * m_recon, axis=0)
    proj_true = np.sum(weight * m_true, axis=0)
    denom = max(float(np.linalg.norm(proj_true)), np.finfo(np.float32).eps)
    return float(np.linalg.norm(proj_recon - proj_true) / denom)


def mz_rmse(m_recon, m_true, rho=None) -> float:
    m_recon = np.asarray(m_recon, dtype=np.float32)
    m_true = np.asarray(m_true, dtype=np.float32)
    diff = m_recon[..., 2] - m_true[..., 2]
    if rho is not None:
        mask = np.asarray(rho, dtype=np.float32) > 0.5
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def depth_correlation(m_recon, m_true, yx: tuple[int, int]) -> float:
    y, x = yx
    recon_profile = np.asarray(m_recon, dtype=np.float32)[:, y, x, :].reshape(-1)
    true_profile = np.asarray(m_true, dtype=np.float32)[:, y, x, :].reshape(-1)
    if np.allclose(recon_profile, true_profile):
        return 1.0
    recon_std = float(np.std(recon_profile))
    true_std = float(np.std(true_profile))
    if recon_std <= np.finfo(np.float32).eps or true_std <= np.finfo(np.float32).eps:
        return 0.0
    return float(np.corrcoef(recon_profile, true_profile)[0, 1])


def vortex_core_z_error(m_recon, m_true, yx: tuple[int, int]) -> float:
    y, x = yx
    recon_profile = np.abs(np.asarray(m_recon, dtype=np.float32)[:, y, x, 2])
    true_profile = np.abs(np.asarray(m_true, dtype=np.float32)[:, y, x, 2])
    return float(abs(int(np.argmax(recon_profile)) - int(np.argmax(true_profile))))


def iterations_to_threshold(history, threshold: float) -> int:
    history = np.asarray(history, dtype=np.float32)
    hits = np.flatnonzero(history <= threshold)
    if hits.size == 0:
        return int(history.size)
    return int(hits[0] + 1)


def equilibrium_residual(m, backend: Any, *, rho) -> float:
    return float(np.asarray(_equilibrium_residual_jax(m, rho, backend)))


def support_center_yx(rho) -> tuple[int, int]:
    support = np.argwhere(np.max(np.asarray(rho, dtype=np.float32), axis=0) > 0.5)
    if support.size == 0:
        raise ValueError("Support mask is empty.")
    center = np.round(np.mean(support, axis=0)).astype(int)
    return int(center[0]), int(center[1])


def analytic_vortex_init(rho) -> jax.Array:
    rho = np.asarray(rho, dtype=np.float32)
    support = (rho > 0.5).astype(np.float32)
    m0 = vortex_magnetization(rho.shape, support_zyx=jnp.asarray(support), dtype=jnp.float32)
    return project_unit_norm(m0, rho)


def _ramp_design(shape: tuple[int, int], pixel_size_nm: float, dtype):
    height, width = shape
    yy, xx = jnp.meshgrid(
        jnp.arange(height, dtype=dtype),
        jnp.arange(width, dtype=dtype),
        indexing="ij",
    )
    px = jnp.asarray(pixel_size_nm, dtype=dtype)
    return jnp.stack([
        jnp.ones_like(yy),
        yy * px,
        xx * px,
    ], axis=-1)


def _fit_ramp(phi_target, phi_base, pixel_size_nm: float):
    design = _ramp_design(phi_target.shape, pixel_size_nm, phi_base.dtype)
    flat_design = design.reshape(-1, 3)
    rhs = (phi_target - phi_base).reshape(-1)
    eye = jnp.eye(3, dtype=phi_base.dtype)
    coeffs = jnp.linalg.solve(flat_design.T @ flat_design + 1e-12 * eye, flat_design.T @ rhs)
    ramp_img = jnp.tensordot(design, coeffs, axes=([-1], [0]))
    ramp_coeffs = RampCoeffs(
        offset=make_quantity(coeffs[0], "rad"),
        slope_y=make_quantity(coeffs[1], "rad/nm"),
        slope_x=make_quantity(coeffs[2], "rad/nm"),
    )
    return ramp_coeffs, phi_base + ramp_img


def _initial_magnetization(rho, init):
    rho = np.asarray(rho, dtype=np.float32)
    support = rho > 0.5
    if isinstance(init, str):
        init_name = init.lower()
        if init_name == "zero":
            m0 = np.zeros(rho.shape + (3,), dtype=np.float32)
        elif init_name == "uniform_x":
            m0 = np.zeros(rho.shape + (3,), dtype=np.float32)
            m0[..., 0] = 1.0
        elif init_name == "analytic_vortex":
            m0 = np.asarray(analytic_vortex_init(rho), dtype=np.float32)
        else:
            raise ValueError(f"Unknown init mode {init!r}")
    else:
        m0 = np.asarray(init, dtype=np.float32)
        if m0.shape != rho.shape + (3,):
            raise ValueError(
                f"Warm-start magnetization must have shape {rho.shape + (3,)}, got {m0.shape}."
            )

    if not np.any(np.linalg.norm(m0[support], axis=-1) > 0.0):
        m0[..., 0] = np.where(support, 1.0, 0.0)
    return project_unit_norm(m0, rho)


def _physics_loss(m, rho, backend: Any, *, objective: str):
    if objective == "energy":
        return _backend_total_energy(m, rho, backend)
    if objective == "torque":
        return _equilibrium_residual_jax(m, rho, backend)
    raise ValueError(f"Unknown physics_objective {objective!r}")


def invert_magnetization(
    phi_target,
    rho,
    backend: Any,
    *,
    pixel_size,
    lambda_phys: float = 0.0,
    max_iter: int = 10,
    lr: float = 5e-2,
    init: str | np.ndarray = "zero",
    fit_ramp: bool = False,
    optimizer: str = "bb",
    physics_objective: str = "energy",
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> InversionResult:
    if physics_objective not in {"energy", "torque"}:
        raise ValueError("physics_objective must be 'energy' or 'torque'")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if optimizer not in {"bb", "lbfgs"}:
        raise ValueError(f"Unknown optimizer {optimizer!r}")

    phi_target = _as_jax_array(phi_target)
    rho = _as_jax_array(rho)
    if isinstance(pixel_size, u.Quantity):
        pixel_size_q = pixel_size
    else:
        pixel_size_q = u.Quantity(pixel_size, "nm")
    pixel_size_nm = float(np.asarray(u.uconvert("nm", pixel_size_q).value))

    m = _initial_magnetization(rho, init)
    prev_m = None
    prev_grad = None
    loss_history: list[float] = []
    best_loss = np.inf
    no_improve = 0

    supports_jax = bool(getattr(backend, "supports_jax", False))

    def _predict_with_optional_ramp(m_current):
        phi_base = forward_phase_from_density_and_magnetization(
            rho=rho,
            magnetization_3d=m_current,
            pixel_size=pixel_size_q,
            axis="z",
        )
        if fit_ramp:
            return _fit_ramp(phi_target, phi_base, pixel_size_nm)
        ramp_coeffs = RampCoeffs.zeros(dtype=phi_base.dtype)
        return ramp_coeffs, phi_base

    def _loss_for_grad(m_current):
        _ramp_coeffs, phi_pred = _predict_with_optional_ramp(m_current)
        residual = phi_pred - phi_target
        data_loss = jnp.mean(residual * residual)
        if lambda_phys == 0.0 or not supports_jax:
            return data_loss
        return data_loss + float(lambda_phys) * _physics_loss(
            m_current,
            rho,
            backend,
            objective=physics_objective,
        )

    loss_and_grad = jax.value_and_grad(_loss_for_grad)

    def _total_loss_for_history(m_current):
        ramp_coeffs, phi_pred = _predict_with_optional_ramp(m_current)
        residual = phi_pred - phi_target
        data_loss = float(np.asarray(jnp.mean(residual * residual)))
        physics_loss = 0.0
        if lambda_phys != 0.0:
            if supports_jax:
                physics_loss = float(np.asarray(_physics_loss(m_current, rho, backend, objective=physics_objective)))
            else:
                terms = backend.energies(backend.prepare(rho, m_current))
                physics_loss = sum(float(np.asarray(value)) for value in terms.values())
        return data_loss + float(lambda_phys) * physics_loss, ramp_coeffs, phi_pred

    for _iteration in range(max_iter):
        current_loss_jax, grad = loss_and_grad(m)
        current_loss = float(np.asarray(current_loss_jax))

        grad_norm = float(np.linalg.norm(np.asarray(grad)))
        if optimizer == "bb" and prev_m is not None and prev_grad is not None:
            s = np.ravel(np.asarray(m - prev_m))
            y = np.ravel(np.asarray(grad - prev_grad))
            denom = float(np.dot(s, y))
            if np.isfinite(denom) and abs(denom) > 1e-12:
                step_size = float(np.clip(abs(np.dot(s, s) / denom), lr * 0.1, lr * 5.0))
            else:
                step_size = float(lr)
        else:
            step_size = float(lr)

        accepted_m = m
        accepted_loss = current_loss
        if grad_norm > 0.0:
            for factor in (1.0, 0.5, 0.25, 0.125):
                candidate = project_unit_norm(m - step_size * factor * grad, rho)
                candidate_loss, _, _ = _total_loss_for_history(candidate)
                if candidate_loss <= current_loss or factor == 0.125:
                    accepted_m = candidate
                    accepted_loss = candidate_loss
                    break

        loss_history.append(float(accepted_loss))
        prev_m = m
        prev_grad = grad
        m = accepted_m

        if accepted_loss + early_stopping_min_delta < best_loss:
            best_loss = accepted_loss
            no_improve = 0
        else:
            no_improve += 1

        if early_stopping_patience is not None and no_improve >= early_stopping_patience:
            break

    _final_loss, ramp_coeffs, phi_pred = _total_loss_for_history(m)
    return InversionResult(
        m_recon=np.asarray(m, dtype=np.float32),
        phi_pred=np.asarray(phi_pred, dtype=np.float32),
        loss_history=np.asarray(loss_history, dtype=np.float32),
        ramp_coeffs=ramp_coeffs,
    )


def run_with_scaled_rho(
    pipeline,
    rho,
    m_true,
    pixel_size,
    *,
    scale: float,
    histogram_bins: int = 32,
) -> ScaledRhoExperimentResult:
    rho = np.asarray(rho, dtype=np.float32)
    m_true = np.asarray(m_true, dtype=np.float32)
    rho_scaled = np.clip(scale * rho, 0.0, 1.0).astype(np.float32)
    m_scaled_truth = np.asarray(project_unit_norm(m_true, rho_scaled), dtype=np.float32)
    phi_scaled = np.asarray(
        forward_phase_from_density_and_magnetization(
            rho=rho_scaled,
            magnetization_3d=m_scaled_truth,
            pixel_size=pixel_size,
            axis="z",
        ),
        dtype=np.float32,
    )

    recon = pipeline(phi_scaled, rho_scaled)
    norms = np.linalg.norm(np.asarray(recon.m_recon, dtype=np.float32), axis=-1)
    hist_counts, hist_edges = np.histogram(norms, bins=histogram_bins, range=(0.0, 1.0))
    mean_abs_m = float(np.mean(norms))
    return ScaledRhoExperimentResult(
        rho_scaled=rho_scaled,
        m_scaled_truth=m_scaled_truth,
        phi_scaled=phi_scaled,
        hist_counts=hist_counts,
        hist_edges=hist_edges,
        mean_abs_m=mean_abs_m,
    )


def plot_loss_history(loss_history, *, label: str | None = None):
    loss_history = np.asarray(loss_history, dtype=np.float32)
    iterations = np.arange(1, loss_history.size + 1, dtype=np.int32)
    fig, ax = plt.subplots()
    ax.plot(iterations, loss_history, label=label)
    if label is not None:
        ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    return fig, ax, {
        "iterations": iterations,
        "loss_history": loss_history,
    }


def plot_depth_profile(m_recon, m_true, yx: tuple[int, int], *, component: int = 2):
    y, x = yx
    recon_profile = np.asarray(m_recon, dtype=np.float32)[:, y, x, component]
    true_profile = np.asarray(m_true, dtype=np.float32)[:, y, x, component]
    depth = np.arange(recon_profile.shape[0], dtype=np.int32)
    fig, ax = plt.subplots()
    ax.plot(depth, true_profile, label="true")
    ax.plot(depth, recon_profile, label="recon")
    ax.legend()
    ax.set_xlabel("z")
    ax.set_ylabel(f"m[{component}]")
    return fig, ax, {
        "yx": yx,
        "component": component,
        "true_profile": true_profile,
        "recon_profile": recon_profile,
    }


def plot_m_slices(m_recon, m_true, *, z_index: int, component: int = 2):
    recon_slice = np.asarray(m_recon, dtype=np.float32)[z_index, ..., component]
    true_slice = np.asarray(m_true, dtype=np.float32)[z_index, ..., component]
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(true_slice)
    axes[0].set_title("true")
    axes[1].imshow(recon_slice)
    axes[1].set_title("recon")
    return fig, axes, {
        "z_index": z_index,
        "component": component,
        "true_slice": true_slice,
        "recon_slice": recon_slice,
    }


def plot_loss_landscape_2d(x_values, y_values, loss_grid, *, x_label: str = "x", y_label: str = "y"):
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    loss_grid = np.asarray(loss_grid, dtype=np.float32)
    min_index = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    minimum = (
        float(x_values[min_index[1]]),
        float(y_values[min_index[0]]),
        float(loss_grid[min_index]),
    )
    fig, ax = plt.subplots()
    image = ax.imshow(loss_grid, origin="lower", aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return fig, ax, {
        "minimum": minimum,
    }


__all__ = [
    "CombinedBackend",
    "EquilibriumTorqueBackend",
    "FieldState",
    "IdentityBackend",
    "InversionResult",
    "NeuralMagCritic",
    "PhysicsBackend",
    "ScaledRhoExperimentResult",
    "SmoothnessBackend",
    "WeightedBackend",
    "analytic_vortex_init",
    "depth_correlation",
    "equilibrium_residual",
    "invert_magnetization",
    "iterations_to_threshold",
    "mz_rmse",
    "phase_residual",
    "plot_depth_profile",
    "plot_loss_history",
    "plot_loss_landscape_2d",
    "plot_m_slices",
    "project_unit_norm",
    "projected_m_error",
    "run_with_scaled_rho",
    "support_center_yx",
    "vortex_core_z_error",
]
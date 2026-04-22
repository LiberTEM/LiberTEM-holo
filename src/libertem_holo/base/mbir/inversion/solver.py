from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import unxt as u

from ..forward import forward_phase_from_density_and_magnetization
from ..synthetic import vortex_magnetization
from ..units import RampCoeffs, _as_ramp_coeffs, _ramp_coeffs_from_array, _ramp_coeffs_to_array
from .backends import PhysicsBackend


@dataclass(frozen=True)
class InversionResult:
    m_recon: jnp.ndarray
    loss_history: jnp.ndarray
    phi_pred: jnp.ndarray
    wall_time_s: float
    converged: bool
    ramp_coeffs: RampCoeffs = field(default_factory=RampCoeffs.zeros)


def _normalize_pixel_size(pixel_size) -> u.Quantity:
    if isinstance(pixel_size, u.Quantity):
        return cast(u.Quantity, pixel_size)
    return u.Quantity(float(pixel_size), "nm")


def _initial_m(rho: jnp.ndarray, init) -> jnp.ndarray:
    shape = rho.shape + (3,)
    if isinstance(init, str):
        if init == "zero":
            return jnp.zeros(shape, dtype=rho.dtype)
        if init == "uniform_x":
            return jnp.broadcast_to(
                jnp.asarray([1.0, 0.0, 0.0], dtype=rho.dtype),
                shape,
            )
        if init == "analytic_vortex":
            core_radius = max(1.5, rho.shape[0] / 32.0)
            return vortex_magnetization(
                (int(rho.shape[0]), int(rho.shape[1]), int(rho.shape[2])),
                support_zyx=rho,
                core_radius=core_radius,
                dtype=rho.dtype,
            )
        raise ValueError(f"Unknown init strategy {init!r}.")

    init_arr = jnp.asarray(init, dtype=rho.dtype)
    if init_arr.shape != shape:
        raise ValueError(
            f"Custom init must have shape {shape}, got {init_arr.shape}."
        )
    return init_arr


def _initial_ramp(dtype, init_ramp_coeffs=None) -> jnp.ndarray:
    if init_ramp_coeffs is None:
        return jnp.zeros((3,), dtype=dtype)
    return jnp.asarray(
        _ramp_coeffs_to_array(
            _as_ramp_coeffs(init_ramp_coeffs, dtype=dtype),
        ),
        dtype=dtype,
    )


def project_unit_norm(
    m,
    rho,
    *,
    threshold: float = 0.5,
    eps: float = 1e-12,
    fallback_direction=(1.0, 0.0, 0.0),
) -> jnp.ndarray:
    m_arr = jnp.asarray(m)
    rho_arr = jnp.asarray(rho)
    if m_arr.shape[:-1] != rho_arr.shape or m_arr.shape[-1] != 3:
        raise ValueError(
            "m must have shape rho.shape + (3,), "
            f"got rho {rho_arr.shape} and m {m_arr.shape}."
        )

    mask = rho_arr > threshold
    # Use a regularised norm to avoid the 0/0 in jnp.linalg.norm's VJP at m=0.
    # jnp.linalg.norm has VJP: g * m / norm.  When m=0 this gives g * 0/0 = NaN
    # even when g=0, because NaN * 0 = NaN in IEEE-754.  Computing the norm as
    # sqrt(sum(m²) + ε²) is always ≥ ε, so its VJP is always finite.
    norms_sq = jnp.sum(m_arr ** 2, axis=-1, keepdims=True)
    safe_norms = jnp.sqrt(norms_sq + eps)
    norms = safe_norms  # alias: the threshold check below uses the same value
    normalized = m_arr / safe_norms
    fallback = jnp.broadcast_to(
        jnp.asarray(fallback_direction, dtype=m_arr.dtype),
        m_arr.shape,
    )
    normalized = jnp.where(norms_sq > eps, normalized, fallback)
    return jnp.where(mask[..., None], normalized, 0.0)


def invert_magnetization(
    phi_meas,
    rho,
    backend: PhysicsBackend,
    *,
    pixel_size,
    lambda_phys: float = 0.0,
    max_iter: int = 500,
    lr: float = 1e-2,
    init: str | Any = "zero",
    axis: str = "z",
    projection_threshold: float = 0.5,
    fit_ramp: bool = False,
    init_ramp_coeffs=None,
    optimizer: str = "adam",
    physics_objective: str = "energy",
    lbfgs_memory_size: int = 10,
    bb_min_step: float = 1e-6,
    bb_max_step: float = 1.0,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> InversionResult:
    rho_arr = jnp.asarray(rho)
    phi_target = jnp.asarray(phi_meas)
    pixel_size_q = cast(u.Quantity, _normalize_pixel_size(pixel_size))

    m0 = project_unit_norm(
        _initial_m(rho_arr, init),
        rho_arr,
        threshold=projection_threshold,
    )
    ramp0 = _initial_ramp(phi_target.dtype, init_ramp_coeffs)
    use_physics = float(lambda_phys) != 0.0
    optimizer_name = optimizer.lower()
    if optimizer_name not in {"adam", "lbfgs", "bb"}:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}. Expected 'adam', 'lbfgs', or 'bb'."
        )
    physics_objective_name = physics_objective.lower()
    if physics_objective_name not in {"energy", "torque"}:
        raise ValueError(
            "physics_objective must be 'energy' or 'torque'."
        )
    if bb_min_step <= 0.0:
        raise ValueError("bb_min_step must be > 0.")
    if bb_max_step < bb_min_step:
        raise ValueError("bb_max_step must be >= bb_min_step.")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1 when provided.")

    if fit_ramp:
        height, width = phi_target.shape
        pixel_size_nm_q = cast(u.Quantity, u.uconvert("nm", pixel_size_q))
        pixel_size_nm = jnp.asarray(pixel_size_nm_q.value, dtype=phi_target.dtype)
        y, x = jnp.meshgrid(
            jnp.arange(height, dtype=phi_target.dtype),
            jnp.arange(width, dtype=phi_target.dtype),
            indexing="ij",
        )
        ramp_basis = jnp.stack(
            [
                jnp.ones_like(y),
                y * pixel_size_nm,
                x * pixel_size_nm,
            ],
            axis=-1,
        )
        ramp_basis_flat = ramp_basis.reshape((-1, 3))
        gram = ramp_basis_flat.T @ ramp_basis_flat
        gram = gram + jnp.eye(3, dtype=phi_target.dtype) * jnp.asarray(1e-12, dtype=phi_target.dtype)
        gram_inv = jnp.asarray(np.linalg.inv(np.asarray(gram)), dtype=phi_target.dtype)

        fit_ramp_to_residual = lambda residual: (
            gram_inv @ (ramp_basis_flat.T @ residual.reshape((-1,))),
            jnp.tensordot(
                ramp_basis,
                gram_inv @ (ramp_basis_flat.T @ residual.reshape((-1,))),
                axes=([-1], [0]),
            ),
        )
    else:
        fit_ramp_to_residual = lambda _residual: (ramp0, jnp.zeros_like(phi_target))

    def total_backend_energy(m_projected):
        field = backend.prepare(rho_arr, m_projected)
        energy_terms = backend.energies(field)
        if not energy_terms:
            return jnp.asarray(0.0, dtype=phi_target.dtype)
        return jnp.sum(
            jnp.stack(
                tuple(
                    jnp.asarray(value, dtype=phi_target.dtype)
                    for value in energy_terms.values()
                )
            )
        )

    def physics_loss_value(m_projected):
        if not use_physics:
            return jnp.asarray(0.0, dtype=phi_target.dtype)
        if physics_objective_name == "energy":
            return total_backend_energy(m_projected)
        grad_energy = jax.grad(total_backend_energy)(m_projected)
        torque = jnp.cross(m_projected, grad_energy)
        return jnp.mean(jnp.sum(torque ** 2, axis=-1))

    def loss_components(m_current):
        m_projected = project_unit_norm(
            m_current,
            rho_arr,
            threshold=projection_threshold,
        )
        phi_mag = forward_phase_from_density_and_magnetization(
            rho=rho_arr,
            magnetization_3d=m_projected,
            pixel_size=pixel_size_q,
            axis=axis,
            ramp_coeffs=None,
        )
        if fit_ramp:
            ramp_current, ramp_image = fit_ramp_to_residual(phi_target - phi_mag)
            phi_pred = phi_mag + ramp_image
        else:
            ramp_current = ramp0
            phi_pred = forward_phase_from_density_and_magnetization(
                rho=rho_arr,
                magnetization_3d=m_projected,
                pixel_size=pixel_size_q,
                axis=axis,
                ramp_coeffs=_ramp_coeffs_from_array(ramp_current),
            )
        data_loss = 0.5 * jnp.mean((phi_pred - phi_target) ** 2)
        physics_loss = physics_loss_value(m_projected)
        total_loss = data_loss + lambda_phys * physics_loss
        return total_loss, (m_projected, phi_pred, ramp_current)

    loss_and_grad = jax.value_and_grad(loss_components, has_aux=True)

    def scalar_loss(m_current):
        loss_value, _aux = loss_components(m_current)
        return loss_value

    opt = None
    if optimizer_name == "lbfgs":
        opt = optax.lbfgs(learning_rate=lr, memory_size=lbfgs_memory_size)

        @jax.jit
        def step(m_current, opt_state_current):
            (loss_value, (m_projected, phi_pred, ramp_current)), grads = loss_and_grad(m_current)
            updates, next_opt_state = opt.update(
                grads,
                opt_state_current,
                m_current,
                value=loss_value,
                grad=grads,
                value_fn=scalar_loss,
            )
            next_m = optax.apply_updates(m_current, updates)
            next_m = project_unit_norm(
                next_m,
                rho_arr,
                threshold=projection_threshold,
            )
            return next_m, next_opt_state, loss_value, m_projected, phi_pred, ramp_current
    elif optimizer_name == "adam":
        opt = optax.adam(lr)

        @jax.jit
        def step(m_current, opt_state_current):
            (loss_value, (m_projected, phi_pred, ramp_current)), grads = loss_and_grad(m_current)
            updates, next_opt_state = opt.update(grads, opt_state_current, m_current)
            next_m = optax.apply_updates(m_current, updates)
            next_m = project_unit_norm(
                next_m,
                rho_arr,
                threshold=projection_threshold,
            )
            return next_m, next_opt_state, loss_value, m_projected, phi_pred, ramp_current
    else:
        bb_min_step_arr = jnp.asarray(bb_min_step, dtype=phi_target.dtype)
        bb_max_step_arr = jnp.asarray(bb_max_step, dtype=phi_target.dtype)
        bb_eps = jnp.asarray(1e-12, dtype=phi_target.dtype)

        @jax.jit
        def step(m_current, opt_state_current):
            prev_m, prev_grads, prev_step_size, step_index = opt_state_current
            (loss_value, (m_projected, phi_pred, ramp_current)), grads = loss_and_grad(m_current)

            delta_m = m_current - prev_m
            delta_grads = grads - prev_grads
            dot_mg = jnp.vdot(delta_m, delta_grads).real
            dot_gg = jnp.vdot(delta_grads, delta_grads).real
            dot_mm = jnp.vdot(delta_m, delta_m).real

            bb1 = dot_mm / jnp.where(jnp.abs(dot_mg) > bb_eps, dot_mg, jnp.nan)
            bb2 = dot_mg / jnp.where(dot_gg > bb_eps, dot_gg, jnp.nan)
            alternating_step = jnp.where((step_index % 2) == 1, bb2, bb1)
            valid_step = (
                (step_index > 0)
                & jnp.isfinite(alternating_step)
                & (alternating_step > 0)
            )
            step_size = jnp.where(
                valid_step,
                jnp.clip(alternating_step, bb_min_step_arr, bb_max_step_arr),
                prev_step_size,
            )

            next_m = m_current - step_size * grads
            next_m = project_unit_norm(
                next_m,
                rho_arr,
                threshold=projection_threshold,
            )
            next_opt_state = (m_current, grads, step_size, step_index + 1)
            return next_m, next_opt_state, loss_value, m_projected, phi_pred, ramp_current

    if optimizer_name == "bb":
        opt_state = (
            m0,
            jnp.zeros_like(m0),
            jnp.asarray(lr, dtype=phi_target.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )
    else:
        assert opt is not None
        opt_state = opt.init(m0)

    history = []
    m_current = m0
    ramp_current = ramp0
    best_m = m0
    best_loss = np.inf
    stale_steps = 0
    start = perf_counter()
    for _ in range(max_iter):
        m_current, opt_state, loss_value, m_projected, _, ramp_current = step(m_current, opt_state)
        history.append(loss_value)
        loss_scalar = float(loss_value)
        if np.isfinite(loss_scalar) and loss_scalar < best_loss - float(early_stopping_min_delta):
            best_loss = loss_scalar
            best_m = m_projected
            stale_steps = 0
        elif early_stopping_patience is not None:
            stale_steps += 1
            if stale_steps >= early_stopping_patience:
                break
    wall_time_s = perf_counter() - start

    if np.isfinite(best_loss):
        m_current = best_m

    _, (m_current, phi_pred, ramp_current) = loss_components(m_current)

    loss_history = jnp.stack(history) if history else jnp.zeros((0,), dtype=phi_pred.dtype)
    converged = bool(
        loss_history.size > 0
        and jnp.all(jnp.isfinite(loss_history))
        and loss_history[-1] <= loss_history[0]
    )
    return InversionResult(
        m_recon=m_current,
        loss_history=loss_history,
        phi_pred=phi_pred,
        wall_time_s=wall_time_s,
        converged=converged,
        ramp_coeffs=_ramp_coeffs_from_array(ramp_current),
    )
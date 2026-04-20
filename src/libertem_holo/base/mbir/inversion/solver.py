from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import optax
import unxt as u

from ..forward import forward_phase_from_density_and_magnetization
from .backends import PhysicsBackend


@dataclass(frozen=True)
class InversionResult:
    m_recon: jnp.ndarray
    loss_history: jnp.ndarray
    phi_pred: jnp.ndarray
    wall_time_s: float
    converged: bool


def _normalize_pixel_size(pixel_size):
    if isinstance(pixel_size, u.Quantity):
        return pixel_size
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
        raise ValueError(f"Unknown init strategy {init!r}.")

    init_arr = jnp.asarray(init, dtype=rho.dtype)
    if init_arr.shape != shape:
        raise ValueError(
            f"Custom init must have shape {shape}, got {init_arr.shape}."
        )
    return init_arr


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
    norms = jnp.linalg.norm(m_arr, axis=-1, keepdims=True)
    safe_norms = jnp.where(norms > eps, norms, 1.0)
    normalized = m_arr / safe_norms
    fallback = jnp.broadcast_to(
        jnp.asarray(fallback_direction, dtype=m_arr.dtype),
        m_arr.shape,
    )
    normalized = jnp.where(norms > eps, normalized, fallback)
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
    init="zero",
    axis: str = "z",
    projection_threshold: float = 0.5,
) -> InversionResult:
    rho_arr = jnp.asarray(rho)
    phi_target = jnp.asarray(phi_meas)
    pixel_size_q = _normalize_pixel_size(pixel_size)

    m0 = project_unit_norm(
        _initial_m(rho_arr, init),
        rho_arr,
        threshold=projection_threshold,
    )

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(m0)

    def loss_components(m_current):
        m_projected = project_unit_norm(
            m_current,
            rho_arr,
            threshold=projection_threshold,
        )
        phi_pred = forward_phase_from_density_and_magnetization(
            rho=rho_arr,
            magnetization_3d=m_projected,
            pixel_size=pixel_size_q,
            axis=axis,
        )
        data_loss = 0.5 * jnp.mean((phi_pred - phi_target) ** 2)
        field = backend.prepare(rho_arr, m_projected)
        energy_terms = backend.energies(field)
        if energy_terms:
            physics_loss = jnp.sum(jnp.stack(tuple(energy_terms.values())))
        else:
            physics_loss = jnp.asarray(0.0, dtype=phi_pred.dtype)
        total_loss = data_loss + lambda_phys * physics_loss
        return total_loss, (m_projected, phi_pred)

    loss_and_grad = jax.value_and_grad(loss_components, has_aux=True)

    @jax.jit
    def step(m_current, opt_state_current):
        (loss_value, (m_projected, phi_pred)), grads = loss_and_grad(m_current)
        updates, next_opt_state = optimizer.update(grads, opt_state_current, m_current)
        next_m = optax.apply_updates(m_current, updates)
        next_m = project_unit_norm(
            next_m,
            rho_arr,
            threshold=projection_threshold,
        )
        return next_m, next_opt_state, loss_value, m_projected, phi_pred

    history = []
    m_current = m0
    phi_pred = forward_phase_from_density_and_magnetization(
        rho=rho_arr,
        magnetization_3d=m_current,
        pixel_size=pixel_size_q,
        axis=axis,
    )
    start = perf_counter()
    for _ in range(max_iter):
        m_current, opt_state, loss_value, _, phi_pred = step(m_current, opt_state)
        history.append(loss_value)
    wall_time_s = perf_counter() - start

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
    )
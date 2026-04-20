from __future__ import annotations

import jax
import jax.numpy as jnp


def _as_spatial_vector_field(values, *, name: str) -> jnp.ndarray:
    arr = jnp.asarray(values)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (Z, Y, X, 3), got {arr.shape}.")
    return arr


def _safe_relative_l2(numerator, denominator, *, eps: float = 1e-12) -> jnp.ndarray:
    num = jnp.linalg.norm(jnp.ravel(jnp.asarray(numerator)))
    den = jnp.linalg.norm(jnp.ravel(jnp.asarray(denominator)))
    return num / jnp.maximum(den, eps)


def phase_residual(phi_pred, phi_true, *, eps: float = 1e-12) -> jnp.ndarray:
    phi_pred_arr = jnp.asarray(phi_pred)
    phi_true_arr = jnp.asarray(phi_true)
    if phi_pred_arr.shape != phi_true_arr.shape:
        raise ValueError(
            f"phi_pred and phi_true must match, got {phi_pred_arr.shape} and {phi_true_arr.shape}."
        )
    return _safe_relative_l2(phi_pred_arr - phi_true_arr, phi_true_arr, eps=eps)


def projected_m_error(m_recon, m_true, *, eps: float = 1e-12) -> jnp.ndarray:
    m_recon_arr = _as_spatial_vector_field(m_recon, name="m_recon")
    m_true_arr = _as_spatial_vector_field(m_true, name="m_true")
    projected_recon = jnp.sum(m_recon_arr[..., :2], axis=0)
    projected_true = jnp.sum(m_true_arr[..., :2], axis=0)
    return _safe_relative_l2(projected_recon - projected_true, projected_true, eps=eps)


def mz_rmse(m_recon, m_true) -> jnp.ndarray:
    m_recon_arr = _as_spatial_vector_field(m_recon, name="m_recon")
    m_true_arr = _as_spatial_vector_field(m_true, name="m_true")
    return jnp.sqrt(jnp.mean((m_recon_arr[..., 2] - m_true_arr[..., 2]) ** 2))


def depth_correlation(m_recon, m_true, yx, *, component: int = 0, eps: float = 1e-12) -> jnp.ndarray:
    m_recon_arr = _as_spatial_vector_field(m_recon, name="m_recon")
    m_true_arr = _as_spatial_vector_field(m_true, name="m_true")
    y, x = map(int, yx)
    recon_profile = m_recon_arr[:, y, x, component]
    true_profile = m_true_arr[:, y, x, component]
    recon_centered = recon_profile - jnp.mean(recon_profile)
    true_centered = true_profile - jnp.mean(true_profile)
    numerator = jnp.sum(recon_centered * true_centered)
    denominator = jnp.linalg.norm(recon_centered) * jnp.linalg.norm(true_centered)
    return numerator / jnp.maximum(denominator, eps)


def vortex_core_z_error(m_recon, m_true, yx) -> jnp.ndarray:
    m_recon_arr = _as_spatial_vector_field(m_recon, name="m_recon")
    m_true_arr = _as_spatial_vector_field(m_true, name="m_true")
    y, x = map(int, yx)
    recon_idx = jnp.argmax(m_recon_arr[:, y, x, 2])
    true_idx = jnp.argmax(m_true_arr[:, y, x, 2])
    return jnp.abs(recon_idx - true_idx)


def equilibrium_residual(
    m_recon,
    backend_alt,
    *,
    rho=None,
    support_threshold: float = 1e-6,
) -> jnp.ndarray:
    """Return an equilibrium-consistency proxy based on the alternative backend.

    The metric computes the RMS norm of the torque proxy
    ``m × dE/dm`` using ``backend_alt`` as the energy functional.

    Parameters
    ----------
    m_recon
        Reconstructed magnetization field with shape ``(Z, Y, X, 3)``.
    backend_alt
        Physics backend used only for this diagnostic. It should differ from
        the prior used in the reconstruction so the metric does not score
        trivially well by re-evaluating the optimized objective.
    rho
        Optional support/density field with shape ``(Z, Y, X)``. When omitted,
        support is inferred from the non-zero norm of ``m_recon``.
    support_threshold
        Threshold used when inferring the support mask from ``m_recon``.
    """
    m_arr = _as_spatial_vector_field(m_recon, name="m_recon")
    if rho is None:
        rho_arr = jnp.asarray(
            jnp.linalg.norm(m_arr, axis=-1) > support_threshold,
            dtype=m_arr.dtype,
        )
    else:
        rho_arr = jnp.asarray(rho, dtype=m_arr.dtype)
        if rho_arr.shape != m_arr.shape[:-1]:
            raise ValueError(
                f"rho must match m_recon spatial shape {m_arr.shape[:-1]}, got {rho_arr.shape}."
            )

    def total_energy(m_current):
        field = backend_alt.prepare(rho_arr, m_current)
        terms = backend_alt.energies(field)
        if not terms:
            raise ValueError("backend_alt must provide at least one energy term.")
        return jnp.sum(jnp.stack([jnp.asarray(value) for value in terms.values()]))

    grad_energy = jax.grad(total_energy)(m_arr)
    torque = jnp.cross(m_arr, grad_energy)
    return jnp.sqrt(jnp.mean(jnp.sum(torque ** 2, axis=-1)))


def iterations_to_threshold(loss_history, threshold) -> int:
    history = jnp.asarray(loss_history)
    if history.ndim != 1:
        raise ValueError(f"loss_history must be 1D, got {history.shape}.")
    hits = jnp.nonzero(history <= threshold, size=1, fill_value=-1)[0]
    hit = int(hits[0])
    if hit < 0:
        return int(history.shape[0])
    return hit + 1
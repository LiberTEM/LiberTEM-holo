from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..forward import forward_phase_from_density_and_magnetization
from ..synthetic import vortex_magnetization


@dataclass(frozen=True)
class ScaledRhoExperimentResult:
    rho_scaled: jnp.ndarray
    m_scaled_truth: jnp.ndarray
    phi_scaled: jnp.ndarray
    reconstruction: Any
    mean_abs_m: float
    hist_counts: np.ndarray
    hist_edges: np.ndarray


def support_center_yx(rho, *, threshold: float = 0.5) -> tuple[int, int]:
    rho_arr = np.asarray(rho)
    if rho_arr.ndim != 3:
        raise ValueError(f"rho must have shape (Z, Y, X), got {rho_arr.shape}.")
    support = np.argwhere(np.max(rho_arr, axis=0) > threshold)
    if support.size == 0:
        raise ValueError("Support mask is empty at the requested threshold.")
    center = np.round(support.mean(axis=0)).astype(int)
    return int(center[0]), int(center[1])


def analytic_vortex_init(
    rho,
    *,
    core_radius: float | None = None,
    chirality: float = 1.0,
    magnitude: float = 1.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    rho_arr = jnp.asarray(rho, dtype=dtype)
    if rho_arr.ndim != 3:
        raise ValueError(f"rho must have shape (Z, Y, X), got {rho_arr.shape}.")
    shape = tuple(int(v) for v in rho_arr.shape)
    if core_radius is None:
        core_radius = max(1.5, shape[0] / 32.0)
    return vortex_magnetization(
        shape,
        support_zyx=rho_arr,
        core_radius=core_radius,
        chirality=chirality,
        magnitude=magnitude,
        dtype=dtype,
    )


def run_with_scaled_rho(
    run_reconstruction: Callable[[jnp.ndarray, jnp.ndarray], Any],
    rho_true,
    m_true,
    pixel_size,
    *,
    scale: float = 1.5,
    axis: str = "z",
    support_threshold: float = 0.5,
    histogram_bins: int = 32,
) -> ScaledRhoExperimentResult:
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}.")

    rho_arr = jnp.asarray(rho_true)
    m_arr = jnp.asarray(m_true)
    if m_arr.shape != rho_arr.shape + (3,):
        raise ValueError(
            "m_true must have shape rho_true.shape + (3,), "
            f"got rho {rho_arr.shape} and m {m_arr.shape}."
        )

    rho_scaled = rho_arr * scale
    m_scaled_truth = m_arr / scale
    phi_scaled = forward_phase_from_density_and_magnetization(
        rho=rho_scaled,
        magnetization_3d=m_scaled_truth,
        pixel_size=pixel_size,
        axis=axis,
    )
    reconstruction = run_reconstruction(phi_scaled, rho_scaled)
    m_recon = getattr(reconstruction, "m_recon", None)
    if m_recon is None:
        raise ValueError(
            "run_reconstruction must return an object with an m_recon attribute."
        )

    m_recon_arr = np.asarray(m_recon)
    magnitude = np.linalg.norm(m_recon_arr, axis=-1)
    support_mask = np.asarray(rho_scaled > support_threshold)
    support_values = magnitude[support_mask]
    if support_values.size == 0:
        hist_counts = np.zeros((histogram_bins,), dtype=int)
        hist_edges = np.linspace(0.0, 1.0, histogram_bins + 1)
        mean_abs_m = float("nan")
    else:
        upper = max(1.5, float(np.max(support_values)))
        hist_counts, hist_edges = np.histogram(
            support_values,
            bins=histogram_bins,
            range=(0.0, upper),
        )
        mean_abs_m = float(np.mean(support_values))

    return ScaledRhoExperimentResult(
        rho_scaled=rho_scaled,
        m_scaled_truth=m_scaled_truth,
        phi_scaled=phi_scaled,
        reconstruction=reconstruction,
        mean_abs_m=mean_abs_m,
        hist_counts=hist_counts,
        hist_edges=hist_edges,
    )
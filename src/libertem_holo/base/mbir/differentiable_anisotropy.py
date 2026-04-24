from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
import unxt as u

from .forward import forward_phase_from_density_and_magnetization
from .kernel import build_rdfc_kernel


def _default_axis2_hint() -> np.ndarray:
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class DifferentiableAnisotropyConfig:
    """Weights and geometry for a differentiable phase/physics objective.

    This is a JAX-native prototype surface for fitting magnetization together
    with anisotropy directions. It intentionally avoids the stateful NeuralMag
    minimizer and instead exposes a scalar objective that can be differentiated
    with respect to both the magnetization field and anisotropy parameters.
    """

    phase_pad: int = 0
    phase_weight: float = 1.0
    smoothness_weight: float = 1e-3
    anisotropy_weight: float = 1e-2
    norm_weight: float = 1e-1
    support_threshold: float = 0.5
    projection_axis: str = "z"
    geometry: str = "disc"
    axis2_hint: np.ndarray = field(default_factory=_default_axis2_hint)


@dataclass(frozen=True)
class DifferentiableAnisotropyFitResult:
    """Result of joint phase/anisotropy optimization."""

    magnetization_zyx: np.ndarray
    axis_angles: np.ndarray
    axis1: np.ndarray
    axis2: np.ndarray
    axis3: np.ndarray
    history: dict[str, np.ndarray]


def _safe_normalize(vector: jax.Array, eps: float = 1e-8) -> jax.Array:
    vector = jnp.asarray(vector)
    return vector / jnp.maximum(jnp.linalg.norm(vector), eps)


def axis_angles_to_unit_vector(theta: jax.Array, phi: jax.Array) -> jax.Array:
    """Convert spherical angles to a unit vector.

    ``theta`` is the polar angle from +z, ``phi`` is the azimuth about z.
    """
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)
    return jnp.stack(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ],
        axis=0,
    )


def unit_vector_to_axis_angles(axis: np.ndarray) -> tuple[float, float]:
    """Convert a unit vector back to spherical angles for notebook setup."""
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    theta = float(np.arccos(np.clip(axis[2], -1.0, 1.0)))
    phi = float(np.arctan2(axis[1], axis[0]))
    return theta, phi


def orthonormalize_anisotropy_axes_jax(
    axis1: jax.Array,
    axis2_hint: jax.Array | np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Construct a differentiable orthonormal basis from a primary axis."""
    axis1 = _safe_normalize(axis1)
    if axis2_hint is None:
        axis2_hint = _default_axis2_hint()
    hint = _safe_normalize(jnp.asarray(axis2_hint, dtype=axis1.dtype))

    axis2 = hint - jnp.dot(hint, axis1) * axis1
    axis2_norm = jnp.linalg.norm(axis2)

    fallback_hint = jnp.where(
        jnp.abs(axis1[0]) < 0.9,
        jnp.array([1.0, 0.0, 0.0], dtype=axis1.dtype),
        jnp.array([0.0, 1.0, 0.0], dtype=axis1.dtype),
    )
    fallback_axis2 = fallback_hint - jnp.dot(fallback_hint, axis1) * axis1
    axis2 = jnp.where(axis2_norm > 1e-6, axis2, fallback_axis2)
    axis2 = _safe_normalize(axis2)
    axis3 = _safe_normalize(jnp.cross(axis1, axis2))
    axis2 = _safe_normalize(jnp.cross(axis3, axis1))
    return axis1, axis2, axis3


def angle_params_to_anisotropy_axes(
    axis_angles: jax.Array,
    axis2_hint: jax.Array | np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map ``[theta, phi]`` to an orthonormal anisotropy basis."""
    axis_angles = jnp.asarray(axis_angles)
    if axis_angles.shape != (2,):
        raise ValueError(f"axis_angles must have shape (2,), got {axis_angles.shape!r}.")
    axis1 = axis_angles_to_unit_vector(axis_angles[0], axis_angles[1])
    return orthonormalize_anisotropy_axes_jax(axis1, axis2_hint=axis2_hint)


def cubic_anisotropy_invariant(
    magnetization_zyx: jax.Array,
    axis1: jax.Array,
    axis2: jax.Array,
    axis3: jax.Array | None = None,
) -> jax.Array:
    """Return the dimensionless cubic-anisotropy invariant per voxel."""
    magnetization_zyx = jnp.asarray(magnetization_zyx)
    axis1 = _safe_normalize(axis1)
    axis2 = _safe_normalize(axis2)
    if axis3 is None:
        axis3 = _safe_normalize(jnp.cross(axis1, axis2))
    else:
        axis3 = _safe_normalize(axis3)

    a1 = jnp.tensordot(magnetization_zyx, axis1, axes=[-1, 0])
    a2 = jnp.tensordot(magnetization_zyx, axis2, axes=[-1, 0])
    a3 = jnp.tensordot(magnetization_zyx, axis3, axes=[-1, 0])
    return (a1 * a1) * (a2 * a2) + (a2 * a2) * (a3 * a3) + (a3 * a3) * (a1 * a1)


def mean_cubic_anisotropy_loss(
    rho_zyx: jax.Array,
    magnetization_zyx: jax.Array,
    axis1: jax.Array,
    axis2: jax.Array,
    *,
    support_threshold: float = 0.5,
) -> jax.Array:
    """Average cubic-anisotropy invariant inside the support."""
    mask = (jnp.asarray(rho_zyx) > support_threshold).astype(jnp.asarray(magnetization_zyx).dtype)
    invariant = cubic_anisotropy_invariant(magnetization_zyx, axis1, axis2)
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(mask * invariant) / denom


def smoothness_loss_3d(
    rho_zyx: jax.Array,
    magnetization_zyx: jax.Array,
    *,
    support_threshold: float = 0.5,
) -> jax.Array:
    """Simple 3D exchange-like smoothness loss over nearest-neighbor pairs."""
    rho_zyx = jnp.asarray(rho_zyx)
    magnetization_zyx = jnp.asarray(magnetization_zyx)
    mask = rho_zyx > support_threshold

    def pair_loss(diff: jax.Array, pair_mask: jax.Array) -> jax.Array:
        weight = pair_mask.astype(diff.dtype)[..., None]
        denom = jnp.maximum(jnp.sum(weight), 1.0)
        return jnp.sum((diff * weight) ** 2) / denom

    loss_z = pair_loss(magnetization_zyx[1:, :, :, :] - magnetization_zyx[:-1, :, :, :], mask[1:, :, :] & mask[:-1, :, :])
    loss_y = pair_loss(magnetization_zyx[:, 1:, :, :] - magnetization_zyx[:, :-1, :, :], mask[:, 1:, :] & mask[:, :-1, :])
    loss_x = pair_loss(magnetization_zyx[:, :, 1:, :] - magnetization_zyx[:, :, :-1, :], mask[:, :, 1:] & mask[:, :, :-1])
    return loss_z + loss_y + loss_x


def support_norm_loss(
    rho_zyx: jax.Array,
    magnetization_zyx: jax.Array,
    *,
    support_threshold: float = 0.5,
) -> jax.Array:
    """Penalize non-unit support vectors and non-zero off-support vectors."""
    rho_zyx = jnp.asarray(rho_zyx)
    magnetization_zyx = jnp.asarray(magnetization_zyx)
    support = (rho_zyx > support_threshold).astype(magnetization_zyx.dtype)
    norms_sq = jnp.sum(magnetization_zyx * magnetization_zyx, axis=-1)
    safe_norms = jnp.sqrt(norms_sq + 1e-12)
    on_support = support * (safe_norms - 1.0) ** 2
    off_support = (1.0 - support) * norms_sq
    denom = jnp.maximum(jnp.asarray(rho_zyx.size, dtype=magnetization_zyx.dtype), 1.0)
    return (jnp.sum(on_support) + jnp.sum(off_support)) / denom


def project_magnetization_to_support_jax(
    rho_zyx: jax.Array,
    magnetization_zyx: jax.Array,
    *,
    support_threshold: float = 0.5,
) -> jax.Array:
    """Project a magnetization field to unit norm on support and zero elsewhere."""
    rho_zyx = jnp.asarray(rho_zyx)
    magnetization_zyx = jnp.asarray(magnetization_zyx)
    support = (rho_zyx > support_threshold).astype(magnetization_zyx.dtype)[..., None]
    norms_sq = jnp.sum(magnetization_zyx * magnetization_zyx, axis=-1, keepdims=True)
    safe_norms = jnp.sqrt(norms_sq + 1e-12)
    return support * (magnetization_zyx / safe_norms)


def _history_row(
    axis_angles: jax.Array,
    terms: dict[str, jax.Array],
    grad_axis: jax.Array,
    grad_m: jax.Array,
) -> dict[str, np.ndarray]:
    return {
        "loss_total": np.asarray(terms["total"]),
        "loss_phase": np.asarray(terms["phase"]),
        "loss_smoothness": np.asarray(terms["smoothness"]),
        "loss_anisotropy": np.asarray(terms["anisotropy"]),
        "loss_norm": np.asarray(terms["norm"]),
        "axis_gradient_norm": np.asarray(jnp.linalg.norm(grad_axis)),
        "magnetization_gradient_norm": np.asarray(jnp.linalg.norm(grad_m)),
        "axis_angles": np.asarray(axis_angles),
        "axis1": np.asarray(terms["axis1"]),
        "axis2": np.asarray(terms["axis2"]),
        "axis3": np.asarray(terms["axis3"]),
    }


def _stack_history(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([row[key] for row in rows], axis=0)
        for key in rows[0]
    }


def pad_phase_view_zyx_jax(
    rho_zyx: jax.Array,
    magnetization_zyx: jax.Array,
    phase_pad: int,
) -> tuple[jax.Array, jax.Array]:
    """Pad a ZYX volume in the projected Y/X plane for phase fitting."""
    rho_zyx = jnp.asarray(rho_zyx)
    magnetization_zyx = jnp.asarray(magnetization_zyx)
    if phase_pad <= 0:
        return rho_zyx, magnetization_zyx
    rho_view = jnp.pad(rho_zyx, ((0, 0), (phase_pad, phase_pad), (phase_pad, phase_pad)))
    magnetization_view = jnp.pad(
        magnetization_zyx,
        ((0, 0), (phase_pad, phase_pad), (phase_pad, phase_pad), (0, 0)),
    )
    return rho_view, magnetization_view


def phase_data_loss(
    rho_zyx: jax.Array,
    magnetization_zyx: jax.Array,
    phase_target: jax.Array,
    *,
    cellsize_nm: float,
    phase_pad: int = 0,
    projection_axis: str = "z",
    geometry: str = "disc",
    rdfc_kernel: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """Mean-squared phase residual for a padded 3D magnetization volume."""
    phase_target = jnp.asarray(phase_target)
    rho_view, magnetization_view = pad_phase_view_zyx_jax(rho_zyx, magnetization_zyx, phase_pad)
    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(tuple(phase_target.shape), geometry=geometry)
    predicted_phase = forward_phase_from_density_and_magnetization(
        rho=rho_view,
        magnetization_3d=magnetization_view,
        pixel_size=u.Quantity(cellsize_nm, "nm"),
        axis=projection_axis,
        geometry=geometry,
        rdfc_kernel=rdfc_kernel,
    )
    residual = predicted_phase - phase_target
    return 0.5 * jnp.mean(residual * residual)


def joint_phase_anisotropy_loss_terms(
    magnetization_zyx: jax.Array,
    axis_angles: jax.Array,
    rho_zyx: jax.Array,
    phase_target: jax.Array,
    *,
    cellsize_nm: float,
    config: DifferentiableAnisotropyConfig | None = None,
    rdfc_kernel: dict[str, jax.Array] | None = None,
) -> dict[str, jax.Array]:
    """Return differentiable phase/physics loss components for joint fitting."""
    config = config or DifferentiableAnisotropyConfig()
    axis1, axis2, axis3 = angle_params_to_anisotropy_axes(axis_angles, axis2_hint=config.axis2_hint)
    phase_term = phase_data_loss(
        rho_zyx,
        magnetization_zyx,
        phase_target,
        cellsize_nm=cellsize_nm,
        phase_pad=config.phase_pad,
        projection_axis=config.projection_axis,
        geometry=config.geometry,
        rdfc_kernel=rdfc_kernel,
    )
    smoothness_term = smoothness_loss_3d(
        rho_zyx,
        magnetization_zyx,
        support_threshold=config.support_threshold,
    )
    anisotropy_term = mean_cubic_anisotropy_loss(
        rho_zyx,
        magnetization_zyx,
        axis1,
        axis2,
        support_threshold=config.support_threshold,
    )
    norm_term = support_norm_loss(
        rho_zyx,
        magnetization_zyx,
        support_threshold=config.support_threshold,
    )
    total = (
        config.phase_weight * phase_term
        + config.smoothness_weight * smoothness_term
        + config.anisotropy_weight * anisotropy_term
        + config.norm_weight * norm_term
    )
    return {
        "total": total,
        "phase": phase_term,
        "smoothness": smoothness_term,
        "anisotropy": anisotropy_term,
        "norm": norm_term,
        "axis1": axis1,
        "axis2": axis2,
        "axis3": axis3,
    }


def joint_phase_anisotropy_loss(
    magnetization_zyx: jax.Array,
    axis_angles: jax.Array,
    rho_zyx: jax.Array,
    phase_target: jax.Array,
    *,
    cellsize_nm: float,
    config: DifferentiableAnisotropyConfig | None = None,
    rdfc_kernel: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """Scalar joint loss for differentiable anisotropy fitting."""
    return joint_phase_anisotropy_loss_terms(
        magnetization_zyx,
        axis_angles,
        rho_zyx,
        phase_target,
        cellsize_nm=cellsize_nm,
        config=config,
        rdfc_kernel=rdfc_kernel,
    )["total"]


def optimize_joint_phase_anisotropy(
    magnetization_zyx: jax.Array,
    axis_angles: jax.Array,
    rho_zyx: jax.Array,
    phase_target: jax.Array,
    *,
    cellsize_nm: float,
    config: DifferentiableAnisotropyConfig | None = None,
    rdfc_kernel: dict[str, jax.Array] | None = None,
    n_iter: int = 10,
    magnetization_learning_rate: float = 1e-2,
    axis_learning_rate: float = 1e-4,
) -> DifferentiableAnisotropyFitResult:
    """Run a small optax-based joint optimization over magnetization and axis angles."""
    if n_iter < 0:
        raise ValueError(f"n_iter must be >= 0, got {n_iter}.")

    config = config or DifferentiableAnisotropyConfig()
    rho_zyx = jnp.asarray(rho_zyx)
    phase_target = jnp.asarray(phase_target)
    axis_angles = jnp.asarray(axis_angles)
    magnetization_zyx = project_magnetization_to_support_jax(
        rho_zyx,
        magnetization_zyx,
        support_threshold=config.support_threshold,
    )

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(tuple(phase_target.shape), geometry=config.geometry)

    axis_optimizer = optax.adam(axis_learning_rate)
    magnetization_optimizer = optax.adam(magnetization_learning_rate)
    axis_opt_state = axis_optimizer.init(axis_angles)
    magnetization_opt_state = magnetization_optimizer.init(magnetization_zyx)

    def loss_with_aux(axis_angles_arg: jax.Array, magnetization_arg: jax.Array):
        terms = joint_phase_anisotropy_loss_terms(
            magnetization_arg,
            axis_angles_arg,
            rho_zyx,
            phase_target,
            cellsize_nm=cellsize_nm,
            config=config,
            rdfc_kernel=rdfc_kernel,
        )
        aux = {
            "phase": terms["phase"],
            "smoothness": terms["smoothness"],
            "anisotropy": terms["anisotropy"],
            "norm": terms["norm"],
            "axis1": terms["axis1"],
            "axis2": terms["axis2"],
            "axis3": terms["axis3"],
        }
        return terms["total"], aux

    evaluate_and_grad = jax.value_and_grad(loss_with_aux, argnums=(0, 1), has_aux=True)

    @jax.jit
    def apply_step(
        axis_angles_arg: jax.Array,
        magnetization_arg: jax.Array,
        grad_axis_arg: jax.Array,
        grad_m_arg: jax.Array,
        axis_opt_state_arg: optax.OptState,
        magnetization_opt_state_arg: optax.OptState,
    ) -> tuple[jax.Array, jax.Array, optax.OptState, optax.OptState]:
        axis_updates, next_axis_opt_state = axis_optimizer.update(
            grad_axis_arg,
            axis_opt_state_arg,
            axis_angles_arg,
        )
        next_axis_angles = optax.apply_updates(axis_angles_arg, axis_updates)
        magnetization_updates, next_magnetization_opt_state = magnetization_optimizer.update(
            grad_m_arg,
            magnetization_opt_state_arg,
            magnetization_arg,
        )
        next_magnetization = optax.apply_updates(magnetization_arg, magnetization_updates)
        next_magnetization = project_magnetization_to_support_jax(
            rho_zyx,
            next_magnetization,
            support_threshold=config.support_threshold,
        )
        return (
            next_axis_angles,
            next_magnetization,
            next_axis_opt_state,
            next_magnetization_opt_state,
        )

    history_rows: list[dict[str, np.ndarray]] = []

    (loss_total, aux_terms), (grad_axis, grad_m) = evaluate_and_grad(axis_angles, magnetization_zyx)
    terms = {"total": loss_total, **aux_terms}
    history_rows.append(_history_row(axis_angles, terms, grad_axis, grad_m))

    for _ in range(n_iter):
        axis_angles, magnetization_zyx, axis_opt_state, magnetization_opt_state = apply_step(
            axis_angles,
            magnetization_zyx,
            grad_axis,
            grad_m,
            axis_opt_state,
            magnetization_opt_state,
        )
        (loss_total, aux_terms), (grad_axis, grad_m) = evaluate_and_grad(axis_angles, magnetization_zyx)
        terms = {"total": loss_total, **aux_terms}
        history_rows.append(_history_row(axis_angles, terms, grad_axis, grad_m))

    final_axis1 = np.asarray(terms["axis1"])
    final_axis2 = np.asarray(terms["axis2"])
    final_axis3 = np.asarray(terms["axis3"])
    return DifferentiableAnisotropyFitResult(
        magnetization_zyx=np.asarray(magnetization_zyx),
        axis_angles=np.asarray(axis_angles),
        axis1=final_axis1,
        axis2=final_axis2,
        axis3=final_axis3,
        history=_stack_history(history_rows),
    )
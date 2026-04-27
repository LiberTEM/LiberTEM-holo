import jax
import jax.numpy as jnp
import numpy as np
import unxt as u

from libertem_holo.base.mbir import (
    DifferentiableAnisotropyConfig,
    angle_params_to_anisotropy_axes,
    joint_phase_anisotropy_loss,
    optimize_joint_phase_anisotropy,
    pad_phase_view_zyx_jax,
    phase_data_loss,
    unit_vector_to_axis_angles,
)
from libertem_holo.base.mbir.forward import forward_phase_from_density_and_magnetization
from libertem_holo.base.mbir.kernel import build_rdfc_kernel


def _random_unit_volume(shape=(2, 3, 4), seed=0):
    rho = np.ones(shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    m = rng.normal(size=shape + (3,)).astype(np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    m = m / np.where(norms > 0.0, norms, 1.0)
    return rho, m


def test_angle_params_to_axes_returns_orthonormal_basis():
    axis = np.array([0.3, -0.4, 0.8660254], dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    theta, phi = unit_vector_to_axis_angles(axis)

    axis1, axis2, axis3 = angle_params_to_anisotropy_axes(jnp.array([theta, phi], dtype=jnp.float32))

    np.testing.assert_allclose(np.linalg.norm(np.asarray(axis1)), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(axis2)), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(axis3)), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(np.asarray(axis1), np.asarray(axis2)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(np.asarray(axis1), np.asarray(axis3)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(np.asarray(axis2), np.asarray(axis3)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(axis1), axis, atol=1e-6)


def test_angle_params_to_axes_handles_parallel_hint_with_finite_gradients():
    axis2_hint = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    axis_angles = jnp.array([0.0, 0.0], dtype=jnp.float32)

    def loss_fn(axis_angles_arg):
        axis1, axis2, axis3 = angle_params_to_anisotropy_axes(axis_angles_arg, axis2_hint=axis2_hint)
        return jnp.sum(axis1 + axis2 + axis3)

    grad = jax.grad(loss_fn)(axis_angles)

    assert grad.shape == (2,)
    assert jnp.all(jnp.isfinite(grad))


def test_phase_data_loss_prefers_matching_magnetization():
    rho, m = _random_unit_volume()
    rho_view, m_view = pad_phase_view_zyx_jax(rho, m, phase_pad=1)
    phase_target = forward_phase_from_density_and_magnetization(
        rho_view,
        m_view,
        pixel_size=u.Quantity(2.0, "nm"),
        axis="z",
        geometry="disc",
    )

    matching = phase_data_loss(
        rho,
        m,
        phase_target,
        cellsize_nm=2.0,
        phase_pad=1,
        projection_axis="z",
        geometry="disc",
    )
    shifted = phase_data_loss(
        rho,
        jnp.roll(jnp.asarray(m), shift=1, axis=1),
        phase_target,
        cellsize_nm=2.0,
        phase_pad=1,
        projection_axis="z",
        geometry="disc",
    )

    assert float(matching) < 1e-8
    assert float(shifted) > float(matching)


def test_joint_phase_anisotropy_loss_has_finite_gradients():
    rho, m = _random_unit_volume(seed=7)
    config = DifferentiableAnisotropyConfig(
        phase_pad=1,
        phase_weight=1.0,
        smoothness_weight=0.05,
        anisotropy_weight=0.05,
        norm_weight=0.05,
    )
    rho_view, m_view = pad_phase_view_zyx_jax(rho, m, phase_pad=config.phase_pad)
    phase_target = forward_phase_from_density_and_magnetization(
        rho_view,
        m_view,
        pixel_size=u.Quantity(2.0, "nm"),
        axis=config.projection_axis,
        geometry=config.geometry,
    )
    rdfc_kernel = build_rdfc_kernel(tuple(phase_target.shape), geometry=config.geometry)

    axis = np.array([0.45, 0.2, 0.87], dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    theta, phi = unit_vector_to_axis_angles(axis)
    axis_angles = jnp.array([theta, phi], dtype=jnp.float32)

    def loss_fn(axis_angles_arg, magnetization_arg):
        return joint_phase_anisotropy_loss(
            magnetization_arg,
            axis_angles_arg,
            rho,
            phase_target,
            cellsize_nm=2.0,
            config=config,
            rdfc_kernel=rdfc_kernel,
        )

    loss_value, (grad_angles, grad_m) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        axis_angles,
        jnp.asarray(m),
    )

    assert jnp.isfinite(loss_value)
    assert grad_angles.shape == (2,)
    assert grad_m.shape == m.shape
    assert jnp.all(jnp.isfinite(grad_angles))
    assert jnp.all(jnp.isfinite(grad_m))


def test_joint_phase_anisotropy_loss_handles_zero_off_support_gradients():
    rho = np.zeros((2, 3, 4), dtype=np.float32)
    rho[:, 1:, 1:3] = 1.0
    _, m = _random_unit_volume(shape=(2, 3, 4), seed=11)
    m = m * rho[..., None]

    config = DifferentiableAnisotropyConfig(
        phase_pad=1,
        phase_weight=1.0,
        smoothness_weight=0.05,
        anisotropy_weight=0.05,
        norm_weight=0.05,
    )
    rho_view, m_view = pad_phase_view_zyx_jax(rho, m, phase_pad=config.phase_pad)
    phase_target = forward_phase_from_density_and_magnetization(
        rho_view,
        m_view,
        pixel_size=u.Quantity(2.0, "nm"),
        axis=config.projection_axis,
        geometry=config.geometry,
    )

    def loss_fn(magnetization_arg):
        return joint_phase_anisotropy_loss(
            magnetization_arg,
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            rho,
            phase_target,
            cellsize_nm=2.0,
            config=config,
        )

    grad_m = jax.grad(loss_fn)(jnp.asarray(m))

    assert grad_m.shape == m.shape
    assert jnp.all(jnp.isfinite(grad_m))


def test_optimize_joint_phase_anisotropy_reduces_loss_and_preserves_support():
    rho = np.ones((2, 3, 4), dtype=np.float32)
    rho[:, 0, 0] = 0.0
    rho[:, 2, 3] = 0.0
    _, m_true = _random_unit_volume(shape=(2, 3, 4), seed=17)
    _, initial_m = _random_unit_volume(shape=(2, 3, 4), seed=23)
    m_true = m_true * rho[..., None]
    initial_m = initial_m * rho[..., None]

    config = DifferentiableAnisotropyConfig(
        phase_pad=1,
        phase_weight=1.0,
        smoothness_weight=0.01,
        anisotropy_weight=0.01,
        norm_weight=0.01,
    )
    rho_view, m_view = pad_phase_view_zyx_jax(rho, m_true, phase_pad=config.phase_pad)
    phase_target = forward_phase_from_density_and_magnetization(
        rho_view,
        m_view,
        pixel_size=u.Quantity(2.0, "nm"),
        axis=config.projection_axis,
        geometry=config.geometry,
    )

    axis_angles0 = jnp.array([0.35, -0.6], dtype=jnp.float32)

    result = optimize_joint_phase_anisotropy(
        initial_m,
        axis_angles0,
        rho,
        phase_target,
        cellsize_nm=2.0,
        config=config,
        n_iter=6,
        magnetization_learning_rate=1e-2,
        axis_learning_rate=1e-4,
    )

    assert result.history["loss_total"].shape == (7,)
    assert result.history["loss_total"][-1] < result.history["loss_total"][0]

    off_support = result.magnetization_zyx[rho < 0.5]
    on_support_norms = np.linalg.norm(result.magnetization_zyx[rho > 0.5], axis=-1)
    np.testing.assert_allclose(off_support, 0.0, atol=1e-6)
    np.testing.assert_allclose(on_support_norms, 1.0, atol=1e-5)
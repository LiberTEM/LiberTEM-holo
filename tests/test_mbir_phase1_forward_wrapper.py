import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

import libertem_holo.base.mbir as mbir
import unxt as u


def _make_vortex_disc_problem(z=3, y=32, x=32):
    xx, yy = jnp.meshgrid(jnp.arange(x), jnp.arange(y), indexing="ij")
    cy = (y - 1) / 2.0
    cx = (x - 1) / 2.0
    dy = yy - cy
    dx = xx - cx
    r = jnp.sqrt(dx**2 + dy**2)

    radius = 0.35 * min(y, x)
    disc = r <= radius
    rho2d = disc.astype(jnp.float64)
    rho = jnp.broadcast_to(rho2d[..., None], (x, y, z))

    r_safe = jnp.where(r > 0, r, 1.0)
    mx2d = jnp.where(disc, -dy / r_safe, 0.0)
    my2d = jnp.where(disc, dx / r_safe, 0.0)
    mz2d = jnp.zeros_like(mx2d)
    m2d = jnp.stack([mx2d, my2d, mz2d], axis=-1)
    # Explicitly re-normalise to guard against fp overshoot from sqrt
    _norms = jnp.linalg.norm(m2d, axis=-1, keepdims=True)
    m2d = jnp.where(_norms > 0, m2d / jnp.where(_norms > 0, _norms, 1.0), m2d)
    m = jnp.broadcast_to(m2d[..., None, :], (x, y, z, 3))

    return rho, m


def _phase_energy(rho, m, pixel_size, axis="z"):
    phase = mbir.phase_from_magnetisation(
        rho=rho,
        magnetization_3d=m,
        pixel_size=pixel_size,
        reference_induction=mbir.B_REF,
        axis=axis,
    )
    return jnp.mean(phase**2)


def test_forward_phase_uniform_out_of_plane_is_near_zero():
    z, y, x = 3, 24, 24
    rho = jnp.ones((x, y, z), dtype=jnp.float64)
    m = jnp.zeros((x, y, z, 3), dtype=jnp.float64).at[..., 2].set(1.0)

    phase = mbir.phase_from_magnetisation(
        rho=rho,
        magnetization_3d=m,
        pixel_size=u.Quantity(1.0, "nm"),
        reference_induction=mbir.B_REF,
        axis="z",
    )

    assert phase.shape == (y, x)
    assert np.max(np.abs(np.asarray(phase))) < 1e-10


def test_forward_phase_vortex_disc_has_antisymmetric_nontrivial_pattern():
    rho, m = _make_vortex_disc_problem()
    phase = mbir.phase_from_magnetisation(
        rho=rho,
        magnetization_3d=m,
        pixel_size=u.Quantity(1.0, "nm"),
        reference_induction=mbir.B_REF,
        axis="z",
        prw_vec=jnp.array([0.2, 0.0], dtype=jnp.float64),
    )
    phase_np = np.asarray(phase)

    assert np.max(np.abs(phase_np)) > 1e-6
    assert np.min(phase_np) < 0 < np.max(phase_np)

    anti = -phase_np[::-1, ::-1]
    corr = np.corrcoef(phase_np.ravel(), anti.ravel())[0, 1]
    assert corr > 0.9


def test_forward_phase_gradients_are_finite_and_match_shapes():
    rho, m = _make_vortex_disc_problem(z=2, y=20, x=20)
    pixel_size = u.Quantity(1.0, "nm")

    grad_rho, grad_m = jax.grad(
        lambda r, mm: _phase_energy(r, mm, pixel_size, axis="z"),
        argnums=(0, 1),
    )(rho, m)

    assert grad_rho.shape == rho.shape
    assert grad_m.shape == m.shape
    assert jnp.all(jnp.isfinite(grad_rho))
    assert jnp.all(jnp.isfinite(grad_m))


def test_forward_phase_gradient_fd_spot_check():
    rho, m = _make_vortex_disc_problem(z=2, y=20, x=20)
    pixel_size = u.Quantity(1.0, "nm")

    grad_rho = jax.grad(lambda r: _phase_energy(r, m, pixel_size, axis="z"))(rho)
    idx = (13, 10, 0)
    eps = 1e-4

    perturb = jnp.zeros_like(rho).at[idx].set(1.0)
    f_plus = _phase_energy(rho + eps * perturb, m, pixel_size, axis="z")
    f_minus = _phase_energy(rho - eps * perturb, m, pixel_size, axis="z")
    fd = (f_plus - f_minus) / (2.0 * eps)

    assert_allclose(float(fd), float(grad_rho[idx]), rtol=5e-2, atol=2e-4)


def test_forward_phase_aliases_match():
    rho, m = _make_vortex_disc_problem(z=2, y=20, x=20)
    pixel_size = u.Quantity(1.0, "nm")

    phase_requested = mbir.phase_from_magnetisation(
        rho=rho,
        magnetization_3d=m,
        pixel_size=pixel_size,
        reference_induction=mbir.B_REF,
        axis="z",
    )
    phase_us = mbir.phase_from_magnetization(
        rho=rho,
        magnetization_3d=m,
        pixel_size=pixel_size,
        reference_induction=mbir.B_REF,
        axis="z",
    )
    phase_density = mbir.phase_from_density_and_magnetization(
        rho=rho,
        magnetization_3d=m,
        pixel_size=pixel_size,
        reference_induction=mbir.B_REF,
        axis="z",
    )
    phase_old = mbir.forward_phase_from_density_and_magnetization(
        rho=rho,
        magnetization_3d=m,
        pixel_size=pixel_size,
        reference_induction=mbir.B_REF,
        axis="z",
    )

    assert_allclose(np.asarray(phase_requested), np.asarray(phase_us))
    assert_allclose(np.asarray(phase_requested), np.asarray(phase_density))
    assert_allclose(np.asarray(phase_requested), np.asarray(phase_old))

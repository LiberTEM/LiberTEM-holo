import jax
import jax.numpy as jnp
import numpy as np
import pytest

from libertem_holo.base.mbir.equilibrium_orientation_fit import (
    EquilibriumOrientationFitConfig,
    build_equilibrium_orientation_problem,
    finite_difference_axis_gradient_check,
    make_vmapped_steepest_descent_multi_start,
    one_step_match_check,
    phase_loss_after_native_relax,
    phase_loss_after_relax,
    phase_loss_and_axis_grad,
    prepare_equilibrium_fit_target,
    relax_magnetization,
    relax_magnetization_native,
    unit_vector_to_axis_angles,
)


pytest.importorskip("neuralmag")


def _random_unit_volume(shape=(4, 4, 4), seed=0):
    rho = np.ones(shape, dtype=np.float32)
    rho[0, 0, 0] = 0.0
    rho[-1, -1, -1] = 0.0
    rng = np.random.default_rng(seed)
    m = rng.normal(size=shape + (3,)).astype(np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    m = m / np.where(norms > 0.0, norms, 1.0)
    return rho, m * rho[..., None]


def _make_problem():
    rho, m_true = _random_unit_volume(seed=7)
    config = EquilibriumOrientationFitConfig(
        coarse_grain_factor=1,
        demag_p=1,
        phase_pad=1,
        minimizer_max_iter=2,
        minimizer_tol=-1.0,
        outer_steps=2,
        axis_learning_rate=1e-2,
    )
    target = prepare_equilibrium_fit_target(
        "synthetic",
        rho,
        m_true,
        cellsize_nm=2.0,
        config=config,
    )
    problem = build_equilibrium_orientation_problem(
        target,
        Msat_A_per_m=480e3,
        Aex_J_per_m=1e-11,
        Kc1_J_per_m3=-1.35e4,
        config=config,
    )
    _, m0 = _random_unit_volume(seed=11)
    axis = np.array([0.25, -0.1, 0.96], dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    axis_angles = jnp.asarray(unit_vector_to_axis_angles(axis), dtype=jnp.float32)
    return problem, axis_angles, m0


def test_phase_loss_and_axis_grad_is_finite():
    problem, axis_angles, m0 = _make_problem()

    loss, grad, aux = phase_loss_and_axis_grad(problem, axis_angles, m0, max_iter=2, tol=-1.0)

    assert jnp.isfinite(loss)
    assert grad.shape == (2,)
    assert jnp.all(jnp.isfinite(grad))
    assert bool(jnp.isfinite(aux["phase_rms"]))
    assert bool(jnp.isfinite(aux["max_g"]))


def test_one_step_match_check_matches_neuralmag_step():
    problem, axis_angles, m0 = _make_problem()

    check = one_step_match_check(problem, axis_angles, m0)

    assert check["jax_n_iter"] == 1
    assert np.isfinite(check["jax_max_g"])
    assert check["max_abs_m_diff"] < 1e-5


def test_finite_difference_axis_gradient_check_is_reasonable():
    problem, axis_angles, m0 = _make_problem()

    check = finite_difference_axis_gradient_check(problem, axis_angles, m0, eps=1e-2, max_iter=2)

    assert np.all(np.isfinite(check["jacfwd_grad"]))
    assert np.all(np.isfinite(check["finite_difference_grad"]))
    assert np.all(check["abs_error"] < 2e-1)


def test_self_consistent_cube_target_has_small_phase_loss():
    rho = np.ones((4, 4, 4), dtype=np.float32)
    rng = np.random.default_rng(7)
    m0 = rng.normal(size=rho.shape + (3,)).astype(np.float32)
    m0 /= np.maximum(np.linalg.norm(m0, axis=-1, keepdims=True), 1e-12)

    config = EquilibriumOrientationFitConfig(
        coarse_grain_factor=1,
        demag_p=1,
        phase_pad=1,
        minimizer_tol=1e3,
        minimizer_relative_tol=1e-2,
        minimizer_min_iter=8,
        minimizer_stall_patience=64,
        minimizer_stall_relative_improvement=1e-4,
        minimizer_max_iter=512,
        outer_steps=2,
        axis_learning_rate=1e-2,
    )
    true_axis = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    true_axis /= np.linalg.norm(true_axis)
    true_angles = jnp.asarray(unit_vector_to_axis_angles(true_axis), dtype=jnp.float32)

    seed_target = prepare_equilibrium_fit_target(
        "seed",
        rho,
        m0,
        cellsize_nm=2.0,
        config=config,
    )
    seed_problem = build_equilibrium_orientation_problem(
        seed_target,
        Msat_A_per_m=480e3,
        Aex_J_per_m=1e-11,
        Kc1_J_per_m3=-1.35e4,
        config=config,
    )
    m_eq, eq_info = relax_magnetization(seed_problem, true_angles, m0)

    assert bool(np.asarray(eq_info["converged"]))

    eq_target = prepare_equilibrium_fit_target(
        "eq",
        rho,
        np.asarray(m_eq, dtype=np.float32),
        cellsize_nm=2.0,
        config=config,
    )
    eq_problem = build_equilibrium_orientation_problem(
        eq_target,
        Msat_A_per_m=480e3,
        Aex_J_per_m=1e-11,
        Kc1_J_per_m3=-1.35e4,
        config=config,
    )
    loss, aux = phase_loss_after_relax(eq_problem, true_angles, np.asarray(m_eq, dtype=np.float32))

    assert bool(np.asarray(aux["converged"]))
    assert float(np.asarray(loss)) < 1e-4
    assert float(np.asarray(aux["phase_rms"])) < 1.5e-2


def test_self_consistent_cube_target_has_small_native_phase_loss():
    rho = np.ones((4, 4, 4), dtype=np.float32)
    rng = np.random.default_rng(11)
    m0 = rng.normal(size=rho.shape + (3,)).astype(np.float32)
    m0 /= np.maximum(np.linalg.norm(m0, axis=-1, keepdims=True), 1e-12)

    config = EquilibriumOrientationFitConfig(
        coarse_grain_factor=1,
        demag_p=1,
        phase_pad=1,
        minimizer_tol=1e3,
        minimizer_relative_tol=1e-2,
        minimizer_min_iter=8,
        minimizer_stall_patience=64,
        minimizer_stall_relative_improvement=1e-4,
        minimizer_max_iter=512,
        outer_steps=2,
        axis_learning_rate=1e-2,
    )
    true_axis = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    true_axis /= np.linalg.norm(true_axis)
    true_angles = jnp.asarray(unit_vector_to_axis_angles(true_axis), dtype=jnp.float32)

    seed_target = prepare_equilibrium_fit_target(
        "seed-native",
        rho,
        m0,
        cellsize_nm=2.0,
        config=config,
    )
    seed_problem = build_equilibrium_orientation_problem(
        seed_target,
        Msat_A_per_m=480e3,
        Aex_J_per_m=1e-11,
        Kc1_J_per_m3=-1.35e4,
        config=config,
    )
    m_eq, eq_info = relax_magnetization_native(
        seed_problem,
        true_angles,
        m0,
        method="alternating",
    )

    assert eq_info["converged"]

    eq_target = prepare_equilibrium_fit_target(
        "eq-native",
        rho,
        np.asarray(m_eq, dtype=np.float32),
        cellsize_nm=2.0,
        config=config,
    )
    eq_problem = build_equilibrium_orientation_problem(
        eq_target,
        Msat_A_per_m=480e3,
        Aex_J_per_m=1e-11,
        Kc1_J_per_m3=-1.35e4,
        config=config,
    )
    loss, aux = phase_loss_after_native_relax(
        eq_problem,
        true_angles,
        np.asarray(m_eq, dtype=np.float32),
        method="alternating",
    )

    assert bool(np.asarray(aux["converged"]))
    assert float(np.asarray(loss)) < 1e-4
    assert float(np.asarray(aux["phase_rms"])) < 1.5e-2


def test_vmapped_steepest_descent_multi_start_is_finite_on_self_consistent_cube() -> None:
    rho = np.ones((4, 4, 4), dtype=np.float32)
    rng = np.random.default_rng(17)
    m0 = rng.normal(size=rho.shape + (3,)).astype(np.float32)
    m0 /= np.maximum(np.linalg.norm(m0, axis=-1, keepdims=True), 1e-12)

    config = EquilibriumOrientationFitConfig(
        coarse_grain_factor=1,
        demag_p=1,
        phase_pad=1,
        minimizer_tol=1e3,
        minimizer_relative_tol=1e-2,
        minimizer_min_iter=8,
        minimizer_stall_patience=64,
        minimizer_stall_relative_improvement=1e-4,
        minimizer_max_iter=512,
        outer_steps=3,
        axis_learning_rate=5e-2,
    )
    true_axis = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    true_axis /= np.linalg.norm(true_axis)
    true_angles = jnp.asarray(unit_vector_to_axis_angles(true_axis), dtype=jnp.float32)

    seed_target = prepare_equilibrium_fit_target(
        "seed-vmap-sd",
        rho,
        m0,
        cellsize_nm=2.0,
        config=config,
    )
    seed_problem = build_equilibrium_orientation_problem(
        seed_target,
        Msat_A_per_m=480e3,
        Aex_J_per_m=1e-11,
        Kc1_J_per_m3=-1.35e4,
        config=config,
    )
    m_eq, eq_info = relax_magnetization(seed_problem, true_angles, m0)

    assert bool(np.asarray(eq_info["converged"]))

    eq_target = prepare_equilibrium_fit_target(
        "eq-vmap-sd",
        rho,
        np.asarray(m_eq, dtype=np.float32),
        cellsize_nm=2.0,
        config=config,
    )
    eq_problem = build_equilibrium_orientation_problem(
        eq_target,
        Msat_A_per_m=480e3,
        Aex_J_per_m=1e-11,
        Kc1_J_per_m3=-1.35e4,
        config=config,
    )

    start_axes = np.array(
        [
            [+1.0, +1.0, +1.0],
            [+1.0, +1.0, -1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, -1.0, -1.0],
            [-1.0, +1.0, +1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    ) / np.sqrt(3.0)
    init_angles_batch = jnp.asarray(
        [unit_vector_to_axis_angles(axis) for axis in start_axes],
        dtype=jnp.float32,
    )

    run_all = make_vmapped_steepest_descent_multi_start(eq_problem, np.asarray(m_eq, dtype=np.float32))
    final_angles, final_losses, final_rms, loss_history, rms_history, axis1_history = run_all(
        init_angles_batch
    )

    assert final_angles.shape == (8, 2)
    assert final_losses.shape == (8,)
    assert final_rms.shape == (8,)
    assert loss_history.shape == (8, config.outer_steps)
    assert rms_history.shape == (8, config.outer_steps)
    assert axis1_history.shape == (8, config.outer_steps, 3)
    assert np.all(np.isfinite(np.asarray(final_angles)))
    assert np.all(np.isfinite(np.asarray(final_losses)))
    assert np.all(np.isfinite(np.asarray(final_rms)))
    assert np.all(np.isfinite(np.asarray(loss_history)))
    assert np.all(np.isfinite(np.asarray(rms_history)))
    assert np.all(np.isfinite(np.asarray(axis1_history)))
    assert float(np.min(np.asarray(final_losses))) < 1.0


def test_phase_loss_after_relax_vmap_matches_single_calls() -> None:
    rho = np.ones((4, 4, 4), dtype=np.float32)
    rng = np.random.default_rng(3)
    m0 = rng.normal(size=rho.shape + (3,)).astype(np.float32)
    m0 /= np.maximum(np.linalg.norm(m0, axis=-1, keepdims=True), 1e-12)

    config = EquilibriumOrientationFitConfig(
        coarse_grain_factor=1,
        demag_p=1,
        phase_pad=1,
        minimizer_tol=1e3,
        minimizer_relative_tol=1e-2,
        minimizer_min_iter=8,
        minimizer_stall_patience=64,
        minimizer_stall_relative_improvement=1e-4,
        minimizer_max_iter=256,
        outer_steps=2,
        axis_learning_rate=1e-2,
    )
    true_axis = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    true_axis /= np.linalg.norm(true_axis)
    true_angles = jnp.asarray(unit_vector_to_axis_angles(true_axis), dtype=jnp.float32)

    seed_target = prepare_equilibrium_fit_target(
        "seed-vmap-compare",
        rho,
        m0,
        cellsize_nm=2.0,
        config=config,
    )
    seed_problem = build_equilibrium_orientation_problem(
        seed_target,
        Msat_A_per_m=480e3,
        Aex_J_per_m=1e-11,
        Kc1_J_per_m3=-1.35e4,
        config=config,
    )
    m_eq, eq_info = relax_magnetization(seed_problem, true_angles, m0, method="alternating")

    assert bool(np.asarray(eq_info["converged"]))

    eq_target = prepare_equilibrium_fit_target(
        "eq-vmap-compare",
        rho,
        np.asarray(m_eq, dtype=np.float32),
        cellsize_nm=2.0,
        config=config,
    )
    eq_problem = build_equilibrium_orientation_problem(
        eq_target,
        Msat_A_per_m=480e3,
        Aex_J_per_m=1e-11,
        Kc1_J_per_m3=-1.35e4,
        config=config,
    )
    m_init = np.asarray(eq_target.m_xyz, dtype=np.float32)

    start_axes = np.array(
        [
            [+1.0, +1.0, +1.0],
            [+1.0, +1.0, -1.0],
            [+1.0, -1.0, +1.0],
            [-1.0, +1.0, +1.0],
        ],
        dtype=np.float32,
    ) / np.sqrt(3.0)
    init_angles_batch = jnp.asarray(
        [unit_vector_to_axis_angles(axis) for axis in start_axes],
        dtype=jnp.float32,
    )

    def run_one(axis_angles):
        loss, aux = phase_loss_after_relax(eq_problem, axis_angles, m_init, method="alternating")
        return loss, aux["phase_rms"], aux["n_iter"], aux["max_g"], aux["converged"]

    single_results = [tuple(np.asarray(value) for value in run_one(axis_angles)) for axis_angles in init_angles_batch]
    batched_results = jax.jit(jax.vmap(run_one))(init_angles_batch)
    loss_batch, rms_batch, n_iter_batch, max_g_batch, converged_batch = [
        np.asarray(value) for value in batched_results
    ]

    single_loss = np.asarray([float(result[0]) for result in single_results], dtype=np.float32)
    single_rms = np.asarray([float(result[1]) for result in single_results], dtype=np.float32)
    single_n_iter = np.asarray([int(result[2]) for result in single_results], dtype=np.int32)
    single_max_g = np.asarray([float(result[3]) for result in single_results], dtype=np.float32)
    single_converged = np.asarray([bool(result[4]) for result in single_results], dtype=bool)

    assert np.allclose(loss_batch, single_loss, rtol=1e-3, atol=1e-12)
    assert np.allclose(rms_batch, single_rms, rtol=1e-3, atol=2e-8)
    assert np.array_equal(n_iter_batch, single_n_iter)
    assert np.all(np.isfinite(max_g_batch))
    assert np.all(max_g_batch > 0.0)
    assert np.array_equal(converged_batch, single_converged)
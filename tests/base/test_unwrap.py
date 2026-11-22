import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.unwrap import (
    IRLSOptions,
    L1BranchesPrior,
    L1PhasePrior,
    amplitude_edge_weights,
    derivative_variance,
    phase_unwrap,
    quality_unwrap,
    unwrap_phase_irls,
    unwrap_phase_laplacian,
)


@pytest.mark.with_numba
def test_quality_unwrap_smoketest():
    rng = np.random.default_rng()
    phase = rng.random(128*128).reshape(128, 128)
    dv = derivative_variance(phase)
    _ = quality_unwrap(phase, dv)


def test_phase_unwrap_helper_smoke():
    rng = np.random.default_rng()
    phase = rng.random(128*128).reshape(128, 128)
    a = phase_unwrap(image=phase, method="skimage")
    b = phase_unwrap(image=np.exp(1j * phase), method="skimage")

    assert np.allclose(a, b)


@pytest.mark.parametrize(
    "method", ["skimage", "quality", "laplacian", "irls", "admm"],
)
def test_phase_unwrap_helper_methods(method: str):
    rng = np.random.default_rng()
    phase = rng.random(128*128).reshape(128, 128)
    _ = phase_unwrap(image=phase, method=method)


def test_unwrap_phase_laplacian():
    rng = np.random.default_rng()
    phase = rng.random(128*128).reshape(128, 128)
    _ = unwrap_phase_laplacian(wrapped_phase=phase)


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_unwrap_laplacian_with_backends(backend):
    rng = np.random.default_rng()
    phase = rng.random(128*128).reshape(128, 128)

    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    phase = xp.asarray(phase)
    _ = unwrap_phase_laplacian(wrapped_phase=phase, xp=xp)


# IRLS unwrapping, checked by properties of a correct unwrap rather than against
# stored results.

def _wrap(value):
    return np.angle(np.exp(1j * value))


def _assert_congruent(unwrapped, wrapped):
    """The result differs from the input by whole turns at every pixel."""
    turns = (unwrapped - wrapped) / (2 * np.pi)
    np.testing.assert_allclose(turns, np.round(turns), rtol=0, atol=1e-9)


def _error_against(unwrapped, truth):
    """Worst difference from the truth, up to the constant an unwrap cannot know."""
    offset = (unwrapped - truth).mean()
    return np.abs(unwrapped - truth - offset).max()


def _vortex(shape, center=None):
    """A residue: no continuous field matches all of its wrapped gradients."""
    rows, cols = shape
    y, x = np.mgrid[:rows, :cols].astype(np.float64)
    cy, cx = center if center is not None else (rows / 2 - 0.5, cols / 2 - 0.5)
    return np.arctan2(y - cy, x - cx)


def _dipole():
    """Two residues of opposite charge, so the solve has a cut to place between them."""
    return _wrap(_vortex((40, 48)) - _vortex((40, 48), center=(12, 30)))


def _cg_iterations(result):
    return sum(r.iterations for r in result.rounds)


def test_irls_linear_ramp_comes_back_exactly():
    y, x = np.mgrid[:24, :31]
    truth = 0.7 * y + 0.9 * x
    wrapped = _wrap(truth)
    result = unwrap_phase_irls(wrapped)
    _assert_congruent(result.phase, wrapped)
    assert result.status == "converged"
    assert _error_against(result.phase, truth) < 1e-6


def test_irls_smooth_field_comes_back():
    y, x = np.mgrid[:32, :32] / 32
    truth = 12 * (y - 0.5) ** 2 + 9 * (x - 0.3) ** 2 + 4 * y * x
    wrapped = _wrap(truth)
    result = unwrap_phase_irls(wrapped)
    _assert_congruent(result.phase, wrapped)
    assert _error_against(result.phase, truth) < 1e-6


def test_irls_residue_is_isolated_by_the_weights():
    wrapped = _vortex((32, 32))
    result = unwrap_phase_irls(wrapped)
    _assert_congruent(result.phase, wrapped)
    # Uniform weights would mean the reweighting never engaged:
    assert result.weights_x.min() < result.weights_x.max() / 10


def test_irls_constant_phase_is_left_alone():
    wrapped = np.full((16, 20), 0.4)
    result = unwrap_phase_irls(wrapped)
    np.testing.assert_allclose(result.phase, 0.4, rtol=0, atol=1e-9)
    assert np.all(result.branches == 0)


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 7), (7, 1), (2, 2), (13, 29), (29, 13)],
)
def test_irls_odd_and_rectangular_shapes(shape):
    y, x = np.mgrid[:shape[0], :shape[1]]
    wrapped = _wrap(0.35 * y + 0.55 * x)
    result = unwrap_phase_irls(wrapped)
    assert result.phase.shape == shape
    assert result.weights_x.shape == (shape[0], shape[1] - 1)
    assert result.weights_y.shape == (shape[0] - 1, shape[1])
    _assert_congruent(result.phase, wrapped)
    assert _error_against(result.phase, 0.35 * y + 0.55 * x) < 1e-6


def test_irls_iteration_limit_is_reported():
    # A residue: a field without residues needs no rounds at all, so it could not
    # hit the iteration limit.
    wrapped = _vortex((40, 40))
    result = unwrap_phase_irls(
        wrapped,
        options=IRLSOptions(iterations_per_round=1, cg_relative_tolerance=1e-14),
    )
    assert result.status == "iteration_limit"
    assert any(not r.converged for r in result.rounds)
    assert np.isfinite(result.final_relative_residual)
    # The result is still congruent with the input:
    _assert_congruent(result.phase, wrapped)


def test_irls_branch_field_is_the_phase():
    wrapped = _dipole()
    result = unwrap_phase_irls(wrapped)
    assert result.branches.dtype == np.int32
    np.testing.assert_allclose(
        result.phase, wrapped + 2 * np.pi * result.branches, rtol=0, atol=1e-12,
    )


@pytest.mark.parametrize("settles", [True, False])
def test_irls_warm_start_from_own_answer(settles):
    if settles:
        y, x = np.mgrid[:32, :40]
        wrapped = _wrap(0.5 * y + 0.3 * x + 3 * np.sin(0.2 * x))
    else:
        # A noise-free field with residues does not settle: the iterate keeps
        # sharpening around the cut, round after round, so only the branches are
        # compared.
        wrapped = _dipole()
    cold = unwrap_phase_irls(wrapped)
    priors = [
        L1BranchesPrior(branches=cold.branches, epsilon=cold.resume_epsilon),
        L1PhasePrior(phase=cold.phase, epsilon=cold.resume_epsilon),
        cold,
    ]
    for prior in priors:
        warm = unwrap_phase_irls(wrapped, prior=prior)
        np.testing.assert_array_equal(warm.branches, cold.branches)
        if settles:
            assert warm.status == "converged"
            assert len(warm.rounds) <= 2
            assert _cg_iterations(warm) <= 1


def test_irls_phase_prior_survives_the_input_crossing_pi():
    # The same field shifted by a constant: the wrapped values of many pixels
    # cross ±π, so their branches change, but the unwrapped field only shifts.
    y, x = np.mgrid[:32, :40]
    truth = 0.5 * y + 0.3 * x
    before = _wrap(truth)
    after = _wrap(truth + 2)
    cold_before = unwrap_phase_irls(before)
    cold_after = unwrap_phase_irls(after)
    warm = unwrap_phase_irls(after, prior=cold_before.as_prior())
    assert warm.status == "converged"
    np.testing.assert_array_equal(warm.branches, cold_after.branches)
    _assert_congruent(warm.phase, after)
    assert _error_against(warm.phase, truth) < 1e-6


def test_irls_wrong_prior_still_converges_to_the_cold_answer():
    wrapped = _dipole()
    cold = unwrap_phase_irls(wrapped)
    branches = cold.branches.copy()
    branches.ravel()[::97] += 1
    warm = unwrap_phase_irls(
        wrapped,
        prior=L1BranchesPrior(branches=branches, epsilon=0.0),
        options=IRLSOptions(iterations_per_round=200, rounds=30),
    )
    _assert_congruent(warm.phase, wrapped)
    np.testing.assert_array_equal(warm.branches, cold.branches)


@pytest.mark.parametrize(
    ("cold_start", "steps"),
    # From zero, the first weights are derived before the first round, so the
    # schedule is one step further after the same number of rounds.
    [("least_squares", 2), ("zero", 3)],
)
def test_irls_resume_epsilon_is_where_the_schedule_stopped(cold_start, steps):
    wrapped = _dipole()
    options = IRLSOptions(rounds=3, iterations_per_round=500, cold_start=cold_start)
    decay = options.decay

    converged = unwrap_phase_irls(wrapped, options=options)
    assert all(r.converged for r in converged.rounds)
    assert converged.selected_round == 2
    assert np.isclose(
        converged.resume_epsilon, options.epsilon_start * decay**steps, rtol=1e-12,
    )

    limited = unwrap_phase_irls(
        wrapped,
        options=IRLSOptions(rounds=3, iterations_per_round=1, cold_start=cold_start),
    )
    assert limited.status == "iteration_limit"
    # One round coarser, since the last round's iterate does not solve its system:
    assert np.isclose(
        limited.resume_epsilon, options.epsilon_start * decay ** (steps - 1), rtol=1e-12,
    )


def test_irls_single_precision_finds_the_same_branches():
    wrapped = _dipole()
    cold = unwrap_phase_irls(wrapped)
    single = unwrap_phase_irls(wrapped.astype(np.float32))
    np.testing.assert_array_equal(single.branches, cold.branches)
    assert single.weights_x.dtype == np.float32
    # The phase is float64, exactly congruent with the float32 input:
    assert single.phase.dtype == np.float64
    _assert_congruent(single.phase, wrapped.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rounds": 0},
        {"iterations_per_round": 0},
        {"cg_relative_tolerance": 0.0},
        {"cg_relative_tolerance": float("nan")},
        {"epsilon_min": 0.0},
        {"epsilon_min": float("nan")},
        {"epsilon_start": 1e-4, "epsilon_min": 1e-3},
        {"epsilon_start": float("nan")},
    ],
)
def test_irls_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        IRLSOptions(**kwargs)


@pytest.mark.parametrize(
    "prior",
    [
        L1BranchesPrior(branches=np.zeros((10, 10), dtype=np.int32), epsilon=0.0),
        L1PhasePrior(phase=np.zeros((10, 10)), epsilon=0.0),
    ],
)
def test_irls_prior_of_the_wrong_shape_is_rejected(prior):
    with pytest.raises(ValueError, match="shape"):
        unwrap_phase_irls(_dipole(), prior=prior)


@pytest.mark.parametrize(
    "wrapped",
    [np.zeros(10), np.zeros((0, 4)), np.zeros((4, 4, 4))],
)
def test_irls_invalid_input_is_rejected(wrapped):
    with pytest.raises(ValueError):
        unwrap_phase_irls(wrapped)


def test_phase_unwrap_irls_from_complex_wave():
    y, x = np.mgrid[:32, :32]
    truth = 0.4 * y + 0.6 * x
    unwrapped = phase_unwrap(np.exp(1j * truth), method="irls")
    assert _error_against(unwrapped, truth) < 1e-6


def _cut_turns(phase, wrapped):
    """Whole turns by which the gradient of `phase` differs from the wrapped one."""
    total = 0
    for axis in (0, 1):
        difference = np.diff(phase, axis=axis) - _wrap(np.diff(wrapped, axis=axis))
        total += int(np.abs(np.rint(difference / (2 * np.pi))).sum())
    return total


def _noisy_cliff(sigma, seed=1):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:96, :96] / 96
    truth = 12 * np.exp(-((y - .5)**2 + (x - .45)**2) / .04) + 6 * y
    truth += 2.5 * (x > 0.75 + 0.1 * (y - 0.5))
    return truth, _wrap(truth + rng.normal(scale=sigma, size=truth.shape))


def test_irls_cut_turns_are_reported():
    _, wrapped = _noisy_cliff(sigma=0.9)
    result = unwrap_phase_irls(wrapped)
    assert result.cut_turns == _cut_turns(result.phase, wrapped)
    assert all(r.cut_turns is not None for r in result.rounds)
    # The selected round is the latest one with the lowest cut cost (here equal
    # to the cut turns):
    cuts = [r.cut_turns for r in result.rounds]
    assert result.cut_turns <= min(cuts)
    if result.selected_round >= 0:
        assert cuts[result.selected_round] == result.cut_turns
        assert result.cut_turns not in cuts[result.selected_round + 1:]


def test_irls_rounds_improve_on_a_short_solve():
    _, wrapped = _noisy_cliff(sigma=0.9)
    # One round of one CG iteration barely moves from the start.
    start = unwrap_phase_irls(
        wrapped, options=IRLSOptions(rounds=1, iterations_per_round=1),
    )
    result = unwrap_phase_irls(wrapped)
    assert result.cut_turns < start.cut_turns


def test_irls_without_cuts_runs_no_rounds():
    y, x = np.mgrid[:24, :31]
    wrapped = _wrap(0.7 * y + 0.9 * x)
    result = unwrap_phase_irls(wrapped)
    assert result.cut_turns == 0
    assert result.rounds == []
    assert result.selected_round == -1
    assert result.status == "converged"
    np.testing.assert_array_equal(result.weights_x, 1)


def test_irls_stops_once_the_cut_count_settles():
    # The rounded L2 solution of the dipole already has the final cut turns, so no
    # round improves on it.
    wrapped = _dipole()
    early = unwrap_phase_irls(wrapped)
    assert len(early.rounds) == IRLSOptions().stop_patience
    full = unwrap_phase_irls(wrapped, options=IRLSOptions(stop_patience=None))
    assert len(full.rounds) == IRLSOptions().rounds
    np.testing.assert_array_equal(early.branches, full.branches)
    assert early.cut_turns == full.cut_turns


def test_irls_stop_patience_is_validated():
    with pytest.raises(ValueError, match="stop_patience"):
        IRLSOptions(stop_patience=0)


@pytest.mark.parametrize("seed", range(4))
def test_irls_zero_start_is_closer_to_the_truth(seed):
    truth, wrapped = _noisy_cliff(sigma=0.9, seed=seed)

    def wrong_pixels(cold_start):
        result = unwrap_phase_irls(wrapped, options=IRLSOptions(cold_start=cold_start))
        error = result.phase - truth
        return np.count_nonzero(np.rint((error - np.median(error)) / (2 * np.pi)))

    assert wrong_pixels("zero") < wrong_pixels("least_squares")


@pytest.mark.parametrize("tolerance", [-0.1, 1.0])
def test_irls_stop_tolerance_is_validated(tolerance):
    with pytest.raises(ValueError, match="stop_tolerance"):
        IRLSOptions(stop_tolerance=tolerance)


def test_irls_cold_start_is_validated():
    with pytest.raises(ValueError, match="cold_start"):
        IRLSOptions(cold_start="random")


def test_amplitude_edge_weights():
    amplitude = np.array([
        [1.0, 2.0, 4.0],
        [0.0, 4.0, 8.0],
    ])
    wx, wy = amplitude_edge_weights(amplitude, floor=0.01, percentile=50)
    assert wx.shape == (2, 2)
    assert wy.shape == (1, 3)
    # The reference is the median, 3.0; each edge takes the smaller of its two
    # pixels, clipped to [floor, 1]:
    np.testing.assert_allclose(wx, [[1 / 3, 2 / 3], [0.01, 1.0]])
    np.testing.assert_allclose(wy, [[0.01, 2 / 3, 1.0]])


def test_amplitude_edge_weights_without_signal():
    wx, wy = amplitude_edge_weights(np.zeros((3, 4)))
    np.testing.assert_array_equal(wx, np.ones((3, 3)))
    np.testing.assert_array_equal(wy, np.ones((2, 4)))


@pytest.mark.parametrize("floor", [0.0, -1.0, 1.5])
def test_amplitude_edge_weights_floor_is_validated(floor):
    with pytest.raises(ValueError, match="floor"):
        amplitude_edge_weights(np.ones((3, 3)), floor=floor)


def _blocked_wave(seed=1, size=128, blobs=40):
    """A noisy wave on a smooth field, with small blobs where the beam is blocked.

    Inside the blobs, the amplitude is low and the phase random. A blob away from
    the border is surrounded by low-noise phase, so its residues balance inside it.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size] / size
    truth = 12 * np.exp(-((y - .5)**2 + (x - .45)**2) / .04) + 6 * y
    blocked = np.zeros(truth.shape, dtype=bool)
    for cy, cx, r in zip(
        rng.uniform(0, 1, blobs), rng.uniform(0, 1, blobs), rng.uniform(0.015, 0.04, blobs),
        strict=True,
    ):
        blocked |= (y - cy)**2 + (x - cx)**2 < r**2
    amplitude = np.where(blocked, 0.02, 1.0)
    phase = truth + rng.normal(scale=0.3, size=truth.shape)
    phase[blocked] = rng.uniform(-np.pi, np.pi, size=blocked.sum())
    return amplitude * np.exp(1j * phase), truth, blocked


def test_irls_complex_input_uses_amplitude_weights():
    wave, _, _ = _blocked_wave()
    from_wave = unwrap_phase_irls(wave)
    explicit = unwrap_phase_irls(
        np.angle(wave), edge_weights=amplitude_edge_weights(np.abs(wave)),
    )
    np.testing.assert_array_equal(from_wave.branches, explicit.branches)
    assert from_wave.cut_cost < from_wave.cut_turns
    np.testing.assert_array_equal(phase_unwrap(wave, method="irls"), from_wave.phase)


def test_irls_amplitude_weights_keep_blocked_regions_from_spoiling_the_rest():
    wave, truth, blocked = _blocked_wave()
    result = unwrap_phase_irls(wave)
    error = (result.phase - truth)[~blocked]
    turns = np.rint((error - np.median(error)) / (2 * np.pi))
    assert np.count_nonzero(turns) == 0


def test_irls_uniform_edge_weights_change_nothing():
    wrapped = _dipole()
    rows, cols = wrapped.shape
    plain = unwrap_phase_irls(wrapped)
    # Only relative weights matter:
    uniform = unwrap_phase_irls(
        wrapped, edge_weights=(np.full((rows, cols - 1), 7.0), np.full((rows - 1, cols), 7.0)),
    )
    np.testing.assert_array_equal(plain.branches, uniform.branches)
    assert uniform.cut_cost == plain.cut_turns


@pytest.mark.parametrize(
    "edge_weights",
    [
        (np.ones((40, 48)), np.ones((39, 48))),
        (np.ones((40, 47)), np.ones((40, 48))),
        (np.zeros((40, 47)), np.ones((39, 48))),
        (np.full((40, 47), np.nan), np.ones((39, 48))),
    ],
)
def test_irls_invalid_edge_weights_are_rejected(edge_weights):
    with pytest.raises(ValueError, match="edge_weights"):
        unwrap_phase_irls(_dipole(), edge_weights=edge_weights)

"""Invalid pixels (NaN or infinite input) in the L1 phase unwrapping, on the CPU.

`test_unwrap_cupy.py` compares the GPU's results with the CPU's.
"""
import numpy as np
import pytest

from libertem_holo.base.unwrap import (
    L1BranchesPrior,
    amplitude_edge_weights,
    irls,
    unwrap_phase_admm,
    unwrap_phase_irls,
)

SOLVERS = [
    pytest.param(unwrap_phase_irls, id="irls"),
    pytest.param(unwrap_phase_admm, id="admm"),
]


def _ramp(shape: tuple[int, int] = (40, 52)) -> np.ndarray:
    y, x = np.mgrid[:shape[0], :shape[1]]
    return 0.45 * x + 0.35 * y


def _noisy_wave(shape: tuple[int, int] = (60, 70), seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]] / max(shape)
    truth = 20 * np.exp(-((y - .5)**2 + (x - .5)**2) / .05) + 8 * y
    amplitude = rng.uniform(0.3, 1.0, shape)
    noise = rng.normal(scale=0.5, size=shape)
    return amplitude * np.exp(1j * (truth + noise))


def _invalidate(field: np.ndarray, kind: str) -> np.ndarray:
    field = field.copy()
    if kind == "pixel":
        field[5, 7] = np.nan
    elif kind == "hole":
        field[10:18, 20:33] = np.nan
    elif kind == "border":
        field[:, :6] = np.nan
        field[-4:, :] = np.nan
    elif kind == "infinite":
        field[8, 9] = np.inf
        field[20, 30] = -np.inf
    else:
        field[...] = np.nan
    return field


KINDS = ["pixel", "hole", "border", "infinite", "all"]


def _check_masked(result: irls.L1Result, invalid: np.ndarray) -> None:
    assert np.isnan(result.phase[invalid]).all()
    assert np.isfinite(result.phase[~invalid]).all()
    assert (result.branches[invalid] == 0).all()
    assert np.isfinite(result.weights_x).all()
    assert np.isfinite(result.weights_y).all()


@pytest.mark.parametrize("solve", SOLVERS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_ramp_with_invalid_pixels_comes_back_exactly(solve, kind, dtype):
    """Edges that touch invalid pixels do not count, so the ramp has no cuts."""
    truth = _ramp()
    wrapped = _invalidate(np.angle(np.exp(1j * truth)), kind).astype(dtype)
    invalid = ~np.isfinite(wrapped)
    result = solve(wrapped)
    _check_masked(result, invalid)
    assert result.cut_turns == 0
    # The rounded L2 solution, which ignores the edge weights, can cut the data
    # near large invalid regions; one round removes those cuts:
    assert len(result.rounds) <= 1
    if not invalid.all():
        difference = result.phase[~invalid] - truth[~invalid]
        np.testing.assert_allclose(difference, difference[0], atol=1e-5)


@pytest.mark.parametrize("solve", SOLVERS)
@pytest.mark.parametrize("kind", KINDS)
def test_invalid_pixels_of_a_wave(solve, kind):
    wave = _invalidate(_noisy_wave(), kind).astype(np.complex64)
    invalid = ~np.isfinite(wave)
    original = wave.copy()
    result = solve(wave)
    np.testing.assert_array_equal(wave, original)  # the caller's input is unchanged
    _check_masked(result, invalid)
    # The same as the phase with edge weights from the amplitude:
    phase = np.where(invalid, np.nan, np.angle(wave))
    from_phase = solve(phase, edge_weights=amplitude_edge_weights(np.abs(wave)))
    np.testing.assert_array_equal(result.branches, from_phase.branches)
    assert result.cut_cost == from_phase.cut_cost


@pytest.mark.parametrize("solve", SOLVERS)
def test_invalid_pixels_with_edge_weights(solve):
    wave = _invalidate(_noisy_wave(), "hole")
    phase = np.where(np.isfinite(wave), np.angle(wave), np.nan)
    rng = np.random.default_rng(1)
    edges = (rng.uniform(0.2, 1, (60, 69)), rng.uniform(0.2, 1, (59, 70)))
    result = solve(phase, edge_weights=edges)
    _check_masked(result, ~np.isfinite(phase))


@pytest.mark.parametrize("solve", SOLVERS)
def test_a_result_with_invalid_pixels_is_a_prior(solve):
    wave = _invalidate(_noisy_wave(), "border")
    invalid = ~np.isfinite(wave)
    cold = solve(wave)
    for prior in (cold, L1BranchesPrior(cold.branches, epsilon=0.05)):
        warm = solve(wave, prior=prior)
        _check_masked(warm, invalid)
        assert warm.cut_cost <= cold.cut_cost * 1.05


def test_edges_that_touch_invalid_pixels_are_weak_and_do_not_count():
    phase = _invalidate(np.angle(_noisy_wave()), "pixel")
    wrapped, edges, invalid = irls._check_input(phase, None, None)  # noqa: SLF001
    assert edges is None
    assert wrapped[5, 7] == 0
    assert np.isnan(phase[5, 7])  # the caller's array is unchanged
    problem = irls._WeightedProblem(wrapped, 1, edges, invalid=invalid)  # noqa: SLF001
    weak_x = problem.edge_x[5, 6:8]
    weak_y = problem.edge_y[4:6, 7]
    np.testing.assert_array_equal(weak_x, irls._INVALID_EDGE_WEIGHT)  # noqa: SLF001
    np.testing.assert_array_equal(weak_y, irls._INVALID_EDGE_WEIGHT)  # noqa: SLF001
    assert (problem.edge_x == 1).sum() == problem.edge_x.size - 2
    # Changing the branch of the invalid pixel adds no cut turns:
    branches = np.zeros(phase.shape, dtype=np.int32)
    turns, _ = problem.cuts(branches)
    branches[5, 7] = 3
    assert problem.cuts(branches)[0] == turns


def test_amplitude_weights_of_invalid_pixels():
    """Invalid pixels count as without signal, and are left out of the percentile."""
    rng = np.random.default_rng(3)
    amplitude = rng.uniform(0.5, 1.5, (20, 30))
    reference = np.percentile(amplitude, 90)
    with_invalid = amplitude.copy()
    with_invalid[4, 5] = np.nan
    with_invalid[10, 11] = np.inf
    valid = np.isfinite(with_invalid)
    weights_x, weights_y = amplitude_edge_weights(with_invalid, floor=0.01)
    assert np.isfinite(weights_x).all()
    assert np.isfinite(weights_y).all()
    assert weights_x[4, 4] == weights_x[4, 5] == 0.01
    assert weights_y[3, 5] == weights_y[4, 5] == 0.01
    relative = np.clip(amplitude / np.percentile(amplitude[valid], 90), 0.01, 1)
    np.testing.assert_array_equal(
        weights_x[0], np.minimum(relative[0, 1:], relative[0, :-1]),
    )
    assert np.percentile(amplitude[valid], 90) != reference
    no_signal = amplitude_edge_weights(np.full((3, 4), np.nan))
    assert all(np.array_equal(w, np.ones_like(w)) for w in no_signal)

"""ADMM phase unwrapping on the CPU, and the numba helpers it shares with IRLS."""
import numpy as np
import pytest

from libertem_holo.base.unwrap import (
    ADMMOptions,
    _admm_numba,
    _irls_numba,
    amplitude_edge_weights,
    irls,
    unwrap_phase_admm,
    unwrap_phase_irls,
)


def _wrap(value: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * value))


def _vortex(shape: tuple[int, int], center: tuple | None = None) -> np.ndarray:
    rows, cols = shape
    y, x = np.mgrid[:rows, :cols].astype(np.float64)
    cy, cx = center if center is not None else (rows / 2 - 0.5, cols / 2 - 0.5)
    return np.arctan2(y - cy, x - cx)


def _dipole() -> np.ndarray:
    return _wrap(_vortex((40, 48)) - _vortex((40, 48), center=(12, 30)))


def _noisy(shape: tuple[int, int] = (70, 90), seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]] / max(shape)
    truth = 25 * np.exp(-((y - .4)**2 + (x - .5)**2) / .05) + 12 * y
    truth += 3.0 * (x > 0.7)
    return _wrap(truth + rng.normal(scale=0.6, size=shape))


def _wave(shape: tuple[int, int] = (64, 80), seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]]
    amplitude = np.where((y - 30)**2 + (x - 40)**2 < 12**2, 0.05, 1.0)
    amplitude = amplitude * rng.uniform(0.8, 1.2, size=shape)
    phase = 0.25 * x + 0.15 * y + rng.normal(scale=0.3, size=shape)
    phase += np.where(amplitude < 0.5, rng.uniform(-np.pi, np.pi, size=shape), 0)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def _assert_congruent(unwrapped: np.ndarray, wrapped: np.ndarray) -> None:
    turns = (unwrapped - wrapped) / (2 * np.pi)
    np.testing.assert_allclose(turns, np.round(turns), rtol=0, atol=1e-9)


def test_admm_linear_ramp_comes_back_exactly():
    y, x = np.mgrid[:24, :31]
    truth = 0.7 * y + 0.9 * x
    wrapped = _wrap(truth)
    result = unwrap_phase_admm(wrapped)
    _assert_congruent(result.phase, wrapped)
    assert result.status == "converged"
    assert result.cut_turns == 0
    offset = (result.phase - truth).mean()
    assert np.abs(result.phase - truth - offset).max() < 1e-6


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "field", [pytest.param(_dipole, id="dipole"), pytest.param(_noisy, id="noisy")],
)
def test_admm_cuts_no_more_than_irls(field, dtype):
    wrapped = field().astype(dtype)
    admm = unwrap_phase_admm(wrapped)
    reference = unwrap_phase_irls(wrapped)
    _assert_congruent(admm.phase, wrapped)
    assert admm.cut_cost <= 1.02 * reference.cut_cost
    assert admm.phase.dtype == np.float64
    assert admm.branches.dtype == np.int32
    assert all(r.iterations == ADMMOptions().iterations_per_round for r in admm.rounds)


def test_admm_on_a_complex_wave():
    wave = _wave()
    result = unwrap_phase_admm(wave)
    _assert_congruent(result.phase, np.angle(wave))
    # Edge weights from the amplitude make cuts through the dark disk cheap:
    plain = unwrap_phase_admm(np.angle(wave))
    assert result.cut_cost <= plain.cut_cost
    assert result.weights_x.shape == (64, 79)
    assert result.weights_y.shape == (63, 80)


def test_admm_warm_start_is_never_worse_than_its_prior():
    wrapped = _noisy()
    cold = unwrap_phase_admm(wrapped)
    warm = unwrap_phase_admm(wrapped, prior=cold)
    assert warm.cut_cost <= cold.cut_cost
    shifted = _wrap(wrapped + 0.4)
    warm = unwrap_phase_admm(shifted, prior=cold)
    _assert_congruent(warm.phase, shifted)


def test_admm_result_does_not_depend_on_the_thread_count():
    wrapped = _noisy(shape=(130, 150))
    one = unwrap_phase_admm(wrapped, workers=1)
    four = unwrap_phase_admm(wrapped, workers=4)
    np.testing.assert_array_equal(one.branches, four.branches)
    np.testing.assert_array_equal(one.phase, four.phase)
    assert one.cut_cost == four.cut_cost


@pytest.mark.parametrize(
    "kwargs",
    [{"iterations_per_round": 3}, {"iterations_per_round": 0}, {"rho_scale": 0},
     {"rounds": 0}, {"epsilon_min": 0}, {"stop_patience": 0},
     {"epsilon_start": 1e-4}, {"cold_start": "magic"}],
)
def test_admm_options_are_checked(kwargs):
    with pytest.raises(ValueError):  # noqa: PT011 (each case has its message)
        ADMMOptions(**kwargs)


# ---- The numba helpers match numpy, or each other, bit for bit -------------------


def test_wave_setup_is_numpys():
    wave = _wave()
    wave[3, 5] = 0
    for dtype in (np.complex64, np.complex128):
        w = wave.astype(dtype)
        phase, (edge_x, edge_y), invalid = irls._wave_setup(w, 4)  # noqa: SLF001
        assert invalid is None
        np.testing.assert_array_equal(phase, np.angle(w))
        expected = irls._check_edge_weights(  # noqa: SLF001
            amplitude_edge_weights(np.abs(w)), w.shape,
        )
        np.testing.assert_array_equal(edge_x, expected[0])
        np.testing.assert_array_equal(edge_y, expected[1])


@pytest.mark.parametrize("seed", range(6))
def test_percentile_is_numpys(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 200_000))
    x = [rng.random(n), rng.integers(0, 4, n).astype(float),
         np.round(rng.normal(size=n), 2)][seed % 3]
    for q in (0, 10, 50, 90, 99.9, 100):
        assert irls._percentile(x, q) == np.percentile(x, q)  # noqa: SLF001
    x[rng.integers(0, n)] = np.nan
    assert np.isnan(irls._percentile(x, 90))  # noqa: SLF001


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(33, 47), (1, 9), (9, 1)])
def test_targets_and_jumps_are_numpys(dtype, shape):
    wrapped = (np.random.default_rng(1).uniform(-4, 4, size=shape)).astype(dtype)
    problem = irls._Edges(wrapped, 2)  # noqa: SLF001
    for axis, target, jumps in ((1, problem.target_x, problem.jumps_x),
                                (0, problem.target_y, problem.jumps_y)):
        difference = np.diff(wrapped, axis=axis)
        expected = irls._wrap(difference)  # noqa: SLF001
        np.testing.assert_array_equal(target, expected)
        np.testing.assert_array_equal(
            jumps, np.rint((difference - expected) / irls.TAU).astype(np.int32),
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(64, 80), (1, 9), (9, 1), (2, 2)])
def test_admm_update_by_rows_is_the_plain_one(dtype, shape):
    rng = np.random.default_rng(2)
    rows, cols = shape
    u = rng.normal(size=shape).astype(dtype)
    gx = rng.normal(size=(rows, cols - 1)).astype(dtype)
    gy = rng.normal(size=(rows - 1, cols)).astype(dtype)
    cx = rng.uniform(0.01, 1, gx.shape).astype(dtype)
    cy = rng.uniform(0.01, 1, gy.shape).astype(dtype)
    lx = rng.normal(size=gx.shape).astype(dtype)
    ly = rng.normal(size=gy.shape).astype(dtype)
    outputs = []
    for update, extra in ((_admm_numba.admm_update, ()),
                          (_admm_numba.admm_update_rows, (5,))):
        out = (np.zeros_like(gx), np.zeros_like(gy), np.empty_like(u))
        update(u, gx, gy, cx, cy, lx, ly, out[0], out[1], 0.7, 0.05, out[2], *extra)
        outputs.append(out)
    for plain, by_rows in zip(*outputs, strict=True):
        np.testing.assert_array_equal(plain, by_rows)


def test_select_pair_handles_equal_and_infinite_values():
    x = np.full(100_000, 0.5)
    assert _irls_numba.select_pair(x, 10, 11) == (0.5, 0.5, False)
    x[7] = np.inf
    x[8] = -np.inf
    a, b, nan = _irls_numba.select_pair(x, 0, 99_999)
    assert (a, b, nan) == (-np.inf, np.inf, False)


def test_phase_unwrap_forwards_keyword_arguments():
    from libertem_holo.base.unwrap import phase_unwrap  # noqa: PLC0415
    wrapped = _noisy()
    options = ADMMOptions(rounds=1, stop_patience=None)
    np.testing.assert_array_equal(
        phase_unwrap(wrapped, method="admm", options=options),
        unwrap_phase_admm(wrapped, options=options).phase,
    )
    with pytest.raises(ValueError, match="unknown phase unwrapping method"):
        phase_unwrap(wrapped, method="magic")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("solve", "options"),
    [(unwrap_phase_irls, ADMMOptions()), (unwrap_phase_admm, irls.IRLSOptions()),
     (unwrap_phase_admm, {"rounds": 3})],
)
def test_each_solver_takes_only_its_own_options(solve, options):
    with pytest.raises(TypeError, match="options must be"):
        solve(_noisy(), options=options)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"workers": 0}, ValueError, "workers"),
        ({"workers": -1}, ValueError, "workers"),
        ({"workers": True}, ValueError, "workers"),
        ({"workers": 1.5}, ValueError, "workers"),
        ({"prior": np.zeros((70, 90))}, TypeError, "prior must be"),
        ({"prior": irls.L1PhasePrior(np.zeros((70, 90)), np.nan)}, ValueError,
         "epsilon"),
        ({"prior": irls.L1BranchesPrior(np.zeros((70, 90), np.int32), -1.0)},
         ValueError, "epsilon"),
        ({"edge_weights": np.ones((2, 70, 89))}, TypeError, "pair of arrays"),
    ],
)
@pytest.mark.parametrize("solve", [unwrap_phase_irls, unwrap_phase_admm])
def test_arguments_are_checked(solve, kwargs, error, match):
    with pytest.raises(error, match=match):
        solve(_noisy(), **kwargs)


def test_boolean_input_is_rejected():
    with pytest.raises(TypeError, match="real or complex"):
        unwrap_phase_admm(np.zeros((4, 5), dtype=bool))

"""Tests of IRLS and ADMM phase unwrapping on the GPU, against the CPU solvers.

The GPU IRLS solver does the same arithmetic per pixel, in the same order, and sums
dot products in float64 in another order. With the same preconditioner padding, it
finds the same branches, rounds and cut turns as the CPU solver; these tests check
that on fields that exercise each part of the solve. By default, the GPU pads the
preconditioner's grid to other lengths than the CPU (faster with cuFFT), which
changes the CG iterates slightly, so these tests use the CPU's padding, except
where they request the `gpu_defaults` fixture. They skip without a GPU.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.unwrap import (
    ADMMOptions,
    IRLSOptions,
    L1BranchesPrior,
    L1PhasePrior,
    L1Result,
    _irls_cupy,
    amplitude_edge_weights,
    irls,
    phase_unwrap,
    unwrap_phase_admm,
    unwrap_phase_irls,
)
from libertem_holo.base.workspace import Workspace, clear_default_workspace

if TYPE_CHECKING:
    import cupy as cp

_devices = detect()
pytestmark = pytest.mark.skipif(
    not _devices["cudas"] or not _devices["has_cupy"],
    reason="No CUDA device or no CuPy",
)


@pytest.fixture
def gpu_defaults() -> None:
    """Solve with the GPU's own padding (see the module docstring)."""


@pytest.fixture(autouse=True)
def _cpu_padding(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "gpu_defaults" not in request.fixturenames:
        monkeypatch.setattr(_irls_cupy, "PADDED_LENGTH", irls._padded_length)  # noqa: SLF001


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
    """Return a smooth field with a step and noise: many residues."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]] / max(shape)
    truth = 25 * np.exp(-((y - .4)**2 + (x - .5)**2) / .05) + 12 * y
    truth += 3.0 * (x > 0.7)
    return _wrap(truth + rng.normal(scale=0.6, size=shape))


def _wave(shape: tuple[int, int] = (64, 80), seed: int = 5) -> np.ndarray:
    """Return a complex wave with a dark disk, for edge weights from its amplitude."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]]
    amplitude = np.where((y - 30)**2 + (x - 40)**2 < 12**2, 0.05, 1.0)
    amplitude = amplitude * rng.uniform(0.8, 1.2, size=shape)
    phase = 0.25 * x + 0.15 * y + rng.normal(scale=0.3, size=shape)
    phase += np.where(amplitude < 0.5, rng.uniform(-np.pi, np.pi, size=shape), 0)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def _assert_same(gpu: L1Result, cpu: L1Result, residual_rtol: float = 1e-3) -> None:
    """Assert the same result, up to the summation order of the dot products.

    Where the system is badly conditioned, the CG residuals depend more on that
    order; `residual_rtol` allows for it.
    """
    import cupy as cp  # noqa: PLC0415
    for name in ("phase", "branches", "weights_x", "weights_y"):
        assert isinstance(getattr(gpu, name), cp.ndarray), name
    np.testing.assert_array_equal(gpu.branches.get(), cpu.branches)
    np.testing.assert_array_equal(gpu.phase.get(), cpu.phase)
    assert gpu.phase.dtype == cpu.phase.dtype
    assert gpu.weights_x.dtype == cpu.weights_x.dtype
    tol = 1e-4 if cpu.weights_x.dtype == np.float32 else 1e-9
    np.testing.assert_allclose(gpu.weights_x.get(), cpu.weights_x, rtol=tol, atol=tol)
    np.testing.assert_allclose(gpu.weights_y.get(), cpu.weights_y, rtol=tol, atol=tol)
    assert [r.iterations for r in gpu.rounds] == [r.iterations for r in cpu.rounds]
    assert [r.converged for r in gpu.rounds] == [r.converged for r in cpu.rounds]
    assert [r.cut_turns for r in gpu.rounds] == [r.cut_turns for r in cpu.rounds]
    for g, c in zip(gpu.rounds, cpu.rounds, strict=True):
        np.testing.assert_allclose(
            g.residuals, c.residuals, rtol=residual_rtol, atol=1e-12,
        )
    assert gpu.selected_round == cpu.selected_round
    assert gpu.cut_turns == cpu.cut_turns
    assert gpu.cut_cost == pytest.approx(cpu.cut_cost, rel=1e-9)
    assert gpu.status == cpu.status
    assert gpu.resume_epsilon == pytest.approx(cpu.resume_epsilon, rel=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "field",
    [
        pytest.param(_dipole, id="dipole"),
        pytest.param(_noisy, id="noisy"),
        pytest.param(lambda: _vortex((33, 47)), id="vortex"),
        pytest.param(lambda: np.full((16, 20), 0.4), id="constant"),
        pytest.param(
            lambda: _wrap(np.add.outer(0.7 * np.arange(24), 0.9 * np.arange(31))),
            id="ramp",
        ),
    ],
)
def test_irls_gpu_matches_cpu(field, dtype):
    import cupy as cp  # noqa: PLC0415
    wrapped = field().astype(dtype)
    _assert_same(unwrap_phase_irls(cp.asarray(wrapped)), unwrap_phase_irls(wrapped))


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.usefixtures("gpu_defaults")
def test_irls_gpu_padding_finds_an_equally_good_result(seed):
    """With its own padding, the GPU's result can differ, but not by much."""
    import cupy as cp  # noqa: PLC0415
    wrapped = _noisy(shape=(99, 75), seed=seed)  # padded differently on the GPU
    for n in wrapped.shape:
        assert _irls_cupy._gpu_padded_length(n) != irls._padded_length(n)  # noqa: SLF001
    cpu = unwrap_phase_irls(wrapped)
    gpu = unwrap_phase_irls(cp.asarray(wrapped))
    turns = (gpu.phase.get() - wrapped) / (2 * np.pi)
    np.testing.assert_allclose(turns, np.round(turns), rtol=0, atol=1e-9)
    assert gpu.cut_cost <= 1.02 * cpu.cut_cost
    offset = np.median(gpu.branches.get() - cpu.branches)
    differ = np.count_nonzero(gpu.branches.get() - cpu.branches - offset)
    assert differ < 0.02 * wrapped.size


@pytest.mark.parametrize("signal", [True, False])
def test_amplitude_edge_weights_on_the_gpu(signal):
    import cupy as cp  # noqa: PLC0415
    amplitude = np.abs(_wave()).astype(np.float64) * signal
    cpu = amplitude_edge_weights(amplitude)
    gpu = amplitude_edge_weights(cp.asarray(amplitude))
    for g, c in zip(gpu, cpu, strict=True):
        assert isinstance(g, cp.ndarray)
        np.testing.assert_allclose(g.get(), c, rtol=1e-12)


def test_irls_gpu_matches_cpu_on_a_complex_wave():
    import cupy as cp  # noqa: PLC0415
    wave = _wave()
    # cupy's and numpy's `angle` may differ in the last bit, so both solve the
    # same phase, with the edge weights from the same amplitude:
    phase = np.angle(wave)
    cpu = unwrap_phase_irls(phase, edge_weights=amplitude_edge_weights(np.abs(wave)))
    weights = amplitude_edge_weights(cp.abs(cp.asarray(wave)))
    gpu = unwrap_phase_irls(cp.asarray(phase), edge_weights=weights)
    _assert_same(gpu, cpu)
    # With the wave itself, the GPU derives the same edge weights:
    from_wave = unwrap_phase_irls(cp.asarray(wave))
    from_phase = unwrap_phase_irls(cp.angle(cp.asarray(wave)), edge_weights=weights)
    np.testing.assert_array_equal(from_wave.branches.get(), from_phase.branches.get())
    assert from_wave.cut_cost == from_phase.cut_cost


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 7), (7, 1), (2, 2), (13, 29), (29, 13)],
)
def test_irls_gpu_odd_and_rectangular_shapes(shape):
    import cupy as cp  # noqa: PLC0415
    y, x = np.mgrid[:shape[0], :shape[1]]
    wrapped = _wrap(0.35 * y + 0.55 * x + 2 * np.sin(0.4 * x * y))
    _assert_same(unwrap_phase_irls(cp.asarray(wrapped)), unwrap_phase_irls(wrapped))


@pytest.mark.parametrize(
    "options",
    [
        # Several CUDA graphs per round, and the iteration limit:
        IRLSOptions(iterations_per_round=200, rounds=5),
        IRLSOptions(iterations_per_round=1, cg_relative_tolerance=1e-14),
        IRLSOptions(iterations_per_round=40, cg_relative_tolerance=1e-3),
        IRLSOptions(cold_start="least_squares", stop_patience=None, rounds=8),
    ],
)
def test_irls_gpu_matches_cpu_with_options(options):
    import cupy as cp  # noqa: PLC0415
    wrapped = _noisy()
    _assert_same(
        unwrap_phase_irls(cp.asarray(wrapped), options=options),
        unwrap_phase_irls(wrapped, options=options),
    )


def test_irls_gpu_warm_starts_match_cpu():
    import cupy as cp  # noqa: PLC0415
    wrapped = _dipole()
    cold = unwrap_phase_irls(wrapped)
    cold_gpu = unwrap_phase_irls(cp.asarray(wrapped))
    priors = [
        (L1BranchesPrior(branches=cold.branches, epsilon=cold.resume_epsilon),
         L1BranchesPrior(branches=cold_gpu.branches, epsilon=cold_gpu.resume_epsilon)),
        (L1PhasePrior(phase=cold.phase, epsilon=cold.resume_epsilon),
         cold_gpu.as_prior()),
        (cold, cold_gpu),
    ]
    shifted = _wrap(wrapped + 2)
    for cpu_prior, gpu_prior in priors:
        _assert_same(
            unwrap_phase_irls(cp.asarray(wrapped), prior=gpu_prior),
            unwrap_phase_irls(wrapped, prior=cpu_prior),
        )
    # A phase prior from another input:
    _assert_same(
        unwrap_phase_irls(cp.asarray(shifted), prior=cold_gpu),
        unwrap_phase_irls(shifted, prior=cold),
    )


def test_irls_gpu_reuses_its_workspace():
    """Repeated solves of same-sized inputs, which reuse one problem, match the CPU."""
    import cupy as cp  # noqa: PLC0415
    fields = [_noisy(seed=s) for s in range(3)]
    singles = [unwrap_phase_irls(f) for f in fields]
    for _ in range(2):
        for field, single in zip(fields, singles, strict=True):
            _assert_same(unwrap_phase_irls(cp.asarray(field)), single)
    # Results do not share memory with the problem, which later solves reuse:
    results = [unwrap_phase_irls(cp.asarray(f)) for f in fields]
    for result, single in zip(results, singles, strict=True):
        _assert_same(result, single)


_LATE_COPY = r"""
extern "C" __global__ void late_copy(const double* src, double* dst, long long n,
                                     long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {}
    for (long long k = blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (long long)gridDim.x * blockDim.x) {
        dst[k] = src[k];
    }
}
"""


def _late_copy(src: cp.ndarray) -> cp.ndarray:
    """Copy `src` on the current stream, after spinning ~20 ms on the GPU."""
    import cupy as cp  # noqa: PLC0415
    kernel = cp.RawKernel(_LATE_COPY, "late_copy")
    dst = cp.empty_like(src)
    rate = cp.cuda.Device().attributes["ClockRate"]  # kHz
    kernel((64,), (256,), (src, dst, np.int64(src.size), np.int64(rate * 20)))
    return dst


def test_irls_gpu_orders_its_stream_with_the_callers():
    """Inputs the caller frees at once, and results it reads on its own stream.

    The problem works on a stream of its own and returns without synchronizing.
    Memory the caller frees goes back to the pool of the caller's stream and must
    not be reused there before the problem's stream has read it; results must be
    complete before the caller's stream reads them.
    """
    import cupy as cp  # noqa: PLC0415

    from libertem_holo.base.unwrap import irls  # noqa: PLC0415
    from libertem_holo.base.unwrap._irls_cupy import problem_for  # noqa: PLC0415
    rng = np.random.default_rng(7)
    shape = (300, 280)
    wrapped = _wrap(rng.normal(scale=2, size=shape))
    cpu = irls._WeightedProblem(wrapped, workers=1)  # noqa: SLF001
    gpu = problem_for(cp.asarray(wrapped), None)
    for epsilon, late in ((0.3, False), (0.1, True), (0.03, False), (0.01, True)):
        phi = rng.normal(size=shape)
        cpu.reweight(phi, epsilon)
        # An input that is freed at once, and maybe still being computed on the
        # caller's stream:
        pending = cp.asarray(phi)
        if late:
            pending = _late_copy(pending)
        gpu.reweight(pending, epsilon)
        del pending
        junk = [cp.full(shape, np.nan) for _ in range(8)]
        weights = gpu.weights_x + 0  # on the caller's stream
        del junk
        np.testing.assert_array_equal(weights.get(), cpu.weights_x)


def test_irls_gpu_rejects_invalid_edge_weights():
    import cupy as cp  # noqa: PLC0415
    wrapped = cp.asarray(_dipole())
    weights = (cp.ones((40, 47)), cp.ones((39, 48)))
    weights[0][3, 4] = cp.nan
    with pytest.raises(ValueError, match="finite"):
        unwrap_phase_irls(wrapped, edge_weights=weights)


def test_phase_unwrap_irls_takes_cupy_input():
    import cupy as cp  # noqa: PLC0415
    wave = _wave()
    phase = phase_unwrap(cp.asarray(wave), method="irls")
    assert isinstance(phase, cp.ndarray)
    assert phase.shape == wave.shape


def _selection_inputs(dtype: type) -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    return [
        rng.normal(size=1001).astype(dtype),
        rng.normal(size=1000).astype(dtype),
        np.round(rng.normal(size=500) * 3).astype(dtype),  # many duplicates
        np.array([1, 1, 2, 2], dtype),
        np.full(77, 0.25, dtype),
        np.array([-3.5], dtype),
        np.array([4.0, -1.0], dtype),
        -np.abs(rng.normal(size=99)).astype(dtype),
        rng.uniform(-np.pi, np.pi, size=300_000).astype(dtype),  # several blocks
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gpu_selection_is_numpys_median_and_percentile(dtype):
    """The radix selection gives exactly np.median and np.percentile (linear)."""
    import cupy as cp  # noqa: PLC0415
    for values in _selection_inputs(dtype):
        n = values.size
        problem = _irls_cupy.CupyWeightedProblem((1, n), dtype)
        x = cp.asarray(values)
        with problem._on_stream():  # noqa: SLF001
            problem._select(x, ((n - 1) // 2, n // 2), mode=0, out_t=problem.offset)  # noqa: SLF001
            median = problem.offset.get()[0]
            virtual = (n - 1) * 0.9
            low = int(np.floor(virtual))
            problem._select(  # noqa: SLF001
                x, (low, min(low + 1, n - 1)), mode=1, t=virtual - low,
            )
            percentile = problem.select_out.get()[0]
        assert median == np.median(values)
        assert median.dtype == np.median(values).dtype
        assert percentile == np.percentile(values.astype(np.float64), 90)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "amplitude",
    [
        pytest.param(lambda shape, rng: rng.uniform(0.01, 2, size=shape), id="random"),
        pytest.param(lambda shape, _: np.zeros(shape), id="no-signal"),
        pytest.param(
            lambda shape, rng: np.where(
                rng.random(shape) < 0.2, np.nan, rng.uniform(0.01, 2, size=shape),
            ),
            id="invalid-pixels",
        ),
        pytest.param(lambda shape, _: np.full(shape, np.nan), id="all-invalid"),
    ],
)
@pytest.mark.parametrize("shape", [(61, 47), (1, 9), (9, 1), (1, 1)])
def test_gpu_wave_setup_is_the_generic_one(dtype, amplitude, shape):
    """`fill_from_wave` equals `angle`, `amplitude_edge_weights` and `fill`.

    Also with invalid pixels (NaN in the wave, and so in its angle).
    """
    import cupy as cp  # noqa: PLC0415
    rng = np.random.default_rng(5)
    wave = amplitude(shape, rng) * np.exp(1j * rng.uniform(-4, 4, size=shape))
    wave = cp.asarray(wave.astype(dtype))
    real = np.float32 if dtype == np.complex64 else np.float64
    fused = _irls_cupy.CupyWeightedProblem(shape, real)
    fused.fill_from_wave(wave, irls._AMPLITUDE_FLOOR, irls._AMPLITUDE_PERCENTILE)  # noqa: SLF001
    generic = _irls_cupy.CupyWeightedProblem(shape, real)
    weights = irls._check_edge_weights(  # noqa: SLF001
        amplitude_edge_weights(cp.abs(wave)), shape,
    )
    generic.fill(cp.angle(wave), weights)
    for name in ("wrapped", "invalid", "target_x", "target_y", "jumps_x", "jumps_y",
                 "edge_x", "edge_y", "weights_x", "weights_y"):
        np.testing.assert_array_equal(
            getattr(fused, name).get(), getattr(generic, name).get(), err_msg=name,
        )


def _start_on(
    problem: _irls_cupy.CupyWeightedProblem,
    start: str,
    options: IRLSOptions,
    prior: L1PhasePrior | L1BranchesPrior | None,
) -> dict:
    """Run a cold or warm start on `problem`; return its results, on the host."""
    if start == "cold":
        phi, best, epsilon = problem.cold_start(problem.wrapped, options)
    else:
        phi, best, epsilon = problem.warm_start(problem.wrapped, prior, options)
    return {
        "phi": phi.get(), "epsilon": epsilon,
        "weights_x": problem.weights_x.get(), "weights_y": problem.weights_y.get(),
        **{f"best.{name}": getattr(best, name) for name in (
            "cut_turns", "cut_cost", "weights_epsilon", "converged", "round_index")},
        **{f"best.{name}": getattr(best, name).get() for name in (
            "phase", "branches", "weights_x", "weights_y")},
    }


@pytest.mark.parametrize("source", ["wave", "phase", "phase with edge weights"])
@pytest.mark.parametrize(
    "start", ["zero", "least_squares", "phase prior", "branches prior"],
)
def test_gpu_start_graphs_are_the_generic_starts(source, start):
    """A staged start (one CUDA graph) equals filling and starting step by step."""
    import cupy as cp  # noqa: PLC0415
    rng = np.random.default_rng(7)
    shape = (37, 29)
    wave = rng.uniform(0.05, 2, size=shape) * np.exp(1j * (
        _vortex(shape) + 0.3 * rng.standard_normal(shape)))
    wave = cp.asarray(wave.astype(np.complex64))
    wrapped = cp.angle(wave)
    edges = None
    if source == "phase with edge weights":
        edges = irls._check_edge_weights(  # noqa: SLF001
            (cp.asarray(rng.uniform(0.2, 1, (shape[0], shape[1] - 1))),
             cp.asarray(rng.uniform(0.2, 1, (shape[0] - 1, shape[1])))), shape,
        )
    options = IRLSOptions(
        cold_start="least_squares" if start == "least_squares" else "zero",
    )
    reference = unwrap_phase_irls(np.asarray(wrapped.get(), dtype=np.float64))
    prior = {
        "phase prior": L1PhasePrior(phase=reference.phase + 0.4, epsilon=0.05),
        "branches prior": L1BranchesPrior(branches=reference.branches, epsilon=0.3),
    }.get(start)
    kind = "cold" if prior is None else "warm"
    floor, percentile = irls._AMPLITUDE_FLOOR, irls._AMPLITUDE_PERCENTILE  # noqa: SLF001

    generic = _irls_cupy.CupyWeightedProblem(shape, np.float32)
    staged = _irls_cupy.CupyWeightedProblem(shape, np.float32)
    if source == "wave":
        generic.fill_from_wave(wave, floor, percentile)
        staged.stage_wave(wave, floor, percentile)
    else:
        generic.fill(wrapped, edges)
        staged.stage_phase(wrapped, edges)
    expected = _start_on(generic, kind, options, prior)
    actual = _start_on(staged, kind, options, prior)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value, err_msg=name)
    # The graph is cached per kind of start; a second start reuses it:
    if source == "wave":
        staged.stage_wave(wave, floor, percentile)
    else:
        staged.stage_phase(wrapped, edges)
    again = _start_on(staged, kind, options, prior)
    for name, value in expected.items():
        np.testing.assert_array_equal(again[name], value, err_msg=name)
    assert len(staged._start_graphs) == 1  # noqa: SLF001


def test_gpu_start_graphs_follow_the_options():
    """A cached start graph is not reused for another epsilon_start."""
    import cupy as cp  # noqa: PLC0415
    wrapped = cp.asarray(_vortex((33, 47)).astype(np.float32))
    staged = _irls_cupy.CupyWeightedProblem(wrapped.shape, np.float32)
    for epsilon_start in (1.0, 2.5, 1.0):
        options = IRLSOptions(epsilon_start=epsilon_start)
        generic = _irls_cupy.CupyWeightedProblem(wrapped.shape, np.float32)
        generic.fill(wrapped, None)
        expected = _start_on(generic, "cold", options, None)
        staged.stage_phase(wrapped, None)
        actual = _start_on(staged, "cold", options, None)
        assert expected["best.cut_turns"] > 0  # so the start reweights at epsilon_start
        assert actual["epsilon"] == epsilon_start
        for name, value in expected.items():
            np.testing.assert_array_equal(actual[name], value, err_msg=name)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_gpu_kernels_compile_only_their_declared_signatures(dtype):
    """Every path launches each kernel with its declared argument types.

    So the compile-only tests in `test_cuda_kernels.py`, which compile
    `_irls_numba_cuda.signatures`, cover what runs, and numba compiles no second
    variant for an argument of another type.
    """
    import cupy as cp  # noqa: PLC0415

    from libertem_holo.base.unwrap import _irls_numba_cuda  # noqa: PLC0415
    rng = np.random.default_rng(3)
    shape = (24, 19)
    wave = cp.asarray((rng.uniform(0.1, 1, shape) * np.exp(
        1j * _vortex(shape) + 0.2j * rng.standard_normal(shape))).astype(dtype))
    edges = (cp.ones((shape[0], shape[1] - 1)), cp.ones((shape[0] - 1, shape[1])))
    with_invalid = wave.copy()
    with_invalid[3, 4] = np.nan  # launches `restore_invalid`
    unwrap_phase_irls(with_invalid)
    for solve, options in (
        (unwrap_phase_irls, IRLSOptions()),
        (unwrap_phase_irls, IRLSOptions(cold_start="least_squares")),
        (unwrap_phase_admm, ADMMOptions()),
        (unwrap_phase_admm, ADMMOptions(cold_start="least_squares")),
    ):
        for wrapped, edge_weights in ((wave, None), (cp.angle(wave), edges)):
            kwargs = {"options": options, "edge_weights": edge_weights}
            result = solve(wrapped, **kwargs)
            for prior in (result, L1BranchesPrior(result.branches, epsilon=0.5)):
                solve(wrapped, prior=prior, **kwargs)
    real = np.float32 if dtype == np.complex64 else np.float64
    kernels = _irls_numba_cuda.kernels(real)
    declared = _irls_numba_cuda.signatures(real)
    assert kernels.keys() == declared.keys()
    for name, kernel in kernels.items():
        assert kernel.signatures == [declared[name]], name


@pytest.mark.parametrize("own_workspaces", [True, False])
def test_irls_gpu_threads_setting_up_at_once(own_workspaces):
    """Threads whose first solves capture their CUDA graphs at the same time.

    Either with a workspace per thread, or all on the default workspace, one at a
    time.
    Each thread uses its own stream, as workers should: a device-wide
    synchronization in another thread would invalidate the captures.
    """
    import threading  # noqa: PLC0415

    import cupy as cp  # noqa: PLC0415
    threads = 8
    fields = [_noisy(shape=(96, 80 + s % 2), seed=s) for s in range(threads)]
    expected = [unwrap_phase_irls(f).branches for f in fields]
    start = threading.Barrier(threads)
    failures = []

    def solve(i: int) -> None:
        workspace = Workspace() if own_workspaces else None
        try:
            with cp.cuda.Stream(non_blocking=True) as stream:
                start.wait()
                for _ in range(3):
                    result = unwrap_phase_irls(
                        cp.asarray(fields[i]), workspace=workspace,
                    )
                    branches = result.branches.get(stream=stream)
                    if not np.array_equal(branches, expected[i]):
                        failures.append((i, "differs"))
        except Exception as e:  # noqa: BLE001
            failures.append((i, repr(e)))

    workers = [threading.Thread(target=solve, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not failures


def _assert_admm_same(gpu: L1Result, cpu: L1Result) -> None:
    """Assert the same ADMM result, up to rounding in the cosine transforms."""
    np.testing.assert_array_equal(gpu.branches.get(), cpu.branches)
    np.testing.assert_allclose(gpu.phase.get(), cpu.phase, rtol=0, atol=1e-5)
    assert [r.cut_turns for r in gpu.rounds] == [r.cut_turns for r in cpu.rounds]
    assert gpu.selected_round == cpu.selected_round
    assert gpu.cut_turns == cpu.cut_turns
    assert gpu.cut_cost == pytest.approx(cpu.cut_cost, rel=1e-9)
    assert gpu.status == cpu.status
    assert gpu.resume_epsilon == pytest.approx(cpu.resume_epsilon, rel=1e-12)
    tol = 1e-5 if cpu.weights_x.dtype == np.float32 else 1e-12
    np.testing.assert_allclose(gpu.weights_x.get(), cpu.weights_x, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "field",
    [
        pytest.param(_dipole, id="dipole"),
        pytest.param(_noisy, id="noisy"),
        pytest.param(lambda: _vortex((33, 47)), id="vortex odd shape"),
        pytest.param(lambda: np.full((16, 20), 0.4), id="constant"),
        pytest.param(lambda: _wave().astype(np.complex64), id="wave"),
    ],
)
def test_admm_gpu_matches_cpu(field, dtype):
    import cupy as cp  # noqa: PLC0415
    data = field()
    if not np.iscomplexobj(data):
        data = data.astype(dtype)
    elif dtype == np.float64:
        data = data.astype(np.complex128)
    _assert_admm_same(unwrap_phase_admm(cp.asarray(data)), unwrap_phase_admm(data))


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_admm_gpu_single_rows_and_columns(shape, complex_input):
    """Solved on the host, with the result on the device, and as a prior."""
    import cupy as cp  # noqa: PLC0415
    y, x = np.mgrid[:shape[0], :shape[1]]
    data = _wrap(1.7 * x + 2.1 * y)
    if complex_input:
        data = np.exp(1j * data)
    cpu = unwrap_phase_admm(data)
    gpu = unwrap_phase_admm(cp.asarray(data))
    assert isinstance(gpu.phase, cp.ndarray)
    assert isinstance(gpu.weights_y, cp.ndarray)
    _assert_admm_same(gpu, cpu)
    assert gpu.cut_turns == 0
    warm = unwrap_phase_admm(cp.asarray(data), prior=gpu)
    _assert_admm_same(warm, unwrap_phase_admm(data, prior=cpu))


def test_admm_gpu_warm_starts_and_edge_weights_match_cpu():
    import cupy as cp  # noqa: PLC0415
    wrapped = _noisy().astype(np.float32)
    rng = np.random.default_rng(6)
    edges = (rng.uniform(0.2, 1, (70, 89)), rng.uniform(0.2, 1, (69, 90)))
    cold = unwrap_phase_admm(wrapped, edge_weights=edges)
    cold_gpu = unwrap_phase_admm(
        cp.asarray(wrapped), edge_weights=tuple(cp.asarray(e) for e in edges),
    )
    _assert_admm_same(cold_gpu, cold)
    options = ADMMOptions(iterations_per_round=4, rho_scale=1.0)
    for prior in (cold, L1BranchesPrior(cold.branches, epsilon=0.02)):
        gpu_prior = (
            L1BranchesPrior(cp.asarray(prior.branches), epsilon=prior.epsilon)
            if isinstance(prior, L1BranchesPrior) else cold_gpu
        )
        _assert_admm_same(
            unwrap_phase_admm(cp.asarray(wrapped), prior=gpu_prior, options=options),
            unwrap_phase_admm(wrapped, prior=prior, options=options),
        )


@pytest.mark.parametrize(
    "method", ["skimage", "quality", "laplacian", "irls", "admm"],
)
def test_phase_unwrap_returns_the_kind_of_array_it_is_given(method):
    """Cupy in, cupy out: on the GPU where the method can, else via the host."""
    import cupy as cp  # noqa: PLC0415
    wave = _wave()
    cpu = phase_unwrap(wave, method=method)
    gpu = phase_unwrap(cp.asarray(wave), method=method)
    assert isinstance(gpu, cp.ndarray)
    assert isinstance(cpu, np.ndarray)
    # The phase of the wave rounds differently on the GPU, by an ulp or so; a
    # wrong branch would be off by 2 pi:
    np.testing.assert_allclose(gpu.get(), cpu, rtol=0, atol=1e-4)


def _with_invalid(field: np.ndarray, kind: str) -> np.ndarray:
    """`field` with invalid pixels of `kind`: a pixel, a hole, a border, infinity."""
    field = field.copy()
    if kind == "pixel":
        field[3, 4] = np.nan
    elif kind == "hole":
        field[20:26, 30:41] = np.nan
    elif kind == "border":
        field[:, :7] = np.nan
        field[-3:, :] = np.nan
    elif kind == "infinite":
        field[11, 12] = np.inf
        field[30, 50] = -np.inf
    else:
        field[...] = np.nan
    return field


_INVALID_KINDS = ["pixel", "hole", "border", "infinite", "all"]


@pytest.mark.parametrize("kind", _INVALID_KINDS)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("input_", ["phase", "phase with edge weights"])
def test_irls_gpu_invalid_pixels_are_masked_as_on_the_cpu(kind, dtype, input_):
    import cupy as cp  # noqa: PLC0415
    wave = _with_invalid(_wave().astype(dtype), kind)
    data = np.where(np.isfinite(wave), np.angle(wave), np.nan)
    kwargs, gpu_kwargs = {}, {}
    if input_ == "phase with edge weights":
        edges = amplitude_edge_weights(np.abs(wave))
        kwargs = {"edge_weights": edges}
        gpu_kwargs = {"edge_weights": tuple(cp.asarray(e) for e in edges)}
    cpu = unwrap_phase_irls(data, **kwargs)
    gpu = unwrap_phase_irls(cp.asarray(data), **gpu_kwargs)
    # The small edge weights at invalid pixels make the system worse conditioned:
    rtol = 1e-2 if dtype == np.complex64 else 1e-3
    _assert_same(gpu, cpu, residual_rtol=rtol)
    invalid = ~np.isfinite(wave)
    assert np.isnan(gpu.phase.get()[invalid]).all()
    assert np.isfinite(gpu.phase.get()[~invalid]).all()
    warm_cpu = unwrap_phase_irls(data, prior=cpu, **kwargs)
    warm_gpu = unwrap_phase_irls(cp.asarray(data), prior=gpu, **gpu_kwargs)
    _assert_same(warm_gpu, warm_cpu, residual_rtol=rtol)


@pytest.mark.parametrize("kind", _INVALID_KINDS)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_admm_gpu_invalid_pixels_are_masked_as_on_the_cpu(kind, dtype):
    import cupy as cp  # noqa: PLC0415
    wave = _with_invalid(_wave().astype(dtype), kind)
    data = np.where(np.isfinite(wave), np.angle(wave), np.nan)
    cpu = unwrap_phase_admm(data)
    gpu = unwrap_phase_admm(cp.asarray(data))
    _assert_admm_same(gpu, cpu)
    invalid = ~np.isfinite(wave)
    assert np.isnan(gpu.phase.get()[invalid]).all()
    assert (gpu.branches.get()[invalid] == 0).all()
    _assert_admm_same(
        unwrap_phase_admm(cp.asarray(data), prior=gpu),
        unwrap_phase_admm(data, prior=cpu),
    )


@pytest.mark.parametrize("solve", [unwrap_phase_irls, unwrap_phase_admm])
@pytest.mark.parametrize("kind", _INVALID_KINDS)
def test_gpu_invalid_pixels_of_a_wave_are_those_of_its_phase(solve, kind):
    """A wave with invalid pixels gives what its phase and amplitude edge weights give.

    Both on the GPU, since numpy's and cupy's angle may differ in the last bit.
    """
    import cupy as cp  # noqa: PLC0415
    wave = cp.asarray(_with_invalid(_wave(), kind))
    from_wave = solve(wave)
    phase = cp.where(cp.isfinite(wave), cp.angle(wave), cp.nan)
    from_phase = solve(phase, edge_weights=amplitude_edge_weights(cp.abs(wave)))
    for name in ("phase", "branches", "weights_x", "weights_y"):
        np.testing.assert_array_equal(
            getattr(from_wave, name).get(), getattr(from_phase, name).get(),
            err_msg=name,
        )
    assert from_wave.cut_cost == from_phase.cut_cost


def test_gpu_amplitude_weights_on_many_blocks_are_the_cpus():
    """The largest weight is found among all blocks (more than 256 on large GPUs).

    The amplitude grows with the row, so that only the last rows have the
    largest weight, and the grid-stride loop hands them to the last blocks.
    """
    import cupy as cp  # noqa: PLC0415
    shape = (1024, 1024)
    amplitude = np.linspace(0.1, 1, shape[0])[:, np.newaxis] * np.ones(shape)
    wave = (amplitude * np.exp(1j * _noisy(shape))).astype(np.complex64)
    _, (edge_x, edge_y), _ = irls._wave_setup(wave, 4)  # noqa: SLF001
    gpu = _irls_cupy.CupyWeightedProblem(shape, np.float32)
    gpu.fill_from_wave(
        cp.asarray(wave), irls._AMPLITUDE_FLOOR, irls._AMPLITUDE_PERCENTILE,  # noqa: SLF001
    )
    # (abs may differ in the last bit between numpy and cupy)
    np.testing.assert_allclose(gpu.edge_x.get(), edge_x, rtol=1e-6)
    np.testing.assert_allclose(gpu.edge_y.get(), edge_y, rtol=1e-6)


def test_irls_gpu_threads_solving_the_same_shape_do_not_interfere():
    """Concurrent solves of same-sized inputs in several threads, on one GPU.

    Each thread has its own workspace, FFT plans included: if threads shared a
    plan's work area, concurrent solves would overwrite each other's intermediate
    results.
    """
    import threading  # noqa: PLC0415

    import cupy as cp  # noqa: PLC0415
    threads, repeats = 4, 3
    fields = [_noisy(shape=(2048, 2048), seed=s) for s in range(threads)]
    expected = [unwrap_phase_irls(cp.asarray(f)).branches.get() for f in fields]
    start = threading.Barrier(threads)
    wrong = []

    def solve(i: int) -> None:
        wave = cp.asarray(fields[i])
        with Workspace() as workspace:
            start.wait()
            for _ in range(repeats):
                branches = unwrap_phase_irls(wave, workspace=workspace).branches.get()
                if not np.array_equal(branches, expected[i]):
                    wrong.append(i)

    workers = [threading.Thread(target=solve, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not wrong


@pytest.mark.parametrize("solve", [unwrap_phase_irls, unwrap_phase_admm])
def test_arrays_on_the_other_device_are_rejected(solve):
    import cupy as cp  # noqa: PLC0415
    wrapped = _noisy().astype(np.float32)
    edges = (np.ones((70, 89)), np.ones((69, 90)))
    result = solve(wrapped)
    on_gpu = cp.asarray(wrapped)
    with pytest.raises(TypeError, match="edge_weights must be on the device"):
        solve(on_gpu, edge_weights=edges)
    with pytest.raises(TypeError, match="edge_weights must be on the device"):
        solve(wrapped, edge_weights=tuple(cp.asarray(e) for e in edges))
    with pytest.raises(TypeError, match="prior's phase must be on the device"):
        solve(on_gpu, prior=result)
    gpu_prior = L1BranchesPrior(cp.asarray(result.branches), epsilon=0.5)
    with pytest.raises(TypeError, match="prior's branches must be on the device"):
        solve(wrapped, prior=gpu_prior)


def test_gpu_default_workspace_holds_one_problem():
    """A solve of another shape (or method) frees the problem and replaces it."""
    import cupy as cp  # noqa: PLC0415
    pool = cp.get_default_memory_pool()
    clear_default_workspace()
    baseline = pool.used_bytes()
    small = cp.asarray(_wave(shape=(64, 80)))
    large = cp.asarray(_wave(shape=(300, 400)))
    inputs = small.nbytes + large.nbytes
    unwrap_phase_irls(large)
    one_large = pool.used_bytes() - baseline
    unwrap_phase_irls(small)
    unwrap_phase_admm(small)
    one_small = pool.used_bytes() - baseline
    assert one_small < (one_large - inputs) / 4 + inputs
    unwrap_phase_irls(large)
    assert pool.used_bytes() - baseline <= one_large
    clear_default_workspace()
    assert pool.used_bytes() - baseline == inputs
    assert pool.total_bytes() - pool.used_bytes() == 0  # the pool was freed


def test_gpu_solves_keep_no_arrays_alive():
    import gc  # noqa: PLC0415
    import weakref  # noqa: PLC0415

    import cupy as cp  # noqa: PLC0415
    wave = _with_invalid(_wave(), "hole")
    for solve in (unwrap_phase_irls, unwrap_phase_admm):
        inputs = cp.asarray(wave)
        result = solve(inputs)
        refs = [weakref.ref(inputs), weakref.ref(result.phase)]
        del inputs, result
        gc.collect()
        assert all(ref() is None for ref in refs)


def test_gpu_workspace_must_be_a_workspace():
    import cupy as cp  # noqa: PLC0415
    with pytest.raises(TypeError, match="workspace must be a Workspace"):
        unwrap_phase_admm(cp.zeros((4, 5)), workspace={})

"""GPU (cupy) implementation of the weighted problems of the IRLS and ADMM unwrapping.

The counterpart of :class:`~libertem_holo.base.unwrap.irls._WeightedProblem`
for cupy input: the same arithmetic in the same order per pixel, and dot products
summed in float64, so the results match the CPU's when both pad to the same lengths
(see `PADDED_LENGTH`). At the image sizes of holography (~500² to 2048²), the work
per CG iteration is a few passes over one image, so kernel launches and host
synchronizations cost more than the arithmetic. Hence:

* The preconditioner (the unweighted Neumann Poisson solve) is a pair of DCTs, each
  computed as a real FFT of the reordered image with twiddles (J. Makhoul, 1980),
  with pre-built cuFFT plans. IRLS pads to even 3-smooth lengths
  (`_gpu_padded_length`); ADMM does not pad.
* The CG scalars (alpha, beta, the residuals, the stop flag) stay on the device. Each
  kernel returns at once when the stop flag is set, so a fixed sequence of
  iterations does what the CPU loop does. It is captured as a CUDA graph once per
  problem and set of options, and replayed: one host synchronization per IRLS round
  of up to `_GRAPH_ITERATIONS` CG iterations. The setup of a solve and its starting
  point are one more graph (see :meth:`CupyWeightedProblem.cold_start`).
* Dot products are reduced in float64 and in a fixed order (per-block partial sums,
  then the last block to finish adds them up), so results do not vary from run to
  run. Some are deferred: the next kernel that needs them adds up the partial sums
  (see `_DEFER_RZ`).

The kernels are numba CUDA kernels (`_irls_numba_cuda`). Launching one from Python
costs ~40 us of host time (measured on a GH200), so the rounds launch them only
while capturing graphs, and then replay the graphs.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import threading
import warnings
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
    from cupy.cuda import cufft
    from cupyx.scipy.fft import get_fft_plan
except ImportError:
    # Importable without cupy (for example for the doctests); only cupy input
    # gets here.
    cp = None

from libertem_holo.base.workspace import Workspace, default_workspace

from .irls import _INVALID_EDGE_WEIGHT, _is_smooth

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .admm import ADMMOptions
    from .irls import IRLSOptions, L1BranchesPrior, L1PhasePrior, L1Round, _Candidate

_BLOCK = 256
# CG iterations per captured graph: the default `iterations_per_round` fits in one.
_GRAPH_ITERATIONS = 16  # even: see `_cg_iterations`
# Grid of the grid-stride kernels: at most `_THREADS_PER_SM` threads per
# multiprocessor (8 blocks of 256), and at least `_MIN_PER_THREAD` elements per
# thread, which keeps the reduction tails of small images short. On a GH200,
# 8 blocks per multiprocessor is full occupancy: 4 left the streaming kernels at
# 45 % of DRAM bandwidth (Nsight Compute), 8 made a CG round 12 % faster at
# 2048^2, and larger grids made the reductions slower.
_THREADS_PER_SM = 2048  # the most a multiprocessor holds at once on a GH200
_MIN_PER_THREAD = 2
# Which CG dot products are added up from their partial sums by the kernels that
# need them, instead of by the last block of the kernel that computes them. That
# takes the last block's serial tail (~0.5 us) off the critical path, which gains
# ~4 % at ~500^2 for all three, but every block of the consuming kernels then adds
# up all partial sums, which costs ~4 % at 2048^2 (measured on a GH200). So they
# are deferred only on grids of up to `_DEFER_MAX_GRID` blocks.
_DEFER_RZ = True  # r . z, in apply_operator (beta) and cg_update (alpha)
_DEFER_ALPHA = True  # p . Ap, in cg_update
_DEFER_RR = True  # r . r, in makhoul_spectrum (the convergence check)
_DEFER_MAX_GRID = 512
# ADMM (`admm.ADMMOptions`): the grid of `admm_update` (no reductions, so one
# pixel per thread), and the rounds per graph launch: 2 up to 512^2 pixels, where
# a round costs little more than a host round trip (~25 us on a GH200), else 1,
# since a round after the stop is wasted work.
_ADMM_GRID = (lambda n: max(1, -(-n // _BLOCK)))  # noqa: E731
_ADMM_CHUNK = (lambda n: 2 if n <= 512 * 512 else 1)  # noqa: E731

# Indices into the double scalars `s` on the device:
(_S_NORM, _S_RZ, _S_ALPHA, _S_BETA, _S_MEAN, _S_TURNS, _S_COST, _S_LARGEST, _S_INVALID,
 _S_INVALID_PIXELS) = range(10)
# Indices into the int flags `f` on the device:
_F_DONE, _F_ITER, _F_ITERATED, _F_ZERO, _F_PENDING = range(5)


def _fft_plans(extended: cp.ndarray, spectrum: cp.ndarray) -> tuple:
    """Return new cuFFT plans for the real FFT pair on the padded grid.

    Each problem gets its own pair: every execution of a plan keeps its
    intermediate results in the plan's work area, so concurrent solves (in
    different threads) that shared plans would overwrite each other's. Creating a
    pair takes ~10 ms (measured on a GH200), once per problem.
    """
    return (
        get_fft_plan(extended, value_type="R2C"),
        get_fft_plan(spectrum, shape=extended.shape, value_type="C2R"),
    )


# The padded lengths of the IRLS preconditioner; None means `_gpu_padded_length`.
# Tests set `irls._padded_length` to get the CPU's results bit for bit.
PADDED_LENGTH: Callable[[int], int] | None = None

# Held while kernels are launched from Python, which includes every graph capture
# and setting up a problem: numba compiles and caches a kernel at its first
# launch, which two threads must not do at once, and captures of different
# threads should not interleave.
_SETUP_LOCK = threading.RLock()

# The slot of the GPU problems in a `Workspace`:
_SLOT = "unwrap"
# The most numba array wrappers a problem caches (see `_numba_args`); a problem
# has about 60 arrays of its own:
_WRAPPER_CACHE_SIZE = 256


def problem_for(
    wrapped: cp.ndarray,
    edge_weights: tuple[cp.ndarray, cp.ndarray] | None,
    *,
    deferred: bool = False,
    method: str = "irls",
    workspace: Workspace | None = None,
) -> CupyWeightedProblem:
    """Return a problem for `wrapped`, from `workspace` (None: the default one).

    Buffers, FFT plans and the captured CUDA graphs only depend on the shape, the
    dtype, the device and the `method` ("irls" or "admm"), so a series of
    same-sized inputs reuses them. The workspace keeps one problem; one for other
    inputs replaces it. The caller must hold the workspace's lock while it uses
    the problem. With `deferred`, the input is only copied into the problem; the
    next call must be :meth:`CupyWeightedProblem.cold_start` or
    :meth:`~CupyWeightedProblem.warm_start`, which sets the problem up.
    """
    problem = _workspace(wrapped.shape, wrapped.dtype, method, workspace)
    if deferred:
        problem.stage_phase(wrapped, edge_weights)
    else:
        problem.fill(wrapped, edge_weights)
    return problem


def problem_for_wave(
    wave: cp.ndarray, floor: float, percentile: float, *, deferred: bool = False,
    method: str = "irls", workspace: Workspace | None = None,
) -> CupyWeightedProblem:
    """Return a problem for the phase of `wave`, with edge weights from its amplitude.

    See :func:`problem_for` and :meth:`CupyWeightedProblem.fill_from_wave`. With
    `deferred`, the wave is only copied into the problem; the next call must be
    :meth:`CupyWeightedProblem.cold_start` or
    :meth:`~CupyWeightedProblem.warm_start`, which sets the problem up.
    """
    dtype = np.float32 if wave.dtype == np.complex64 else np.float64
    wave = cp.ascontiguousarray(wave, dtype=np.result_type(dtype, np.complex64))
    problem = _workspace(wave.shape, np.dtype(dtype), method, workspace)
    if deferred:
        problem.stage_wave(wave, floor, percentile)
    else:
        problem.fill_from_wave(wave, floor, percentile)
    return problem


def _workspace(
    shape: tuple[int, int],
    dtype: np.dtype,
    method: str,
    workspace: Workspace | None,
) -> CupyWeightedProblem:
    """Return the problem of `workspace` for these inputs, on the current device.

    See :func:`problem_for`.
    """
    if workspace is None:
        workspace = default_workspace()
    key = (
        shape, np.dtype(dtype).str, cp.cuda.Device().id, method,
        PADDED_LENGTH, _BLOCK, _MIN_PER_THREAD,
        _DEFER_RZ, _DEFER_ALPHA, _DEFER_RR, _DEFER_MAX_GRID,
    )

    def build() -> CupyWeightedProblem:
        with _SETUP_LOCK:
            return CupyWeightedProblem(
                shape, dtype, padded_length=PADDED_LENGTH, method=method,
            )

    return workspace.entry(_SLOT, key, build)


def _gpu_padded_length(length: int) -> int:
    """Return the smallest even 3-smooth number that is at least `length`.

    Lengths with factors of 5, 7 or 11, which the CPU version
    (`irls._padded_length`) allows, are much slower with cuFFT for some sizes
    (500 against 512, 1008 against 1024), and were slower overall on a GH200,
    padding included.
    """
    candidate = max(length, 2)
    while candidate % 2 != 0 or not _is_smooth(candidate, limit=3):
        candidate += 1
    return candidate


def _pinned(size: int, dtype: type) -> np.ndarray:
    """Return a numpy array of `size` elements in pinned (page-locked) memory."""
    dtype = np.dtype(dtype)
    memory = cp.cuda.alloc_pinned_memory(max(size, 1) * dtype.itemsize)
    return np.frombuffer(memory, dtype=dtype, count=size)


_multiprocessors: dict[int, int] = {}


def _multiprocessor_count() -> int:
    device = cp.cuda.Device()
    if device.id not in _multiprocessors:
        # `Device.attributes` queries all attributes, which takes milliseconds:
        _multiprocessors[device.id] = cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrMultiProcessorCount, device.id,
        )
    return _multiprocessors[device.id]


class CupyWeightedProblem:
    """The weighted problem of IRLS or ADMM rounds, on the GPU.

    Same interface as :class:`~libertem_holo.base.unwrap.irls._WeightedProblem`.
    All arrays are cupy arrays. A problem holds the buffers, FFT plans and
    captured CUDA graphs for one shape, dtype, method and device, and is reused
    by later solves. Use :func:`problem_for` to get one.

    Everything runs on a private non-blocking stream (stream capture needs one),
    ordered with the caller's stream on entry and exit (:meth:`_on_stream`).
    """

    xp = cp

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: np.dtype,
        padded_length: Callable[[int], int] | None = None,
        method: str = "irls",
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.shape = rows, cols = shape
        self.size = rows * cols
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.block = _BLOCK
        self.grid = max(1, min(
            -(-self.size // (self.block * _MIN_PER_THREAD)),
            min(_THREADS_PER_SM // self.block, 32) * _multiprocessor_count(),
        ))
        self._init_kernels()
        self._set_deferrals()
        image = lambda shape=shape: cp.empty(shape, dtype=self.dtype)  # noqa: E731
        with self.stream:
            edges_x, edges_y = (rows, cols - 1), (rows - 1, cols)
            self.target_x, self.target_y = image(edges_x), image(edges_y)
            self.jumps_x = cp.empty((rows, cols - 1), dtype=cp.int32)
            self.jumps_y = cp.empty((rows - 1, cols), dtype=cp.int32)
            n_x, n_y = rows * (cols - 1), (rows - 1) * cols
            self.edges = cp.empty(n_x + n_y, dtype=self.dtype)
            self.edge_x = self.edges[:n_x].reshape(edges_x)
            self.edge_y = self.edges[n_x:].reshape(edges_y)
            self.weights_x, self.weights_y = image(edges_x), image(edges_y)
            self.unit_x = cp.ones((rows, cols - 1), dtype=self.dtype)
            self.unit_y = cp.ones((rows - 1, cols), dtype=self.dtype)

            self._allocate_scalars()

            # ADMM's u-step must solve its least-squares problem exactly: on a
            # padded grid, the padding's edges would pull on the result.
            self.method = method
            self._setup_preconditioner(
                (lambda n: n) if method == "admm"
                else padded_length or _gpu_padded_length,
            )

            # CG work arrays; the graphs iterate on `phi` (see `run_round`):
            self.phi, self.b, self.q, self.r, self.z = (image() for _ in range(5))
            # The CG direction, in two buffers: an iteration computes the next one
            # from the current one (see `apply_operator`, mode 2).
            self.directions = (image(), image())
            self._direction = 0
            self.difference = image()
            # Inputs at fixed addresses, for the captured graphs:
            self.wrapped = image()
            # 1 for the invalid pixels, set up with the input:
            self.invalid = cp.zeros(self.size, dtype=cp.uint8)
            self.wave = None  # a staged wave (see `stage_wave`)
            self.prior_phase = None  # a staged prior (see `warm_start`)
            self.prior_branches = None
            # The parameters of `charbonnier_weights` (eps^2, "unless settled"):
            self.start_params = cp.zeros(2, dtype=cp.float64)  # for a warm start
            self.l2 = image()  # the L2 solution of the cold start
            self.round_params = cp.zeros(3, dtype=cp.float64)  # per round; [2] unused
            # set on the device, for `reweight` and the cold start from zero:
            self.direct_params = cp.zeros(2, dtype=cp.float64)
            # Two slots for a round's candidate (phase, branches, weights): a round
            # writes into the slot that does not hold the kept candidate, so that
            # keeping a candidate needs no copy (see `keep`).
            self.slots = tuple(
                (cp.empty(self.shape, dtype=cp.float64),
                 cp.empty(self.shape, dtype=cp.int32),
                 image(edges_x), image(edges_y))
                for _ in range(2)
            )
            self._kept: int | None = None
            # Reused for the stream ordering of `_on_stream`:
            self._event_caller = cp.cuda.Event(disable_timing=True)
            self._event_own = cp.cuda.Event(disable_timing=True)
            # State of the radix selection (see `_select`):
            self.select_state = cp.zeros(4, dtype=cp.uint64)
            self.select_hist = cp.zeros(2 * 2048, dtype=cp.uint32)
            self.select_counter = cp.zeros(1, dtype=cp.uint32)
            self.select_out = cp.zeros(4, dtype=cp.float64)
            self.offset = cp.zeros(1, dtype=self.dtype)
        if self.extended.size >= 2**31:
            msg = f"image too large for the GPU solver: {shape}"
            raise ValueError(msg)
        self._init_host_state()

    def _set_deferrals(self) -> None:
        """Decide which dot products to defer, for this grid (see `_DEFER_RZ`)."""
        small = self.grid <= _DEFER_MAX_GRID
        self.defer_rz = _DEFER_RZ and small
        self.defer_alpha = _DEFER_ALPHA and small
        self.defer_rr = _DEFER_RR and small

    def _init_host_state(self) -> None:
        """Set up the pinned host buffers, the graph caches, and a solve's state."""
        self._host_s = _pinned(self.s.size, np.float64)
        self._host_params = _pinned(3, np.float64)
        self._admm_graphs: dict[bool, cp.cuda.graph.Graph] = {}
        self._admm_key = None
        self._history_length = 0
        self._graph = None
        self._graph_key = None
        self._start_graphs: dict[tuple, tuple[tuple, cp.cuda.graph.Graph]] = {}
        self._host_start_params = _pinned(2, np.float64)
        self._check_weights = False
        # whether _host_s holds the number of invalid pixels:
        self._invalid_on_host = False
        self._staged = None
        self._iterations = 0

    def _setup_preconditioner(self, padded_length: Callable[[int], int]) -> None:
        """Set up the Neumann Poisson solve on the zero-padded grid (P1, P2).

        For ADMM, `padded_length` is the identity: no padding.

        DCTs by FFTs of size (P1, P2) of the reordered image, with twiddles. The
        preconditioner is C3 D C2 for the orthonormal DCTs C2, C3 = C2^-1 and the
        diagonal D; the normalizations of the DCTs cancel around D, so it is B D F
        for the unnormalized forward transform F and its inverse B. Both are a
        reordering, an FFT and a twiddle.
        """
        rows, cols = self.shape
        self.padded_shape = p1, p2 = padded_length(rows), padded_length(cols)
        cdtype = cp.complex64 if self.dtype == np.float32 else cp.complex128
        u = cp.arange(p1, dtype=cp.float64)[:, None]
        v = cp.arange(p2, dtype=cp.float64)[None, :]
        # B F = 2, and 1/N of the inverse FFT:
        scale = 2 * p1 * p2
        self.twiddles = [
            cp.exp(-1j * np.pi * cp.arange(n + 1) / (2 * n)).astype(cdtype)
            for n in (p1, p2)
        ]
        self.spectrum_out = cp.zeros((p1, p2 // 2 + 1), dtype=cdtype)
        # The FFT's input, which `cg_update` writes into directly. Separate from
        # `extended`, which the inverse FFT overwrites, so that its padding stays
        # zero.
        self.fft_input = cp.zeros((p1, p2), dtype=self.dtype)
        symbol = 4 - 2 * cp.cos(np.pi * u / p1) - 2 * cp.cos(np.pi * v / p2)
        inverse = 1.0 / (cp.where(symbol == 0, 1.0, symbol) * scale)
        inverse[symbol == 0] = 0
        self.inverse_symbol = inverse.astype(self.dtype)
        self.extended = cp.zeros((p1, p2), dtype=self.dtype)
        self.spectrum = cp.zeros((p1, p2 // 2 + 1), dtype=cdtype)
        self.plan_forward, self.plan_inverse = _fft_plans(self.extended, self.spectrum)

    def fill(
        self,
        wrapped: cp.ndarray,
        edge_weights: tuple[cp.ndarray, cp.ndarray] | None,
    ) -> None:
        """Set up the problem for `wrapped`."""
        with self._on_stream():
            if edge_weights is not None:
                self.edge_x[...] = edge_weights[0]
                self.edge_y[...] = edge_weights[1]
                self._memcpy(self.weights_x, self.edge_x)
                self._memcpy(self.weights_y, self.edge_y)
            self._prepare_targets(
                cp.ascontiguousarray(wrapped, dtype=self.dtype),
                unit_edges=edge_weights is None, detect=True,
            )
        self._invalid_on_host = False
        self._iterations = 0
        self._kept = None
        self._staged = None

    def fill_from_wave(self, wave: cp.ndarray, floor: float, percentile: float) -> None:
        """Set up the problem for the phase of `wave`, with amplitude edge weights.

        The same as `fill(angle(wave), amplitude_edge_weights(abs(wave), floor,
        percentile))`, in a few kernels instead of a few dozen cupy calls. Whether
        the weights are valid is checked at the next :meth:`cuts`, which
        synchronizes anyway.
        """
        with self._on_stream():
            self._enqueue_fill_from_wave(wave, floor, percentile)
        self._invalid_on_host = False
        self._check_weights = True
        self._iterations = 0
        self._kept = None
        self._staged = None

    def _enqueue_fill_from_wave(
        self, wave: cp.ndarray, floor: float, percentile: float,
    ) -> None:
        rows, cols = self.shape
        n = self.size
        amplitude = self.difference  # free until the first `congruent`
        self._launch(
            "wave_phase_amplitude",
            (wave, self.wrapped, amplitude, self.invalid, np.int32(n), self.partials,
             self.counter, self.s, np.float64(percentile), self.select_state,
             self.select_out),
        )
        # The percentile of the valid amplitudes (the invalid ones are +inf), at
        # the ranks that `wave_phase_amplitude` set:
        self._select(amplitude, (0, 0), mode=1, device_ranks=True)
        for mode in (0, 1):
            self._launch(
                "amplitude_edges",
                (amplitude, self.invalid, self.select_out, np.float64(floor),
                 np.float64(_INVALID_EDGE_WEIGHT), np.int32(rows), np.int32(cols),
                 np.int32(mode), self.partials, self.counter, self.s, self.edge_x,
                 self.edge_y, self.weights_x, self.weights_y),
            )
        self._prepare_targets(self.wrapped, unit_edges=False, detect=False)

    def stage_wave(self, wave: cp.ndarray, floor: float, percentile: float) -> None:
        """Copy `wave` into the problem; the start graph sets the problem up from it."""
        with self._on_stream():
            if self.wave is None or self.wave.dtype != wave.dtype:
                self.wave = cp.empty(self.shape, dtype=wave.dtype)
            self._memcpy(self.wave, wave)
        self._stage(("wave", floor, percentile))

    def stage_phase(
        self,
        wrapped: cp.ndarray,
        edge_weights: tuple[cp.ndarray, cp.ndarray] | None,
    ) -> None:
        """Copy the wrapped phase (and the edge weights, if given) into the problem."""
        with self._on_stream():
            self._memcpy(self.wrapped, cp.ascontiguousarray(wrapped, dtype=self.dtype))
            if edge_weights is not None:
                self.edge_x[...] = edge_weights[0]
                self.edge_y[...] = edge_weights[1]
        self._stage(("phase", edge_weights is not None))

    def _stage(self, staged: tuple) -> None:
        self._staged = staged
        self._check_weights = False
        self._iterations = 0
        self._kept = None

    def cold_start(
        self, wrapped: cp.ndarray, options: IRLSOptions,
    ) -> tuple[cp.ndarray, _Candidate, float]:
        """Start a solve without a prior; see `irls._cold_start`.

        If the input was staged (`deferred`), one CUDA graph and one
        synchronization: the setup, the rounded L2 solution as the first
        candidate (into slot 0), and the starting point.
        """
        from .irls import _cold_start  # noqa: PLC0415

        if self._staged is None:
            return _cold_start(self, wrapped, options)
        best = self._run_start(options.cold_start, options, weights_epsilon=math.inf)
        if options.cold_start == "zero" and best.cut_turns > 0:
            return self.phi, best, options.epsilon_start
        return self.phi, best, math.inf

    def warm_start(
        self,
        wrapped: cp.ndarray,
        prior: L1PhasePrior | L1BranchesPrior,
        options: IRLSOptions,
    ) -> tuple[cp.ndarray, _Candidate, float]:
        """Start a solve from a prior; see `irls._warm_start`.

        If the input was staged (`deferred`), one CUDA graph: the setup, the
        starting phase from the prior, its weights, and the rounded starting phase
        as the first candidate. The epsilon of the weights comes from pinned
        memory, so one graph serves every prior of the same type.
        """
        from .irls import L1PhasePrior, _warm_start  # noqa: PLC0415

        if self._staged is None:
            return _warm_start(self, wrapped, prior, options)
        epsilon = min(max(prior.epsilon, options.epsilon_min), options.epsilon_start)
        phase_prior = isinstance(prior, L1PhasePrior)
        with self._on_stream():
            if self.prior_phase is None:
                self.prior_phase = cp.empty(self.shape, dtype=cp.float64)
                self.prior_branches = cp.empty(self.shape, dtype=cp.int32)
            if phase_prior:
                target, source = self.prior_phase, prior.phase
            else:
                target, source = self.prior_branches, prior.branches
            source = cp.ascontiguousarray(cp.asarray(source), dtype=target.dtype)
            self._memcpy(target, source)
        self._host_start_params[0] = epsilon * epsilon
        self._host_start_params[1] = 0.0
        best = self._run_start(
            "phase_prior" if phase_prior else "branches_prior", options,
            weights_epsilon=epsilon,
        )
        return self.phi, best, epsilon

    def _run_start(
        self, kind: str, options: IRLSOptions, *, weights_epsilon: float,
    ) -> _Candidate:
        """Run the start graph of `kind` on the staged input; return its candidate."""
        from .irls import _Candidate  # noqa: PLC0415

        staged, self._staged = self._staged, None
        self._invalid_on_host = True
        # One graph per kind of staged input and of start, captured again when
        # the staged parameters or the options change:
        slot = (staged[0], kind)
        key = (staged, kind, options.epsilon_start if kind == "zero" else None)
        entry = self._start_graphs.get(slot)
        if entry is None or entry[0] != key:
            self._start_graphs.pop(slot, None)
            entry = (key, self._capture(
                lambda: self._enqueue_start(staged, kind, options),
            ))
            self._start_graphs[slot] = entry
        graph = entry[1]
        with self._on_stream():
            graph.launch(self.stream)
            self.stream.synchronize()
        s = self._host_s
        if staged[0] == "wave" and s[_S_INVALID] > 0:
            msg = "edge_weights must be finite and greater than zero"
            raise ValueError(msg)
        phase, branches, weights_x, weights_y = self.slots[0]
        self._kept = 0
        return _Candidate(
            phase=phase, branches=branches,
            cut_turns=int(s[_S_TURNS]), cut_cost=float(s[_S_COST]),
            weights_x=weights_x, weights_y=weights_y,
            weights_epsilon=weights_epsilon, converged=True, round_index=-1,
        )

    def _enqueue_start(self, staged: tuple, kind: str, options: IRLSOptions) -> None:
        """Enqueue a start: the setup, the starting point, and the first candidate.

        `staged` is ("wave", floor, percentile) or ("phase", whether edge weights
        were given);
        `kind` is "zero" or "least_squares" (a cold start), "phase_prior" or
        "branches_prior". Mirrors `irls._cold_start` and `irls._warm_start`.
        """
        rows, cols = self.shape
        dims = (np.int32(rows), np.int32(cols))
        if staged[0] == "wave":
            self._enqueue_fill_from_wave(self.wave, staged[1], staged[2])
        else:
            if staged[1]:
                self._memcpy(self.weights_x, self.edge_x)
                self._memcpy(self.weights_y, self.edge_y)
            self._prepare_targets(self.wrapped, unit_edges=not staged[1], detect=True)
        phase, branches, weights_x, weights_y = self.slots[0]

        def candidate(phi: cp.ndarray) -> None:
            self._congruent_into(phi, self.wrapped, phase, branches)
            self._launch(
                "cuts",
                (branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
                 self.invalid, *dims, self.partials, self.counter, self.s),
            )
            self._memcpy(weights_x, self.weights_x)
            self._memcpy(weights_y, self.weights_y)

        def reweight(params: cp.ndarray) -> None:
            self._launch(
                "charbonnier_weights",
                (self.phi, self.target_x, self.target_y, self.edge_x, self.edge_y,
                 params, self.weights_x, self.weights_y, *dims, self.f),
            )

        if kind in ("zero", "least_squares"):
            # The rounded L2 solution is the first candidate:
            self._launch(
                "right_hand_side",
                (self.unit_x, self.target_x, self.unit_y, self.target_y, self.b,
                 *dims, self.partials, self.counter, self.s),
            )
            self._precondition(self.b, self.l2, self.f_free, mode=2)
            candidate(self.l2)
            if kind == "zero":
                # Start from zero, with the weights there; `cold_start` uses
                # this start only if the L2 candidate has cut turns:
                cp.cuda.runtime.memsetAsync(
                    self.phi.data.ptr, 0, self.phi.nbytes, self.stream.ptr,
                )
                eps = options.epsilon_start
                self._launch(
                    "set_params",
                    (self.direct_params, np.float64(eps * eps), np.float64(0)),
                    grid=1, block=1,
                )
                reweight(self.direct_params)
            else:
                self._memcpy(self.phi, self.l2)
        else:
            self._memcpy(self.start_params, self._host_start_params)
            self._launch(
                "starting_phase",
                (self.wrapped, self.prior_phase, self.prior_branches,
                 np.int32(kind == "branches_prior"), self.phi, np.int64(self.size)),
            )
            reweight(self.start_params)
            candidate(self.phi)
        self._memcpy(self._host_s, self.s)

    def _prepare_targets(
        self, wrapped: cp.ndarray, *, unit_edges: bool, detect: bool,
    ) -> None:
        """Enqueue the targets; with `detect`, find the invalid pixels of `wrapped`."""
        rows, cols = self.shape
        self._launch(
            "prepare_targets",
            (wrapped, self.wrapped, self.invalid, self.target_x, self.target_y,
             self.jumps_x, self.jumps_y, self.edge_x, self.edge_y, self.weights_x,
             self.weights_y, np.int32(rows), np.int32(cols), np.int32(unit_edges),
             np.int32(detect), np.float64(_INVALID_EDGE_WEIGHT), self.partials,
             self.counter, self.s),
        )
        if detect:
            self._launch(
                "zero_invalid",
                (self.wrapped, self.invalid, self.s, np.int64(self.size)),
            )

    # -- launches ------------------------------------------------------------

    @contextlib.contextmanager
    def _on_stream(self) -> Iterator[None]:
        """Run on the problem's stream, ordered after and before the caller's stream.

        The problem's stream first waits for the caller's, whose arrays it is
        about to use. At the end, the caller's stream waits for the problem's:
        that makes the results safe to use there, and it keeps the memory pool from
        handing memory that the caller frees after the call to a new array while
        the problem's stream is still reading it. Neither costs a host
        synchronization.
        """
        caller = cp.cuda.get_current_stream()
        if caller == self.stream:
            yield
            return
        # A wait takes the event's state when it is enqueued, so the events can
        # be recorded again at the next call.
        self._event_caller.record(caller)
        self.stream.wait_event(self._event_caller)
        try:
            with self.stream:
                yield
        finally:
            self._event_own.record(self.stream)
            caller.wait_event(self._event_own)

    def _launch(
        self, name: str, args: tuple, grid: int | None = None, block: int | None = None,
    ) -> None:
        with _SETUP_LOCK, warnings.catch_warnings():
            # Small grids are deliberate (flags, small images); numba warns of
            # low occupancy:
            warnings.filterwarnings("ignore", message="Grid size", category=Warning)
            self.kernels[name][
                grid or self.grid, block or self.block, self._numba_stream,
            ](*self._numba_args(args))

    def _allocate_scalars(self) -> None:
        """Allocate the device scalars, flags, and partial sums of reductions."""
        self.s = cp.zeros(10, dtype=cp.float64)
        self.f = cp.zeros(5, dtype=cp.int32)
        # Flags that are never set, for unconditional launches:
        self.f_free = cp.zeros(5, dtype=cp.int32)
        self.counter = cp.zeros(1, dtype=cp.uint32)
        self.partials = cp.zeros(2 * self.grid, dtype=cp.float64)
        # Partial sums of the deferred dot products (see `_DEFER_RZ`); r . z in
        # two buffers, which alternate: the latest and the one before.
        self.partials_rz = tuple(
            cp.zeros(self.grid, dtype=cp.float64) for _ in range(2)
        )
        self._rz_index = 0
        self.partials_pap = cp.zeros(self.grid, dtype=cp.float64)
        self.partials_rr = cp.zeros(self.grid, dtype=cp.float64)

    def _init_kernels(self) -> None:
        """Get the numba kernels for the dtype, and wrap the stream for them."""
        from numba import cuda as numba_cuda  # noqa: PLC0415

        from . import _irls_numba_cuda  # noqa: PLC0415
        if self.block != _irls_numba_cuda.BLOCK:
            msg = (
                f"block size {self.block} does not match the block size the numba "
                f"kernels are compiled for ({_irls_numba_cuda.BLOCK})"
            )
            raise ValueError(msg)
        self.kernels = _irls_numba_cuda.kernels(self.dtype)
        self._numba_stream = numba_cuda.external_stream(self.stream.ptr)
        self._numba_arrays: dict[tuple, object] = {}
        self._numba_cuda = numba_cuda
        # An absent (optional) array argument:
        self.absent_f64 = cp.empty(0, dtype=cp.float64)

    def _numba_args(self, args: tuple) -> tuple:
        """Return the arguments for a numba kernel.

        Arrays are wrapped without the stream synchronization of the array
        interface, which graph capture forbids. A wrapper only describes the
        memory (address, shape, dtype), and does not keep the array alive, so
        it is cached by that description: for another array at the same address
        it is the same. The cache is emptied when it grows large.
        """
        out = []
        for arg in args:
            if isinstance(arg, cp.ndarray):
                key = (arg.data.ptr, arg.shape, arg.dtype.str)
                wrapper = self._numba_arrays.get(key)
                if wrapper is None:
                    if len(self._numba_arrays) >= _WRAPPER_CACHE_SIZE:
                        self._numba_arrays.clear()
                    wrapper = self._numba_cuda.from_cuda_array_interface(
                        arg.ravel().__cuda_array_interface__, owner=None, sync=False,
                    )
                    self._numba_arrays[key] = wrapper
                out.append(wrapper)
            else:
                out.append(arg)
        return tuple(out)

    def _precondition(
        self,
        residual: cp.ndarray,
        out: cp.ndarray,
        flags: cp.ndarray,
        mode: int,
        *,
        reordered: bool = False,
        direction: cp.ndarray | None = None,
        rz_out: cp.ndarray | None = None,
        check: IRLSOptions | None = None,
    ) -> None:
        """Enqueue `out` = M `residual`; `mode` as in the `crop` kernel.

        With `reordered`, `cg_update` has already written the reordered residual
        into the FFT's input. Modes 0 and 4 set `direction` to
        `out`; modes 3 and 4 write the partial sums of r . z to `rz_out`. With
        `check`, the spectral step first checks the convergence of the CG
        iteration from the partial sums of r . r (see `_DEFER_RR`).
        """
        rows, cols = self.shape
        p1, p2 = self.padded_shape
        size = self.extended.size
        grid = max(1, min(-(-size // self.block), 8 * self.grid))
        dims = (np.int32(rows), np.int32(cols), np.int32(p1), np.int32(p2))
        if not reordered:
            self._launch("reorder", (residual, self.fft_input, *dims, flags), grid=grid)
        self.plan_forward.fft(self.fft_input, self.spectrum, cufft.CUFFT_FORWARD)
        history = self.history if check is not None else self.s
        self._launch(
            "makhoul_spectrum",
            (self.spectrum, self.spectrum_out, self.inverse_symbol,
             *self.twiddles, np.int32(p1), np.int32(p2), flags,
             self.partials_rr if check is not None else self.absent_f64,
             np.int32(self.grid), self.s, history,
             np.float64(check.cg_relative_tolerance if check else 0),
             np.int32(check.iterations_per_round if check else 0)),
            grid=grid,
        )
        self.plan_inverse.fft(self.spectrum_out, self.extended, cufft.CUFFT_INVERSE)
        self._launch(
            "crop",
            (self.extended, residual, out,
             direction if direction is not None else out, *dims,
             rz_out if mode in (3, 4) else self.partials, self.counter, self.s,
             flags, np.int32(mode), np.int32(check is not None)),
        )

    def _select(
        self,
        x: cp.ndarray,
        ranks: tuple[int, int],
        *,
        mode: int,
        t: float = 0.0,
        out_t: cp.ndarray | None = None,
        device_ranks: bool = False,
    ) -> None:
        """Enqueue the radix selection of the values of ranks `ranks` in `x`.

        mode 0: their mean in the dtype of `x` to `out_t` (the median, for the two
        middle ranks); mode 1: the linear interpolation with weight `t` between
        them, in float64, to `self.select_out[0]`. With `device_ranks`, the ranks
        and `t` come from the device instead (see `percentile_ranks`).
        """
        n = x.size
        bits = 8 * x.dtype.itemsize
        grid = max(1, min(-(-n // (self.block * 8)), 2 * _multiprocessor_count()))
        for p in range(-(-bits // 11)):
            self._launch(
                "select_pass",
                (x, np.int32(n), np.int32(p), np.uint64(ranks[0]), np.uint64(ranks[1]),
                 self.select_state, self.select_hist, self.select_counter,
                 np.int32(mode), np.float64(t),
                 out_t if out_t is not None else self.offset, self.select_out,
                 np.int32(device_ranks)),
                grid=grid,
            )

    def _apply(
        self,
        p: cp.ndarray,
        out: cp.ndarray,
        mode: int,
        *,
        z: cp.ndarray | None = None,
        p_new: cp.ndarray | None = None,
    ) -> None:
        """Enqueue `out` = A p; see the `apply_operator` kernel for the modes.

        In the CG iterations (modes 1 and 2), with the deferred dot products, r . z
        comes from the latest partial sums (`self._rz_index`) and the ones before.
        """
        rows, cols = self.shape
        deferred_rz = mode != 0 and self.defer_rz
        latest = self.partials_rz[self._rz_index]
        before = self.partials_rz[1 - self._rz_index]
        self._launch(
            "apply_operator",
            (p, z if z is not None else p, p_new if p_new is not None else p,
             self.weights_x, self.weights_y, out, np.int32(rows), np.int32(cols),
             self.partials, self.counter, self.s, self.f, np.int32(mode),
             np.float64(np.finfo(self.dtype).tiny), latest, before,
             np.int32(self.grid if deferred_rz else 0),
             self.partials_pap if mode != 0 and self.defer_alpha else self.absent_f64),
        )

    def _cg_preamble(self, options: IRLSOptions) -> None:
        """Enqueue the start of a round's CG on `self.phi`."""
        rows, cols = self.shape
        n = np.int64(self.size)
        tol = np.float64(options.cg_relative_tolerance)
        phi = self.phi
        self._launch("reset_flags", (self.f,), grid=1, block=1)
        self._launch(
            "right_hand_side",
            (self.weights_x, self.target_x, self.weights_y, self.target_y, self.b,
             np.int32(rows), np.int32(cols), self.partials, self.counter, self.s),
        )
        self._launch(
            "center_and_norm",
            (self.b, n, self.partials, self.counter, self.s, self.f, self.history),
        )
        self._apply(phi, self.q, mode=0)
        self._launch(
            "initial_residual",
            (self.b, self.q, self.r, n, self.partials, self.counter, self.s, self.f,
             self.history, tol),
        )
        # z = M r; d = z; rz (as partial sums, into the first buffer, if deferred)
        self._direction = 0
        self._rz_index = 0
        self._precondition(
            self.r, self.z, self.f, mode=4 if self.defer_rz else 0,
            direction=self.directions[0], rz_out=self.partials_rz[0],
        )

    def _cg_iterations(
        self, count: int, options: IRLSOptions, *, first: bool,
    ) -> None:
        """Enqueue `count` CG iterations, then remove the mean if CG has iterated.

        Iterations after CG has stopped (converged, or at the limit) do nothing,
        except for the FFTs, which cannot be skipped from the device; the mean is
        removed only once CG has stopped. Except for
        the first iteration after the preamble (`first`), each iteration computes
        its direction from the previous one into the other direction buffer, and
        the deferred r . z alternates between its two buffers likewise. So `count`
        must be even for a graph that is replayed after itself.
        """
        _, cols = self.shape
        p1, p2 = self.padded_shape
        n = np.int64(self.size)
        tol = np.float64(options.cg_relative_tolerance)
        limit = np.int32(options.iterations_per_round)
        absent = self.absent_f64
        phi = self.phi
        for i in range(count):
            current = self.directions[self._direction]
            if first and i == 0:  # d = z, from the preamble
                self._apply(current, self.q, mode=1)  # q = A d; alpha
            else:  # d = beta d + z; q = A d; alpha
                self._direction = 1 - self._direction
                self._apply(
                    current, self.q, mode=2,
                    z=self.z, p_new=self.directions[self._direction],
                )
            d = self.directions[self._direction]
            latest_rz = self.partials_rz[self._rz_index]
            self._launch(
                "cg_update",
                (phi, d, self.r, self.q, n, self.partials, self.counter, self.s,
                 self.f, self.history, tol, limit, self.fft_input, np.int32(cols),
                 np.int32(p1), np.int32(p2),
                 self.partials_pap if self.defer_alpha else absent,
                 latest_rz if self.defer_rz else absent, np.int32(self.grid),
                 np.float64(np.finfo(self.dtype).tiny),
                 self.partials_rr if self.defer_rr else absent),
            )
            # z = M r; r . z: beta now, or the partial sums, into the other buffer
            self._precondition(
                self.r, self.z, self.f, mode=3 if self.defer_rz else 1,
                reordered=True, rz_out=self.partials_rz[1 - self._rz_index],
                check=options if self.defer_rr else None,
            )
            if self.defer_rz:
                self._rz_index = 1 - self._rz_index
        self._launch("mean", (phi, n, self.partials, self.counter, self.s, self.f))
        self._launch("subtract_mean_if_iterated", (phi, n, self.s, self.f))

    def _capture(self, enqueue: Callable[[], None]) -> cp.cuda.graph.Graph:
        """Capture what `enqueue` enqueues on the problem's stream as a graph.

        A device-wide synchronization (`cudaDeviceSynchronize`) in any thread
        while this captures invalidates the capture. Captures happen at the first
        solve of a problem and when the options change, one at a time
        (`_SETUP_LOCK`); other threads should synchronize their own streams
        instead.
        """
        with _SETUP_LOCK, self.stream:
            self.stream.begin_capture()
            try:
                enqueue()
            finally:
                graph = self.stream.end_capture()
        return graph

    def _memcpy(
        self, dst: cp.ndarray | np.ndarray, src: cp.ndarray | np.ndarray,
    ) -> None:
        """Enqueue a copy between device and/or pinned host arrays (capturable)."""
        if src.nbytes == 0:
            return
        kinds = {(True, True): 3, (True, False): 2, (False, True): 1}  # cudaMemcpyKind
        on_device = (isinstance(src, cp.ndarray), isinstance(dst, cp.ndarray))
        pointer = lambda a: a.data.ptr if isinstance(a, cp.ndarray) else a.ctypes.data  # noqa: E731
        cp.cuda.runtime.memcpyAsync(
            pointer(dst), pointer(src), src.nbytes, kinds[on_device], self.stream.ptr,
        )

    def _round_tail(self, slot: int) -> None:
        """Enqueue the end of a round, after its CG.

        Reweight (unless settled), round onto the branches, count the cuts, keep
        the weights with the candidate (all into `slot`), and copy the report to
        the host.
        """
        rows, cols = self.shape
        phase, branches, weights_x, weights_y = self.slots[slot]
        self._launch(
            "charbonnier_weights",
            (self.phi, self.target_x, self.target_y, self.edge_x, self.edge_y,
             self.round_params, self.weights_x, self.weights_y,
             np.int32(rows), np.int32(cols), self.f),
        )
        self._congruent_into(self.phi, self.wrapped, phase, branches)
        self._launch(
            "cuts",
            (branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
             self.invalid, np.int32(rows), np.int32(cols), self.partials,
             self.counter, self.s),
        )
        self._memcpy(weights_x, self.weights_x)
        self._memcpy(weights_y, self.weights_y)
        self._memcpy(self._host_flags, self.f)
        self._memcpy(self._host_history, self.history)
        self._memcpy(self._host_s, self.s)

    def _prepare_graph(self, options: IRLSOptions) -> None:
        """Capture a whole round as CUDA graphs.

        With up to `_GRAPH_ITERATIONS` CG iterations, one graph: the round's
        parameters from pinned memory, the CG, and the round's tail. With more, the
        first graph stops after that many iterations, `_graph_more` runs as many
        more, and `_graph_tails` runs the tail. The graphs with the tail exist once
        per candidate slot.
        """
        key = (options.iterations_per_round, options.cg_relative_tolerance)
        if self._graph_key == key:
            return
        if self._history_length < options.iterations_per_round + 1:
            self._history_length = options.iterations_per_round + 1
            with self.stream:
                self.history = cp.zeros(self._history_length, dtype=cp.float64)
            self._host_flags = _pinned(self.f.size, np.int32)
            self._host_history = _pinned(self.history.size, np.float64)
        chunk = min(options.iterations_per_round, _GRAPH_ITERATIONS)
        single = options.iterations_per_round <= chunk
        # A replayed chunk must leave the direction in the buffer it found it in:
        assert single or chunk % 2 == 0  # noqa: S101

        def first(slot: int) -> None:
            self._memcpy(self.round_params, self._host_params)
            self._cg_preamble(options)
            self._cg_iterations(chunk, options, first=True)
            if single:
                self._round_tail(slot)

        # Both first graphs end with the direction in the same buffer, where
        # `_graph_more` starts.
        self._graphs = tuple(self._capture(lambda s=s: first(s)) for s in (0, 1))
        self._graph_more = (
            None if single
            else self._capture(lambda: self._cg_iterations(chunk, options, first=False))
        )
        self._graph_tails = (
            None if single
            else tuple(self._capture(lambda s=s: self._round_tail(s)) for s in (0, 1))
        )
        self._graph = self._graphs[0]
        self._graph_key = key

    # -- the interface of _WeightedProblem -----------------------------------

    def run_round(
        self,
        phi: cp.ndarray,
        wrapped: cp.ndarray,  # noqa: ARG002 (filled in by `fill`)
        *,
        epsilon: float,
        at_end: bool,
        options: IRLSOptions,
        index: int,
    ) -> tuple[L1Round, _Candidate]:
        """Run one IRLS round on `phi` (in place); see `_WeightedProblem.run_round`.

        One graph launch and one synchronization (with up to `_GRAPH_ITERATIONS`
        CG iterations). The candidate's arrays are a candidate slot of the
        problem, valid until the next round unless the candidate is kept
        (:meth:`keep`), and until the end of the solve (:meth:`release`) if it is.
        """
        from .irls import L1Round, _Candidate  # noqa: PLC0415

        self._prepare_graph(options)
        # Read by the graph when it runs; safe to write, since the previous round
        # has synchronized.
        self._host_params[0] = epsilon * epsilon
        self._host_params[1] = float(at_end)
        slot = 1 if self._kept == 0 else 0
        with self._on_stream():
            own = phi.data.ptr == self.phi.data.ptr
            if not own:
                self._memcpy(self.phi, phi)
            self._graphs[slot].launch(self.stream)
            if self._graph_more is not None:
                while True:  # more iterations than one graph holds
                    self._memcpy(self._host_flags, self.f)
                    self.stream.synchronize()
                    if self._host_flags[_F_DONE]:
                        break
                    self._graph_more.launch(self.stream)
                self._graph_tails[slot].launch(self.stream)
            if not own:
                self._memcpy(phi, self.phi)
            self.stream.synchronize()
        flags = self._host_flags
        iterations = int(flags[_F_ITER])
        self._iterations = iterations
        if flags[_F_ZERO]:
            report = L1Round(iterations=0, residuals=[0.0], converged=True)
        else:
            report = L1Round(
                iterations=iterations,
                residuals=[float(x) for x in self._host_history[: iterations + 1]],
                converged=bool(flags[_F_DONE] == 1),
            )
        phase, branches, weights_x, weights_y = self.slots[slot]
        candidate = _Candidate(
            phase=phase,
            branches=branches,
            cut_turns=int(self._host_s[_S_TURNS]),
            cut_cost=float(self._host_s[_S_COST]),
            weights_x=weights_x,
            weights_y=weights_y,
            weights_epsilon=math.nan,  # set by the caller
            converged=False,  # set by the caller
            round_index=index,
        )
        return report, candidate

    def run_rounds(
        self,
        phi: cp.ndarray,
        wrapped: cp.ndarray,
        weights_epsilon: float,
        options: IRLSOptions | ADMMOptions,
        *,
        best: _Candidate,
    ) -> tuple[list[L1Round], _Candidate]:
        """Run the rounds of a solve; see `irls._irls_rounds`.

        IRLS runs `irls._irls_rounds`, one graph launch per round. ADMM runs its
        stop rule on the device (in `cuts_decide`), and a graph launch runs
        `_ADMM_CHUNK` rounds: one host synchronization per chunk.
        """
        from .irls import _MIN_IMPROVEMENT, L1Round, _Candidate, _irls_rounds  # noqa: PLC0415

        if self.method != "admm":
            return _irls_rounds(self, phi, wrapped, weights_epsilon, options, best=best)
        self._prepare_admm(options)
        eps = (
            options.epsilon_start if math.isinf(weights_epsilon)
            else max(options.epsilon_min, weights_epsilon * options.decay)
        )
        state = self._host_round_state
        state[:] = 0.0
        state[0], state[1], state[2] = best.cut_cost, best.cut_turns, -1.0
        state[5], state[6] = eps, best.weights_epsilon
        state[8] = options.rounds
        state[9] = 1e300 if options.stop_patience is None else options.stop_patience
        state[10], state[11] = options.stop_tolerance, options.epsilon_min
        state[12], state[13] = options.decay, _MIN_IMPROVEMENT
        state[16], state[17] = 1.0, 0.0  # the candidate slot, the best's slot
        self._host_params[0], self._host_params[1] = eps * eps, 0.0
        with self._on_stream():
            if phi.data.ptr != self.phi.data.ptr:
                self._memcpy(self.phi, phi)
            first = True
            while True:
                self._admm_graphs[first].launch(self.stream)
                self.stream.synchronize()
                first = False
                if state[4]:
                    break
        hist = self._host_round_hist
        rounds = [
            L1Round(iterations=options.iterations_per_round, residuals=[],
                    converged=True, cut_turns=int(hist[2 * i]),
                    cut_cost=float(hist[2 * i + 1]))
            for i in range(int(state[7]))
        ]
        best_slot = int(state[17])
        self._kept = best_slot
        if state[2] < 0:  # no round improved on the start
            return rounds, best
        phase, branches, weights_x, weights_y = self.slots[best_slot]
        return rounds, _Candidate(
            phase=phase, branches=branches,
            cut_turns=int(state[1]), cut_cost=float(state[0]),
            weights_x=weights_x, weights_y=weights_y,
            weights_epsilon=float(state[6]), converged=True, round_index=int(state[2]),
        )

    def _prepare_admm(self, options: ADMMOptions) -> None:
        """Set up ADMM's buffers, and capture the graphs of its rounds.

        One graph for the first chunk of rounds of a solve, one for later chunks.
        The layout of `round_state` is documented at `cuts_decide`.
        """
        key = (options.rounds, options.iterations_per_round, options.rho_scale)
        if self._admm_key == key:
            return
        rows, cols = self.shape
        with self.stream:
            self.rho = cp.full(1, options.rho_scale / math.pi, dtype=cp.float64)
            self.lam = tuple(
                (cp.zeros((rows, cols - 1), dtype=self.dtype),
                 cp.zeros((rows - 1, cols), dtype=self.dtype))
                for _ in range(2)
            )
            self.round_state = cp.zeros(32, dtype=cp.float64)
            self.round_hist = cp.zeros(2 * options.rounds, dtype=cp.float64)
        self._host_round_state = _pinned(32, np.float64)
        self._host_round_hist = _pinned(2 * options.rounds, np.float64)
        p1, p2 = self.padded_shape
        dims = (np.int32(rows), np.int32(cols), np.int32(p1), np.int32(p2))
        tau = self.dtype.type(2 * np.pi)
        n = np.int64(self.size)
        (ph0, br0, wx0, wy0), (ph1, br1, wx1, wy1) = self.slots
        spectrum_grid = max(1, min(-(-self.extended.size // self.block), 8 * self.grid))

        def one_round() -> None:
            for it in range(options.iterations_per_round):
                (lx, ly), (lx_new, ly_new) = self.lam[it % 2], self.lam[1 - it % 2]
                self._launch(
                    "admm_update",
                    (self.extended if it else self.phi, np.int32(it > 0),
                     self.target_x, self.target_y, self.edge_x, self.edge_y,
                     lx, ly, lx_new, ly_new, self.fft_input, *dims,
                     self.round_params, self.rho),
                    grid=_ADMM_GRID(self.size),
                )
                self.plan_forward.fft(
                    self.fft_input, self.spectrum, cufft.CUFFT_FORWARD,
                )
                self._launch(
                    "makhoul_spectrum",
                    (self.spectrum, self.spectrum_out, self.inverse_symbol,
                     *self.twiddles, np.int32(p1), np.int32(p2), self.f_free,
                     self.absent_f64, np.int32(self.grid), self.s, self.s,
                     np.float64(0), np.int32(0)),
                    grid=spectrum_grid,
                )
                self.plan_inverse.fft(
                    self.spectrum_out, self.extended, cufft.CUFFT_INVERSE,
                )
            # u of the last solve into phi (crop, mode 2: only z):
            self._launch(
                "crop",
                (self.extended, self.r, self.phi, self.phi, *dims, self.partials,
                 self.counter, self.s, self.f_free, np.int32(2), np.int32(0)),
            )
            self._launch(
                "circular_offset",
                (self.phi, self.wrapped, tau, self.difference, n, self.partials,
                 self.counter, self.offset),
            )
            self._launch(
                "branches_and_phase_slot",
                (self.phi, self.wrapped, self.offset, tau, br0, ph0, br1, ph1,
                 self.round_state, n),
            )
            self._launch(
                "cuts_decide",
                (br0, br1, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
                 self.invalid, np.int32(rows), np.int32(cols), self.partials,
                 self.counter, self.s,
                 self.round_state, self.round_hist, self.round_params),
            )

        def chunk(*, first: bool) -> None:
            if first:  # a new solve: its state, rho = scale * median(C) / pi, duals 0
                self._memcpy(self.round_params, self._host_params)
                self._memcpy(self.round_state, self._host_round_state)
                edges = self.edges.size
                if edges:
                    self._select(
                        self.edges, ((edges - 1) // 2, edges // 2), mode=1, t=0.5,
                    )
                    self._launch(
                        "scale_into",
                        (self.select_out, self.rho,
                         np.float64(options.rho_scale / math.pi)),
                        grid=1, block=1,
                    )
                for lam in self.lam:
                    for x in lam:
                        cp.cuda.runtime.memsetAsync(
                            x.data.ptr, 0, x.nbytes, self.stream.ptr)
            for _ in range(_ADMM_CHUNK(self.size)):
                one_round()
            # once the rounds have stopped, the weights of the best candidate:
            self._launch(
                "admm_weights",
                (self.round_state, ph0, ph1, self.target_x, self.target_y,
                 self.edge_x, self.edge_y, wx0, wy0, wx1, wy1,
                 np.int32(rows), np.int32(cols)),
            )
            self._memcpy(self._host_round_state, self.round_state)
            self._memcpy(self._host_round_hist, self.round_hist)

        self._admm_graphs = {
            first: self._capture(lambda first=first: chunk(first=first))
            for first in (True, False)
        }
        self._admm_key = key

    def _slot_of(self, candidate: _Candidate) -> int | None:
        for index, slot in enumerate(self.slots):
            if candidate.phase is slot[0]:
                return index
        return None

    def keep(self, candidate: _Candidate) -> _Candidate:
        """Keep `candidate` past the next round, which writes into the other slot."""
        slot = self._slot_of(candidate)
        if slot is not None:
            self._kept = slot
        return candidate

    def release(self, candidate: _Candidate) -> _Candidate:
        """Copy `candidate` out of the problem, whose slots later solves overwrite.

        The copy has NaN as the phase and 0 as the branch of the invalid pixels.
        """
        if self._slot_of(candidate) is None:
            return candidate
        self._kept = None
        with self._on_stream():
            arrays = {}
            for name in ("phase", "branches", "weights_x", "weights_y"):
                source = getattr(candidate, name)
                arrays[name] = cp.empty_like(source)
                self._memcpy(arrays[name], source)
            if self._invalid_pixels():
                self._launch(
                    "restore_invalid",
                    (self.invalid, arrays["phase"], arrays["branches"],
                     np.int64(self.size)),
                )
            return dataclasses.replace(candidate, **arrays)

    def _invalid_pixels(self) -> int:
        """Return the number of invalid pixels of the current problem."""
        if not self._invalid_on_host:  # set up without a start graph
            self._memcpy(self._host_s, self.s)
            self.stream.synchronize()
            self._invalid_on_host = True
        return int(self._host_s[_S_INVALID_PIXELS])

    def cuts(self, branches: cp.ndarray) -> tuple[int, float]:
        """Count the cut turns of `branches`, and their cost. Synchronizes."""
        rows, cols = self.shape
        with self._on_stream():
            self._launch(
                "cuts",
                (branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
                 self.invalid, np.int32(rows), np.int32(cols), self.partials,
                 self.counter, self.s),
            )
            s = self.s.get(stream=self.stream)
        if self._check_weights:
            self._check_weights = False
            if s[_S_INVALID] > 0:
                msg = "edge_weights must be finite and greater than zero"
                raise ValueError(msg)
        return int(s[_S_TURNS]), float(s[_S_COST])

    def reweight(
        self, phi: cp.ndarray, epsilon: float, *, unless_settled: bool = False,
    ) -> None:
        """Set the weights from the iterate `phi` at `epsilon`.

        With `unless_settled`, only if the last CG ran an iteration; decided on the
        device.
        """
        rows, cols = self.shape
        with self._on_stream():
            self._launch(
                "set_params",
                (self.direct_params, np.float64(epsilon * epsilon),
                 np.float64(unless_settled)),
                grid=1, block=1,
            )
            self._launch(
                "charbonnier_weights",
                (phi, self.target_x, self.target_y, self.edge_x, self.edge_y,
                 self.direct_params, self.weights_x, self.weights_y,
                 np.int32(rows), np.int32(cols), self.f),
            )

    def l2_solution(self) -> cp.ndarray:
        """Solve the unweighted problem, approximately; see `_WeightedProblem`."""
        rows, cols = self.shape
        with self._on_stream():
            out = cp.empty(self.shape, dtype=self.dtype)
            self._launch(
                "right_hand_side",
                (self.unit_x, self.target_x, self.unit_y, self.target_y, self.b,
                 np.int32(rows), np.int32(cols), self.partials, self.counter, self.s),
            )
            self._precondition(self.b, out, self.f_free, mode=2)
        return out

    def congruent(
        self, phi: cp.ndarray, wrapped: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Snap `phi` onto the branches of `wrapped`, see `irls._congruent`."""
        with self._on_stream():
            branches = cp.empty(self.shape, dtype=cp.int32)
            phase = cp.empty(self.shape, dtype=cp.float64)
            self._congruent_into(phi, wrapped, phase, branches)
        return phase, branches

    def _congruent_into(
        self,
        phi: cp.ndarray,
        wrapped: cp.ndarray,
        phase: cp.ndarray,
        branches: cp.ndarray,
    ) -> None:
        tau = self.dtype.type(2 * np.pi)
        n = np.int64(self.size)
        if self.method == "admm":
            # ADMM: around the circular mean instead of the median, in one pass
            self._launch(
                "circular_offset",
                (phi, wrapped, tau, self.difference, n, self.partials, self.counter,
                 self.offset),
            )
            self._launch(
                "branches_and_phase",
                (phi, wrapped, self.offset, tau, branches, phase, n),
            )
            return
        self._launch("wrapped_difference", (phi, wrapped, tau, self.difference, n))
        # The median, as np.median: the mean of the two middle values.
        self._select(
            self.difference, ((self.size - 1) // 2, self.size // 2),
            mode=0, out_t=self.offset,
        )
        self._launch(
            "branches_and_phase",
            (phi, wrapped, self.offset, tau, branches, phase, n),
        )

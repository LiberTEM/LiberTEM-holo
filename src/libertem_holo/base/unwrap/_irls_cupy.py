"""GPU (cupy) implementation of the weighted problem of the IRLS phase unwrapping.

PROTOTYPE. The counterpart of :class:`~libertem_holo.base.unwrap.irls._WeightedProblem`
for cupy input, with the same results: the same arithmetic in the same order per
pixel, and dot products summed in float64. At the image sizes of holography (~500²
to 2048²), the work per CG iteration is a few passes over one image, so kernel
launches and host synchronizations cost more than the arithmetic. Hence:

* The preconditioner (the unweighted Neumann Poisson solve) is a pair of DCTs, each
  computed as a real FFT of the reordered image with twiddles (J. Makhoul, 1980), on
  3-smooth padded lengths, with pre-built cuFFT plans.
* The CG scalars (alpha, beta, the residuals, the stop flag) stay on the device. Each
  kernel returns at once when the stop flag is set, so a fixed sequence of
  iterations does what the CPU loop does. It is captured as a CUDA graph, once per
  workspace, and replayed: one host synchronization per IRLS round. The setup of a
  solve and its starting point are one more graph (see
  :meth:`CupyWeightedProblem.cold_start`).
* Dot products are reduced in float64 and in a fixed order (per-block partial sums,
  then the last block to finish adds them up), so results do not vary from run to
  run. Some are deferred: the next kernel that needs them adds up the partial sums
  (see `_DEFER_RZ`).

The kernels are numba CUDA kernels (`_irls_numba_cuda`). Launching one from Python
costs ~40 us of host time, so they are only launched while capturing a graph.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import threading
import warnings
from collections import OrderedDict
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

from .irls import _is_smooth

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .admm import ADMMOptions
    from .irls import IRLSOptions, L1BranchesPrior, L1PhasePrior, L1Round, _Candidate

_BLOCK = 256
# CG iterations per captured graph: the default `iterations_per_round` fits in one.
_GRAPH_ITERATIONS = 16  # even: see `_cg_iterations`
# Grid of the grid-stride kernels: at most 8 blocks of 256 threads per
# multiprocessor, which is full occupancy (4 left the streaming kernels at 45 %
# of DRAM bandwidth, measured with Nsight Compute; 8 made a CG round 12 % faster
# at 2048^2, more made the reductions slower), and at least 2 elements per
# thread, which keeps the reduction tails of small images short.
_THREADS_PER_SM = 2048  # the most an SM holds at once
_MIN_PER_THREAD = 2
# PROTOTYPE switches: which CG dot products are added up from their partial sums
# by the kernels that need them, instead of by the last block of the kernel that
# computes them. That takes the last block's serial tail (~0.5 us) off the
# critical path, which gains ~4 % at ~500^2 for all three, but every block of the
# consuming kernels then adds up all partial sums, which costs ~4 % at 2048^2
# (measured on the GH200). So only up to `_DEFER_MAX_GRID` blocks.
_DEFER_RZ = True  # r . z, in apply_operator (beta) and cg_update (alpha)
_DEFER_ALPHA = True  # p . Ap, in cg_update
_DEFER_RR = True  # r . r, in makhoul_spectrum (the convergence check)
_DEFER_MAX_GRID = 512
# EXPERIMENT: rounds of a solve that use the Poisson (Makhoul) preconditioner; the
# later ones use the Jacobi preconditioner. None: all rounds Poisson.
# ADMM (`admm.ADMMOptions`): the grid of `admm_update` (no reductions, so one
# pixel per thread), and the rounds per graph launch: 2 on small images (a round
# costs little more than a host round trip, ~25 us), else 1 (a round after the
# stop would be wasted work).
_ADMM_GRID = (lambda n: max(1, -(-n // _BLOCK)))  # noqa: E731
_ADMM_CHUNK = (lambda n: 2 if n <= 512 * 512 else 1)  # noqa: E731
_POISSON_ROUNDS: int | None = (
    int(os.environ["IRLS_POISSON_ROUNDS"])
    if "IRLS_POISSON_ROUNDS" in os.environ else None
)

# Indices into the double scalars `s` on the device:
(_S_NORM, _S_RZ, _S_ALPHA, _S_BETA, _S_MEAN, _S_TURNS, _S_COST, _S_LARGEST, _S_INVALID,
 _S_NAN) = range(10)
# Indices into the int flags `f` on the device:
_F_DONE, _F_ITER, _F_ITERATED, _F_ZERO, _F_PENDING = range(5)


def _fft_plans(extended: cp.ndarray, spectrum: cp.ndarray) -> tuple:
    """Return new cuFFT plans for the real FFT pair on the extended grid.

    Each workspace has its own: a plan has a work area for its intermediate
    results, which every execution of the plan uses, and its stream is set at each
    call. Shared between workspaces, and so between threads, concurrent solves of
    the same shape would overwrite each other's intermediate results. A plan pair
    costs ~10 ms to create, once per workspace.
    """
    return (
        get_fft_plan(extended, value_type="R2C"),
        get_fft_plan(spectrum, shape=extended.shape, value_type="C2R"),
    )


# PROTOTYPE switch, for comparisons with the CPU:
PADDED_LENGTH: Callable[[int], int] | None = None  # None: irls._padded_length

_local = threading.local()
_CACHE_SIZE = 2


def problem_for(
    wrapped: cp.ndarray,
    edge_weights: tuple[cp.ndarray, cp.ndarray] | None,
    *,
    deferred: bool = False,
    method: str = "irls",
) -> CupyWeightedProblem:
    """Return a problem for `wrapped`, reusing a workspace of the same shape.

    Buffers, FFT plans and the captured CUDA graph only depend on the shape and the
    dtype, so a series of same-sized inputs reuses them. Workspaces are kept per
    thread (two concurrent solves must not share one), for the last
    :code:`_CACHE_SIZE` shapes. With `deferred`, the input is only copied into the
    workspace, and :meth:`CupyWeightedProblem.cold_start` or
    :meth:`~CupyWeightedProblem.warm_start` sets the problem up, which must come
    next. Workspaces are kept per `method` ("irls" or "admm").
    """
    problem = _workspace(wrapped.shape, wrapped.dtype, method)
    if deferred:
        problem.stage_phase(wrapped, edge_weights)
    else:
        problem.fill(wrapped, edge_weights)
    return problem


def problem_for_wave(
    wave: cp.ndarray, floor: float, percentile: float, *, deferred: bool = False,
    method: str = "irls",
) -> CupyWeightedProblem:
    """Return a problem for the phase of `wave`, with edge weights from its amplitude.

    See :func:`problem_for` and :meth:`CupyWeightedProblem.fill_from_wave`. With
    `deferred`, the wave is only copied into the workspace, and the problem is set
    up by :meth:`CupyWeightedProblem.cold_start` or
    :meth:`~CupyWeightedProblem.warm_start`, which must come next.
    """
    dtype = np.float32 if wave.dtype == np.complex64 else np.float64
    wave = cp.ascontiguousarray(wave, dtype=np.result_type(dtype, np.complex64))
    problem = _workspace(wave.shape, np.dtype(dtype), method)
    if deferred:
        problem.stage_wave(wave, floor, percentile)
    else:
        problem.fill_from_wave(wave, floor, percentile)
    return problem


def _workspace(
    shape: tuple[int, int], dtype: np.dtype, method: str = "irls",
) -> CupyWeightedProblem:
    """Return the cached workspace for `shape` and `dtype` (see :func:`problem_for`)."""
    cache = getattr(_local, "cache", None)
    if cache is None:
        cache = _local.cache = OrderedDict()
    key = (
        shape, np.dtype(dtype).str, cp.cuda.Device().id, method,
        PADDED_LENGTH, _BLOCK, _MIN_PER_THREAD,
        _DEFER_RZ, _DEFER_ALPHA, _DEFER_RR, _DEFER_MAX_GRID,
    )
    problem = cache.pop(key, None)
    if problem is None:
        problem = CupyWeightedProblem(
            shape, dtype, padded_length=PADDED_LENGTH, method=method,
        )
    cache[key] = problem
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)
    return problem


def _gpu_padded_length(length: int) -> int:
    """Return the smallest even 3-smooth number that is at least `length`.

    cuFFT is fastest on these lengths. Lengths with factors of 5 or 7, which the
    CPU version allows, are much slower with cuFFT for some sizes (500 against
    512, 1008 against 1024), and measured slower overall, padding included.
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
    """The weighted problem of one IRLS round, on the GPU.

    Same interface as :class:`~libertem_holo.base.unwrap.irls._WeightedProblem`.
    All arrays are cupy arrays. Use :func:`problem_for` to get one.

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

            # CG work arrays; the graph runs on `phi`, see `solve`:
            self.phi, self.b, self.q, self.r, self.z = (image() for _ in range(5))
            # The CG direction, in two buffers: an iteration computes the next one
            # from the current one (see `apply_operator`, mode 2).
            self.directions = (image(), image())
            self._direction = 0
            self.difference = image()
            # Inputs at fixed addresses, for the captured graphs:
            self.wrapped = image()
            self.wave = None  # a staged wave (see `stage_wave`)
            self.prior_phase = None  # a staged prior (see `warm_start`)
            self.prior_branches = None
            self.start_params = cp.zeros(2, dtype=cp.float64)  # eps^2 of a prior
            self.l2 = image()  # the L2 solution of the cold start
            # eps^2, unless settled (and unused):
            self.round_params = cp.zeros(3, dtype=cp.float64)
            self.direct_params = cp.zeros(2, dtype=cp.float64)  # the same, ungraphed
            # Two slots for a round's candidate result (phase, branches, weights):
            # a round writes into the slot that does not hold the best result so
            # far, so that keeping a candidate needs no copy (see `keep`).
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
            # The radix selection (see `select_pass`):
            self.select_state = cp.zeros(4, dtype=cp.uint64)
            self.select_hist = cp.zeros(2 * 2048, dtype=cp.uint32)
            self.select_counter = cp.zeros(1, dtype=cp.uint32)
            self.select_out = cp.zeros(3, dtype=cp.float64)
            self.offset = cp.zeros(1, dtype=self.dtype)
        if self.extended.size >= 2**31:
            msg = f"image too large for the GPU solver: {shape}"
            raise ValueError(msg)
        self._init_host_state()

    def _set_deferrals(self) -> None:
        """Which dot products to defer, for this grid (see `_DEFER_RZ`)."""
        small = self.grid <= _DEFER_MAX_GRID
        self.defer_rz = _DEFER_RZ and small
        self.defer_alpha = _DEFER_ALPHA and small
        self.defer_rr = _DEFER_RR and small

    def _init_host_state(self) -> None:
        """Pinned host buffers, captured graphs, and the state of a solve."""
        self._host_s = _pinned(self.s.size, np.float64)
        self._host_params = _pinned(3, np.float64)
        self._admm_graphs: dict[bool, cp.cuda.graph.Graph] = {}
        self._admm_key = None
        self._history_length = 0
        self._graph = None
        self._graph_key = None
        self._graph_sets: dict[tuple, tuple] = {}
        self._kind = "poisson"  # the preconditioner of the current round
        self._start_graphs: dict[tuple, cp.cuda.graph.Graph] = {}
        self._host_start_params = _pinned(2, np.float64)
        self._check_weights = False
        self._staged = None
        self._iterations = 0

    def _setup_preconditioner(self, padded_length: Callable[[int], int]) -> None:
        """Set up the Neumann Poisson solve on the zero-padded grid (P1, P2).

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
        """Set up the problem for `wrapped`, in this workspace."""
        with self._on_stream():
            if edge_weights is not None:
                self.edge_x[...] = edge_weights[0]
                self.edge_y[...] = edge_weights[1]
                self._memcpy(self.weights_x, self.edge_x)
                self._memcpy(self.weights_y, self.edge_y)
            self._prepare_targets(
                cp.ascontiguousarray(wrapped, dtype=self.dtype),
                unit_edges=edge_weights is None,
            )
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
        virtual = (n - 1) * (percentile / 100)  # numpy's "linear" method
        low = math.floor(virtual)
        self._launch(
            "wave_phase_amplitude",
            (wave, self.wrapped, amplitude, np.int32(n), self.partials,
             self.counter, self.s),
        )
        self._select(amplitude, (low, min(low + 1, n - 1)), mode=1, t=virtual - low)
        for mode in (0, 1):
            self._launch(
                "amplitude_edges",
                (amplitude, self.select_out, np.float64(floor), np.int32(rows),
                 np.int32(cols), np.int32(mode), self.partials, self.counter,
                 self.s, self.edge_x, self.edge_y, self.weights_x, self.weights_y),
            )
        self._prepare_targets(self.wrapped, unit_edges=False)

    def stage_wave(self, wave: cp.ndarray, floor: float, percentile: float) -> None:
        """Copy `wave` into the workspace; the start graph sets up from it."""
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
        """Copy the wrapped phase (and edge weights) into the workspace."""
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

        After staging, as one CUDA graph: the setup, the rounded L2 solution as
        the first candidate (into slot 0), and the starting point. One
        synchronization.
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

        After staging, as one CUDA graph: the setup, the starting phase from the
        prior, its weights, and it rounded as the first candidate. The epsilon of
        the weights comes from pinned memory, so one graph serves every prior.
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
        """Launch the start graph of `kind` for the staged input; its candidate."""
        from .irls import _Candidate  # noqa: PLC0415

        staged, self._staged = self._staged, None
        key = (staged, kind, options.epsilon_start if kind == "zero" else None)
        graph = self._start_graphs.get(key)
        if graph is None:
            graph = self._capture(lambda: self._enqueue_start(staged, kind, options))
            self._start_graphs[key] = graph
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

        `staged` is ("wave", floor, percentile) or ("phase", edge weights given);
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
            self._prepare_targets(self.wrapped, unit_edges=not staged[1])
        phase, branches, weights_x, weights_y = self.slots[0]

        def candidate(phi: cp.ndarray) -> None:
            self._congruent_into(phi, self.wrapped, phase, branches)
            self._launch(
                "cuts",
                (branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
                 *dims, self.partials, self.counter, self.s),
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
                # From zero, with its weights (used only if there are cuts):
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

    def _prepare_targets(self, wrapped: cp.ndarray, *, unit_edges: bool) -> None:
        rows, cols = self.shape
        self._launch(
            "prepare_targets",
            (wrapped, self.wrapped, self.target_x, self.target_y, self.jumps_x,
             self.jumps_y, self.edge_x, self.edge_y, self.weights_x, self.weights_y,
             np.int32(rows), np.int32(cols), np.int32(unit_edges)),
        )

    # -- launches ------------------------------------------------------------

    @contextlib.contextmanager
    def _on_stream(self) -> Iterator[None]:
        """Run on our stream, ordered after and before the caller's stream.

        Our stream first waits for the caller's, whose arrays we are about to use.
        At the end, the caller's stream waits for ours: that makes our results
        safe to use there, and it keeps the memory pool from handing memory that
        the caller frees after the call to a new array while our stream is still
        reading it. Neither costs a host synchronization.
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
        with warnings.catch_warnings():
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
            msg = "the numba kernels are compiled for their own block size"
            raise ValueError(msg)
        self.kernels = _irls_numba_cuda.kernels(self.dtype)
        self._numba_stream = numba_cuda.external_stream(self.stream.ptr)
        self._numba_arrays: dict[tuple, object] = {}
        self._numba_cuda = numba_cuda
        # An absent (optional) array argument:
        self.absent_f64 = cp.empty(0, dtype=cp.float64)

    def _numba_args(self, args: tuple) -> tuple:
        """Return the arguments for a numba kernel.

        Arrays are wrapped once, without the stream synchronization of the array
        interface, which graph capture forbids.
        """
        out = []
        for arg in args:
            if isinstance(arg, cp.ndarray):
                key = (arg.data.ptr, arg.shape, arg.dtype.str)
                if key not in self._numba_arrays:
                    self._numba_arrays[key] = self._numba_cuda.as_cuda_array(
                        arg.ravel(), sync=False,
                    )
                out.append(self._numba_arrays[key])
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
        if self._kind == "sgs":
            self._launch(
                "sgs_black",
                (residual, self.weights_x, self.weights_y, out, np.int32(rows),
                 np.int32(cols), flags),
            )
            self._launch(
                "sgs_red",
                (residual, self.weights_x, self.weights_y, out,
                 direction if direction is not None else out,
                 np.int32(rows), np.int32(cols),
                 rz_out if mode in (3, 4) else self.partials, self.counter, self.s,
                 flags, np.int32(mode)),
            )
            return
        if self._kind == "jacobi":
            self._launch(
                "jacobi",
                (residual, self.weights_x, self.weights_y, out,
                 direction if direction is not None else out,
                 np.int32(rows), np.int32(cols),
                 rz_out if mode in (3, 4) else self.partials, self.counter, self.s,
                 flags, np.int32(mode)),
            )
            return
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
             flags, np.int32(mode), np.int32(check is not None),
             self.weights_x, self.weights_y, self.round_params,
             np.int32(self._kind == "additive")),
        )

    def _select(
        self,
        x: cp.ndarray,
        ranks: tuple[int, int],
        *,
        mode: int,
        t: float = 0.0,
        out_t: cp.ndarray | None = None,
    ) -> None:
        """Enqueue the radix selection of the values of ranks `ranks` in `x`.

        mode 0: their mean in the dtype of `x` to `out_t` (the median, for the two
        middle ranks); mode 1: the linear interpolation with weight `t` between
        them, in float64, to `self.select_out[0]`.
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
                 out_t if out_t is not None else self.offset, self.select_out),
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
        """Enqueue `count` CG iterations, then the mean removal once CG stopped.

        Iterations after CG has stopped (converged, or at the limit) do nothing,
        except for the FFTs, which cannot be skipped from the device. Each
        iteration but the `first` after the preamble computes its direction from
        the last one, in the other direction buffer, and the deferred r . z
        alternates between its two buffers likewise: `count` must be even for a
        graph that is replayed after itself.
        """
        _, cols = self.shape
        p1, p2 = self.padded_shape
        n = np.int64(self.size)
        tol = np.float64(options.cg_relative_tolerance)
        limit = np.int32(options.iterations_per_round)
        absent = self.absent_f64
        poisson = self._kind in ("poisson", "additive")
        defer_rr = self.defer_rr and poisson
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
                 self.partials_rr if defer_rr else absent),
            )
            # z = M r; r . z: beta now, or the partial sums, into the other buffer
            self._precondition(
                self.r, self.z, self.f, mode=3 if self.defer_rz else 1,
                reordered=poisson, rz_out=self.partials_rz[1 - self._rz_index],
                check=options if defer_rr else None,
            )
            if self.defer_rz:
                self._rz_index = 1 - self._rz_index
        self._launch("mean", (phi, n, self.partials, self.counter, self.s, self.f))
        self._launch("subtract_mean_if_iterated", (phi, n, self.s, self.f))

    def _capture(self, enqueue: Callable[[], None]) -> cp.cuda.graph.Graph:
        """Capture what `enqueue` enqueues on the workspace's stream as a graph.

        A device-wide synchronization (`cudaDeviceSynchronize`) in any thread
        while this captures invalidates the capture. That happens only while a
        workspace is set up (the first solve of a shape in a thread); workers
        should synchronize their own streams instead.
        """
        with self.stream:
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
             np.int32(rows), np.int32(cols), self.partials, self.counter, self.s),
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
        more, and the tail graph the tail. The graphs with the tail exist once per
        candidate slot.
        """
        key = (options.iterations_per_round, options.cg_relative_tolerance, self._kind)
        if self._graph_key == key:
            return
        if key in self._graph_sets:
            (self._graphs, self._graph_more, self._graph_tails) = self._graph_sets[key]
            self._graph, self._graph_key = self._graphs[0], key
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

        # Both first graphs end with the direction in the same buffer, where the
        # more graph starts.
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
        self._graph_sets[key] = (self._graphs, self._graph_more, self._graph_tails)

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

        One graph launch and one synchronization. The candidate's arrays are a
        workspace slot, valid until the next round unless it is kept (:meth:`keep`),
        and until the end of the solve (:meth:`release`) if it is.
        """
        from .irls import L1Round, _Candidate  # noqa: PLC0415

        self._kind = (
            os.environ.get("IRLS_GLOBAL", "poisson")
            if _POISSON_ROUNDS is None or index < _POISSON_ROUNDS
            else os.environ.get("IRLS_LOCAL", "jacobi")
        )
        # EXPERIMENT: CG iterations per kind of round
        override = os.environ.get(
            "IRLS_POISSON_ITERS" if self._kind in ("poisson", "additive")
            else "IRLS_LOCAL_ITERS",
        )
        if override:
            options = dataclasses.replace(options, iterations_per_round=int(override))
        self._prepare_graph(options)
        self._kind = "poisson"  # the default, e.g. for the start graphs
        # Read by the graph when it runs; the last round has synchronized.
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
        """Buffers and the graphs of ADMM's rounds: a solve's first chunk, and later.

        The round state `round_state` is documented at `cuts_decide`.
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
                 self.counter, self.s, self.f_free, np.int32(2), np.int32(0),
                 self.weights_x, self.weights_y, self.round_params, np.int32(0)),
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
                 np.int32(rows), np.int32(cols), self.partials, self.counter, self.s,
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
            # once the rounds have stopped, the result's weights:
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
        """Keep a candidate past the next round: protect its slot from it."""
        slot = self._slot_of(candidate)
        if slot is not None:
            self._kept = slot
        return candidate

    def release(self, candidate: _Candidate) -> _Candidate:
        """Copy the result out of the workspace, which later solves reuse."""
        if self._slot_of(candidate) is None:
            return candidate
        self._kept = None
        with self._on_stream():
            arrays = {}
            for name in ("phase", "branches", "weights_x", "weights_y"):
                source = getattr(candidate, name)
                arrays[name] = cp.empty_like(source)
                self._memcpy(arrays[name], source)
            return dataclasses.replace(candidate, **arrays)

    def cuts(self, branches: cp.ndarray) -> tuple[int, float]:
        """Count the cut turns of `branches`, and their cost. Synchronizes."""
        rows, cols = self.shape
        with self._on_stream():
            self._launch(
                "cuts",
                (branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
                 np.int32(rows), np.int32(cols), self.partials, self.counter, self.s),
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
        """Set the weights from `phi` at `epsilon`.

        With `unless_settled`, only if the last :meth:`solve` ran an iteration;
        decided on the device.
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
            # the circular mean instead of the median, in one pass
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

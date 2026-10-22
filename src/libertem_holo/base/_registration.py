"""Pairwise graph registration of frame stacks.

Instead of registering every frame against one reference, the shift of many
frame *pairs* is measured, each measurement is weighted by its
peak-to-sidelobe ratio (PSR), and a robust weighted least-squares fit gives
one drift vector per frame.

This module is private; its public names are exported by
:mod:`libertem_holo.base.align`.
"""

from __future__ import annotations

import contextlib
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numba
import numpy as np
from libertem.common.threading import set_num_threads
from tqdm import tqdm

from libertem_holo.base.utils import to_cpu

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from libertem_holo.base.align import Correlator
    from libertem_holo.base.reconstr import XPType

REFERENCE_NODE = -1
"""Value of :attr:`StackRegistrationResult.first` for edges that measure a
frame against the averaged reference of the anchored topology."""


@dataclass(frozen=True)
class GraphSolverParams:
    """Parameters of :func:`solve_graph`.

    Parameters
    ----------
    sidelobe_exclusion
        Half-width of the square around the correlation peak that is excluded
        from the sidelobe statistics of the PSR (3: a 7x7 square).
    weight_exponent
        The base weight of an edge is
        :code:`clip((psr / median(psr)) ** weight_exponent, *weight_clip)`.
    weight_clip
        Bounds of the base weight.
    irls_iterations
        Number of weighted least-squares solves; after each one the robust
        weights are updated from the residuals.
    cutoff_floor
        Lower bound of the residual cutoff, in pixels.
    cutoff_sigmas
        Residuals above :code:`median + cutoff_sigmas * mad_scale * MAD` are
        down-weighted (Huber-like: weight :code:`cutoff / residual`).
    mad_scale
        Factor that turns the median absolute deviation into a standard
        deviation estimate for Gaussian data.

    """

    sidelobe_exclusion: int = 3
    weight_exponent: float = 2.0
    weight_clip: tuple[float, float] = (0.1, 10.0)
    irls_iterations: int = 8
    cutoff_floor: float = 0.1
    cutoff_sigmas: float = 2.5
    mad_scale: float = 1.4826

    def __post_init__(self) -> None:
        if self.sidelobe_exclusion < 0:
            msg = "sidelobe_exclusion must be >= 0"
            raise ValueError(msg)
        if self.irls_iterations < 1:
            msg = "irls_iterations must be >= 1"
            raise ValueError(msg)
        low, high = self.weight_clip
        if not 0 < low <= high:
            msg = "weight_clip must satisfy 0 < low <= high"
            raise ValueError(msg)


@dataclass(frozen=True)
class StarRegistration:
    """Register every frame against the first one.

    One measurement per frame, so there is nothing to weigh or to solve;
    this is what :func:`~libertem_holo.base.convenience.reconstruct_stack`
    does by default.
    """

    topology: ClassVar[str] = "star"


@dataclass(frozen=True)
class AnchoredRegistration:
    """Register batches of frames pairwise against an anchored reference.

    The first `batch_size` frames are registered among each other (all
    pairs), aligned and averaged into a reference; every later batch is
    registered against that reference and among itself. Each batch is solved
    as one PSR-weighted, robust least-squares problem, see
    :func:`solve_graph`.

    Parameters
    ----------
    batch_size
        Number of frames registered jointly. The number of pairs per batch
        grows with its square. With 1, every frame is registered against
        frame 0 alone, which is :class:`StarRegistration`.
    solver
        Parameters of the graph solver and the PSR

    """

    batch_size: int = 16
    solver: GraphSolverParams = field(default_factory=GraphSolverParams)
    topology: ClassVar[str] = "anchored"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            msg = "batch_size must be at least 1"
            raise ValueError(msg)


Registration = StarRegistration | AnchoredRegistration
RegistrationArg = Registration | Literal["star", "anchored"] | None
"""What :func:`~libertem_holo.base.convenience.reconstruct_stack` accepts as
`registration`: the name of a topology with its defaults, an instance with
parameters, or None for the default."""


def resolve_registration(value: RegistrationArg) -> Registration:
    """Turn a `registration` argument into a :data:`Registration` instance.

    :code:`None` and :code:`"star"` give :class:`StarRegistration`,
    :code:`"anchored"` gives :class:`AnchoredRegistration` with its defaults.
    """
    if value is None or value == "star":
        return StarRegistration()
    if value == "anchored":
        return AnchoredRegistration()
    if isinstance(value, (StarRegistration, AnchoredRegistration)):
        return value
    msg = (
        f"unknown registration {value!r}: expected 'star', 'anchored', "
        "StarRegistration or AnchoredRegistration"
    )
    raise ValueError(msg)


# frames per chunk of the star topology; only bounds the memory for the
# prepared inputs, the results don't depend on it
_STAR_CHUNK = 16


@dataclass
class StackRegistrationResult:
    """Drifts of a stack and the pairwise measurements they were solved from.

    One entry per measured pair ("edge") in the per-edge arrays. The shift of
    an edge is what the correlator measured: applied to frame `second`, it
    aligns it with frame `first`. `first` is :data:`REFERENCE_NODE` for edges
    against the averaged reference of the anchored topology.

    """

    drifts: np.ndarray
    """Shape (N, 2): the shift to apply to each frame to align it with
    frame 0 (or with the reference)."""
    first: np.ndarray
    second: np.ndarray
    shift: np.ndarray
    psr: np.ndarray
    weight: np.ndarray
    """The weight of each edge in the final least-squares solve."""
    residual: np.ndarray
    """The residual of each edge after the final solve, in pixels."""

    def to_metadata(self) -> dict[str, list]:
        """Return the per-edge arrays as JSON-serializable lists."""
        return {
            "first": self.first.tolist(),
            "second": self.second.tolist(),
            "shift": self.shift.tolist(),
            "psr": self.psr.tolist(),
            "weight": self.weight.tolist(),
            "residual": self.residual.tolist(),
        }


@numba.njit(cache=True, nogil=True)
def _psr_cpu(corrmap: np.ndarray, exclusion: int) -> float:  # noqa: C901
    """Peak-to-sidelobe ratio of a surface, see :func:`peak_to_sidelobe_ratio`."""
    height, width = corrmap.shape
    best = -math.inf
    best_y = 0
    best_x = 0
    for y in range(height):
        for x in range(width):
            value = corrmap[y, x]
            if value > best:
                best = value
                best_y = y
                best_x = x
    total = 0.0
    count = 0
    for y in range(height):
        dy = abs(y - best_y)
        dy = min(dy, height - dy)
        for x in range(width):
            dx = abs(x - best_x)
            dx = min(dx, width - dx)
            if dy > exclusion or dx > exclusion:
                total += corrmap[y, x]
                count += 1
    if count == 0:
        return 0.0
    mean = total / count
    total_sq = 0.0
    for y in range(height):
        dy = abs(y - best_y)
        dy = min(dy, height - dy)
        for x in range(width):
            dx = abs(x - best_x)
            dx = min(dx, width - dx)
            if dy > exclusion or dx > exclusion:
                deviation = corrmap[y, x] - mean
                total_sq += deviation * deviation
    std = math.sqrt(total_sq / count)
    return (best - mean) / max(std, 1e-15)


def _psr_device(corrmap: np.ndarray, exclusion: int, xp: XPType) -> np.ndarray:
    """Vectorized :func:`peak_to_sidelobe_ratio` for any array module.

    Returns a 0-d `xp` array and never touches the host: no boolean-mask
    indexing (which needs the mask's count on the host) and no index arrays
    of the surface's size. Statistics are accumulated in float64.
    """
    height, width = corrmap.shape
    values = corrmap.astype(np.float64, copy=False)
    flat_index = xp.argmax(values)
    peak_y = flat_index // width
    peak_x = flat_index % width
    dy = abs(xp.arange(height) - peak_y)
    dy = xp.minimum(dy, height - dy)
    dx = abs(xp.arange(width) - peak_x)
    dx = xp.minimum(dx, width - dx)
    mask = (dy[:, None] > exclusion) | (dx[None, :] > exclusion)
    count = mask.sum()
    denominator = xp.maximum(count, 1)
    mean = xp.where(mask, values, 0.0).sum() / denominator
    variance = xp.where(mask, (values - mean) ** 2, 0.0).sum() / denominator
    # the peak value is the maximum (argmax takes the first one)
    psr = (values.max() - mean) / xp.maximum(xp.sqrt(variance), 1e-15)
    return xp.where(count > 0, psr, 0.0)


def _psr_xp(corrmap: np.ndarray, exclusion: int, xp: XPType) -> float:
    """:func:`_psr_device` as a Python float (one device-to-host transfer)."""
    return float(to_cpu(_psr_device(corrmap, exclusion, xp)))


def peak_to_sidelobe_ratio(corrmap: np.ndarray, exclusion: int = 3) -> float:
    """Peak-to-sidelobe ratio (PSR) of a correlation surface.

    How far the peak stands out of the rest of the surface:
    :code:`(peak - mean(sidelobes)) / std(sidelobes)`, where the sidelobes
    are all pixels outside the square of half-width `exclusion` around the
    peak (wrapping around the edges). A high PSR means an unambiguous
    correlation.

    Parameters
    ----------
    corrmap
        The correlation surface, e.g. :attr:`RegResult.corrmap`; numpy or cupy
    exclusion
        Half-width of the excluded square

    """
    if isinstance(corrmap, np.ndarray):
        return float(_psr_cpu(np.ascontiguousarray(corrmap), exclusion))
    import cupy as cp  # noqa: PLC0415

    return _psr_xp(corrmap, exclusion, cp)


def star_pairs(count: int) -> list[tuple[int, int]]:
    """Pairs :code:`(0, i)` for nodes :code:`1 .. count - 1`."""
    return [(0, i) for i in range(1, count)]


def all_pairs(count: int) -> list[tuple[int, int]]:
    """All pairs :code:`(a, b)` with :code:`a < b` of `count` nodes."""
    return [(a, b) for a in range(count) for b in range(a + 1, count)]


def anchored_pairs(count: int) -> list[tuple[int, int]]:
    """Node 0 against each of `count` frames, then all pairs among the frames."""
    return star_pairs(count + 1) + [
        (a, b) for a in range(1, count + 1) for b in range(a + 1, count + 1)
    ]


def solve_graph(  # noqa: PLR0917
    first: np.ndarray,
    second: np.ndarray,
    shift: np.ndarray,
    psr: np.ndarray,
    count: int,
    params: GraphSolverParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve per-node drifts from pairwise shift measurements.

    Node 0 is pinned at zero. Each edge is weighted by its PSR relative to
    the median PSR (relative to the maximum if the median is zero), and
    iteratively reweighted (IRLS) by its residual, so that measurements that
    disagree with the rest of the graph lose influence. Both axes are solved
    jointly, with a common residual.

    Parameters
    ----------
    first, second
        Node indices of each edge, :code:`0 <= first < second < count`
    shift
        Shape (E, 2): the measured shift that aligns node `second` with
        node `first`
    psr
        Shape (E,): the PSR of each measurement
    count
        Number of nodes
    params
        Solver parameters, default :class:`GraphSolverParams`

    Returns
    -------
    drifts
        Shape (count, 2), :code:`drifts[0] == 0`
    weight
        Shape (E,): the weight of each edge in the final solve
    residual
        Shape (E,): the residual of each edge after the final solve

    """
    if params is None:
        params = GraphSolverParams()
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    measured = np.asarray(shift, dtype=np.float64).reshape(-1, 2)
    psr = np.asarray(psr, dtype=np.float64)
    edges = len(first)
    if count <= 1 or edges == 0:
        return np.zeros((count, 2)), np.ones(edges), np.zeros(edges)
    if (first < 0).any() or (second >= count).any() or (first >= second).any():
        msg = "edges must satisfy 0 <= first < second < count"
        raise ValueError(msg)

    design = np.zeros((edges, count - 1))
    rows = np.arange(edges)
    has_first = first > 0
    design[rows[has_first], first[has_first] - 1] = -1.0
    design[rows, second - 1] = 1.0

    # normalize by the median PSR; if that is zero (more than half of the
    # edges have no peak at all), by the largest, so that the edges with a
    # peak still outweigh the others; all zero: equal weights
    scale = np.median(psr)
    if scale <= 0:
        scale = psr.max()
    if scale > 0:
        base_weights = np.clip(
            (psr / scale) ** params.weight_exponent, *params.weight_clip,
        )
    else:
        base_weights = np.ones(edges)
    robust_weights = np.ones(edges)
    weights = base_weights
    residual = np.zeros(edges)
    solution = np.zeros((count - 1, 2))
    for _ in range(params.irls_iterations):
        weights = base_weights * robust_weights
        root_weight = np.sqrt(weights)[:, None]
        solution, *_ = np.linalg.lstsq(
            design * root_weight, measured * root_weight, rcond=None,
        )
        residual = np.linalg.norm(design @ solution - measured, axis=1)
        median = np.median(residual)
        scale = params.mad_scale * np.median(np.abs(residual - median))
        cutoff = max(params.cutoff_floor, median + params.cutoff_sigmas * scale)
        robust_weights = np.minimum(1.0, cutoff / np.maximum(residual, 1e-12))
    drifts = np.vstack([np.zeros((1, 2)), solution])
    return drifts, weights, residual


def measure_pairs(
    correlator: Correlator,
    prepared: Sequence[Any],
    pairs: Sequence[tuple[int, int]],
    *,
    n_threads: int = 1,
    device: Any = None,  # noqa: ANN401
    exclusion: int = 3,
    progress: tqdm | None = None,
    xp: XPType = np,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the shift and PSR of each pair, in a thread pool.

    On cupy, the measurements stay on the device until all pairs are done
    and are transferred to the host once, so the GPU queue is never drained
    per pair.

    Parameters
    ----------
    correlator
        Its :code:`correlate` is called with the two prepared inputs of a
        pair
    prepared
        The prepared inputs (:code:`correlator.prepare_input`), indexed by the
        node indices in `pairs`
    pairs
        Node index pairs :code:`(first, second)`
    n_threads
        Size of the thread pool. BLAS is limited to one thread while it runs.
    device
        A context manager entered in each worker, e.g. the caller's
        :code:`cupy.cuda.Device()`
    exclusion
        Sidelobe exclusion half-width of the PSR
    progress
        A progress bar to advance by one per pair
    xp
        The array module of the correlator

    Returns
    -------
    shift
        Shape (E, 2), float64, in the order of `pairs`
    psr
        Shape (E,)

    """
    if device is None:
        device = contextlib.nullcontext()

    def measure(k: int) -> tuple[Any, Any]:
        a, b = pairs[k]
        with device:
            res = correlator.correlate(prepared[a], prepared[b])
            # elements may be Python floats, numpy scalars or cupy 0-d arrays;
            # NoopCorrelator returns a scalar
            raw: Any = res.shift
            if np.isscalar(raw) or getattr(raw, "ndim", 1) == 0:
                raw = (raw, raw)
            if xp is np:
                psr: Any = peak_to_sidelobe_ratio(res.corrmap, exclusion)
                shift: Any = np.asarray([float(x) for x in raw], dtype=np.float64)
            else:
                psr = _psr_device(res.corrmap, exclusion, xp)
                shift = xp.stack([xp.asarray(x, dtype=np.float64) for x in raw])
        if progress is not None:
            progress.update(1)
        return shift, psr

    with (
        set_num_threads(1 if n_threads > 1 else None),
        ThreadPoolExecutor(max_workers=n_threads) as pool,
    ):
        results = list(pool.map(measure, range(len(pairs))))
    if not results:
        return np.zeros((0, 2)), np.zeros(0)
    if xp is np:
        shifts = np.array([r[0] for r in results], dtype=np.float64)
        psrs = np.array([r[1] for r in results], dtype=np.float64)
    else:
        shifts = to_cpu(xp.stack([r[0] for r in results]))
        psrs = to_cpu(xp.stack([xp.asarray(r[1], dtype=np.float64) for r in results]))
    return shifts.reshape(-1, 2), psrs


def register_stack(  # noqa: C901, PLR0912, PLR0915
    frames: Sequence[Any],
    correlator: Correlator,
    options: Registration | None = None,
    *,
    align_batch: Callable[[range, np.ndarray], None] | None = None,
    make_reference: Callable[[range], Any] | None = None,
    n_threads: int | None = None,
    xp: XPType = np,
    progress: bool = False,
) -> StackRegistrationResult:
    """Register the frames of a stack pairwise, batch by batch.

    Parameters
    ----------
    frames
        Indexable sequence of the images to register (whatever the
        correlator's :code:`prepare_input` takes)
    correlator
        Measures the shift of a pair; any :class:`~libertem_holo.base.align.Correlator`
    options
        :class:`StarRegistration` (the default) or :class:`AnchoredRegistration`
    align_batch
        Called as :code:`align_batch(indices, drifts)` with the frame
        indices of a batch (a :code:`range`) and their solved drifts, as
        soon as they are known; e.g. to apply the shifts in place
    make_reference
        Anchored topology only: called with the indices of the first batch
        after :code:`align_batch`, must return the reference image (e.g. the
        average of the aligned frames); it is prepared with the correlator
        and becomes node 0 of every later batch.
    n_threads
        Threads for preparing and correlating. Default: the number of CPUs
        for numpy, 1 for cupy.
    xp
        The array module of the correlator; on cupy, the workers use the
        caller's current device
    progress
        Show a progress bar over the pairs

    """
    if options is None:
        options = StarRegistration()
    if n_threads is None:
        n_threads = (os.cpu_count() or 1) if xp is np else 1
    anchored = isinstance(options, AnchoredRegistration)
    if anchored and make_reference is None:
        msg = "the anchored topology needs `make_reference`"
        raise ValueError(msg)
    device = contextlib.nullcontext() if xp is np else xp.cuda.Device()
    n_frames = len(frames)
    batch_size = options.batch_size if anchored else _STAR_CHUNK
    solver = options.solver if anchored else GraphSolverParams()
    drifts = np.zeros((n_frames, 2))
    first_all: list[np.ndarray] = []
    second_all: list[np.ndarray] = []
    shift_all: list[np.ndarray] = []
    psr_all: list[np.ndarray] = []
    weight_all: list[np.ndarray] = []
    residual_all: list[np.ndarray] = []

    def prepare_many(indices: Iterable[int]) -> list[Any]:
        def prepare(i: int) -> Any:  # noqa: ANN401
            with device:
                return correlator.prepare_input(frames[i])

        with (
            set_num_threads(1 if n_threads > 1 else None),
            ThreadPoolExecutor(max_workers=n_threads) as pool,
        ):
            return list(pool.map(prepare, indices))

    def batches(start: int) -> Iterable[range]:
        while start < n_frames:
            end = min(n_frames, start + batch_size)
            yield range(start, end)
            start = end

    if not anchored:
        total = max(n_frames - 1, 0)
    else:
        first_end = min(n_frames, batch_size)
        total = len(all_pairs(first_end)) + sum(
            len(anchored_pairs(len(b))) for b in batches(first_end)
        )
    bar = tqdm(total=total, disable=not progress, desc="registration")

    def solve_batch(
        pairs: list[tuple[int, int]],
        prepared: list[Any],
        count: int,
        node_index: Callable[[int], int],
    ) -> np.ndarray:
        """Measure and solve one batch; record its edges with global indices."""
        shifts, psrs = measure_pairs(
            correlator, prepared, pairs, n_threads=n_threads,
            device=device, exclusion=solver.sidelobe_exclusion, progress=bar, xp=xp,
        )
        first = np.array([a for a, _ in pairs], dtype=np.int64)
        second = np.array([b for _, b in pairs], dtype=np.int64)
        local_drifts, weights, residuals = solve_graph(
            first, second, shifts, psrs, count, solver,
        )
        first_all.append(np.array([node_index(a) for a in first], dtype=np.int64))
        second_all.append(np.array([node_index(b) for b in second], dtype=np.int64))
        shift_all.append(shifts)
        psr_all.append(psrs)
        weight_all.append(weights)
        residual_all.append(residuals)
        return local_drifts

    with bar:
        if not anchored and n_frames > 0:
            reference = prepare_many([0])[0]
            if align_batch is not None:
                align_batch(range(1), drifts[0:1])
            for batch in batches(1):
                prepared = [reference, *prepare_many(batch)]
                local = solve_batch(
                    star_pairs(len(batch) + 1), prepared, len(batch) + 1,
                    lambda i, start=batch.start: 0 if i == 0 else start + i - 1,
                )
                drifts[batch.start:batch.stop] = local[1:]
                if align_batch is not None:
                    align_batch(batch, drifts[batch.start:batch.stop])
        elif anchored and n_frames > 0:
            first_batch = range(min(n_frames, batch_size))
            prepared = prepare_many(first_batch)
            local = solve_batch(
                all_pairs(len(first_batch)), prepared, len(first_batch), lambda i: i,
            )
            drifts[first_batch.start:first_batch.stop] = local
            if align_batch is not None:
                align_batch(first_batch, drifts[first_batch.start:first_batch.stop])
            if first_batch.stop < n_frames:
                assert make_reference is not None  # noqa: S101
                with device:
                    reference = correlator.prepare_input(make_reference(first_batch))
                for batch in batches(first_batch.stop):
                    prepared = [reference, *prepare_many(batch)]
                    local = solve_batch(
                        anchored_pairs(len(batch)), prepared, len(batch) + 1,
                        lambda i, start=batch.start: (
                            REFERENCE_NODE if i == 0 else start + i - 1
                        ),
                    )
                    drifts[batch.start:batch.stop] = local[1:]
                    if align_batch is not None:
                        align_batch(batch, drifts[batch.start:batch.stop])

    def concat(
        parts: list[np.ndarray], shape: tuple[int, ...], dtype: type,
    ) -> np.ndarray:
        if not parts:
            return np.zeros(shape, dtype=dtype)
        return np.concatenate(parts)

    return StackRegistrationResult(
        drifts=drifts,
        first=concat(first_all, (0,), np.int64),
        second=concat(second_all, (0,), np.int64),
        shift=concat(shift_all, (0, 2), np.float64),
        psr=concat(psr_all, (0,), np.float64),
        weight=concat(weight_all, (0,), np.float64),
        residual=concat(residual_all, (0,), np.float64),
    )

"""Tests of the pairwise graph registration (`libertem_holo.base._registration`)."""
# ruff: noqa: ANN202, D205, D209
# ruff: noqa: FBT002, ARG002, SLF001, PLC0415, RUF046
from __future__ import annotations

import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base import _registration
from libertem_holo.base._registration import (
    REFERENCE_NODE,
    all_pairs,
    anchored_pairs,
    register_stack,
    star_pairs,
)
from libertem_holo.base.align import (
    AnchoredRegistration,
    Correlator,
    GraphSolverParams,
    ImageCorrelator,
    NoopCorrelator,
    RegResult,
    StarRegistration,
    cross_correlate,
    peak_to_sidelobe_ratio,
    solve_graph,
)
from libertem_holo.base.convenience import _align_waves


def _cupy():
    d = detect()
    if not d["cudas"] or not d["has_cupy"]:
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    import cupy as cp

    return cp


def _planted_surface(shape, peak, seed=0):
    rng = np.random.default_rng(seed)
    surface = rng.random(shape)
    surface[peak] = 10.0
    return surface


@pytest.mark.parametrize("shape", [(40, 33), (33, 40)])
@pytest.mark.parametrize("peak", [(0, 0), (12, 7), (39, 32), (20, 0)])
@pytest.mark.parametrize("exclusion", [0, 3, 5])
def test_psr_numba_matches_vectorized(shape, peak, exclusion):
    peak = (min(peak[0], shape[0] - 1), min(peak[1], shape[1] - 1))
    surface = _planted_surface(shape, peak)
    expected = _registration._psr_xp(surface, exclusion, np)
    psr = peak_to_sidelobe_ratio(surface, exclusion)
    assert psr == pytest.approx(expected, rel=1e-9)
    # float32 surfaces, as from float32 input:
    psr32 = peak_to_sidelobe_ratio(surface.astype(np.float32), exclusion)
    assert psr32 == pytest.approx(expected, rel=1e-5)


def test_psr_device_is_device_side():
    """`_psr_device` returns a 0-d array; no NaN when no sidelobes remain."""
    surface = _planted_surface((12, 11), (5, 6))
    psr = _registration._psr_device(surface, 3, np)
    assert isinstance(psr, np.ndarray)
    assert psr.shape == ()
    expected = peak_to_sidelobe_ratio(surface, 3)
    assert float(psr) == pytest.approx(expected, rel=1e-9)
    with np.errstate(all="raise"):
        assert float(_registration._psr_device(surface, 20, np)) == 0.0
        assert float(_registration._psr_device(np.ones((12, 11)), 3, np)) == 0.0
    # float32 surfaces are accumulated in float64:
    psr32 = _registration._psr_device(surface.astype(np.float32), 3, np)
    assert psr32.dtype == np.float64
    assert float(psr32) == pytest.approx(float(psr), rel=1e-5)


@pytest.mark.with_numba
def test_psr_small():
    surface = _planted_surface((12, 11), (5, 6))
    expected = _registration._psr_xp(surface, 3, np)
    assert peak_to_sidelobe_ratio(surface, 3) == pytest.approx(expected, rel=1e-9)
    assert expected > 5
    # no sidelobes left: 0
    assert peak_to_sidelobe_ratio(surface, 20) == 0.0
    assert _registration._psr_xp(surface, 20, np) == 0.0
    # a flat surface has a zero (not NaN) PSR
    assert peak_to_sidelobe_ratio(np.ones((12, 11)), 3) == 0.0


def test_psr_wrapped_distance_is_roll_invariant():
    """The sidelobe exclusion wraps around, so rolling the surface changes nothing."""
    surface = _planted_surface((32, 36), (3, 33))
    reference = peak_to_sidelobe_ratio(surface, 3)
    for roll in [(5, 0), (0, -7), (-3, 20), (31, 35)]:
        rolled = np.roll(surface, roll, axis=(0, 1))
        assert peak_to_sidelobe_ratio(rolled, 3) == pytest.approx(reference, rel=1e-12)


def test_pairs():
    assert star_pairs(1) == []
    assert star_pairs(4) == [(0, 1), (0, 2), (0, 3)]
    assert all_pairs(3) == [(0, 1), (0, 2), (1, 2)]
    assert anchored_pairs(3) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert anchored_pairs(1) == [(0, 1)]


def test_solve_graph_hand_worked():
    """An inconsistent 3-node graph: 0->1 = 2, 0->2 = 5.5, 1->2 = 3."""
    first = [0, 0, 1]
    second = [1, 2, 2]
    shift = [[2.0, 0.0], [5.5, 0.0], [3.0, 0.0]]
    psr = [10.0, 10.0, 10.0]
    drifts, weight, residual = solve_graph(first, second, shift, psr, 3)
    assert drifts.shape == (3, 2)
    assert np.array_equal(drifts[0], [0.0, 0.0])
    assert drifts[:, 0] == pytest.approx([0.0, 2.1667, 5.3333], abs=1e-4)
    assert np.all(drifts[:, 1] == 0)
    assert weight.shape == residual.shape == (3,)
    assert np.all(weight > 0)


def test_solve_graph_axes_independent():
    first = [0, 0, 1]
    second = [1, 2, 2]
    shift = [[2.0, -2.0], [5.5, -5.5], [3.0, -3.0]]
    drifts, _, _ = solve_graph(first, second, shift, [1.0, 1.0, 1.0], 3)
    assert drifts[:, 0] == pytest.approx(-drifts[:, 1])
    assert drifts[:, 0] == pytest.approx([0.0, 2.1667, 5.3333], abs=1e-4)


def test_solve_graph_consistent_is_exact():
    truth = np.array([[0.0, 0.0], [1.5, -2.0], [-3.25, 0.5], [4.0, 4.0], [0.1, -0.1]])
    pairs = all_pairs(5)
    first = [a for a, _ in pairs]
    second = [b for _, b in pairs]
    shift = [truth[b] - truth[a] for a, b in pairs]
    drifts, weight, residual = solve_graph(first, second, shift, np.ones(len(pairs)), 5)
    assert np.allclose(drifts, truth, atol=1e-12)
    assert np.allclose(residual, 0, atol=1e-12)
    assert np.allclose(weight, 1.0)


def test_solve_graph_outlier_downweighted():
    truth = np.array([[0.0, 0.0], [1.5, -2.0], [-3.25, 0.5], [4.0, 4.0], [0.1, -0.1]])
    pairs = all_pairs(5)
    first = [a for a, _ in pairs]
    second = [b for _, b in pairs]
    shift = np.array([truth[b] - truth[a] for a, b in pairs])
    rng = np.random.default_rng(3)
    shift += rng.normal(scale=0.01, size=shift.shape)
    shift[4] += [10.0, -7.0]  # edge (1, 2) is way off
    drifts, weight, residual = solve_graph(first, second, shift, np.ones(len(pairs)), 5)
    assert np.allclose(drifts, truth, atol=0.05)
    assert residual[4] > 5
    assert weight[4] < 0.1
    assert np.all(weight[np.arange(len(pairs)) != 4] > 0.5)


def test_solve_graph_psr_weighting():
    """A low-PSR edge loses against the others even without robust reweighting."""
    first = [0, 0, 1]
    second = [1, 2, 2]
    shift = [[2.0, 0.0], [5.5, 0.0], [3.0, 0.0]]
    params = GraphSolverParams(irls_iterations=1)
    equal, _, _ = solve_graph(first, second, shift, [5.0, 5.0, 5.0], 3, params)
    low_direct, _, _ = solve_graph(first, second, shift, [5.0, 1.0, 5.0], 3, params)
    # with the 0->2 measurement de-weighted, node 2 moves towards 2 + 3 = 5:
    assert low_direct[2, 0] < equal[2, 0]
    assert low_direct[2, 0] == pytest.approx(5.0, abs=0.1)


def test_solve_graph_edge_cases():
    drifts, weight, residual = solve_graph([], [], np.zeros((0, 2)), [], 1)
    assert drifts.shape == (1, 2)
    assert weight.shape == residual.shape == (0,)
    drifts, _, _ = solve_graph([], [], np.zeros((0, 2)), [], 0)
    assert drifts.shape == (0, 2)
    # all PSRs zero (NoopCorrelator): no NaNs
    drifts, weight, _ = solve_graph([0, 0], [1, 2], [[1.0, 0], [2.0, 0]], [0.0, 0.0], 3)
    assert np.all(np.isfinite(drifts))
    assert np.allclose(drifts[:, 0], [0, 1, 2])
    assert np.allclose(weight, 1.0)
    # median zero but not all zero: the edges with a peak dominate
    _, weight, _ = solve_graph(
        [0, 0, 1], [1, 2, 2], [[1.0, 0], [2.0, 0], [1.0, 0]], [0.0, 0.0, 10.0], 3,
    )
    assert np.allclose(weight, [0.1, 0.1, 1.0])
    with pytest.raises(ValueError, match="first < second"):
        solve_graph([1], [0], [[1.0, 0.0]], [1.0], 2)
    with pytest.raises(ValueError, match="first < second"):
        solve_graph([0], [2], [[1.0, 0.0]], [1.0], 2)


def test_solver_params_validation():
    with pytest.raises(ValueError, match="irls_iterations"):
        GraphSolverParams(irls_iterations=0)
    with pytest.raises(ValueError, match="weight_clip"):
        GraphSolverParams(weight_clip=(1.0, 0.5))
    with pytest.raises(ValueError, match="sidelobe_exclusion"):
        GraphSolverParams(sidelobe_exclusion=-1)
    with pytest.raises(ValueError, match="batch_size"):
        AnchoredRegistration(batch_size=0)
    assert AnchoredRegistration().batch_size == 16
    assert StarRegistration().topology == "star"
    assert AnchoredRegistration().topology == "anchored"


def _disk_stack(n=6, shape=(96, 80), noise=0.05, seed=3):
    """Build a stack of a disk rolled by integer shifts, plus noise."""
    sy, sx = shape
    yy, xx = np.mgrid[:sy, :sx]
    disk = (np.sqrt((yy - sy / 2) ** 2 + (xx - sx / 2) ** 2) < 12).astype(np.float32)
    rng = np.random.default_rng(seed)
    shifts_true = [(0, 0), (2, -3), (-4, 1), (5, 5), (-1, -6), (3, 0), (1, 1), (-2, 4),
                   (6, -1), (0, 3)][:n]
    frames = np.stack([np.roll(disk, s, axis=(0, 1)) for s in shifts_true])
    frames += noise * rng.random((n, sy, sx)).astype(np.float32)
    return frames, np.asarray(shifts_true, dtype=float)


def _rolling_aligner(frames):
    """`align_batch` / `make_reference` callbacks: roll by the rounded drift,
    average the aligned frames of a batch."""
    aligned = np.zeros_like(frames)
    calls = []

    def align_batch(indices, drifts):
        calls.append((indices, drifts.copy()))
        for i, d in zip(indices, drifts, strict=True):
            rounded = tuple(int(round(x)) for x in d)
            aligned[i] = np.roll(frames[i], rounded, axis=(0, 1))

    def make_reference(indices):
        return aligned[indices.start:indices.stop].mean(axis=0)

    return align_batch, make_reference, calls


def test_register_stack_star_equals_align_waves():
    bfs, shifts_true = _disk_stack()
    waves = bfs.astype(np.complex128)
    correlator = ImageCorrelator(hanning=True, binning=1, upsample_factor=10)
    _, _, expected = _align_waves(waves.copy(), bfs.copy(), correlator, progress=False)

    result = register_stack(bfs, correlator, StarRegistration())
    assert result.drifts.shape == (6, 2)
    assert np.allclose(result.drifts, expected, atol=1e-9)
    assert np.allclose(result.drifts, -shifts_true, atol=0.11)
    assert np.all(result.first == 0)
    assert result.second.tolist() == [1, 2, 3, 4, 5]
    assert np.allclose(result.shift, result.drifts[1:])
    assert np.all(result.psr > 5)
    # one edge per unknown: exact fit, so only the PSR base weights remain
    base = np.clip((result.psr / np.median(result.psr)) ** 2, 0.1, 10.0)
    assert np.allclose(result.weight, base)
    assert np.allclose(result.residual, 0.0, atol=1e-9)
    meta = result.to_metadata()
    assert set(meta) == {"first", "second", "shift", "psr", "weight", "residual"}
    assert all(len(v) == 5 for v in meta.values())


@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("batch_size", [4, 16])
def test_register_stack_anchored(n_threads, batch_size):
    bfs, shifts_true = _disk_stack(n=10)
    correlator = ImageCorrelator(hanning=True, binning=1, upsample_factor=10)
    align_batch, make_reference, calls = _rolling_aligner(bfs)

    options = AnchoredRegistration(batch_size=batch_size)
    result = register_stack(
        bfs, correlator, options,
        align_batch=align_batch, make_reference=make_reference, n_threads=n_threads,
    )
    assert np.allclose(result.drifts, -shifts_true, atol=0.11)
    assert np.array_equal(result.drifts[0], [0, 0])
    n_first = min(10, batch_size)
    expected_edges = len(all_pairs(n_first))
    start = n_first
    while start < 10:
        count = min(10, start + batch_size) - start
        expected_edges += len(anchored_pairs(count))
        start += count
    for name in ("first", "second", "psr", "weight", "residual"):
        assert getattr(result, name).shape == (expected_edges,), name
    assert result.shift.shape == (expected_edges, 2)
    reference_edges = result.first == REFERENCE_NODE
    if batch_size < 10:
        assert reference_edges.sum() == 10 - n_first
        assert np.all(result.second[reference_edges] >= n_first)
    else:
        assert not reference_edges.any()
    assert np.all(result.psr > 5)
    assert np.all(result.second > np.where(reference_edges, -1, result.first))
    # every frame was handed to align_batch exactly once, in order:
    seen = [i for indices, _ in calls for i in indices]
    assert seen == list(range(10))
    for indices, drifts in calls:
        assert np.array_equal(drifts, result.drifts[indices.start:indices.stop])


def test_register_stack_threads_identical():
    bfs, _ = _disk_stack(n=7)
    correlator = ImageCorrelator(hanning=True, binning=1, upsample_factor=10)
    results = []
    for n_threads in (1, 4):
        align_batch, make_reference, _ = _rolling_aligner(bfs)
        results.append(register_stack(
            bfs, correlator, AnchoredRegistration(batch_size=4),
            align_batch=align_batch, make_reference=make_reference, n_threads=n_threads,
        ))
    a, b = results
    for name in ("drifts", "first", "second", "shift", "psr", "weight", "residual"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


class _PlainCorrelator(Correlator):
    """A correlator with only the base contract: no spectrum caching, no `weight`."""

    def prepare_input(self, img):
        window = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
        return np.asarray(img) * window

    def correlate(self, ref_image, moving_image, plot=False):
        pos, corrmap = cross_correlate(ref_image, moving_image, upsample_factor=10)
        shape = moving_image.shape
        shift = (pos[0] - shape[0] // 2, pos[1] - shape[1] // 2)
        return RegResult(maximum=pos, shift=shift, corrmap=corrmap)


def test_register_stack_any_correlator():
    """The graph needs nothing beyond `prepare_input` and `correlate`."""
    bfs, shifts_true = _disk_stack(n=5)
    options = AnchoredRegistration(batch_size=3)

    align_batch, make_reference, _ = _rolling_aligner(bfs)
    plain = register_stack(
        bfs, _PlainCorrelator(), options, align_batch=align_batch,
        make_reference=make_reference,
    )
    align_batch, make_reference, _ = _rolling_aligner(bfs)
    fast = register_stack(
        bfs, ImageCorrelator(hanning=True, upsample_factor=10), options,
        align_batch=align_batch, make_reference=make_reference,
    )
    assert np.allclose(plain.drifts, -shifts_true, atol=0.11)
    assert np.allclose(plain.drifts, fast.drifts, atol=1e-6)
    assert np.allclose(plain.psr, fast.psr, rtol=1e-6)

    with pytest.raises(ValueError, match="make_reference"):
        register_stack(bfs, _PlainCorrelator(), options)


def test_register_stack_noop_and_empty():
    bfs, _ = _disk_stack(n=3)
    result = register_stack(bfs, NoopCorrelator(), StarRegistration())
    assert np.all(result.drifts == 0)
    assert np.all(result.psr == 0)
    assert np.all(np.isfinite(result.weight))
    star = StarRegistration()
    result = register_stack(bfs[:0], NoopCorrelator(), star)
    assert result.drifts.shape == (0, 2)
    assert result.first.shape == (0,)
    result = register_stack(bfs[:1], NoopCorrelator(), star)
    assert result.drifts.shape == (1, 2)


@pytest.mark.parametrize("topology", ["star", "anchored"])
def test_register_stack_cupy(topology):
    cp = _cupy()
    bfs, shifts_true = _disk_stack(n=6)
    anchored = AnchoredRegistration(batch_size=4)
    options = StarRegistration() if topology == "star" else anchored

    # the drifts reach `align_batch` on the host, so the numpy aligner serves
    # both runs; only the reference goes to the device
    align_np, make_reference_np, _ = _rolling_aligner(bfs)
    align_cp, make_reference_host, _ = _rolling_aligner(bfs)

    def make_reference_cp(indices):
        return cp.asarray(make_reference_host(indices))

    expected = register_stack(
        bfs, ImageCorrelator(upsample_factor=10), options,
        align_batch=align_np, make_reference=make_reference_np,
    )
    result = register_stack(
        cp.asarray(bfs), ImageCorrelator(upsample_factor=10, xp=cp), options,
        align_batch=align_cp, make_reference=make_reference_cp, xp=cp,
    )
    for name in ("drifts", "first", "second", "shift", "psr", "weight", "residual"):
        assert isinstance(getattr(result, name), np.ndarray), name
    assert np.allclose(result.drifts, -shifts_true, atol=0.11)
    assert np.allclose(result.drifts, expected.drifts, atol=0.21)
    assert np.allclose(result.psr, expected.psr, rtol=1e-3)
    surface = np.asarray(_planted_surface((40, 33), (12, 7)))
    assert peak_to_sidelobe_ratio(cp.asarray(surface)) == pytest.approx(
        peak_to_sidelobe_ratio(surface), rel=1e-6,
    )
    device_psr = _registration._psr_device(cp.asarray(surface), 3, cp)
    assert isinstance(device_psr, cp.ndarray)
    assert float(device_psr) == pytest.approx(peak_to_sidelobe_ratio(surface), rel=1e-6)
    # the correlator keeps its result on the device:
    corr = ImageCorrelator(upsample_factor=10, xp=cp)
    pa = corr.prepare_input(cp.asarray(bfs[0]))
    pb = corr.prepare_input(cp.asarray(bfs[1]))
    res = corr.correlate(pa, pb)
    assert isinstance(res.maximum, cp.ndarray)
    assert res.maximum.dtype == np.float64
    assert all(isinstance(x, cp.ndarray) for x in res.shift)

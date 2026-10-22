"""Pairwise graph registration on the real 30-frame stack.

Compares the anchored topology against the default star alignment in-process,
so it needs only the stack files, not the reference results.
"""
# ruff: noqa: TC003
from __future__ import annotations

import pathlib

import numpy as np
import pytest
from libertem.api import Context
from skimage.restoration import unwrap_phase

from libertem_holo.base._registration import REFERENCE_NODE
from libertem_holo.base.align import AnchoredRegistration, WaveCorrelator
from libertem_holo.base.convenience import reconstruct_stack
from libertem_holo.base.io import InputData, Results
from libertem_holo.base.utils import remove_phase_ramp

try:
    import cupy as cp
except ImportError:
    cp = None

N_FRAMES = 30


@pytest.fixture(scope="module")
def cpu_ctx():
    with Context.make_with("dask", gpus=0) as ctx:
        yield ctx


@pytest.fixture(scope="module")
def stack_results(cpu_ctx: Context):
    """Run `reconstruct_stack` at most once per backend and registration."""
    cache: dict[tuple, Results] = {}

    def _run(
        dm_testdata_path: pathlib.Path, backend: str, registration: str,
    ) -> Results:
        if backend == "cupy" and cp is None:
            pytest.skip("no cupy")
        key = (backend, registration)
        if key not in cache:
            path_obj = dm_testdata_path / "holo/reconstr/stack_obj_minus.dm4"
            path_ref = dm_testdata_path / "holo/reconstr/stack_ref_minus.dm4"
            stack_obj = InputData.load_from_dm(path_obj)
            stack_ref = InputData.load_from_dm(path_ref)
            ctx = cpu_ctx if backend == "libertem" else None
            xp = np if backend == "libertem" else cp
            kwargs = {}
            if registration.startswith("anchored"):
                kwargs["registration"] = AnchoredRegistration()
            if registration.endswith("wave"):
                kwargs["correlator"] = WaveCorrelator(xp=xp)
            cache[key] = reconstruct_stack(
                stack=stack_obj, stack_ref=stack_ref,
                backend=backend, ctx=ctx, progress=False, **kwargs,
            )
        return cache[key]
    return _run


def _drifts(res: Results) -> np.ndarray:
    return np.vstack([res.metadata["drifts_x"], res.metadata["drifts_y"]]).T


def _flat_phase(res: Results) -> np.ndarray:
    phase = unwrap_phase(np.angle(res.complex_wave), rng=42)
    roi = np.s_[50:150, 250:400]
    phase, _, _ = remove_phase_ramp(phase, roi=roi)
    return phase - np.mean(phase[roi])


@pytest.mark.parametrize("backend", ["cupy", "libertem"])
def test_anchored_measures_like_star(
    dm_testdata_path: pathlib.Path, stack_results, backend,
):
    """The anchored topology on the default (brightfield) correlator.

    Its first batch contains the direct edges (0, i), which measure exactly
    what the star alignment measures. The solved drifts differ from the star
    drifts by up to a few pixels on this stack: the brightfield images carry
    detector-fixed biprism edges that the phase-normalized correlation locks
    onto, so the star drifts jump between frames while the graph solution
    is smooth (see `test_wave_topologies_agree`).
    """
    star = stack_results(dm_testdata_path, backend, "star")
    anchored = stack_results(dm_testdata_path, backend, "anchored")

    drifts_star = _drifts(star)
    drifts = _drifts(anchored)
    assert drifts.shape == (N_FRAMES, 2)
    assert np.all(np.isfinite(drifts))
    assert np.array_equal(drifts[0], [0, 0])
    assert "registration" not in star.metadata

    meta = anchored.metadata["registration"]
    assert meta["topology"] == "anchored"
    assert meta["batch_size"] == 16
    # all pairs of 16, then 14 against the reference plus all pairs of 14:
    n_edges = 120 + (14 + 91)
    for key in ("first", "second", "psr", "weight", "residual", "shift"):
        assert len(meta[key]) == n_edges, key
    first = np.asarray(meta["first"])
    second = np.asarray(meta["second"])
    psr = np.asarray(meta["psr"])
    shift = np.asarray(meta["shift"])
    assert np.all(np.isfinite(psr))
    assert np.all(psr > 0)
    assert (first == REFERENCE_NODE).sum() == 14
    assert np.all(second[first == REFERENCE_NODE] >= 16)
    # the direct edges of the first batch are the star measurements:
    direct = (first == 0) & (second < 16)
    assert direct.sum() == 15
    assert np.allclose(shift[direct], drifts_star[second[direct]], atol=1e-9)
    # the graph solution is smooth where the star drifts jump:
    assert np.abs(np.diff(drifts, axis=0)).max() < 1.0
    assert np.abs(drifts - drifts_star).max() < 4.0


@pytest.mark.parametrize("backend", ["cupy", "libertem"])
def test_wave_topologies_agree(dm_testdata_path: pathlib.Path, stack_results, backend):
    """With the wave correlator, star and anchored find the same smooth drift."""
    star = stack_results(dm_testdata_path, backend, "star-wave")
    anchored = stack_results(dm_testdata_path, backend, "anchored-wave")
    drifts_star = _drifts(star)
    drifts = _drifts(anchored)
    assert np.all(np.isfinite(drifts))
    assert np.abs(drifts - drifts_star).max() < 0.25
    assert np.abs(np.diff(drifts, axis=0)).max() < 0.5
    # a steady drift of a few pixels over the stack:
    assert 3 < np.abs(drifts[-1]).max() < 6
    assert len(anchored.metadata["registration"]["psr"]) == 120 + 14 + 91

    # wrapped difference: the unwrapping may differ by 2 pi in places
    diff = np.abs(np.angle(np.exp(1j * (_flat_phase(anchored) - _flat_phase(star)))))
    assert diff.mean() < 0.03


def test_metadata_roundtrip(dm_testdata_path: pathlib.Path, stack_results, tmp_path):
    res = stack_results(dm_testdata_path, "libertem", "anchored")
    path = tmp_path / "anchored.npz"
    res.save(path)
    loaded = Results.load(path)
    assert loaded.metadata["registration"] == res.metadata["registration"]

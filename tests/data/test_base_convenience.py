import pathlib

import numpy as np
import pytest
from libertem.api import Context
from skimage.restoration import unwrap_phase

from libertem_holo.base.convenience import reconstruct_stack
from libertem_holo.base.io import InputData, Results
from libertem_holo.base.utils import remove_phase_ramp

try:
    import cupy as cp
except ImportError:
    cp = None


@pytest.fixture(scope="module")
def cpu_ctx():
    with Context.make_with("dask", gpus=0) as ctx:
        yield ctx


@pytest.fixture(scope="module")
def stack_results(cpu_ctx: Context):
    """Run `reconstruct_stack` at most once per backend in this module."""
    cache: dict[str, Results] = {}

    def _run(dm_testdata_path: pathlib.Path, backend: str) -> Results:
        if backend == "cupy" and cp is None:
            pytest.skip("no cupy")
        if backend not in cache:
            path_obj = dm_testdata_path / "holo/reconstr/stack_obj_minus.dm4"
            path_ref = dm_testdata_path / "holo/reconstr/stack_ref_minus.dm4"
            stack_obj = InputData.load_from_dm(path_obj)
            stack_ref = InputData.load_from_dm(path_ref)
            ctx = cpu_ctx if backend == "libertem" else None
            cache[backend] = reconstruct_stack(
                stack=stack_obj, stack_ref=stack_ref,
                backend=backend, ctx=ctx,
            )
        return cache[backend]
    return _run


def _drifts(res: Results) -> np.ndarray:
    return np.vstack([res.metadata["drifts_x"], res.metadata["drifts_y"]]).T


@pytest.mark.parametrize("backend", ["cupy", "libertem"])
def test_reconstruct_stack(dm_testdata_path: pathlib.Path, stack_results, backend):
    res = stack_results(dm_testdata_path, backend)
    wave = res.complex_wave
    drifts = _drifts(res)
    phase = unwrap_phase(np.angle(wave), rng=42)
    roi = np.s_[50:150, 250:400]
    phase, _, _ = remove_phase_ramp(phase, roi=roi)
    phase -= np.mean(phase[roi])

    if False:
        arrays = {"drift": drifts}
        np.savez("/home/alex/minus_drifts_2.npz", **arrays, allow_pickle=False)
        res = Results(complex_wave=wave, unwrapped_phase=phase, brightfield=res.brightfield)
        res.save("/home/alex/minus_2.npz")

    expected = Results.load(dm_testdata_path / "holo/align/minus_2.npz")
    expected_drifts = np.load(dm_testdata_path / "holo/align/minus_drifts_2.npz")["drift"]
    assert expected.unwrapped_phase is not None
    np.testing.assert_allclose(phase, expected.unwrapped_phase, atol=0.05)
    np.testing.assert_allclose(drifts, expected_drifts, atol=0.1)


def test_reconstruct_stack_cupy_vs_libertem(dm_testdata_path: pathlib.Path, stack_results):
    res_cp = stack_results(dm_testdata_path, "cupy")
    res_lt = stack_results(dm_testdata_path, "libertem")

    # GPU and CPU FFTs differ in rounding, which can move the maximum of the
    # upsampled cross-correlation by one step (1/10 px, times binning 2):
    np.testing.assert_allclose(_drifts(res_cp), _drifts(res_lt), atol=0.21)
    wave_cp, wave_lt = res_cp.complex_wave, res_lt.complex_wave
    assert np.linalg.norm(wave_cp - wave_lt) / np.linalg.norm(wave_cp) < 1e-2
    bf_cp, bf_lt = res_cp.brightfield, res_lt.brightfield
    assert np.linalg.norm(bf_cp - bf_lt) / np.linalg.norm(bf_cp) < 1e-2

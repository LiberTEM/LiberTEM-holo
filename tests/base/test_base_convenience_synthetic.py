import numpy as np
import pytest
from libertem.api import Context

from libertem_holo.base.convenience import _reconstruct_stack, reconstruct_stack
from libertem_holo.base.io import InputData, Results
from libertem_holo.base.utils import HoloParams

try:
    import cupy as cp
except ImportError:
    cp = None


def _params(holo: np.ndarray, xp) -> HoloParams:
    return HoloParams.from_hologram(
        holo[0],
        out_shape=(64, 64),
        central_band_mask_radius=10,
        xp=xp,
    )


def _assert_results_close(a: Results, b: Results) -> None:
    scale = np.max(np.abs(a.complex_wave))
    np.testing.assert_allclose(a.complex_wave, b.complex_wave, atol=1e-6 * scale)
    np.testing.assert_allclose(
        a.brightfield, b.brightfield,
        atol=1e-5 * np.max(np.abs(a.brightfield)),
    )
    np.testing.assert_allclose(
        a.metadata["drifts_x"], b.metadata["drifts_x"], atol=1e-6,
    )
    np.testing.assert_allclose(
        a.metadata["drifts_y"], b.metadata["drifts_y"], atol=1e-6,
    )


def _assert_results_similar(a: Results, b: Results) -> None:
    # GPU and CPU FFTs differ in rounding, which can move the maximum of the
    # upsampled cross-correlation by one step (1/10 px, times binning 2):
    np.testing.assert_allclose(
        a.metadata["drifts_x"], b.metadata["drifts_x"], atol=0.21,
    )
    np.testing.assert_allclose(
        a.metadata["drifts_y"], b.metadata["drifts_y"], atol=0.21,
    )
    for x, y in [(a.complex_wave, b.complex_wave), (a.brightfield, b.brightfield)]:
        assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-2


@pytest.mark.parametrize("with_ref", [False, True])
def test_reconstruct_stack_libertem_vs_numpy(holo_stack, lt_ctx: Context, with_ref):
    holo, ref, _, _ = holo_stack
    stack = InputData.from_array(holo.astype(np.float32), exposure_time=1.0)
    stack_ref = None
    if with_ref:
        stack_ref = InputData.from_array(ref.astype(np.float32), exposure_time=1.0)

    res_lt = reconstruct_stack(
        stack, stack_ref, holoparams=_params(holo, np),
        backend="libertem", ctx=lt_ctx, progress=False,
    )
    # the frame-by-frame implementation, executed on the CPU:
    res_np = _reconstruct_stack(
        stack, stack_ref, _params(holo, np), None,
        xp=np, ctx=None, progress=False,
    )
    assert res_lt.complex_wave.shape == (64, 64)
    assert res_lt.brightfield.shape == (64, 64)
    assert len(res_lt.metadata["drifts_x"]) == holo.shape[0]
    # the object moves, so there should be drift:
    assert np.max(np.abs(res_lt.metadata["drifts_x"])) > 1
    _assert_results_close(res_np, res_lt)


def test_reconstruct_stack_cupy_vs_libertem(holo_stack, lt_ctx: Context):
    if cp is None:
        pytest.skip("no cupy")
    holo, ref, _, _ = holo_stack
    stack = InputData.from_array(holo.astype(np.float32), exposure_time=1.0)
    stack_ref = InputData.from_array(ref.astype(np.float32), exposure_time=1.0)

    res_lt = reconstruct_stack(
        stack, stack_ref, holoparams=_params(holo, np),
        backend="libertem", ctx=lt_ctx, progress=False,
    )
    res_cp = reconstruct_stack(
        stack, stack_ref, holoparams=_params(holo, cp),
        # the context is ignored by the cupy backend:
        backend="cupy", ctx=lt_ctx, progress=False,
    )
    _assert_results_similar(res_cp, res_lt)


def test_reconstruct_stack_backend_args(holo_stack, lt_ctx: Context):
    holo, _, _, _ = holo_stack
    stack = InputData.from_array(holo)
    with pytest.raises(ValueError, match="needs a `Context`"):
        reconstruct_stack(stack, backend="libertem")
    with pytest.raises(ValueError, match="unknown backend"):
        reconstruct_stack(stack, backend="foo", ctx=lt_ctx)


def test_into_libertem_dataset_from_array(lt_ctx: Context):
    data = np.random.default_rng(42).random((3, 16, 16), dtype=np.float32)
    input_data = InputData.from_array(data)
    ds = input_data.into_libertem_dataset(ctx=lt_ctx)
    assert tuple(ds.shape) == (3, 16, 16)

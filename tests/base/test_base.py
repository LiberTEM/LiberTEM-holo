import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.utils import HoloParams
from libertem_holo.base.reconstr import (
    HologramFFT,
    get_phase,
    phase_offset_correction,
    reconstruct_bf,
    reconstruct_frame,
)
from libertem_holo.base.utils import to_cpu

try:
    import cupy as cp
except ImportError:
    cp = None


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_get_phase(backend: str, holo_data) -> None:
    holo, ref, phase_ref, slice_crop = holo_data

    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    p = HoloParams.from_hologram(
        ref[0, 0],
        central_band_mask_radius=1,
        out_shape=(32, 32),
        line_filter_width=None,
        xp=xp,
    )

    _ = get_phase(holo[0, 0], params=p, xp=xp)

    # TODO: ensure deterministic results for `get_phase`?
    # assert np.allclose(phase_ref[slice_crop][0, 0], phase[...], rtol=0.12)


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
@pytest.mark.parametrize(
    "line_filter_width", [1, 10, None],
)
def test_params_from_hologram(backend: str, holo_data, line_filter_width) -> None:
    holo, ref, phase_ref, slice_crop = holo_data

    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    HoloParams.from_hologram(
        ref[0, 0],
        central_band_mask_radius=1,
        out_shape=(32, 32),
        line_filter_width=line_filter_width,
        xp=xp,
    )


def test_params_sb_upper_lower(holo_data):
    holo, ref, phase_ref, slice_crop = holo_data
    p1 = HoloParams.from_hologram(
        ref[0, 0],
        central_band_mask_radius=1,
        out_shape=(32, 32),
        sb='upper',
    )
    p2 = HoloParams.from_hologram(
        ref[0, 0],
        central_band_mask_radius=1,
        out_shape=(32, 32),
        sb='lower',
    )
    assert p1.sb_position_int == (53, 58)
    assert p2.sb_position_int == (11, 6)


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_phase_offset(backend: str, holo_data, lt_ctx) -> None:
    from libertem.io.dataset.memory import MemoryDataSet
    from libertem_holo.base.filters import disk_aperture
    from libertem_holo.udf.reconstr import HoloReconstructUDF
    from libertem.common.backend import set_use_cpu, set_use_cuda
    holo, ref, phase_ref, slice_crop = holo_data

    if backend == "cupy":
        d = detect()
        cudas = detect()["cudas"]
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    # Prepare LT datasets and do reconstruction
    dataset_holo = MemoryDataSet(data=holo, num_partitions=2, sig_dims=2)

    sb_position = [11, 6]
    sb_size = 6.26498204

    out_shape = dataset_holo.shape.sig
    aperture = np.fft.fftshift(disk_aperture(out_shape=out_shape, radius=sb_size))
    holo_udf = HoloReconstructUDF(out_shape=out_shape,
                                  sb_position=sb_position,
                                  aperture=aperture)
    try:
        if backend == "cupy":
            set_use_cuda(cudas[0])
        w_holo = lt_ctx.run_udf(dataset=dataset_holo, udf=holo_udf)["wave"].data
    finally:
        set_use_cpu(0)

    w_holo = w_holo.reshape((-1,) + tuple(out_shape))

    # pick two holograms and align the phase offset
    # (in case of cupy, input data is implicitly moved to device):
    averaged, phase_offsets, stack = phase_offset_correction(
        w_holo[:2], return_stack=True, xp=xp,
    )

    # with explicit conversion it should still work:
    averaged, phase_offsets, stack = phase_offset_correction(
        xp.asarray(w_holo[:2]), return_stack=True, xp=xp,
    )


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
@pytest.mark.parametrize(
    "precision", [True, False],
)
def test_hologram_fft_equivalence(backend: str, precision, holo_stack) -> None:
    holo, _, _, _ = holo_stack

    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        xp = cp
    else:
        xp = np

    frame = holo[0].astype(np.float32)
    p = HoloParams.from_hologram(
        frame,
        out_shape=(64, 64),
        central_band_mask_radius=10,
        xp=xp,
    )
    holo_fft = HologramFFT(frame, precision=precision, xp=xp)

    wave_ref = reconstruct_frame(
        frame, sb_pos=p.sb_position, aperture=p.aperture, slice_fft=p.slice_fft,
        precision=precision, xp=xp,
    )
    wave = reconstruct_frame(
        holo_fft, sb_pos=p.sb_position, aperture=p.aperture, slice_fft=p.slice_fft,
        xp=xp,
    )
    # the same `holo_fft` is re-used for the brightfield:
    bf_ref = reconstruct_bf(
        frame, aperture=p.aperture_bf, slice_fft=p.slice_fft,
        precision=precision, xp=xp,
    )
    bf = reconstruct_bf(
        holo_fft, aperture=p.aperture_bf, slice_fft=p.slice_fft, xp=xp,
    )

    # the array path transforms only the needed columns of the spectrum, the
    # HologramFFT path the complete half spectrum, so they agree up to rounding:
    tol = 1e-10 if precision else 1e-5
    assert wave.dtype == wave_ref.dtype
    wave, wave_ref = to_cpu(wave), to_cpu(wave_ref)
    assert np.allclose(wave, wave_ref, rtol=tol, atol=tol * np.abs(wave_ref).max())
    assert bf.dtype == bf_ref.dtype
    bf, bf_ref = to_cpu(bf), to_cpu(bf_ref)
    assert np.allclose(bf, bf_ref, rtol=tol, atol=tol * np.abs(bf_ref).max())


def test_hologram_fft_xp_mismatch(holo_stack) -> None:
    d = detect()
    if not d["cudas"] or not d["has_cupy"]:
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    holo, _, _, _ = holo_stack
    p = HoloParams.from_hologram(
        holo[0], out_shape=(64, 64), central_band_mask_radius=10,
    )
    holo_fft = HologramFFT(holo[0], xp=cp)
    with pytest.raises(ValueError, match="was computed with cupy"):
        reconstruct_bf(holo_fft, aperture=p.aperture_bf, slice_fft=p.slice_fft, xp=np)


@pytest.fixture
def frame_and_params(holo_stack):
    holo, _, _, _ = holo_stack
    frame = holo[0]
    p = HoloParams.from_hologram(frame, out_shape=(64, 64), central_band_mask_radius=10)
    return frame, p


def test_reconstruct_frame_holoparams(frame_and_params) -> None:
    """`holoparams=` is the same as passing the parameters explicitly (#176)."""
    frame, p = frame_and_params
    expected = reconstruct_frame(frame, p.sb_position, p.aperture, p.slice_fft)
    wave = reconstruct_frame(frame, holoparams=p)
    assert np.array_equal(wave, expected)
    assert wave.shape == (64, 64)

    holo_fft = HologramFFT(frame)
    assert np.allclose(reconstruct_frame(holo_fft, holoparams=p), expected)


def test_reconstruct_frame_holoparams_override(frame_and_params) -> None:
    """Explicit arguments take precedence over `holoparams`."""
    frame, p = frame_and_params
    expected = reconstruct_frame(frame, (0, 0), p.aperture, p.slice_fft)
    wave = reconstruct_frame(frame, sb_pos=(0, 0), holoparams=p)
    assert np.array_equal(wave, expected)

    other = HoloParams.from_hologram(
        frame, out_shape=(32, 32), central_band_mask_radius=10,
    )
    wave = reconstruct_frame(
        frame, aperture=other.aperture, slice_fft=other.slice_fft, holoparams=p,
    )
    assert wave.shape == (32, 32)


def test_reconstruct_frame_estimate_params(frame_and_params) -> None:
    """Without any parameters, they are estimated with `from_hologram`."""
    frame, _ = frame_and_params
    p = HoloParams.from_hologram(frame)
    wave = reconstruct_frame(frame)
    assert np.array_equal(wave, reconstruct_frame(frame, holoparams=p))
    assert wave.shape == (frame.shape[0] // 4, frame.shape[1] // 4)


@pytest.mark.parametrize("given", ["sb_pos", "aperture", "slice_fft"])
def test_reconstruct_frame_partial_params(frame_and_params, given: str) -> None:
    frame, p = frame_and_params
    values = {"sb_pos": p.sb_position, "aperture": p.aperture, "slice_fft": p.slice_fft}
    with pytest.raises(ValueError, match="missing"):
        reconstruct_frame(frame, **{given: values[given]})


def test_reconstruct_frame_hologram_fft_needs_params(frame_and_params) -> None:
    frame, _ = frame_and_params
    with pytest.raises(TypeError, match="HologramFFT"):
        reconstruct_frame(HologramFFT(frame))


def test_reconstruct_frame_holoparams_shape_mismatch(frame_and_params) -> None:
    frame, p = frame_and_params
    with pytest.raises(ValueError, match="shape"):
        reconstruct_frame(frame[:128, :128], holoparams=p)


def test_reconstruct_frame_holoparams_cupy(frame_and_params) -> None:
    """`xp` defaults to `holoparams.xp`."""
    d = detect()
    if not d["cudas"] or not d["has_cupy"]:
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    frame, _ = frame_and_params
    p_cp = HoloParams.from_hologram(
        cp.asarray(frame), out_shape=(64, 64), central_band_mask_radius=10, xp=cp,
    )
    wave = reconstruct_frame(cp.asarray(frame), holoparams=p_cp)
    assert isinstance(wave, cp.ndarray)


@pytest.mark.need_cupy
def test_cupy_is_available():
    """
    Ensure that the GPU test suite fails if cupy is not available
    """
    d = detect()
    if not d["cudas"] or not d["has_cupy"]:
        raise RuntimeError("must have working CUPY for GPU tests!")

import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.utils import HoloParams
from libertem_holo.base.reconstr import get_phase, phase_offset_correction
from libertem_holo.base.filters import butterworth_disk, butterworth_line


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


def test_butterworth_disk_cpu_gpu_equiv():
    d = detect()
    if not d['cudas'] or not d['has_cupy']:
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    import cupy as cp
    disk_gpu = butterworth_disk(shape=(512, 511), radius=128.0, order=12, xp=cp)
    disk_cpu = butterworth_disk(shape=(512, 511), radius=128.0, order=12, xp=np)

    assert np.allclose(disk_gpu.get(), disk_cpu)


def test_butterworth_line_cpu_gpu_equiv():
    d = detect()
    if not d['cudas'] or not d['has_cupy']:
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    import cupy as cp

    line_cpu = butterworth_line(
        shape=(512, 511),
        width=3,
        sb_position=(100.1, 100),
        length_ratio=0.9,
        order=12,
        xp=np,
    )
    line_gpu = butterworth_line(
        shape=(512, 511),
        width=3,
        sb_position=(100.1, 100),
        length_ratio=0.9,
        order=12,
        xp=cp,
    )

    assert np.allclose(line_gpu.get(), line_cpu)


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

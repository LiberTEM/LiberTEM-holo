import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.utils import HoloParams
from libertem_holo.base.reconstr import get_phase
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
    disk_gpu = butterworth_disk(shape=(512, 512), radius=128.0, order=12, xp=cp)
    disk_cpu = butterworth_disk(shape=(512, 512), radius=128.0, order=12, xp=np)

    assert np.allclose(disk_gpu.get(), disk_cpu)


def test_butterworth_line_cpu_gpu_equiv():
    d = detect()
    if not d['cudas'] or not d['has_cupy']:
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    import cupy as cp

    line_cpu = butterworth_line(
        shape=(512, 512),
        width=3,
        sb_position=(100.1, 100),
        length_ratio=0.9,
        order=12,
        xp=np,
    )
    line_gpu = butterworth_line(
        shape=(512, 512),
        width=3,
        sb_position=(100.1, 100),
        length_ratio=0.9,
        order=12,
        xp=cp,
    )

    assert np.allclose(line_gpu.get(), line_cpu)

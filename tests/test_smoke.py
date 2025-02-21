import pytest
import numpy as np

from libertem.utils.devices import detect

from libertem_holo.base.utils import HoloParams
from libertem_holo.base.reconstr import phase_offset_correction


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_holo_params_happy_case(backend: str, holo_data) -> None:
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
        out_shape=(64, 64),
        line_filter_length=0.9,
        line_filter_width=2,
        xp=xp,
    )

    assert p.sb_position_int == (53, 58)

    # post-process the aperture with a gaussian:
    p2 = p.filter_aperture_gaussian(sigma=1)

    assert p2.sb_position_int == (53, 58)


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_phase_offset_happy_case(backend: str, holo_data) -> None:
    holo, ref, phase_ref, slice_crop = holo_data

    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    phase_ref = xp.asarray(phase_ref)

    phase_offset_correction(
        phase_ref.reshape((-1, phase_ref.shape[2], phase_ref.shape[3])),
        xp=xp
    )


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_hard_aperture_disk_non_square(backend: str) -> None:
    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    from libertem_holo.base.utils import _hard_disk_aperture

    aperture = _hard_disk_aperture((512, 511), radius=7, xp=xp)
    assert aperture.shape == (512, 511)

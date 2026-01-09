import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.utils import HoloParams
from libertem_holo.base.generate import hologram_frame
from libertem_holo.base.filters import butterworth_disk, butterworth_line


def test_linefilter_orientation_l_r():
    sy, sx = 64, 64
    ref = hologram_frame(np.ones((sy, sx)), np.zeros((sy, sx)), f_angle=30)
    ref_rot = hologram_frame(np.ones((sy, sx)), np.zeros((sy, sx)), f_angle=120)
    p = HoloParams.from_hologram(ref)
    p_rot = HoloParams.from_hologram(ref_rot)
    line = butterworth_line(
        shape=(64, 64),
        width=10,
        sb_position=p.sb_position_int,
        order=25
    )
    line_rot = butterworth_line(
        shape=(64, 64),
        width=10,
        sb_position=p_rot.sb_position_int,
        order=25
    )

    # line filter comes from top-left:
    assert np.allclose(line[0, 0], 0)

    # bottom right is all-ones:
    assert np.allclose(line[-1, -1], 1)

    # in the rotated case, line filter comes from bottom left:
    assert np.allclose(line_rot[-1, 0], 0)


def test_linefilter_orientation_upper_lower():
    sy, sx = 64, 64
    ref = hologram_frame(np.ones((sy, sx)), np.zeros((sy, sx)), f_angle=30)
    p_upper = HoloParams.from_hologram(ref, sb='upper')
    p_lower = HoloParams.from_hologram(ref, sb='lower')
    line_upper = butterworth_line(
        shape=(64, 64),
        width=10,
        sb_position=p_upper.sb_position_int,
        order=25
    )
    line_lower = butterworth_line(
        shape=(64, 64),
        width=10,
        sb_position=p_lower.sb_position_int,
        order=25
    )

    # upper: line filter comes from top-left:
    assert np.allclose(line_upper[0, 0], 0)
    # bottom right is all-ones:
    assert np.allclose(line_upper[-1, -1], 1)

    # lower: line filter comes from bottom-right:
    assert np.allclose(line_lower[-1, -1], 0)
    # top left is all-ones:
    assert np.allclose(line_lower[0, 0], 1)


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

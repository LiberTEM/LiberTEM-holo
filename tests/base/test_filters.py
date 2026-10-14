import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.utils import (
    HoloParams, draw_lf_rect, fft_shift_coords, line_filter_coords,
)
from libertem_holo.base.generate import hologram_frame
from libertem_holo.base.filters import butterworth_disk, butterworth_line, line_filter


@pytest.mark.with_numba
@pytest.mark.parametrize(
    ("f_angle", "sb", "quadrant"), [
        (30, "upper", (-1, -1)),
        (30, "lower", (1, 1)),
        (120, "upper", (-1, 1)),
        (120, "lower", (1, -1)),
    ]
)
def test_linefilter_orientation(f_angle, sb, quadrant):
    """Line points to the quadrant of the sideband.

    With `sb_position` in fftshifted coordinates, as used by
    `HoloParams.from_hologram`.
    """
    shape = (64, 64)
    ref = hologram_frame(np.ones(shape), np.zeros(shape), f_angle=f_angle)
    p = HoloParams.from_hologram(ref, sb=sb)
    sb_shifted = fft_shift_coords(p.sb_position_int, shape)
    line = butterworth_line(
        shape=shape,
        width=3,
        sb_position=sb_shifted,
        order=25
    )
    center = np.array([shape[0] / 2 - 1, shape[1] / 2 - 1])
    direction = np.array(sb_shifted) - center
    assert tuple(np.sign(direction)) == quadrant

    def at(t: float) -> float:
        return line[tuple(np.round(center + t * direction).astype(int))]

    # line comes from the edge of the quadrant of the sideband:
    assert np.allclose(at(1.0), 0)
    assert np.allclose(at(2.0), 0)

    # the opposite direction is all-ones:
    assert np.allclose(at(-1.0), 1)
    assert np.allclose(at(-2.0), 1)


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


def _butterworth_line_reference(
    shape: tuple[int, int],
    width: float,
    sb_position: tuple[float, float],
    length_ratio: float,
    order: int,
) -> np.ndarray:
    # vectorized reference, including the offsets marked FIXME in the implementation
    center = np.array([shape[0] / 2 - 1, shape[1] / 2 - 1])
    direction = np.array(sb_position) - center
    unit = direction / np.linalg.norm(direction)
    start = center + (1 - length_ratio) * direction
    axis_origin = start + np.array([1, 0])
    straight_begin = start + np.array([0, -1])

    pos = np.moveaxis(np.mgrid[0:shape[0], 0:shape[1]], 0, -1).astype(np.float64)
    past_start = (pos - straight_begin) @ direction > 0
    rel = pos - axis_origin
    dist_axis = np.abs(rel[..., 0] * unit[1] - rel[..., 1] * unit[0])
    dist_cap = np.linalg.norm(rel, axis=-1)
    dist = np.where(past_start, dist_axis, dist_cap)
    return 1 - 1 / np.sqrt(1 + (dist / width)**(2 * order))


@pytest.mark.with_numba
@pytest.mark.parametrize("angle", np.arange(0, 360, 15) + 0.5)
@pytest.mark.parametrize("shape", [(64, 64), (65, 50)])
def test_butterworth_line_reference(angle, shape):
    center = np.array([shape[0] / 2 - 1, shape[1] / 2 - 1])
    rad = np.deg2rad(angle)
    sb_position = tuple(center + 20 * np.array([np.sin(rad), np.cos(rad)]))
    kwargs = {"width": 2.5, "sb_position": sb_position, "length_ratio": 0.8, "order": 3}
    line = butterworth_line(shape=shape, **kwargs)
    assert line.dtype == np.float32
    ref = _butterworth_line_reference(shape=shape, **kwargs)
    assert np.allclose(line, ref, atol=1e-6)


@pytest.mark.with_numba
@pytest.mark.parametrize("angle", np.arange(0, 360, 10))
def test_butterworth_line_direction(angle):
    """Line points from the center towards `sb_position`, for all angles.

    The center and the opposite direction stay free, including for
    axis-aligned positions.
    """
    shape = (128, 128)
    center = np.array([shape[0] / 2 - 1, shape[1] / 2 - 1])
    rad = np.deg2rad(angle)
    sb_position = tuple(np.round(center + 50 * np.array([np.sin(rad), np.cos(rad)])))
    direction = np.array(sb_position) - center
    line = butterworth_line(
        shape=shape, width=2, sb_position=sb_position, length_ratio=0.9, order=12,
    )

    def at(t: float) -> float:
        return line[tuple(np.round(center + t * direction).astype(int))]

    for t in (0.3, 0.6, 1.0, 1.2):
        assert at(t) < 0.01, t
    for t in (-1.2, -0.6, -0.3, 0):
        assert at(t) > 0.99, t


def test_butterworth_line_at_center():
    with pytest.raises(ValueError, match="at the center"):
        butterworth_line(shape=(64, 64), width=2, sb_position=(31, 31))


@pytest.mark.with_numba
@pytest.mark.parametrize("f_angle", [10, 30, 60, 100, 150])
@pytest.mark.parametrize("sb", ["upper", "lower"])
def test_aperture_line_points_to_central_band(f_angle, sb):
    """Line in the sideband aperture lies between sideband and central band.

    It must point towards where the central band actually is after shifting
    the sideband to the center (as in `reconstruct_frame`), and not cover the
    sideband.
    """
    sy, sx = 512, 512
    holo = hologram_frame(
        np.ones((sy, sx)), np.zeros((sy, sx)), f_angle=f_angle, sampling=3,
    )
    p = HoloParams.from_hologram(holo, sb=sb)

    sb_pos = np.array(p.sb_position).astype(np.int64)
    fft = np.roll(np.fft.fft2(holo), sb_pos, axis=(0, 1))
    fft = np.abs(np.fft.fftshift(fft))
    center = np.array(fft.shape) // 2
    # mask out the sideband itself:
    fft[center[0] - 5:center[0] + 6, center[1] - 5:center[1] + 6] = 0
    cb_pos = np.array(np.unravel_index(np.argmax(fft), fft.shape))
    to_cb = (cb_pos - center) / np.linalg.norm(cb_pos - center)

    aperture = np.fft.fftshift(p.aperture)
    ap_center = np.array(aperture.shape) // 2

    def at(t: float) -> float:
        return aperture[tuple(np.round(ap_center + t * p.sb_size * to_cb).astype(int))]

    assert at(0) > 0.9
    assert at(0.6) < 0.1
    assert at(-0.6) > 0.9


def test_line_filter_deprecated():
    shape = (64, 64)
    with pytest.warns(DeprecationWarning, match="line_filter is deprecated"):
        lf = line_filter(sb_position=(10, 12), out_shape=(32, 32), orig_shape=shape)
    assert lf.shape == (32, 32)
    assert lf.any()

    with pytest.warns(DeprecationWarning, match="draw_lf_rect is deprecated"):
        draw_lf_rect(
            np.zeros(shape, dtype=bool),
            orig_shape=shape, sb_position_shifted=(20, 22), length_ratio=0.9, width=3,
        )
    with pytest.warns(DeprecationWarning, match="line_filter_coords is deprecated"):
        line_filter_coords(
            length_ratio=0.9, sb_position_shifted=(20, 22), width=3, orig_shape=shape,
        )

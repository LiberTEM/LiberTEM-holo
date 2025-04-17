import numpy as np
import pytest

from libertem_holo.base.utils import remove_phase_ramp


@pytest.mark.parametrize(
    "method", ["gradient", "fit"],
)
@pytest.mark.parametrize(
    "shape", [
        (64, 64),
        (32, 64),
        (31, 17),
    ],
)
@pytest.mark.parametrize(
    "ramp_yx", [
        (0, 0),
        (0, 7),
        (0.01, 12),
        (1, -3.001),
        (-1, 3.001),
        (-1, -3.001),
    ],
)
def test_remove_phase_ramp(shape, ramp_yx, method):
    ramp_y, ramp_x = ramp_yx
    yy = np.arange(0, shape[0], 1)
    xx = np.arange(0, shape[1], 1)
    y, x = np.meshgrid(yy, xx, indexing='ij')
    ramp = ramp_x * x + ramp_y * y

    img_without_ramp, detected_ramp, (ramp_y_det, ramp_x_det) = remove_phase_ramp(
        ramp,
        roi=None,
        method=method,
    )

    assert np.allclose(img_without_ramp, 0)
    assert np.allclose(detected_ramp, ramp)


@pytest.mark.parametrize(
    "method", ["gradient", "fit"],
)
@pytest.mark.parametrize(
    "shape", [
        (64, 64),
        (32, 64),
        (31, 17),
    ],
)
@pytest.mark.parametrize(
    "ramp_yx", [
        (0, 0),
        (0, 7),
        (0.01, 12),
        (1, -3.001),
        (-1, 3.001),
        (-1, -3.001),
    ],
)
@pytest.mark.parametrize(
    "roi_method", [
        "arr", "slice"
    ]
)
def test_remove_phase_ramp_with_roi(shape, ramp_yx, method, roi_method):
    ramp_y, ramp_x = ramp_yx
    yy = np.arange(0, shape[0], 1)
    xx = np.arange(0, shape[1], 1)
    y, x = np.meshgrid(yy, xx, indexing='ij')
    ramp = ramp_x * x + ramp_y * y

    # put the ramp into a larger zero array to ensure we are looking at the
    # correct ROI:
    slice_in_shape = np.s_[0:shape[0], 0:shape[1]]
    space = np.zeros((128, 128), dtype=np.float64)
    space[slice_in_shape] = ramp

    roi = slice_in_shape
    if roi_method == "arr":
        roi = space[slice_in_shape]

    img_without_ramp, detected_ramp, (ramp_y_det, ramp_x_det) = remove_phase_ramp(
        space,
        roi=roi,
        method=method,
    )

    assert np.allclose(img_without_ramp[slice_in_shape], 0)
    if ramp_yx != (0, 0):
        assert not np.allclose(img_without_ramp, 0)
    assert np.allclose(detected_ramp[slice_in_shape], ramp)

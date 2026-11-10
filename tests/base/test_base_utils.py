import numpy as np
import pytest

from libertem_holo.base.generate import hologram_frame
from libertem_holo.base.reconstr import reconstruct_frame
from libertem_holo.base.utils import (
    HoloParams,
    _hard_disk_aperture,
    fringe_contrast,
    fringe_contrast_map,
    fringe_period,
    remove_phase_ramp,
)

try:
    import cupy as cp
except ImportError:
    cp = None


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
    y, x = np.meshgrid(yy, xx, indexing="ij")
    ramp = ramp_x * x + ramp_y * y

    img_without_ramp, detected_ramp, (_ramp_y_det, _ramp_x_det) = remove_phase_ramp(
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
        "arr", "slice",
    ],
)
def test_remove_phase_ramp_with_roi(shape, ramp_yx, method, roi_method):
    ramp_y, ramp_x = ramp_yx
    yy = np.arange(0, shape[0], 1)
    xx = np.arange(0, shape[1], 1)
    y, x = np.meshgrid(yy, xx, indexing="ij")
    ramp = ramp_x * x + ramp_y * y

    # put the ramp into a larger zero array to ensure we are looking at the
    # correct ROI:
    slice_in_shape = np.s_[0:shape[0], 0:shape[1]]
    space = np.zeros((128, 128), dtype=np.float64)
    space[slice_in_shape] = ramp

    roi = slice_in_shape
    if roi_method == "arr":
        roi = space[slice_in_shape]

    img_without_ramp, detected_ramp, (_ramp_y_det, _ramp_x_det) = remove_phase_ramp(
        space,
        roi=roi,
        method=method,
    )

    assert np.allclose(img_without_ramp[slice_in_shape], 0)
    if ramp_yx != (0, 0):
        assert not np.allclose(img_without_ramp, 0)
    assert np.allclose(detected_ramp[slice_in_shape], ramp)


def test_line_filter_width_0(holo_data) -> None:
    _holo, ref, _phase_ref, _slice_crop = holo_data

    p = HoloParams.from_hologram(
        ref[0, 0],
        central_band_mask_radius=1,
        out_shape=(32, 32),
        line_filter_width=0.0,
        xp=np,
    )

    wave = reconstruct_frame(
        ref[0, 0], sb_pos=p.sb_position, aperture=p.aperture,
        slice_fft=p.slice_fft,
    )

    assert not np.any(np.isnan(wave))


@pytest.mark.parametrize("shape", [(64, 64), (63, 65), (31, 17)])
def test_hard_disk_aperture_centered(shape):
    aperture = _hard_disk_aperture(shape, radius=5.5)
    # centered on the zero frequency, i.e. symmetric under (y, x) -> (-y, -x):
    assert aperture[0, 0]
    flipped = np.roll(aperture[::-1, ::-1], (1, 1), axis=(0, 1))
    assert np.array_equal(aperture, flipped)


def _vacuum_hologram(
    visibility: float, *, shape=(512, 512), **kwargs: float,
) -> np.ndarray:
    return hologram_frame(
        np.ones(shape), np.zeros(shape), visibility=visibility, **kwargs,
    )


@pytest.mark.parametrize(
    ("sb_position", "direction", "period"), [
        ((0, 16), (0.0, 1.0), 4.0),
        ((0, 48), (0.0, -1.0), 4.0),
        ((8, 0), (1.0, 0.0), 8.0),
        ((56, 56), (-np.sqrt(0.5), -np.sqrt(0.5)), 8 / np.sqrt(2)),
    ],
)
def test_fringe_period(sb_position, direction, period):
    d, p = fringe_period(sb_position, shape=(64, 64))
    assert np.allclose(d, direction)
    assert np.isclose(p, period)


def test_fringe_period_zero_frequency():
    with pytest.raises(ValueError, match="zero frequency"):
        fringe_period((0, 0), shape=(64, 64))


@pytest.mark.parametrize("f_angle", [0, 30, 75])
@pytest.mark.parametrize("sampling", [4, 5, 8])
@pytest.mark.parametrize("visibility", [0.3, 0.7])
def test_fringe_contrast_sideband(visibility, sampling, f_angle):
    holo = _vacuum_hologram(visibility, sampling=sampling, f_angle=f_angle)
    contrast = fringe_contrast(holo, method="sideband", size=200)
    assert np.isclose(contrast, visibility, rtol=1e-3)


@pytest.mark.parametrize("f_angle", [0, 30, 75])
@pytest.mark.parametrize("sampling", [4, 5, 8])
@pytest.mark.parametrize("visibility", [0.3, 0.7])
def test_fringe_contrast_profile(visibility, sampling, f_angle):
    holo = _vacuum_hologram(visibility, sampling=sampling, f_angle=f_angle)
    contrast = fringe_contrast(holo, method="profile", size=200)
    # sampling at whole pixels misses the extrema a bit, so the profile
    # underestimates the contrast, by up to ~20 % at 4 px per fringe:
    assert 0.78 * visibility < contrast <= visibility + 1e-9


def test_fringe_contrast_poisson_noise():
    # Poisson noise like `hologram_frame(..., poisson_noise=0.01)`, which
    # means ~100 electrons per pixel, but with a seeded generator:
    counts = 1000
    noise_scale = 0.01 * counts
    rng = np.random.default_rng(42)
    holo = noise_scale * rng.poisson(_vacuum_hologram(0.5) / noise_scale)
    assert np.isclose(fringe_contrast(holo, method="sideband"), 0.5, rtol=0.02)
    assert 0.4 < fringe_contrast(holo, method="profile") < 0.5


def test_fringe_contrast_map():
    holo = _vacuum_hologram(0.6, shape=(256, 256))
    contrast = fringe_contrast_map(holo, out_shape=(64, 96))
    assert contrast.shape == (64, 96)
    # the fringes are not periodic over the hologram, which disturbs the edges
    # and causes some ripples:
    interior = contrast[8:-8, 8:-8]
    assert np.allclose(interior, 0.6, rtol=0.05)
    assert np.isclose(interior.mean(), 0.6, rtol=1e-2)


def test_fringe_contrast_map_default_out_shape():
    """The default `out_shape` grows to fit the aperture."""
    # 3 px per fringe: the aperture is larger than a quarter of the hologram
    holo = _vacuum_hologram(0.6, shape=(256, 256), sampling=3)
    contrast = fringe_contrast_map(holo)
    assert min(contrast.shape) > 64
    assert np.isclose(contrast[16:-16, 16:-16].mean(), 0.6, rtol=1e-2)


def test_fringe_contrast_map_aperture_too_large():
    holo = _vacuum_hologram(0.6, shape=(256, 256))
    with pytest.raises(ValueError, match="doesn't fit into out_shape"):
        fringe_contrast_map(holo, out_shape=(32, 48))


def test_fringe_contrast_map_cb_zero():
    """Zero where the centerband vanishes, instead of nan."""
    holo = np.zeros((64, 64))
    contrast = fringe_contrast_map(holo, sb_position=(0, 16), sb_size=4)
    assert np.all(contrast == 0)


def test_fringe_contrast_local():
    """The region selects which part of the hologram is measured."""
    shape = (512, 512)
    holo = np.concatenate([
        _vacuum_hologram(0.3, shape=shape)[:256],
        _vacuum_hologram(0.7, shape=shape)[256:],
    ])
    for method in ("sideband", "profile"):
        top = fringe_contrast(holo, method=method, center=(110, 256), size=150)
        bottom = fringe_contrast(holo, method=method, center=(400, 256), size=150)
        assert np.isclose(top, 0.3, rtol=0.2)
        assert np.isclose(bottom, 0.7, rtol=0.2)


@pytest.mark.parametrize("method", ["sideband", "profile"])
def test_fringe_contrast_region_too_large(method):
    holo = _vacuum_hologram(0.5, shape=(128, 128))
    with pytest.raises(ValueError, match="doesn't fit"):
        # the default of 50 fringe periods is 250 px:
        fringe_contrast(holo, method=method)


def test_fringe_contrast_profile_too_short():
    holo = _vacuum_hologram(0.5, shape=(128, 128))
    with pytest.raises(ValueError, match="maxima"):
        fringe_contrast(holo, method="profile", size=3)


def test_fringe_contrast_unknown_method():
    holo = _vacuum_hologram(0.5, shape=(64, 64))
    with pytest.raises(ValueError, match="unknown method"):
        fringe_contrast(holo, method="foo")


def test_fringe_contrast_cupy():
    if cp is None:
        pytest.skip("no cupy")
    holo = _vacuum_hologram(0.5)
    expected = fringe_contrast(holo, method="sideband")
    contrast = fringe_contrast(cp.asarray(holo), method="sideband", xp=cp)
    assert np.isclose(contrast, expected)
    assert isinstance(fringe_contrast_map(cp.asarray(holo), xp=cp), cp.ndarray)
    profile = fringe_contrast(cp.asarray(holo), method="profile", xp=cp)
    assert np.isclose(profile, fringe_contrast(holo, method="profile"))

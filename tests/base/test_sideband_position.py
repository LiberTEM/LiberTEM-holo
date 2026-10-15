"""Tests for `estimate_sideband_position`, against a reference implementation."""
import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.generate import hologram_frame
from libertem_holo.base.utils import (
    _mask_threshold,
    _sideband_max,
    estimate_sideband_position,
)

try:
    import cupy as cp
except ImportError:
    cp = None


def _estimate_sideband_position_reference(
    holo_data, holo_sampling, central_band_mask_radius=None, sb="lower",
) -> tuple[float, float]:
    """Estimate the sideband position like the previous, array-based implementation.

    Kept verbatim (numpy only), except for the centre of the central band
    mask, which used `fftshift` and was off by one pixel for odd shapes.
    """
    f_freq_1d_y = np.fft.fftfreq(holo_data.shape[0], holo_sampling[0])
    f_freq_1d_x = np.fft.fftfreq(holo_data.shape[1], holo_sampling[1])
    f_freq_mesh = np.meshgrid(f_freq_1d_x, f_freq_1d_y)
    f_freq = np.hypot(f_freq_mesh[0], f_freq_mesh[1])

    if central_band_mask_radius is None:
        central_band_mask_radius = np.mean(holo_data.shape) / 20.0 * np.max(f_freq)

    shape = holo_data.shape
    cy = shape[0]//2
    cx = shape[1]//2
    ys, xs = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    aperture = np.zeros(shape, dtype=bool)
    aperture[np.sqrt((xs - cx)**2 + (ys - cy)**2) < central_band_mask_radius] = 1
    aperture = np.fft.ifftshift(aperture)  # was: `fftshift`

    aperture_central_band = np.subtract(1.0, aperture)
    fft_holo = np.fft.fft2(holo_data) / np.prod(holo_data.shape)
    fft_filtered = fft_holo * aperture_central_band

    if sb == "lower":
        fft_sb = fft_filtered[: int(fft_filtered.shape[0] / 2), :]
        sb_position = np.unravel_index(np.abs(fft_sb).argmax(), fft_sb.shape)
    elif sb == "upper":
        fft_sb = fft_filtered[int(fft_filtered.shape[0] / 2):, :]
        sb_position = np.unravel_index(np.abs(fft_sb).argmax(), fft_sb.shape)
        sb_position = np.add(
            np.asarray(sb_position),
            np.asarray([int(fft_filtered.shape[0] / 2), 0]),
        )
    return tuple(float(c) for c in sb_position)


def _get_xp(backend):  # noqa: ANN202
    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        return cp
    return np


def _assert_same_sideband(holo, result, expected, sb):  # noqa: ANN202
    """Assert that `result` ties with `expected`, up to FFT rounding errors.

    Where the maximum is not unique, like for conjugate pixels in row 0, or if
    the searched half only contains noise, the position depends on how the FFT
    implementation rounds (cuFFT vs. numpy).
    """
    if sb == "lower":
        row_start, row_stop = 0, int(holo.shape[0] / 2)
    else:
        row_start, row_stop = int(holo.shape[0] / 2), holo.shape[0]
    assert row_start <= result[0] < row_stop
    magnitude = np.abs(np.fft.fft2(holo))
    pos = tuple(int(c) for c in result)
    pos_expected = tuple(int(c) for c in expected)
    # rounding errors of the FFT are a few `eps * max`, real differences
    # between peaks are orders of magnitude larger:
    atol = 1e-9 * magnitude.max()
    assert np.isclose(magnitude[pos], magnitude[pos_expected], rtol=0, atol=atol), (
        f"{result} doesn't tie with {expected}"
    )


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("shape", [(64, 64), (48, 80), (63, 65), (31, 17), (128, 128)])
@pytest.mark.parametrize("f_angle", [0, 30, 90, 137.5])
@pytest.mark.parametrize("sampling", [5.0, 3.3])
@pytest.mark.parametrize("sb", ["lower", "upper"])
@pytest.mark.parametrize("radius", [None, 2.2, 4.0, 5.0])
def test_estimate_sideband_position_vs_reference(  # noqa: PLR0917
    backend, shape, f_angle, sampling, sb, radius,
):
    xp = _get_xp(backend)
    holo = hologram_frame(
        np.ones(shape), np.zeros(shape), sampling=sampling, f_angle=f_angle,
    )
    expected = _estimate_sideband_position_reference(
        holo, (1, 1), central_band_mask_radius=radius, sb=sb,
    )
    result = estimate_sideband_position(
        xp.asarray(holo), (1, 1), central_band_mask_radius=radius, sb=sb, xp=xp,
    )
    if backend == "numpy":
        assert result == expected
    else:
        _assert_same_sideband(holo, result, expected, sb)
    assert all(isinstance(c, float) for c in result)


@pytest.mark.parametrize(
    ("shape", "sampling", "sb"), [
        ((48, 80), 3.3, "lower"),
        ((63, 65), 5.0, "upper"),
    ],
)
@pytest.mark.parametrize(
    # With f_angle=90, the sideband is in row 0: for `lower`, |F(0, k)| and
    # |F(0, -k)| tie exactly with numpy's FFT, and for `upper`, only noise is
    # left, so the position depends on the rounding of the FFT. These are the
    # configurations that failed with cuFFT before ties were accepted.
    "f_angle", [0, 30, 90, 137.5],
)
def test_sideband_position_fft_rounding(shape, sampling, sb, f_angle):
    """`_assert_same_sideband` accepts results from a differently rounding FFT."""
    holo = hologram_frame(
        np.ones(shape), np.zeros(shape), sampling=sampling, f_angle=f_angle,
    )
    expected = _estimate_sideband_position_reference(
        holo, (1, 1), central_band_mask_radius=2.2, sb=sb,
    )
    fft = np.fft.fft2(holo)
    if sb == "lower":
        row_start, row_stop = 0, int(shape[0] / 2)
    else:
        row_start, row_stop = int(shape[0] / 2), shape[0]
    # emulate a different FFT implementation (like cuFFT) by errors of the size
    # of the FFT's rounding errors:
    scale = np.finfo(np.float64).eps * np.abs(fft).max()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(fft.shape) + 1j * rng.standard_normal(fft.shape)
        result = _sideband_max(
            fft + scale * noise, row_start, row_stop, _mask_threshold(2.2), xp=np,
        )
        _assert_same_sideband(holo, result, expected, sb)


@pytest.mark.parametrize("f_angle", [30, 90])
def test_assert_same_sideband_rejects_other_peak(f_angle):
    shape = (48, 80)
    holo = hologram_frame(
        np.ones(shape), np.zeros(shape), sampling=3.3, f_angle=f_angle,
    )
    expected = _estimate_sideband_position_reference(
        holo, (1, 1), central_band_mask_radius=2.2, sb="lower",
    )
    # the next smaller peak, after removing the sideband (and its tie):
    fft = np.fft.fft2(holo)
    peak = np.abs(fft[tuple(int(c) for c in expected)])
    fft[np.abs(fft) >= 0.99 * peak] = 0
    other = _sideband_max(fft, 0, int(shape[0] / 2), _mask_threshold(2.2), xp=np)
    with pytest.raises(AssertionError, match="doesn't tie"):
        _assert_same_sideband(holo, other, expected, "lower")


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("shape", [(64, 64), (63, 65), (31, 17), (9, 300)])
@pytest.mark.parametrize("sb", ["lower", "upper"])
@pytest.mark.parametrize("holo_sampling", [(1, 1), (0.5, 2)])
# seed 22 has a tie for shape (9, 300), sb="upper", radius 2.0:
@pytest.mark.parametrize("seed", [*range(8), 22])
@pytest.mark.parametrize("radius", [None, 2.0])
def test_estimate_sideband_position_noise_vs_reference(  # noqa: PLR0917
    backend, shape, sb, holo_sampling, seed, radius,
):
    # No clear peak, so there can be exact ties: with an odd number of rows,
    # both pixels of a conjugate pair |F(k)| == |F(-k)| can be in the searched
    # half. Comparing |z|^2 instead of abs(z) can then pick the other one, so
    # check that the result is an equally valid maximum instead of the position:
    xp = _get_xp(backend)
    holo = np.random.default_rng(seed).random(shape)
    expected = _estimate_sideband_position_reference(
        holo, holo_sampling, central_band_mask_radius=radius, sb=sb,
    )
    result = estimate_sideband_position(
        xp.asarray(holo), holo_sampling, central_band_mask_radius=radius, sb=sb, xp=xp,
    )
    fft_abs = np.abs(np.fft.fft2(holo))
    expected_idx = tuple(int(c) for c in expected)
    result_idx = tuple(int(c) for c in result)
    assert np.isclose(fft_abs[result_idx], fft_abs[expected_idx], rtol=1e-12, atol=0)


@pytest.mark.with_numba
@pytest.mark.parametrize("sb", ["lower", "upper"])
def test_estimate_sideband_position_numba(sb):
    # one small case per sideband, for coverage of the kernel with JIT disabled
    holo = hologram_frame(np.ones((32, 33)), np.zeros((32, 33)), f_angle=30)
    expected = _estimate_sideband_position_reference(holo, (1, 1), sb=sb)
    assert estimate_sideband_position(holo, (1, 1), sb=sb) == expected

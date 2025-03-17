import numpy as np
import pytest
from libertem.utils.devices import detect
from libertem_holo.base.align import cross_correlate
from sparseconverter import for_backend, NUMPY


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
@pytest.mark.parametrize(
    "norm", ["phase", None],
)
@pytest.mark.parametrize(
    "upsample_factor", (10, 25)
)
@pytest.mark.parametrize(
    "sig_shape", (
        (64, 64),
        (61, 67),
        (57, 60),
        (63, 72),
    )
)
@pytest.mark.parametrize(
    "shift,crop", (
        ((0, 0), 0),
        ((-3.7, 4.2), 6),
        ((6.7, 8.1), 10),
        ((0.2, 9.7), 11),
    )
)
def test_cross_correlation(backend, upsample_factor, sig_shape, shift, crop, norm):
    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
        from cupyx.scipy import ndimage as ni
    else:
        xp = np
        from scipy import ndimage as ni

    input_data = xp.zeros(sig_shape, dtype=np.float32)
    # draw a rectangle into the input data:
    input_data[32:38, 16:37] = 1.2345

    # shift by some subpixel value:
    if shift != (0, 0):
        input_shifted = np.fft.ifft2(
            ni.fourier_shift(np.fft.fft2(input_data), shift=shift)
        ).real
    else:
        input_shifted = input_data

    # crop off the edges that wrap around when shifted, meaning we select a
    # slightly smaller ROI for cross correlation:
    crop_s = np.s_[crop:-crop - 1, crop:-crop - 1]
    input_data = input_data[crop_s]
    input_shifted = input_shifted[crop_s]

    pos, _corrmap = cross_correlate(
        input_shifted,
        input_data,
        plot=False,
        normalization=norm,
        upsample_factor=upsample_factor,
        xp=xp,
    )
    pos = for_backend(pos, NUMPY)
    pos_rel = (
        pos[0] - (input_data.shape[0]) // 2,
        pos[1] - (input_data.shape[1]) // 2,
    )

    assert pos_rel == pytest.approx(shift, abs=1/upsample_factor)
    # assert pos_rel[0] == pytest.approx(shift[0], abs=1/upsample_factor)
    # assert pos_rel[1] == pytest.approx(shift[1], abs=1/upsample_factor)

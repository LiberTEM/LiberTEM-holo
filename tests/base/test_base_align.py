import numpy as np
import pytest
from skimage.registration import phase_cross_correlation
from libertem.utils.devices import detect
from sparseconverter import for_backend, NUMPY
from libertem_holo.base.align import cross_correlate, align_stack
from libertem_holo.base.filters import _butterworth_disk_cpu, hanning_2d


def _test_data_shifted(shape, shift):
    """
    Make two disks, the second one shifted by `shift`
    """
    cy = shape[0]/2
    cx = shape[1]/2
    order = 20
    radius = 16
    orig = _butterworth_disk_cpu(shape, radius=radius, order=order, cx=cx, cy=cy)
    shifted = _butterworth_disk_cpu(
        shape,
        radius=radius,
        order=order,
        cy=cy+shift[0],
        cx=cx+shift[1],
    )
    return orig, shifted


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
@pytest.mark.parametrize(
    "norm", [
        "phase",
        # None
    ],
)
@pytest.mark.parametrize(
    "upsample_factor", (1, 5, 10),
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
    "shift", (
        (0, 0),
        (-3.7, 4.2),
        (6.7, 8.1),
        (0.2, 9.7),
    )
)
def test_cross_correlation(backend, upsample_factor, sig_shape, shift, norm):
    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    input_data, input_shifted = _test_data_shifted(shape=sig_shape, shift=shift)
    input_data, input_shifted = xp.asarray(input_data), xp.asarray(input_shifted)
    my_filter = hanning_2d(shape=sig_shape, xp=xp)
    # my_filter = 1

    input_data -= input_data.mean()
    input_shifted -= input_shifted.mean()
    input_data_f = input_data * my_filter
    input_shifted_f = input_shifted * my_filter

    pos, _corrmap = cross_correlate(
        input_shifted_f,
        input_data_f,
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

    sk_shift, _, _ = phase_cross_correlation(
        moving_image=for_backend(input_shifted, NUMPY),
        reference_image=for_backend(input_data, NUMPY),
        upsample_factor=upsample_factor,
        normalization='phase',
    )
    sk_shift = tuple(-sk_shift)
    assert pos_rel == pytest.approx(sk_shift, abs=1/upsample_factor + 1e-5)
    assert pos_rel == pytest.approx(shift, abs=1/upsample_factor + 1e-5)


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_align_stack(backend):
    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    stack = xp.zeros((10, 64, 64), dtype=np.float32)
    shifts = xp.zeros((10, 2), dtype=np.int64)

    for i in range(stack.shape[0]):
        shift = (i, i)
        input_data, input_shifted = _test_data_shifted(shape=stack.shape[1:], shift=shift)
        input_data, input_shifted = xp.asarray(input_data), xp.asarray(input_shifted)
        stack[i] = input_shifted
        shifts[i] = xp.stack(xp.asarray(shift))

    aligned, shifts_found, reference, corrs = align_stack(
        stack=stack,
        wave_stack=stack,
        static=None,
        correlator=None,
        xp=xp,
    )
    assert np.allclose(-shifts_found, shifts)

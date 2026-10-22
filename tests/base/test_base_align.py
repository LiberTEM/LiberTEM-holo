import contextlib
import numpy as np
from scipy.ndimage import gaussian_filter
import pytest
from skimage.registration import phase_cross_correlation
from libertem.utils.devices import detect
from sparseconverter import for_backend, NUMPY
from libertem_holo.base.utils import HoloParams, to_cpu
from libertem_holo.base.align import (
    cross_correlate, align_stack, ImageCorrelator, AmplitudeCorrelator,
    BiprismDeletionCorrelator, BrightFieldCorrelator, PhaseImageCorrelator,
    GradAngleCorrelator, GradXYCorrelator, NoopCorrelator, WaveCorrelator, Correlator,
    RegResult,
)
from libertem_holo.base.filters import _butterworth_disk_cpu, hanning_2d


# inspired from https://stackoverflow.com/a/71901243/540644
@contextlib.contextmanager
def noninteractive_plot():
    """Utility to disable plot interactivity for tests."""
    import matplotlib

    try:
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        yield
    finally:
        matplotlib.use(backend)


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


def test_correlator_prepare_should_coerce_input():
    d = detect()
    if not d["cudas"] or not d["has_cupy"]:
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    import cupy as cp
    xp = cp

    # inputs are _numpy_ arrays, not cupy!
    a = np.zeros((128, 128), dtype=np.float32)
    b = np.zeros((128, 128), dtype=np.float32)

    params = HoloParams.from_hologram(
        a,
        central_band_mask_radius=1,
        out_shape=(64, 64),
        line_filter_length=0.9,
        line_filter_width=2,
        sb_position=(32, 32),
        xp=xp,
    )

    factories = [
        lambda: ImageCorrelator(xp=xp),
        lambda: BiprismDeletionCorrelator(xp=xp, mask=np.zeros((128, 128), dtype=bool)),
        lambda: AmplitudeCorrelator(xp=xp, holoparams=params),
        lambda: BrightFieldCorrelator(xp=xp, holoparams=params),
        lambda: PhaseImageCorrelator(xp=xp, holoparams=params),
        lambda: GradAngleCorrelator(xp=xp, holoparams=params),
        lambda: GradXYCorrelator(xp=xp, holoparams=params),
        NoopCorrelator,
    ]

    for fn in factories:
        corr = fn()
        a1 = corr.prepare_input(a)
        b1 = corr.prepare_input(b)
        _ = corr.correlate(ref_image=a1, moving_image=b1)


def test_plot_should_work_with_cupy():
    d = detect()
    if not d["cudas"] or not d["has_cupy"]:
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    import cupy as cp
    xp = cp

    # inputs are _numpy_ arrays, not cupy!
    a = np.zeros((128, 128), dtype=np.float32)
    b = np.zeros((128, 128), dtype=np.float32)

    params = HoloParams.from_hologram(
        a,
        central_band_mask_radius=1,
        out_shape=(64, 64),
        line_filter_length=0.9,
        line_filter_width=2,
        sb_position=(32, 32),
        xp=xp,
    )

    factories = [
        lambda: ImageCorrelator(xp=xp),
        lambda: BiprismDeletionCorrelator(
            xp=xp, mask=np.zeros((128, 128), dtype=bool)
        ),
        lambda: AmplitudeCorrelator(xp=xp, holoparams=params),
        lambda: BrightFieldCorrelator(xp=xp, holoparams=params),
        lambda: PhaseImageCorrelator(xp=xp, holoparams=params),
        lambda: GradAngleCorrelator(xp=xp, holoparams=params),
        lambda: GradXYCorrelator(xp=xp, holoparams=params),
        lambda: NoopCorrelator(),
    ]

    with noninteractive_plot():
        for fn in factories:
            corr = fn()
            a1 = corr.prepare_input(a)
            b1 = corr.prepare_input(b)
            _ = corr.correlate(ref_image=a1, moving_image=b1, plot=True)


def _upsampled_dft_legacy(corrspecs, frequencies, upsampled_region_size, axis_offsets):
    """Verbatim copy of `_upsampled_dft` before the spectrum-level refactoring."""
    im2pi = -1j * 2 * np.pi
    upsampled = corrspecs
    for (ax_freq, ax_offset) in zip(frequencies[::-1], axis_offsets[::-1]):
        kernel = np.linspace(
            -ax_offset,
            (-ax_offset + upsampled_region_size - 1),
            num=int(upsampled_region_size),
        )
        kernel = np.exp(kernel[:, None] * ax_freq * im2pi, dtype=np.complex64)
        upsampled = np.tensordot(kernel, upsampled, axes=(1, -1))
    return upsampled


def _cross_correlate_legacy(src, target, normalization='phase', upsample_factor=1, xp=np):
    """Verbatim copy of `cross_correlate` (minus plotting) before the
    spectrum-level refactoring; guards that the results stay bit-identical.
    """
    src = xp.asarray(src)
    target = xp.asarray(target)
    src_freq = xp.fft.fftn(src)
    target_freq = xp.fft.fftn(target)
    image_product = src_freq * target_freq.conj()

    if normalization == 'phase':
        eps = np.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError(f"unknown normalization {normalization}")

    cross_correlation = xp.fft.ifftn(image_product)
    shifted_corr = xp.fft.fftshift(np.abs(cross_correlation))

    maxima = xp.unravel_index(
        xp.argmax(shifted_corr),
        shifted_corr.shape
    )
    float_dtype = image_product.real.dtype
    midpoint = xp.array([xp.fix(axis_size / 2) for axis_size in src_freq.shape])
    shift = xp.stack(maxima).astype(float_dtype, copy=False)
    shift -= midpoint

    if upsample_factor > 1:
        frequencies = (
            xp.fft.fftfreq(src.shape[0], upsample_factor),
            xp.fft.fftfreq(src.shape[1], upsample_factor),
        )
        upsample_factor = xp.array(upsample_factor, dtype=float_dtype)
        shift = xp.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = xp.ceil(upsample_factor * 1.5)
        dftshift = xp.fix(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - np.round(shift * upsample_factor)
        cross_correlation = _upsampled_dft_legacy(
            image_product.conj(),
            frequencies,
            upsampled_region_size,
            sample_region_offset,
        ).conj()
        maxima = xp.unravel_index(
            xp.argmax(xp.abs(cross_correlation)), cross_correlation.shape
        )
        maxima = xp.stack(maxima).astype(float_dtype, copy=False)
        maxima -= dftshift
        shift += maxima / upsample_factor

    if xp is np:
        shift = tuple(float(x) for x in shift)
    else:
        shift = tuple(float(to_cpu(x)) for x in shift)
    pos = xp.array(shift) + midpoint
    return pos, shifted_corr


def _xp_for(backend):
    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        return cp
    return np


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("norm", ["phase", None])
@pytest.mark.parametrize("upsample_factor", (1, 5, 10))
@pytest.mark.parametrize("sig_shape", ((64, 64), (61, 67), (57, 60), (63, 72)))
@pytest.mark.parametrize("shift", ((0, 0), (-3.7, 4.2), (6.7, 8.1), (0.2, 9.7)))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cross_correlate_refactor_identical(
    backend, norm, upsample_factor, sig_shape, shift, dtype,
):
    """The spectrum-level pieces reproduce the original `cross_correlate` bit for bit."""
    xp = _xp_for(backend)
    input_data, input_shifted = _test_data_shifted(shape=sig_shape, shift=shift)
    my_filter = hanning_2d(shape=sig_shape, xp=np)
    src = xp.asarray(((input_shifted - input_shifted.mean()) * my_filter).astype(dtype))
    target = xp.asarray(((input_data - input_data.mean()) * my_filter).astype(dtype))

    pos, corrmap = cross_correlate(
        src, target, normalization=norm, upsample_factor=upsample_factor, xp=xp,
    )
    pos_legacy, corrmap_legacy = _cross_correlate_legacy(
        src, target, normalization=norm, upsample_factor=upsample_factor, xp=xp,
    )
    assert for_backend(pos, NUMPY).tolist() == for_backend(pos_legacy, NUMPY).tolist()
    assert corrmap.dtype == corrmap_legacy.dtype
    assert xp.array_equal(corrmap, corrmap_legacy)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("norm", ["phase", None])
@pytest.mark.parametrize("binning", (1, 2))
@pytest.mark.parametrize("upsample_factor", (1, 10))
def test_image_correlator_prepared_identical(backend, norm, binning, upsample_factor):
    """Correlating prepared images (cached spectra) equals the original
    `cross_correlate` on the pre-processed images bit for bit."""
    xp = _xp_for(backend)
    a, b = _test_data_shifted(shape=(61, 67), shift=(-3.7, 4.2))
    a, b = xp.asarray(a), xp.asarray(b)
    corr = ImageCorrelator(
        upsample_factor=upsample_factor, normalization=norm, hanning=True,
        binning=binning, xp=xp,
    )
    pa, pb = corr.prepare_input(a), corr.prepare_input(b)
    assert pa.shape == pb.shape
    # idempotent:
    assert corr.prepare_input(pa) is pa
    # the pre-processed image can be recovered for inspection:
    assert xp.allclose(pa.image.real, corr._preprocess(a), atol=1e-6)
    res = corr.correlate(pa, pb)

    # what the correlator fed to the FFT before: hanning, then binning
    a_pre = corr._preprocess(a)
    b_pre = corr._preprocess(b)
    pos, corrmap = _cross_correlate_legacy(
        a_pre, b_pre, normalization=norm, upsample_factor=upsample_factor, xp=xp,
    )
    pos_rel = (pos[0] - a_pre.shape[0] // 2, pos[1] - a_pre.shape[1] // 2)
    if binning != 1:
        pos_rel = (pos_rel[0] * binning, pos_rel[1] * binning)
    assert for_backend(res.maximum, NUMPY).tolist() == for_backend(pos, NUMPY).tolist()
    assert [float(to_cpu(x)) for x in res.shift] == [float(to_cpu(x)) for x in pos_rel]
    assert xp.array_equal(res.corrmap, corrmap)
    # and the raw images are prepared on the fly:
    res_raw = corr.correlate(a, b)
    assert [float(to_cpu(x)) for x in res_raw.shift] == [float(to_cpu(x)) for x in res.shift]


def _synthetic_wave(shape, shift, seed=7, sigma=2.0):
    """A complex wave with smooth random amplitude and phase texture across the
    whole field (like a specimen), rolled by `shift`."""
    rng = np.random.default_rng(seed)
    t1 = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    t2 = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    amp = 0.7 + 0.25 * t1 / t1.std()
    wave = amp * np.exp(1j * t2 / t2.std())
    wave = wave + 0.02 * (rng.random(shape) + 1j * rng.random(shape))
    return np.roll(wave, shift, axis=(0, 1))


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("shift", ((0, 0), (3, -4), (-5, 2)))
def test_wave_correlator(backend, shift):
    xp = _xp_for(backend)
    ref = _synthetic_wave((72, 80), (0, 0))
    moving = _synthetic_wave((72, 80), shift)
    corr = WaveCorrelator(upsample_factor=10, xp=xp)
    assert corr.correlate_on == "wave"
    pa, pb = corr.prepare_input(xp.asarray(ref)), corr.prepare_input(xp.asarray(moving))
    assert pa.spectrum.dtype == np.complex128
    res = corr.correlate(pa, pb)
    found = tuple(float(to_cpu(x)) for x in res.shift)
    # the shift to apply to the moving image to align it with the reference;
    # the ordinary correlation of windowed waves has a small bias towards zero
    assert found == pytest.approx(tuple(-s for s in shift), abs=0.15)


def test_correlate_on_attribute():
    assert Correlator.correlate_on == "bf"
    assert ImageCorrelator().correlate_on == "bf"
    assert WaveCorrelator().correlate_on == "wave"
    for cls in (
        BrightFieldCorrelator, PhaseImageCorrelator, AmplitudeCorrelator,
        GradAngleCorrelator, GradXYCorrelator,
    ):
        assert cls.correlate_on == "hologram"
    assert NoopCorrelator.correlate_on == "bf"

    class DuckCorrelator:
        def prepare_input(self, img):
            return img

        def correlate(self, ref_image, moving_image, plot=False):
            return RegResult(maximum=(0.0, 0.0), shift=(0.0, 0.0), corrmap=ref_image)

    assert getattr(DuckCorrelator(), "correlate_on", "bf") == "bf"
    # per instance for the input-agnostic ImageCorrelator:
    assert ImageCorrelator(correlate_on="wave").correlate_on == "wave"
    assert ImageCorrelator.correlate_on == "bf"


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("region", (2, 6, 15))
def test_upsampled_dft_kernel_formulation(backend, dtype, region):
    """The kernel built from `arange - offset` equals the linspace form for the
    integer-valued offsets that occur (`dftshift - round(shift * factor)`)."""
    xp = _xp_for(backend)
    from libertem_holo.base.align import _upsampled_dft_kernel

    freqs = xp.fft.fftfreq(37, 10).astype(dtype)
    im2pi = -1j * 2 * np.pi
    for offset in (-23.0, 0.0, 7.0, 52.0):
        off = xp.asarray(offset, dtype=dtype)
        kernel = _upsampled_dft_kernel(freqs, off, region)
        positions = np.linspace(-off, (-off + region - 1), num=region)
        expected = np.exp(positions[:, None] * freqs * im2pi, dtype=np.complex64)
        assert kernel.dtype == expected.dtype == np.complex64
        assert xp.array_equal(kernel, expected), offset
    # non-integer offsets differ by roundoff at most (not used in practice)
    off = xp.asarray(3.3, dtype=dtype)
    kernel = _upsampled_dft_kernel(freqs, off, region)
    positions = np.linspace(-off, (-off + region - 1), num=region)
    expected = np.exp(positions[:, None] * freqs * im2pi, dtype=np.complex64)
    assert xp.allclose(kernel, expected, rtol=1e-5, atol=1e-6)

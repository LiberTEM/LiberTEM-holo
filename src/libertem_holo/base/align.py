from __future__ import annotations

import math
import typing
from typing import Literal, NamedTuple

try:
    import cupy as cp
except ImportError:
    cp = None
import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from libertem_holo.base.reconstr import (
    HoloParams,
    XPType,
    get_phase,
    reconstruct_bf,
    reconstruct_frame,
)

from .utils import to_cpu

log = logging.getLogger(__name__)


def _hanning_2d(img, xp):
    return xp.asarray(img) * xp.outer(xp.hanning(img.shape[0]), xp.hanning(img.shape[1]))


def _upsampled_dft(
    corrspecs: npt.NDArray,
    frequencies: tuple[np.ndarray, np.ndarray],
    upsampled_region_size: int,
    axis_offsets: tuple[float, float],
) -> np.ndarray:
    """
    From https://github.com/LiberTEM/LiberTEM-blobfinder, which is itself
    heavily adapted from skimage.registration._phase_cross_correlation.py
    which is itself based on code by Manuel Guizar released initially under a
    BSD 3-Clause license @ https://www.mathworks.com/matlabcentral/fileexchange/18401

    :meta private:
    """
    upsampled = corrspecs
    for (ax_freq, ax_offset) in zip(frequencies[::-1], axis_offsets[::-1]):
        kernel = _upsampled_dft_kernel(ax_freq, ax_offset, upsampled_region_size)
        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        upsampled = np.tensordot(kernel, upsampled, axes=(1, -1))
    return upsampled


def _upsampled_dft_kernel(
    ax_freq: np.ndarray,
    ax_offset: float | np.ndarray,
    upsampled_region_size: int,
) -> np.ndarray:
    """DFT matrix of one axis for :func:`_upsampled_dft`.

    Shape :code:`(upsampled_region_size, len(ax_freq))`, complex64 like the
    original skimage code.
    """
    im2pi = -1j * 2 * np.pi
    # The sample positions -ax_offset + i, as a difference so that a
    # device-scalar offset stays on the device (no host round trip). In the
    # offset's floating dtype, like np.linspace(-ax_offset, ...) gave: the
    # positions are integer-valued, so the cast is exact.
    xp = _array_module(ax_freq)
    offset = xp.asarray(ax_offset)
    dtype = offset.dtype if offset.dtype.kind == "f" else np.float64
    positions = (xp.arange(upsampled_region_size) - offset).astype(dtype, copy=False)
    return np.exp(positions[:, None] * ax_freq * im2pi, dtype=np.complex64)


def _array_module(arr: typing.Any) -> XPType:  # noqa: ANN401
    """numpy or cupy, whichever `arr` belongs to."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def _plot_cross_correlate(*, shifted_corr, pos, plot_title, src, target):
    pos = tuple(to_cpu(i) for i in pos)
    fig, ax = plt.subplots(3, sharex=True, sharey=True)
    ax[0].imshow(to_cpu(shifted_corr))
    ax[0].plot(pos[1], pos[0], 'x', color='red')
    ax[1].imshow(to_cpu(src))
    ax[1].plot(pos[1], pos[0], 'x', color='red')
    ax[2].imshow(to_cpu(target))
    ax[2].plot(pos[1], pos[0], 'x', color='red')
    fig.suptitle(plot_title)


def cross_correlate(
    src,
    target,
    plot: bool = False,
    plot_title: str = "",
    normalization: Literal['phase'] | None = 'phase',
    upsample_factor=1,
    xp=np,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid image registration by cross-correlation.

    Supports optional phase normalization. Based on the
    `phase_cross_correlation` function of scikit-image, but with added GPU
    support via cupy, and some debugging facilities built in.

    Parameters
    ==========
    src
        The static image, either a numpy or cupy array
        (if cupy, you should set `xp=cp`, too)

    target
        The moving image, either a numpy or cupy array
        (if cupy, you should set `xp=cp`, too)

    normalization
        'phase' or None, same as for `phase_cross_correlation`

    upsample_factor
        Subpixel scaling factor, same as for `phase_cross_correlation`.
        Note that we test with a factor of up to 10; with larger values
        the results may not improve.

    xp
        numpy or cupy
    """
    src = xp.asarray(src)
    target = xp.asarray(target)
    src_freq = xp.fft.fftn(src)
    target_freq = xp.fft.fftn(target)
    shift, shifted_corr, midpoint = _correlate_spectra(
        src_freq,
        target_freq,
        normalization=normalization,
        upsample_factor=upsample_factor,
        xp=xp,
    )

    # for "backwards compat", return correlation maxima and not shift
    pos = shift + midpoint

    if plot:
        _plot_cross_correlate(
            shifted_corr=shifted_corr,
            pos=pos,
            plot_title=plot_title,
            src=src,
            target=target,
        )

    return pos, shifted_corr


def _correlate_spectra(
    src_freq: np.ndarray,
    target_freq: np.ndarray,
    *,
    normalization: Literal["phase"] | None,
    upsample_factor: int,
    xp: XPType = np,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correlate two spectra: :func:`cross_correlate` after the forward FFTs.

    Parameters
    ----------
    src_freq, target_freq
        Spectra (:code:`xp.fft.fftn`) of the static and the moving image
    normalization, upsample_factor, xp
        As for :func:`cross_correlate`

    Returns
    -------
    shift
        The shift of the moving image relative to the static one, a float64
        `xp` array of two elements. Nothing is transferred to the host: on
        cupy, the result stays on the device.
    shifted_corr
        The fftshifted magnitude of the cross-correlation
    midpoint
        The position of zero shift in `shifted_corr`

    """
    image_product = _cross_power_spectrum(src_freq, target_freq, normalization)
    shifted_corr, shift, midpoint = _coarse_peak(image_product, xp)

    # estimate sublixel shifts using the upsampled DFT method:
    if upsample_factor > 1:
        shift = _refine_shift(image_product, shift, upsample_factor, xp)

    return shift.astype(np.float64, copy=False), shifted_corr, midpoint


def _cross_power_spectrum(
    src_freq: np.ndarray,
    target_freq: np.ndarray,
    normalization: Literal["phase"] | None,
) -> np.ndarray:
    """Cross power spectrum of two spectra, optionally phase-normalized."""
    image_product = src_freq * target_freq.conj()

    if normalization == "phase":
        eps = np.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError(f"unknown normalization {normalization}")
    return image_product


def _coarse_peak(
    image_product: np.ndarray, xp: XPType,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer-pixel peak of the cross-correlation.

    Returns the fftshifted magnitude of the correlation, the integer shift
    (float dtype) and the midpoint, i.e. the position of zero shift.
    """
    cross_correlation = xp.fft.ifftn(image_product)
    shifted_corr = xp.fft.fftshift(np.abs(cross_correlation))

    # unravel the flat argmax with integer arithmetic: on cupy, this keeps
    # the index on the device
    width = shifted_corr.shape[1]
    flat_index = xp.argmax(shifted_corr)
    maxima = (flat_index // width, flat_index % width)
    float_dtype = image_product.real.dtype
    midpoint = xp.asarray([float(np.fix(axis_size / 2)) for axis_size in image_product.shape])
    shift = xp.stack(maxima).astype(float_dtype, copy=False)
    shift -= midpoint
    return shifted_corr, shift, midpoint


def _refine_shift(
    image_product: np.ndarray, shift: np.ndarray, upsample_factor: int, xp: XPType,
) -> np.ndarray:
    """Refine an integer shift to :code:`1/upsample_factor` pixels.

    Matrix-multiply DFT of the cross power spectrum in a window of
    :code:`ceil(1.5 * upsample_factor)` upsampled pixels around the current
    estimate (Guizar-Sicairos et al., as in scikit-image).
    """
    float_dtype = image_product.real.dtype
    frequencies = (
        xp.fft.fftfreq(image_product.shape[0], upsample_factor),
        xp.fft.fftfreq(image_product.shape[1], upsample_factor),
    )

    # The lattice parameters are Python scalars: they act as weak scalars, so
    # a float32 shift stays float32, and nothing of the device is touched
    # from the host.
    factor = float(upsample_factor)
    # Initial shift estimate in upsampled grid
    shift = xp.round(shift * factor) / factor
    upsampled_region_size = math.ceil(factor * 1.5)
    # Center of output array at dftshift + 1
    dftshift = float(np.fix(upsampled_region_size / 2.0))
    # Matrix multiply DFT around the current shift estimate
    sample_region_offset = dftshift - xp.round(shift * factor)
    cross_correlation = _upsampled_dft(
        image_product.conj(),
        frequencies,
        upsampled_region_size,
        sample_region_offset,
    ).conj()
    # Locate maximum and map back to original pixel grid
    flat_index = xp.argmax(xp.abs(cross_correlation))
    maxima = (flat_index // upsampled_region_size, flat_index % upsampled_region_size)

    maxima = xp.stack(maxima).astype(float_dtype, copy=False)
    maxima -= dftshift

    shift += maxima / factor
    return shift


def gradient(image: np.ndarray, scale=1):
    scale = [scale] * image.ndim
    gradients = np.gradient(np.asarray(image), *scale)
    return np.stack(gradients, axis=-1)


def get_grad_angle(image, scale=3):
    """From an image, get the angle of the gradient."""
    grad = gradient(image, scale=scale)
    return np.arctan2(grad[..., 0], grad[..., 1])


def get_grad_xy(image, scale=3):
    """From an image, get the angle of the gradient."""
    grad = gradient(image, scale=scale)
    return (grad[..., 0], grad[..., 1])


def is_left(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """
    Points a and b are points on a line and result is an array of
    True or False if c is left or right of line resp.
    """
    return (b[1] - a[1])*(c[0] - a[0]) - (b[0] - a[0])*(c[1] - a[1]) > 0


class RegResult(NamedTuple):
    """Result of registering one pair of images, see :meth:`Correlator.correlate`."""

    maximum: typing.Any
    """Position of the correlation maximum, an `xp` array of two floats"""
    shift: typing.Any
    """The shift to apply to the moving image to align it with the
    reference, a tuple of two floats (0-d `xp` arrays on cupy)"""
    corrmap: np.ndarray


class PreparedImage(NamedTuple):
    """Output of :meth:`Correlator.prepare_input` for the FFT-based correlators.

    Holds the spectrum (:code:`xp.fft.fftn`, native FFT layout) of the
    pre-processed image, so that an image that takes part in several
    correlations is transformed only once.
    """

    spectrum: typing.Any

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the pre-processed image (the same as the spectrum's)."""
        return self.spectrum.shape

    @property
    def image(self) -> typing.Any:
        """The pre-processed image, recovered by an inverse FFT.

        Complex, with a negligible imaginary part for real images; meant for
        inspection and plotting, e.g. :code:`plt.imshow(prepared.image.real)`.
        """
        # dispatches to cupy for cupy spectra
        return np.fft.ifftn(self.spectrum)


class Correlator:
    """Base class for image registration.

    :meth:`prepare_input` pre-processes one image, and :meth:`correlate`
    registers two outputs of :meth:`prepare_input`; what :meth:`prepare_input`
    returns is up to the correlator. :attr:`correlate_on` tells
    :func:`~libertem_holo.base.convenience.reconstruct_stack` which array of a
    reconstructed stack to feed to :meth:`prepare_input`.
    """

    correlate_on: Literal["bf", "wave", "hologram"] = "bf"
    """What the correlator correlates on, i.e. what
    :func:`~libertem_holo.base.convenience.reconstruct_stack` passes to
    :meth:`prepare_input`: the brightfield image (:code:`"bf"`), the complex
    object wave (:code:`"wave"`) or the raw hologram (:code:`"hologram"`)."""

    def prepare_input(
        self,
        img: np.ndarray,
    ) -> typing.Any:
        raise NotImplementedError()

    def correlate(
        self,
        ref_image: typing.Any,
        moving_image: typing.Any,
        plot: bool = False,
    ) -> RegResult:
        raise NotImplementedError()


class _FFTCorrelator(Correlator):
    """Shared implementation of the correlators built on :func:`cross_correlate`.

    Subclasses implement :meth:`_preprocess`; :meth:`prepare_input` transforms
    its result into a :class:`PreparedImage` and :meth:`correlate` works on
    those.
    """

    _xp: XPType
    _normalization: Literal["phase"] | None
    _upsample_factor: int

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def prepare_input(
        self,
        img: np.ndarray | PreparedImage,
    ) -> PreparedImage:
        """Pre-process `img` and return its spectrum as a :class:`PreparedImage`.

        A :class:`PreparedImage` is returned unchanged.
        """
        if isinstance(img, PreparedImage):
            return img
        xp = self._xp
        return PreparedImage(spectrum=xp.fft.fftn(xp.asarray(self._preprocess(img))))

    def correlate(
        self,
        ref_image: PreparedImage,
        moving_image: PreparedImage,
        plot: bool = False,
    ) -> RegResult:
        """Register `moving_image` against `ref_image`.

        Both are outputs of :meth:`prepare_input`; raw images are prepared on
        the fly. The returned :attr:`RegResult.shift` is the shift to apply
        to the moving image to align it with the reference.
        """
        xp = self._xp
        ref_image = self.prepare_input(ref_image)
        moving_image = self.prepare_input(moving_image)
        shift, corrmap, midpoint = _correlate_spectra(
            ref_image.spectrum,
            moving_image.spectrum,
            normalization=self._normalization,
            upsample_factor=self._upsample_factor,
            xp=xp,
        )
        pos = shift + midpoint
        if plot:
            _plot_cross_correlate(
                shifted_corr=corrmap,
                pos=pos,
                plot_title="",
                src=_prepared_to_image(ref_image, xp),
                target=_prepared_to_image(moving_image, xp),
            )
        pos_rel = (
            pos[0] - (moving_image.shape[0]) // 2,
            pos[1] - (moving_image.shape[1]) // 2,
        )
        return RegResult(maximum=pos, shift=pos_rel, corrmap=corrmap)


def _prepared_to_image(prepared: PreparedImage, xp: XPType) -> np.ndarray:
    """Real-space image of a :class:`PreparedImage`, for plotting only."""
    image = prepared.image
    if xp.abs(image.imag).max() <= 1e-6 * xp.abs(image).max():
        return image.real
    return xp.abs(image)


class ImageCorrelator(_FFTCorrelator):
    """
    Cross correlation based registration of any real-space 2D images
    (brightfield, phase or amplitude images, TEM or Lorentz TEM images, ...)
    with some pre-filtering: optional Hanning window and binning.

    `correlate_on` says which image of a reconstructed stack
    :func:`~libertem_holo.base.convenience.reconstruct_stack` gives it: the
    brightfield images by default, or the complex waves
    (:code:`correlate_on="wave"`).
    """
    def __init__(
        self,
        upsample_factor: int = 1,
        normalization: Literal['phase'] | None = 'phase',
        hanning: bool = True,
        binning: int = 1,
        xp: XPType = np,
        correlate_on: Literal["bf", "wave"] = "bf",
    ) -> None:
        if correlate_on not in ("bf", "wave"):
            msg = f"correlate_on must be 'bf' or 'wave', not {correlate_on!r}"
            raise ValueError(msg)
        self._xp = xp
        self._upsample_factor = upsample_factor
        self._normalization = normalization
        self._hanning = hanning
        self._binning = binning
        self._zoom_factor = 1 / binning
        self.correlate_on = correlate_on

    def _preprocess(
        self,
        img: np.ndarray,
    ) -> typing.Any:
        xp = self._xp
        if xp is np:
            import scipy.ndimage as ni
        else:
            import cupyx.scipy.ndimage as ni

        img = xp.asarray(img)

        # apply hanning filter:
        if self._hanning:
            img = _hanning_2d(img, xp=xp)

        # apply binning:
        if self._zoom_factor != 1:
            img = ni.zoom(img, self._zoom_factor)

        return img

    def correlate(
        self,
        ref_image: PreparedImage,
        moving_image: PreparedImage,
        plot: bool = False,
    ) -> RegResult:
        res = super().correlate(ref_image, moving_image, plot=plot)
        if self._binning != 1:
            pos_rel = (
                res.shift[0] / self._zoom_factor,
                res.shift[1] / self._zoom_factor,
            )
            res = res._replace(shift=pos_rel)
        return res


class WaveCorrelator(_FFTCorrelator):
    """
    Cross correlation of complex object waves: the Hanning-windowed waves are
    correlated as they are (ordinary, non-normalized correlation by default),
    so both amplitude and phase contribute to the registration.

    In :func:`~libertem_holo.base.convenience.reconstruct_stack`, it registers
    the reconstructed waves (:code:`correlate_on = "wave"`). Unlike the
    brightfield images, the waves don't carry the detector-fixed biprism
    edges and their Fresnel fringes, which a brightfield correlation can
    lock onto instead of the drifting specimen.
    """

    correlate_on = "wave"

    def __init__(
        self,
        upsample_factor: int = 10,
        normalization: Literal["phase"] | None = None,
        hanning: bool = True,
        xp: XPType = np,
    ) -> None:
        self._xp = xp
        self._upsample_factor = upsample_factor
        self._normalization = normalization
        self._hanning = hanning

    def _preprocess(
        self,
        img: np.ndarray,
    ) -> typing.Any:
        xp = self._xp
        img = xp.asarray(img)
        if self._hanning:
            img = _hanning_2d(img, xp=xp)
        return img


class BiprismDeletionCorrelator(_FFTCorrelator):
    """
    Cross correlation on low magnification while removing biprism.
    """
    def __init__(
        self,
        mask: np.ndarray,
        upsample_factor: int = 1,
        normalization: Literal['phase'] | None = 'phase',
        xp: XPType = np,
    ) -> None:
        self._mask = mask
        self._xp = xp
        self._upsample_factor = upsample_factor
        self._normalization = normalization

    def _preprocess(
        self,
        img: np.ndarray,
    ) -> typing.Any:
        overview = np.zeros_like(img)
        overview[:] = img
        overview[self._mask] = img.mean()
        return overview

    @classmethod
    def plot_get_coords(cls, img, coords_out):
        """
        At low magnification, plot image of area with biprism visible.
        Click on edges of biprism to create the coordinates to mask it out,
        for cross correlation.
        First, click one edge of biprism from left side, then same edge, right side.
        Then, click other edge of biprism from left side, then right side.
        -----1-------2-----
        -----3-------4-----
        """
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        def onclick(event):
            plt.plot(event.xdata, event.ydata, 'ro')
            coords_out.append((event.ydata, event.xdata))
            if len(coords_out) == 4:
                fig.canvas.mpl_disconnect(cid)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

    @classmethod
    def get_masked(cls, img, coords):
        """Uses coordinates from plot_get_coords to create a mask of biprism."""
        yx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        mask = is_left(coords[0], coords[1], yx) & ~ is_left(coords[2], coords[3], yx)
        return mask


class BrightFieldCorrelator(_FFTCorrelator):
    """
    Cross correlation on bright field of hologram.
    """

    correlate_on = "hologram"

    def __init__(
        self,
        holoparams: HoloParams,
        upsample_factor: int = 1,
        normalization: Literal["phase"] | None = "phase",
        hanning: bool = True,
        xp: XPType = np,
    ) -> None:
        self._holoparams = holoparams
        self._xp = xp
        self._normalization = normalization
        self._upsample_factor = upsample_factor
        self._hanning = hanning

    def _preprocess(
        self,
        img: np.ndarray,
    ) -> typing.Any:
        holoparams = self._holoparams
        holo_bf = np.abs(
            reconstruct_bf(
                frame=img,
                aperture=holoparams.aperture_bf,
                slice_fft=holoparams.slice_fft,
                xp=self._xp,
            ),
        )
        holo_bf = np.gradient(holo_bf)[0]
        # apply hanning filter:
        if self._hanning:
            holo_bf = _hanning_2d(holo_bf, xp=self._xp)
        return holo_bf


class PhaseImageCorrelator(_FFTCorrelator):
    """
    Cross correlation on reconstructed phase image.
    """

    correlate_on = "hologram"

    def __init__(
        self,
        holoparams: HoloParams,
        upsample_factor: int = 1,
        normalization: Literal['phase'] | None = 'phase',
        hanning: bool = True,
        xp: XPType = np,
    ) -> None:
        self._holoparams = holoparams
        self._xp = xp
        self._normalization = normalization
        self._upsample_factor = upsample_factor
        self._hanning = hanning

    def _preprocess(
        self,
        img: np.ndarray,
    ) -> typing.Any:
        holoparams = self._holoparams
        img = self._xp.asarray(img)
        phase = get_phase(img, holoparams, xp=self._xp)

        # apply hanning filter:
        if self._hanning:
            phase = _hanning_2d(phase, xp=self._xp)

        return phase


class AmplitudeCorrelator(_FFTCorrelator):
    """
    Cross correlation on reconstructed amplitude.
    """

    correlate_on = "hologram"

    def __init__(
        self,
        holoparams: HoloParams,
        upsample_factor: int = 1,
        normalization: Literal['phase'] | None = 'phase',
        hanning: bool = False,
        xp: XPType = np,
    ) -> None:
        self._holoparams = holoparams
        self._xp = xp
        self._hanning = hanning
        self._normalization = normalization
        self._upsample_factor = upsample_factor

    def _preprocess(
        self,
        img: np.ndarray,
    ) -> typing.Any:
        holoparams = self._holoparams
        slice_fft = holoparams.slice_fft
        amp = np.abs(
            reconstruct_frame(
                img, sb_pos=holoparams.sb_position,
                aperture=holoparams.aperture,
                slice_fft=slice_fft, xp=self._xp
            )
        )

        # apply hanning filter:
        if self._hanning:
            amp = _hanning_2d(amp, xp=self._xp)

        return amp


class GradAngleCorrelator(_FFTCorrelator):
    """
    Cross correlation on gradient angle of phase image.
    """

    correlate_on = "hologram"

    def __init__(
        self,
        holoparams: HoloParams,
        upsample_factor: int = 1,
        normalization: Literal['phase'] | None = 'phase',
        xp: XPType = np,
    ) -> None:
        self._holoparams = holoparams
        self._xp = xp
        self._normalization = normalization
        self._upsample_factor = upsample_factor

    def _preprocess(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        holoparams = self._holoparams
        grad_angle = get_grad_angle(get_phase(img, holoparams, xp=self._xp))
        return grad_angle


class GradXYCorrelator(Correlator):
    """
    Cross correlation on gradient x and Y, correlation maps summed.
    """

    correlate_on = "hologram"

    def __init__(
        self,
        holoparams: HoloParams,
        xp: XPType = np,
    ) -> None:
        self._holoparams = holoparams
        self._xp = xp

    def prepare_input(
        self,
        img: np.ndarray,
    ) -> typing.Any:
        holoparams = self._holoparams
        (grad_x, grad_y) = get_grad_xy(
            get_phase(img, holoparams, xp=self._xp),
            scale=3,
        )
        # because `gradient` interpolates at the edge, we get a nice
        # vertical artifact that the cross correlation latches onto,
        # so we need to slice the edges away. take care to slice
        # everything the same, so the shapes match, and we need to
        # slice enough such that the interpolated region is removed
        # completely (I think this relates to the `scale` argument
        # above):
        return (grad_x[4:-5, 4:-5], grad_y[4:-5, 4:-5])

    def correlate(
        self,
        ref_image: typing.Any,
        moving_image: typing.Any,
        plot: bool = False,
    ) -> RegResult:
        xp = self._xp
        ref_image_x, ref_image_y = ref_image
        moving_image_x, moving_image_y = moving_image
        pos_x, corrmap_x = cross_correlate(
            ref_image_x,
            moving_image_x,
            xp=xp,
            plot=plot,
        )
        pos_y, corrmap_y = cross_correlate(
            ref_image_y,
            moving_image_y,
            xp=xp,
            plot=plot,
        )
        corrmap = corrmap_x + corrmap_y
        pos = xp.unravel_index(xp.argmax(corrmap), corrmap.shape)
        pos = tuple(float(to_cpu(x)) for x in pos)
        pos_rel = (
            pos[0] - (moving_image_y.shape[0]) // 2,
            pos[1] - (moving_image_y.shape[1]) // 2,
        )
        return RegResult(maximum=pos, shift=pos_rel, corrmap=corrmap)


class NoopCorrelator(Correlator):
    """Do nothing, successfully."""

    def prepare_input(
        self,
        img: np.ndarray,
    ) -> typing.Any:
        return img

    def correlate(
        self,
        ref_image: typing.Any,
        moving_image: typing.Any,
        plot: bool = False,
    ) -> RegResult:
        return RegResult(maximum=0.0, shift=0.0, corrmap=np.zeros_like(ref_image))


def align_stack(
    stack: np.ndarray,
    wave_stack: np.ndarray,
    static: np.ndarray | None = None,
    correlator: Correlator | None = None,
    xp=np,
) -> tuple[np.ndarray, np.ndarray, typing.Any, np.ndarray]:
    """Align stacks of N holograms.

    Parameters
    ==========

    stack
        Stack of images with shape (N, h, w) which is used
        for the alignment. It should be of dtype float32 or float64,
        so for holograms, you may want to align on the abs(wave).
        You can also slice the original data, if you want to align
        to a region of interest.

    wave_stack
        The (complex) result of the reconstruction. This should have
        shape (N, h', w'), meaning it should have the same number of
        complex images as the stack argument, but the 2d items in the
        stack can have a different shape.

    static
        A reference image to align against. If this is not given,
        the first image of the stack will be taken.

    correlator
        A :class:`Correlator` instance. By default, an :class:`ImageCorrelator`
        is used, with reasonable default parameters. If you need control over
        these, you can construct your own :class:`Correlator` and pass it in.
        A requirement is that it has to work on the value given as the
        stack parameter.

    Returns
    =======

    aligned_stack
        The aligned stack. Same shape and dtype as wave_stack

    shifts
        The shifts that were applied to the stack. Shape (N, 2)

    reference
        The pre-processed reference, as returned by the correlator's
        :code:`prepare_input` (a :class:`PreparedImage` holding the spectrum
        for the FFT-based correlators). Useful for debugging.

    corrs
        The correlation maps, as returned from the correlator. Useful for
        debugging.
    """
    wave_stack = xp.asarray(wave_stack)
    stack = xp.asarray(stack)
    aligned_stack = xp.zeros_like(wave_stack)

    if correlator is None:
        correlator = ImageCorrelator(
            upsample_factor=10,
            normalization='phase',
            hanning=True,
            binning=1,
            xp=xp,
        )

    if static is None:
        reference = stack[0]
    else:
        reference = static
    reference = correlator.prepare_input(reference)

    corrs = xp.zeros((stack.shape[0],) + reference.shape, dtype=np.float32)
    shifts = xp.zeros((wave_stack.shape[0], 2), dtype="float32")

    if xp is np:
        import scipy.ndimage as ni
    else:
        import cupyx.scipy.ndimage as ni

    for i, (reg_frame, wave_frame) in enumerate(zip(stack, wave_stack)):
        wave_frame = xp.asarray(wave_frame)
        pre_reg_frame = correlator.prepare_input(reg_frame)

        reg_result = correlator.correlate(
            reference,
            pre_reg_frame,
            plot=False,
        )
        corrs[i] = reg_result.corrmap

        shifted = xp.fft.ifft2(ni.fourier_shift(
            xp.fft.fftn(wave_frame),
            xp.asarray(reg_result.shift),
        ))

        # support for non-complex data: explicitly discard imaginary part
        if not np.iscomplexobj(wave_stack):
            shifted = shifted.real

        aligned_stack[i] = shifted
        shifts[i] = xp.stack(reg_result.shift)
    return aligned_stack, shifts, reference, corrs


def stack_alignment_quality(wave_stack: np.ndarray, shifts):
    """Stack quality ,judged by standard deviation on the stacking axis.

    This should be mostly noise, if not, there may be issues from the alignment.
    """
    std_image = np.std(np.abs(wave_stack), axis=0)
    offset = math.ceil(shifts.max()) + 1
    return std_image[offset:-offset, offset:-offset]


# The pairwise graph registration lives in a private module; its public API
# is part of this module (imported last to avoid a cycle):
from libertem_holo.base._registration import (  # noqa: E402
    AnchoredRegistration,
    GraphSolverParams,
    StarRegistration,
    peak_to_sidelobe_ratio,
    solve_graph,
)

__all__ = [
    "AmplitudeCorrelator",
    "AnchoredRegistration",
    "BiprismDeletionCorrelator",
    "BrightFieldCorrelator",
    "Correlator",
    "GradAngleCorrelator",
    "GradXYCorrelator",
    "GraphSolverParams",
    "ImageCorrelator",
    "NoopCorrelator",
    "PhaseImageCorrelator",
    "PreparedImage",
    "RegResult",
    "StarRegistration",
    "WaveCorrelator",
    "align_stack",
    "cross_correlate",
    "get_grad_angle",
    "get_grad_xy",
    "gradient",
    "is_left",
    "peak_to_sidelobe_ratio",
    "solve_graph",
    "stack_alignment_quality",
]

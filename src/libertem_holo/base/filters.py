"""Useful image filtering helpers."""
import math
import warnings
from typing import NamedTuple

import numpy as np
import numba
from numba import cuda
import sparse
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage.filters import window
from skimage.restoration import unwrap_phase

from libertem.masks import radial_bins
from libertem_holo.base.utils import (
    fft_shift_coords, other_sb, _draw_lf_rect, get_slice_fft,
)


def disk_aperture(out_shape: tuple[int, int], radius: float, xp=np) -> np.ndarray:
    """Generate a disk-shaped aperture

    Parameters
    ----------
    out_shape : interger
        Shape of the output array
    radius : float
        Radius of the disk
    xp
        Either numpy or cupy

    Returns
    -------
    aperture
        2d array containing aperture

    """
    center = int(out_shape[0] / 2), int(out_shape[1] / 2)

    bins = xp.asarray(radial_bins(
        centerX=center[1],
        centerY=center[0],
        imageSizeX=out_shape[1],
        imageSizeY=out_shape[0],
        radius=float(radius),
        n_bins=1,
        use_sparse=False,
    ))

    return bins[0]


def butterworth_disk(shape: tuple[int, int], radius: float, order: int = 12, xp=np):
    """Generate a filered disk-shaped aperture.

    The edges are filtered with a butterworth filter of the given order.

    The result is in fftshifted layout: the disk is centered at
    `(shape[0] / 2, shape[1] / 2)`, which is the zero frequency of an
    fftshifted array for even sizes (and half a pixel past it for odd sizes).

    Parameters
    ----------
    shape
        output shape of the aperture

    radius
        radius in pixels

    order
        order of the butterworth filter

    xp
        Either numpy or cupy
    """
    if xp is np:
        cy = shape[0]/2
        cx = shape[1]/2
        return _butterworth_disk_cpu(shape, radius, cy, cx, order)
    else:
        import cupy as cp
        result = cp.zeros(shape, dtype=np.float32)
        size = 32
        threadsperblock = (size, size)
        blockspergrid_x = math.ceil(result.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(result.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        _butterworth_disk_gpu[blockspergrid, threadsperblock](
            result,
            radius,
            order
        )
        return result


@numba.njit(cache=True, inline="always")
def _butterworth_disk_kernel(y, x, cy, cx, radius, order):
    d = math.sqrt((y-cy)**2 + (x-cx)**2)
    return 1/math.sqrt(1 + math.pow((d/radius), 2*order))


@numba.njit(cache=True, parallel=True)
def _butterworth_disk_cpu(
    shape: tuple[int, int],
    radius: float,
    cy: float,
    cx: float,
    order: int = 12,
):
    result = np.zeros(shape, dtype=np.float32)
    for y in numba.prange(shape[0]):
        for x in range(shape[1]):
            result[y, x] = _butterworth_disk_kernel(y, x, cy, cx, radius, order)
    return result


@cuda.jit(cache=True)
def _butterworth_disk_gpu(result, radius: float, order: int = 12):
    y, x = cuda.grid(2)
    cy = result.shape[0]/2
    cx = result.shape[1]/2
    if x < result.shape[1] and y < result.shape[0]:
        result[y, x] = _butterworth_disk_kernel(y, x, cy, cx, radius, order)


def highpass(img: np.ndarray, sigma: float = 2) -> np.ndarray:
    """Return highpass by subtracting a gaussian lowpass filter."""
    return img - ndimage.gaussian_filter(img, sigma=sigma)


def exclusion_mask(img: np.ndarray, sigma: float = 6) -> np.ndarray:
    """Generate outlier mask.

    Return a mask with `True` entries for pixels deviating more
    than `sigma` from the mean.
    """
    return np.abs(img) > (img.mean() + sigma * img.std())


def clipped(img: np.ndarray, sigma: float = 6):
    """Mask out outliers.

    Return `img`, but with pixels deviating more than `sigma` from the mean
    masked out.

    Useful for plotting:

    >>> plt.imshow(img, vmax=np.max(clipped(img)))  # doctest: +SKIP
    """
    sigma_mask = exclusion_mask(img, sigma=sigma)
    sigma_mask = ~sigma_mask
    return img[sigma_mask]


def phase_unwrap(image):
    """
    A phase_unwrap function that is unwrap the complex / wrapped phase image.

    Parameters
    ----------
    image : 2d nd array
        Complex or Wrapped phase image
    Returns
    -------
        2d nd array of the unwrapped phase image
    """

    if image.dtype.kind != 'c':
        image_new = unwrap_phase(image)
    else:
        angle = np.angle(image)
        image_new = unwrap_phase(angle)

    return image_new


def remove_dead_pixels(img, sigma_lowpass=2.0, sigma_exclusion=6.0):
    """Remove dead pixels.

    Parameters
    ----------
    img : np.array
        Input array
    sigma_lowpass : float
        How much of the low frequencies should be removed before
        finding bad pixels
    sigma_exclusion : float
        Pixels deviating more than this value from the mean will be
        removed

    """
    from libertem.corrections.detector import correct
    mask = exclusion_mask(highpass(img, sigma=sigma_lowpass), sigma=sigma_exclusion)

    coords = sparse.COO(mask)
    return correct(
        buffer=img.reshape((1, *img.shape)),
        excluded_pixels=coords.coords,
        sig_shape=tuple(img.shape),
    ).squeeze()


def window_filter(input_array, window_type, window_shape):
    """Apply window-based filter.

    Return a filtered array with the same size of the input array

    Parameters
    ----------
    input_array: array
        Input array
    window_type : string, float or tuple
        The type of window to be created. Any window type supported by
        ``scipy.signal.get_window`` is allowed here. See notes below for a
        current list, or the SciPy documentation for the version of SciPy
        on your machine.
    window_shape : tuple of int or int
        The shape of the window. If an integer is provided,
        a 2D window is generated.

    Notes
    -----
    This function is based on ``scipy.signal.get_window`` and thus can access
    all of the window types available to that function
    (e.g., ``"hann"``, ``"boxcar"``). Note that certain window types require
    parameters that have to be supplied with the window name as a tuple
    (e.g., ``("tukey", 0.8)``). If only a float is supplied, it is interpreted
    as the beta parameter of the Kaiser window.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html

    it is recommended to check the that after fft shift, the input array has value of 0
    at the border.

    """
    if isinstance(window_shape, int):
        window_shape = (window_shape, window_shape)
    win = window(window_type, window_shape)
    array_filtered = np.fft.fftshift(fftconvolve(np.fft.fftshift(input_array), win, mode="same"))
    return array_filtered / np.max(array_filtered)


def line_filter(
    sb_position,
    out_shape,
    orig_shape,
    length_ratio=0.9,
    width=20,
    crop_to_out_shape=True,
):
    """Return a line filter for the sideband.

    It can be applied by multiplying it with the aperture.

    `sb_position` is in native (unshifted) FFT coordinates of `orig_shape`, as
    in :attr:`HoloParams.sb_position <libertem_holo.base.utils.HoloParams>`.
    The result is a boolean mask in fftshifted layout (zero frequency at the
    center), cropped to `out_shape` with
    :func:`~libertem_holo.base.utils.get_slice_fft` if `crop_to_out_shape`
    is set.

    .. deprecated:: 1.1.0
        Use :func:`butterworth_line` instead, which is what
        :meth:`~libertem_holo.base.utils.HoloParams.from_hologram` uses.
    """
    warnings.warn(
        "line_filter is deprecated and will be removed in a future release; "
        "use butterworth_line instead",
        DeprecationWarning,
        stacklevel=2,
    )
    # we work in original shape, and crop at the end.
    dest = np.zeros(orig_shape, dtype=bool)

    # approx. positions of both sidebands (inferred from symmetry):
    sb_pos_shifted = fft_shift_coords(sb_position, orig_shape)

    _draw_lf_rect(
        dest,
        orig_shape=orig_shape,
        sb_position_shifted=sb_pos_shifted,
        length_ratio=length_ratio,
        width=width
    )

    if crop_to_out_shape:
        slice_fft = get_slice_fft(out_shape=out_shape, sig_shape=orig_shape)
        return dest[slice_fft]
    else:
        return dest


def central_line_filter(
    sb_position: tuple[int, int],
    out_shape: tuple[int, int],
    orig_shape: tuple[int, int],
    length_ratio: float = 0.9,
    width: float = 10.0,
    crop_to_out_shape: bool = True,
    order: int = 2,
) -> np.ndarray[int, int]:
    """Return a line filter for the central band.

    It can be applied by multiplying it with the aperture.

    Two lines from close to the center towards both sidebands, made with
    :func:`butterworth_line`.

    Parameters
    ----------
    sb_position
        Position of the sideband in native (unshifted) FFT coordinates of
        `orig_shape`, as in
        :attr:`HoloParams.sb_position <libertem_holo.base.utils.HoloParams>`.
        It is converted to fftshifted coordinates internally.

    out_shape
        Shape to crop to, if `crop_to_out_shape` is set.

    orig_shape
        Shape of the hologram.

    length_ratio, width, order
        See :func:`butterworth_line`.

    crop_to_out_shape
        Crop the result to `out_shape`, around the center, with
        :func:`~libertem_holo.base.utils.get_slice_fft`.

    Returns
    -------
    np.ndarray
        The filter in fftshifted layout, centered on the central band (zero
        frequency at the center). Apply :func:`numpy.fft.fftshift` after
        cropping to get the native layout used by
        :attr:`HoloParams.aperture_bf <libertem_holo.base.utils.HoloParams>`.

    """
    lf = butterworth_line(
        shape=orig_shape,
        width=width,
        sb_position=fft_shift_coords(sb_position, orig_shape),
        length_ratio=length_ratio,
        order=order,
        xp=np,
    )
    lf2 = butterworth_line(
        shape=orig_shape,
        width=width,
        sb_position=fft_shift_coords(other_sb(sb_position, orig_shape), orig_shape),
        length_ratio=length_ratio,
        order=order,
        xp=np,
    )

    dest = lf * lf2

    if crop_to_out_shape:
        slice_fft = get_slice_fft(out_shape=out_shape, sig_shape=orig_shape)
        return dest[slice_fft]
    else:
        return dest


class _LineGeometry(NamedTuple):
    r"""Line filter geometry, in pixel coordinates `(y, x)` of the output array.

    The output array is in fftshifted layout (zero frequency at the center),
    so these are fftshifted coordinates. The center `c` is the band the
    aperture is centered on, and `target` is the band the line points to::

          (1 - length_ratio) * |dir|
         |<------>|
         c ------ start ============== target ==============> edge of array
                 (                    the line continues past `target`
                  round cap at the start

         dir = target - c,  |dir| = target_dist

    For the sideband aperture, `c` is the sideband and `target` the central
    band; for the brightfield aperture, it is the other way around.

    FIXME: these offsets are kept so that the aperture doesn't change within a
    semver-compatible release:

    * The center `c` is at `shape / 2 - 1`; the zero frequency of an
      fftshifted array is at `shape // 2`, which is also what
      :func:`butterworth_disk` uses.
    * The axis of the line, and the center of its round cap, is at
      `start + (1, 0)`, i.e. 1 px below (+y) the start point.
    * The straight part begins at the perpendicular through
      `start + (0, -1)`, instead of at the center of the round cap.
    """

    # the line starts at (start_y, start_x)...
    start_y: float
    start_x: float
    # ...and extends in direction (dir_y, dir_x) up to the edge of the array:
    dir_y: float
    dir_x: float
    # dir_y / dir_x, the slope of the axis (inf if it is vertical):
    slope: float
    # the slope of the perpendicular, only used to decide whether a pixel is
    # past the start (inf if the axis is horizontal):
    perp_slope: float
    # sqrt(slope**2 + 1); dividing the residual of the axis' line equation,
    # |slope * x - y + intercept|, by it gives the perpendicular distance:
    dist_scale: float


def _butterworth_line_geometry(
    shape: tuple[int, int],
    target: tuple[float, float],
    length_ratio: float,
) -> _LineGeometry:
    """Compute the geometry of the line filter, see :class:`_LineGeometry`.

    Parameters
    ----------
    shape
        Shape of the output array, which is in fftshifted layout.

    target
        Position `(y, x)` the line points towards, in fftshifted coordinates
        of `shape`, i.e. an index into the output array.

    length_ratio
        See :func:`butterworth_line`.

    Returns
    -------
    _LineGeometry
        In the same (fftshifted) coordinates as `target`.

    """
    # center, see the FIXME in `_LineGeometry`:
    cy = shape[0] / 2 - 1
    cx = shape[1] / 2 - 1

    dir_y = target[0] - cy
    dir_x = target[1] - cx
    target_dist = math.sqrt(dir_y**2 + dir_x**2)
    if target_dist == 0:
        msg = (
            f"sb_position {tuple(target)} is at the center of the aperture, "
            "so the direction of the line filter is undefined"
        )
        raise ValueError(msg)
    length = target_dist * (1 - length_ratio)
    slope = dir_y / dir_x if dir_x != 0 else math.inf
    perp_slope = -1 / slope if dir_y != 0 else math.inf
    return _LineGeometry(
        start_y=cy + length * dir_y / target_dist,
        start_x=cx + length * dir_x / target_dist,
        dir_y=dir_y,
        dir_x=dir_x,
        slope=slope,
        perp_slope=perp_slope,
        dist_scale=math.sqrt(slope**2 + 1),
    )


# The kernel below is written in terms of slopes rather than vectors, so that
# the floating-point results, including which side of the start pixels exactly
# on the boundary end up on, are bit-identical to earlier versions.
@numba.njit(cache=True, inline="always")
def _butterworth_line_kernel(
    y: int, x: int, geometry: _LineGeometry, width: float, order: int,
) -> np.float32:
    """Value of the line filter at pixel `(y, x)`.

    `(y, x)` is an index into the output array, `geometry` is in the same
    coordinates. Returns 0 on the line and 1 far away from it. The `+ 1` /
    `- 1` below are the offsets marked FIXME in :class:`_LineGeometry`.
    """
    g = geometry
    off_y = y - g.start_y
    off_x = x - g.start_x

    # is the pixel past the start, in the direction of the line?
    # equivalent to `dir_y * off_y + dir_x * (off_x + 1) > 0`
    if g.dir_y == 0:
        past_start = g.dir_x * (off_x + 1) > 0
    else:
        intercept = off_y - g.perp_slope * off_x
        past_start = g.dir_y * (intercept - g.perp_slope) > 0

    if past_start:
        # straight part: perpendicular distance to the axis
        if g.dir_x == 0:
            dist = abs(off_x)
        else:
            dist = abs(g.slope * off_x - off_y + 1) / g.dist_scale
    else:
        # round cap: distance to its center
        dist = math.sqrt((off_y - 1)**2 + off_x**2)
    passed = 1 / math.sqrt(1 + math.pow((dist / width), 2 * order))
    return np.float32(1) - np.float32(passed)


@numba.njit(cache=True, parallel=True)
def _butterworth_line_cpu(
    shape: tuple[int, int], geometry: _LineGeometry, width: float, order: int,
) -> np.ndarray:
    """Line filter of the given `shape`, computed on the CPU.

    `geometry` is in pixel coordinates of the result.
    """
    result = np.empty(shape, dtype=np.float32)
    for y in numba.prange(shape[0]):
        for x in range(shape[1]):
            result[y, x] = _butterworth_line_kernel(y, x, geometry, width, order)
    return result


@cuda.jit
def _butterworth_line_gpu(
    result: np.ndarray, geometry: _LineGeometry, width: float, order: int,
) -> None:
    """Write the line filter into `result`, on the GPU.

    `geometry` is in pixel coordinates of `result`.
    """
    y, x = cuda.grid(2)
    if x < result.shape[1] and y < result.shape[0]:
        result[y, x] = _butterworth_line_kernel(y, x, geometry, width, order)


def butterworth_line(
    shape: tuple[int, int],
    width: float,
    sb_position: tuple[float, float],
    length_ratio: float = 0.9,
    order: int = 12,
    xp=np
):
    """Generate a line filter.

    This is useful to remove fresnel fringes. The result is a mask that is one
    everywhere except for a line with butterworth-filtered edges, which
    is zero. The line starts close to the center of the array and points
    towards `sb_position`, extending up to the edge of the array.

    Both `sb_position` and the result are in fftshifted layout (zero frequency
    at the center), like :func:`butterworth_disk`. To get the native layout
    used by :class:`~libertem_holo.base.utils.HoloParams`, crop with
    :func:`~libertem_holo.base.utils.get_slice_fft` and apply
    :func:`numpy.fft.fftshift`, as in
    :meth:`~libertem_holo.base.utils.HoloParams.from_hologram`.

    Parameters
    ----------
    shape
        output shape of the aperture

    width
        width of the line in pixels

    sb_position
        Position `(y, x)` that the line points towards, in fftshifted
        coordinates of `shape`, i.e. an index into the result. For an aperture
        centered on the sideband, this is where the central band is (in
        :meth:`~libertem_holo.base.utils.HoloParams.from_hologram`, that is
        :code:`fft_shift_coords(sb_position, shape)`, with the native
        :attr:`HoloParams.sb_position <libertem_holo.base.utils.HoloParams>`),
        and vice versa.

    length_ratio
        Length of the line from its start to `sb_position`, as a fraction of
        the distance from the center to `sb_position`. The part close to the
        center is left out.

    order
        order of the butterworth filter

    xp
        Either numpy or cupy
    """
    geometry = _butterworth_line_geometry(
        shape=shape,
        target=sb_position,
        length_ratio=length_ratio,
    )
    if xp is np:
        return _butterworth_line_cpu(shape, geometry, width, order)
    else:
        import cupy as cp
        result = cp.empty(shape, dtype=np.float32)
        size = 16
        threadsperblock = (size, size)
        blockspergrid_x = math.ceil(result.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(result.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        _butterworth_line_gpu[blockspergrid, threadsperblock](
            result, geometry, width, order,
        )
        return result


def hanning_2d(shape: tuple[int, int], xp=np):
    return xp.outer(xp.hanning(shape[0]), xp.hanning(shape[1]))

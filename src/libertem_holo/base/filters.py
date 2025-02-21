"""Useful image filtering helpers."""
import math

import numpy as np
import numba
from numba import cuda
import sparse
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from skimage.filters import window
from skimage.restoration import unwrap_phase

from libertem.masks import radial_bins
from libertem_holo.base.utils import (
    fft_shift_coords, other_sb, draw_lf_rect, get_slice_fft,
)


def disk_aperture(out_shape: tuple[int, int], radius: float, xp=np) -> np.ndarray:
    """Generate a disk-shaped aperture, fft-shifted.

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

    return xp.fft.fftshift(bins[0])


def butterworth_disk(shape: tuple[int, int], radius: float, order: int = 12, xp=np):
    if xp is np:
        return _butterworth_disk_cpu(shape, radius, order)
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


@numba.njit(cache=True)
def _butterworth_disk_cpu(shape: tuple[int, int], radius: float, order: int = 12):
    result = np.zeros(shape, dtype=np.float32)
    cy = shape[0]/2
    cx = shape[1]/2
    for y in range(shape[0]):
        for x in range(shape[1]):
            d = np.sqrt((y-cy)**2 + (x-cx)**2)
            result[y, x] = 1/np.sqrt(1 + np.pow((d/radius), 2*order))
    return result


@cuda.jit(cache=True)
def _butterworth_disk_gpu(result, radius: float, order: int = 12):
    y, x = cuda.grid(2)
    cy = result.shape[0]/2
    cx = result.shape[1]/2
    if x < result.shape[1] and y < result.shape[0]:
        d = math.sqrt((y-cy)**2 + (x-cx)**2)
        result[y, x] = 1/math.sqrt(1 + math.pow((d/radius), 2*order))


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


def phase_ramp_finding(img, order=1):
    """Find a phase ramp in `img`.

    A phase ramp finding function that is used to find the phase ramp across the
    field of view.

    Parameters
    ----------
    img : 2d nd array
        Complex image or phase image.
    order : int
        Phase ramp, 1 (default) is linear.
    ramp : 2d tuple, ()
        Phase ramp in y, x, if not None.

    Returns
    -------
        ramp, order, tuple, float

    """
    # The ramp is determined by the maximum and minimum values of the image.
    # TODO least-square-fitting, polynomial order
    # TODO How to find phase ramps in complex images
    if img.dtype.kind != 'c':
        if order == 1:
            ramp_x = np.mean(np.gradient(img, axis=0))
            ramp_y = np.mean(np.gradient(img, axis=1))
            ramp = (ramp_y, ramp_x)
        else:
            raise ValueError(f"can only handle `order=1` for now, not order={order}")
    else:
        raise ValueError(f"cannot handle input of type {img.dtype}")

    return ramp


def phase_ramp_removal(size, order=1, ramp=None):
    """Remove phase ramp.

    A phase ramp removal function that finds and removes a phase ramp across the
    field of view.

    Parameters
    ----------
    size : 2d tuple, ()
        Size of the Complex image or phase image
    order : int
        Phase ramp, 1 (default) is linear.
    ramp : 2d tuple, ()
        Phase ramp in y, x, if not None.

    Returns
    -------
        2d nd array of the corrected image

    """
    # TODO How to find phase ramps in complex images
    img = np.zeros(size)
    if ramp is None:
        ramp = phase_ramp_finding(size, order=1)
    else:
        (ramp_y, ramp_x) = ramp
    yy = np.arange(0, size[0], 1)
    xx = np.arange(0, size[1], 1)
    y, x = np.meshgrid(yy, xx)
    if order == 1:
        img = ramp_x * x + ramp_y * y
    else:
        # To be expanded.
        raise ValueError(f"cannot handle order={order}")

    return img


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


def ramp_compensation(image):
    """
    A ramp or wedge compensation for a 2D image with a linear optimization methods.

    Parameters
    ----------
    image : 2D-Array
        Input array
    """

    def linear_gradient(c, dy, dx, y, x):
        return c+y*dy+x*dx
    x = np.linspace(0, image.shape[0]-1, image.shape[0])
    y = np.linspace(0, image.shape[1]-1, image.shape[1])

    def fun(initial_value):
        function = image_not_compensated - linear_gradient(initial_value[0], initial_value[1],
                                                           initial_value[2], yv, xv)
        return function.reshape((-1, ))
    yv, xv = np.meshgrid(y, x)

    image_not_compensated = np.copy(image)
    m_initial = np.gradient(image_not_compensated)
    dy_initial = np.mean(m_initial[0])
    dx_initial = np.mean(m_initial[1])
    c_initial = image[0, 0]
    initial_value = np.array([c_initial, dy_initial, dx_initial])
    res1 = least_squares(fun, initial_value)
    gradient_compensation = linear_gradient(res1.x[0], res1.x[1],
                                            res1.x[2], yv, xv)
    return image - gradient_compensation


def line_filter(
    sb_position,
    out_shape,
    orig_shape,
    length_ratio=0.9,
    width=20,
    crop_to_out_shape=True,
):
    """Return a line filter for the sideband.

    It can be applied by multiplying it with the aperture. The filter has the
    zero frequency at the center of the image.
    """
    # we work in original shape, and crop at the end.
    dest = np.zeros(orig_shape, dtype=bool)

    # approx. positions of both sidebands (inferred from symmetry):
    sb_pos_shifted = fft_shift_coords(sb_position, orig_shape)

    draw_lf_rect(
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
    sb_position,
    out_shape,
    orig_shape,
    length_ratio=0.9,
    width=20,
    crop_to_out_shape=False,
):
    """
    Return a line filter for the central band, that can be applied
    by multiplying it with the aperture.
    """
    # we are working in npn-fft-shifted space, meaning with the zero
    # frequency at the center of the image. we work in original shape,
    # and crop at the end.
    dest = np.zeros(orig_shape, dtype=bool)

    # approx. positions of both sidebands (inferred from symmetry):
    sb_pos_shifted = fft_shift_coords(sb_position, orig_shape)

    other_sb_pos = fft_shift_coords(other_sb(sb_position, orig_shape), orig_shape)

    draw_lf_rect(
        dest,
        orig_shape=orig_shape,
        sb_position_shifted=sb_pos_shifted,
        length_ratio=length_ratio,
        width=width,
    )
    draw_lf_rect(
        dest,
        orig_shape=orig_shape,
        sb_position_shifted=other_sb_pos,
        length_ratio=length_ratio,
        width=width,
    )

    if crop_to_out_shape:
        slice_fft = get_slice_fft(out_shape=out_shape, sig_shape=orig_shape)
        return dest[slice_fft]
    else:
        return dest


@numba.njit
def _butterworth_line_cpu(shape, width, sb_position, length_ratio=0.9, order=12):
    result = np.zeros(shape, dtype=np.float32)
    cy = shape[0] / 2 - 1
    cx = shape[1] / 2 - 1
    a = (sb_position[0] - cy) / (sb_position[1] - cx)
    b = 1

    # determine starting point:
    sb_dist = np.sqrt((sb_position[0] - cy)**2 + (sb_position[1] - cx)**2)
    length = sb_dist * (1 - length_ratio)

    sb_sel = np.sign(sb_position[0] - cy)

    # shift to starting point
    cy += length * (sb_position[0] - cy) / sb_dist
    cx += length * (sb_position[1] - cx) / sb_dist

    c = -1/a
    for y in range(shape[0]):
        for x in range(shape[1]):
            x0 = x - cx
            y0 = y - cy
            d = y0 - c * x0
            xc = (d-c)/(a-b)

            if sb_sel * xc < 0:
                dist = np.abs(a*x0 - y0 + b)/np.sqrt(a**2 + 1)
            else:
                dist = np.sqrt((y-cy-1)**2 + (x-cx)**2)
            result[y, x] = 1/np.sqrt(1 + np.pow((dist/width), 2*order))
    return 1 - result


def butterworth_line(
    shape: tuple[int, int],
    width: float,
    sb_position: tuple[int, int],
    length_ratio: float = 0.9,
    order: int = 12,
    xp=np
):
    """
    shape: output shape of the aperture

    width: width of the line in pixels

    order: order of the butterworth filter
    """
    if xp is np:
        return _butterworth_line_cpu(
            shape=shape,
            width=width,
            sb_position=sb_position,
            length_ratio=length_ratio,
            order=order,
        )
    else:
        import cupy as cp
        result = cp.zeros(shape, dtype=np.float32)
        size = 16
        threadsperblock = (size, size)
        blockspergrid_x = math.ceil(result.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(result.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        _butterworth_line_gpu[blockspergrid, threadsperblock](
            result,
            width,
            sb_position,
            length_ratio,
            order,
        )
        return 1 - result


@cuda.jit
def _butterworth_line_gpu(
    result, width, sb_position, length_ratio=0.9, order=12
):
    y, x = cuda.grid(2)
    if x < result.shape[1] and y < result.shape[0]:
        cy = result.shape[0] / 2 - 1
        cx = result.shape[1] / 2 - 1
        a = (sb_position[0] - cy) / (sb_position[1] - cx)
        b = 1

        # determine starting point:
        sb_dist = math.sqrt((sb_position[0] - cy)**2 + (sb_position[1] - cx)**2)
        length = sb_dist * (1 - length_ratio)

        if sb_position[0] - cy >= 0:
            sb_sel = 1
        else:
            sb_sel = -1

        # shift to starting point
        cy += length * (sb_position[0] - cy) / sb_dist
        cx += length * (sb_position[1] - cx) / sb_dist

        c = -1/a
        x0 = x - cx
        y0 = y - cy
        d = y0 - c * x0
        xc = (d-c)/(a-b)

        if sb_sel * xc < 0:
            dist = abs(a*x0 - y0 + b)/math.sqrt(a**2 + 1)
        else:
            dist = math.sqrt((y-cy-1)**2 + (x-cx)**2)
        result[y, x] = 1/math.sqrt(1 + math.pow((dist/width), 2*order))

"""Useful image filtering helpers."""
import numpy as np
import sparse
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from skimage.filters import window
from skimage.restoration import unwrap_phase

from libertem_holo.base.utils import (
    fft_shift_coords, other_sb, draw_lf_rect, get_slice_fft,
)


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
    crop_to_out_shape=False,
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

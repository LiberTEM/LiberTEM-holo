import sparse
import numpy as np
from scipy import ndimage
from skimage.restoration import unwrap_phase
from skimage.filters import window
from scipy.signal import fftconvolve


def highpass(img, sigma=2):
    """
    Return highpass by subtracting a gaussian lowpass filter
    """
    return img - ndimage.gaussian_filter(img, sigma=sigma)


def exclusion_mask(img, sigma=6):
    """
    Return a mask with `True` entries for pixels deviating more than `sigma` from the mean
    """
    return np.abs(img) > (img.mean() + sigma * img.std())


def clipped(img, sigma=6):
    """
    Return `img`, but with pixels deviating more than `sigma` from the mean masked out

    Useful for plotting:

    >>> plt.imshow(img, vmax=np.max(clipped(img)))
    """
    sigma_mask = exclusion_mask(img)
    sigma_mask = ~sigma_mask
    return img[sigma_mask]


def phase_ramp_finding(img, order=1):
    """
    A phase ramp finding function that is used to find the phase ramp across the field of view.

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
            pass
    else:
        pass

    return ramp


def phase_ramp_removal(size, order=1, ramp=None):
    """
    A phase ramp removal function that is remove to find the phase ramp across the field of view.

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
        pass

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
    """
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
        buffer=img.reshape((1,) + img.shape),
        excluded_pixels=coords.coords,
        sig_shape=tuple(img.shape)
    ).squeeze()


def window_filter(input_array, window_type, window_shape):
    """
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
    array_filtered = array_filtered / np.max(array_filtered)
    return array_filtered

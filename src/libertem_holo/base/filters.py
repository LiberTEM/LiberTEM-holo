import sparse
import numpy as np
from scipy import ndimage
from skimage.restoration import unwrap_phase
from numpy import unwrap as np_unwrap


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


def phase_unwrap(image, alg=1):
    """
    A phase_unwrap function that is unwrap the complex / wrapped phase image.

    Parameters
    ----------
    image : 2d nd array
        Complex or Wrapped phase image
    alg : int
        Define which algorithm for phase unwrapping.
        1 by default to use skimage. 2 to use numpy.umwrap
    Returns
    -------
        2d nd array of the unwrapped phase image
    """

    if image.dtype.kind == 'c':
        img = np.angle(image)
    else:
        img = image
    if alg == 1:
        image_new = unwrap_phase(img)

    elif alg == 2:
        image_new = np_unwrap(img)

    else:
        pass

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

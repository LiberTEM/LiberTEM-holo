import numpy as np
from scipy import ndimage
from skimage.restoration import unwrap_phase


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
    if img is not complex:
        if order == 1:
            ramp_x = np.mean(np.gradient(img, axis=0))
            ramp_y = np.mean(np.gradient(img, axis=1))
            ramp = (ramp_y, ramp_x)
        else:
            pass
    if img is complex:
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

    image_new = np.zeros_like(image)
    if image is not complex:
        image_new = unwrap_phase(image)
    else:
        angle = np.angle(image)
        image_new = unwrap_phase(angle)

    return image_new
import numpy as np
from scipy import ndimage


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

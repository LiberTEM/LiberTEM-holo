import numpy as np
from skimage.restoration import unwrap_phase

from .laplace import prepare_laplacian_unwrap, unwrap_phase_laplacian
from .quality import derivative_variance, quality_unwrap


def phase_unwrap(image: np.ndarray[tuple[int, int]]) -> np.ndarray[tuple[int, int]]:
    """Unwrap the complex / wrapped phase image.

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


__all__ = [
    "derivative_variance",
    "phase_unwrap",
    "prepare_laplacian_unwrap",
    "quality_unwrap",
    "unwrap_phase_laplacian",
]

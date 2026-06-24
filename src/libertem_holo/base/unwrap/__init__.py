"""Various phase unwrapping tools and algorithms."""
from typing import Literal

import numpy as np
import numpy.dtypes as nd
from skimage.restoration import unwrap_phase

from .laplace import prepare_laplacian_unwrap, unwrap_phase_laplacian
from .quality import derivative_variance, quality_unwrap

ComplexOrFloat = (
    nd.Float16DType | nd.Float32DType | nd.Float64DType
    | nd.Complex64DType | nd.Complex128DType
)


def phase_unwrap(
    image: np.ndarray[tuple[int, int], ComplexOrFloat],
    method: Literal["skimage", "quality", "laplacian"] = "skimage",
) -> np.ndarray[tuple[int, int]]:
    """Unwrap phase from wrapped phase or complex wave.

    You can choose different methods, which might perform better or worse depending
    on the input data. For tweaking the unwrapping, please call the specific unwrap
    methods directly.

    Parameters
    ----------
    image
        Complex wave or wrapped phase image

    method
        The method to use for unwrapping the phase.
            - skimage: use the unwrap_phase function from scikit-image
            - quality: unwrap based on the variance of the derivative
            - laplacian: qualitative unwrapping based on

    Returns
    -------
        2d nd array of the unwrapped phase image

    """
    phase = np.angle(image) if image.dtype.kind == "c" else image

    if method == "skimage":
        return unwrap_phase(phase)
    if method == "quality":
        quality = derivative_variance(phase)
        return quality_unwrap(phase, quality)
    if method == "laplacian":
        return unwrap_phase_laplacian(wrapped_phase=phase)
    msg = f"unknown phase unwrapping method '{method}'"  # pyright: ignore[reportUnreachable]
    raise ValueError(msg)


__all__ = [
    "derivative_variance",
    "phase_unwrap",
    "prepare_laplacian_unwrap",
    "quality_unwrap",
    "unwrap_phase_laplacian",
]

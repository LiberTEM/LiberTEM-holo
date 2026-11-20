"""Various phase unwrapping tools and algorithms."""
from typing import Literal

import numpy as np
import numpy.dtypes as nd
from skimage.restoration import unwrap_phase

from .irls import (
    BranchesPrior,
    CGRound,
    IRLSOptions,
    IRLSResult,
    PhasePrior,
    amplitude_edge_weights,
    unwrap_phase_irls,
)
from .laplace import prepare_laplacian_unwrap, unwrap_phase_laplacian
from .quality import derivative_variance, quality_unwrap

ComplexOrFloat = (
    nd.Float16DType | nd.Float32DType | nd.Float64DType
    | nd.Complex64DType | nd.Complex128DType
)


def phase_unwrap(
    image: np.ndarray[tuple[int, int], ComplexOrFloat],
    method: Literal["skimage", "quality", "laplacian", "irls"] = "skimage",
) -> np.ndarray[tuple[int, int]]:
    """Unwrap phase from wrapped phase or complex wave.

    You can choose different methods, which might perform better or worse depending
    on the input data. For tweaking the unwrapping, please call the specific unwrap
    methods directly.

    Parameters
    ----------
    image
        Complex wave or wrapped phase image. Prefer the complex wave: methods
        that can use its amplitude (:code:`"irls"`) then weight the image by it,
        so that discontinuities go where the amplitude is low.

    method
        The method to use for unwrapping the phase.
            - skimage: use the unwrap_phase function from scikit-image
            - quality: unwrap based on the variance of the derivative
            - laplacian: least-squares unwrapping by the Laplacian, see
              :func:`unwrap_phase_laplacian`; not congruent with the input
            - irls: iteratively reweighted least squares, see
              :func:`unwrap_phase_irls`; for a complex wave, with edge weights
              from its amplitude

    Returns
    -------
        2d nd array of the unwrapped phase image

    """
    if method == "irls":
        # Takes the complex wave, for edge weights from its amplitude:
        return unwrap_phase_irls(image).phase
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
    "BranchesPrior",
    "CGRound",
    "IRLSOptions",
    "IRLSResult",
    "PhasePrior",
    "amplitude_edge_weights",
    "derivative_variance",
    "phase_unwrap",
    "prepare_laplacian_unwrap",
    "quality_unwrap",
    "unwrap_phase_irls",
    "unwrap_phase_laplacian",
]

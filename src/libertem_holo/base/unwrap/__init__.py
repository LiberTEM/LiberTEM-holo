"""Various phase unwrapping tools and algorithms."""
from typing import Any, Literal, TypeVar

from skimage.restoration import unwrap_phase

from libertem_holo.base.align import _array_module
from libertem_holo.base.utils import to_cpu

from .admm import ADMMOptions, unwrap_phase_admm
from .irls import (
    IRLSOptions,
    L1BranchesPrior,
    L1PhasePrior,
    L1Result,
    L1Round,
    amplitude_edge_weights,
    unwrap_phase_irls,
)
from .laplace import prepare_laplacian_unwrap, unwrap_phase_laplacian
from .quality import derivative_variance, quality_unwrap

# numpy or cupy arrays: `phase_unwrap` returns the kind it is given.
ArrayT = TypeVar("ArrayT")

_METHODS = ("skimage", "quality", "laplacian", "irls", "admm")


def phase_unwrap(
    image: ArrayT,
    method: Literal["skimage", "quality", "laplacian", "irls", "admm"] = "skimage",
    **kwargs: Any,  # noqa: ANN401 (forwarded to the method's function)
) -> ArrayT:
    """Unwrap phase from wrapped phase or complex wave.

    You can choose different methods, which might perform better or worse depending
    on the input data. For tweaking the unwrapping, pass the keyword arguments of
    the method's function, or call it directly.

    Parameters
    ----------
    image
        Complex wave or wrapped phase image, as a numpy or a cupy array. Prefer
        the complex wave: methods that can use its amplitude (:code:`"irls"`,
        :code:`"admm"`) then weight the image by it, so that discontinuities go
        where the amplitude is low.
    method
        The method to use for unwrapping the phase.

            - skimage: use the unwrap_phase function from scikit-image (on the
              CPU; `kwargs` go to it)
            - quality: unwrap based on the variance of the derivative (on the
              CPU; `kwargs` go to :func:`derivative_variance`)
            - laplacian: least-squares unwrapping by the Laplacian, see
              :func:`unwrap_phase_laplacian`; not congruent with the input
            - irls: L1 unwrapping by iteratively reweighted least squares, see
              :func:`unwrap_phase_irls`; for a complex wave, with edge weights
              from its amplitude
            - admm: L1 unwrapping by ADMM, see :func:`unwrap_phase_admm`; the same
              objective as irls, faster
    kwargs
        Keyword arguments of the method's function, for example `options` or
        `prior` for :code:`"irls"` and :code:`"admm"`.

    Returns
    -------
        The unwrapped phase, an array of the same kind as `image`: for a cupy
        array, the CPU-only methods copy it to the host and their result back.

    """
    if method not in _METHODS:
        msg = f"unknown phase unwrapping method '{method}'"
        raise ValueError(msg)
    if method == "irls":
        # Takes the complex wave, for edge weights from its amplitude:
        return unwrap_phase_irls(image, **kwargs).phase
    if method == "admm":
        return unwrap_phase_admm(image, **kwargs).phase
    xp = _array_module(image)
    phase = xp.angle(image) if image.dtype.kind == "c" else image
    if method == "laplacian":
        return unwrap_phase_laplacian(wrapped_phase=phase, xp=xp, **kwargs)
    # The CPU-only methods:
    phase = to_cpu(phase)
    if method == "skimage":
        result = unwrap_phase(phase, **kwargs)
    else:
        result = quality_unwrap(phase, derivative_variance(phase, **kwargs))
    return xp.asarray(result)


__all__ = [
    "ADMMOptions",
    "IRLSOptions",
    "L1BranchesPrior",
    "L1PhasePrior",
    "L1Result",
    "L1Round",
    "amplitude_edge_weights",
    "derivative_variance",
    "phase_unwrap",
    "prepare_laplacian_unwrap",
    "quality_unwrap",
    "unwrap_phase_admm",
    "unwrap_phase_irls",
    "unwrap_phase_laplacian",
]

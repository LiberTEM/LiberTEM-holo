"""Phase unwrapping tools and algorithms.

:func:`phase_unwrap` gives access to all methods. The L1 methods,
:func:`unwrap_phase_irls` and :func:`unwrap_phase_admm`, run on the GPU for
cupy input, and then return cupy arrays.

Invalid pixels
--------------

For the L1 methods, pixels whose input is not finite (NaN or infinite) are
invalid. The solve treats their phase as 0, and gives the edges that touch them
a weight of :code:`1e-3`, where the largest weight is 1, so that cuts run
through them rather than through the data. These edges do not count in the cut
turns and the cut cost. The result is NaN at invalid pixels, with branch 0.

GPU memory and threads
----------------------

On the GPU, a solve keeps its buffers, FFT plans and CUDA graphs in a
:class:`~libertem_holo.base.workspace.Workspace`, for the next solve of the same
shape, dtype, method and device. Setting them up costs more than a solve of a
small image: 20 to 75 ms on a GH200, most of it in Python on the CPU. A
workspace keeps one set; a solve with other inputs replaces it, and frees the
old one first. The set takes about 33 to 43 times the memory of the image in
the precision of the solve (float32 for complex64 input): 30 to 65 MiB for
470 by 460 pixels, 0.6 to 1.2 GiB for 2048 by 2048.

Without a :code:`workspace` argument, solves use the default workspace. Solves
in several threads then share it, one at a time. For parallel solves, give each
thread its own workspace:

.. code-block:: python

    from libertem_holo.base.workspace import Workspace

    def unwrap_frames(frames):
        with Workspace() as workspace:  # cleared at the end
            return [unwrap_phase_admm(frame, workspace=workspace) for frame in frames]

:func:`~libertem_holo.base.workspace.clear_default_workspace` frees the default
workspace, and returns the unused memory of cupy's memory pools.

The first GPU solve in a process compiles the CUDA kernels, about 8 s on a GH200
(for one dtype). They are cached on disk: later processes load them in under a
second. Setting up a solve in a workspace captures CUDA graphs. A device-wide
synchronization, such as :code:`cupy.cuda.Device().synchronize()`, in another
thread at that time makes the capture fail: other threads should synchronize
their own streams instead.
"""
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
        :code:`"admm"`) then derive edge weights from it, so that
        discontinuities go where the amplitude is low.
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
        msg = (
            f"unknown phase unwrapping method {method!r}, "
            f"expected one of {', '.join(_METHODS)}"
        )
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

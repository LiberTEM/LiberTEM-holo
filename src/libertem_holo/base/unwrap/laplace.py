"""2D Laplacian Phase Unwrapping."""
from dataclasses import dataclass

import numpy as np


@dataclass
class LaplaceParams:
    """Container for reusable unwrapping parameters."""

    del_op: np.ndarray
    del_inv: np.ndarray


def prepare_laplacian_unwrap(shape: tuple[int, int], xp=np) -> LaplaceParams:
    """Prepare unwrap parameteres for a given shape.

    Create the analytical discrete laplace operator and it's inverse
    in the frequency domain.

    You can use this if you plan to repeatedly unwrap
    phases of the same shape.
    """
    rows, cols = shape
    u = xp.arange(rows)[:, xp.newaxis]
    v = xp.arange(cols)[xp.newaxis, :]

    del_op = 1 * (-4 + 2 * xp.cos(2 * xp.pi * u / rows) + 2 * xp.cos(2 * xp.pi * v / cols))

    # Inverse operator:
    del_inv = xp.zeros_like(del_op)
    mask = del_op != 0
    del_inv[mask] = 1 / del_op[mask]
    return LaplaceParams(del_op=del_op, del_inv=del_inv)


def unwrap_phase_laplacian(
    wrapped_phase: np.ndarray[tuple[int, int]],
    params: LaplaceParams | None = None,
    xp=np,
) -> np.ndarray[tuple[int, int]]:
    """2D Laplacian Phase Unwrapping.

    Note that the result is qualitative and not guaranteed to be
    an integer multiple of 2π from the original phase. A potential use
    case is a quick phase preview, as this method is GPU-accelerated
    by passing in `xp=cp`. The result can also be used as a starting
    point for further processing.

    Implements part of: Marvin A. Schofield and Yimei Zhu, "Fast phase
    unwrapping algorithm for interferometric applications," Opt. Lett. 28,
    1194-1196 (2003) https://doi.org/10.1364/OL.28.001194

    Parameters:
    -----------
    wrapped_phase
        The input phase, wrapped between -π and +π

    params
        Optional pre-computed parameters using `prepare_laplacian_unwrap`,
        useful if many phases of the same shape need to be
        unwrapped.

    xp
        numpy or cupy array module

    """
    fft2 = xp.fft.fft2
    ifft2 = xp.fft.ifft2

    if params is None:
        params = prepare_laplacian_unwrap(wrapped_phase.shape, xp=xp)
    del_op = params.del_op
    del_inv = params.del_inv

    exp_phase = xp.exp(1j * wrapped_phase)

    # forward discrete laplace operator:
    laplacian_exp = ifft2(fft2(exp_phase) * del_op)

    del_phase = xp.imag(xp.conj(exp_phase) * laplacian_exp)

    # inverse discrete laplace operator:
    unwrapped_phase = ifft2(fft2(del_phase) * del_inv).real

    return unwrapped_phase

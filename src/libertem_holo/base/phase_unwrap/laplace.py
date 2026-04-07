import numpy as np
from dataclasses import dataclass


@dataclass
class LaplaceParams:
    del_op: np.ndarray
    del_inv: np.ndarray


def prepare_unwrap_for_shape(shape: tuple[int, int], xp=np) -> LaplaceParams:
    """
    Create the analytical discrete laplace operator and it's inverse
    in the frequency domain
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
    wrapped_phase: np.ndarray,
    params: LaplaceParams | None = None,
    xp=np,
):
    """
    2D Laplacian Phase Unwrapping. `params` are pre-computed
    using `prepare_unwrap_for_shape`.

    Implements part of: Marvin A. Schofield and Yimei Zhu, "Fast phase
    unwrapping algorithm for interferometric applications," Opt. Lett. 28,
    1194-1196 (2003) https://doi.org/10.1364/OL.28.001194
    """
    fft2 = xp.fft.fft2
    ifft2 = xp.fft.ifft2

    if params is None:
        params = prepare_unwrap_for_shape(wrapped_phase.shape)
    del_op = params.del_op
    del_inv = params.del_inv

    exp_phase = xp.exp(1j * wrapped_phase)

    # forward discrete laplace operator:
    laplacian_exp = ifft2(fft2(exp_phase) * del_op)

    del_phase = xp.imag(xp.conj(exp_phase) * laplacian_exp)

    # inverse discrete laplace operator:
    unwrapped_phase = ifft2(fft2(del_phase) * del_inv).real

    return unwrapped_phase

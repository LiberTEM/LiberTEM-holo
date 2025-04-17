"""Core reconstruction functions of LiberTEM-holo.

This includes both FFT-based and phase-shifting approaches.
"""

# Functions freq_array, aperture_function, estimate_sideband_position
# estimate_sideband_size are adopted from Hyperspy
# and are subject of following copyright:
#
#  Copyright 2007-2016 The HyperSpy developers
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2019 The LiberTEM developers
#
#  LiberTEM is distributed under the terms of the GNU General
# Public License as published by the Free Software Foundation,
# version 3 of the License.
# see: https://github.com/LiberTEM/LiberTEM
from __future__ import annotations

import typing
from typing import Literal
import time
from sparseconverter import NUMPY, for_backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import logging
from scipy.sparse.linalg import eigsh

from libertem_holo.base.filters import phase_unwrap
from libertem_holo.base.utils import get_slice_fft, HoloParams

log = logging.getLogger(__name__)

XPType = typing.Any  # Union[Module("numpy"), Module("cupy")]


def reconstruct_frame(
    frame: np.ndarray,
    sb_pos: tuple[float, float],
    aperture: np.ndarray,
    slice_fft: tuple[slice, slice],
    *,
    precision: bool = True,
    xp: XPType = np,
) -> np.ndarray:
    """Reconstruct a single hologram.

    Parameters
    ----------
    frame
        A numpy or cupy array containing the input hologram
    sb_pos
        The sideband position, for example as returned by
        `estimate_sideband_position`
    aperture
        A numpy or cupy array containing the aperture to apply in fourier space
    slice_fft
        A slice to crop to the selected output shape in fourier space,
        as returned by `get_slice_fft`.
    precision
        Defines precision of the reconstruction, True for complex128 for the
        resulting complex wave, otherwise results will be complex64
    xp
        Pass in either the numpy or cupy module to select CPU or GPU processing

    """
    frame = xp.array(frame)

    if not precision:
        frame = frame.astype(np.float32)
    frame_size = frame.shape

    fft_frame = xp.fft.fft2(frame) / np.prod(frame_size)
    fft_frame = xp.roll(fft_frame, xp.array(sb_pos).astype(xp.int64), axis=(0, 1))

    fft_frame = xp.fft.fftshift(xp.fft.fftshift(fft_frame)[slice_fft])

    fft_frame = fft_frame * aperture

    return xp.fft.ifft2(fft_frame) * np.prod(frame_size)


def reconstruct_double_resolution(
    frames: np.ndarray,
    sb_pos: tuple[float, float],
    aperture: np.ndarray,
    slice_fft: tuple[slice, slice],
    *,
    precision: bool = True,
    xp: XPType = np,
) -> np.ndarray:
    """Reconstruct a stack of phase shifted holography with double resolution method.

    Parameters
    ----------
    frames : array_like
        Two holograms taken at a phase offset of π; shape (2, height, width)
    sb_pos
        The sideband position, for example as returned by
        `estimate_sideband_position`
    aperture
        A numpy or cupy array containing the aperture to apply in fourier space
    slice_fft
        A slice to crop to the selected output shape in fourier space,
        as returned by `get_slice_fft`.
    precision
        Defines precision of the reconstruction, True for complex128 for the
        resulting complex wave, otherwise results will be complex64
    xp
        Pass in either the numpy or cupy module to select CPU or GPU processing

    Returns
    -------
    wav : complex array
        the reconstructed complex image

    """
    image_double_resolution = frames[1] - frames[0]
    return reconstruct_frame(
        image_double_resolution,
        sb_pos,
        aperture,
        slice_fft,
        precision=precision,
        xp=xp,
    )


def estimate_omega(
    image: np.ndarray,
    sideband_position: tuple[float, float],
    *,
    flip: bool = False,
) -> tuple[float, float]:
    """Estimates the frequency carrier of the hologram.

    Parameters
    ----------
    image : array
            Holographic data array
    sideband_position : tuple
        The sideband position (y, x), referred to the non-shifted FFT.
    flip
        ??? TODO

    Returns
    -------
    omega : tuple
        frequency carrier in y and x axis

    """
    width = image.shape[-1]

    if sideband_position[1] >= width / 2:
        omega = (-sideband_position[0], (width - sideband_position[1]))
    elif sideband_position[1] < width / 2:
        omega = (-sideband_position[0], sideband_position[1])

    if flip:
        omega = (-omega[0], -omega[1])

    return omega


def reconstruct_direct_euler(
    image: np.ndarray,
    omega: tuple[float, float],
) -> np.ndarray:
    """Reconstruct a stack of phase shifted holograms.

    This is using the direct reconstruction method by Ru et.al. 1994 (euler
    form)

    Parameters
    ----------
    image : array
        Holographic data array
    omega: tuple
        frequency carrier in y and x axis

    Returns
    -------
    phase final : 2D array
        the reconstructed phase image

    """
    number_of_images = image.shape[1]
    phase_initial_euler = np.zeros(number_of_images, dtype="complex128")

    n = np.arange(number_of_images)
    phase_initial = 2 * np.pi * n / number_of_images
    phase_initial_euler = np.exp(1j * phase_initial)

    c22 = 0
    for i in range(number_of_images):
        c22 = c22 + (image[i] * phase_initial_euler[i])
    c22 = c22 / number_of_images

    x = np.linspace(0, omega[1], image.shape[2], endpoint=False)
    y = np.linspace(0, omega[0], image.shape[1], endpoint=False)
    irow, icol = np.meshgrid(x, y, indexing="xy")

    ramp_carrier = np.exp(1j * 2 * np.pi * (irow + icol))

    c22 = c22 / ramp_carrier
    return np.angle(c22)


def reconstruct_direct(
    stack: np.ndarray,
    omega: tuple[float, float],
) -> np.ndarray:
    """Reconstruct a stack of phase shifted holograms.

    This is using the direct reconstruction method.

    Parameters
    ----------
    stack : array_like
        Stack of holograms
    number_of_images : int
        The number of images inside the stack
    omega: tuple
        frequency carrier in y and x axis

    Returns
    -------
    phase final : 2D array
        the reconstructed phase image

    """
    xspace = omega[1]
    yspace = omega[0]
    compfront = np.zeros((stack.shape[1], stack.shape[2]), dtype="complex64")
    compcar = np.zeros((stack.shape[1], stack.shape[2]), dtype="complex64")
    sin_value_sum = np.zeros((stack.shape[1], stack.shape[2]), dtype="complex64")
    cos_value_sum = np.zeros((stack.shape[1], stack.shape[2]), dtype="complex64")
    coscar = np.zeros((stack.shape[1], stack.shape[2]), dtype="complex64")
    sincar = np.zeros((stack.shape[1], stack.shape[2]), dtype="complex64")

    number_of_images = stack.shape[0]
    cos_value_sum = np.array(0)
    sin_value_sum = np.array(0)
    n = np.arange(number_of_images)
    initial_phase_change = 2 * np.pi * (n / number_of_images)

    for i in range(number_of_images):
        cos_value = stack[i] * np.cos(initial_phase_change[i])
        sin_value = stack[i] * np.sin(initial_phase_change[i])
        cos_value_sum = cos_value_sum + cos_value
        sin_value_sum = sin_value_sum + sin_value
    compfront = cos_value_sum + sin_value_sum * 1j

    x = np.linspace(0, xspace, stack.shape[2], endpoint=False)
    y = np.linspace(0, yspace, stack.shape[1], endpoint=False)
    irow, icol = np.meshgrid(x, y, indexing="xy")

    coscar = np.cos(2 * np.pi * (icol + irow)).astype("float32")
    sincar = np.sin(2 * np.pi * (icol + irow)).astype("float32")
    compcar = coscar + sincar * 1j

    compfinal = compfront / compcar
    return np.arctan2(np.imag(compfinal), np.real(compfinal))


def display_fft_image(
    image: np.ndarray,
    sb_position: tuple,
    slice_fft: np.ndarray,
    mask: np.ndarray = 1,
    detail: bool = True,
) -> None:
    """Display an fft image.

    This function helps to show the steps of the reconstruction and to define
    the best length and width of line mask

    Parameters
    ----------
    image : array
        the input image
    sb_position : tuple
        the sideband position (y, x), referred to the non-shifted FFT.
    slice fft : array
        contain minimum and maximum value in y and x axis.
    mask : array
        contain the aperture and line mask
    detail : bolean

    """
    fft_original_image = np.fft.fft2(image) / np.prod(image.shape)
    fft_original_image1 = np.roll(fft_original_image, sb_position, axis=(0, 1))
    fft_original_image2 = np.fft.fftshift(fft_original_image1)
    fft_original_image3 = fft_original_image2[slice_fft]
    fft_original_image4 = np.fft.fftshift(fft_original_image3)
    fft_with_aperture = fft_original_image4 * mask
    if detail:
        plt.figure()
        plt.imshow(np.abs(fft_original_image1), norm=LogNorm(), cmap="gray")
        plt.figure()
        plt.imshow(np.abs(fft_original_image2), norm=LogNorm(), cmap="gray")
        plt.figure()
        plt.imshow(np.abs(fft_original_image3), norm=LogNorm(), cmap="gray")
        plt.figure()
        plt.imshow(np.abs(fft_original_image4), norm=LogNorm(), cmap="gray")
        figure, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.abs(fft_original_image4), norm=LogNorm(), cmap="gray")
        ax1.set_title("Without Aperture Mask")
        ax2.imshow(np.log1p(np.abs(fft_with_aperture)), cmap="gray")
        ax2.set_title("FFT with Aperture Mask")
    else:
        figure, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.abs(fft_original_image4), norm=LogNorm(), cmap="gray")
        ax1.set_title("Without Aperture Mask")
        ax2.imshow(np.abs(fft_with_aperture), vmax=0.01, cmap="gray")
        ax2.set_title("FFT with Aperture Mask")


def get_phase(
    hologram: np.ndarray,
    params: HoloParams,
    xp: XPType = np,
) -> np.ndarray:
    """Reconstruct hologram using HoloParams and extract and unwrap phase."""
    t0 = time.perf_counter()

    slice_fft = get_slice_fft(params.out_shape, hologram.shape)
    phase_amp = reconstruct_frame(
        hologram,
        sb_pos=params.sb_position,
        aperture=params.aperture,
        slice_fft=slice_fft,
        xp=xp
    )

    # phase_unwrap is numpy-only:
    phase = for_backend(np.angle(phase_amp), NUMPY)

    t1 = time.perf_counter()

    phase_unwrapped = phase_unwrap(phase)

    t2 = time.perf_counter()

    log.debug(
        f"time: total={t2-t0:.3f}s, "
        f"reconstruction={t1-t0:.3f}s, unwrapping={t2-t1:.3f}s"
    )
    return phase_unwrapped


def reconstruct_bf(
    frame: np.ndarray,
    aperture: np.ndarray,
    slice_fft: tuple[slice, slice],
    *,
    xp=np,
) -> np.ndarray:
    """Reconstruct a brightfield image from a hologram.

    Please use :func:`libertem_holo.base.filter.central_line_filter` to
    filter out fresnel fringes as appropriate.
    """
    frame = xp.array(frame)
    fft_frame = xp.fft.fft2(frame)
    fft_frame = xp.fft.fftshift(xp.fft.fftshift(fft_frame)[slice_fft])

    fft_frame = fft_frame * xp.array(aperture)

    return xp.fft.ifft2(fft_frame)


def phase_offset_correction(
    aligned_stack,
    wtype: Literal['weighted'] | Literal['unweighted'] = 'weighted',
    threshold: float = 1e-12,
    return_stack: bool = False,
    xp=np,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    This part of the code is to correct for the phase drift in the holograms due
    to the biprism drift with time.  Since we are dealing with the phase of the
    image which wraps from -pi to pi a simple scalar correction of the phase
    doesnt work as the phase wraps around the phase axis. So the options for a
    solution would be either iterative solution which is computationally heavy
    or an eigenvalue solution which is implemented here.

    We start with a stack of holograms acquired using Holoworks or stack
    acquisition and this function returns a phase corrected complex image
    and optionally a stack of the phase corrected complex images before
    averaging.

    Adapted from a function by oleh.melnyk:
    https://github.com/Ptychography-4-0/ptychography/blob/master/src/ptychography40/stitching/stitching.py
    for more details, see https://arxiv.org/pdf/2005.02032.pdf

    Parameters
    ----------
    aligned_stack
        Array of shape (N, sy, sx) where N is the number of complex images;
        should have dtype complex64 or complex128.

    threshold
        Minimum absolute value to be considered in finding the phase match for
        stitching.

    wtype
        Selected type of weights to be used in the angular synchronization
        There are 2 possible choices:
        'unweighted' or 'weighted', depending on usage or not of the weights for
        angular synchronization (explained above)

    return_stack
        Also returns a stack shaped like the input stack, where the phase
        offset correction is applied.

    xp
        Either numpy or cupy for GPU support
    """
    aligned_stack = xp.asarray(aligned_stack)
    R = aligned_stack.shape[0]
    orig_R = R

    # Otherwise eigsh is unhappy
    if R <= 2:
        # Not sure how broadcasting works here, but try to be equivalent
        new_shape = (3,) + aligned_stack.shape[1:]
        new_aligned_stack = np.zeros_like(aligned_stack, shape=new_shape)
        new_aligned_stack[:R, :, :] = aligned_stack
        R = 3
        aligned_stack = new_aligned_stack

    ph_diff = xp.zeros((R, R), dtype=complex)
    d1 = aligned_stack.shape[1]
    d2 = aligned_stack.shape[2]

    result_stack = xp.zeros_like(aligned_stack)

    for r1 in range(R):
        for r2 in range(r1+1, R):
            ph_diff[r1, r2] = np.einsum('ij,ij', aligned_stack[r1], aligned_stack[r2].conj())
            ph_diff[r2, r1] = ph_diff[r1, r2].conj()

    if wtype == 'weighted':
        weights = np.abs(ph_diff)
    elif wtype == 'unweighted':
        weights = (np.abs(ph_diff) > threshold).astype(float)

    idx = weights > 0
    ph_diff[idx] = ph_diff[idx]/np.abs(ph_diff[idx])
    degree = np.sum(weights, axis=1)
    laplacian = np.diag(degree) - ph_diff * weights
    # because cupyx.scipy.sparse.linalg.eigsh doesn't support which='SM',
    # we transfer to CPU here (see also https://github.com/cupy/cupy/issues/4692):
    _, phases = eigsh(for_backend(laplacian, NUMPY), 1, which='SM')
    phases = xp.asarray(phases)
    idx = np.abs(phases) > threshold
    phases[idx] = phases[idx]/np.abs(phases[idx])
    phases[~idx] = 1
    phases *= phases[0].conj()
    result = xp.zeros((d1, d2), dtype=complex)
    count = xp.zeros((d1, d2))
    for r in range(R):
        result[:, :] += phases[r].conj() * aligned_stack[r]
        count[:, :] += np.abs(aligned_stack[r]) > threshold
        if return_stack:
            result_stack[r] = phases[r].conj() * aligned_stack[r]

    mask = count != 0
    result[mask] = result[mask] / count[mask]

    if return_stack:
        return result, phases, result_stack[:orig_R]
    else:
        return result, phases, None

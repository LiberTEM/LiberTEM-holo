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

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.fft import fft2

from libertem_holo.base.mask import disk_aperture

XPType = Any  # Union[Module("numpy"), Module("cupy")]


def freq_array(
    shape: tuple[int, int],
    sampling: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Generate a frequency array.

    Parameters
    ----------
    shape : (int, int)
        The shape of the array.
    sampling: (float, float), optional, (Default: (1., 1.))
        The sampling rates of the array.

    Returns
    -------
        Array of the frequencies.

    """
    f_freq_1d_y = np.fft.fftfreq(shape[0], sampling[0])
    f_freq_1d_x = np.fft.fftfreq(shape[1], sampling[1])
    f_freq_mesh = np.meshgrid(f_freq_1d_x, f_freq_1d_y)
    return np.hypot(f_freq_mesh[0], f_freq_mesh[1])


def get_slice_fft(
    out_shape: tuple[int, int],
    sig_shape: tuple[int, int],
) -> tuple[slice, slice]:
    """Get a slice in fourier space to achieve the given output shape."""
    sy, sx = sig_shape
    oy, ox = out_shape

    y_min = int(sy / 2 - oy / 2)
    y_max = int(sy / 2 + oy / 2)
    x_min = int(sx / 2 - ox / 2)
    x_max = int(sx / 2 + ox / 2)
    return (slice(y_min, y_max), slice(x_min, x_max))


def estimate_sideband_position(
    holo_data: np.ndarray,
    holo_sampling: tuple[float, float],
    central_band_mask_radius: float | None = None,
    sb: Literal["lower", "upper"] = "lower",
    xp: XPType = np,
) -> tuple[float, float]:
    """Find the position of the sideband and return its position.

    Parameters
    ----------
    holo_data: ndarray
        The data of the hologram.
    holo_sampling: tuple
        The sampling rate in both image directions.
    central_band_mask_radius: float, optional
        The aperture radius used to mask out the centerband.
    sb : str, optional
        Chooses which sideband is taken. 'lower' or 'upper'
    xp
        Pass in either the numpy or cupy module to select CPU or GPU processing

    Returns
    -------
    Tuple of the sideband position (y, x), referred to the unshifted FFT.

    """
    sb_position = (0, 0)

    f_freq = freq_array(holo_data.shape, holo_sampling)

    # If aperture radius of centerband is not given, it will be set to 5 % of
    # the Nyquist frequency.
    if central_band_mask_radius is None:
        central_band_mask_radius = 1 / 20.0 * np.max(f_freq)

    aperture = disk_aperture(
        holo_data.shape,
        central_band_mask_radius,
        xp=xp,
    )

    # A small aperture masking out the centerband.
    aperture_central_band = np.subtract(1.0, aperture)
    # imitates 0

    fft_holo = fft2(holo_data) / np.prod(holo_data.shape)
    fft_filtered = fft_holo * aperture_central_band

    # Sideband position in pixels referred to unshifted FFT
    if sb == "lower":
        fft_sb = fft_filtered[: int(fft_filtered.shape[0] / 2), :]
        sb_position = xp.asarray(
            np.unravel_index(np.abs(fft_sb).argmax(), fft_sb.shape),
        )
    elif sb == "upper":
        fft_sb = fft_filtered[int(fft_filtered.shape[0] / 2):, :]
        sb_position = np.unravel_index(np.abs(fft_sb).argmax(), fft_sb.shape)
        sb_position = xp.asarray(
            xp.add(
                xp.asarray(sb_position),
                xp.asarray([int(fft_filtered.shape[0] / 2), 0]),
            ),
        )

    return tuple(sb_position)


def estimate_sideband_size(
    sb_position: tuple[float, float],
    holo_shape: tuple[int, int],
    sb_size_ratio: float = 0.5,
    xp: XPType = np,
) -> float:
    """Estimate the size of sideband filter.

    Parameters
    ----------
    holo_shape : array_like
            Holographic data array
    sb_position : tuple
        The sideband position (y, x), referred to the non-shifted FFT.
    sb_size_ratio : float, optional
        Size of sideband as a fraction of the distance to central band
    xp
        Pass in either the numpy or cupy module to select CPU or GPU processing

    Returns
    -------
    sb_size : float
        Size of sideband filter

    """
    h = (
        xp.array(
            (
                xp.asarray(sb_position) - xp.asarray([0, 0]),
                xp.asarray(sb_position) - xp.asarray([0, holo_shape[1]]),
                xp.asarray(sb_position) - xp.asarray([holo_shape[0], 0]),
                xp.asarray(sb_position) - xp.asarray(holo_shape),
            ),
        )
        * sb_size_ratio
    )
    return xp.min(xp.linalg.norm(h, axis=1))


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
    fft_frame = xp.roll(fft_frame, xp.array(sb_pos), axis=(0, 1))

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


def display_fft_image(image, sb_position, slice_fft, mask=1, detail=True):
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

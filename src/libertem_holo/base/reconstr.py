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

import numpy as np
from numpy.fft import fft2
from libertem_holo.base.mask import disk_aperture
from libertem_holo.base.filters import ramp_compensation
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.restoration import unwrap_phase


def freq_array(shape, sampling=(1., 1.)):
    """
    Makes up a frequency array.

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
    f_freq = np.hypot(f_freq_mesh[0], f_freq_mesh[1])

    return f_freq


def get_slice_fft(out_shape, sig_shape):
    sy, sx = sig_shape
    oy, ox = out_shape

    y_min = int(sy / 2 - oy / 2)
    y_max = int(sy / 2 + oy / 2)
    x_min = int(sx / 2 - ox / 2)
    x_max = int(sx / 2 + ox / 2)
    slice_fft = (slice(y_min, y_max), slice(x_min, x_max))

    return slice_fft


def estimate_sideband_position(holo_data, holo_sampling, central_band_mask_radius=None, sb='lower'):
    """
    Finds the position of the sideband and returns its position.

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

    Returns
    -------
    Tuple of the sideband position (y, x), referred to the unshifted FFT.
    """

    sb_position = (0, 0)

    f_freq = freq_array(holo_data.shape, holo_sampling)

    # If aperture radius of centerband is not given, it will be set to 5 % of the Nyquist
    # frequency.
    if central_band_mask_radius is None:
        central_band_mask_radius = 1 / 20. * np.max(f_freq)

    aperture = disk_aperture(holo_data.shape, central_band_mask_radius)

    # A small aperture masking out the centerband.
    aperture_central_band = np.subtract(1.0, aperture)
    # imitates 0

    fft_holo = fft2(holo_data) / np.prod(holo_data.shape)
    fft_filtered = fft_holo * aperture_central_band

    # Sideband position in pixels referred to unshifted FFT
    if sb == 'lower':
        fft_sb = fft_filtered[:int(fft_filtered.shape[0] / 2), :]
        sb_position = np.asarray(np.unravel_index(fft_sb.argmax(), fft_sb.shape))
    elif sb == 'upper':
        fft_sb = fft_filtered[int(fft_filtered.shape[0] / 2):, :]
        sb_position = (np.unravel_index(fft_sb.argmax(), fft_sb.shape))
        sb_position = np.asarray(np.add(sb_position, (int(fft_filtered.shape[0] / 2), 0)))

    return sb_position


def estimate_sideband_size(sb_position, holo_shape, sb_size_ratio=0.5):
    """
    Estimates the size of sideband filter

    Parameters
    ----------
    holo_shape : array_like
            Holographic data array
    sb_position : tuple
        The sideband position (y, x), referred to the non-shifted FFT.
    sb_size_ratio : float, optional
        Size of sideband as a fraction of the distance to central band

    Returns
    -------
    sb_size : float
        Size of sideband filter

    """

    h = np.array((np.asarray(sb_position) - np.asarray([0, 0]),
                  np.asarray(sb_position) - np.asarray([0, holo_shape[1]]),
                  np.asarray(sb_position) - np.asarray([holo_shape[0], 0]),
                  np.asarray(sb_position) - np.asarray(holo_shape))) * sb_size_ratio
    return np.min(np.linalg.norm(h, axis=1))


def reconstruct_frame(frame, sb_pos, aperture, slice_fft, precision=True):
    if not precision:
        frame = frame.astype(np.float32)
    frame_size = frame.shape

    fft_frame = np.fft.fft2(frame) / np.prod(frame_size)
    fft_frame = np.roll(fft_frame, sb_pos, axis=(0, 1))

    fft_frame = np.fft.fftshift(np.fft.fftshift(fft_frame)[slice_fft])

    fft_frame = fft_frame * aperture

    wav = np.fft.ifft2(fft_frame) * np.prod(frame_size)
    return wav


def reconstruct_double_resolution(frame, sb_pos, aperture, slice_fft, precision=True):
    """
    Reconstruct a stack of phase shifted holography with double resolution method.

    Parameters
    ----------
    Frame : array_like
        Holographic data array
    sb_pos : tuple
        The sideband position (y, x), referred to the non-shifted FFT.
    aperture : 2D Array
        array containing the mask for reconstruction

    Returns
    -------
    wav : complex array
        the reconstructed complex image

    """
    image_double_resolution = frame[1]-frame[0]
    wav = reconstruct_frame(image_double_resolution,
                            sb_pos, aperture, slice_fft)
    return wav


def reconstruct_direct(frame, number_of_images, omega):
    """
    Reconstruct a stack of phase shifted holography with direct
    reconstruction method

    Parameters
    ----------
    Frame : array_like
        Holographic data array
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
    compfront = np.zeros((frame.shape[1], frame.shape[2]), dtype="complex64")
    compcar = np.zeros((frame.shape[1], frame.shape[2]), dtype="complex64")
    sin_value_sum = np.zeros((frame.shape[1], frame.shape[2]), dtype="complex64")
    cos_value_sum = np.zeros((frame.shape[1], frame.shape[2]), dtype="complex64")
    coscar = np.zeros((frame.shape[1], frame.shape[2]), dtype="complex64")
    sincar = np.zeros((frame.shape[1], frame.shape[2]), dtype="complex64")

    cos_value_sum = np.array(0)
    sin_value_sum = np.array(0)
    n = np.arange(number_of_images)
    initial_phase_change = 2 * np.pi * (n/number_of_images)

    for i in range(number_of_images):
        cos_value = frame[i] * np.cos(initial_phase_change[i])
        sin_value = frame[i] * np.sin(initial_phase_change[i])
        cos_value_sum = cos_value_sum + cos_value
        sin_value_sum = sin_value_sum + sin_value
    compfront = cos_value_sum + sin_value_sum * 1j

    x = np.linspace(0, yspace, frame.shape[1], endpoint=False)
    y = np.linspace(0, xspace, frame.shape[2], endpoint=False)
    irow, icol = np.meshgrid(x, y, indexing='ij')

    coscar = np.cos(2 * np.pi * (icol+irow)).astype('float32')
    sincar = np.sin(2 * np.pi * (icol+irow)).astype('float32')
    compcar = coscar + sincar * 1j

    compfinal = compfront/compcar
    phase_final = np.arctan2(np.imag(compfinal), np.real(compfinal))

    return phase_final


def display_fft_image(image, sb_position, slice_fft, mask=1, detail=True):
    """
    Display the fft image
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
        ax1.set_title('Without Aperture Mask')
        ax2.imshow(np.log1p(np.abs(fft_with_aperture)), cmap="gray")
        ax2.set_title('FFT with Aperture Mask')
    else:
        figure, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.abs(fft_original_image4), norm=LogNorm(), cmap="gray")
        ax1.set_title('Without Aperture Mask')
        ax2.imshow(np.abs(fft_with_aperture), vmax=0.01, cmap="gray")
        ax2.set_title('FFT with Aperture Mask')


def phase_shifting_reconstruction(image, number_of_image, centerband_position, sb_position):
    """
    This function allow a reconstruction with phase shifting methods.

    Parameters
    ----------
    image : array
        the input image
    number_of_image : int
        the number of images in the stack
    centerband_position : tuple
        the centerband position (y, x), referred to the shifted FFT.
    sb_position : tuple
        the sideband position (y, x), referred to the shifted FFT.
    """
    image_FT_stack = np.fft.fftshift((np.fft.fft2(image)))
    phase_initial = np.zeros(number_of_image)
    centerband_value = np.zeros(number_of_image, dtype=np.complex128)
    sideband_value = np.zeros(number_of_image, dtype=np.complex128)
    for i in range(len(image_FT_stack)):
        sideband_value[i] = image_FT_stack[i, sb_position[0], sb_position[1]]
        centerband_value[i] = image_FT_stack[i, centerband_position[0], centerband_position[1]]
        phase_initial[i] = np.angle(sideband_value[i])-np.angle(centerband_value[i])
    c2 = 0
    for i in range(number_of_image):
        c2 = c2 + (image[i] * np.exp(-1j * phase_initial[i]))
    c2 = c2 / number_of_image
    phase_distribution_noisy = ramp_compensation(unwrap_phase(np.angle(c2)))
    return phase_distribution_noisy

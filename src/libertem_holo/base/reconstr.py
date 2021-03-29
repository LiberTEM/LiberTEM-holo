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


def aperture_function(r, apradius, rsmooth):
    """
    A smooth aperture function that decays from apradius-rsmooth to apradius+rsmooth.
    Parameters
    ----------
    r : ndarray
        Array of input data (e.g. frequencies)
    apradius : float
        Radius (center) of the smooth aperture. Decay starts at apradius - rsmooth.
    rsmooth : float
        Smoothness in halfwidth. rsmooth = 1 will cause a decay from 1 to 0 over 2 pixel.
    Returns
    -------
        2d array containing aperture
    """

    return 0.5 * (1. - np.tanh((np.absolute(r) - apradius) / (0.5 * rsmooth)))

def aperture_function_gauss(r,radius,sigma) :
    """
    An aperture function with gaussian filter and skimage draw disk
    ----------
    r : input array
        Array of input data (e.g. frequency or input image)
    radius : float
        Radius of the disk or circle 
    sigma : float
        Sigma value of gaussian filter
    Returns
    -------
        2d array containing aperture
    """
    size = r.shape 
    aperture = np.zeros((size))
    rr,cc = disk((0,0),radius)
    aperture[rr,cc] = 1
    aperture = gaussian_filter(aperture,sigma=sigma)
    return aperture

def get_aperture(out_shape, sb_size, sb_smoothness, sig_shape):
    sy, sx = sig_shape
    oy, ox = out_shape
    f_sampling = (1. / oy, 1. / ox)
    sb_size = sb_size * np.mean(f_sampling)
    sb_smoothness = sb_size * sb_smoothness * np.mean(f_sampling)

    f_freq = freq_array(out_shape)
    aperture = aperture_function(f_freq, sb_size, sb_smoothness)

    y_min = int(sy / 2 - oy / 2)
    y_max = int(sy / 2 + oy / 2)
    x_min = int(sx / 2 - ox / 2)
    x_max = int(sx / 2 + oy / 2)
    slice_fft = (slice(y_min, y_max), slice(x_min, x_max))

    return slice_fft, aperture

def get_aperture_gauss(out_shape, sb_size, sb_smoothness, sig_shape):
    sy, sx = sig_shape
    oy, ox = out_shape
    f_sampling = (1. / oy, 1. / ox)
    f_freq = freq_array(out_shape)
    aperture = aperture_function_gauss(f_freq, sb_size, sb_smoothness)

    y_min = int(sy / 2 - oy / 2)
    y_max = int(sy / 2 + oy / 2)
    x_min = int(sx / 2 - ox / 2)
    x_max = int(sx / 2 + oy / 2)
    slice_fft = (slice(y_min, y_max), slice(x_min, x_max))

    return slice_fft, aperture

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

    aperture = aperture_function(f_freq, central_band_mask_radius, 1e-6)
    # A small aperture masking out the centerband.
    aperture_central_band = np.subtract(1.0, aperture)  # 1e-6
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

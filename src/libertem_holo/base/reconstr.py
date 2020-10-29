# Functions freq_array, aperture_function, are adopted from Hyperspy
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

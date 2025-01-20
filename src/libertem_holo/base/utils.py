
"""Utility functions for working with holography data."""

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
import typing
from numpy.fft import fft2
from libertem_holo.base.mask import disk_aperture
from sparseconverter import NUMPY, for_backend
import numpy as np
import logging


log = logging.getLogger(__name__)

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




class HoloParams(typing.NamedTuple):
    """Here."""

    sb_size: tuple[float, float]
    sb_position: tuple[float, float]
    aperture: np.ndarray  # actually can be a cupy ndarray, too! hm...
    orig_shape: tuple[int, int]
    out_shape: tuple[int, int]
    scale_factor: float  # by how much is the phase image scaled down

    @property
    def sb_position_raw(self) -> tuple[float, float]:
        """The actual sb_position contains arrays, convert them to floats."""
        return tuple(
            float(for_backend(c, NUMPY))
            for c in self.sb_position
        )

    @property
    def sb_position_int(self) -> tuple[int, int]:
        """Sideband position from float to int."""
        return tuple(
            int(c)
            for c in self.sb_position_raw
        )

    @classmethod
    def from_hologram(
        cls,
        hologram: np.ndarray,
        central_band_mask_radius: int,
        out_shape: tuple = None,
        xp: XPType = np,
    ) -> HoloParams:
        """Return reconstruction parameters."""
        hologram = xp.asarray(hologram)

        sb_position = estimate_sideband_position(
            holo_data=hologram,
            holo_sampling=(1, 1),
            sb='upper',
            central_band_mask_radius=central_band_mask_radius,
            xp=xp,
        )
        sb_size = estimate_sideband_size(sb_position, hologram.shape, xp=xp)

        if out_shape is None:
            out_side = 2 * int(sb_size) + 16
            out_shape = (out_side, out_side)

        #  Disk aperture, add line filter ?
        aperture = disk_aperture(out_shape, sb_size, xp=xp)
        aperture = xp.asarray(aperture)

        return cls(
            sb_size=sb_size,
            sb_position=sb_position,
            aperture=aperture,
            out_shape=out_shape,
            orig_shape=hologram.shape,
            scale_factor=out_shape[0] / hologram.shape[0],
        )

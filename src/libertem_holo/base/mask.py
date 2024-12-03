"""Functions for building apertures."""
from __future__ import annotations

import numpy as np
from libertem.masks import radial_bins
from skimage.draw import line


def disk_aperture(out_shape: tuple[int, int], radius: float, xp=np) -> np.ndarray:
    """Generate a disk-shaped aperture, fft-shifted.

    Parameters
    ----------
    out_shape : interger
        Shape of the output array
    radius : float
        Radius of the disk
    xp
        Either numpy or cupy

    Returns
    -------
    aperture
        2d array containing aperture

    """
    center = int(out_shape[0] / 2), int(out_shape[1] / 2)

    bins = xp.asarray(radial_bins(
        centerX=center[1],
        centerY=center[0],
        imageSizeX=out_shape[1],
        imageSizeY=out_shape[0],
        radius=float(radius),
        n_bins=1,
        use_sparse=False,
    ))

    return xp.fft.fftshift(bins[0])


def line_filter(
    orig_shape: tuple[int, int],
    sb_pos: tuple[int, int],
    width: int,
    length: int,
    slice_fft: slice,
) -> np.ndarray:
    """Remove Fresnel fringes from biprism with a line filter in fourier space.

    The starting point is the sideband position. The end points depend on the
    length and in the direction to top right image.

    Parameters
    ----------
    orig_shape
        the shape of the image.
    sb_pos
        Position of the sideband that is used for reconstruction of holograms.
    width
        Width of the line (rectangle) in pixels.
    length
        Length of the line (rectangle) in pixels.
    slice_fft
        A slice in fft shifted coordinates

    Returns
    -------
        2d array containing line filter

    """
    start_pos = (sb_pos[0], sb_pos[1])
    angle = np.arctan2(sb_pos[0], orig_shape[1] - sb_pos[1])
    end_pos = (sb_pos[0] - int(np.floor(length * np.sin(angle))),
               sb_pos[1] + int(np.floor(length * np.cos(angle))))

    # FIXME: replace with `skimage.draw.polygon`?
    rr, cc = line(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
    mask = np.ones(orig_shape)
    mask[rr, cc] = 0

    for i in range(int(np.ceil(width/2))):
        rr, cc = line(start_pos[0], start_pos[1] + i, end_pos[0] + i, end_pos[1])
        mask[rr, cc] = 0
        rr, cc = line(start_pos[0], start_pos[1] - i, end_pos[0], end_pos[1] - i)
        mask[rr, cc] = 0

    return np.fft.fftshift(np.fft.fftshift(mask)[slice_fft])

import numpy as np
from skimage.draw import line
from libertem.masks import radial_bins


def disk_aperture(out_shape, radius, xp=np):
    """
    A disk-shaped aperture, fft-shifted.

    Parameters
    ----------
    out_shape : interger
        Shape of the output array
    radius : float
        Radius of the disk

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
        use_sparse=False
    ))

    return xp.fft.fftshift(bins[0])


def line_filter(
    shape: tuple[int, int],
    sidebandpos: tuple[int, int],
    width: int,
    length: int,
    slice_fft,
):
    """
    A line filter function that is used to remove Fresnel fringes from biprism.
    Parameters. The line will be created with skimage.draw.line. The starting points are
    the sideband position. The end points depend on the length and in the direction to top
    right image.

    Parameters
    ----------
    shape : 2D tuple, ()
        the shape of the image.
    sidebandpos : 2d tuple, ()
        Position of the sideband that is used for reconstruction of holograms.
    width: int
        Width of the line (rectangle) in pixels.
    length : int
        Length of the line (rectangle) in pixels.
    slice_fft : array
        contain minimum and maximum value in y and x axis.

    Returns
    -------
        2d array containing line filter
    """
    start_pos = (sidebandpos[0], sidebandpos[1])
    angle = np.arctan2(sidebandpos[0], shape[1] - sidebandpos[1])
    end_pos = (sidebandpos[0] - int(np.floor(length * np.sin(angle))),
               sidebandpos[1] + int(np.floor(length * np.cos(angle))))

    rr, cc = line(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
    mask = np.ones(shape)
    mask[rr, cc] = 0

    for i in range(0, int(np.ceil(width/2))):
        rr, cc = line(start_pos[0], start_pos[1] + i, end_pos[0] + i, end_pos[1])
        mask[rr, cc] = 0
        rr, cc = line(start_pos[0], start_pos[1] - i, end_pos[0], end_pos[1] - i)
        mask[rr, cc] = 0

    mask = np.fft.fftshift(np.fft.fftshift(mask)[slice_fft])

    return mask

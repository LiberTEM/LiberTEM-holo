import numpy as np

from libertem.udf import UDF

from skimage.draw import polygon


def line_filter(size, sidebandpos, width, length):
    """
    A line filter function that is used to remove Fresnel fringes from biprism.

    Parameters
    ----------
    size : 2d tuple, ()
        size of the FFT of the hologram.
    sidebandpos : 2d tuple, ()
        Position of the sideband that is used for reconstruction of holograms.
    width: float
        Width of the line (rectangle).
    length : float
        Length of the line (rectangle).
    Returns
    -------
        2d array containing line filter
    """

    angle = np.arctan2(size[0] / 2 + 1 - sidebandpos[0],  size[1] / 2 + 1 - sidebandpos[1])
    mid_pos = ((size[0] / 2 + 1 + sidebandpos[0]) / 2,
               (size[1] / 2 + 1 + sidebandpos[1]) / 2)
    left_bottom = (mid_pos[0] - length / 2 * np.cos(angle) + width / 2 * np.sin(angle),
                   (mid_pos[1] - length / 2 * np.sin(angle) - width / 2 * np.cos(angle)))
    right_bottom = (left_bottom[0] + np.cos(angle) * length,
                    left_bottom[1] + np.sin(angle) * length)
    left_top = (left_bottom[0] - np.sin(angle) * width,
                left_bottom[1] + np.cos(angle) * width)
    right_top = (right_bottom[0] + left_top[0] - left_bottom[0],
                right_bottom[1] + left_top[1] - left_bottom[1])
    r = np.array([left_bottom[0], right_bottom[0], right_top[0], left_top[0]], dtype=int)
    c = np.array([left_bottom[1], right_bottom[1], right_top[1], left_top[1]], dtype=int)
    rr, cc = polygon(r, c)
    mask = np.ones(size)
    mask[rr, cc] = 0

    return mask
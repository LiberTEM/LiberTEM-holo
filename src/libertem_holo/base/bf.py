import typing
from functools import lru_cache

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from sparseconverter import NUMPY, for_backend
from skimage.draw import polygon

from libertem_holo.base.reconstr import get_slice_fft


@lru_cache
def shifted_coords_for_shape(shape):
    return np.fft.fftshift(np.moveaxis(np.mgrid[0:shape[0], 0:shape[1]], 0, -1), axes=(0, 1))

def fft_shift_coords(pos, shape):
    coords = shifted_coords_for_shape(shape)
    return tuple(int(n) for n in coords[pos[0], pos[1]])

def other_sb(sb_position, shape):
    """
    Given the sb_position (as from the estimate function in fft coordinates),
    and the shape of the hologram, calculate the position of the other sideband
    position (also in fft coordinates)
    """
    sb_pos_shifted = fft_shift_coords(sb_position, shape)
    center = (shape[0]//2, shape[1]//2)
    center_to_sb = (
        sb_pos_shifted[0]-center[0],
        sb_pos_shifted[1]-center[1],
    )
    other_sb = (
        center[0] - center_to_sb[0],
        center[1] - center_to_sb[1],
    )
    return fft_shift_coords(other_sb, shape)

def _line_filter_coords(length_ratio, sb_position_shifted, width, orig_shape):
    # let's start from a "unit rectangle" which is 1x1 and not rotated:
    coords = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ]).astype(np.float32)

    # shift to the origin in y direction:
    coords -= np.array([0.5, 0])

    # let's determine the length from the sideband position and ratio:
    center = (orig_shape[0]//2, orig_shape[1]//2)
    center_to_sb = (
        sb_position_shifted[0]-center[0],
        sb_position_shifted[1]-center[1],
    )
    sb_dist = np.linalg.norm(np.array(center_to_sb))
    length = sb_dist * length_ratio
    length_rest = sb_dist * (1 - length_ratio)

    # stretch such that the width (in x direction) corresponds to the length,
    # and the height (y direction) corresponds to the width of the filter:
    scale = np.array([
        [width, 0],
        [0, length],
    ])

    # angle from -pi to +pi between the "x-axis" and the vector from center to sb:
    angle = np.arctan2(*center_to_sb)

    rotate = np.array([
        [np.cos(-angle), np.sin(-angle)],
        [-np.sin(-angle), np.cos(-angle)],
    ])

    # print(center_to_sb, angle/np.pi*180)

    # apply scale:
    coords = coords @ scale

    # move to the right:
    coords += np.array([0, length_rest])

    # rotate:
    coords = coords @ rotate

    coords += np.array(center)

    # return angle, sb_dist, length, coords
    return coords


def _draw_lf_rect(dest, orig_shape, sb_position_shifted, length_ratio, width):
    # we "draw" a rotated rectangle into `dest`, starting at `sb_position_shifted`
    # and ending at `length_ratio` times the vector in the direction to the center of `out_shape`.
    coords = _line_filter_coords(length_ratio=length_ratio, sb_position_shifted=sb_position_shifted, width=width, orig_shape=orig_shape)
    # print(coords)
    rr, cc = polygon(coords[:, 0], coords[:, 1], shape=dest.shape)
    dest[rr, cc] = True


def central_line_filter(sb_position, out_shape, orig_shape, length_ratio=0.7, width=20, crop_to_out_shape=False):
    """
    Return a line filter for the central band, that can be applied
    by multiplying it with the aperture.
    """
    # we are working in npn-fft-shifted space, meaning with the zero
    # frequency at the center of the image. we work in original shape,
    # and crop at the end.
    dest = np.zeros(orig_shape, dtype=bool)

    # approx. positions of both sidebands (inferred from symmetry):
    sb_pos_shifted = fft_shift_coords(sb_position, orig_shape)

    # vector from the center to the sb:
    center = (orig_shape[0]//2, orig_shape[1]//2)
    center_to_sb = (
        sb_pos_shifted[0]-center[0],
        sb_pos_shifted[1]-center[1],
    )
    other_sb_pos = fft_shift_coords(other_sb(sb_position, orig_shape), orig_shape)

    _draw_lf_rect(dest, orig_shape=orig_shape, sb_position_shifted=sb_pos_shifted, length_ratio=length_ratio, width=width)
    _draw_lf_rect(dest, orig_shape=orig_shape, sb_position_shifted=other_sb_pos, length_ratio=length_ratio, width=width)

    if crop_to_out_shape:
        slice_fft = get_slice_fft(out_shape=out_shape, sig_shape=orig_shape)
        return dest[slice_fft]
    else:
        return dest


def reconstruct_bf(
    frame: np.ndarray,
    aperture: np.ndarray,
    slice_fft: tuple[slice, slice],
    *,
    xp = np,
) -> np.ndarray:
    frame = xp.array(frame)
    frame_size = frame.shape
    fft_frame = xp.fft.fft2(frame) # / np.prod(frame_size)
    fft_frame = xp.fft.fftshift(xp.fft.fftshift(fft_frame)[slice_fft])

    fft_frame = fft_frame * xp.array(aperture)

    return xp.fft.ifft2(fft_frame) # * np.prod(frame_size)

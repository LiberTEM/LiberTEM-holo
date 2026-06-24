"""2D quality-guided unwrapping."""
import heapq

import numba
import numpy as np


@numba.njit
def wrap(val: np.ndarray | float):
    """Returns phase offset."""
    return np.angle(np.exp(val * 1j))


def derivative_variance(
    array: np.ndarray[tuple[int, int]],
    k: int = 3,
) -> np.ndarray[tuple[int, int]]:
    """Calculate variance of the derivative of `array`.

    Can be used as a quality map for
    :func:`libertem_holo.base.unwrap.quality_unwrap`.

    Parameters:
    -----------
    array
        Input phase with a float dtype
    k
        Window size
    """
    # Adapted from
    # https://github.com/theilen/PyMRR/tree/master/mrr/unwrapping/
    diff = np.stack((
        np.diff(array, axis=0, prepend=0),
        np.diff(array, axis=1, prepend=0),
    ), axis=0)
    diff = wrap(diff)
    diff = np.pad(
        diff,
        ((0, 0), (1, 1), (1, 1)),
        mode="edge",
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        diff,
        (k, k),
        axis=(1, 2),
    )
    return np.var(windows, axis=(-2, -1)).sum(axis=0)


@numba.njit
def get_neighbours_ud(idx: int, im_size: int, width: int):
    above = idx - width
    below = idx + width
    if above > 0:
        yield above
    if below < im_size:
        yield below


@numba.njit
def get_neighbours(idx: int, height: int, width: int, connectivity: int):
    im_size = height * width
    for neigh in get_neighbours_ud(idx, im_size, width):
        yield neigh
    left = idx - 1
    right = idx + 1
    col = idx % width
    if col > 0:
        yield left
        if connectivity > 4:
            for neigh in get_neighbours_ud(left, im_size, width):
                yield neigh
    if width - col > 1:
        yield right
        if connectivity > 4:
            for neigh in get_neighbours_ud(right, im_size, width):
                yield neigh


@numba.njit
def unwrap_heap(heap, flat_phase, flat_q, flat_to_q, height, width, uw_phase, connectivity):
    other_heap = [(1., -1, -1)]
    _ = heapq.heappop(other_heap)

    # Unwrap in heap order
    while len(heap) > 0:
        _, idx, parent_idx = heapq.heappop(heap)
        phase_diff = flat_phase[idx] - flat_phase[parent_idx]
        uw_phase[idx] = uw_phase[parent_idx] + wrap(phase_diff)
        # Queue neighbours
        for n_idx in get_neighbours(idx, height, width, connectivity):
            if flat_to_q[n_idx] > 0:
                heapq.heappush(heap, (flat_q[n_idx], n_idx, idx))
                flat_to_q[n_idx] = 0
            elif flat_to_q[n_idx] < 0:
                heapq.heappush(other_heap, (flat_q[n_idx], n_idx, idx))
                flat_to_q[n_idx] = 0

    # This will unwrap any disjoint additional areas in the seed mask
    # but there is no likelihood that they unwrap with the same scale
    # as the first unwrapped area, so this is disabled
    # remaining_mask = flat_to_q > 0
    # if remaining_mask.any():
    #     (nonzero_remaining,) = np.nonzero(remaining_mask)
    #     pos = nonzero_remaining[np.argmin(flat_q[remaining_mask])]
    #     heapq.heappush(heap, (flat_q[pos], pos, pos))
    #     flat_to_q[pos] = 0
    #     unwrap_heap(
    #         heap, flat_phase, flat_q, flat_to_q, height, width, uw_phase, connectivity
    #     )

    # If we had any postponed pixels set them to_q == 1 and re-run
    if len(other_heap) > 0:
        flat_to_q = np.abs(flat_to_q)
        unwrap_heap(
            other_heap, flat_phase, flat_q, flat_to_q, height, width, uw_phase, connectivity
        )


def quality_unwrap(
    phase: np.ndarray[tuple[int, int]],
    quality: np.ndarray[tuple[int, int]],
):
    """Unwrap phase guided by a quality map.

    Parameters:
    -----------
    phase
        Input phase as a 2D ndarray
    quality
        Inverse score: lower values are higher quality
    """
    # quality is lowest => best
    assert -np.pi <= phase.min() <= np.pi
    assert -np.pi <= phase.max() <= np.pi
    assert phase.ndim == 2
    assert phase.shape == quality.shape
    img_shape = phase.shape

    # Flat views and results array
    flat_quality = quality.ravel()
    flat_phase = phase.ravel()
    flat_uw_phase = phase.copy().ravel()  # result array
    flat_to_q = np.ones(flat_phase.shape, dtype=np.int8)

    pos = np.argmin(flat_quality)
    heap = [(flat_quality[pos], pos, pos)]
    flat_to_q[pos] = 0  # first position already in q

    unwrap_heap(
        heap,
        flat_phase,
        flat_quality,
        flat_to_q,
        *img_shape,
        flat_uw_phase,
        connectivity=8,
    )

    return flat_uw_phase.reshape(img_shape)


__all__ = ["derivative_variance", "quality_unwrap"]

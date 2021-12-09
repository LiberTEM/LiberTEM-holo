'''
Direct phase-shifting hologram reconstruction

Based on work by Ru et al. https://doi.org/10.1016/0304-3991(94)90171-6 and
Patrick Adrian Gunawan.

This implements the full-form reconstruction for arbitrary phases. Only
condition is that the resulting characteristic matrix is invertible. The enginge
is structured in such a way that "divide and conquer" such as tiled processing
can be implemented.

Shifted images can be handled efficiently. This allows drift compensation and
processing data where the phase shifting also couples to image shift, i.e. it is
not only shifting the fringes.
'''

import numpy as np
import numba


def reconstruction_recipe(phases):
    '''
    Calculate parameters for the reconstruction engine.

    This inverts the characteristic matrix and extracts the portion that
    is relevant for the reconstruction.

    Parameters
    ----------

    phases : np.ndarray
        Phase angles of the holograms to be processed

    Returns
    -------
    rotation
        np.exp(1j*phases)
    recipe
        Last row of the inverted characteristic matrix, parameter for :func:`reconstruction_engine`.
    '''
    N = len(phases)
    rotation = np.exp(1j*phases)
    characteristic_matrix = np.array([
        (N, np.sum(rotation), np.sum(1/rotation)),
        (np.sum(1/rotation), N, np.sum(1/rotation**2)),
        (np.sum(rotation), np.sum(rotation**2), N)
    ])
    work_matrix = np.linalg.inv(characteristic_matrix)
    return rotation, work_matrix[2]


@numba.njit(fastmath=True)
def reconstruction_engine(holograms, y_offsets, x_offsets, rotation, recipe, out):
    '''
    Numerical work horse for the reconstruction.

    This function can be called chunk-wise to process tiles and aggregate the result in
    :code:`out`, provided the correct subset of per-frame parameters is provided
    and offsets are adjusted by the tile origin.

    Parameters
    ----------

    holograms : numpy.ndarray
        Hologram stack, resp. tile from hologram stack
    y_offsets, x_offsets : numpy.ndarray
        Offset of each item in :code:`holograms` relative to output.
    rotation : numpy.ndarray
        Direction of each hologram in the complex plane. Can be calculated with
        :func:`reconstruction_recipe`.
    recipe : numpy.ndarray
        Reconstruction recipe from :func:`reconstruction_recipe`, see there.
    out : numpy.ndarray
        Buffer where to aggregate.
    '''
    size_y, size_x = out.shape
    input_y, input_x = holograms.shape[1:]
    count = len(holograms)
    acc = np.zeros((3, size_x), dtype=np.complex128)
    # order (y, i, x) for optimization:
    # Sequential access on fast x axis with pre-calcuated range and offsets,
    # aggregation over i to increase leverage for vector product,
    # row by row to reduce accumulator size for cache efficiency
    for y in range(size_y):
        for i in range(count):
            rot = rotation[i]
            source_y = y + y_offsets[i]
            if source_y >= 0 and source_y < input_y:
                x_offset = x_offsets[i]
                start_x = max(0, -x_offset)
                stop_x = min(size_x, input_x + x_offset)
                for x in range(start_x, stop_x):
                    source_x = x + x_offset
                    source_data = holograms[i, source_y, source_x]
                    acc[0, x] += source_data
                    acc[1, x] += source_data / rot
                    acc[2, x] += source_data * rot
        out[y] += recipe @ acc
        acc[:] = 0


def direct_reconstruction(holograms, phases, reference_wave, y_offsets=None, x_offsets=None):
    '''
    Convenience function to perform reconstruction of a stack of holograms.

    Parameters
    ----------

    holograms : numpy.ndarray
        Hologram stack, resp. tile from hologram stack
    phases : numpy.ndarray
        Phase angle for each of the holograms.
    reference_wave : numpy.ndarray
        Reconstruction of a vacuum hologram stack, or a virtual reference.
        For simulated holograms, the complex conjugate of the reference wave
        can be used.
    y_offsets, x_offsets : numpy.ndarray
        Offset of each item in :code:`holograms` relative to output. No offset if None.

    Returns
    -------

    Reconstructed hologram
    '''
    size_y, size_x = holograms.shape[1:]
    if y_offsets is None:
        y_offsets = np.zeros(len(holograms), dtype=int)
    if x_offsets is None:
        x_offsets = np.zeros(len(holograms), dtype=int)
    N = len(phases)

    if len(holograms) != N:
        raise ValueError("Number of holograms and number of phases should be equal.")
    if len(y_offsets) != N:
        raise ValueError("Number of holograms and number of y offsets should be equal.")
    if len(x_offsets) != N:
        raise ValueError("Number of holograms and number of x offsets should be equal.")

    rotation, recipe = reconstruction_recipe(phases)
    out = np.zeros((size_y, size_x), dtype=np.complex128)
    reconstruction_engine(
        holograms,
        y_offsets=y_offsets,
        x_offsets=x_offsets,
        rotation=rotation,
        recipe=recipe,
        out=out
    )
    return out / reference_wave

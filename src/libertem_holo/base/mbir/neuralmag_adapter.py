"""Adapters between NeuralMag state tensors and MBIR-friendly arrays."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _require_isotropic_voxel(voxel_size: float | Sequence[float] | np.ndarray) -> None:
    sizes = np.asarray(voxel_size, dtype=float).reshape(-1)
    if sizes.size == 0:
        raise ValueError("voxel_size must not be empty")
    if sizes.size == 1:
        size = float(sizes[0])
    elif sizes.size == 3:
        if not np.allclose(sizes, sizes[0]):
            raise ValueError(
                "NeuralMag adapter currently requires isotropic voxels; "
                f"got voxel_size={tuple(float(v) for v in sizes)}."
            )
        size = float(sizes[0])
    else:
        raise ValueError(
            "voxel_size must be a scalar or a length-3 iterable, "
            f"got shape {tuple(sizes.shape)}."
        )
    if size <= 0:
        raise ValueError(f"voxel_size must be positive, got {size}.")


def _nodal_to_cell(values: np.ndarray) -> np.ndarray:
    if values.ndim != 4:
        raise ValueError(f"Expected 4D nodal state array, got shape {values.shape}.")
    nx, ny, nz, _ = values.shape
    if min(nx, ny, nz) < 2:
        raise ValueError(
            "Nodal state must have at least two points per spatial axis to define cells."
        )
    return (
        values[:-1, :-1, :-1, :]
        + values[1:, :-1, :-1, :]
        + values[:-1, 1:, :-1, :]
        + values[:-1, :-1, 1:, :]
        + values[1:, 1:, :-1, :]
        + values[1:, :-1, 1:, :]
        + values[:-1, 1:, 1:, :]
        + values[1:, 1:, 1:, :]
    ) / 8.0


def neuralmag_state_to_mbir_rho_m(
    state: np.ndarray,
    voxel_size: float | Sequence[float] | np.ndarray,
    *,
    state_is_nodal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert NeuralMag state ``(x, y, z, [rho, mx, my, mz])`` to MBIR arrays.

    Returns:
        ``rho``: cell-centered array in LiberTEM ordering ``(z, y, x)``.
        ``m``: cell-centered magnetization array in LiberTEM ordering
        ``(z, y, x, 3)`` with components ``(mx, my, mz)``.
    """
    _require_isotropic_voxel(voxel_size)

    state_arr = np.asarray(state)
    if state_arr.ndim != 4 or state_arr.shape[-1] != 4:
        raise ValueError(
            "Expected NeuralMag state shape (x, y, z, 4) "
            f"with [rho, mx, my, mz], got {state_arr.shape}."
        )

    cell_state = _nodal_to_cell(state_arr) if state_is_nodal else state_arr
    rho = np.transpose(cell_state[..., 0], (2, 1, 0))
    m = np.transpose(cell_state[..., 1:], (2, 1, 0, 3))
    return rho, m

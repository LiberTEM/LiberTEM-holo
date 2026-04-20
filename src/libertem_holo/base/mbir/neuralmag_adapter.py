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
    """Convert NeuralMag state ``(z, y, x, [rho, mx, my, mz])`` to MBIR arrays.

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
    rho = cell_state[..., 0]
    m = cell_state[..., 1:]
    return rho, m


def _mbir_cell_to_neuralmag_xyz(
    rho_3d: np.ndarray,
    m_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rho_arr = np.asarray(rho_3d)
    m_arr = np.asarray(m_3d)

    if rho_arr.ndim != 3:
        raise ValueError(f"Expected rho shape (z, y, x), got {rho_arr.shape}.")
    if m_arr.ndim != 4 or m_arr.shape[-1] != 3:
        raise ValueError(
            "Expected magnetization shape (z, y, x, 3), "
            f"got {m_arr.shape}."
        )
    if m_arr.shape[:3] != rho_arr.shape:
        raise ValueError(
            "rho and m must have matching spatial shapes, "
            f"got rho {rho_arr.shape} and m {m_arr.shape[:3]}."
        )

    return rho_arr, m_arr


def _normalize_vector_field(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normalized = values / safe_norms
    return np.where(support[..., np.newaxis] > 0.0, normalized, 0.0)


def mbir_rho_m_to_neuralmag_state(
    rho_3d: np.ndarray,
    m_3d: np.ndarray,
    mesh_or_state,
    *,
    state_is_nodal: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """Convert MBIR cell arrays to a nodal NeuralMag-style state tensor.

    Args:
        rho_3d: MBIR density/support array in LiberTEM ordering ``(z, y, x)``.
        m_3d: MBIR magnetization array in LiberTEM ordering ``(z, y, x, 3)``.
        mesh_or_state: NeuralMag ``Mesh`` or ``State`` used to construct the
            cell- and node-based function objects for projection.
        state_is_nodal: If true, project the cell-centered MBIR fields onto the
            NeuralMag nodal layout. If false, return the exact cell-centered
            NeuralMag state layout.
        normalize: If true, renormalize the nodal magnetization inside the
            projected support before returning.

    Returns:
        State tensor in NeuralMag ordering ``(z, y, x, [rho, mx, my, mz])``.
    """
    rho_xyz, m_xyz = _mbir_cell_to_neuralmag_xyz(rho_3d, m_3d)

    if not state_is_nodal:
        return np.concatenate((rho_xyz[..., np.newaxis], m_xyz), axis=-1)

    try:
        import neuralmag as nm
    except ImportError as exc:
        raise ImportError(
            "mbir_rho_m_to_neuralmag_state requires the neuralmag package."
        ) from exc

    state = mesh_or_state if hasattr(mesh_or_state, "tensor") else nm.State(mesh_or_state)

    rho_cell = nm.CellFunction(state, tensor=state.tensor(rho_xyz))
    m_cell = nm.VectorCellFunction(state, tensor=state.tensor(m_xyz))

    rho_node = np.asarray(rho_cell.to_node().tensor)
    m_node = np.asarray(m_cell.to_node().tensor)
    if normalize:
        m_node = _normalize_vector_field(m_node, support=rho_node)

    return np.concatenate((rho_node[..., np.newaxis], m_node), axis=-1)


def mbir_rho_m_to_neuralmag(
    rho_3d: np.ndarray,
    m_3d: np.ndarray,
    mesh_or_state,
    *,
    normalize: bool = False,
):
    """Convert MBIR cell arrays to a NeuralMag nodal ``VectorFunction``.

    This is the reverse-direction helper for setting ``state.m`` from MBIR's
    cell-centered representation. Support ``rho`` remains a separate cell-based
    quantity in NeuralMag, so this function returns magnetization only.
    """
    try:
        import neuralmag as nm
    except ImportError as exc:
        raise ImportError("mbir_rho_m_to_neuralmag requires the neuralmag package.") from exc

    state = mesh_or_state if hasattr(mesh_or_state, "tensor") else nm.State(mesh_or_state)
    state_tensor = mbir_rho_m_to_neuralmag_state(
        rho_3d,
        m_3d,
        state,
        state_is_nodal=True,
        normalize=normalize,
    )
    return nm.VectorFunction(state, tensor=state.tensor(state_tensor[..., 1:]))

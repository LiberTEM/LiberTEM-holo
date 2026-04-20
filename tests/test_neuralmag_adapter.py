import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


_ADAPTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "libertem_holo"
    / "base"
    / "mbir"
    / "neuralmag_adapter.py"
)
_SPEC = importlib.util.spec_from_file_location("neuralmag_adapter", _ADAPTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
neuralmag_state_to_mbir_rho_m = _MODULE.neuralmag_state_to_mbir_rho_m


def test_neuralmag_state_to_mbir_rho_m_nodal_to_cell_and_transpose():
    state = np.arange(3 * 4 * 5 * 4, dtype=float).reshape(3, 4, 5, 4)

    rho, m = neuralmag_state_to_mbir_rho_m(state, voxel_size=(1.0, 1.0, 1.0))

    cell = (
        state[:-1, :-1, :-1, :]
        + state[1:, :-1, :-1, :]
        + state[:-1, 1:, :-1, :]
        + state[:-1, :-1, 1:, :]
        + state[1:, 1:, :-1, :]
        + state[1:, :-1, 1:, :]
        + state[:-1, 1:, 1:, :]
        + state[1:, 1:, 1:, :]
    ) / 8.0
    expected_rho = np.transpose(cell[..., 0], (2, 1, 0))
    expected_m = np.transpose(cell[..., 1:], (2, 1, 0, 3))

    assert rho.shape == (4, 3, 2)
    assert m.shape == (4, 3, 2, 3)
    assert_allclose(rho, expected_rho)
    assert_allclose(m, expected_m)


def test_neuralmag_state_to_mbir_rho_m_cell_centered_state_keeps_values():
    state = np.arange(2 * 3 * 4 * 4, dtype=float).reshape(2, 3, 4, 4)

    rho, m = neuralmag_state_to_mbir_rho_m(
        state,
        voxel_size=1.0,
        state_is_nodal=False,
    )

    assert rho.shape == (4, 3, 2)
    assert m.shape == (4, 3, 2, 3)
    assert rho[2, 1, 0] == state[0, 1, 2, 0]
    assert_allclose(m[2, 1, 0], state[0, 1, 2, 1:])


def test_neuralmag_state_to_mbir_rho_m_rejects_anisotropic_voxel_sizes():
    state = np.zeros((2, 2, 2, 4), dtype=float)

    with pytest.raises(ValueError, match="isotropic voxels"):
        neuralmag_state_to_mbir_rho_m(
            state,
            voxel_size=(1.0, 2.0, 1.0),
            state_is_nodal=False,
        )


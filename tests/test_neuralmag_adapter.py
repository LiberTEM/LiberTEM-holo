import importlib.util
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest
from numpy.testing import assert_allclose


nm = pytest.importorskip("neuralmag")


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
mbir_rho_m_to_neuralmag = _MODULE.mbir_rho_m_to_neuralmag
mbir_rho_m_to_neuralmag_state = _MODULE.mbir_rho_m_to_neuralmag_state


class VortexDiscFixture(TypedDict):
    rho_true: np.ndarray
    m_true: np.ndarray
    phi_true: np.ndarray
    pixel_size_nm: float


def _load_cached_vortex_disc_fixture(size: int) -> VortexDiscFixture:
    fixture_path = (
        Path(__file__).resolve().parent / "test_mbir_data" / f"vortex_disc_{size}_ku0.npz"
    )
    with np.load(fixture_path) as data:
        return {
            "rho_true": data["rho_true"],
            "m_true": data["m_true"],
            "phi_true": data["phi_true"],
            "pixel_size_nm": float(np.asarray(data["pixel_size_nm"])),
        }


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
    expected_rho = cell[..., 0]
    expected_m = cell[..., 1:]

    assert rho.shape == (2, 3, 4)
    assert m.shape == (2, 3, 4, 3)
    assert_allclose(rho, expected_rho)
    assert_allclose(m, expected_m)


def test_neuralmag_state_to_mbir_rho_m_cell_centered_state_keeps_values():
    state = np.arange(2 * 3 * 4 * 4, dtype=float).reshape(2, 3, 4, 4)

    rho, m = neuralmag_state_to_mbir_rho_m(
        state,
        voxel_size=1.0,
        state_is_nodal=False,
    )

    assert rho.shape == (2, 3, 4)
    assert m.shape == (2, 3, 4, 3)
    assert rho[0, 1, 2] == state[0, 1, 2, 0]
    assert_allclose(m[0, 1, 2], state[0, 1, 2, 1:])


def test_neuralmag_state_to_mbir_rho_m_rejects_anisotropic_voxel_sizes():
    state = np.zeros((2, 2, 2, 4), dtype=float)

    with pytest.raises(ValueError, match="isotropic voxels"):
        neuralmag_state_to_mbir_rho_m(
            state,
            voxel_size=(1.0, 2.0, 1.0),
            state_is_nodal=False,
        )


def test_mbir_rho_m_to_neuralmag_state_round_trip_cell_data():
    mesh = nm.Mesh((4, 5, 6), (1.0, 1.0, 1.0))
    z, y, x = np.meshgrid(
        np.arange(4, dtype=float),
        np.arange(5, dtype=float),
        np.arange(6, dtype=float),
        indexing="ij",
    )
    rho = ((x - 2.5) ** 2 + (y - 2.0) ** 2 <= 4.0).astype(float)
    m = np.stack(
        [
            np.cos(0.2 * x) * rho,
            np.sin(0.3 * y) * rho,
            np.tanh(z - 1.5) * rho,
        ],
        axis=-1,
    )

    state_tensor = mbir_rho_m_to_neuralmag_state(rho, m, mesh, state_is_nodal=False)
    rho_rt, m_rt = neuralmag_state_to_mbir_rho_m(
        state_tensor,
        voxel_size=1.0,
        state_is_nodal=False,
    )

    assert state_tensor.shape == (4, 5, 6, 4)
    assert rho_rt.shape == rho.shape
    assert m_rt.shape == m.shape
    assert_allclose(rho_rt, rho, atol=1e-12)
    assert_allclose(m_rt, m, atol=1e-12)


def test_mbir_rho_m_to_neuralmag_returns_vector_function_with_unit_norm_preserved():
    mesh = nm.Mesh((3, 4, 5), (1.0, 1.0, 1.0))
    rho = np.ones((3, 4, 5), dtype=float)
    m = np.zeros((3, 4, 5, 3), dtype=float)
    m[..., 0] = 1.0

    m_node = mbir_rho_m_to_neuralmag(rho, m, mesh)

    assert isinstance(m_node, nm.VectorFunction)
    back_to_cell = np.asarray(m_node.to_cell().tensor)
    norms = np.linalg.norm(back_to_cell, axis=-1)
    assert_allclose(norms, 1.0, atol=1e-12)


def test_mbir_rho_m_to_neuralmag_cached_vortex_round_trip_rmse():
    fixture = _load_cached_vortex_disc_fixture(32)
    rho = fixture["rho_true"]
    m = fixture["m_true"]
    pixel_size_nm = fixture["pixel_size_nm"]

    mesh = nm.Mesh(rho.shape, (pixel_size_nm * 1e-9,) * 3)
    state = nm.State(mesh)
    m_node = mbir_rho_m_to_neuralmag(rho, m, state, normalize=False)
    m_back = np.asarray(m_node.to_cell().tensor)

    rmse = np.sqrt(np.mean((m_back - m) ** 2))

    assert rmse < 1e-2


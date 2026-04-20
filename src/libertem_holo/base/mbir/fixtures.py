from __future__ import annotations

from pathlib import Path

import numpy as np
import unxt as u

from .forward import forward_phase_from_density_and_magnetization
from .neuralmag_adapter import mbir_rho_m_to_neuralmag
from .synthetic import soft_disc_support, vortex_magnetization


DEFAULT_VORTEX_RADIUS_FRACTION = 0.32
DEFAULT_EDGE_WIDTH_CELLS = 1.2
DEFAULT_VOXEL_SIZE_NM = 5.0
DEFAULT_MS_A_PER_M = 8e5
DEFAULT_A_J_PER_M = 1.3e-11
DEFAULT_KU_J_PER_M3 = 0.0
DEFAULT_ALPHA = 0.5
DEFAULT_RELAX_TOLERANCE_1_PER_S = 1e9
DEFAULT_MAX_STEPS = 256
DEFAULT_RELAX_DT_S = 3e-12


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_fixture_dir() -> Path:
    return _repo_root() / "tests" / "test_mbir_data"


def vortex_disc_fixture_path(
    size: int,
    *,
    ku: float = DEFAULT_KU_J_PER_M3,
    data_dir: str | Path | None = None,
) -> Path:
    suffix = f"{int(round(ku))}" if float(ku).is_integer() else str(ku).replace(".", "p")
    root = _default_fixture_dir() if data_dir is None else Path(data_dir)
    return root / f"vortex_disc_{size}_ku{suffix}.npz"


def generate_vortex_disc_fixture(
    size: int,
    *,
    ku: float = DEFAULT_KU_J_PER_M3,
    voxel_size_nm: float = DEFAULT_VOXEL_SIZE_NM,
    ms_a_per_m: float = DEFAULT_MS_A_PER_M,
    a_j_per_m: float = DEFAULT_A_J_PER_M,
    alpha: float = DEFAULT_ALPHA,
    radius_fraction: float = DEFAULT_VORTEX_RADIUS_FRACTION,
    edge_width_cells: float = DEFAULT_EDGE_WIDTH_CELLS,
    relax_tolerance_1_per_s: float = DEFAULT_RELAX_TOLERANCE_1_PER_S,
    relax_dt_s: float = DEFAULT_RELAX_DT_S,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, np.ndarray | float]:
    import neuralmag as nm

    mesh = nm.Mesh((size, size, size), (voxel_size_nm * 1e-9,) * 3)
    state = nm.State(mesh)

    radius_cells = radius_fraction * size
    rho_true = np.asarray(
        soft_disc_support(
            (size, size, size),
            radius=radius_cells,
            edge_width=edge_width_cells,
            dtype=np.float32,
        )
    )
    rho_nm = np.clip(rho_true, state.eps, 1.0).astype(np.float32)
    state.rho = nm.CellFunction(state, tensor=state.tensor(rho_nm))

    state.material.Ms = nm.CellFunction(state).fill(ms_a_per_m)
    state.material.A = nm.CellFunction(state).fill(a_j_per_m)
    state.material.Ku = nm.CellFunction(state).fill(ku)
    state.material.alpha = nm.CellFunction(state).fill(alpha)

    if ku != 0.0:
        ku_axis = np.zeros((size, size, size, 3), dtype=np.float32)
        ku_axis[..., 1] = 1.0
        state.material.Ku_axis = nm.VectorCellFunction(state, tensor=state.tensor(ku_axis))

    m0 = np.asarray(
        vortex_magnetization(
            (size, size, size),
            support_zyx=rho_true,
            core_radius=max(1.5, size / 32.0),
            dtype=np.float32,
        )
    )
    state.m = mbir_rho_m_to_neuralmag(rho_true, m0, state, normalize=True)

    nm.ExchangeField().register(state, "exchange")
    nm.DemagField(p=3).register(state, "demag")
    nm.TotalField("exchange", "demag").register(state)
    llg = nm.LLGSolver(state, max_steps=max_steps)
    llg.relax(tol=relax_tolerance_1_per_s, dt=relax_dt_s)

    m_true = np.asarray(state.m.to_cell().tensor)
    rho_true = np.asarray(state.rho.tensor)
    phi_true = np.asarray(
        forward_phase_from_density_and_magnetization(
            rho=rho_true,
            magnetization_3d=m_true,
            pixel_size=u.Quantity(voxel_size_nm, "nm"),
            axis="z",
        )
    )

    return {
        "rho_true": rho_true.astype(np.float32),
        "m_true": m_true.astype(np.float32),
        "phi_true": phi_true.astype(np.float32),
        "pixel_size_nm": float(voxel_size_nm),
    }


def save_vortex_disc_fixture(
    size: int,
    *,
    ku: float = DEFAULT_KU_J_PER_M3,
    data_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    path = vortex_disc_fixture_path(size, ku=ku, data_dir=data_dir)
    if path.exists() and not overwrite:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    fixture = generate_vortex_disc_fixture(size, ku=ku)
    np.savez_compressed(path, **fixture)
    return path


def load_vortex_disc_fixture(
    size: int,
    *,
    ku: float = DEFAULT_KU_J_PER_M3,
    data_dir: str | Path | None = None,
    generate_if_missing: bool = False,
) -> dict[str, np.ndarray | float]:
    path = vortex_disc_fixture_path(size, ku=ku, data_dir=data_dir)
    if not path.exists():
        if not generate_if_missing:
            raise FileNotFoundError(
                f"Fixture {path} does not exist. Generate it first or pass generate_if_missing=True."
            )
        save_vortex_disc_fixture(size, ku=ku, data_dir=data_dir)

    with np.load(path) as data:
        return {
            "rho_true": data["rho_true"],
            "m_true": data["m_true"],
            "phi_true": data["phi_true"],
            "pixel_size_nm": float(np.asarray(data["pixel_size_nm"])),
        }
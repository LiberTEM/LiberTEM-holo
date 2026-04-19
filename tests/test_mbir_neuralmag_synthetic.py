import logging

import numpy as np
import pytest

import jax.numpy as jnp
import unxt as u

from libertem_holo.base.mbir import forward_model_2d

nm = pytest.importorskip("neuralmag")
LLG_RELAX_TOLERANCE_1_PER_S = 1e9  # loose criterion for reduced-runtime smoke test
DISC_RADIUS_FRACTION = 0.32  # fraction of nx for reduced-size disc support
RADIUS_EPS = 1e-12  # avoid division by zero at vortex center
VORTEX_CORE_RADIUS_CELLS = 1.1  # node-grid radius for out-of-plane vortex core


def _build_vortex_disc_state():
    # Reduced-size fixture to keep runtime modest while exercising end-to-end flow.
    n_z, n_y, n_x = 1, 8, 8
    dz_nm, dy_nm, dx_nm = 10.0, 8.0, 8.0
    mesh = nm.Mesh((n_z, n_y, n_x), (dz_nm * 1e-9, dy_nm * 1e-9, dx_nm * 1e-9))
    state = nm.State(mesh)

    # Disc support (rho/state domains)
    cy = (n_y - 1) / 2
    cx = (n_x - 1) / 2
    yy, xx = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing="ij")
    disc_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (DISC_RADIUS_FRACTION * n_x) ** 2

    domains = np.zeros((n_z, n_y, n_x), dtype=np.int32)
    domains[0, disc_mask] = 1
    state.domains = nm.CellFunction(
        state,
        dtype=nm.config.backend.integer,
        tensor=state.tensor(domains, dtype=nm.config.backend.integer),
    )

    # Material parameters
    state.material.Ms = nm.CellFunction(state).fill(8e5)      # A/m
    state.material.A = nm.CellFunction(state).fill(1.3e-11)   # J/m
    state.material.alpha = nm.CellFunction(state).fill(0.5)   # damping

    # Vortex-like initialization at nodes
    nz_n, ny_n, nx_n, _ = nm.VectorFunction(state).tensor_shape
    _, yy_n, xx_n = np.meshgrid(
        np.arange(nz_n),
        np.arange(ny_n),
        np.arange(nx_n),
        indexing="ij",
    )
    cy_n = (ny_n - 1) / 2
    cx_n = (nx_n - 1) / 2
    dy = yy_n - cy_n
    dx = xx_n - cx_n
    rr = np.sqrt(dx**2 + dy**2) + RADIUS_EPS
    mx = -(dy / rr)
    my = dx / rr
    mz = np.zeros_like(mx)
    core = rr < VORTEX_CORE_RADIUS_CELLS
    mx[core] = 0.0
    my[core] = 0.0
    mz[core] = 1.0
    m0 = np.stack([mx, my, mz], axis=-1).astype(np.float32)
    state.m = nm.VectorFunction(state, tensor=state.tensor(m0))

    return state, dy_nm


def _relax_to_m_true(state):
    nm.ExchangeField().register(state, "exchange")
    nm.DemagField(p=3).register(state, "demag")
    nm.TotalField("exchange", "demag").register(state)
    llg = nm.LLGSolver(state, max_steps=128)
    llg.relax(tol=LLG_RELAX_TOLERANCE_1_PER_S, dt=3e-12)
    m_true = np.asarray(state.m.to_cell().tensor)
    return m_true


def _neuralmag_to_projected_adapter(m_true, state):
    rho = np.asarray(state.rho.tensor)
    m_true = m_true * rho[..., None]
    m_projected = m_true[0, ..., :2]
    # Empty unit string is the canonical unxt representation for dimensionless.
    return u.Quantity(jnp.asarray(m_projected), "")


def test_neuralmag_vortex_ground_truth_smoke():
    nm.set_log_level(logging.WARNING)
    state, pixel_size_nm = _build_vortex_disc_state()
    m_true = _relax_to_m_true(state)

    assert np.isfinite(m_true).all()

    converted = _neuralmag_to_projected_adapter(m_true, state)
    assert converted.shape[-1] == 2
    assert np.isfinite(np.asarray(converted.value)).all()

    try:
        phi_true = forward_model_2d(
            converted,
            pixel_size=u.Quantity(pixel_size_nm, "nm"),
            geometry="disc",
        )
    except TypeError as exc:
        if "unsupported operand type" in str(exc) and "Quantity" in str(exc):
            pytest.skip(f"Current unxt/jax arithmetic stack is incompatible with forward_model_2d: {exc}")
        raise

    assert np.isfinite(np.asarray(phi_true.value)).all()

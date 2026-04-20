import numpy as np
import pytest
import unxt as u

from libertem_holo.base.mbir import (
    depth_correlation,
    equilibrium_residual,
    forward_phase_from_density_and_magnetization,
    load_vortex_disc_fixture,
    mz_rmse,
    phase_residual,
    projected_m_error,
    vortex_core_z_error,
)
from libertem_holo.base.mbir.inversion import (
    IdentityBackend,
    NeuralMagCritic,
    SmoothnessBackend,
    invert_magnetization,
    project_unit_norm,
)


def _crop_center_volume(values, size: int):
    start = (values.shape[0] - size) // 2
    stop = start + size
    if values.ndim == 3:
        return values[start:stop, start:stop, start:stop]
    return values[start:stop, start:stop, start:stop, :]


def _load_smoke_fixture(size: int = 16):
    fixture = load_vortex_disc_fixture(32)
    rho = _crop_center_volume(np.asarray(fixture["rho_true"], dtype=np.float32), size)
    m = _crop_center_volume(np.asarray(fixture["m_true"], dtype=np.float32), size)
    pixel_size = u.Quantity(float(fixture["pixel_size_nm"]), "nm")
    phi = np.asarray(
        forward_phase_from_density_and_magnetization(
            rho=rho,
            magnetization_3d=m,
            pixel_size=pixel_size,
            axis="z",
        ),
        dtype=np.float32,
    )
    return rho, m, phi, pixel_size


def _support_center_yx(rho):
    support = np.argwhere(rho.max(axis=0) > 0.5)
    if support.size == 0:
        raise ValueError("Support mask is empty.")
    center = np.round(support.mean(axis=0)).astype(int)
    return int(center[0]), int(center[1])


def _make_neuralmag_backend(rho, pixel_size_nm: float):
    nm = pytest.importorskip("neuralmag")

    mesh = nm.Mesh(rho.shape, (pixel_size_nm * 1e-9,) * 3)
    state = nm.State(mesh)
    state.rho = nm.CellFunction(state, tensor=state.tensor(np.asarray(rho, dtype=np.float32)))
    state.material.Ms = nm.CellFunction(state).fill(8e5)
    state.material.A = nm.CellFunction(state).fill(1.3e-11)
    state.m = nm.VectorCellFunction(state, tensor=state.tensor(np.zeros(rho.shape + (3,), dtype=np.float32)))
    nm.ExchangeField().register(state, "exchange")
    nm.DemagField(p=1).register(state, "demag")
    return NeuralMagCritic.from_state(state, terms=("exchange", "demag"))


def test_project_unit_norm_enforces_support_contract():
    rho = np.zeros((3, 3, 3), dtype=np.float32)
    rho[1, 1, 1] = 1.0
    rho[1, 1, 2] = 1.0
    rho[1, 2, 1] = 1.0
    m = np.zeros((3, 3, 3, 3), dtype=np.float32)
    m[1, 1, 2] = np.array([0.0, 3.0, 4.0], dtype=np.float32)
    m[1, 2, 1] = np.array([1.0, 2.0, 2.0], dtype=np.float32)

    projected = np.asarray(project_unit_norm(m, rho))
    norms = np.linalg.norm(projected, axis=-1)

    assert np.allclose(norms[rho > 0.5], 1.0, atol=1e-6)
    assert np.count_nonzero(projected[rho <= 0.5]) == 0


@pytest.mark.parametrize("init", ["zero", "uniform_x", "analytic_vortex"])
def test_regime_a_init_options_respect_support_contract(init):
    rho, _m_true, phi_true, pixel_size = _load_smoke_fixture(size=16)

    result = invert_magnetization(
        phi_true,
        rho,
        IdentityBackend(),
        pixel_size=pixel_size,
        lambda_phys=0.0,
        max_iter=1,
        lr=5e-2,
        init=init,
    )

    norms = np.linalg.norm(np.asarray(result.m_recon), axis=-1)
    assert np.allclose(norms[rho > 0.5], 1.0, atol=1e-6)
    assert np.count_nonzero(np.asarray(result.m_recon)[rho <= 0.5]) == 0


def test_regime_a_accepts_array_warm_start():
    rho, _m_true, phi_true, pixel_size = _load_smoke_fixture(size=16)
    warm_start = np.zeros(rho.shape + (3,), dtype=np.float32)
    warm_start[..., 1] = 1.0

    result = invert_magnetization(
        phi_true,
        rho,
        IdentityBackend(),
        pixel_size=pixel_size,
        lambda_phys=0.0,
        max_iter=1,
        lr=5e-2,
        init=warm_start,
    )

    norms = np.linalg.norm(np.asarray(result.m_recon), axis=-1)
    assert np.allclose(norms[rho > 0.5], 1.0, atol=1e-6)


@pytest.mark.parametrize(
    ("backend_factory", "lambda_phys"),
    [
        (lambda rho, px: IdentityBackend(), 0.0),
        (lambda rho, px: SmoothnessBackend(), 1e-3),
        (_make_neuralmag_backend, 1e-8),
    ],
)
def test_regime_a_backends_reduce_loss_on_cached_fixture(backend_factory, lambda_phys):
    rho, m_true, phi_true, pixel_size = _load_smoke_fixture(size=16)
    backend = backend_factory(rho, float(pixel_size.value))

    result = invert_magnetization(
        phi_true,
        rho,
        backend,
        pixel_size=pixel_size,
        lambda_phys=lambda_phys,
        max_iter=10,
        lr=5e-2,
        init="zero",
    )

    assert result.loss_history.shape == (10,)
    assert np.all(np.isfinite(np.asarray(result.loss_history)))
    assert float(result.loss_history[-1]) < float(result.loss_history[0])
    assert result.m_recon.shape == rho.shape + (3,)
    assert result.phi_pred.shape == phi_true.shape

    yz = _support_center_yx(rho)
    rel_phase = float(phase_residual(result.phi_pred, phi_true))
    rel_proj = float(projected_m_error(result.m_recon, m_true))
    mz_err = float(mz_rmse(result.m_recon, m_true))
    depth_corr = float(depth_correlation(result.m_recon, m_true, yz))
    core_z_err = float(vortex_core_z_error(result.m_recon, m_true, yz))
    eq_residual = float(equilibrium_residual(result.m_recon, SmoothnessBackend(), rho=rho))

    assert np.isfinite(rel_phase)
    assert np.isfinite(rel_proj)
    assert np.isfinite(mz_err)
    assert np.isfinite(depth_corr)
    assert np.isfinite(core_z_err)
    assert np.isfinite(eq_residual)
    assert -1.0 <= depth_corr <= 1.0
    assert core_z_err >= 0.0
    assert eq_residual >= 0.0
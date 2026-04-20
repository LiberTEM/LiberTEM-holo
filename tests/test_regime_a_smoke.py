import numpy as np
import pytest
import unxt as u

from libertem_holo.base.mbir import (
    forward_phase_from_density_and_magnetization,
    load_vortex_disc_fixture,
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
    m = np.zeros((3, 3, 3, 3), dtype=np.float32)

    projected = np.asarray(project_unit_norm(m, rho))
    norms = np.linalg.norm(projected, axis=-1)

    assert norms[1, 1, 1] == pytest.approx(1.0)
    assert np.count_nonzero(projected[rho <= 0.5]) == 0


@pytest.mark.parametrize(
    ("backend_factory", "lambda_phys"),
    [
        (lambda rho, px: IdentityBackend(), 0.0),
        (lambda rho, px: SmoothnessBackend(), 1e-3),
        (_make_neuralmag_backend, 1e-8),
    ],
)
def test_regime_a_backends_reduce_loss_on_cached_fixture(backend_factory, lambda_phys):
    rho, _m_true, phi_true, pixel_size = _load_smoke_fixture(size=16)
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
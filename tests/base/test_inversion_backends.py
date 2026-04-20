import numpy as np
import pytest
from numpy.testing import assert_allclose

from libertem_holo.base.mbir import soft_disc_support, vortex_magnetization
from libertem_holo.base.mbir.energy_backend import NeuralMagEnergyBackend
from libertem_holo.base.mbir.inversion import IdentityBackend, NeuralMagCritic, SmoothnessBackend


def test_identity_backend_returns_empty_terms():
    rho = np.ones((4, 4, 4), dtype=np.float32)
    m = np.zeros((4, 4, 4, 3), dtype=np.float32)

    backend = IdentityBackend()
    field = backend.prepare(rho, m)

    assert field.rho.shape == rho.shape
    assert field.m.shape == m.shape
    assert backend.energies(field) == {}


def test_smoothness_backend_zero_for_uniform_and_positive_for_vortex():
    rho = np.asarray(soft_disc_support((8, 8, 8), radius=2.8, edge_width=1.2), dtype=np.float32)
    uniform_m = np.zeros((8, 8, 8, 3), dtype=np.float32)
    uniform_m[..., 0] = 1.0
    vortex_m = np.asarray(vortex_magnetization((8, 8, 8), support_zyx=rho), dtype=np.float32)

    backend = SmoothnessBackend()

    uniform_energy = backend.energies(backend.prepare(rho, uniform_m))["smoothness"]
    vortex_energy = backend.energies(backend.prepare(rho, vortex_m))["smoothness"]

    assert float(uniform_energy) == pytest.approx(0.0, abs=1e-6)
    assert float(vortex_energy) > 0.0


def test_neuralmag_critic_matches_wrapped_backend():
    nm = pytest.importorskip("neuralmag")

    mesh = nm.Mesh((2, 2, 2), (5e-9, 5e-9, 5e-9))
    state = nm.State(mesh)
    state.material.Ms = nm.CellFunction(state).fill(8e5)
    state.material.A = nm.CellFunction(state).fill(1.3e-11)

    m0 = np.zeros((2, 2, 2, 3), dtype=np.float32)
    m0[..., 0] = 1.0
    state.m = nm.VectorCellFunction(state, tensor=state.tensor(m0))

    nm.ExchangeField().register(state, "exchange")
    nm.DemagField(p=1).register(state, "demag")

    rho = np.ones((2, 2, 2), dtype=np.float32)
    m = m0.copy()

    wrapped = NeuralMagEnergyBackend.from_state(state, ("exchange", "demag"))
    critic = NeuralMagCritic(wrapped)

    direct_terms = wrapped.energy_terms(rho, m)
    critic_terms = critic.energies(critic.prepare(rho, m))

    assert set(critic_terms) == {"exchange", "demag"}
    assert_allclose(critic_terms["exchange"], direct_terms["exchange"])
    assert_allclose(critic_terms["demag"], direct_terms["demag"])
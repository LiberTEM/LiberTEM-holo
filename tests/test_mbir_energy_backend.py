import numpy as np
import pytest
from numpy.testing import assert_allclose
import jax
import jax.numpy as jnp

from libertem_holo.base.mbir_energy_backend import NeuralMagEnergyBackend
from libertem_holo.base.mbir.neuralmag_adapter import neuralmag_state_to_mbir_rho_m


def test_resolves_terms_once_at_initialization():
    calls: list[str] = []

    def resolver(term):
        calls.append(term)
        return lambda weighted_m: float(np.sum(weighted_m))

    backend = NeuralMagEnergyBackend(("exchange", "anisotropy"), resolver=resolver)

    assert backend.terms == ("exchange", "anisotropy")
    assert calls == ["exchange", "anisotropy"]

    rho = np.ones((2, 2), dtype=float)
    m = np.ones((2, 2, 3), dtype=float)
    backend.energies(rho=rho, m=m)
    assert calls == ["exchange", "anisotropy"]


def test_energies_returns_finite_scalars_and_uses_rho_times_m():
    seen: dict[str, np.ndarray] = {}

    def resolver(term):
        def fn(weighted_m):
            seen[term] = weighted_m
            return np.linalg.norm(weighted_m)

        return fn

    backend = NeuralMagEnergyBackend(("exchange",), resolver=resolver)
    rho = np.array([[2.0, 0.5], [1.0, 3.0]])
    m = np.array(
        [
            [[1.0, 2.0, 0.0], [4.0, -2.0, 1.0]],
            [[-1.0, 0.0, 2.0], [0.5, 0.5, 0.5]],
        ]
    )

    energies = backend.energies(rho=rho, m=m)

    assert set(energies) == {"exchange"}
    assert isinstance(energies["exchange"], float)
    assert np.isfinite(energies["exchange"])
    np.testing.assert_allclose(seen["exchange"], rho[..., None] * m)


def test_energies_uses_custom_input_adapter_before_resolver():
    seen: dict[str, np.ndarray] = {}

    def resolver(term):
        def fn(weighted_m):
            seen[term] = weighted_m
            return np.sum(weighted_m)

        return fn

    def input_adapter(rho, m):
        return np.transpose(np.asarray(rho)[..., None] * np.asarray(m), (2, 1, 0, 3))

    backend = NeuralMagEnergyBackend(("exchange",), resolver=resolver, input_adapter=input_adapter)
    rho = np.arange(24, dtype=float).reshape(2, 3, 4) + 1.0
    m = np.ones((2, 3, 4, 3), dtype=float)

    energies = backend.energies(rho=rho, m=m)

    assert set(energies) == {"exchange"}
    assert np.isfinite(energies["exchange"])
    assert seen["exchange"].shape == (4, 3, 2, 3)
    assert_allclose(seen["exchange"], np.transpose(rho[..., None] * m, (2, 1, 0, 3)))


def test_from_state_resolves_terms_once_and_uses_state_layout():
    resolved_calls: list[tuple[str, tuple[str, ...]]] = []
    seen: dict[str, np.ndarray] = {}

    class DummyM:
        spaces = "ccc"

    class DummyState:
        m = DummyM()

        def resolve(self, name, args):
            resolved_calls.append((name, tuple(args)))

            def fn(weighted_m):
                seen[name] = weighted_m
                return np.linalg.norm(weighted_m)

            return fn

    backend = NeuralMagEnergyBackend.from_state(DummyState(), ("exchange", "demag"))
    rho = np.arange(24, dtype=float).reshape(2, 3, 4) + 1.0
    m = np.ones((2, 3, 4, 3), dtype=float)

    energies = backend.energies(rho=rho, m=m)

    assert resolved_calls == [
        ("E_exchange", ("m",)),
        ("E_demag", ("m",)),
    ]
    assert set(energies) == {"exchange", "demag"}
    assert_allclose(seen["E_exchange"], rho[..., None] * m)
    assert_allclose(seen["E_demag"], rho[..., None] * m)


def test_from_state_with_real_neuralmag_resolve_returns_finite_energies():
    nm = pytest.importorskip("neuralmag")

    mesh = nm.Mesh((2, 2, 2), (5e-9, 5e-9, 5e-9))
    state = nm.State(mesh)
    state.material.Ms = nm.CellFunction(state).fill(8e5)
    state.material.A = nm.CellFunction(state).fill(1.3e-11)

    nz, ny, nx, _ = nm.VectorFunction(state).tensor_shape
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=float),
        np.arange(ny, dtype=float),
        np.arange(nx, dtype=float),
        indexing="ij",
    )
    mx = -(yy - 1.0)
    my = xx - 1.0
    mz = 0.25 * (zz - 1.0)
    m = np.stack([mx, my, mz], axis=-1)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    m = m / np.where(norms > 0.0, norms, 1.0)
    state.m = nm.VectorFunction(state, tensor=state.tensor(m.astype(np.float32)))

    nm.ExchangeField().register(state, "exchange")
    nm.DemagField(p=1).register(state, "demag")

    cell_state = np.concatenate(
        [
            np.asarray(state.rho.tensor)[..., np.newaxis],
            np.asarray(state.m.to_cell().tensor),
        ],
        axis=-1,
    )
    rho_mbir, m_mbir = neuralmag_state_to_mbir_rho_m(
        cell_state,
        voxel_size=(5.0, 5.0, 5.0),
        state_is_nodal=False,
    )

    backend = NeuralMagEnergyBackend.from_state(state, ("exchange", "demag"))
    energies = backend.energies(rho=rho_mbir, m=m_mbir)

    assert np.isfinite(energies["exchange"])
    assert np.isfinite(energies["demag"])
    assert energies["exchange"] >= 0.0
    assert energies["demag"] >= 0.0


def test_from_state_real_resolved_callables_work_in_jitted_loss_path():
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
    backend = NeuralMagEnergyBackend.from_state(state, ("exchange", "demag"))

    rho = jnp.ones((2, 2, 2), dtype=jnp.float32)
    m = jnp.asarray(m0)

    @jax.jit
    def loss_fn(m_arg):
        terms = backend.energy_terms(rho=rho, m=m_arg)
        return terms["exchange"] + terms["demag"]

    value = loss_fn(m)
    grad = jax.grad(loss_fn)(m)

    assert jnp.isfinite(value)
    assert grad.shape == m.shape
    assert jnp.all(jnp.isfinite(grad))

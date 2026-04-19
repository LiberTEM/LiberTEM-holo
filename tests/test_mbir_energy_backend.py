import numpy as np

from libertem_holo.base.mbir_energy_backend import NeuralMagEnergyBackend


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

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.numpy as jnp

from ..energy_backend import NeuralMagEnergyBackend


@dataclass(frozen=True)
class FieldState:
    rho: jnp.ndarray
    m: jnp.ndarray


def _validate_rho_m(rho, m) -> tuple[jnp.ndarray, jnp.ndarray]:
    rho_arr = jnp.asarray(rho)
    m_arr = jnp.asarray(m)
    if rho_arr.ndim != 3:
        raise ValueError(f"rho must have shape (Z, Y, X), got {rho_arr.shape}.")
    if m_arr.ndim != 4 or m_arr.shape[-1] != 3:
        raise ValueError(
            "m must have shape (Z, Y, X, 3), "
            f"got {m_arr.shape}."
        )
    if m_arr.shape[:-1] != rho_arr.shape:
        raise ValueError(
            "rho and m must share spatial dimensions, "
            f"got rho {rho_arr.shape} and m {m_arr.shape[:-1]}."
        )
    return rho_arr, m_arr


@runtime_checkable
class PhysicsBackend(Protocol):
    def prepare(self, rho, m) -> FieldState: ...

    def energies(self, field: FieldState) -> dict[str, jnp.ndarray]: ...


class IdentityBackend:
    def prepare(self, rho, m) -> FieldState:
        rho_arr, m_arr = _validate_rho_m(rho, m)
        return FieldState(rho=rho_arr, m=m_arr)

    def energies(self, field: FieldState) -> dict[str, jnp.ndarray]:
        return {}


class SmoothnessBackend:
    def __init__(self, *, weight_by_rho: bool = True) -> None:
        self._weight_by_rho = weight_by_rho

    def prepare(self, rho, m) -> FieldState:
        rho_arr, m_arr = _validate_rho_m(rho, m)
        return FieldState(rho=rho_arr, m=m_arr)

    def energies(self, field: FieldState) -> dict[str, jnp.ndarray]:
        return {"smoothness": self._smoothness_energy(field.rho, field.m)}

    def _smoothness_energy(self, rho: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
        total = jnp.asarray(0.0, dtype=m.dtype)
        for axis in range(3):
            diffs = jnp.diff(m, axis=axis)
            if self._weight_by_rho:
                left = jnp.take(rho, indices=jnp.arange(rho.shape[axis] - 1), axis=axis)
                right = jnp.take(rho, indices=jnp.arange(1, rho.shape[axis]), axis=axis)
                weight = 0.5 * (left + right)
            else:
                weight = jnp.ones(diffs.shape[:-1], dtype=m.dtype)
            total = total + jnp.sum(weight[..., None] * diffs ** 2)
        return total


class WeightedBackend:
    def __init__(self, backend: PhysicsBackend, *, weight: float) -> None:
        self._backend = backend
        self._weight = float(weight)

    def prepare(self, rho, m) -> FieldState:
        return self._backend.prepare(rho, m)

    def energies(self, field: FieldState) -> dict[str, jnp.ndarray]:
        return {
            name: jnp.asarray(self._weight, dtype=value.dtype) * value
            for name, value in self._backend.energies(field).items()
        }


class CombinedBackend:
    def __init__(self, *backends: PhysicsBackend) -> None:
        if not backends:
            raise ValueError("CombinedBackend requires at least one backend.")
        self._backends = tuple(backends)

    def prepare(self, rho, m) -> FieldState:
        rho_arr, m_arr = _validate_rho_m(rho, m)
        return FieldState(rho=rho_arr, m=m_arr)

    def energies(self, field: FieldState) -> dict[str, jnp.ndarray]:
        energy_terms: dict[str, jnp.ndarray] = {}
        for backend in self._backends:
            prepared = backend.prepare(field.rho, field.m)
            terms = backend.energies(prepared)
            duplicate_terms = set(energy_terms).intersection(terms)
            if duplicate_terms:
                duplicate_str = ", ".join(sorted(duplicate_terms))
                raise ValueError(
                    "CombinedBackend received duplicate energy term names: "
                    f"{duplicate_str}."
                )
            energy_terms.update(terms)
        return energy_terms


class NeuralMagCritic:
    def __init__(self, backend: NeuralMagEnergyBackend) -> None:
        self._backend = backend

    @classmethod
    def from_state(
        cls,
        state,
        terms=("exchange", "demag"),
    ) -> NeuralMagCritic:
        return cls(NeuralMagEnergyBackend.from_state(state, terms))

    def prepare(self, rho, m) -> FieldState:
        rho_arr, m_arr = _validate_rho_m(rho, m)
        return FieldState(rho=rho_arr, m=m_arr)

    def energies(self, field: FieldState) -> dict[str, jnp.ndarray]:
        return self._backend.energy_terms(field.rho, field.m)
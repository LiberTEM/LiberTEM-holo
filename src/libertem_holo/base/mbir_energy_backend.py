from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib import import_module

import numpy as np


ResolveFn = Callable[[str], Callable[[np.ndarray], float]]
AdapterFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _resolve_neuralmag_energy_term(term: str) -> Callable[[np.ndarray], float]:
    module_names = (
        "neuralmag.energy",
        "neuralmag.energies",
    )
    for module_name in module_names:
        try:
            module = import_module(module_name)
        except ImportError:
            continue

        attr = getattr(module, term, None)
        if callable(attr):
            return attr
        for resolver_name in ("resolve", "resolve_energy", "get_energy"):
            resolver = getattr(module, resolver_name, None)
            if callable(resolver):
                resolved = resolver(term)
                if callable(resolved):
                    return resolved
    msg = f"Could not resolve NeuralMag energy term '{term}'."
    raise ValueError(msg)


class NeuralMagEnergyBackend:
    """Evaluate pre-resolved NeuralMag energy terms on MBIR magnetization arrays.

    The JIT contract is:
    - call ``state.resolve(...)`` once, outside any ``jax.jit`` boundary
    - use :meth:`energy_terms` inside a jitted loss path when the configured
      ``input_adapter`` is itself JIT-safe
    - use :meth:`energies` for eager validation and reporting when Python floats
      are desired
    """

    def __init__(
        self,
        terms: Iterable[str],
        *,
        resolver: ResolveFn | None = None,
        input_adapter: AdapterFn | None = None,
    ) -> None:
        self._terms = tuple(terms)
        self._resolver = _resolve_neuralmag_energy_term if resolver is None else resolver
        self._input_adapter = (
            (lambda rho, m: np.asarray(rho)[..., np.newaxis] * np.asarray(m))
            if input_adapter is None
            else input_adapter
        )
        self._energy_functions: Mapping[str, Callable[[np.ndarray], float]] = {
            term: self._resolver(term) for term in self._terms
        }

    @classmethod
    def from_state(
        cls,
        state,
        terms: Iterable[str],
    ) -> NeuralMagEnergyBackend:
        def resolver(term: str) -> Callable[[np.ndarray], float]:
            return state.resolve(f"E_{term}", ["m"])

        def input_adapter(rho: np.ndarray, m: np.ndarray) -> np.ndarray:
            spaces = getattr(getattr(state, "m", None), "spaces", "")
            if set(spaces) == {"c"}:
                return rho[..., np.newaxis] * m

            from libertem_holo.base.mbir.neuralmag_adapter import mbir_rho_m_to_neuralmag

            return np.asarray(mbir_rho_m_to_neuralmag(rho, m, state).tensor)

        return cls(terms, resolver=resolver, input_adapter=input_adapter)

    @property
    def terms(self) -> tuple[str, ...]:
        return self._terms

    def energy_terms(self, rho: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
        weighted_m = self._input_adapter(rho, m)
        return {term: fn(weighted_m) for term, fn in self._energy_functions.items()}

    def energies(self, rho: np.ndarray, m: np.ndarray) -> dict[str, float]:
        raw_terms = self.energy_terms(rho, m)
        out: dict[str, float] = {}
        for term, value in raw_terms.items():
            energy = float(np.asarray(value))
            if not np.isfinite(energy):
                msg = f"NeuralMag energy '{term}' evaluated to non-finite value."
                raise ValueError(msg)
            out[term] = energy
        return out

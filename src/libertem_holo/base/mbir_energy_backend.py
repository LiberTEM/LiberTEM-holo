from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib import import_module

import numpy as np


ResolveFn = Callable[[str], Callable[[np.ndarray], float]]


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
    def __init__(
        self,
        terms: Iterable[str],
        *,
        resolver: ResolveFn | None = None,
    ) -> None:
        self._terms = tuple(terms)
        self._resolver = _resolve_neuralmag_energy_term if resolver is None else resolver
        self._energy_functions: Mapping[str, Callable[[np.ndarray], float]] = {
            term: self._resolver(term) for term in self._terms
        }

    @property
    def terms(self) -> tuple[str, ...]:
        return self._terms

    def energies(self, rho: np.ndarray, m: np.ndarray) -> dict[str, float]:
        rho_arr = np.asarray(rho)
        m_arr = np.asarray(m)
        weighted_m = rho_arr[..., np.newaxis] * m_arr
        out: dict[str, float] = {}
        for term, fn in self._energy_functions.items():
            energy = float(np.asarray(fn(weighted_m)))
            if not np.isfinite(energy):
                msg = f"NeuralMag energy '{term}' evaluated to non-finite value."
                raise ValueError(msg)
            out[term] = energy
        return out

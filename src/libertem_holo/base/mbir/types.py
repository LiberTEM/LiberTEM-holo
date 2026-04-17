"""Result types, solver configuration, and Newton-CG preconditioners.

Holds the ``NamedTuple`` result containers returned by the public API
(:class:`SolverResult`, :class:`LCurveResult`,
:class:`BootstrapThresholdResult`), the ``NewtonCGConfig`` solver config,
the ``_resolve_solver_config`` dispatcher, and the experimental Newton-CG
preconditioners.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import unxt as u

from .units import (
    RampCoeffs,
    _assert_quantity_compatible,
    _assert_ramp_coeffs_units,
)

def _assert_solver_result_units(result: SolverResult) -> None:
    """Assert the public unit contract of SolverResult."""
    _assert_quantity_compatible(result.magnetization, "", "result.magnetization")
    _assert_ramp_coeffs_units(result.ramp_coeffs, "result.ramp_coeffs")
    _assert_quantity_compatible(result.loss_history, "rad2", "result.loss_history")


def _assert_lcurve_result_units(result: LCurveResult) -> None:
    """Assert the public unit contract of LCurveResult."""
    _assert_quantity_compatible(result.lambdas, "rad2", "result.lambdas")
    _assert_quantity_compatible(result.data_misfits, "rad2", "result.data_misfits")
    _assert_quantity_compatible(result.reg_norms, "", "result.reg_norms")
    _assert_quantity_compatible(result.magnetizations, "", "result.magnetizations")
    _assert_ramp_coeffs_units(result.ramp_coeffs, "result.ramp_coeffs")

class SolverResult(NamedTuple):
    """Result returned by :func:`solve_mbir_2d`.

    Attributes
    ----------
    magnetization : Quantity["dimensionless"]
        Reconstructed **projected** (thickness-integrated) in-plane
        magnetization of shape ``(N, M, 2)``.  Dimensionless
        (:math:`M / M_s`).
    ramp_coeffs : RampCoeffs
        Background phase-ramp coefficients with explicit units.
    loss_history : Quantity["rad2"]
        Per-step loss values.
    converged : bool
        Whether the CG solver converged within the iteration budget.
    """
    magnetization: u.Quantity
    ramp_coeffs: RampCoeffs
    loss_history: u.Quantity
    converged: bool


class LCurveResult(NamedTuple):
    """Result returned by :func:`lcurve_sweep` and :func:`lcurve_sweep_vmap`.

    Attributes
    ----------
    lambdas : Quantity["rad2"]
        Regularization weights used in the sweep.
    data_misfits : Quantity["rad2"]
        Data-fidelity term for each lambda.
    reg_norms : Quantity["dimensionless"]
        Unweighted regularization norm for each lambda.
    magnetizations : Quantity["dimensionless"]
        Reconstructed projected magnetizations, shape
        ``(n_lambdas, N, M, 2)``.
    ramp_coeffs : RampCoeffs
        Background ramp coefficients per lambda (batched
        ``Quantity`` fields with leading lambda dimension).
    corner_index : int
        Index of the detected L-curve corner.
    """
    lambdas: u.Quantity
    data_misfits: u.Quantity
    reg_norms: u.Quantity
    magnetizations: u.Quantity
    ramp_coeffs: RampCoeffs
    corner_index: int


class BootstrapThresholdResult(NamedTuple):
    """Result returned by :func:`bootstrap_threshold_uncertainty_2d`."""
    threshold: float
    threshold_low: float
    threshold_high: float
    threshold_draws: np.ndarray
    magnetizations: u.Quantity  # Quantity["dimensionless"]
    mean_magnetization: u.Quantity  # Quantity["dimensionless"]
    mean_norm: u.Quantity  # Quantity["dimensionless"]
    norm_low: u.Quantity  # Quantity["dimensionless"]
    norm_high: u.Quantity  # Quantity["dimensionless"]
    norm_ci95: u.Quantity  # Quantity["dimensionless"]
    relative_ci95: u.Quantity  # Quantity["dimensionless"]
    mask_frequency: np.ndarray
    local_induction_mean_samples: u.Quantity | None  # Quantity["T"]
    local_induction_mean: u.Quantity | None  # Quantity["T"]
    local_induction_mean_low: u.Quantity | None  # Quantity["T"]
    local_induction_mean_high: u.Quantity | None  # Quantity["T"]
    local_induction_mean_ci95: u.Quantity | None  # Quantity["T"]
    local_induction_roi_pixels: np.ndarray | None


@dataclasses.dataclass(frozen=True)
class NewtonCGConfig:
    """Configuration for the Newton-CG solver.

    Parameters
    ----------
    cg_maxiter : int
        Maximum number of conjugate-gradient iterations used to
        solve the Newton system ``H @ delta = -g``.
    cg_tol : float
        CG convergence tolerance (relative residual norm).
    preconditioner : {None, "block_jacobi"}
        Experimental inverse preconditioner for the Newton-CG linear
        solve. ``"block_jacobi"`` uses one curvature scale for the
        flattened magnetization block and individual curvature scales
        for the three ramp parameters.
    """
    cg_maxiter: int = 10000
    cg_tol: float = 1e-9
    preconditioner: Literal["block_jacobi"] | None = None


@dataclasses.dataclass(frozen=True)
class RegConfig:
    """Regularization configuration for the MBIR solver.

    Parameters
    ----------
    lambda_exchange : float or Quantity["rad2"]
        Exchange-energy regularization weight.  A plain float is
        interpreted as having units of rad².  Default is ``0.0``
        (no regularization).
    """
    lambda_exchange: float | u.Quantity = 0.0


SolverConfig = Union[NewtonCGConfig]

_SOLVER_DEFAULTS = {
    "newton_cg": NewtonCGConfig,
}


def _resolve_solver_config(
    solver: str | SolverConfig,
    solver_config: SolverConfig | None = None,
) -> SolverConfig:
    """Resolve a solver string or config object (and optional override) to a SolverConfig.

    When *solver_config* is provided it takes precedence over *solver*.
    A string selects the default config for the named solver.
    """
    if solver_config is not None:
        return solver_config
    if isinstance(solver, str):
        solver_name = solver.lower()
        if solver_name not in _SOLVER_DEFAULTS:
            raise ValueError(
                f"Unknown solver {solver!r}. "
                f"Choose from {list(_SOLVER_DEFAULTS)}"
            )
        return _SOLVER_DEFAULTS[solver_name]()
    if isinstance(solver, NewtonCGConfig):
        return solver
    raise TypeError(
        f"solver must be a string or a SolverConfig instance, got {type(solver)}"
    )

_NEWTON_CG_PRECONDITIONERS = {None, "block_jacobi"}


def _validate_newton_cg_preconditioner(preconditioner: str | None) -> str | None:
    """Validate the requested experimental Newton-CG preconditioner."""
    if preconditioner not in _NEWTON_CG_PRECONDITIONERS:
        raise ValueError(
            f"Unknown Newton-CG preconditioner {preconditioner!r}. "
            f"Choose from {sorted(name for name in _NEWTON_CG_PRECONDITIONERS if name is not None)} "
            "or None."
        )
    return preconditioner


def _make_block_jacobi_preconditioner(
    x0_flat: jax.Array,
    matvec_hvp,
    magnetization_size: int,
):
    """Build a cheap block-diagonal inverse preconditioner in flat space."""
    dtype = x0_flat.dtype
    eps = jnp.sqrt(jnp.finfo(dtype).eps)
    total_size = x0_flat.size
    ramp_size = total_size - magnetization_size

    if ramp_size != 3:
        raise ValueError(
            "Newton-CG block Jacobi preconditioner expects three ramp coefficients, "
            f"got {ramp_size}."
        )

    mag_probe = jnp.concatenate([
        jnp.ones((magnetization_size,), dtype=dtype),
        jnp.zeros((ramp_size,), dtype=dtype),
    ])
    mag_curvature = jnp.vdot(mag_probe, matvec_hvp(mag_probe)).real / magnetization_size

    ramp_curvatures = []
    for ramp_index in range(ramp_size):
        basis = jnp.zeros_like(x0_flat)
        basis = basis.at[magnetization_size + ramp_index].set(1)
        ramp_curvatures.append(matvec_hvp(basis)[magnetization_size + ramp_index].real)

    diag = jnp.concatenate([
        jnp.full((magnetization_size,), mag_curvature, dtype=dtype),
        jnp.asarray(ramp_curvatures, dtype=dtype),
    ])
    diag = jnp.where(jnp.isfinite(diag) & (diag > eps), diag, jnp.ones_like(diag))

    def inverse_preconditioner(v):
        return v / diag

    return inverse_preconditioner


def _make_newton_cg_preconditioner(
    preconditioner: str | None,
    x0_flat: jax.Array,
    matvec_hvp,
    magnetization_size: int,
):
    """Create an experimental inverse preconditioner for JAX CG."""
    preconditioner = _validate_newton_cg_preconditioner(preconditioner)
    if preconditioner is None:
        return None
    if preconditioner == "block_jacobi":
        return _make_block_jacobi_preconditioner(
            x0_flat,
            matvec_hvp,
            magnetization_size,
        )
    raise AssertionError(f"Unhandled Newton-CG preconditioner: {preconditioner}")

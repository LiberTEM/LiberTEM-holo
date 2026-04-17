"""MBIR loss, Newton-CG solver, and user-facing reconstruction entry points.

Implements :func:`mbir_loss_2d`, the internal
:func:`_run_newton_cg_solver_2d`, and the user-facing
:func:`solve_mbir_2d`, :func:`reconstruct_2d`, and
:func:`reconstruct_2d_ensemble`.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import quaxed.numpy as qnp
import unxt as u
from jax.flatten_util import ravel_pytree

from .units import (
    RampCoeffs,
    _as_angle_quantity,
    _as_dimensionless_quantity,
    _as_length_quantity,
    _as_ramp_coeffs,
    _assert_quantity_compatible,
    _to_lambda_exchange,
    _validate_positive,
    make_quantity,
)
from .types import (
    NewtonCGConfig,
    RegConfig,
    SolverConfig,
    SolverResult,
    _assert_solver_result_units,
    _make_newton_cg_preconditioner,
    _resolve_solver_config,
)
from .regularization import exchange_loss_fn
from .kernel import build_rdfc_kernel
from .forward import apply_ramp, forward_model_single_rdfc_2d

def _normalize_reg_config(reg_config: RegConfig | dict[str, Any] | None) -> RegConfig:
    """Convert a dict or None to a RegConfig for backwards compatibility."""
    if reg_config is None:
        return RegConfig()
    if isinstance(reg_config, RegConfig):
        return reg_config
    if isinstance(reg_config, dict):
        return RegConfig(lambda_exchange=reg_config.get("lambda_exchange", 0.0))
    raise TypeError(
        f"reg_config must be a RegConfig, dict, or None, got {type(reg_config)}"
    )

def mbir_loss_2d(
    params: tuple[u.Quantity, RampCoeffs],
    mask: jax.Array,
    phase: u.Quantity,
    rdfc_kernel: dict[str, Any],
    pixel_size: u.Quantity,
    reg_config: RegConfig | dict[str, Any],
    reg_mask: jax.Array | None = None,
) -> u.Quantity:
    """Compute the MBIR loss for 2D projected magnetization.

    The total loss is the sum of a least-squares data-fidelity term
    and optional exchange-energy regularization.

    Parameters
    ----------
    params
        Tuple of ``(magnetization, ramp_coeffs)`` where
        *magnetization* has shape ``(N, M, 2)`` and *ramp_coeffs*
        has shape ``(3,)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization before the forward model.
    phase
        Observed phase image of shape ``(H, W)``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    pixel_size
        Pixel size in nanometres.
    reg_config
        Regularization configuration.  Pass a :class:`RegConfig` instance
        or a dict with key ``'lambda_exchange'`` (for backwards compat).
    reg_mask
        Optional regularization mask of shape ``(N, M)`` passed to
        :func:`exchange_loss_fn`.  Defaults to *mask* when not
        provided.

    Returns
    -------
    jax.Array
        Scalar loss value.
    """
    if reg_mask is None:
        reg_mask = mask
    magnetization, ramp_coeffs = params
    mask = jnp.asarray(mask)
    reg_mask = jnp.asarray(reg_mask)
    phase_q = _as_angle_quantity(phase)
    magnetization_q = _as_dimensionless_quantity(magnetization)
    ramp_coeffs_q = _as_ramp_coeffs(
        ramp_coeffs,
        dtype=magnetization_q.value.dtype,
    )
    pixel_size_q = _as_length_quantity(pixel_size)

    magnetization_q = qnp.stack([
        magnetization_q[..., 0] * mask,
        magnetization_q[..., 1] * mask,
    ], axis=-1)

    predictions = forward_model_single_rdfc_2d(
        magnetization_q,
        ramp_coeffs_q,
        rdfc_kernel,
        pixel_size_q,
    )

    residuals = predictions - phase_q
    loss = 0.5 * qnp.sum(residuals ** 2)

    rc = _normalize_reg_config(reg_config)
    lambda_exchange = _to_lambda_exchange(rc.lambda_exchange)

    loss += lambda_exchange * exchange_loss_fn(magnetization_q, reg_mask)
    _assert_quantity_compatible(loss, "rad2", "loss")

    return loss


def _run_newton_cg_solver_2d(
    phase: u.Quantity,
    init_mag: u.Quantity,
    mask: jax.Array,
    pixel_size: u.Quantity,
    reg_config: RegConfig | dict[str, Any] | None = None,
    rdfc_kernel: dict[str, Any] | None = None,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 10000,
    preconditioner: str | None = None,
    init_ramp_coeffs: RampCoeffs | None = None,
    reg_mask: jax.Array | None = None,
) -> tuple[tuple[u.Quantity, RampCoeffs], u.Quantity, bool]:
    """Minimize :func:`mbir_loss_2d` using a single Newton-CG solve.

    The MBIR objective is quadratic in the reconstruction
    parameters, so Newton-CG reduces to a single linear solve of
    ``H @ delta = -g``.  The solver accuracy is therefore fully
    controlled by the inner CG tolerance and iteration budget.

    Parameters
    ----------
    phase
        Observed phase image of shape ``(H, W)``.
    init_mag
        Initial magnetization of shape ``(N, M, 2)``.
    mask
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size
        Pixel size in nanometres.
    reg_config
        Regularization configuration dictionary (see
        :func:`mbir_loss_2d`), default ``{}``.
    rdfc_kernel
        Kernel dictionary as returned by :func:`build_rdfc_kernel`.
    cg_tol
        Tolerance for the CG solver, default 1e-8.
    cg_maxiter
        Maximum number of CG iterations for the Newton solve,
        default 10000.
    preconditioner
        Experimental inverse preconditioner for the CG solve.
    init_ramp_coeffs
        Initial ramp coefficients of shape ``(3,)``.  Defaults to
        zeros.
    reg_mask
        Optional regularization mask of shape ``(N, M)``.
        Defaults to *mask*.

    Returns
    -------
    (magnetization, ramp_coeffs) : tuple[Quantity[""], RampCoeffs]
        Optimized magnetization ``(N, M, 2)`` and typed ramp coefficients.
    loss_history : Quantity["rad2"]
        Length-1 array containing the loss after the Newton update.
    converged : bool
        Whether the CG solver converged within the iteration budget.
    """
    phase_q = _as_angle_quantity(phase)
    init_mag_q = _as_dimensionless_quantity(init_mag)
    pixel_size_q = _as_length_quantity(pixel_size)
    init_ramp_coeffs_q = _as_ramp_coeffs(
        init_ramp_coeffs,
        dtype=init_mag_q.value.dtype,
    )
    if reg_config is None:
        reg_config = RegConfig()
    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(phase_q.shape)
    kernel = cast(dict[str, Any], rdfc_kernel)

    x0_tree = (init_mag_q, init_ramp_coeffs_q)
    x0_flat, unravel = ravel_pytree(x0_tree)

    def objective_flat(x_flat):
        mag, ramp = unravel(x_flat)
        return mbir_loss_2d(
            (mag, ramp),
            mask,
            phase_q,
            kernel,
            pixel_size_q,
            reg_config,
            reg_mask=reg_mask,
        ).value

    loss_grad = jax.grad(objective_flat)

    grad_at_x0 = loss_grad(x0_flat)

    def matvec_hvp(v):
        return jax.jvp(loss_grad, (x0_flat,), (v,))[1]

    preconditioner_fn = _make_newton_cg_preconditioner(
        preconditioner,
        x0_flat,
        matvec_hvp,
        init_mag_q.value.size,
    )

    delta, cg_info = jax.scipy.sparse.linalg.cg(
        matvec_hvp,
        -grad_at_x0,
        tol=cg_tol,
        maxiter=cg_maxiter,
        M=preconditioner_fn,
    )
    # JAX's cg returns None for info; compute convergence from the residual.
    # The system is: H @ delta = b where b = -grad_at_x0.
    b = -grad_at_x0
    residual = matvec_hvp(delta) - b
    b_norm = jnp.linalg.norm(b)
    residual_norm = jnp.linalg.norm(residual)
    # CG convention: converged when ||r|| / ||b|| < tol (or b == 0).
    converged = (b_norm == 0) | (residual_norm / (b_norm + 1e-30) < cg_tol)
    final_flat = x0_flat + delta
    history = make_quantity(jnp.expand_dims(objective_flat(final_flat), axis=0), "rad2")

    final_mag, final_ramp = unravel(final_flat)

    return (final_mag, final_ramp), history, converged


def solve_mbir_2d(
    phase,
    init_mag,
    mask,
    pixel_size,
    solver: str | SolverConfig = "newton_cg",
    reg_config: RegConfig | dict[str, Any] | None = None,
    rdfc_kernel=None,
    init_ramp_coeffs=None,
    reg_mask=None,
):
    """
    Unified MBIR solver for 2D projected magnetization reconstruction.

    Parameters
    ----------
    phase : array_like
        Measured phase image in **radians**.
    init_mag : array_like
        Initial magnetization estimate, shape ``(N, M, 2)``.
    mask : array_like
        Binary mask of shape ``(N, M)`` applied to the
        magnetization.
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units.
    solver : str or SolverConfig, optional
        Solver selection. Pass ``"newton_cg"`` or a
        :class:`NewtonCGConfig` for full control. Default is ``"newton_cg"``.
    reg_config : RegConfig or dict, optional
        Regularization configuration.  Pass a :class:`RegConfig` or a dict
        (e.g. ``{"lambda_exchange": 1.0}``) for backwards compatibility.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
    init_ramp_coeffs : array_like, optional
        Initial ramp coefficients ``[offset, slope_y, slope_x]``
        in units of **[rad, rad/nm, rad/nm]**.
    reg_mask : array_like, optional
        Regularization mask of shape ``(N, M)``.  Defaults to *mask*.

    Returns
    -------
    SolverResult
        Named tuple with fields ``magnetization``, ``ramp_coeffs``, and
        ``loss_history``.

    """
    phase = _as_angle_quantity(phase)
    init_mag = _as_dimensionless_quantity(init_mag)
    pixel_size = _as_length_quantity(pixel_size)
    if init_ramp_coeffs is not None:
        init_ramp_coeffs = _as_ramp_coeffs(
            init_ramp_coeffs,
            dtype=init_mag.value.dtype,
        )
    if isinstance(solver, str):
        config = _resolve_solver_config(solver)
    elif isinstance(solver, NewtonCGConfig):
        config = solver
    else:
        raise TypeError(
            f"solver must be a string or a SolverConfig instance, got {type(solver)}"
        )

    shared = dict(
        phase=phase,
        init_mag=init_mag,
        mask=mask,
        pixel_size=pixel_size,
        reg_config=reg_config,
        rdfc_kernel=rdfc_kernel,
        init_ramp_coeffs=init_ramp_coeffs,
        reg_mask=reg_mask,
    )

    if isinstance(config, NewtonCGConfig):
        (mag, ramp), loss_history, converged_jax = _run_newton_cg_solver_2d(
            **shared,
            cg_tol=config.cg_tol,
            cg_maxiter=config.cg_maxiter,
            preconditioner=config.preconditioner,
        )
        converged = bool(converged_jax)
    else:
        raise AssertionError(f"Unhandled solver config type: {type(config)}")

    result = SolverResult(
        magnetization=mag,
        ramp_coeffs=cast(RampCoeffs, ramp),
        loss_history=loss_history,
        converged=converged,
    )
    _assert_solver_result_units(result)
    if not result.converged:
        warnings.warn(
            f"CG solver did not converge within {config.cg_maxiter} iterations "
            f"(tol={config.cg_tol}). Consider increasing cg_maxiter or "
            f"relaxing cg_tol in NewtonCGConfig.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def reconstruct_2d(
    phase,
    pixel_size,
    mask=None,
    lam=1e-3,
    solver: str | SolverConfig = "newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config: SolverConfig | None = None,
):
    """Convenience wrapper for 2D MBIR magnetization reconstruction.

    Provides a simple interface similar to pyramid's
    ``reconstruction_2d_from_phasemap``.  Builds the RDFC kernel,
    initial magnetization guess, and mask automatically.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(N, M)`` in **radians**.
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
    mask : array_like, optional
        Binary mask of shape ``(N, M)``.  Defaults to all ones.
    lam : Quantity["rad2"], optional
        Regularization weight (``lambda_exchange``), default ``Quantity(1e-3, "rad2")``.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``) or a :class:`SolverConfig` instance.
        Ignored when *solver_config* is provided.
        Default is ``"newton_cg"``.
    reg_mask : array_like, optional
        Separate regularization mask of shape ``(N, M)``.
        Defaults to *mask*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.
    solver_config : SolverConfig, optional
        Explicit solver configuration object.  When provided,
        the *solver* string argument is ignored.

    Returns
    -------
    SolverResult
        Named tuple with fields ``magnetization``, ``ramp_coeffs``,
        and ``loss_history``.
    """
    pixel_size = _as_length_quantity(pixel_size)
    lam = _to_lambda_exchange(lam)
    _validate_positive(pixel_size, "pixel_size")

    phase = _as_angle_quantity(phase)
    if mask is None:
        mask = jnp.ones(phase.shape, dtype=bool)
    else:
        mask = jnp.asarray(mask, dtype=bool)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    init_mag = make_quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")
    reg_config = RegConfig(lambda_exchange=lam)

    if solver_config is not None:
        solver = solver_config

    result = solve_mbir_2d(
        phase=phase,
        init_mag=init_mag,
        mask=mask,
        pixel_size=pixel_size,
        solver=solver,
        reg_config=reg_config,
        rdfc_kernel=rdfc_kernel,
        reg_mask=reg_mask,
    )
    _assert_solver_result_units(result)
    return result


def reconstruct_2d_ensemble(
    phase,
    masks,
    pixel_size,
    lam=1e-3,
    solver: str | SolverConfig = "newton_cg",
    reg_masks=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
) -> u.Quantity:
    """Batched MBIR reconstruction over an ensemble of bootstrap masks.

    Runs :func:`reconstruct_2d` for each mask in the ensemble using
    ``jax.vmap`` for efficient parallel execution on GPU.

    Parameters
    ----------
    phase : array_like
        Measured phase image of shape ``(H, W)`` in **radians**.
    masks : array_like
        Bootstrap mask ensemble of shape ``(N_boot, H, W)``.
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
    lam : float, optional
        Regularization weight (``lambda_exchange``), default 1e-3.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``) or a
        :class:`NewtonCGConfig` instance.
        Ignored when *solver_config* is provided.
        Default is ``"newton_cg"``.
    reg_masks : array_like, optional
        Separate regularization masks of shape ``(N_boot, H, W)``.
        Defaults to *masks*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel (``"disc"`` or
        ``"slab"``), default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
        Built automatically when not provided.
    solver_config : SolverConfig, optional
        Explicit solver configuration object.  When provided,
        the *solver* string argument is ignored.

    Returns
    -------
    Quantity["dimensionless"]
        Reconstructed magnetization ensemble of shape
        ``(N_boot, H, W, 2)``.

    """
    pixel_size = _as_length_quantity(pixel_size)
    _validate_positive(pixel_size, "pixel_size")

    phase = _as_angle_quantity(phase)
    masks = jnp.asarray(masks)
    if reg_masks is None:
        reg_masks = masks
    else:
        reg_masks = jnp.asarray(reg_masks)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    # Resolve solver config once (Python-level dispatch, outside vmap)
    config = _resolve_solver_config(solver, solver_config)

    init_mag = make_quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")
    reg_config = RegConfig(lambda_exchange=_to_lambda_exchange(lam))

    # Build a vmappable function for the chosen solver
    def _solve_single_newton(mask, reg_mask):
        (mag, _ramp), _loss, converged = _run_newton_cg_solver_2d(
            phase=phase,
            init_mag=init_mag,
            mask=mask,
            pixel_size=pixel_size,
            reg_config=reg_config,
            rdfc_kernel=rdfc_kernel,
            cg_tol=config.cg_tol,
            cg_maxiter=config.cg_maxiter,
            reg_mask=reg_mask,
        )
        return cast(u.Quantity, mag).value, converged

    solve_single = _solve_single_newton

    solve_batch = jax.jit(jax.vmap(solve_single, in_axes=(0, 0)))
    all_mag, all_converged = solve_batch(masks, reg_masks)

    converged_arr = np.asarray(all_converged)
    if not np.all(converged_arr):
        n_failed = int(np.sum(~converged_arr))
        warnings.warn(
            f"CG solver did not converge for {n_failed} of {len(converged_arr)} "
            f"ensemble member(s). Consider increasing cg_maxiter or "
            f"relaxing cg_tol in NewtonCGConfig.",
            RuntimeWarning,
            stacklevel=2,
        )

    return make_quantity(all_mag, "")


# Mapping from projection axis to (sum_axis, coeff_matrix, need_transpose).

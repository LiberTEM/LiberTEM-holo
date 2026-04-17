"""L-curve regularization sweep and related loss decomposition."""

from __future__ import annotations

import warnings
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import quaxed.numpy as qnp
import unxt as u

from .units import (
    RampCoeffs,
    _as_angle_quantity,
    _as_dimensionless_quantity,
    _as_length_quantity,
    _as_ramp_coeffs,
    _assert_quantity_compatible,
    _ramp_coeffs_from_array,
    _ramp_coeffs_to_array,
    _to_lambda_exchange,
    _validate_positive,
    make_quantity,
)
from .types import (
    LCurveResult,
    NewtonCGConfig,
    RegConfig,
    SolverConfig,
    _assert_lcurve_result_units,
    _resolve_solver_config,
)
from .regularization import exchange_loss_fn
from .kernel import build_rdfc_kernel
from .forward import forward_model_single_rdfc_2d
from .solver import _run_newton_cg_solver_2d, solve_mbir_2d

def decompose_loss(
    magnetization,
    ramp_coeffs,
    phase,
    mask,
    reg_mask,
    rdfc_kernel,
    pixel_size,
) -> tuple[u.Quantity, u.Quantity]:
    """Decompose the MBIR loss into data-fidelity and regularization terms.

    Evaluates the two components of the loss **without** the
    ``lambda_exchange`` multiplier on the regularization term, so
    that they can be compared on an L-curve plot.

    Parameters
    ----------
    magnetization : Quantity["dimensionless"]
        Reconstructed magnetization of shape ``(N, M, 2)``.
    ramp_coeffs : RampCoeffs
        Background ramp coefficients with ``unxt.Quantity`` fields.
    phase : Quantity["angle"]
        Observed phase image of shape ``(N, M)``.
    mask : array_like
        Binary mask of shape ``(N, M)`` applied to the
        magnetization before the forward model.
    reg_mask : array_like
        Regularization mask of shape ``(N, M)`` passed to
        :func:`exchange_loss_fn`.
    rdfc_kernel : dict
        Pre-built RDFC kernel from :func:`build_rdfc_kernel`.
    pixel_size : Quantity["length"]
        Pixel size converted to nanometres internally.

    Returns
    -------
    data_misfit : Quantity["rad2"]
        ``sum((predicted - observed)**2)`` — the squared-residual
        norm, matching Pyramid's ``chisq_m`` convention (no 1/2
        factor).
    exchange_norm : Quantity["dimensionless"]
        Unweighted exchange regularization norm (no lambda
        multiplier) computed with :func:`exchange_loss_fn`.
    """
    pixel_size = _as_length_quantity(pixel_size)
    magnetization = _as_dimensionless_quantity(magnetization)
    ramp_coeffs = _as_ramp_coeffs(
        ramp_coeffs,
        dtype=magnetization.value.dtype,
    )
    phase = _as_angle_quantity(phase)
    mask = jnp.asarray(mask)
    reg_mask = jnp.asarray(reg_mask)

    masked_mag = qnp.stack([
        magnetization[..., 0] * mask,
        magnetization[..., 1] * mask,
    ], axis=-1)

    predicted = forward_model_single_rdfc_2d(
        masked_mag, ramp_coeffs, rdfc_kernel, pixel_size,
    )
    residuals = predicted - phase
    data_misfit = qnp.sum(residuals ** 2)
    exchange_norm = exchange_loss_fn(masked_mag, reg_mask)

    _assert_quantity_compatible(cast(u.Quantity, data_misfit), "rad2", "data_misfit")
    _assert_quantity_compatible(cast(u.Quantity, exchange_norm), "", "exchange_norm")

    return data_misfit, exchange_norm

def kneedle_corner(data_misfits, reg_norms):
    """Find the L-curve corner using the Kneedle algorithm.

    Projects points in log-log space onto the line connecting the
    two endpoints and returns the index of the point with the
    largest perpendicular distance (the "elbow").

    Parameters
    ----------
    data_misfits : array_like
        1D array of data-fidelity values, one per lambda.
    reg_norms : array_like
        1D array of regularization norms (unweighted), one per
        lambda.

    Returns
    -------
    corner_index : int
        Index into the input arrays of the detected corner point.
        Returns ``-1`` when fewer than 3 points are provided.
    score : float
        Maximum normalized perpendicular distance.  Larger values
        indicate a more pronounced corner.
    """
    x = np.log10(np.asarray(reg_norms, dtype=np.float64))
    y = np.log10(np.asarray(data_misfits, dtype=np.float64))

    if len(x) < 3:
        return -1, 0.0

    order = np.argsort(x)
    x, y = x[order], y[order]

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    if x_range == 0 or y_range == 0:
        return -1, 0.0

    x_n = (x - x.min()) / (x_range + 1e-30)
    y_n = (y - y.min()) / (y_range + 1e-30)

    # Perpendicular distance from the line joining first and last point
    x1, y1 = x_n[0], y_n[0]
    x2, y2 = x_n[-1], y_n[-1]
    line_len = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-30
    d = np.abs(
        (y2 - y1) * x_n - (x2 - x1) * y_n + x2 * y1 - y2 * x1
    ) / line_len

    best_sorted = int(np.argmax(d))
    return int(order[best_sorted]), float(d[best_sorted])


def lcurve_sweep(
    phase,
    mask,
    pixel_size,
    lambdas,
    solver: str | SolverConfig = "newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
) -> LCurveResult:
    """Sequential L-curve sweep over regularization weights.

    Runs :func:`solve_mbir_2d` for each value in *lambdas*,
    collects the data-fidelity and regularization norms, and
    detects the L-curve corner via :func:`kneedle_corner`.

    Parameters
    ----------
    phase : Quantity["angle"]
        Measured phase image of shape ``(N, M)`` in **radians**.
    mask : array_like
        Binary mask of shape ``(N, M)``.
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
    lambdas : array_like or Quantity["rad2"]
        1D array of ``lambda_exchange`` values to sweep. Plain scalars
        are promoted to ``Quantity["rad2"]``.
    solver : str or SolverConfig, optional
        Solver selection string or config object.  Ignored when
        *solver_config* is provided.  Default ``"newton_cg"``.
    reg_mask : array_like, optional
        Regularization mask of shape ``(N, M)``.  Defaults to
        *mask*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel, default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel.  Built automatically when ``None``.
    solver_config : SolverConfig, optional
        Explicit solver configuration.
    Returns
    -------
    LCurveResult
        Named tuple with ``lambdas``, ``data_misfits``,
        ``reg_norms``, ``magnetizations``, ``ramp_coeffs``, and
        ``corner_index``.
    """
    phase = _as_angle_quantity(phase)
    pixel_size = _as_length_quantity(pixel_size)
    lambdas_q = _to_lambda_exchange(lambdas)
    lambdas = np.atleast_1d(np.asarray(lambdas_q.value, dtype=np.float64))

    _validate_positive(pixel_size, "pixel_size")

    mask = jnp.asarray(mask, dtype=bool)
    if reg_mask is None:
        reg_mask = mask
    else:
        reg_mask = jnp.asarray(reg_mask)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    actual_solver = solver_config if solver_config is not None else solver

    data_misfits = []
    reg_norms = []
    mag_list = []
    ramp_list = []

    init_mag = make_quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")

    for lam in lambdas:
        reg_config = RegConfig(lambda_exchange=make_quantity(lam, "rad2"))

        result = solve_mbir_2d(
            phase=phase,
            init_mag=init_mag,
            mask=mask,
            pixel_size=pixel_size,
            solver=actual_solver,
            reg_config=reg_config,
            rdfc_kernel=rdfc_kernel,
            reg_mask=reg_mask,
        )

        dm, rn = decompose_loss(
            result.magnetization, result.ramp_coeffs,
            phase, mask, reg_mask, rdfc_kernel, pixel_size,
        )
        data_misfits.append(float(dm.value))
        reg_norms.append(float(rn.value))
        mag_list.append(result.magnetization.value)
        ramp_list.append(jnp.array([
            result.ramp_coeffs.offset.value,
            result.ramp_coeffs.slope_y.value,
            result.ramp_coeffs.slope_x.value,
        ]))

    data_misfits = np.array(data_misfits)
    reg_norms = np.array(reg_norms)
    corner_idx, _ = kneedle_corner(data_misfits, reg_norms)

    all_mag = jnp.stack(mag_list)
    all_ramp = jnp.stack(ramp_list)

    result = LCurveResult(
        lambdas=make_quantity(lambdas, "rad2"),
        data_misfits=make_quantity(data_misfits, "rad2"),
        reg_norms=make_quantity(reg_norms, ""),
        magnetizations=make_quantity(all_mag, ""),
        ramp_coeffs=RampCoeffs(
            offset=make_quantity(all_ramp[:, 0], "rad"),
            slope_y=make_quantity(all_ramp[:, 1], "rad/nm"),
            slope_x=make_quantity(all_ramp[:, 2], "rad/nm"),
        ),
        corner_index=corner_idx,
    )
    _assert_lcurve_result_units(result)
    return result


def lcurve_sweep_vmap(
    phase,
    mask,
    pixel_size,
    lambdas,
    solver: str | SolverConfig = "newton_cg",
    reg_mask=None,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
) -> LCurveResult:
    """Parallel L-curve sweep using ``jax.vmap`` over lambda values.

    Runs all reconstructions in parallel (no warm-starting).
    This is faster on GPU when many lambda values are evaluated,
    but uses more memory than :func:`lcurve_sweep`.

    Parameters
    ----------
    phase : Quantity["angle"]
        Measured phase image of shape ``(N, M)`` in **radians**.
    mask : array_like
        Binary mask of shape ``(N, M)``.
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units. Must be positive.
    lambdas : array_like or Quantity["rad2"]
        1D array of ``lambda_exchange`` values to sweep. Plain scalars
        are promoted to ``Quantity["rad2"]``.
    solver : str or SolverConfig, optional
        Solver selection string or config object.  Ignored when
        *solver_config* is provided.  Default ``"newton_cg"``.
    reg_mask : array_like, optional
        Regularization mask of shape ``(N, M)``.  Defaults to
        *mask*.
    geometry : str, optional
        Voxel geometry for the RDFC kernel, default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Pre-built RDFC kernel.  Built automatically when ``None``.
    solver_config : SolverConfig, optional
        Explicit solver configuration.
    Returns
    -------
    LCurveResult
        Named tuple with ``lambdas``, ``data_misfits``,
        ``reg_norms``, ``magnetizations``, ``ramp_coeffs``, and
        ``corner_index``.

    """
    phase = _as_angle_quantity(phase)
    pixel_size = _as_length_quantity(pixel_size)
    lambdas_q = _to_lambda_exchange(lambdas)
    lambdas_np = np.atleast_1d(np.asarray(lambdas_q.value, dtype=np.float64))
    lambdas_jax = jnp.asarray(lambdas_np)
    _validate_positive(pixel_size, "pixel_size")

    mask = jnp.asarray(mask, dtype=bool)
    if reg_mask is None:
        reg_mask = mask
    else:
        reg_mask = jnp.asarray(reg_mask)

    if rdfc_kernel is None:
        rdfc_kernel = build_rdfc_kernel(
            phase.shape,
            geometry=geometry,
            prw_vec=prw_vec,
        )

    config = _resolve_solver_config(solver, solver_config)

    init_mag = make_quantity(jnp.zeros((*phase.shape, 2), dtype=jnp.float64), "")

    def _solve_for_lam_newton(lam):
        reg_config = RegConfig(lambda_exchange=make_quantity(lam, "rad2"))
        (mag, ramp), _loss, converged = _run_newton_cg_solver_2d(
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
        return cast(u.Quantity, mag).value, _ramp_coeffs_to_array(cast(RampCoeffs, ramp)), converged

    solve_for_lam = _solve_for_lam_newton

    solve_batch = jax.jit(jax.vmap(solve_for_lam))
    all_mag, all_ramp, all_converged = solve_batch(lambdas_jax)

    converged_arr = np.asarray(all_converged)
    if not np.all(converged_arr):
        failed = np.where(~converged_arr)[0]
        failed_lams = lambdas_np[failed]
        warnings.warn(
            f"CG solver did not converge for {len(failed)} of {len(lambdas_np)} "
            f"lambda value(s): {failed_lams.tolist()}. Consider increasing "
            f"cg_maxiter or relaxing cg_tol in NewtonCGConfig.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _decompose_single(mag, ramp):
        masked_mag = u.Quantity(
            jnp.stack([
                mag[..., 0] * mask,
                mag[..., 1] * mask,
            ], axis=-1),
            "",
        )
        predicted = forward_model_single_rdfc_2d(
            masked_mag, _ramp_coeffs_from_array(ramp), rdfc_kernel, pixel_size,
        )
        residuals = predicted - phase
        dm = cast(u.Quantity, qnp.sum(residuals ** 2)).value
        rn = cast(u.Quantity, exchange_loss_fn(masked_mag, reg_mask)).value
        return dm, rn

    decompose_batch = jax.jit(jax.vmap(_decompose_single))
    all_dm, all_rn = decompose_batch(all_mag, all_ramp)

    data_misfits = np.asarray(all_dm)
    reg_norms = np.asarray(all_rn)
    corner_idx, _ = kneedle_corner(data_misfits, reg_norms)

    result = LCurveResult(
        lambdas=make_quantity(lambdas_np, "rad2"),
        data_misfits=make_quantity(data_misfits, "rad2"),
        reg_norms=make_quantity(reg_norms, ""),
        magnetizations=make_quantity(all_mag, ""),
        ramp_coeffs=RampCoeffs(
            offset=make_quantity(all_ramp[:, 0], "rad"),
            slope_y=make_quantity(all_ramp[:, 1], "rad/nm"),
            slope_x=make_quantity(all_ramp[:, 2], "rad/nm"),
        ),
        corner_index=corner_idx,
    )
    _assert_lcurve_result_units(result)
    return result


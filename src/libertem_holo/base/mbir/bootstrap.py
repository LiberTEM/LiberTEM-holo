"""Bootstrap uncertainty estimation over threshold-defined support masks."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import numpy as np
import quaxed.numpy as qnp
import unxt as u

from .units import (
    _as_angle_quantity,
    _as_induction_quantity,
    _as_length_quantity,
    _as_dimensionless_quantity,
    _as_threshold_scalar,
    _validate_positive,
    make_quantity,
)
from .types import BootstrapThresholdResult, SolverConfig
from .solver import reconstruct_2d_ensemble
from .physical import to_local_induction, to_projected_induction_integral
from .units import B_REF


def _generate_threshold_draws_and_masks(
    mip_phase_q: u.Quantity,
    threshold_value: float,
    threshold_low_value: float,
    threshold_high_value: float,
    n_boot: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample threshold draws and build binary bootstrap masks.

    Returns
    -------
    threshold_draws : ndarray of shape (n_boot,)
    reference_mask : ndarray of shape (H, W)
    bootstrap_masks : ndarray of shape (n_boot, H, W)
    """
    rng = np.random.default_rng(rng_seed)
    threshold_draws = rng.uniform(
        low=threshold_low_value,
        high=threshold_high_value,
        size=n_boot,
    )

    mip_phase_rad = cast(u.Quantity, u.uconvert("rad", mip_phase_q))
    mip_abs = np.abs(np.asarray(mip_phase_rad.value))
    reference_mask = (mip_abs > threshold_value).astype(np.float64)
    bootstrap_masks = (
        mip_abs[None, ...] > threshold_draws[:, None, None]
    ).astype(np.float64)

    for draw_index in range(n_boot):
        if bootstrap_masks[draw_index].sum() == 0:
            bootstrap_masks[draw_index] = reference_mask

    return threshold_draws, reference_mask, bootstrap_masks


def _compute_percentile_statistics(
    bootstrap_mag: u.Quantity,
) -> tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """Compute mean magnetization, norm percentiles, and relative CI.

    Returns
    -------
    mean_magnetization, mean_norm, norm_low, norm_high, norm_ci95, relative_ci95
    """
    mean_magnetization = _as_dimensionless_quantity(qnp.mean(bootstrap_mag, axis=0))
    mean_norm = _as_dimensionless_quantity(
        qnp.sqrt(qnp.sum(mean_magnetization ** 2, axis=-1))
    )

    norm_samples = _as_dimensionless_quantity(
        qnp.sqrt(qnp.sum(bootstrap_mag ** 2, axis=-1))
    )
    norm_low = make_quantity(
        np.percentile(np.asarray(norm_samples.value), 2.5, axis=0),
        str(norm_samples.unit),
    )
    norm_high = make_quantity(
        np.percentile(np.asarray(norm_samples.value), 97.5, axis=0),
        str(norm_samples.unit),
    )
    norm_ci95 = norm_high - norm_low
    relative_ci95 = _as_dimensionless_quantity(
        norm_ci95 / (mean_norm + make_quantity(1e-12, ""))
    )
    return mean_magnetization, mean_norm, norm_low, norm_high, norm_ci95, relative_ci95


def _compute_local_induction_statistics(
    bootstrap_mag: u.Quantity,
    bootstrap_masks: np.ndarray,
    pixel_size: u.Quantity,
    thickness: u.Quantity,
    reference_induction: u.Quantity,
    min_effective_thickness: u.Quantity | None,
    invalid_to_nan: bool,
) -> tuple[
    u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, np.ndarray,
]:
    """Compute per-draw mean local induction inside each bootstrap mask.

    Returns
    -------
    samples, mean, low, high, ci95, roi_pixels
    """
    n_boot = bootstrap_masks.shape[0]
    thickness_q = _as_length_quantity(thickness, name="thickness")
    reference_induction_q = _as_induction_quantity(reference_induction)
    draw_means = np.empty(n_boot, dtype=np.float64)
    draw_pixels = np.zeros(n_boot, dtype=np.int64)
    bootstrap_mag_values = np.asarray(bootstrap_mag.value)
    bootstrap_mag_unit = str(bootstrap_mag.unit)

    for draw_index in range(n_boot):
        mag_draw = make_quantity(bootstrap_mag_values[draw_index], bootstrap_mag_unit)
        projected_draw = to_projected_induction_integral(
            mag_draw,
            pixel_size,
            reference_induction=reference_induction_q,
        )
        local_draw = to_local_induction(
            projected_draw,
            thickness_q,
            min_effective_thickness=min_effective_thickness,
            invalid_to_nan=invalid_to_nan,
        )
        local_draw_norm = np.linalg.norm(np.asarray(local_draw.value), axis=-1)
        roi_values = local_draw_norm[np.asarray(bootstrap_masks[draw_index], dtype=bool)]
        finite_roi_values = roi_values[np.isfinite(roi_values)]
        draw_pixels[draw_index] = finite_roi_values.size
        draw_means[draw_index] = (
            float(finite_roi_values.mean()) if finite_roi_values.size > 0 else np.nan
        )

    samples = make_quantity(draw_means, "T")
    mean = make_quantity(np.nanmean(draw_means), "T")
    low = make_quantity(np.nanpercentile(draw_means, 2.5), "T")
    high = make_quantity(np.nanpercentile(draw_means, 97.5), "T")
    ci95 = high - low
    return samples, mean, low, high, ci95, draw_pixels

def bootstrap_threshold_uncertainty_2d(
    phase,
    mip_phase,
    threshold,
    pixel_size,
    lam=1e-3,
    solver: str | SolverConfig = "newton_cg",
    n_boot=50,
    threshold_low=None,
    threshold_high=None,
    rng_seed=0,
    geometry="disc",
    prw_vec=None,
    rdfc_kernel=None,
    solver_config=None,
    thickness: u.Quantity | None = None,
    reference_induction: u.Quantity = B_REF,
    min_effective_thickness: u.Quantity | None = None,
    invalid_to_nan: bool = False,
) -> BootstrapThresholdResult:
    """Bootstrap a thresholded mask ensemble and summarize the uncertainty.

    Threshold draws are sampled uniformly from ``[threshold_low, threshold_high]``.
    Each draw defines a binary mask from ``abs(mip_phase) > threshold_draw``.
    The masks are passed through :func:`reconstruct_2d_ensemble` and the
    resulting magnetization ensemble is summarized with percentile maps for
    ``|M|``.

    Parameters
    ----------
    phase : Quantity["angle"]
        Observed phase image of shape ``(H, W)`` in **radians**.
    mip_phase : Quantity["angle"]
        MIP phase image used for thresholding, shape ``(H, W)``.
    threshold : float
        Central threshold value around which the bootstrap draws are sampled.
        This is a plain scalar applied to ``abs(mip_phase.value)`` after the
        MIP phase has been converted to radians.
    pixel_size : Quantity["length"]
        Pixel size as a ``unxt.Quantity`` with length units.
    lam : float or Quantity["rad2"], optional
        Regularization weight (``lambda_exchange``). Plain scalars are
        promoted to ``Quantity["rad2"]``. Default ``1e-3``.
    solver : str or SolverConfig, optional
        Solver selection string (``"newton_cg"``) or a
        :class:`NewtonCGConfig` instance. Ignored when
        *solver_config* is provided. Default ``"newton_cg"``.
    n_boot : int, optional
        Number of threshold draws, default 50.
    threshold_low : float, optional
        Lower bound for the threshold draws. Defaults to ``threshold - 0.25``.
    threshold_high : float, optional
        Upper bound for the threshold draws. Defaults to ``threshold + 0.25``.
    rng_seed : int, optional
        Seed for the pseudo-random number generator, default 0.
    geometry : str, optional
        Voxel geometry for the RDFC kernel, default ``"disc"``.
    prw_vec : array_like, optional
        Projected reference wave vector ``(v, u)``.
    rdfc_kernel : dict, optional
        Optional pre-built RDFC kernel dictionary for advanced or
        performance-sensitive reuse. Built automatically when ``None``.
    solver_config : SolverConfig, optional
        Explicit solver configuration object.
    thickness : Quantity["length"], optional
        Physical thickness map or scalar thickness used to convert the
        reconstructed projected induction to local induction. When provided,
        the result also includes per-draw mean ``|B_local|`` values inside
        each draw's bootstrap mask and their 95% bootstrap interval.
    reference_induction : Quantity["magnetic induction"], optional
        Physical induction scale corresponding to unit normalized
        magnetization. Default is the module-level 1 T reference.
    min_effective_thickness : Quantity["length"], optional
        Positive lower bound passed to :func:`to_local_induction` to stabilize
        the local-induction conversion.
    invalid_to_nan : bool, optional
        Forwarded to :func:`to_local_induction`. When ``True``, pixels below
        ``min_effective_thickness`` are excluded from the per-draw mean.

    Returns
    -------
    BootstrapThresholdResult
        Summary object containing the threshold draws, reconstructed
        magnetizations, 2.5th and 97.5th percentile maps, their 95% width,
        the relative 95% width, the mask inclusion frequency, and optionally
        the mean local-induction magnitude inside each draw's segmented
        object plus its 95% bootstrap interval.
    """
    phase = _as_angle_quantity(phase)
    mip_phase = _as_angle_quantity(mip_phase)
    pixel_size = _as_length_quantity(pixel_size)
    threshold_value = _as_threshold_scalar(threshold, "threshold")
    _validate_positive(pixel_size, "pixel_size")
    if phase.shape != mip_phase.shape:
        raise ValueError(
            f"phase and mip_phase must have the same shape; got {phase.shape} and {mip_phase.shape}."
        )

    if threshold_low is None:
        threshold_low_value = threshold_value - 0.25
    else:
        threshold_low_value = _as_threshold_scalar(threshold_low, "threshold_low")
    if threshold_high is None:
        threshold_high_value = threshold_value + 0.25
    else:
        threshold_high_value = _as_threshold_scalar(threshold_high, "threshold_high")

    if threshold_high_value <= threshold_low_value:
        raise ValueError(
            "threshold_high must be greater than threshold_low; "
            f"got {threshold_high_value} and {threshold_low_value}."
        )

    # --- Threshold draws and masks ---
    threshold_draws, _reference_mask, bootstrap_masks = _generate_threshold_draws_and_masks(
        mip_phase, threshold_value, threshold_low_value, threshold_high_value,
        n_boot, rng_seed,
    )

    # --- Ensemble reconstruction ---
    bootstrap_mag = reconstruct_2d_ensemble(
        phase=phase,
        masks=bootstrap_masks,
        pixel_size=pixel_size,
        lam=lam,
        solver=solver,
        reg_masks=bootstrap_masks,
        geometry=geometry,
        prw_vec=prw_vec,
        rdfc_kernel=rdfc_kernel,
        solver_config=solver_config,
    )
    bootstrap_mag = _as_dimensionless_quantity(bootstrap_mag)

    # --- Percentile statistics ---
    (
        mean_magnetization, mean_norm,
        norm_low, norm_high, norm_ci95, relative_ci95,
    ) = _compute_percentile_statistics(bootstrap_mag)

    mask_frequency = bootstrap_masks.mean(axis=0)

    # --- Optional local-induction statistics ---
    local_induction_mean_samples = None
    local_induction_mean = None
    local_induction_mean_low = None
    local_induction_mean_high = None
    local_induction_mean_ci95 = None
    local_induction_roi_pixels = None

    if thickness is not None:
        (
            local_induction_mean_samples,
            local_induction_mean,
            local_induction_mean_low,
            local_induction_mean_high,
            local_induction_mean_ci95,
            local_induction_roi_pixels,
        ) = _compute_local_induction_statistics(
            bootstrap_mag, bootstrap_masks, pixel_size, thickness,
            reference_induction, min_effective_thickness, invalid_to_nan,
        )

    return BootstrapThresholdResult(
        threshold=threshold_value,
        threshold_low=threshold_low_value,
        threshold_high=threshold_high_value,
        threshold_draws=threshold_draws,
        magnetizations=bootstrap_mag,
        mean_magnetization=mean_magnetization,
        mean_norm=mean_norm,
        norm_low=norm_low,
        norm_high=norm_high,
        norm_ci95=norm_ci95,
        relative_ci95=relative_ci95,
        mask_frequency=mask_frequency,
        local_induction_mean_samples=local_induction_mean_samples,
        local_induction_mean=local_induction_mean,
        local_induction_mean_low=local_induction_mean_low,
        local_induction_mean_high=local_induction_mean_high,
        local_induction_mean_ci95=local_induction_mean_ci95,
        local_induction_roi_pixels=local_induction_roi_pixels,
    )


"""Matplotlib plotting helpers for MBIR results.

These helpers import :mod:`matplotlib` lazily inside each function so
the core MBIR modules do not pull matplotlib as an import-time dependency.
"""

from __future__ import annotations

import numpy as np
import unxt as u

from .units import _as_dimensionless_quantity, _as_length_quantity
from .physical import (
    to_local_induction,
    to_local_magnetization,
    to_projected_induction_integral,
    to_projected_magnetization_integral,
)
from .types import BootstrapThresholdResult

def plot_bootstrap_mask_summary(
    result: BootstrapThresholdResult,
    *,
    ax=None,
    stable_frequency: float = 0.5,
    cmap: str = "gray",
):
    """Plot the aggregate bootstrap support mask frequency.

    This is a compact view of how often each pixel was included in the
    threshold-defined support across all bootstrap draws. The red contour
    marks the default ``stable_frequency`` boundary.

    Parameters
    ----------
    result
        Bootstrap summary returned by :func:`bootstrap_threshold_uncertainty_2d`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. When omitted, a new figure and axes are created.
    stable_frequency : float, optional
        Contour level used to mark the stable-support boundary. Must lie in
        the closed interval ``[0, 1]``. Default is ``0.5``.
    cmap : str, optional
        Matplotlib colormap for the inclusion-frequency image.

    Returns
    -------
    fig, ax, info
        Figure, axes, and a dictionary with the rendered mask-frequency array,
        stable-support mask, draw count, and threshold metadata.
    """
    import matplotlib.pyplot as plt

    if not 0.0 <= stable_frequency <= 1.0:
        raise ValueError(
            f"stable_frequency must lie in [0, 1], got {stable_frequency}."
        )

    mask_frequency = np.asarray(result.mask_frequency)
    if mask_frequency.ndim != 2:
        raise ValueError(
            "result.mask_frequency must be a 2D image; "
            f"got shape {mask_frequency.shape}."
        )

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    else:
        fig = ax.figure

    image = ax.imshow(mask_frequency, cmap=cmap, origin="lower", vmin=0, vmax=1)
    stable_support = mask_frequency >= stable_frequency
    if np.any(mask_frequency > 0) and np.any(mask_frequency < 1):
        ax.contour(
            mask_frequency,
            levels=[stable_frequency],
            colors="tab:red",
            linewidths=1.0,
        )

    n_draws = len(result.threshold_draws)
    ax.set_title(
        "Bootstrap mask inclusion frequency\n"
        f"{n_draws} draws in [{result.threshold_low:.3f}, {result.threshold_high:.3f}]",
        fontsize=10,
    )
    ax.set_xlabel(
        f"Stable support at frequency >= {stable_frequency:.2f}: "
        f"{int(np.count_nonzero(stable_support))} pixels"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, label="Inclusion frequency")

    info = {
        "mask_frequency": mask_frequency,
        "stable_support": stable_support,
        "stable_frequency": stable_frequency,
        "n_draws": n_draws,
        "threshold": result.threshold,
        "threshold_low": result.threshold_low,
        "threshold_high": result.threshold_high,
    }
    return fig, ax, info


def _compute_physical_bootstrap_data(
    result: BootstrapThresholdResult,
    magnetization,
    pixel_size,
    thickness,
    *,
    support_mask=None,
    min_effective_thickness: u.Quantity | None = None,
    invalid_to_nan: bool = False,
    max_relative_ci_for_display: float = 1.0,
    min_mask_frequency_for_display: float = 0.5,
) -> dict:
    """Compute physical-unit maps and certainty-weighting from bootstrap results.

    This is the pure-computation half of
    :func:`plot_physical_bootstrap_uncertainty`.  It converts the
    reconstructed magnetization to physical units, computes certainty-alpha
    maps, and returns all derived arrays in a single dictionary.

    Parameters
    ----------
    result
        Bootstrap summary returned by :func:`bootstrap_threshold_uncertainty_2d`.
    magnetization
        Final reconstructed magnetization field of shape ``(H, W, 2)``.
    pixel_size : Quantity["length"]
        Pixel size used for the physical conversion.
    thickness : Quantity["length"]
        Thickness map or scalar thickness for the local-unit conversion.
    support_mask : array_like, optional
        Boolean mask that defines the displayed reconstruction support.
        Defaults to all pixels.
    min_effective_thickness : Quantity["length"], optional
        Lower bound passed to :func:`to_local_induction` /
        :func:`to_local_magnetization`.
    invalid_to_nan : bool, optional
        Forwarded to the local conversion helpers.
    max_relative_ci_for_display : float, optional
        Relative 95% CI width where display certainty reaches zero.
    min_mask_frequency_for_display : float, optional
        Minimum threshold-inclusion frequency for the ``reliable_support``
        mask.

    Returns
    -------
    dict
        Keys include ``projected_induction_integral``,
        ``projected_magnetization_integral``, ``local_induction``,
        ``local_magnetization``, various masked/display arrays,
        ``certainty_alpha``, ``reliable_support``, and
        certainty-weighted summary scalars.
    """
    def as_array(value):
        if isinstance(value, u.Quantity):
            return np.asarray(value.value)
        return np.asarray(value)

    magnetization = _as_dimensionless_quantity(magnetization)
    pixel_size = _as_length_quantity(pixel_size)
    thickness = _as_length_quantity(thickness, name="thickness")

    image_shape = magnetization.shape[:-1]
    if support_mask is None:
        support_mask_arr = np.ones(image_shape, dtype=bool)
    else:
        support_mask_arr = np.asarray(support_mask, dtype=bool)
        if support_mask_arr.shape != image_shape:
            raise ValueError(
                "support_mask must match the spatial shape of magnetization; "
                f"got {support_mask_arr.shape} and {image_shape}."
            )

    relative_ci_map = as_array(result.relative_ci95)
    mask_frequency = as_array(result.mask_frequency)
    if relative_ci_map.shape != image_shape:
        raise ValueError(
            "result.relative_ci95 must match the spatial shape of magnetization; "
            f"got {relative_ci_map.shape} and {image_shape}."
        )
    if mask_frequency.shape != image_shape:
        raise ValueError(
            "result.mask_frequency must match the spatial shape of magnetization; "
            f"got {mask_frequency.shape} and {image_shape}."
        )

    projected_induction_integral = to_projected_induction_integral(
        magnetization,
        pixel_size,
    )
    projected_magnetization_integral = to_projected_magnetization_integral(
        magnetization,
        pixel_size,
    )
    local_induction = to_local_induction(
        projected_induction_integral,
        thickness,
        min_effective_thickness=min_effective_thickness,
        invalid_to_nan=invalid_to_nan,
    )
    local_magnetization = to_local_magnetization(
        projected_magnetization_integral,
        thickness,
        min_effective_thickness=min_effective_thickness,
        invalid_to_nan=invalid_to_nan,
    )

    projected_induction_integral_norm = np.linalg.norm(
        as_array(projected_induction_integral),
        axis=-1,
    )
    projected_magnetization_integral_norm = np.linalg.norm(
        as_array(projected_magnetization_integral),
        axis=-1,
    )
    local_induction_norm = np.linalg.norm(as_array(local_induction), axis=-1)
    local_magnetization_norm = np.linalg.norm(as_array(local_magnetization), axis=-1)

    masked_projected_induction = np.where(
        support_mask_arr,
        projected_induction_integral_norm,
        np.nan,
    )
    masked_projected_magnetization = np.where(
        support_mask_arr,
        projected_magnetization_integral_norm,
        np.nan,
    )

    certainty_alpha = np.clip(
        1.0 - relative_ci_map / max_relative_ci_for_display,
        0.0,
        1.0,
    )
    certainty_alpha *= mask_frequency
    certainty_alpha = np.where(support_mask_arr, certainty_alpha, 0.0)

    display_local_induction = np.where(
        support_mask_arr & np.isfinite(local_induction_norm),
        local_induction_norm * certainty_alpha,
        np.nan,
    )
    display_local_magnetization = np.where(
        support_mask_arr & np.isfinite(local_magnetization_norm),
        local_magnetization_norm * certainty_alpha,
        np.nan,
    )

    reliable_support = (
        support_mask_arr
        & np.isfinite(local_induction_norm)
        & np.isfinite(local_magnetization_norm)
        & (relative_ci_map <= max_relative_ci_for_display)
        & (mask_frequency >= min_mask_frequency_for_display)
    )
    masked_relative_ci_percent = np.where(
        support_mask_arr,
        100.0 * relative_ci_map,
        np.nan,
    )
    masked_certainty = np.where(support_mask_arr, certainty_alpha, np.nan)

    weighted_support = (
        support_mask_arr
        & np.isfinite(local_induction_norm)
        & np.isfinite(local_magnetization_norm)
    )
    certainty_weights = np.where(weighted_support, certainty_alpha, 0.0)
    weight_sum = float(np.sum(certainty_weights))
    if weight_sum > 0:
        normalized_weights = certainty_weights / weight_sum
        certainty_weighted_mean_local_induction = float(
            np.sum(
                normalized_weights
                * np.where(weighted_support, local_induction_norm, 0.0)
            )
        )
        certainty_weighted_mean_local_magnetization = float(
            np.sum(
                normalized_weights
                * np.where(weighted_support, local_magnetization_norm, 0.0)
            )
        )
        certainty_weighted_mean_thickness = float(
            np.sum(
                normalized_weights
                * np.where(weighted_support, as_array(thickness), 0.0)
            )
        )
    else:
        normalized_weights = np.zeros_like(certainty_weights)
        certainty_weighted_mean_local_induction = np.nan
        certainty_weighted_mean_local_magnetization = np.nan
        certainty_weighted_mean_thickness = np.nan

    mean_certainty_inside_support = (
        float(np.mean(certainty_alpha[support_mask_arr]))
        if np.any(support_mask_arr)
        else np.nan
    )

    return {
        "projected_induction_integral": projected_induction_integral,
        "projected_magnetization_integral": projected_magnetization_integral,
        "local_induction": local_induction,
        "local_magnetization": local_magnetization,
        "masked_projected_induction": masked_projected_induction,
        "masked_projected_magnetization": masked_projected_magnetization,
        "display_local_induction": display_local_induction,
        "display_local_magnetization": display_local_magnetization,
        "masked_relative_ci_percent": masked_relative_ci_percent,
        "masked_certainty": masked_certainty,
        "certainty_alpha": certainty_alpha,
        "reliable_support": reliable_support,
        "normalized_weights": normalized_weights,
        "certainty_weighted_mean_local_induction": certainty_weighted_mean_local_induction,
        "certainty_weighted_mean_local_magnetization": certainty_weighted_mean_local_magnetization,
        "certainty_weighted_mean_thickness": certainty_weighted_mean_thickness,
        "mean_certainty_inside_support": mean_certainty_inside_support,
        "equivalent_fully_certain_pixels": weight_sum,
        "max_relative_ci_for_display": max_relative_ci_for_display,
        "min_mask_frequency_for_display": min_mask_frequency_for_display,
    }


def plot_physical_bootstrap_uncertainty(
    result: BootstrapThresholdResult,
    magnetization,
    pixel_size,
    thickness,
    *,
    support_mask=None,
    min_effective_thickness: u.Quantity | None = None,
    invalid_to_nan: bool = False,
    max_relative_ci_for_display: float = 1.0,
    min_mask_frequency_for_display: float = 0.5,
):
    """Plot physical-unit maps weighted by bootstrap stability.

    Delegates to :func:`_compute_physical_bootstrap_data` for the
    computation and renders a 2×3 panel figure from the returned data.

    Parameters
    ----------
    result
        Bootstrap summary returned by :func:`bootstrap_threshold_uncertainty_2d`.
    magnetization
        Final reconstructed magnetization field of shape ``(H, W, 2)``.
    pixel_size : Quantity["length"]
        Pixel size used for the physical conversion.
    thickness : Quantity["length"]
        Thickness map or scalar thickness for the local-unit conversion.
    support_mask : array_like, optional
        Boolean mask that defines the displayed reconstruction support.
        Defaults to all pixels.
    min_effective_thickness : Quantity["length"], optional
        Lower bound passed to :func:`to_local_induction` and
        :func:`to_local_magnetization`.
    invalid_to_nan : bool, optional
        Forwarded to the local conversion helpers.
    max_relative_ci_for_display : float, optional
        Relative 95% CI width where display certainty reaches zero.
    min_mask_frequency_for_display : float, optional
        Minimum threshold-inclusion frequency for the ``reliable_support``
        summary mask returned in the info dictionary.

    Returns
    -------
    fig, axs, info
        Matplotlib figure and axes plus a dictionary containing the derived
        physical quantities, display masks, and summary scalars.
    """
    import matplotlib.pyplot as plt

    info = _compute_physical_bootstrap_data(
        result, magnetization, pixel_size, thickness,
        support_mask=support_mask,
        min_effective_thickness=min_effective_thickness,
        invalid_to_nan=invalid_to_nan,
        max_relative_ci_for_display=max_relative_ci_for_display,
        min_mask_frequency_for_display=min_mask_frequency_for_display,
    )

    projected_induction_integral = info["projected_induction_integral"]
    projected_magnetization_integral = info["projected_magnetization_integral"]
    local_induction = info["local_induction"]
    local_magnetization = info["local_magnetization"]

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    im = axs[0, 0].imshow(info["masked_projected_induction"], cmap="magma", origin="lower")
    axs[0, 0].set_title(rf"Projected $|B|$ integral ({projected_induction_integral.unit})")
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046, label=projected_induction_integral.unit)

    im = axs[0, 1].imshow(info["masked_projected_magnetization"], cmap="cividis", origin="lower")
    axs[0, 1].set_title(rf"Projected $|M|$ integral ({projected_magnetization_integral.unit})")
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046, label=projected_magnetization_integral.unit)

    im = axs[0, 2].imshow(info["masked_relative_ci_percent"], cmap="magma_r", origin="lower")
    axs[0, 2].set_title("Relative 95% CI width of |M| (%)")
    plt.colorbar(im, ax=axs[0, 2], fraction=0.046, label="%")

    im = axs[1, 0].imshow(info["display_local_induction"], cmap="magma", origin="lower")
    axs[1, 0].set_title(rf"Certainty-weighted local $|B|$ ({local_induction.unit})")
    plt.colorbar(im, ax=axs[1, 0], fraction=0.046, label=local_induction.unit)

    im = axs[1, 1].imshow(info["display_local_magnetization"], cmap="cividis", origin="lower")
    axs[1, 1].set_title(rf"Certainty-weighted local $|M|$ ({local_magnetization.unit})")
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046, label=local_magnetization.unit)

    im = axs[1, 2].imshow(info["masked_certainty"], cmap="gray", origin="lower", vmin=0, vmax=1)
    axs[1, 2].set_title("Display certainty / alpha")
    plt.colorbar(im, ax=axs[1, 2], fraction=0.046, label="alpha")

    fig.suptitle(
        "Physical-unit maps with bootstrap uncertainty used to suppress unstable edge pixels",
        fontsize=13,
    )

    return fig, axs, info


def plot_lcurve(
    lcurve_result,
    pyramid_style=False,
    ax=None,
    cmap="nipy_spectral",
    colorbar=True,
    **kwargs,
):
    """
    Plot an L-curve from an LCurveResult, with optional Pyramid-style axes.

    Parameters
    ----------
    lcurve_result : LCurveResult
        Result from lcurve_sweep or lcurve_sweep_vmap.
    pyramid_style : bool, optional
        If True, plot y = reg_norm / lambda vs x = data_misfit (Pyramid style).
        If False (default), plot y = data_misfit vs x = reg_norm (standard style).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.
    cmap : str, optional
        Colormap for lambda values.
    colorbar : bool, optional
        Whether to show a colorbar. Default True.
    **kwargs :
        Additional arguments passed to scatter/plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if isinstance(lcurve_result.lambdas, u.Quantity):
        lambdas = np.asarray(lcurve_result.lambdas.value)
    else:
        lambdas = np.asarray(lcurve_result.lambdas)

    if isinstance(lcurve_result.reg_norms, u.Quantity):
        reg_norms = np.asarray(lcurve_result.reg_norms.value)
    else:
        reg_norms = np.asarray(lcurve_result.reg_norms)

    if isinstance(lcurve_result.data_misfits, u.Quantity):
        data_misfits = np.asarray(lcurve_result.data_misfits.value)
    else:
        data_misfits = np.asarray(lcurve_result.data_misfits)

    if pyramid_style:
        x = data_misfits
        y = reg_norms
        xlabel = r"$\Vert\mathbf{F}(\mathbf{x})-\mathbf{y}\Vert^2$"
        ylabel = r"$\Vert\mathbf{D}\mathbf{x}\Vert^2$"
    else:
        x = reg_norms
        y = data_misfits
        xlabel = "Regularisation norm (exchange)"
        ylabel = "Data misfit"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x, y, c=lambdas, cmap=cmap, norm=LogNorm(), s=80, zorder=2, **kwargs)
    ax.plot(x, y, "k-", lw=1.5, zorder=1)
    if hasattr(lcurve_result, "corner_index") and lcurve_result.corner_index >= 0:
        ax.plot(
            x[lcurve_result.corner_index],
            y[lcurve_result.corner_index],
            "r*", ms=18, zorder=3,
            label=f"corner λ={lambdas[lcurve_result.corner_index]:.2e}",
        )
        ax.legend(frameon=False, fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title("MBIR L-curve" + (" (Pyramid style)" if pyramid_style else ""))
    ax.grid(alpha=0.25)
    if colorbar:
        plt.colorbar(sc, ax=ax, label=r"$\lambda$")
    return ax

from __future__ import annotations

import numpy as np


def plot_loss_history(loss_history, *, ax=None, label: str | None = None):
    import matplotlib.pyplot as plt

    history = np.asarray(loss_history, dtype=float)
    if history.ndim != 1:
        raise ValueError(f"loss_history must be 1D, got {history.shape}.")

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    else:
        fig = ax.figure

    x = np.arange(1, history.size + 1)
    ax.plot(x, history, marker="o", linewidth=1.5, markersize=3, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Regime A loss history")
    if label is not None:
        ax.legend()

    return fig, ax, {"iterations": x, "loss_history": history}


def plot_depth_profile(m_recon, m_true, yx, *, component: int = 0, ax=None):
    import matplotlib.pyplot as plt

    m_recon_arr = np.asarray(m_recon)
    m_true_arr = np.asarray(m_true)
    y, x = map(int, yx)
    recon_profile = m_recon_arr[:, y, x, component]
    true_profile = m_true_arr[:, y, x, component]
    z = np.arange(recon_profile.shape[0])

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    else:
        fig = ax.figure

    ax.plot(z, true_profile, label="truth", linewidth=2.0)
    ax.plot(z, recon_profile, label="reconstruction", linewidth=1.5, linestyle="--")
    ax.set_xlabel("z index")
    ax.set_ylabel(f"m[..., {component}]")
    ax.set_title(f"Depth profile at (y={y}, x={x})")
    ax.legend()

    return fig, ax, {
        "z": z,
        "true_profile": true_profile,
        "recon_profile": recon_profile,
        "yx": (y, x),
        "component": component,
    }


def plot_m_slices(m_recon, m_true, *, z_index: int | None = None, component: int = 2, axes=None):
    import matplotlib.pyplot as plt

    m_recon_arr = np.asarray(m_recon)
    m_true_arr = np.asarray(m_true)
    if m_recon_arr.shape != m_true_arr.shape:
        raise ValueError(
            f"m_recon and m_true must match, got {m_recon_arr.shape} and {m_true_arr.shape}."
        )

    if z_index is None:
        z_index = m_recon_arr.shape[0] // 2

    recon_slice = m_recon_arr[z_index, ..., component]
    true_slice = m_true_arr[z_index, ..., component]
    vmax = max(np.max(np.abs(recon_slice)), np.max(np.abs(true_slice)), 1e-12)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)
    else:
        fig = axes[0].figure

    im0 = axes[0].imshow(true_slice, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"Truth z={z_index}, c={component}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(recon_slice, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Recon z={z_index}, c={component}")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.colorbar(im1, ax=list(axes), fraction=0.046, label="Magnetization")

    return fig, axes, {
        "z_index": z_index,
        "component": component,
        "true_slice": true_slice,
        "recon_slice": recon_slice,
    }


def plot_loss_landscape_2d(
    x_values,
    y_values,
    loss_grid,
    *,
    x_label: str = "x",
    y_label: str = "y",
    ax=None,
    mark_minimum: bool = True,
):
    import matplotlib.pyplot as plt

    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    loss_arr = np.asarray(loss_grid, dtype=float)
    if loss_arr.shape != (y_arr.size, x_arr.size):
        raise ValueError(
            "loss_grid must have shape (len(y_values), len(x_values)), "
            f"got {loss_arr.shape} for {(y_arr.size, x_arr.size)}."
        )

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    else:
        fig = ax.figure

    image = ax.imshow(
        loss_arr,
        origin="lower",
        aspect="auto",
        extent=(x_arr.min(), x_arr.max(), y_arr.min(), y_arr.max()),
        cmap="viridis",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Loss landscape")
    fig.colorbar(image, ax=ax, fraction=0.046, label="Loss")

    minimum = None
    if mark_minimum:
        min_index = np.unravel_index(np.argmin(loss_arr), loss_arr.shape)
        minimum = (x_arr[min_index[1]], y_arr[min_index[0]], loss_arr[min_index])
        ax.scatter(minimum[0], minimum[1], marker="x", color="white", s=48, linewidths=1.5)

    return fig, ax, {
        "x_values": x_arr,
        "y_values": y_arr,
        "loss_grid": loss_arr,
        "minimum": minimum,
    }
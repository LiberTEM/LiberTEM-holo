"""Magnetic phase analysis for electron holography.

This module provides tools for quantitative analysis of magnetic phase images,
including radial integration and analytical phase modeling for uniformly magnetized
rods and spheres.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import hbar, mu_0, e
from scipy.ndimage import gaussian_filter

# --- Physical constants (SI) ---
PHI_0 = 2.067833848e-15  # Magnetic flux quantum (T·m²)
MU_B = 9.274009e-24       # Bohr magneton (A·m²)


def magnetic_contours(
        phase: np.ndarray,
        smooth: int = 4,
        contours: int = 8,
) -> np.ndarray:
    """Smooth with gaussian filter and apply enhanced cosine contours to phase image.

    Parameters
    ----------
    phase : np.ndarray
        2D array of phase values in radians.
    smooth : int, optional
        Standard deviation for Gaussian smoothing (default: 4).
    contours : int, optional
        Number of contour levels to apply (default: 8).

    Returns
    -------
    np.ndarray
        2D array of contour values.

    """
    return np.cos(gaussian_filter(phase, sigma=smooth) * contours)


def plot_radius_ranges(
    phase_image: np.ndarray,
    pixel_size_m: float,
    min_radius_m: float,
    max_radius_m: float,
    center: tuple = None,
) -> None:
    """Plot the minimum and maximum radius ranges on the magnetic phase image.

    Parameters
    ----------
    phase_image : numpy.ndarray
        2D phase image in radians.
    pixel_size_m : float
        Size of a pixel in meters.
    min_radius_m : float
        Minimum radius in meters.
    max_radius_m : float
        Maximum radius in meters.

    """
    ny, nx = phase_image.shape
    if center is None:
        x_center, y_center = nx // 2, ny // 2
    else:
        x_center, y_center = center

    min_radius_px = int(min_radius_m / pixel_size_m)
    max_radius_px = int(max_radius_m / pixel_size_m)

    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    x_min = min_radius_px * np.cos(theta)
    y_min = min_radius_px * np.sin(theta)
    x_max = max_radius_px * np.cos(theta)
    y_max = max_radius_px * np.sin(theta)

    plt.figure()
    plt.imshow(phase_image, cmap='gray', origin='lower')
    plt.plot(
        x_min + x_center, y_min + y_center,
        'r--', label=f'Min radius: {min_radius_m:.1e} m',
        )
    plt.plot(
        x_max + x_center, y_max + y_center,
        'b--', label=f'Max radius: {max_radius_m:.1e} m',
        )
    plt.legend()
    plt.title('Radius Ranges for Radial Integration')
    plt.show()


def radial_integration(
    phase_image: np.ndarray,
    pixel_size_m: float,
    min_radius_m: float,
    max_radius_m: float,
    num_radii: int = 50,
    center: tuple = None,
) -> dict:
    """Perform iterative radial integration of a magnetic phase image.

    Parameters
    ----------
    phase_image : numpy.ndarray
        2D phase image in radians.
    pixel_size_m : float
        Size of a pixel in meters.
    min_radius_m : float
        Minimum integration radius in meters.
    max_radius_m : float
        Maximum integration radius in meters.
    num_radii : int, optional
        Number of radii to sample between min and max (default: 50).

    Returns
    -------
    dict
        Dictionary containing:
        - radii: Array of radii used for integration (in m).
        - m_B_components: Inductive moment components for each radius (in Am²).
        - m_components: Magnetic moment components for each radius (in Am²).

    """
    min_radius_px = int(min_radius_m / pixel_size_m)
    max_radius_px = int(max_radius_m / pixel_size_m)

    ny, nx = phase_image.shape
    y, x = np.indices((ny, nx))
    if center is None:
        x_center, y_center = nx // 2, ny // 2
    else:
        x_center, y_center = center

    radii = np.linspace(
        min_radius_px, max_radius_px, num_radii, endpoint=True
        ) * pixel_size_m
    m_B_components = []

    for radius_m in radii:
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        x_loop = radius_m * np.cos(theta)
        y_loop = radius_m * np.sin(theta)

        x_loop_px = (x_loop / pixel_size_m + x_center).astype(int)
        y_loop_px = (y_loop / pixel_size_m + y_center).astype(int)

        x_loop_px = np.clip(x_loop_px, 0, nx - 1)
        y_loop_px = np.clip(y_loop_px, 0, ny - 1)

        phase_loop = phase_image[y_loop_px, x_loop_px]
        integrand_x = -np.sin(theta) * phase_loop
        integrand_y = np.cos(theta) * phase_loop

        d_theta = theta[1] - theta[0]  # Angular step size (assumes uniform theta)
        m_B_x = (hbar / (e * mu_0)) * radius_m * np.sum(integrand_x) * d_theta
        m_B_y = (hbar / (e * mu_0)) * radius_m * np.sum(integrand_y) * d_theta

        m_B_components.append((m_B_x, m_B_y))

    m_B_components = np.array(m_B_components)
    m_components = 2 * m_B_components

    return {
        "radii": radii,
        "m_B_components": m_B_components,
        "m_components": m_components,
    }


def fit_magnetic_moment(
        result: dict,
        plot: bool = False
) -> tuple:
    """Fit the radial integration results to extract the magnetic moment.

    Parameters
    ----------
    result : dict
        Output of `radial_integration`.
    plot : bool, optional
        If True, display a plot of the fit. Default is False.

    Returns
    -------
    tuple
        Fitted magnetic moment components (m_x, m_y) in A·m².

    """
    def _quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * x**2 + b * x + c

    fit_x = curve_fit(_quadratic, result['radii'], result['m_components'][:, 0])
    fit_y = curve_fit(_quadratic, result['radii'], result['m_components'][:, 1])

    m_x = _quadratic(0, *fit_x[0])
    m_y = _quadratic(0, *fit_y[0])

    if plot:
        radii = np.linspace(0, np.max(result['radii']), 100)
        fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
        ax[0].scatter(
            result['radii']*1e9, result['m_components'][:, 0], label='m_x data'
            )
        ax[1].scatter(
            result['radii']*1e9, result['m_components'][:, 1], label='m_y data'
            )
        ax[0].plot(
            radii*1e9, _quadratic(radii, *fit_x[0]),
            'r--', label='m_x fit'
            )
        ax[1].plot(
            radii*1e9, _quadratic(radii, *fit_y[0]),
            'b--', label='m_y fit'
            )
        plt.xlabel('Radius (nm)')
        plt.ylabel('Magnetic moment (Am²)')
        plt.title('Quadratic fit of magnetic moment components')
        plt.legend()
        plt.show()

    return m_x, m_y


def generate_rectangular_lines(
    center: tuple,
    a: float,
    b: float,
    pixel_size: float,
    n_samples: int = 100,
    image: np.ndarray = None,
    show_plot: bool = True,
) -> dict:
    """Generate pixel coordinates for rectangular lines centered at a given point.
    Parameters
    ----------
    center : tuple
        (x, y) coordinates of the rectangle center in pixels.
    a : float
        Half-width of the rectangle in meters.
    b : float
        Half-height of the rectangle in meters.
    pixel_size : float
        Size of a pixel in meters.
    n_samples : int, optional
        Number of sample points along each line (default: 100).
    image : np.ndarray, optional
        Optional image to display the lines on (default: None).
    show_plot : bool, optional
        Whether to display the plot (default: True).
    Returns
    -------
    dict
        Dictionary containing pixel coordinates for 'top', 'bottom', 'left', and 'right' lines.

    """
    cx, cy = center

    ax = a / pixel_size
    by = b / pixel_size

    t = np.linspace(-1, 1, 2 * n_samples + 1)

    top = np.column_stack((cx + ax * t, np.full_like(t, cy - by)))
    bottom = np.column_stack((cx + ax * t, np.full_like(t, cy + by)))
    left = np.column_stack((np.full_like(t, cx - ax), cy + by * t))
    right = np.column_stack((np.full_like(t, cx + ax), cy + by * t))

    lines = {
        "top": np.round(top).astype(int),
        "bottom": np.round(bottom).astype(int),
        "left": np.round(left).astype(int),
        "right": np.round(right).astype(int),
    }

    if show_plot and image is not None:
        plt.imshow(image, cmap="gray")
        for line in lines.values():
            plt.plot(line[:, 1], line[:, 0], linewidth=1.5)
        plt.scatter(cy, cx, c="red", s=30)
        plt.title("Integration rectangle")
        plt.axis("off")
        plt.show()

    return lines


def phase_uniform_rod(
    dim: tuple,
    Lx: float,
    Ly: float,
    Lz: float,
    b_0: float,
    phi: float = np.pi/2,
    theta: float = np.pi/2,
    a: float = 1e-9,
) -> np.ndarray:
    """Calculate the 2D phase map for a uniformly magnetized rod in SI units.

    Parameters
    ----------
    dim : tuple (int, int)
        Dimensions of the field of view in pixels (y_dim, x_dim).
    Lx, Ly, Lz : float
        Dimensions of the rod in meters.
    b_0 : float
        Magnetic induction in tesla (T).
    phi : float, optional
        Azimuthal angle of magnetization in radians (default: π/2).
    theta : float, optional
        Polar angle of magnetization in radians (default: π/2).
    a : float, optional
        Grid spacing in meters (default: 1e-9 m).

    Returns
    -------
    numpy.ndarray
        2D array of phase values in radians.

    """
    y_dim, x_dim = dim
    coeff = -b_0 / (4 * PHI_0)

    def _F_0(x: float, y: float) -> float:
        """Factor for geometry of rod."""
        A = np.log(x**2 + y**2 + 1E-30)
        B = np.arctan2(y, x)
        return x * A - 2 * x + 2 * y * B

    x = (np.arange(x_dim) - (x_dim - 1) / 2) * a
    y = (np.arange(y_dim) - (y_dim - 1) / 2) * a
    xx, yy = np.meshgrid(x, y)

    phase_map = coeff * Lz * (
        -np.cos(phi) * np.sin(theta) * (
            -_F_0(xx - Lx/2, yy - Ly/2)
            + _F_0(xx + Lx/2, yy - Ly/2)
            + _F_0(xx - Lx/2, yy + Ly/2)
            - _F_0(xx + Lx/2, yy + Ly/2)
        )
        + np.sin(phi) * np.sin(theta) * (
            -_F_0(yy - Ly/2, xx - Lx/2)
            + _F_0(yy + Ly/2, xx - Lx/2)
            + _F_0(yy - Ly/2, xx + Lx/2)
            - _F_0(yy + Ly/2, xx + Lx/2)
        )
    )
    return phase_map


def phase_uniform_sphere(
    dim: tuple,
    R: float,
    b_0: float,
    phi: float = np.pi/2,
    theta: float = np.pi/2,
    a: float = 1e-9,
) -> np.ndarray:
    """Calculate the 2D phase map for a uniformly magnetized sphere in SI units.

    Parameters
    ----------
    dim : tuple (int, int)
        Dimensions of the field of view in pixels (y_dim, x_dim).
    R : float
        Radius of the sphere in meters.
    b_0 : float
        Magnetic induction in tesla (T).
    phi : float, optional
        Azimuthal angle of magnetization in radians (default: π/2).
    theta : float, optional
        Polar angle of magnetization in radians (default: π/2).
    a : float, optional
        Pixel size in meters (default: 1e-9 m).

    Returns
    -------
    numpy.ndarray
        2D array of phase values in radians.

    """
    y_dim, x_dim = dim
    x = (np.arange(x_dim) - (x_dim - 1) / 2) * a
    y = (np.arange(y_dim) - (y_dim - 1) / 2) * a
    xx, yy = np.meshgrid(x, y)
    b_perp = b_0 * np.sin(theta)
    coeff = -(2.0 / 3.0) * np.pi * b_perp / PHI_0
    r_squared = np.maximum(xx**2 + yy**2, 1e-30)
    angular = yy * np.cos(phi) - xx * np.sin(phi)
    radial = 1 - np.clip(1 - r_squared / (R**2 + 1e-30), 0, 1) ** (3 / 2)
    phase_map = coeff * R**3 * angular / r_squared * radial
    return np.nan_to_num(phase_map)


def profile_uniform_sphere(
    x: np.ndarray,
    B_perp: float,
    a: float,
) -> np.ndarray:
    """Calculate the 2D phase map for a uniformly magnetized sphere.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates in meters.
    B_perp : float
        Perpendicular magnetic induction in tesla (T).
    a : float
        Radius of the sphere in meters.

    Returns
    -------
    numpy.ndarray
        Phase values in radians.

    """
    epsilon = 1e-12
    x = np.array(x)
    result = np.zeros_like(x)
    abs_x = np.abs(x)
    inside = abs_x <= a
    num = (a**3 - (a**2 - x[inside]**2)**(3/2))
    result[inside] = (
        (e / hbar) * B_perp * num / (x[inside] + epsilon)
    )
    outside = abs_x > a
    result[outside] = (
        (e / hbar) * B_perp * (a**3 / (x[outside] + epsilon))
    )
    return result

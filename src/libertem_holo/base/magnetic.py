"""Magnetic phase analysis."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from libertem_holo.base.filters import clipped

# --- Physical constants (SI) ---
HBAR = 1.054_571_817e-34      # Js
MU0  = 4 * np.pi * 1e-7       # H/m
E    = 1.602_176_634e-19      # C
PHI_0 = 2.067833848e-15  # T*m^2
MU_B = 9.274009e-24 # Am2

def to_xy(
    coords: np.ndarray,
) -> np.ndarray:
    """Convert to (x, y) format for plotting."""
    return np.array([[y, x] for x, y in coords])

def round_coords(
    arr: np.ndarray,
) -> np.ndarray:
    """Round coordinates to ensure integer values."""
    return np.round(arr).astype(int)

def estimate_magnetic_moment_loop_integral(
        phase_image: np.ndarray,
        pixel_size_m: float = 1.0e-9,
        min_radius_m: float = 12.5e-9,
        max_radius_m: float = 32.0e-9,
    ) -> dict:
    """Estimate the magnetic moment from a phase image using a loop integral.

    Parameters
    ----------
    phase_image : numpy.ndarray
        Phase image in radians.
    pixel_size_m : float
        Size of a pixel in meters (default: 1 nm/px).
    min_radius_m : float
        Minimum integration radius in meters.
    max_radius_m : float
        Maximum integration radius in meters.

    Returns
    -------
    dict
        Dictionary containing:
        - radii: List of radii used for integration (in meters).
        - m_B_components: Inductive moment components for each radius (in A·m²).
        - m_components: Magnetic moment components for each radius (in A·m²).

    """
    min_radius_px = int(min_radius_m / pixel_size_m)
    max_radius_px = int(max_radius_m / pixel_size_m)

    ny, nx = phase_image.shape
    y, x = np.indices((ny, nx))
    x_center, y_center = nx // 2, ny // 2

    radii = np.linspace(min_radius_px, max_radius_px, 50, endpoint=True) * pixel_size_m
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

        m_B_x = (HBAR / (E * MU0)) * radius_m * np.trapz(integrand_x, theta)
        m_B_y = (HBAR / (E * MU0)) * radius_m * np.trapz(integrand_y, theta)

        m_B_components.append((m_B_x, m_B_y))

    m_B_components = np.array(m_B_components)
    m_components = 2 * m_B_components

    return {
        "radii": radii,
        "m_B_components": m_B_components,
        "m_components": m_components,
    }

def fit_result(result: dict) -> tuple:
    """Fit quadratic curve to radial integration of phase."""
    def _quadratic(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a*x**2+b

    fit=curve_fit(_quadratic, result['radii'], result['m_components'][...,0])
    mx_fit = _quadratic(0, *fit[0])
    fit=curve_fit(_quadratic, result['radii'], result['m_components'][...,1])
    my_fit = _quadratic(0, *fit[0])
    return mx_fit, my_fit

def get_samp_params(
    res: float,
    shape: tuple,
) -> dict:
    """Get experimental sampling parameters of phase images."""
    n, m = shape
    p = res/n
    constants = {
        'n': n,
        'm': m,
        'p': p,
        'phi0': 2.07e3,
        'mu0': 4 * np.pi * 1e-7,
        'muB': 9.274e3
    }
    constants['fov'] = constants['n'] * constants['p']
    constants['fact'] = constants['phi0']/(np.pi * constants['mu0'] * constants['muB'])
    return constants

def define_region(
    img: np.ndarray,
    samp: int,
    center: tuple,
    radius: tuple,
    plot: bool = False,
) -> np.ndarray:
    """Define rectangular region from mininum and maximum radius length."""
    a = radius[0]
    b = radius[1]
    p = get_samp_params['p']
    i_vals = np.linspace(-samp, samp, 2 * samp + 1)

    cira1 = round_coords(
        np.array([[center[0] + (a/p)*i/samp, center[1] - (b/p)] for i in i_vals])
    )
    cira2 = round_coords(
        np.array([[center[0] + (a/p)*i/samp, center[1] + (b/p)] for i in i_vals])
    )
    cirb1 = round_coords(
        np.array([[center[0] - (a/p), center[1] + (b/p)*i/samp] for i in i_vals])
    )
    cirb2 = round_coords(
        np.array([[center[0] + (a/p), center[1] + (b/p)*i/samp] for i in i_vals])
    )

    if plot:
        for coords in [cira1, cira2, cirb1, cirb2]:
            xy = to_xy(coords)
        plt.figure()
        plt.imshow(
            img, cmap='gray', origin='lower',
            vmin=np.min(clipped(img)), vmax=np.max(clipped(img)),
        )
        plt.plot(xy[:, 0], xy[:, 1], marker='o', markersize=2, linestyle='None', c='r')
        plt.plot(center[1], center[0], 'bo', label='Center')

    return np.array([cira1, cira2, cirb1, cirb2])


def integrate(
    img: np.ndarray,
    coords: np.ndarray,
) -> float:
    """Integrate phase around region."""
    region = define_region()
    integrations = []
    for coords in region:
        values = []
        for x, y in coords:
            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                values.append(img[x, y])
        integrations.append(np.sum(values))

def profile_uniform_sphere(
        x: np.ndarray,
        B_perp: float,
        a: float,
    ) -> np.ndarray:
    """Model line profile for a uniformly magnetized sphere from pyramid (maybe)."""
    epsilon=1e-12
    x = np.array(x)
    result = np.zeros_like(x)
    abs_x = np.abs(x)
    inside = abs_x <= a
    result[inside] = (
        (E / HBAR) * B_perp *
        ((a**3 - (a**2 - x[inside]**2)**(3/2)) / (x[inside] + epsilon))
    )
    outside = abs_x > a
    result[outside] = (
        (E / HBAR) * B_perp *
        (a**3 / (x[outside] + epsilon))
    )
    return result

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

    From pyramid but numpy friendly.

    Parameters
    ----------
    dim : tuple (int, int)
        Dimensions of the field of view in pixels (y_dim, x_dim).
    Lx, Ly, Lz : float
        Dimensions of the rod in meters.
    b_0 : float
        Magnetic induction in tesla (T).
    phi : float, optional
        Azimuthal angle of magnetization in radians (default: 0.0, along y-axis).
    theta : float, optional
        Polar angle of magnetization in radians (default: np.pi/2, in xy-plane).
    a : float, optional
        Grid spacing in meters (default: 1 nm = 1e-9 m).

    Returns
    -------
    phase_map : numpy.ndarray (dim[0], dim[1])
        2D array of phase values in radians.

    """
    y_dim, x_dim = dim

    coeff = -b_0 / (4 * PHI_0)

    def _F_0(
            x: float,
            y: float,
        ) -> float:
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


def profile_uniform_rod(
        x: np.ndarray,
        Lx: float,
        Ly: float,
        Lz: float,
        b_0: float,
        phi: float = np.pi/2,
        theta: float = np.pi/2,
    ) -> np.ndarray:
    """Model for phase profile along x at y=0 for a uniformly magnetized rod.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates along the line profile in meters.
    Lx, Ly, Lz : float
        Dimensions of the rod in meters.
    b_0 : float
        Magnetic induction in tesla (T).
    phi : float, optional
        Azimuthal angle of magnetization in radians.
    theta : float, optional
        Polar angle of magnetization in radians.

    Returns
    -------
    phase_profile : numpy.ndarray
        Phase values along the line profile in radians.

    """
    PHI_0 = 2.067833848e-15  # Magnetic flux quantum in T*m^2
    coeff = -b_0 / (4 * PHI_0)

    def _F_0(
            x: float,
            y: float,
        ) -> float:
        """Factor for geometry of rod."""
        A = np.log(x**2 + y**2 + 1E-30)
        B = np.arctan2(y, x)
        return x * A - 2 * x + 2 * y * B

    # y=0 for the line profile
    phase_profile = coeff * Lz * (
        -np.cos(phi) * np.sin(theta) * (
            -_F_0(x - Lx/2, -Ly/2)
            + _F_0(x + Lx/2, -Ly/2)
            + _F_0(x - Lx/2, +Ly/2)
            - _F_0(x + Lx/2, +Ly/2)
        )
        + np.sin(phi) * np.sin(theta) * (
            -_F_0(-Ly/2, x - Lx/2)
            + _F_0(+Ly/2, x - Lx/2)
            + _F_0(-Ly/2, x + Lx/2)
            - _F_0(+Ly/2, x + Lx/2)
        )
    )

    return phase_profile

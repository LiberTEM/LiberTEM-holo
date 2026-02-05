import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import pi, mu_0
from libertem_holo.base.filters import clipped
from scipy.ndimage import map_coordinates

# --- Physical constants (SI) ---
HBAR = 1.054_571_817e-34      # J·s
MU0  = 4 * np.pi * 1e-7       # H/m
E    = 1.602_176_634e-19      # C



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
import numpy as np

def estimate_magnetic_moment_loop_integral(phase_image, pixel_size_m=1.0e-9, min_radius_m=12.5e-9, max_radius_m=32.0e-9):
    """
    Estimate the magnetic moment from a phase image using a loop integral in polar coordinates.

    Parameters:
    - phase_image: 2D NumPy array, the phase image in radians.
    - pixel_size_m: float, size of a pixel in meters (default: 1.0e-9 m/px).
    - min_radius_m: float, minimum integration radius in meters.
    - max_radius_m: float, maximum integration radius in meters.

    Returns:
    - m_B: Inductive moment (in A·m²).
    - m: Magnetic moment (in A·m²).
    - radii: List of radii used for integration (in meters).
    - m_B_components: List of inductive moment components for each radius (in A·m²).
    """
    # Physical constants (SI units)
    hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
    e = 1.602176634e-19    # Elementary charge (C)
    mu_0 = 4 * np.pi * 1e-7 # Vacuum permeability (N/A²)

    # Convert radii to pixels
    min_radius_px = int(min_radius_m / pixel_size_m)
    max_radius_px = int(max_radius_m / pixel_size_m)

    # Generate a grid of coordinates (in meters)
    ny, nx = phase_image.shape
    y, x = np.indices((ny, nx))
    x_center, y_center = nx // 2, ny // 2
    x_m = (x - x_center) * pixel_size_m
    y_m = (y - y_center) * pixel_size_m

    # Initialize lists to store results
    radii = np.linspace(min_radius_px, max_radius_px, 50, endpoint=True) * pixel_size_m
    m_B_components = []

    for radius_m in radii:
        # Convert radius to pixels
        radius_px = int(radius_m / pixel_size_m)

        # Generate polar coordinates for the loop integral
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        x_loop = radius_m * np.cos(theta)
        y_loop = radius_m * np.sin(theta)

        # Convert loop coordinates to pixel indices
        x_loop_px = (x_loop / pixel_size_m + x_center).astype(int)
        y_loop_px = (y_loop / pixel_size_m + y_center).astype(int)

        # Clip indices to stay within the image bounds
        x_loop_px = np.clip(x_loop_px, 0, nx - 1)
        y_loop_px = np.clip(y_loop_px, 0, ny - 1)

        # Extract phase values along the loop
        phase_loop = phase_image[y_loop_px, x_loop_px]

        # Compute the loop integral for m_B_x and m_B_y
        integrand_x = -np.sin(theta) * phase_loop
        integrand_y = np.cos(theta) * phase_loop

        # Trapezoidal rule for numerical integration
        m_B_x = (hbar / (e * mu_0)) * radius_m * np.trapz(integrand_x, theta)
        m_B_y = (hbar / (e * mu_0)) * radius_m * np.trapz(integrand_y, theta)

        m_B_components.append((m_B_x, m_B_y))

    # Convert to array for easier handling
    m_B_components = np.array(m_B_components)

    # Calculate the magnetic moment (m = 2 * m_B)
    m_components = 2 * m_B_components

    return {
        "radii": radii,
        "m_B_components": m_B_components,
        "m_components": m_components,
    }

def fit_result(result):

    def quadratic(x,a,b):
        return a*x**2+b

    fit=curve_fit(quadratic, result['radii'], result['m_components'][...,0])
    mx_fit = quadratic(0, *fit[0])
    fit=curve_fit(quadratic, result['radii'], result['m_components'][...,1])
    my_fit = quadratic(0, *fit[0])
    return mx_fit, my_fit

def get_samp_params(
    res: float,
    shape: tuple,
) -> dict:
    n, m = shape
    p = res/n
    constants = {
        'n': n,
        'm': m,
        'p': p,
        'phi0': 2.07e3,
        'mu0': 4 * pi * 1e-7,
        'muB': 9.274e3
    }
    constants['fov'] = constants['n'] * constants['p']
    constants['fact'] = constants['phi0'] / (pi * constants['mu0'] * constants['muB'])
    return constants

def define_region(
    img: np.ndarray,
    samp: int,
    center: tuple,
    radius: tuple,
    plot: bool = False,
) -> np.ndarray:
    """Define rectangular region from mininum and maximum radius length around a center."""
    a = radius[0]
    b = radius[1]
    p = get_samp_params['p']
    i_vals = np.linspace(-samp, samp, 2 * samp + 1)

    cira1 = round_coords(np.array([[center[0] + (a/p)*i/samp, center[1] - (b/p)] for i in i_vals]))
    cira2 = round_coords(np.array([[center[0] + (a/p)*i/samp, center[1] + (b/p)] for i in i_vals]))
    cirb1 = round_coords(np.array([[center[0] - (a/p), center[1] + (b/p)*i/samp] for i in i_vals]))
    cirb2 = round_coords(np.array([[center[0] + (a/p), center[1] + (b/p)*i/samp] for i in i_vals]))

    if plot:
        for coords in [cira1, cira2, cirb1, cirb2]:
            xy = to_xy(coords)
        plt.figure()
        plt.imshow(img, cmap='gray', origin='lower', vmin=np.min(clipped(img)), vmax=np.max(clipped(img)))
        plt.plot(xy[:, 0], xy[:, 1], marker='o', markersize=2, linestyle='None', color='red')
        plt.plot(center[1], center[0], 'bo', label='Center')

    return np.array([cira1, cira2, cirb1, cirb2])


def integrate(
    img: np.ndarray,
    coords: np.ndarray,
) -> float:
    region = define_region()
    integrations = []
    for coords in region:
        values = []
        for x, y in coords:
            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                values.append(img[x, y])
        I = np.sum(values)
        integrations.append(I)

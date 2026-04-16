import numpy as np
from scipy.constants import mu_0

from libertem_holo.base.magnetic import radial_integration, fit_magnetic_moment

PHI_0 = 2.067833848e-15  # Magnetic flux quantum (T·m²)


def test_radial_integration(
) -> np.ndarray:
    dim = (128, 128)
    a = 1e-9
    b_perp = 1.256
    R = 20e-9
    phi = np.pi/6
    y_dim, x_dim = dim
    x = (np.arange(x_dim) - (x_dim - 1) / 2) * a
    y = (np.arange(y_dim) - (y_dim - 1) / 2) * a
    xx, yy = np.meshgrid(x, y)
    coeff = -(2.0 / 3.0) * np.pi * b_perp / PHI_0
    r_squared = np.maximum(xx**2 + yy**2, 1e-30)
    angular = yy * np.cos(phi) - xx * np.sin(phi)
    radial = 1 - np.clip(1 - r_squared / (R**2 + 1e-30), 0, 1) ** (3 / 2)
    phase_map = coeff * R**3 * angular / r_squared * radial
    phase_map = np.nan_to_num(phase_map)
    Bx = b_perp * np.cos(phi)
    By = b_perp * np.sin(phi)
    V = 4 / 3 * np.pi * R**3
    Mx = Bx / mu_0
    My = By / mu_0
    mx_true = Mx * V
    my_true = My * V
    results = radial_integration(phase_map, pixel_size_m=a, min_radius_m=25e-9, max_radius_m=120e-9)
    np.testing.assert_allclose(
        fit_magnetic_moment(result=results), [mx_true, my_true], atol=5e-18,
    )

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import pi, mu_0
from libertem_holo.base.filters import clipped


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


class Integrator:
    """Create integration class for general purpose."""
    def get_samp_params(
        self,
        res: float,
        shape: tuple,
    ) -> dict:
        raise NotImplementedError()

    def define_region(
        self,
        center: tuple,
        radius: float,
        plot: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError()

    def integrate(
        self,
        img: np.ndarray,
        coords: np.ndarray,
    ) -> float:
        raise NotImplementedError()

    def sweep_region(
        self,
        limits: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError()

    def get_moments(
        self,
        radius: np.ndarray,
        integration: np.ndarray,
    ) -> tuple:
        raise NotImplementedError()


class RadialIntegration(Integrator):
    pass

class RectangularIntegration(Integrator):
    def __init__(
        self,
        res: float,
        shape: tuple,
        img: np.ndarray,
    ) -> None:
        self._res = res
        self._shape = shape
        self._img = img
    
    def get_samp_params(
        self,
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
        self,
        samp: int,
        center: tuple,
        radius: tuple,
        plot: bool = False,
    ) -> np.ndarray:
        """Define rectangular region from mininum and maximum radius length around a center."""
        a = radius[0]
        b = radius[1]
        p = self.get_samp_params['p']
        i_vals = np.linspace(-samp, samp, 2 * samp + 1)

        cira1 = round_coords(np.array([[center[0] + (a/p)*i/samp, center[1] - (b/p)] for i in i_vals]))
        cira2 = round_coords(np.array([[center[0] + (a/p)*i/samp, center[1] + (b/p)] for i in i_vals]))
        cirb1 = round_coords(np.array([[center[0] - (a/p), center[1] + (b/p)*i/samp] for i in i_vals]))
        cirb2 = round_coords(np.array([[center[0] + (a/p), center[1] + (b/p)*i/samp] for i in i_vals]))

        img = self._img
        if plot:
            for coords in [cira1, cira2, cirb1, cirb2]:
                xy = to_xy(coords)
            plt.figure()
            plt.imshow(img, cmap='gray', origin='lower', vmin=np.min(clipped(img)), vmax=np.max(clipped(img)))
            plt.plot(xy[:, 0], xy[:, 1], marker='o', markersize=2, linestyle='None', color='red')
            plt.plot(center[1], center[0], 'bo', label='Center')

        return np.array([cira1, cira2, cirb1, cirb2])


    def integrate(
        self,
        img: np.ndarray,
        coords: np.ndarray,
        ) -> float:
        region = self.define_region()
        integrations = []
        for coords in region:
            values = []
            for x, y in coords:
                if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                    values.append(img[x, y])
            I = np.sum(values)
            integrations.append(I)


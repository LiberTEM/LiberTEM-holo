import numpy as np
from scipy.ndimage.filters import gaussian_filter


def hologram_frame(amp, phi,
                   counts=1000.,
                   sampling=5.,
                   visibility=1.,
                   f_angle=30.,
                   gaussian_blur=None,
                   poisson_noise=False):
    """
    Generates holograms using phase and amplitude as an input

    See :ref:`holography app` for detailed application example

    .. versionchanged:: 0.1.0
        * Renamed :code:`gaussian_noise` to :code:`gaussian_blur` to correctly reflect the
          functionality.
        * Apply Gaussian blur before Poisson noise to simulate loss of contrast in the imaging
          system. To simulate detector MTF, one can apply blur after applying Poisson noise.
        * Changed the implementation of Poisson noise to correctly reflect the impact of
          :code:`counts` on the noise.

    Notes
    -----
    Theoretical basis for hologram simulations see in:
    Lichte, H., and M. Lehmann. Rep. Prog. Phys. 71 (2008): 016102.
    doi:10.1088/0034-4885/71/1/016102
    :cite:`Lichte2008`

    Parameters
    ----------
    amp, phi: np.ndarray, 2d
        normalized amplitude and phase images of the same shape

    counts: float, default: 1000.
        Number of electron counts in vacuum

    sampling: float, default: 5.
        Hologram fringe sampling (number of pixels per fringe)

    visibility: float, default: 1.
        Hologram fringe visibility (aka fringe contrast)

    f_angle: float, default: 30.
        Angle in degrees of hologram fringes with respect to X-axis

    gaussian_blur: float or int or None, default: None.
        Amount of Gaussian smoothing determined by sigma parameter
        applied to the hologram simulating effect of focus spread.

    poisson_noise: boolean, default: False.
        Poisson noise applied to the hologram.

    Returns
    -------
    holo: np.ndarray, 2d
        hologram image
    """
    if not amp.shape == phi.shape:
        raise ValueError('Amplitude and phase should be 2d arrays of the same shape.')
    sy, sx = phi.shape
    x, y = np.meshgrid(np.arange(sx), np.arange(sy))
    f_angle = f_angle / 180. * np.pi

    holo = counts / 2 * (1. + amp ** 2 + 2. * amp * visibility
                         * np.cos(2. * np.pi * y / sampling * np.cos(f_angle)
                                  + 2. * np.pi * x / sampling * np.sin(f_angle)
                                  - phi))

    if gaussian_blur is not None:
        if not isinstance(gaussian_blur, (float, int)):
            raise ValueError("gaussian_blur parameter should be float or int or None.")
        holo = gaussian_filter(holo, gaussian_blur)

    if poisson_noise:
        if not isinstance(poisson_noise, bool):
            raise ValueError(
                "poisson_noise parameter should be boolean, "
                "implementation was changed compared to the version in LiberTEM. "
                "See also https://github.com/LiberTEM/LiberTEM/issues/1156."
            )
        holo = np.random.poisson(holo)

    return holo

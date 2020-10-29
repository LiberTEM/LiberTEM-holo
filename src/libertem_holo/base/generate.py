import numpy as np
from scipy.ndimage.filters import gaussian_filter


def hologram_frame(amp, phi,
                   counts=1000.,
                   sampling=5.,
                   visibility=1.,
                   f_angle=30.,
                   gaussian_noise=None,
                   poisson_noise=None):
    """
    Generates holograms using phase and amplitude as an input

    See :ref:`holography app` for detailed application example

    .. versionadded:: 0.3.0

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

    gaussian_noise: float or int or None, default: None.
        Amount of Gaussian smoothing determined by sigma parameter
        applied to the hologram simulating effect of focus spread or
        PSF of the detector.

    poisson_noise: float or int or None, default: None.
        Amount of Poisson applied to the hologram.

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

    if poisson_noise:
        if not isinstance(poisson_noise, (float, int)):
            raise ValueError("poisson_noise parameter should be float or int or None.")
        noise_scale = poisson_noise * counts
        holo = noise_scale * np.random.poisson(holo / noise_scale)

    if gaussian_noise:
        if not isinstance(gaussian_noise, (float, int)):
            raise ValueError("gaussian_noise parameter should be float or int or None.")
        holo = gaussian_filter(holo, gaussian_noise)

    return holo

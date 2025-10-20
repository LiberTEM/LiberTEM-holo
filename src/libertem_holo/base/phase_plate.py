import numpy as np


def apply_phase_plate(
    wave: np.ndarray,
    px_size: float,
    defocus: float,
    wavelength: float = 1e-12,
) -> np.ndarray:
    """Apply phase plate to complex wave in SI units (only defocus for now).

    See equation (2.2.8) in Barthel, J. Ultra-precise measurement of optical
    aberrations for sub-Aangstroem transmission electron microscopy.
    Germany: N. p., 2008. Web.

    Parameters
    ----------
    wave
        complex wave you want to defocus
    px_size
        pixel size in meter
    wavelength
        wavelength in meter
    defocus
        aberration factor c20 in meter

    """
    wave_f = np.fft.fft2(wave) / np.prod(wave.shape)
    xx = np.fft.fftfreq(n=wave.shape[1], d=px_size)
    yy = np.fft.fftfreq(n=wave.shape[0], d=px_size)
    gx, gy = np.meshgrid(xx, yy, indexing='ij')
    g = gx + 1j * gy
    chi = np.pi * wavelength * np.real(defocus * g * np.conjugate(g))
    wave_f_defocused = wave_f * np.exp(1j * chi)
    wave_defocused = np.fft.ifft2(wave_f_defocused) * np.prod(wave.shape)
    return wave_defocused

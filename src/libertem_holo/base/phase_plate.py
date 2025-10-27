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
    gx, gy = np.meshgrid(xx, yy, indexing='xy')
    g = gx + 1j * gy
    chi = np.pi * wavelength * np.real(defocus * g * np.conjugate(g))
    wave_f_defocused = wave_f * np.exp(1j * chi)
    wave_defocused = np.fft.ifft2(wave_f_defocused) * np.prod(wave.shape)
    return wave_defocused


def apply_phase_plate_v2(
    wave: np.ndarray,
    px_size: float,
    c20: float,
    c22: float,
    c31: float,
    c33: float,
    c40: float,
    c42: float,
    c44: float,
    wavelength: float = 1e-12,
    padding: bool = True,
    hanning: bool = True,
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
    c20, c22, c31, c33, c40, c42, c44
        aberration factor in meter
    padding
        pads wave
    hanning
        apply hanning filter

    """
    if padding:
        wave_padded = np.zeros((wave.shape[0]*3, wave.shape[1]*3), dtype=wave.dtype)
        wave_padded[wave.shape[0]:wave.shape[0]*2, wave.shape[1]:wave.shape[1]*2] = wave
    else:
        wave_padded = wave
    if hanning:
        wave_padded = wave_padded * np.outer(
            np.hanning(wave_padded.shape[0]), np.hanning(wave_padded.shape[1])
        )
    wave_f = np.fft.fft2(wave_padded) / np.prod(wave_padded.shape)

    xx = np.fft.fftfreq(n=wave_padded.shape[1], d=px_size)
    yy = np.fft.fftfreq(n=wave_padded.shape[0], d=px_size)
    gy, gx = np.meshgrid(xx, yy, indexing='xy')
    g = gx + 1j * gy

    chi_c20 = (wavelength**2 / 2) * np.real(c20 * g * np.conjugate(g))
    chi_c22 = (wavelength**2 / 2) * np.real(c22 * np.conjugate(g)**2)
    chi_c31 = (wavelength**3 / 3) * np.real(c31 * np.conjugate(g)**2 * g)
    chi_c33 = (wavelength**3 / 3) * np.real(c33 * np.conjugate(g)**3)
    chi_c40 = (wavelength**4 / 4) * np.real(c40 * np.conjugate(g)**2 * g**2)
    chi_c42 = (wavelength**4 / 4) * np.real(c42 * np.conjugate(g)**3 * g)
    chi_c44 = (wavelength**4 / 4) * np.real(c44 * np.conjugate(g)**4)

    chi = (2*np.pi / wavelength) * (
        chi_c20 + chi_c22 + chi_c31 + chi_c33 + chi_c40 + chi_c42 + chi_c44
    )

    wave_f_defocused = wave_f * np.exp(1j * chi)
    wave_defocused = np.fft.ifft2(wave_f_defocused) * np.prod(wave_padded.shape)

    if padding:
        wave_defocused_unpadded = np.zeros_like(wave)
        wave_defocused_unpadded = wave_defocused[
            wave.shape[0]:wave.shape[0]*2, wave.shape[1]:wave.shape[1]*2
        ]
    else:
        wave_defocused_unpadded = wave_defocused

    return wave_defocused_unpadded

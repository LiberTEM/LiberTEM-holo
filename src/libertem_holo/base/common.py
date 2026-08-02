import numpy as np

# FIXME perhaps we have to move this elsewhere to keep dependencies clean?
from ptychography40.reconstruction.common import ifftshift_coords, fftshift_coords


def spatial_frequency(sideband_pos, shape):
    '''
    Transform ifftshifted sideband position to spatial frequency
    '''
    transformer = ifftshift_coords(reconstruct_shape=shape)
    origin = transformer((0, 0))
    sideband = transformer(sideband_pos)
    return sideband - origin


def sideband_position(freq, shape):
    '''
    Transform spatial frequency to ifftshifted sideband position
    '''
    forward = ifftshift_coords(reconstruct_shape=shape)
    origin = forward((0, 0))
    sideband = origin + freq
    return fftshift_coords(reconstruct_shape=shape)(sideband)


def phase_shift(image_shift, freq, shape):
    '''
    Calculate the phase shift that corresponds to an image shift of fringes.

    To stablize real-world phase-shifting holography, one has to track the
    relative shift of fringes versus image as well as the image shift. The shift
    of fringes in an image series relative to a fixed region of interest is easy to
    track, see :func:`fringe_phase`.

    The image shift can be tracked from a bandpass-filtered image, i.e. with
    only the center band.

    The image shift has to be compensated in the reconstruction to make sure the
    image is not blurred. This function removes the necessity to re-track the
    phase shift after image shift was compensated since it allows to calculates the
    corresponding phase shift difference instead.

    FIXME Figure out the sign properly
    '''
    phase_shift = (image_shift @ (np.array(freq) / np.array(shape))) * -2 * np.pi
    return phase_shift % (2 * np.pi)


def fringe_params(hologram, ref_wave):
    '''
    Calculate phase and amplitude of the fringes as well as total intensity of a hologram.

    The phase and amplitude correspond to the phase and amplitude of the hologram's Fourier
    transform at the sideband position. This function calculates a single value of the Fourier
    transform directly instead of calculating the complete Fourier transform.

    Since calculating the reference wave is relatively expensive and it can be re-used,
    it is supplied as a parameter instead of just the frequency/side band position.

    Parameters
    ----------
    hologram : numpy.ndarray
        Can also be a stack of holograms for efficiency.
    ref_wave : numpy.ndarray
        Reference wave for the fringes, see :func:`wave`.
    '''
    ft = np.sum(hologram*np.conj(ref_wave), axis=(-1, -2))
    tot = np.sum(hologram, axis=(-1, -2))
    return np.angle(ft), np.abs(ft), tot


def wave(shape, freq):
    '''
    Calculate a complex wave that is modulated with :code:`freq`,
    i.e. includes a phase ramp.
    '''
    y, x = np.ogrid[:shape[0], :shape[1]]
    return np.exp(1j*(freq[-2] * y/shape[0] + freq[-1] * x/shape[1]) * 2*np.pi)

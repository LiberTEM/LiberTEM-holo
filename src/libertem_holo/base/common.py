import numpy as np

# FIXME perhaps we have to move this elsewhere to keep dependencies clean?
from ptychography40.reconstruction.common import ifftshift_coords


def spatial_frequency(sideband_pos, shape):
    '''
    Transform ifftshifted sideband position to spatial frequency
    '''
    transformer = ifftshift_coords(reconstruct_shape=shape)
    origin = transformer((0, 0))
    sideband = transformer(sideband_pos)
    return sideband - origin


def phase_shift(image_shift, freq, shape):
    '''
    Calculate the phase shift that corresponds to an image shift of fringes.

    To stablize real-world phase-shifting holography, one has to track the
    relative shift of fringes versus image as well as the image shift. The shift
    of fringes in an image series relative to a fixed region of interest is easy to
    track as the phase of the Fourier transform at the sideband position.

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

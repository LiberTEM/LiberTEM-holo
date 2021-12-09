import numpy as np
import scipy.ndimage
import pytest

from libertem_holo.base.reconstr import estimate_sideband_position
from libertem_holo.base.common import spatial_frequency, phase_shift


@pytest.fixture
def ref_hologram():
    shape = (131, 255)
    freq = (6, -15)
    obj_wave = np.ones(shape, dtype=np.complex128)
    y, x = np.ogrid[:shape[0], :shape[1]]
    ref_wave = np.exp(1j*(freq[0] * y/shape[0] + freq[1] * x/shape[1]) * 2*np.pi)
    ref_hologram = np.abs(obj_wave + ref_wave)**2
    return (ref_hologram, freq)


def test_spatial_frequency(ref_hologram):
    ref_hologram, ref_freq = ref_hologram
    sideband_pos = estimate_sideband_position(ref_hologram, (1, 1))
    freq = spatial_frequency(sideband_pos, ref_hologram.shape)
    assert np.allclose(ref_freq, freq)


def test_shift(ref_hologram):
    ref_hologram, ref_freq = ref_hologram
    sideband_pos = estimate_sideband_position(ref_hologram, (1, 1))
    fft_holo = np.fft.fft2(ref_hologram)
    at_sideband = fft_holo[sideband_pos[0], sideband_pos[1]]
    angle = np.angle(at_sideband)

    shift = np.random.randint(-10, 10, 2)
    # The hologram wraps around perfectly if freq is integer
    shifted_hologram = scipy.ndimage.shift(ref_hologram, shift, mode='grid-wrap')
    shifted_fft_holo = np.fft.fft2(shifted_hologram)
    shifted_at_sideband = shifted_fft_holo[sideband_pos[0], sideband_pos[1]]
    new_angle = np.angle(shifted_at_sideband)

    calculated_phase_shift = phase_shift(shift, ref_freq, ref_hologram.shape)

    # Compare complex number to be robust against angle wrap-around
    assert np.allclose(np.exp(1j*(angle + calculated_phase_shift)), np.exp(1j*new_angle))

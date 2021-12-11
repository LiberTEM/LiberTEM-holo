import numpy as np
import scipy.ndimage
import pytest

from libertem_holo.base.reconstr import estimate_sideband_position
from libertem_holo.base.common import (
    spatial_frequency, phase_shift, wave, fringe_params, sideband_position
)


@pytest.fixture
def ref_hologram():
    shape = (131, 255)
    freq = (6, -15)
    obj_wave = wave(shape, (0, 0))
    ref_wave = wave(shape, freq)
    ref_hologram = np.abs(obj_wave + ref_wave)**2
    return (ref_hologram, freq)


def test_spatial_frequency(ref_hologram):
    ref_hologram, ref_freq = ref_hologram
    sideband_pos = estimate_sideband_position(ref_hologram, (1, 1))
    freq = spatial_frequency(sideband_pos, ref_hologram.shape)
    assert np.allclose(ref_freq, freq)


def test_shift(ref_hologram):
    ref_hologram, ref_freq = ref_hologram
    ref_wave = wave(ref_hologram.shape, ref_freq)
    sideband_pos = estimate_sideband_position(ref_hologram, (1, 1))
    fft_holo = np.fft.fft2(ref_hologram)
    at_sideband = fft_holo[sideband_pos[0], sideband_pos[1]]
    angle_ref = np.angle(at_sideband)
    angle, _, _ = fringe_params(ref_hologram, ref_wave)
    # Compare complex number to be robust against angle wrap-around
    assert np.allclose(np.exp(1j*angle_ref), np.exp(1j*angle))

    shift = np.random.randint(-10, 10, 2)
    # The hologram wraps around perfectly if freq is integer
    shifted_hologram = scipy.ndimage.shift(ref_hologram, shift, mode='grid-wrap')
    shifted_fft_holo = np.fft.fft2(shifted_hologram)
    shifted_at_sideband = shifted_fft_holo[sideband_pos[0], sideband_pos[1]]
    new_angle_ref = np.angle(shifted_at_sideband)
    new_angle, _, _ = fringe_params(shifted_hologram, ref_wave)

    # Test zero shift for zero shift and stack processing
    calculated_phase_shift = phase_shift([(0, 0), shift], ref_freq, ref_hologram.shape)
    assert np.allclose(calculated_phase_shift[0], 0)

    print(new_angle_ref, new_angle)

    assert np.allclose(np.exp(1j*new_angle_ref), np.exp(1j*new_angle))
    assert np.allclose(np.exp(1j*(angle + calculated_phase_shift[1])), np.exp(1j*new_angle))


def test_sideband():
    shape = (23, 42)
    sideband_pos = (3, 39)
    freq = spatial_frequency(sideband_pos, shape)
    print(freq, sideband_position(freq, shape))
    assert np.allclose(sideband_pos, sideband_position(freq, shape))

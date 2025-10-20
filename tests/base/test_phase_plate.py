import numpy as np
import pytest

from libertem_holo.base.phase_plate import apply_phase_plate


@pytest.mark.parametrize(
    "shape", [
        (64, 64),
        (32, 64),
        (31, 17),
    ],
)
@pytest.mark.parametrize(
    "defocus", [
        1e-6,
        -1e-6,
        100e-6,
        10e-6+1j*1e-6,
        1,
    ],
)
def test_apply_phase_plate(shape, defocus):
    wavelength = 1e-12
    px_size = 0.2e-9
    amp = np.ones(shape)
    phase = np.random.uniform(low=0, high=1, size=shape)
    wave = amp * np.exp(1j*phase)

    wave_defocused = apply_phase_plate(
        wave=wave, px_size=px_size, wavelength=wavelength, defocus=defocus
    )

    wave_de_defocused = apply_phase_plate(
        wave=wave_defocused, px_size=px_size, wavelength=wavelength, defocus=-defocus
    )

    assert np.allclose(wave, wave_de_defocused)

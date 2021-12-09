import numpy as np
import pytest
import scipy.ndimage


from libertem_holo.base.direct import (
    direct_reconstruction, reconstruction_engine, reconstruction_recipe
)


@pytest.fixture
def obj():
    size = 256

    obj = np.ones((size, size), dtype=np.complex64)
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]

    outline = (((y*1.2)**2 + x**2) > 110**2) & ((((y*1.2)**2 + x**2) < 120**2))
    obj[outline] = 0

    left_eye = ((y + 40)**2 + (x + 40)**2) < 20**2
    obj[left_eye] = 0
    right_eye = (np.abs(y + 40) < 15) & (np.abs(x - 40) < 30)
    obj[right_eye] = 0

    nose = (y + 20 + x > 0) & (x < 0) & (y < 10)

    obj[nose] = (0.05j * x + 0.05j * y)[nose]

    mouth = (((y*1)**2 + x**2) > 50**2) & ((((y*1)**2 + x**2) < 70**2)) & (y > 20)

    obj[mouth] = 0

    tongue = (((y - 50)**2 + (x - 50)**2) < 20**2) & ((y**2 + x**2) > 70**2)
    obj[tongue] = 0

    # This wave modulation introduces a strong signature in the diffraction pattern
    # that allows to confirm the correct scale and orientation.
    signature_wave = np.exp(1j*(3 * y + 7 * x) * 2*np.pi/size)

    obj += 0.3*signature_wave - 0.3

    return obj


@pytest.fixture
def simulated(obj):
    freq = (43, 29)
    size_y, size_x = obj.shape
    y, x = np.ogrid[:size_y, :size_x]
    obj_wave = np.ones(obj.shape, dtype=np.complex128)
    angles = np.random.random(23) * 2*np.pi
    directions = np.exp(1j*angles)
    ref_waves = np.exp(
        1j*(freq[0] * y/size_y + freq[1] * x/size_x) * 2*np.pi
    ) * directions[:, np.newaxis, np.newaxis]
    ref_holograms = np.abs(obj_wave + ref_waves)**2
    holograms = np.abs(obj_wave*obj + ref_waves)**2
    return (holograms, ref_holograms, angles)


def test_direct(obj, simulated):
    holograms, ref_holograms, angles = simulated
    ref_rec = direct_reconstruction(ref_holograms, angles, 1)
    rec = direct_reconstruction(holograms, angles, ref_rec)
    assert np.allclose(obj, rec)


def test_direct_shifted(obj, simulated):
    holograms, ref_holograms, angles = simulated
    y_offsets = np.random.randint(-20, 20, len(holograms))
    x_offsets = np.random.randint(-20, 20, len(holograms))

    shifted_holograms = np.zeros_like(holograms)
    for i in range(len(holograms)):
        shifted_holograms[i] = scipy.ndimage.shift(
            holograms[i],
            (y_offsets[i], x_offsets[i])
        )

    ref_rec = direct_reconstruction(ref_holograms, angles, 1)
    rec = direct_reconstruction(shifted_holograms, angles, ref_rec, y_offsets, x_offsets)
    assert np.allclose(obj[21:-21, 21:-21], rec[21:-21, 21:-21])


def test_direct_shifted_tiled(obj, simulated):
    holograms, ref_holograms, angles = simulated

    y_offsets = np.random.randint(-20, 20, len(holograms))
    x_offsets = np.random.randint(-20, 20, len(holograms))

    shifted_holograms = np.zeros_like(holograms)
    for i in range(len(holograms)):
        shifted_holograms[i] = scipy.ndimage.shift(
            holograms[i],
            (y_offsets[i], x_offsets[i])
        )

    tile_00 = shifted_holograms[:7, :101]
    tile_01 = shifted_holograms[:7, 101:]
    tile_10 = shifted_holograms[7:, :101]
    tile_11 = shifted_holograms[7:, 101:]

    rotation, recipe = reconstruction_recipe(angles)

    out = np.zeros(obj.shape, dtype=np.complex128)
    reconstruction_engine(
        tile_00,
        y_offsets=y_offsets[:7],
        x_offsets=x_offsets[:7],
        rotation=rotation[:7],
        recipe=recipe,
        out=out
    )
    reconstruction_engine(
        tile_01,
        y_offsets=y_offsets[:7] - 101,
        x_offsets=x_offsets[:7],
        rotation=rotation[:7],
        recipe=recipe,
        out=out
    )
    reconstruction_engine(
        tile_10,
        y_offsets=y_offsets[7:],
        x_offsets=x_offsets[7:],
        rotation=rotation[7:],
        recipe=recipe,
        out=out
    )
    reconstruction_engine(
        tile_11,
        y_offsets=y_offsets[7:] - 101,
        x_offsets=x_offsets[7:],
        rotation=rotation[7:],
        recipe=recipe,
        out=out
    )

    ref_rec = direct_reconstruction(ref_holograms, angles, 1)
    rec = out / ref_rec
    assert np.allclose(obj[21:-21, 21:-21], rec[21:-21, 21:-21])

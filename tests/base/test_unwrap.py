import pytest
import numpy as np
from libertem_holo.base.unwrap import (
    derivative_variance,
    quality_unwrap,
    unwrap_phase_laplacian,
    phase_unwrap,
)


@pytest.mark.with_numba
def test_quality_unwrap_smoketest():
    phase = np.random.random(128*128).reshape(128, 128)
    dv = derivative_variance(phase)
    _ = quality_unwrap(phase, dv)


def test_phase_unwrap_helper_smoke():
    rng = np.random.default_rng()
    phase = rng.random(128*128).reshape(128, 128)
    a = phase_unwrap(image=phase, method="skimage")
    b = phase_unwrap(image=np.exp(1j * phase), method="skimage")

    assert np.allclose(a, b)


@pytest.mark.parametrize(
    "method", ["skimage", "quality", "laplacian"],
)
def test_phase_unwrap_helper_methods(method: str):
    rng = np.random.default_rng()
    phase = rng.random(128*128).reshape(128, 128)
    _ = phase_unwrap(image=phase, method=method)


def test_unwrap_phase_laplacian():
    rng = np.random.default_rng()
    phase = rng.random(128*128).reshape(128, 128)
    _ = unwrap_phase_laplacian(wrapped_phase=phase)

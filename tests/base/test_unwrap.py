import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.unwrap import (
    derivative_variance,
    phase_unwrap,
    quality_unwrap,
    unwrap_phase_laplacian,
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


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_unwrap_laplacian_with_backends(backend):
    rng = np.random.default_rng()
    phase = rng.random(128*128).reshape(128, 128)

    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    phase = xp.asarray(phase)
    _ = unwrap_phase_laplacian(wrapped_phase=phase, xp=xp)

import pytest
import numpy as np

from sparseconverter import for_backend, NUMPY
from libertem.utils.devices import detect

from libertem_holo.base.filters import (
    butterworth_disk,
    butterworth_line,
    disk_aperture,
)


@pytest.mark.benchmark(
    group="filters"
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy'],
)
def test_butterworth_disk(backend, benchmark, lt_ctx):
    if backend == 'cupy':
        d = detect()
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    if backend == 'cupy':
        import cupy as xp
    else:
        xp = np

    benchmark(
        lambda: for_backend(
            butterworth_disk(shape=(4096, 4096), radius=128.0, order=12, xp=xp),
            NUMPY
        )
    )


@pytest.mark.benchmark(
    group="filters"
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy'],
)
def test_butterworth_line(backend, benchmark, lt_ctx):
    if backend == 'cupy':
        d = detect()
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    if backend == 'cupy':
        import cupy as xp
    else:
        xp = np

    benchmark(
        lambda: for_backend(
            butterworth_line(
                shape=(4096, 4096),
                width=3,
                sb_position=(100.1, 100),
                length_ratio=0.9,
                order=12,
                xp=xp
            ),
            NUMPY
        )
    )


@pytest.mark.benchmark(
    group="filters"
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy'],
)
def test_disk_aperture(backend, benchmark, lt_ctx):
    if backend == 'cupy':
        d = detect()
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    if backend == 'cupy':
        import cupy as xp
    else:
        xp = np

    benchmark(
        lambda: disk_aperture(out_shape=(4096, 4096), radius=128.0, xp=xp),
    )

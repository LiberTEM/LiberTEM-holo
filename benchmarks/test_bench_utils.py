import pytest
import numpy as np

from libertem.utils.devices import detect
from libertem_holo.base.utils import HoloParams


@pytest.mark.benchmark(
    group="utils"
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy'],
)
def test_params_from_hologram(backend, benchmark, lt_ctx, large_holo_data):
    npy_path, ds = large_holo_data
    holo = np.load(str(npy_path), mmap_mode='r')

    if backend == 'cupy':
        d = detect()
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    if backend == 'cupy':
        import cupy as xp
    else:
        xp = np

    benchmark(
        lambda: HoloParams.from_hologram(
            holo[0, 0], central_band_mask_radius=100, xp=xp
        ),
    )

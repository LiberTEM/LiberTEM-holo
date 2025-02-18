import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.utils import HoloParams
from libertem_holo.base.reconstr import get_phase


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_get_phase(backend: str, holo_data) -> None:
    holo, ref, phase_ref, slice_crop = holo_data

    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp
        xp = cp
    else:
        xp = np

    p = HoloParams.from_hologram(
        ref[0, 0],
        central_band_mask_radius=1,
        out_shape=(32, 32),
        line_filter_width=None,
        xp=xp,
    )

    _ = get_phase(holo[0, 0], params=p, xp=xp)

    # TODO: ensure deterministic results for `get_phase`?
    # assert np.allclose(phase_ref[slice_crop][0, 0], phase[...], rtol=0.12)

"""Benchmarks for sideband position estimation."""
# ruff: noqa: INP001
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.generate import hologram_frame
from libertem_holo.base.utils import HoloParams, estimate_sideband_position

SHAPE = (4096, 4096)


def _get_xp(backend: str) -> Any:  # noqa: ANN401
    if backend == "cupy":
        d = detect()
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        import cupy as cp  # noqa: PLC0415
        return cp
    return np


def _synced(fn: Callable[[], Any], xp: Any) -> Callable[[], Any]:  # noqa: ANN401
    """Wait for the GPU to finish, so we time the actual work."""
    if xp is np:
        return fn

    def _inner() -> Any:  # noqa: ANN401
        res = fn()
        xp.cuda.Device().synchronize()
        return res
    return _inner


@pytest.fixture(scope="module")
def holo_4k() -> np.ndarray:
    """Synthetic 4k x 4k hologram with a phase ramp."""
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    phase = np.pi * (xx + yy) / SHAPE[1]
    return hologram_frame(np.ones(SHAPE), phase).astype(np.float32)


@pytest.mark.benchmark(
    group="sideband",
)
@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_estimate_sideband_position(
    backend: str, benchmark: Any, holo_4k: np.ndarray,  # noqa: ANN401
) -> None:
    """Time `estimate_sideband_position`, including the FFT."""
    xp = _get_xp(backend)
    holo = xp.asarray(holo_4k)

    def _run() -> tuple[float, float]:
        return estimate_sideband_position(
            holo, holo_sampling=(1, 1), central_band_mask_radius=100, xp=xp,
        )
    # warm-up outside of the timing, e.g. for JIT compilation:
    _run()
    benchmark(_synced(_run, xp))


@pytest.mark.benchmark(
    group="sideband",
)
@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_holoparams_from_hologram(
    backend: str, benchmark: Any, holo_4k: np.ndarray,  # noqa: ANN401
) -> None:
    """Time `HoloParams.from_hologram`, which includes sideband estimation."""
    xp = _get_xp(backend)
    holo = xp.asarray(holo_4k)

    def _run() -> HoloParams:
        return HoloParams.from_hologram(
            holo, out_shape=(1024, 1024), central_band_mask_radius=100, xp=xp,
        )
    _run()
    benchmark(_synced(_run, xp))

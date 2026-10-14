"""Benchmarks for sharing the hologram FFT between reconstruction calls."""
# ruff: noqa: INP001
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from libertem.utils.devices import detect

from libertem_holo.base.generate import hologram_frame
from libertem_holo.base.reconstr import (
    HologramFFT,
    reconstruct_bf,
    reconstruct_frame,
)
from libertem_holo.base.utils import HoloParams

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
    group="reconstruct_frame",
)
@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
@pytest.mark.parametrize(
    "input_kind", ["array", "pre_fft"],
)
def test_reconstruct_frame(
    backend: str, input_kind: str, benchmark: Any, holo_4k: np.ndarray,  # noqa: ANN401
) -> None:
    """Time `reconstruct_frame` on a plain array vs. a pre-computed FFT."""
    xp = _get_xp(backend)
    params = HoloParams.from_hologram(
        holo_4k, out_shape=(1024, 1024), central_band_mask_radius=100, xp=xp,
    )
    aperture = xp.asarray(params.aperture)
    frame = HologramFFT(holo_4k, xp=xp) if input_kind == "pre_fft" else holo_4k

    benchmark(_synced(lambda: reconstruct_frame(
        frame,
        sb_pos=params.sb_position,
        aperture=aperture,
        slice_fft=params.slice_fft,
        xp=xp,
    ), xp))


@pytest.mark.benchmark(
    group="reconstruct_wave_bf",
)
@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
@pytest.mark.parametrize(
    "shared_fft", [False, True],
)
def test_reconstruct_wave_bf(
    backend: str, shared_fft: bool, benchmark: Any, holo_4k: np.ndarray,  # noqa: ANN401, FBT001
) -> None:
    """Time wave + brightfield from one hologram, with and without shared FFT."""
    xp = _get_xp(backend)
    params = HoloParams.from_hologram(
        holo_4k, out_shape=(1024, 1024), central_band_mask_radius=100, xp=xp,
    )
    aperture = xp.asarray(params.aperture)
    aperture_bf = xp.asarray(params.aperture_bf)

    def _wave_bf() -> tuple[Any, Any]:
        # including the FFT, which is the part we save on:
        frame = HologramFFT(holo_4k, xp=xp) if shared_fft else holo_4k
        wave = reconstruct_frame(
            frame,
            sb_pos=params.sb_position,
            aperture=aperture,
            slice_fft=params.slice_fft,
            xp=xp,
        )
        bf = reconstruct_bf(
            frame, aperture=aperture_bf, slice_fft=params.slice_fft, xp=xp,
        )
        return wave, bf

    benchmark(_synced(_wave_bf, xp))

"""Benchmarks of the pairwise registration building blocks.

The `upsampled_dft` group timed a numba kernel for the separable upsampled
DFT with a fused argmax against the BLAS tensordot: single-threaded, the
tensordot was 2-5x faster at every size (240x232 to 2048x2048, complex64 and
complex128), so the kernel was not adopted and only the tensordot remains.
"""
# ruff: noqa: INP001, S101, SLF001
from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pytest
from libertem.common.threading import set_num_threads

from libertem_holo.base import align
from libertem_holo.base.align import ImageCorrelator, WaveCorrelator

SHAPES = {
    "240x232": (240, 232),  # brightfield images of a 1920x1856 hologram, binned 2
    "512": (512, 512),
    "1024": (1024, 1024),
    "2048": (2048, 2048),
}
UPSAMPLE_FACTOR = 10


@cache
def _images(shape: tuple[int, int], dtype: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a = rng.random(shape).astype(dtype)
    b = np.roll(a, (3, -5), axis=(0, 1))
    return a, b


def _product(shape: tuple[int, int], dtype: str) -> np.ndarray:
    a, b = _images(shape, dtype)
    return np.fft.fftn(a) * np.fft.fftn(b).conj()


@pytest.mark.benchmark(group="upsampled_dft")
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_upsampled_dft(benchmark: Any, shape_name: str, dtype: str) -> None:  # noqa: ANN401
    """One sub-pixel refinement step (BLAS tensordot), single-threaded."""
    shape = SHAPES[shape_name]
    product = _product(shape, dtype)
    region = int(np.ceil(UPSAMPLE_FACTOR * 1.5))
    dftshift = np.fix(region / 2)
    freqs = (
        np.fft.fftfreq(shape[0], UPSAMPLE_FACTOR),
        np.fft.fftfreq(shape[1], UPSAMPLE_FACTOR),
    )
    # b is a rolled by (3, -5), so the shift that aligns b with a is (-3, 5):
    offsets = (dftshift + 3 * UPSAMPLE_FACTOR, dftshift - 5 * UPSAMPLE_FACTOR)
    benchmark.group = f"upsampled_dft {shape_name} {dtype}"

    def run() -> tuple[int, int]:
        cc = align._upsampled_dft(product.conj(), freqs, region, offsets).conj()
        return np.unravel_index(np.argmax(np.abs(cc)), cc.shape)

    with set_num_threads(1):
        peak = run()
        assert peak == (int(dftshift), int(dftshift))
        benchmark.pedantic(run, rounds=20, iterations=1, warmup_rounds=2)


@pytest.mark.benchmark(group="correlate_pair")
@pytest.mark.parametrize("correlator", ["image", "wave"])
@pytest.mark.parametrize("shape_name", ["240x232", "1024"])
def test_correlate_pair(
    benchmark: Any, shape_name: str, correlator: str,  # noqa: ANN401
) -> None:
    """One pair on prepared inputs (cached spectra), single-threaded."""
    shape = SHAPES[shape_name]
    a, b = _images(shape, "float32")
    if correlator == "image":
        corr = ImageCorrelator(upsample_factor=UPSAMPLE_FACTOR, hanning=True, binning=1)
    else:
        corr = WaveCorrelator(upsample_factor=UPSAMPLE_FACTOR)
        a, b = a * np.exp(1j * b), b * np.exp(1j * a)
    pa, pb = corr.prepare_input(a), corr.prepare_input(b)
    benchmark.group = f"correlate_pair {shape_name} {correlator}"
    with set_num_threads(1):
        benchmark.pedantic(
            lambda: corr.correlate(pa, pb), rounds=20, iterations=1, warmup_rounds=2,
        )

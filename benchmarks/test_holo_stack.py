"""End-to-end benchmark of `reconstruct_stack` on a synthetic stack."""
# ruff: noqa: INP001, S101
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from libertem_holo.base.convenience import reconstruct_stack
from libertem_holo.base.generate import hologram_frame
from libertem_holo.base.io import InputData

if TYPE_CHECKING:
    from libertem.api import Context

N_FRAMES = 10
SHAPES = {
    "2k": (2048, 2048),
    "k2/2": (1919, 1855),  # non-smooth, half a K2 frame
}


@cache
def _stack(shape: tuple[int, int]) -> np.ndarray:
    """Synthetic drifting object in a static fringe pattern."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    cy, cx = shape[0] / 2, shape[1] / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # a disk with a soft edge of a few pixels, so that registration has
    # something to lock on to:
    disk = 1 / (1 + np.exp((r - shape[0] / 8) / 2.0))
    frames = np.empty((N_FRAMES, *shape), dtype=np.float32)
    for i in range(N_FRAMES):
        moved = np.roll(disk, (3 * i, -5 * i), axis=(0, 1))
        frames[i] = hologram_frame(1 - 0.3 * moved, 2.0 * moved)
    return frames


@pytest.mark.benchmark(group="reconstruct_stack")
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_reconstruct_stack(
    benchmark: Any, lt_ctx: Context, shape_name: str,  # noqa: ANN401
) -> None:
    """Reconstruct, align and average 10 frames with the libertem backend."""
    shape = SHAPES[shape_name]
    stack = InputData.from_array(_stack(shape), exposure_time=1.0)
    benchmark.group = f"reconstruct_stack {shape_name} {shape}"

    res = benchmark(lambda: reconstruct_stack(
        stack, backend="libertem", ctx=lt_ctx, progress=False,
    ))
    drifts = np.asarray(res.metadata["drifts_x"])
    assert np.abs(drifts).max() > 1

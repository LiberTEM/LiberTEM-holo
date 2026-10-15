"""`reconstruct_stack` on real data with a dask-based LiberTEM `Context`.

Needs `TESTDATA_BASE_PATH` (see `conftest.py`); skipped otherwise. The
30-frame 3838x3710 object and reference stacks are read by the LiberTEM
workers from the `.dm4` files (from the page cache after the warmup round).
Each variant gets a warmup round so that worker startup, imports and numba
compilation are not timed.
"""
# ruff: noqa: INP001, S101
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from libertem.api import Context

from libertem_holo.base.convenience import (
    _reconstruct_waves_udf,
    reconstruct_stack,
)
from libertem_holo.base.io import InputData
from libertem_holo.base.utils import HoloParams

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator

ROUNDS = 5


@pytest.fixture(scope="module")
def dask_cpu_ctx() -> Iterator[Context]:
    """One dask-based CPU context for the whole module."""
    with Context.make_with("dask", gpus=0) as ctx:
        yield ctx


@pytest.fixture
def stacks(dm_testdata_path: pathlib.Path) -> tuple[InputData, InputData]:
    """Object and reference stack, memory-mapped (function scoped)."""
    return (
        InputData.load_from_dm(dm_testdata_path / "holo/reconstr/stack_obj_minus.dm4"),
        InputData.load_from_dm(dm_testdata_path / "holo/reconstr/stack_ref_minus.dm4"),
    )


def _params(stack: InputData) -> HoloParams:
    """Parameters as `_reconstruct_stack` would estimate them."""
    sig_shape = stack.data[0].shape
    return HoloParams.from_hologram(
        stack.data[0],
        out_shape=(sig_shape[0] // 8, sig_shape[1] // 8),
        line_filter_width=2,
        line_filter_length=0.9,
        central_band_mask_radius=100,
    )


@pytest.mark.benchmark(group="data_holoparams")
def test_holoparams_from_hologram(
    benchmark: Any, stacks: tuple[InputData, InputData],  # noqa: ANN401
) -> None:
    """Parameter estimation on the first real frame (sideband search + filters)."""
    stack_obj, _ = stacks
    benchmark.pedantic(
        lambda: _params(stack_obj), rounds=ROUNDS, warmup_rounds=1,
    )


@pytest.mark.benchmark(group="data_reconstruct_waves_udf")
def test_reconstruct_waves_udf(
    benchmark: Any, dask_cpu_ctx: Context, stacks: tuple[InputData, InputData],  # noqa: ANN401
) -> None:
    """Only the UDF run (reading + reconstruction) of the 30-frame object stack."""
    stack_obj, _ = stacks
    params = _params(stack_obj)
    benchmark.pedantic(
        lambda: _reconstruct_waves_udf(
            stack_obj, params, with_bf=True, ctx=dask_cpu_ctx, progress=False,
        ),
        rounds=ROUNDS, warmup_rounds=1,
    )


@pytest.mark.benchmark(group="data_reconstruct_stack")
def test_reconstruct_stack(
    benchmark: Any, dask_cpu_ctx: Context, stacks: tuple[InputData, InputData],  # noqa: ANN401
) -> None:
    """Full `reconstruct_stack` with object and reference stacks (60 frames)."""
    stack_obj, stack_ref = stacks
    params = _params(stack_obj)
    res = benchmark.pedantic(
        lambda: reconstruct_stack(
            stack_obj, stack_ref, holoparams=params, backend="libertem",
            ctx=dask_cpu_ctx, progress=False,
        ),
        rounds=ROUNDS, warmup_rounds=1,
    )
    assert np.abs(np.asarray(res.metadata["drifts_x"])).max() > 0


def test_phase_breakdown(
    dask_cpu_ctx: Context, stacks: tuple[InputData, InputData], capsys: Any,  # noqa: ANN401
) -> None:
    """Wall time per phase of `reconstruct_stack`, printed, not benchmarked."""
    from libertem_holo.base.align import ImageCorrelator  # noqa: PLC0415
    from libertem_holo.base.convenience import _align_waves  # noqa: PLC0415
    from libertem_holo.base.reconstr import phase_offset_correction  # noqa: PLC0415

    stack_obj, stack_ref = stacks
    params = _params(stack_obj)
    timings: dict[str, float] = {}
    for _ in range(2):  # the first pass is the warmup
        t0 = time.perf_counter()
        waves, bfs = _reconstruct_waves_udf(
            stack_obj, params, with_bf=True, ctx=dask_cpu_ctx, progress=False,
        )
        t1 = time.perf_counter()
        correlator = ImageCorrelator(
            hanning=True, normalization="phase", binning=2, upsample_factor=10,
        )
        waves, bfs, _ = _align_waves(waves, bfs, correlator, progress=False)
        t2 = time.perf_counter()
        phase_offset_correction(waves)
        t3 = time.perf_counter()
        _reconstruct_waves_udf(
            stack_ref, params, with_bf=False, ctx=dask_cpu_ctx, progress=False,
        )
        t4 = time.perf_counter()
        timings = {
            "udf obj (30 frames, wave+bf)": t1 - t0,
            "align (30 frames)": t2 - t1,
            "phase offset correction": t3 - t2,
            "udf ref (30 frames, wave)": t4 - t3,
        }
    parts = ", ".join(f"{k}={v:.2f}s" for k, v in timings.items())
    with capsys.disabled():
        print(f"\nPHASES: {parts}")  # noqa: T201

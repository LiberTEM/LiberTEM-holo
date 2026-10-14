"""Compile the CUDA kernels to PTX, without a GPU (marker `cuda_compile`).

Needs NVVM and NVRTC, for example from the `cuda-toolkit` pip wheels, see the
`cuda-compile` tox env. Deselected by default (`pytest.ini`).

What the kernels compute on a GPU is tested by the cupy tests, which skip
without a GPU. There are deliberately no tests in numba's CUDA simulator: it is
slow, and its shared arrays race under coverage.
"""
# the kernels under test are private:
# ruff: noqa: SLF001
import os

import pytest
from numba import cuda, typeof, types

from libertem_holo.base import filters

# In CI, a missing CUDA toolkit is an error, not a reason to skip:
REQUIRE_CUDA_COMPILE = os.environ.get("LIBERTEM_HOLO_REQUIRE_CUDA_COMPILE") == "1"


def _compute_capabilities() -> list[tuple[int, int]]:
    """Lowest and highest compute capability supported by the CUDA toolkit."""
    try:
        from numba.cuda.cudadrv import nvrtc  # noqa: PLC0415
        ccs = nvrtc.get_supported_ccs()
    except Exception as e:  # depends on numba(-cuda) version and toolkit
        if REQUIRE_CUDA_COMPILE:
            raise
        pytest.skip(f"CUDA toolkit not available: {e!r}")
    return sorted({min(ccs), max(ccs)})


def _line_geometry_type() -> types.Type:
    geometry = filters._butterworth_line_geometry(
        shape=(64, 64), target=(40.0, 50.0), length_ratio=0.9,
    )
    return typeof(geometry)


@pytest.mark.cuda_compile
@pytest.mark.parametrize(
    ("kernel", "arg_types"), [
        pytest.param(
            "_butterworth_disk_gpu",
            lambda: (types.float32[:, :], types.float64, types.int64),
            id="disk",
        ),
        pytest.param(
            "_butterworth_line_gpu",
            lambda: (
                types.float32[:, :], _line_geometry_type(), types.float64, types.int64,
            ),
            id="line-float-width",
        ),
        pytest.param(
            "_butterworth_line_gpu",
            lambda: (
                types.float32[:, :], _line_geometry_type(), types.int64, types.int64,
            ),
            id="line-int-width",
        ),
    ],
)
def test_compile_kernel(kernel, arg_types):
    for cc in _compute_capabilities():
        ptx, _ = cuda.compile_ptx(
            getattr(filters, kernel).py_func, arg_types(), cc=cc,
        )
        assert f".target sm_{cc[0]}{cc[1]}" in ptx


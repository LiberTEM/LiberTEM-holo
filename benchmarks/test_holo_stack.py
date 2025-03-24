import pytest
import numpy as np

from libertem.api import Context
from libertem.utils.devices import detect
from libertem.common.backend import set_use_cpu, set_use_cuda

from libertem_holo.base.align import align_stack
from libertem_holo.udf import HoloReconstructUDF


@pytest.fixture(scope="module")
def gpu_ctx():
    d = detect()
    if not d['cudas'] or not d['has_cupy']:
        yield None
    else:
        ctx = Context.make_with(gpus=1, cpus=0)
        try:
            yield ctx
        finally:
            ctx.close()


@pytest.fixture(scope="module")
def cpu_ctx():
    ctx = Context.make_with(gpus=0)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.mark.benchmark(
    group="stack"
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy'],
)
def test_stack_reconstr(backend, benchmark, lt_ctx, large_holo_data):
    path, ds = large_holo_data
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        udf = HoloReconstructUDF.with_default_aperture(
            out_shape=(1024, 1024),
            sb_size=512,
            sb_position=(1024, 1024),
            precision=True,
        )
        benchmark(lt_ctx.run_udf, udf=udf, dataset=ds)
    finally:
        set_use_cpu(0)


@pytest.mark.benchmark(
    group="stack"
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy'],
)
def test_stack_reconstr_parallel(
    backend, benchmark, large_holo_data, gpu_ctx, cpu_ctx
):
    path, _ = large_holo_data
    out_shape = (1024, 1024)
    if backend == 'cupy':
        if gpu_ctx is None:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        ctx = gpu_ctx
    else:
        ctx = cpu_ctx

    ds = ctx.load("npy", str(path))

    with ctx:
        udf = HoloReconstructUDF.with_default_aperture(
            out_shape=out_shape,
            sb_size=out_shape[0] - 32,
            sb_position=(1024, 1024),
            precision=True,
        )
        benchmark(ctx.run_udf, udf=udf, dataset=ds)


@pytest.mark.benchmark(
    group="stack"
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy'],
)
def test_stack_alignment(
    backend, benchmark, large_holo_data, gpu_ctx, cpu_ctx
):
    path, _ = large_holo_data
    out_shape = (1024, 1024)
    if backend == 'cupy':
        if gpu_ctx is None:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
        ctx = gpu_ctx
        import cupy as xp
    else:
        ctx = cpu_ctx
        xp = np

    ds = ctx.load("npy", str(path))

    with ctx:
        udf = HoloReconstructUDF.with_default_aperture(
            out_shape=out_shape,
            sb_size=out_shape[0] - 32,
            sb_position=(1024, 1024),
            precision=True,
        )
        result = ctx.run_udf(dataset=ds, udf=udf)
        wave_stack = result['wave'].data.reshape((-1, 1024, 1024))
        amp_stack = np.abs(wave_stack)

        benchmark(
            align_stack,
            stack=amp_stack,
            wave_stack=wave_stack,
            static=None,
            correlator=None,
            xp=xp,
        )

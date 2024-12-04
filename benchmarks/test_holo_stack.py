import pytest

from libertem.utils.devices import detect
from libertem.udf.sum import SumUDF
from libertem.common.backend import set_use_cpu, set_use_cuda

from libertem_holo.udf import HoloReconstructUDF


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

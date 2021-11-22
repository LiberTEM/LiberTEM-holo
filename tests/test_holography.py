import pytest
import numpy as np

from libertem.io.dataset.memory import MemoryDataSet
from libertem.utils.devices import detect
from libertem.common.backend import set_use_cpu, set_use_cuda

from libertem_holo.udf.reconstr import HoloReconstructUDF
from libertem_holo.base.generate import hologram_frame


@pytest.mark.parametrize(
    # CuPy support deactivated due to https://github.com/LiberTEM/LiberTEM/issues/815
    # 'backend', ['numpy', 'cupy']
    'backend', ['numpy']
)
def test_holo_reconstruction(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    # Prepare image parameters and mesh
    nx, ny = (5, 7)
    sx, sy = (64, 64)
    slice_crop = (slice(None),
                  slice(None),
                  slice(sx // 4, sx // 4 * 3),
                  slice(sy // 4, sy // 4 * 3))

    lnx = np.arange(nx)
    lny = np.arange(ny)
    lsx = np.arange(sx)
    lsy = np.arange(sy)

    mnx, mny, msx, msy = np.meshgrid(lnx, lny, lsx, lsy)

    # Prepare phase image
    phase_ref = np.pi * msx * (mnx.max() - mnx) * mny / sx**2 \
        + np.pi * msy * mnx * (mny.max() - mny) / sy**2

    # Generate holograms
    holo = np.zeros_like(phase_ref)
    ref = np.zeros_like(phase_ref)

    for i in range(nx):
        for j in range(ny):
            holo[j, i, :, :] = hologram_frame(np.ones((sx, sy)), phase_ref[j, i, :, :])
            ref[j, i, :, :] = hologram_frame(np.ones((sx, sy)), np.zeros((sx, sy)))

    # Prepare LT datasets and do reconstruction
    dataset_holo = MemoryDataSet(data=holo, tileshape=(ny, sx, sy),
                                 num_partitions=2, sig_dims=2)

    dataset_ref = MemoryDataSet(data=ref, tileshape=(ny, sx, sy),
                                num_partitions=1, sig_dims=2)

    sb_position = [11, 6]
    sb_size = 6.26498204

    holo_job = HoloReconstructUDF(out_shape=(sx, sy),
                                  sb_position=sb_position,
                                  sb_size=sb_size)
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        w_holo = lt_ctx.run_udf(dataset=dataset_holo, udf=holo_job)['wave'].data
        w_ref = lt_ctx.run_udf(dataset=dataset_ref, udf=holo_job)['wave'].data
    finally:
        set_use_cpu(0)

    w = w_holo / w_ref

    phase = np.angle(w)

    assert np.allclose(phase_ref[slice_crop], phase[slice_crop], rtol=0.12)


def test_poisson_infinity():
    # Make sure that the value with Poisson noise
    # approaches the noiseless case for many counts
    counts = 1000000000  # maximum for np.random.poisson()
    large_counts = hologram_frame(
        amp=np.ones((16, 16)),
        phi=np.zeros((16, 16)),
        counts=counts,
        visibility=0.1,  # weak fringe contrast, enough counts everywhere
        poisson_noise=True
    )
    noiseless = hologram_frame(
        amp=np.ones((16, 16)),
        phi=np.zeros((16, 16)),
        counts=counts,
        visibility=0.1,  # weak fringe contrast, enough counts everywhere
        poisson_noise=False
    )
    # Choose rtol in such a way that test failures are improbable
    assert np.allclose(large_counts, noiseless, rtol=1/np.sqrt(counts) * 50)


@pytest.mark.parametrize(
    'counts', (1000, 10000)
)
def test_poisson_scaling(counts):
    # Make sure that the result with Poisson noise
    # satisfies some properties of a Poisson distribution.

    # Aggregating `counts` frames with dose 1 should correspond
    # to one frame with dose `counts`.
    aggregate = np.sum([
        hologram_frame(
            amp=np.ones((16, 16)),
            phi=np.zeros((16, 16)),
            counts=1,
            visibility=0,  # no fringe contrast, flat full intensity result
            poisson_noise=True
        ) for i in range(counts)
    ], axis=0)

    full = hologram_frame(
        amp=np.ones((16, 16)),
        phi=np.zeros((16, 16)),
        counts=counts,
        visibility=0,  # no fringe contrast, flat full intensity result
        poisson_noise=True
    )
    rtol = 1/np.sqrt(counts)*50
    # See https://en.wikipedia.org/wiki/Poisson_distribution#Descriptive_statistics
    assert np.allclose(np.var(aggregate), counts, rtol=rtol)
    assert np.allclose(np.var(full), counts, rtol=rtol)

    assert np.allclose(np.mean(aggregate), counts, rtol=rtol)
    assert np.allclose(np.mean(full), counts, rtol=rtol)

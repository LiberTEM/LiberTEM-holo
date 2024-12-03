import numpy as np
import pytest
from libertem.api import Context
from libertem.common.backend import set_use_cpu, set_use_cuda
from libertem.io.dataset.memory import MemoryDataSet
from libertem.utils.devices import detect

from libertem_holo.base.generate import hologram_frame
from libertem_holo.base.mask import disk_aperture
from libertem_holo.udf.reconstr import HoloReconstructUDF


@pytest.fixture
def holo_data():
    # Prepare image parameters and mesh
    ny, nx = (5, 7)
    sy, sx = (64, 64)
    slice_crop = (slice(None),
                  slice(None),
                  slice(sy // 4, sy // 4 * 3),
                  slice(sx // 4, sx // 4 * 3))

    lny = np.arange(ny)
    lnx = np.arange(nx)
    lsy = np.arange(sy)
    lsx = np.arange(sx)

    mny, mnx, msy, msx = np.meshgrid(lny, lnx, lsy, lsx)

    # Prepare phase image
    phase_ref = np.pi * msx * (mnx.max() - mnx) * mny / sx**2 \
        + np.pi * msy * mnx * (mny.max() - mny) / sy**2

    # Generate holograms
    holo = np.zeros_like(phase_ref)
    ref = np.zeros_like(phase_ref)

    for i in range(ny):
        for j in range(nx):
            holo[j, i, :, :] = hologram_frame(np.ones((sy, sx)), phase_ref[j, i, :, :])
            ref[j, i, :, :] = hologram_frame(np.ones((sy, sx)), np.zeros((sy, sx)))

    return holo, ref, phase_ref, slice_crop


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_holo_reconstruction(lt_ctx: Context, backend: str, holo_data) -> None:
    holo, ref, phase_ref, slice_crop = holo_data

    if backend == "cupy":
        d = detect()
        cudas = detect()["cudas"]
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    # Prepare LT datasets and do reconstruction
    dataset_holo = MemoryDataSet(data=holo, num_partitions=2, sig_dims=2)

    dataset_ref = MemoryDataSet(data=ref, num_partitions=1, sig_dims=2)

    sb_position = [11, 6]
    sb_size = 6.26498204

    out_shape = dataset_holo.shape.sig
    aperture = disk_aperture(out_shape=out_shape, radius=sb_size)
    holo_udf = HoloReconstructUDF(out_shape=out_shape,
                                  sb_position=sb_position,
                                  aperture=aperture)
    try:
        if backend == "cupy":
            set_use_cuda(cudas[0])
        w_holo = lt_ctx.run_udf(dataset=dataset_holo, udf=holo_udf)["wave"].data
        w_ref = lt_ctx.run_udf(dataset=dataset_ref, udf=holo_udf)["wave"].data
    finally:
        set_use_cpu(0)

    w = w_holo / w_ref

    phase = np.angle(w)

    assert np.allclose(phase_ref[slice_crop], phase[slice_crop], rtol=0.12)


@pytest.mark.parametrize(
    "backend", ["numpy", "cupy"],
)
def test_default_aperture(lt_ctx: Context, backend: str, holo_data) -> None:
    holo, ref, phase_ref, slice_crop = holo_data

    if backend == "cupy":
        d = detect()
        cudas = detect()["cudas"]
        if not d["cudas"] or not d["has_cupy"]:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    # Prepare LT datasets and do reconstruction
    dataset_holo = MemoryDataSet(data=holo, num_partitions=2, sig_dims=2)
    dataset_ref = MemoryDataSet(data=ref, num_partitions=1, sig_dims=2)

    sb_position = [11, 6]
    sb_size = 6.26498204
    out_shape = dataset_holo.shape.sig

    holo_udf = HoloReconstructUDF.with_default_aperture(
        sb_size=sb_size,
        out_shape=out_shape,
        sb_position=sb_position,
    )

    try:
        if backend == "cupy":
            set_use_cuda(cudas[0])
        w_holo = lt_ctx.run_udf(dataset=dataset_holo, udf=holo_udf)["wave"].data
        w_ref = lt_ctx.run_udf(dataset=dataset_ref, udf=holo_udf)["wave"].data
    finally:
        set_use_cpu(0)

    w = w_holo / w_ref

    phase = np.angle(w)

    assert np.allclose(phase_ref[slice_crop], phase[slice_crop], rtol=0.12)

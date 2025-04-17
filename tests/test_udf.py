import numpy as np
import pytest
from libertem.api import Context
from libertem.common.backend import set_use_cpu, set_use_cuda
from libertem.io.dataset.memory import MemoryDataSet
from libertem.utils.devices import detect

from libertem_holo.base.filters import disk_aperture
from libertem_holo.udf.reconstr import HoloReconstructUDF


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
    aperture = np.fft.fftshift(disk_aperture(out_shape=out_shape, radius=sb_size))
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

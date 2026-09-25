import pathlib

import numpy as np
from libertem.api import Context

from libertem_holo.base.io import InputData
from libertem_holo.base.utils import HoloParams
from libertem_holo.udf.reconstr import HoloReconstructUDF


def test_simple_into_libertem_ds(dm_testdata_path: pathlib.Path, lt_ctx: Context):
    path_3d = dm_testdata_path / "3D"
    obj_path = path_3d / "alpha-50_obj.dm3"
    input_data = InputData.load_from_dm(obj_path)
    ds = input_data.into_libertem_dataset(ctx=lt_ctx)

    assert tuple(ds.shape) == (20, 3838, 3710)
    assert ds.dtype == np.dtype("float32")


def test_run_reconstruction_udf_on_ds(
    dm_testdata_path: pathlib.Path, dask_ctx: Context,
):
    path_3d = dm_testdata_path / "3D"
    obj_path = path_3d / "alpha-50_obj.dm3"
    input_data = InputData.load_from_dm(obj_path)
    ds = input_data.into_libertem_dataset(ctx=dask_ctx)

    params = HoloParams.from_hologram(input_data.zslice(0))
    holo_udf = HoloReconstructUDF(
        out_shape=params.out_shape,
        sb_position=params.sb_position,
        aperture=params.aperture,
        aperture_bf=params.aperture_bf,
    )
    res = dask_ctx.run_udf(dataset=ds, udf=holo_udf)
    assert res["wave"].data.shape == (20, *params.out_shape)
    assert res["bf"].data.shape == (20, *params.out_shape)

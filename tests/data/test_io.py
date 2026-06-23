import pathlib

import numpy as np

from libertem_holo.base.io import InputData


def test_load_3d(dm_testdata_path: pathlib.Path):
    path_3d = dm_testdata_path / "3D"
    obj_path = path_3d / "alpha-50_obj.dm3"
    input_data = InputData.load_from_dm(obj_path)

    assert input_data.shape == (20, 3838, 3710)
    assert input_data.dtype == np.dtype("float32")
    assert np.allclose(input_data.data[0].astype("float64").sum(), 833924293.1328387)
    assert np.isclose(input_data.pixelsize, 0.16711573e-9)
    assert input_data.exposure_time == 120.0
    tags = input_data.tags_for_slice(0)
    assert tags is not None
    assert tags["DataBar Device Name"] == "K2-0001"


def test_load_2d(dm_testdata_path: pathlib.Path):
    path_2d = dm_testdata_path
    obj_path = path_2d / "2018-7-17 15_29_0000.dm4"
    input_data = InputData.load_from_dm(obj_path)

    assert input_data.shape == (1, 3838, 3710)
    assert input_data.dtype == np.dtype("float32")
    assert np.allclose(input_data.data[0][0].astype("float64").sum(), 61350.586)
    assert np.isclose(input_data.pixelsize, 0.4504859e-9)
    assert input_data.exposure_time == 2
    tags = input_data.tags_for_slice(0)
    assert tags is not None
    assert tags["DataBar Device Name"] == "K2-0001"


def test_load_from_glob(dm_testdata_path: pathlib.Path):
    input_data = InputData.load_from_glob(
        base_path=dm_testdata_path,
        pattern="2018-7-17 15_29_*.dm4",
    )
    assert input_data.shape == (10, 3838, 3710)
    assert input_data.dtype == np.dtype("float32")
    assert np.isclose(input_data.pixelsize, 0.4504859e-9)
    assert input_data.exposure_time == 20.0
    tags = input_data.tags_for_slice(9)
    assert tags is not None
    assert tags["DataBar Device Name"] == "K2-0001"

    assert input_data.data[7].shape == (3838, 3710)


def test_pixelsize_micrometer(dm_testdata_path: pathlib.Path):
    # test data is some spectrum STEM data, but has µm pixel size:
    input_data = InputData.load_from_dm(
        dm_testdata_path / "ADF Image.dm4",
    )
    assert input_data.shape == (1, 6, 31)
    assert input_data.dtype == np.dtype("float32")
    assert np.isclose(input_data.pixelsize, 0.0010295984e-6)
    tags = input_data.tags_for_slice(0)
    assert tags is not None


def test_dm3_data(dm_testdata_path: pathlib.Path):
    input_data = InputData.load_from_dm(
        dm_testdata_path / "NF100_fil10_04_malika.dm3",
    )
    assert input_data.shape == (1, 2048, 2048)
    assert input_data.dtype == np.dtype("int32")
    assert np.isclose(input_data.pixelsize, 0.4504859e-9)
    assert input_data.exposure_time == 1.0
    tags = input_data.tags_for_slice(0)
    assert tags is not None
    assert tags["DataBar Device Name"] == "BM-UltraScan"

    assert input_data.data[0].shape == (2048, 2048)

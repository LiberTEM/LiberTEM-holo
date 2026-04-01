import numpy as np
from libertem_holo.base.io import InputData


def test_load_3d(dm_testdata_path):
    path_3d = dm_testdata_path / '3D'
    obj_path = path_3d / 'alpha-50_obj.dm3'
    input_data = InputData.load_from_dm(obj_path)

    assert input_data.data.shape == (20, 3838, 3710)
    assert input_data.data.dtype == np.dtype('float32')
    assert input_data.data[0].sum() == 833924288.0
    assert input_data.pixelsize == 0.16711573
    assert input_data.exposure_time == 120.0
    assert input_data.tags['DataBar Device Name'] == 'K2-0001'


def test_load_2d(dm_testdata_path):
    path_2d = dm_testdata_path
    obj_path = path_2d / '2018-7-17 15_29_0000.dm4'
    input_data = InputData.load_from_dm(obj_path)

    assert input_data.data.shape == (3838, 3710)
    assert input_data.data.dtype == np.dtype('float32')
    assert input_data.data[0].sum() == 61350.586
    assert input_data.pixelsize == 0.4504859
    assert input_data.exposure_time == 2
    assert input_data.tags['DataBar Device Name'] == 'K2-0001'

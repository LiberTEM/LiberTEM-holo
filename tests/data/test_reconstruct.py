import pytest
from libertem_holo.base.io import InputData
from libertem_holo.base.utils import HoloParams
from libertem_holo.base.reconstr import reconstruct_frame


@pytest.fixture
def dm_3d(dm_testdata_path):
    path_3d = dm_testdata_path / '3D'
    obj_path = path_3d / 'alpha-50_obj.dm3'
    yield InputData.load_from_dm(obj_path)


def test_reconstruct_single(dm_3d):
    params = HoloParams.from_hologram(
        dm_3d.data[0],
        out_shape=(256, 256),
    )
    _ = reconstruct_frame(
        frame=dm_3d.data[0],
        sb_pos=params.sb_position,
        aperture=params.aperture,
        slice_fft=params.slice_fft,
    )

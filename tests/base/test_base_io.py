import numpy as np

from libertem_holo.base.io import Results, InputData
from libertem_holo.base.utils import HoloParams


def test_input_data_from_array_optional_fields():
    # an InputData object can be constructed from just an array:
    arr = np.random.random((2, 128, 128))
    inp = InputData.from_array(data=arr)
    assert np.allclose(arr[0], inp.data[0])


def test_result_metadata_from_input_data(tmp_path, holo_data):
    holo, ref, phase_ref, slice_crop = holo_data
    holo = holo.reshape((-1, 64, 64))
    inp = InputData.from_array(
        data=holo,
        exposure_time=24.6,
        pixelsize=0.106725,
        tags={
            'DataBar Acquisition Time (OS)': np.uint64(133857343696094653),
        },
    )
    params = HoloParams.from_hologram(
        holo[0],
        out_shape=(holo.shape[1] // 4, holo.shape[2] // 4),
    )
    data = np.random.random((128, 128)) + 1j * np.random.random((128, 128))
    phase = np.random.random((128, 128))
    brightfield = np.random.random((128, 128))
    res = Results(
        complex_wave=data,
        unwrapped_phase=phase,
        brightfield=brightfield,
        metadata={"stuff": 6.54},
    )
    res.metadata_from_input(
        input_data=inp,
        params=params,
    )
    assert res.metadata['stack_shape'] == [35, 64, 64]
    assert res.metadata['exposure_time'] == 24.6

    # pixel size is 4 times larger than the input pixel size, as that
    # is our out_shape relative to our input data:
    assert res.metadata['effective_pixelsize'] == 0.4269
    assert res.metadata['acquisition_timestamp'] == '2025-03-06T11:32:49.609465+00:00'

    # and our custom metadata survived:
    assert res.metadata['stuff'] == 6.54

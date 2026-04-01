import numpy as np

from libertem_holo.base.io import Results, InputData
from libertem_holo.base.utils import HoloParams


def test_save_results(tmp_path):
    res = Results(
        complex_wave=np.zeros((128, 128), dtype=np.complex128),
        metadata={"stuff": 6.54},
    )
    res.save(tmp_path / "test1.npz")


def test_no_metadata(tmp_path):
    data = np.random.random((128, 128)) + 1j * np.random.random((128, 128))
    res = Results(
        complex_wave=data,
    )
    res.save(tmp_path / "test1.npz")

    res_l = Results.load(tmp_path / "test1.npz")
    assert np.allclose(res.complex_wave, res_l.complex_wave)
    assert res.metadata is None


def test_save_roudtrip(tmp_path):
    data = np.random.random((128, 128)) + 1j * np.random.random((128, 128))
    res = Results(
        complex_wave=data,
        metadata={"stuff": 6.54},
    )
    res.save(tmp_path / "test1.npz")

    res_l = Results.load(tmp_path / "test1.npz")
    assert np.allclose(res.complex_wave, res_l.complex_wave)
    assert res.metadata == res_l.metadata


def test_optional_fields(tmp_path):
    data = np.random.random((128, 128)) + 1j * np.random.random((128, 128))
    phase = np.random.random((128, 128))
    brightfield = np.random.random((128, 128))
    res = Results(
        complex_wave=data,
        unwrapped_phase=phase,
        brightfield=brightfield,
        metadata={"stuff": 6.54},
    )
    res.save(tmp_path / "test1.npz")

    res_l = Results.load(tmp_path / "test1.npz")
    assert np.allclose(res.complex_wave, res_l.complex_wave)
    assert np.allclose(res.unwrapped_phase, res_l.unwrapped_phase)
    assert np.allclose(res.brightfield, res_l.brightfield)
    assert res.metadata == res_l.metadata


def test_input_data_from_array_optional_fields():
    # an InputData object can be constructed from just an array:
    arr = np.random.random((2, 128, 128))
    inp = InputData(data=arr)
    assert np.allclose(arr, inp.data)


def test_result_metadata_from_input_data(tmp_path, holo_data):
    holo, ref, phase_ref, slice_crop = holo_data
    holo = holo.reshape((-1, 64, 64))
    inp = InputData(
        data=holo,
        exposure_time=24.6,
        pixelsize=0.106725,
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

    # and our custom metadata survived:
    assert res.metadata['stuff'] == 6.54

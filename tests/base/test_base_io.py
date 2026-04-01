import numpy as np

from libertem_holo.base.io import Results, InputData


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

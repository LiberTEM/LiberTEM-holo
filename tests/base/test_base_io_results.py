import pathlib

import numpy as np

from libertem_holo.base.io import Results


def test_save_results(tmp_path: pathlib.Path):
    res = Results(
        complex_wave=np.zeros((128, 128), dtype=np.complex128),
        metadata={"stuff": 6.54},
    )
    res.save(tmp_path / "test1.npz")


def test_no_metadata(tmp_path: pathlib.Path):
    rng = np.random.default_rng()
    data = rng.random((128, 128)) + 1j * rng.random((128, 128))
    res = Results(
        complex_wave=data,
    )
    res.save(tmp_path / "test1.npz")

    res_l = Results.load(tmp_path / "test1.npz")
    assert np.allclose(res.complex_wave, res_l.complex_wave)
    assert res.metadata is None


def test_save_roudtrip(tmp_path: pathlib.Path):
    rng = np.random.default_rng()
    data = rng.random((128, 128)) + 1j * rng.random((128, 128))
    res = Results(
        complex_wave=data,
        metadata={"stuff": 6.54},
    )
    res.save(tmp_path / "test1.npz")

    res_l = Results.load(tmp_path / "test1.npz")
    assert np.allclose(res.complex_wave, res_l.complex_wave)
    assert res.metadata == res_l.metadata


def test_optional_fields(tmp_path: pathlib.Path):
    rng = np.random.default_rng()
    wave = rng.random((128, 128)) + 1j * rng.random((128, 128))
    phase = rng.random((128, 128))
    brightfield = rng.random((128, 128))
    res = Results(
        complex_wave=wave,
        unwrapped_phase=phase,
        brightfield=brightfield,
        metadata={"stuff": 6.54},
    )
    res.save(tmp_path / "test1.npz")

    res_l = Results.load(tmp_path / "test1.npz")
    assert np.allclose(res.complex_wave, res_l.complex_wave)
    assert res.unwrapped_phase is not None
    assert res_l.unwrapped_phase is not None
    assert np.allclose(res.unwrapped_phase, res_l.unwrapped_phase)
    assert res.brightfield is not None
    assert res_l.brightfield is not None
    assert np.allclose(res.brightfield, res_l.brightfield)
    assert res.metadata == res_l.metadata

import numpy as np
import pytest
from libertem.io.dataset.memory import MemoryDataSet

from libertem_holo.base.io import InputData, Results
from libertem_holo.base.utils import HoloParams


def test_input_data_from_array_optional_fields():
    rng = np.random.default_rng()
    # an InputData object can be constructed from just an array:
    arr = rng.random((2, 128, 128))
    inp = InputData.from_array(data=arr)
    assert np.allclose(arr[0], inp.data[0])


def test_result_metadata_from_input_data(holo_data):
    rng = np.random.default_rng()
    holo, ref, phase_ref, slice_crop = holo_data
    holo = holo.reshape((-1, 64, 64))
    inp = InputData.from_array(
        data=holo,
        exposure_time=24.6,
        pixelsize=0.106725,
        tags={
            "DataBar Acquisition Time (OS)": np.uint64(133857343696094653),
        },
    )
    params = HoloParams.from_hologram(
        holo[0],
        out_shape=(holo.shape[1] // 4, holo.shape[2] // 4),
    )
    data = rng.random((128, 128)) + 1j * rng.random((128, 128))
    phase = rng.random((128, 128))
    brightfield = rng.random((128, 128))
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
    assert res.metadata is not None
    assert res.metadata["stack_shape"] == [35, 64, 64]
    assert res.metadata["exposure_time"] == 24.6

    # pixel size is 4 times larger than the input pixel size, as that
    # is our out_shape relative to our input data:
    assert res.metadata["effective_pixelsize"] == 0.4269
    assert res.metadata["acquisition_timestamp"] == "2025-03-06T11:32:49.609465+00:00"

    # and our custom metadata survived:
    assert res.metadata["stuff"] == 6.54


@pytest.mark.parametrize("shape", [(6, 16, 16), (2, 3, 16, 16)])
def test_input_data_from_libertem_dataset(shape):
    rng = np.random.default_rng()
    arr = rng.random(shape).astype(np.float32)
    ds = MemoryDataSet(data=arr, num_partitions=2, sig_dims=2)
    inp = InputData.load_from_libertem_dataset(
        ds, pixelsize=1e-9, exposure_time=12.0, tags={"a": 1},
    )
    flat = arr.reshape((-1, 16, 16))
    assert inp.shape == flat.shape
    assert inp.dtype == np.dtype(np.float32)
    assert inp.pixelsize == 1e-9
    assert inp.exposure_time == 12.0
    assert inp.tags_for_slice(0) == {"a": 1}
    for z in range(flat.shape[0]):
        assert np.array_equal(inp.zslice(z), flat[z])
        assert np.array_equal(inp.data[z], flat[z])
    with pytest.raises(ValueError, match="out of bounds"):
        inp.zslice(flat.shape[0])


def test_into_libertem_dataset_mixed_sources(lt_ctx):
    arr = np.zeros((2, 16, 16), dtype=np.float32)
    ds = MemoryDataSet(data=arr, num_partitions=1, sig_dims=2)
    from_ds = InputData.load_from_libertem_dataset(ds)
    assert from_ds.into_libertem_dataset(ctx=lt_ctx) is ds
    mixed = InputData(files=[*from_ds.files, *InputData.from_array(arr).files])
    with pytest.raises(ValueError, match="can't combine"):
        mixed.into_libertem_dataset(ctx=lt_ctx)

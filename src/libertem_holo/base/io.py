"""
Basic I/O for holography data.

We mostly want to support loading holograms from DM{3,4} files, and save
results as numpy .npz files
"""

import numpy as np
import json
from typing import Any, NamedTuple
from ncmepy.io.dm import fileDM


def save_results(
    filename: str,
    complex_wave: np.ndarray,
    unwrapped_phase: np.ndarray | None = None,
    brightfield: np.ndarray | None = None,
    metadata: dict[str, Any] = None,
):
    """
    Save a typical reconstruction result.

    Parameters
    ----------
    complex_wave
        2D numpy array (dtype complex64 or complex128)

    unwrapped_phase
        2D numpy array of unwrapped phase (dtype float32 or float64)

    brightfield
        2D numpy array of a brightfield reconstruction from the centerband
        (dtype float32 or float64)

    metadata
        Dictionary of custom metadata. The values have to be json-serializable
        (roughly numbers, strings, lists or dicts of these)
    """
    if metadata is None:
        metadata = {}
    arrays = {
        'complex_wave': complex_wave,
        'metadata': json.dumps(metadata),
    }
    if unwrapped_phase is not None:
        arrays['unwrapped_phase'] = unwrapped_phase
    if brightfield is not None:
        arrays['brightfield'] = brightfield
    np.savez(filename, **arrays, allow_pickle=False)


class InputData(NamedTuple):
    """
    2D or 3D input data (holograms)
    """
    data: np.ndarray

    # in nm
    pixelsize: float | None

    # raw tags from DM
    tags: dict[str, Any] | None

    # in seconds, for the whole stack in the 3D case
    exposure_time: float | None

    @classmethod
    def load_from_dm(cls, filename) -> "InputData":
        dm = fileDM(filename)
        ds = dm.getDataset(0)
        assert ds['pixelUnit'] == 'nm'
        assert ds['pixelSize'][0] == ds['pixelSize'][1]
        pixelsize = ds['pixelSize'][0]
        assert len(ds['data'].shape) in (2, 3), "data should be 2D or 3D"
        exposure_time = dm.getMetadata(0)['DataBar Exposure Time (s)']
        if len(ds['data'].shape) == 3:
            exposure_time *= ds['data'].shape[0]
        return cls(
            data=ds['data'],
            pixelsize=pixelsize,
            tags=dm.getMetadata(0),
            exposure_time=exposure_time,
        )

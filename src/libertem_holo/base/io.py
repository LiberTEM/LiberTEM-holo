"""
Basic I/O for holography data.

We mostly want to support loading holograms from DM{3,4} files, and save
results as numpy .npz files
"""

import numpy as np
import json
from typing import Any
from ncempy.io.dm import fileDM
from dataclasses import dataclass


@dataclass
class Results:
    """Reconstruction results

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
    complex_wave: np.ndarray
    unwrapped_phase: np.ndarray | None = None
    brightfield: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def save(self, filename: str):
        assert str(filename).endswith(".npz")
        if self.metadata is None:
            metadata = {}
        else:
            metadata = self.metadata
        arrays = {
            'complex_wave': self.complex_wave,
            'metadata': json.dumps(metadata),
        }
        if self.unwrapped_phase is not None:
            arrays['unwrapped_phase'] = self.unwrapped_phase
        if self.brightfield is not None:
            arrays['brightfield'] = self.brightfield
        np.savez(filename, **arrays, allow_pickle=False)

    @classmethod
    def load(cls, filename: str) -> "Results":
        arrz = np.load(filename, allow_pickle=False)
        kwargs = {}
        for name in ['complex_wave', 'unwrapped_phase', 'brightfield']:
            kwargs[name] = arrz.get(name)
        kwargs['metadata'] = json.loads(str(arrz['metadata']))
        return cls(**kwargs)


@dataclass
class InputData:
    """
    2D or 3D input data (holograms)
    """
    data: np.ndarray

    # in nm
    pixelsize: float | None = None

    # raw tags from DM
    tags: dict[str, Any] | None = None

    # in seconds, for the whole stack in the 3D case
    exposure_time: float | None = None

    @classmethod
    def load_from_dm(cls, filename) -> "InputData":
        """
        Load .dm3 or .dm4 data. Assumes a single 2D or 3D data set per file.
        """
        dm = fileDM(filename)
        ds = dm.getDataset(0)

        # [z, y, x] in 3D case, but we don't care about z
        units = ds['pixelUnit'][-2:]
        sizes = ds['pixelSize'][-2:]

        assert units[0] == 'nm', f'pixelUnit should be nm, is {ds["pixelUnit"]}'
        assert sizes[0] == sizes[1]
        pixelsize = sizes[0]
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

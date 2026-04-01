from __future__ import annotations
"""
Basic I/O for holography data.

We mostly want to support loading holograms from DM{3,4} files, and save
results as numpy .npz files
"""

import json
import pathlib
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass
import datetime

import numpy as np
from ncempy.io.dm import fileDM

if TYPE_CHECKING:
    from libertem_holo.base.utils import HoloParams


# inspired by https://stackoverflow.com/a/38880683/540644
def dt_from_filetime(ft):
    """
    Convert a windows FILETIME to a datetime
    """
    EPOCH_AS_FILETIME = 116444736000000000
    us = (ft - EPOCH_AS_FILETIME) // 10
    return datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=us)


@dataclass
class Results:
    """Reconstruction results

    Parameters
    ----------
    complex_wave
        the (averaged) complex wave as 2D numpy array (dtype complex64 or
        complex128)

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

    def metadata_from_input(
        self,
        input_data: "InputData",
        params: HoloParams | None = None,
    ):
        """Update `metadata` from `input_data`.

        The following keys will be set:
         - `stack_shape`
         - `exposure_time`
         - `effective_pixelsize` if `params` are given and `input_data` has a pixel size
         - `acquisition_timestamp` if available in the input data tags

        Parameters
        ----------
        input_data
            The input hologram or hologram stack

        params
            The :class:`HoloParams` used for the reconstruction
        """
        self.metadata['stack_shape'] = list(input_data.data.shape)
        self.metadata['exposure_time'] = float(input_data.exposure_time)
        if params is not None and input_data.pixelsize is not None:
            pxs = input_data.pixelsize / params.scale_factor
            self.metadata['effective_pixelsize'] = pxs
        if input_data.tags is not None:
            ft = input_data.tags.get('DataBar Acquisition Time (OS)')
            if ft is not None:
                ft = int(ft)
                dt = dt_from_filetime(ft)
                self.metadata['acquisition_timestamp'] = dt.isoformat()

    def save(
        self,
        path: str | pathlib.Path,
    ):
        """Save result data as npz file.

        Parameters
        ----------
        path
            The path to the .npz file that will be created
        """
        assert str(path).endswith(".npz")

        arrays = {
            'complex_wave': self.complex_wave,
            'metadata': json.dumps(self.metadata or {}),
        }
        if self.unwrapped_phase is not None:
            arrays['unwrapped_phase'] = self.unwrapped_phase
        if self.brightfield is not None:
            arrays['brightfield'] = self.brightfield
        np.savez(path, **arrays, allow_pickle=False)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Results":
        arrz = np.load(path, allow_pickle=False)
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
    """the data array"""

    pixelsize: float | None = None
    """in nm"""

    tags: dict[str, Any] | None = None
    """raw tags from the DM file"""

    exposure_time: float | None = None
    """in seconds, for the whole stack in the 3D case"""

    @classmethod
    def load_from_dm(cls, path) -> "InputData":
        """
        Load .dm3 or .dm4 data. Assumes a single 2D or 3D data set per file.
        """
        dm = fileDM(path)
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

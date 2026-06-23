"""Classes for saving and loading results.

Can take metadata from the input, but also supports
custom metadata fields.
"""

from __future__ import annotations

import json
import pathlib
import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from libertem_holo.base.utils import HoloParams
from libertem_holo.base.io.reader import InputData

if TYPE_CHECKING:
    pass


@dataclass
class Results:
    """Reconstruction results.

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
        input_data: InputData,
        params: HoloParams | None = None,
    ) -> None:
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
        if self.metadata is None:
            self.metadata = {}
        self.metadata["stack_shape"] = list(input_data.shape)
        self.metadata["exposure_time"] = float(input_data.exposure_time)
        if params is not None and input_data.pixelsize is not None:
            pxs = input_data.pixelsize / params.scale_factor
            self.metadata["effective_pixelsize"] = pxs
        tags = input_data.tags_for_slice(0)
        if tags is not None:
            ft = tags.get("DataBar Acquisition Time (OS)")
            if ft is not None:
                ft = int(ft)
                dt = dt_from_filetime(ft)
                self.metadata["acquisition_timestamp"] = dt.isoformat()

    def save(
        self,
        path: str | pathlib.Path,
    ) -> None:
        """Save result data as npz file.

        Parameters
        ----------
        path
            The path to the .npz file that will be created

        """
        if not str(path).endswith(".npz"):
            msg = "path should have an .npz file extension"
            raise ValueError(msg)

        arrays = {
            "complex_wave": self.complex_wave,
            "metadata": json.dumps(self.metadata or {}),
        }
        if self.unwrapped_phase is not None:
            arrays["unwrapped_phase"] = self.unwrapped_phase
        if self.brightfield is not None:
            arrays["brightfield"] = self.brightfield
        np.savez(path, **arrays, allow_pickle=False)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Results:
        """Load a result that was previously saved using the `save` method."""
        arrz = cast("dict[str, np.ndarray]", np.load(path, allow_pickle=False))
        kwargs = {}
        for name in ["complex_wave", "unwrapped_phase", "brightfield"]:
            kwargs[name] = arrz.get(name)
        kwargs["metadata"] = json.loads(str(arrz["metadata"]))
        return cls(**kwargs)


# inspired by https://stackoverflow.com/a/38880683/540644
def dt_from_filetime(ft: int) -> datetime.datetime:
    """Convert a windows FILETIME to a datetime."""
    epoch_as_filetime = 116444736000000000
    us = (ft - epoch_as_filetime) // 10
    return datetime.datetime(
        1970,
        1,
        1,
        tzinfo=datetime.timezone.utc,
    ) + datetime.timedelta(microseconds=us)

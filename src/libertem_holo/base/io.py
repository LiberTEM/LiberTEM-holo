"""Basic I/O for holography data.

We mostly want to support loading holograms from DM{3,4} files, and save
results as numpy .npz files
"""

from __future__ import annotations

import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import natsort
import numpy as np
from ncempy.io.dm import fileDM

if TYPE_CHECKING:
    from collections.abc import Sequence

    from libertem_holo.base.utils import HoloParams


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
        if input_data.tags is not None:
            ft = input_data.tags.get("DataBar Acquisition Time (OS)")
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


class InputSlicer:
    def __init__(self, data: InputData):
        self._data = data

    def __getitem__(self, z: int) -> np.ndarray:
        """Return slice `z` of the input data."""
        return self._data.zslice(z)


@dataclass
class InputData:
    """Input data from one or more files."""

    files: list[InputFile]
    """ordered list of input files"""

    def zslice(self, z: int) -> np.ndarray:
        """From a 3D stack, load a single image/slice."""
        # translate z into two indices:
        # 1) the correct InputFile in self.files
        # 2) the index in that file
        raise NotImplementedError

    def tags_for_slice(self, z: int) -> dict[str, Any]:
        # which InputFile does z lie in? return its tags
        raise NotImplementedError

    @property
    def data(self) -> InputSlicer:
        """Access slices of the input data.

        Example:
        -------
        >>> i = InputData.load_from_dm(path="something.dm4")
        >>> i.data[8]  # return the 8th slice of the input data
        array([...])

        """
        return InputSlicer(self)

    @property
    def shape(self) -> list[int]:
        raise NotImplementedError

    @property
    def exposure_time(self) -> float:
        """Exposure time, in seconds.

        In case of a stack, this is the sum of all exposure times.
        """
        exp_sum = 0.0
        for in_file in self.files:
            if in_file.exposure_time is None:
                path = in_file.path
                msg = (
                    f"At least one of the input files ({path})"
                    " has no defined exposure time"
                )
                raise ValueError(msg)
            exp_sum += in_file.exposure_time
        return exp_sum

    @classmethod
    def load_from_dm(cls, path: str | pathlib.Path) -> InputData:
        """Load .dm3 or .dm4. Assumes a single 2D or 3D data set per file."""
        return cls(files=[InputFile.load_from_dm(path)])

    @classmethod
    def load_from_list(cls, paths: Sequence[str | pathlib.Path]) -> InputData:
        """Load from an ordered list of .dm3 or .dm4 files.

        Each file should contain a single 2D image.
        """
        return cls(files=[
            InputFile.load_from_dm(path)
            for path in paths
        ])

    @classmethod
    def load_from_glob(
        cls,
        *,
        base_path: str | pathlib.Path,
        pattern: str,
        sort_files: bool = True,
    ) -> InputData:
        """Load from a glob (for example `stack_1/image_*.dm4`).

        Parameters
        ----------
        base_path
            path prefix where the data is stored

        pattern
            glob pattern to match against

        sort_files
            Whether the list of files should be sorted naturally
            (using natsort) - if disabled, files will be read in
            an undefined order.

        Example
        -------
        >>> InputData.load_from_glob(base_path="/data/stack-1/", pattern="*.dm4")

        """
        base_path = pathlib.Path(base_path)
        paths = list(base_path.glob(pattern))
        if sort_files:
            paths = natsort.natsorted(paths)
        return cls.load_from_list(paths)


@dataclass
class InputFile:
    """2D or 3D input data from a single file (holograms)."""

    data: np.ndarray
    """the data array"""

    path: pathlib.Path
    """the path to the file on the filesystem"""

    pixelsize: float | None = None
    """in nm"""

    tags: dict[str, Any] | None = None
    """raw tags from the DM file"""

    exposure_time: float | None = None
    """in seconds, for the whole stack in the 3D case"""

    @classmethod
    def load_from_dm(cls, path: str | pathlib.Path) -> InputFile:
        """Load .dm3 or .dm4 data. Assumes a single 2D or 3D data set per file."""
        dm = fileDM(path)
        ds = dm.getDataset(0)

        # [z, y, x] in 3D case, but we don't care about z
        units = ds["pixelUnit"][-2:]
        sizes = ds["pixelSize"][-2:]

        assert units[0] == "nm", f'pixelUnit should be nm, is {ds["pixelUnit"]}'
        assert sizes[0] == sizes[1]
        pixelsize = sizes[0]
        assert len(ds["data"].shape) in (2, 3), "data should be 2D or 3D"
        exposure_time = dm.getMetadata(0)["DataBar Exposure Time (s)"]
        if len(ds["data"].shape) == 3:
            exposure_time *= ds["data"].shape[0]
        return cls(
            data=ds["data"],
            pixelsize=pixelsize,
            tags=dm.getMetadata(0),
            exposure_time=exposure_time,
            path=pathlib.Path(path),
        )

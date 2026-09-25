"""Data loading support.

Use cases:

- Load a stack from a single dm3/dm4 file
- Load a stack from a list or a glob of files
- Construct InputData manually (potentially with missing parts)
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import natsort
import numpy as np
from ncempy.io.dm import fileDM

if TYPE_CHECKING:
    from collections.abc import Sequence
    from libertem.api import Context
    from libertem.io.dataset import DataSet


class InputSlicer:
    def __init__(self, data: InputData):
        self._data = data

    def __getitem__(self, z: int) -> np.ndarray:
        """Return slice `z` of the input data."""
        return self._data.zslice(z)

    def __len__(self) -> int:
        """Return number of slices of the input data."""
        return self._data.shape[0]


@dataclass
class InputData:
    """Input data from one or more files."""

    files: list[InputFile]
    """ordered list of input files"""

    @property
    def pixelsize(self) -> float | None:
        """Pixel size in m."""
        return self.files[0].pixelsize

    def zslice(self, z: int) -> np.ndarray:
        """From a 3D stack, load a single image/slice."""
        # translate z into two indices:
        # 1) the correct InputFile in self.files
        # 2) the index in that file
        offset = 0
        for i_f in self.files:
            if z >= offset and z < offset + i_f.shape_3d[0]:
                in_file_offset = z - offset
                return i_f.data_3d[in_file_offset]
            offset += i_f.shape_3d[0]
        msg = f"z out of bounds: {z}"
        raise ValueError(msg)

    def tags_for_slice(self, z: int) -> dict[str, Any] | None:
        """Raw dictionary of tags for slize `z`."""
        # which InputFile does z lie in? return its tags
        i_f = self._file_for_z(z)
        return i_f.tags

    def _file_for_z(self, z: int) -> InputFile:
        offset = 0
        for i_f in self.files:
            if z >= offset and z < offset + i_f.shape_3d[0]:
                return i_f
            offset += i_f.shape_3d[0]
        msg = f"z out of bounds: {z}"
        raise ValueError(msg)

    @property
    def data(self) -> InputSlicer:
        """Access slices of the input data.

        Example:
        --------
        >>> i = InputData.load_from_dm(path="something.dm4")  # doctest: +SKIP
        >>> i.data[8]  # return the 8th slice of the input data  # doctest: +SKIP
        array([...])

        """
        return InputSlicer(self)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The 3D shape of the data."""
        shapes = [
            file.shape_3d
            for file in self.files
        ]
        sig_shape = shapes[0][1:]
        nav_shape = sum(s[0] for s in shapes)
        return (nav_shape,  *sig_shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        """The numpy dtype of the data."""
        return self.files[0].data.dtype

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

    def into_libertem_dataset(self, ctx: Context) -> DataSet:
        """Turn this InputData into a LiberTEM dataset usable for running a UDF."""
        if len(self.files) == 1:
            # single-file dm dataset
            ds = ctx.load("dm", path=self.files[0].path)
        else:
            ds = ctx.load("fm", files=[f.path for f in self.files])
        return ds

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        pixelsize: float | None = None,
        exposure_time: float | None = None,
        tags: dict[str, Any] | None = None,
    ) -> InputData:
        """Create InputData from an array."""
        i_f = InputFile.from_array(
            data=data,
            pixelsize=pixelsize,
            exposure_time=exposure_time,
            tags=tags,
        )
        return InputData(
            files=[i_f],
        )

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
        >>> InputData.load_from_glob(
        ...     base_path="/data/stack-1/",
        ...     pattern="*.dm4")  # doctest: +SKIP

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

    path: pathlib.Path | None = None
    """the path to the file on the filesystem"""

    pixelsize: float | None = None
    """in m"""

    tags: dict[str, Any] | None = None
    """raw tags from the DM file"""

    exposure_time: float | None = None
    """in seconds, for the whole stack in the 3D case"""

    @property
    def data_3d(self) -> np.ndarray:
        """Data as a 3D shape."""
        return self.data.reshape(self.shape_3d)

    @property
    def shape_3d(self) -> tuple[int, int, int]:
        """Shape as 3D.

        In the 2D case, the first dimension will have shape 1.
        """
        shape = self.data.shape
        if len(shape) == 3:
            return shape
        if len(shape) == 2:
            return (1, *tuple(shape))
        msg = f"shape should be 2d or 3d; is {shape}"
        raise ValueError(msg)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        pixelsize: float | None = None,
        exposure_time: float | None = None,
        tags: dict[str, Any] | None = None,
    ) -> InputFile:
        """Create InputFile from an array."""
        return cls(
            data=data,
            pixelsize=pixelsize,
            exposure_time=exposure_time,
            tags=tags,
        )

    @classmethod
    def load_from_dm(cls, path: str | pathlib.Path) -> InputFile:
        """Load .dm3 or .dm4 data. Assumes a single 2D or 3D data set per file."""
        dm = fileDM(path)
        data = dm.getMemmap(0)
        tags = dm.getMetadata(0)

        units, scales = _pixel_scale_and_units(tags)

        # [z, y, x] in 3D case, but we don't care about z
        units = units[-2:]
        scales = scales[-2:]

        if scales[0] != scales[1]:
            msg = "pixel size should be the same for both axes"
            raise ValueError(msg)
        pixelsize = scales[0]
        if units[0] == "nm":
            pix_mult = 1e-9
        elif units[0] == "µm":
            pix_mult = 1e-6
        else:
            msg = (
                'pixelUnit should be nm or µm,'
                f' is {units}'
            )
            raise ValueError(msg)
        pixelsize = float(pixelsize) * pix_mult

        if len(data.shape) not in (2, 3):
            msg = "data should be 2D or 3D"
            raise ValueError(msg)

        exposure_time = tags.get("DataBar Exposure Time (s)")
        if len(data.shape) == 3:
            exposure_time *= data.shape[0]
        return cls(
            data=data,
            pixelsize=pixelsize,
            tags=tags,
            exposure_time=exposure_time,
            path=pathlib.Path(path),
        )


def _pixel_scale_and_units(tags):
    units = []
    scales = []
    for i in range(5):
        unit = tags.get(f"Calibrations Dimension {i} Units")
        scale = tags.get(f"Calibrations Dimension {i} Scale")
        if unit is not None and scale is not None:
            units.append(unit)
            scales.append(scale)
    return units[::-1], scales[::-1]

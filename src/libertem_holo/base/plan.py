"""Hologram reconstruction with real-to-complex and pruned FFTs.

The per-frame reconstruction only ever uses an ``out_shape``-sized window of
the hologram's spectrum, and the hologram is real. :class:`ReconstructionPlan`
precomputes, once per parameter set, which spectrum columns that window needs
and then reconstructs each frame with

* a real-to-complex FFT along the rows (half the spectrum is redundant for
  real input),
* a complex FFT along the columns only for the columns that the sideband
  and central band windows need,
* a gather of the window directly from the unshifted spectrum by a numba
  kernel (CPU) or CUDA kernel (cupy) that computes the spectrum index of
  each output element on the fly, conjugates the elements that are only
  stored as their mirror image, and applies the aperture, all in one pass,
  so there are no full-size ``roll``, ``fftshift`` or scaling passes and no
  index arrays of the size of the output window (only the 1D column set
  for the pruned FFT),
* reuse of the large intermediate buffer between frames (numpy only).

The results agree with the previous ``roll`` / ``fftshift`` / crop
implementation up to floating point rounding, including its layout for odd
sizes; see ``tests/base/test_plan.py``.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import numba
import numpy as np
from numba import cuda

if TYPE_CHECKING:
    from libertem_holo.base.utils import HoloParams

XPType = typing.Any  # Union[Module("numpy"), Module("cupy")]
SpectrumKind = Literal["half", "pruned"]


def real_dtype(precision: bool) -> np.dtype:  # noqa: FBT001 (mirrors the `precision` API)
    """Return the floating point dtype that `precision` selects (float64 or float32)."""
    return np.dtype(np.float64 if precision else np.float32)


def prepare_frame(
    frame: np.ndarray,
    *,
    sig_shape: tuple[int, int],
    precision: bool,
    xp: XPType = np,
) -> np.ndarray:
    """Check the shape of `frame` and cast it according to `precision`.

    Returns the input unchanged if it already has the right dtype.
    """
    frame = xp.asarray(frame)
    if tuple(frame.shape) != tuple(sig_shape):
        msg = f"frame shape {frame.shape} != sig_shape {tuple(sig_shape)}"
        raise ValueError(msg)
    dtype = real_dtype(precision)
    if frame.dtype != dtype:
        frame = frame.astype(dtype)
    return frame


def gather_idx_1d(n: int, o: int, s: int) -> np.ndarray:
    """Spectrum indices of the output window along one axis.

    For an unshifted spectrum of length `n`, output length `o` and an
    integer sideband shift `s`, element ``j`` of the output window (in
    unshifted order, as fed to the inverse FFT) is spectrum element
    ``gather_idx_1d(n, o, s)[j]``.

    This replicates exactly what ``roll(+s)``, ``fftshift``, the centered
    slice of :func:`libertem_holo.base.utils.get_slice_fft` and a final
    ``fftshift`` of the window do, including their behavior for odd sizes:
    for an even window in an odd spectrum the window is one bin off-centre,
    and for odd by odd the DC element lands at index ``o - 1`` of the window
    instead of 0. Changing that is a separate, result-changing step.

    :func:`_window_index_1d` is the same formula for one element, as used
    by the gather kernels.
    """
    a = int(n / 2 - o / 2)  # slice start, as in `get_slice_fft`
    j = np.arange(o)
    c = (j - o // 2) % o  # `fftshift` of the window
    return (a + c - n // 2 - s) % n  # `fftshift` of the spectrum and `roll(+s)`


@numba.njit(cache=True, inline="always")
def _window_index_1d(j: int, n: int, o: int, s: int) -> int:
    """Spectrum index of output element `j`; see :func:`gather_idx_1d`.

    Requires ``o <= n``, which :class:`ReconstructionPlan` enforces; then
    ``(n - o) // 2`` is the slice start ``int(n / 2 - o / 2)``.
    """
    a = (n - o) // 2
    c = (j - o // 2) % o
    return (a + c - n // 2 - s) % n


@numba.njit(cache=True, inline="always")
def _column_index(k: int, sx: int, ox: int, shift_x: int) -> tuple[int, bool]:
    """Half spectrum column of output column `k`, and whether it is mirrored.

    Columns beyond the Nyquist column are only stored as the conjugate of
    their mirror image in a real-to-complex spectrum.
    """
    ix = _window_index_1d(k, sx, ox, shift_x)
    if ix > sx // 2:
        return sx - ix, True
    return ix, False


@numba.njit(cache=True, nogil=True)
def _gather_cpu(  # noqa: PLR0917 (kernel arguments are positional)
    data: np.ndarray,
    col_of_ixp: np.ndarray,
    aperture: np.ndarray,
    shift_y: int,
    shift_x: int,
    sy: int,
    sx: int,
    out: np.ndarray,
) -> None:
    """Gather the window around `shift` from `data` into `out`, times `aperture`.

    `data` holds spectrum columns of a frame of shape ``(sy, sx)``;
    `col_of_ixp` maps a half spectrum column to its column in `data`.
    Single-threaded: on the CPU, this runs inside a LiberTEM worker.
    """
    oy, ox = out.shape
    cols = np.empty(ox, dtype=np.int64)
    mirrored = np.empty(ox, dtype=np.bool_)
    for k in range(ox):
        ixp, m = _column_index(k, sx, ox, shift_x)
        cols[k] = col_of_ixp[ixp]
        mirrored[k] = m
    for j in range(oy):
        iy = _window_index_1d(j, sy, oy, shift_y)
        iy_m = (-iy) % sy
        for k in range(ox):
            v = data[iy_m, cols[k]].conjugate() if mirrored[k] else data[iy, cols[k]]
            out[j, k] = v * aperture[j, k]


_GATHER_BLOCK = (16, 16)


@cuda.jit(cache=True)
def _gather_gpu(  # noqa: PLR0917 (kernel arguments are positional)
    data: np.ndarray,
    col_of_ixp: np.ndarray,
    aperture: np.ndarray,
    shift_y: int,
    shift_x: int,
    sy: int,
    sx: int,
    out: np.ndarray,
) -> None:
    # one thread per output element, same index arithmetic as `_gather_cpu`
    j, k = cuda.grid(2)
    oy, ox = out.shape
    if j < oy and k < ox:
        iy = _window_index_1d(j, sy, oy, shift_y)
        ixp, m = _column_index(k, sx, ox, shift_x)
        c = col_of_ixp[ixp]
        v = data[(-iy) % sy, c].conjugate() if m else data[iy, c]
        out[j, k] = v * aperture[j, k]


@dataclass(frozen=True)
class _Indices:
    """The column set of a plan; what depends only on geometry and array module."""

    #: unique columns of the half spectrum that the windows need, host
    cols_host: np.ndarray
    #: the same, on the compute device
    cols: typing.Any
    #: column in the pruned spectrum for each half spectrum column (-1: not needed)
    col_of_ixp: typing.Any
    #: identity map, for gathering from a complete half spectrum
    col_identity: typing.Any


@lru_cache(maxsize=8)
def _build_indices(
    *,
    sig_shape: tuple[int, int],
    out_shape: tuple[int, int],
    shift: tuple[int, int],
    with_bf: bool,
    xp: XPType,
    device: int | None,  # noqa: ARG001 (part of the cache key)
) -> _Indices:
    """Column set of a plan on the device of `xp`; cached, so transferred only once.

    `device` is the current GPU for cupy (None for numpy): it only serves as
    part of the cache key, as arrays live on the device they were created on.
    Use ``_build_indices.cache_clear()`` to release the cached arrays.
    """
    sx = sig_shape[1]
    shifts = [shift, (0, 0)] if with_bf else [shift]
    ix = np.concatenate([gather_idx_1d(sx, out_shape[1], s[1]) for s in shifts])
    ixp = np.where(ix > sx // 2, sx - ix, ix)
    cols = np.unique(ixp)
    col_of_ixp = np.full(sx // 2 + 1, -1, dtype=np.int64)
    col_of_ixp[cols] = np.arange(len(cols))
    for arr in (cols, col_of_ixp):
        arr.setflags(write=False)
    return _Indices(
        cols_host=cols,
        cols=xp.asarray(cols),
        col_of_ixp=xp.asarray(col_of_ixp),
        col_identity=xp.arange(sx // 2 + 1),
    )


@dataclass
class PartialSpectrum:
    """The part of a hologram's spectrum that reconstruction needs.

    Pass to :meth:`ReconstructionPlan.wave` and :meth:`ReconstructionPlan.bf`
    to share one FFT between the wave and the brightfield reconstruction.

    `kind` is ``"half"`` for the complete half spectrum of a real frame
    (``rfft2``, shape ``(sy, sx // 2 + 1)``, see :func:`half_spectrum`), or
    ``"pruned"`` for the column block that :meth:`ReconstructionPlan.spectrum`
    computes. `sig_shape` is the shape of the frame, which the half spectrum
    alone does not determine.
    """

    kind: SpectrumKind
    data: typing.Any
    sig_shape: tuple[int, int]


def half_spectrum(
    frame: np.ndarray,
    *,
    precision: bool = True,
    xp: XPType = np,
) -> PartialSpectrum:
    """Compute the unnormalized, unshifted half spectrum (``rfft2``) of a hologram.

    Any :class:`ReconstructionPlan` for the same frame shape can reconstruct
    from it, with any sideband position and output shape.
    """
    frame = xp.asarray(frame)
    sig_shape = (int(frame.shape[0]), int(frame.shape[1]))
    frame = prepare_frame(frame, sig_shape=sig_shape, precision=precision, xp=xp)
    return PartialSpectrum(kind="half", data=xp.fft.rfft2(frame), sig_shape=sig_shape)


class ReconstructionPlan:
    """Precomputed column set and buffers to reconstruct holograms of one shape.

    Parameters
    ----------
    sig_shape
        Shape of the input frames
    out_shape
        Shape of the reconstructed wave and brightfield image
    sb_pos
        Sideband position (y, x), referring to the unshifted FFT, as returned
        by :func:`libertem_holo.base.utils.estimate_sideband_position`.
        Truncated to integers, like the previous implementation.
    aperture
        Sideband aperture of shape `out_shape`, in unshifted layout
        (like :attr:`libertem_holo.base.utils.HoloParams.aperture`). A
        floating point aperture that is not wider than the computation's
        precision is used as is, wider ones are cast down.
    aperture_bf
        Central band aperture for the brightfield image, same layout;
        optional
    precision
        True computes in float64/complex128, False in float32/complex64.
        Input frames are cast accordingly.
    xp
        numpy or cupy
    reuse_buffers
        Keep the large intermediate array between frames (numpy only).
        A plan with buffers must not be shared between threads.

    Notes
    -----
    `out_shape` must not exceed `sig_shape` along either axis. The column
    set is cached per `sig_shape`, `out_shape`, integer sideband position,
    presence of `aperture_bf` and `xp` (and GPU device), and shared
    read-only between plans, so building many plans for the same parameters
    is cheap and does not transfer anything to the GPU. The gather kernels
    are compiled by numba on first use (cached on disk).

    Examples
    --------
    >>> from libertem_holo.base.generate import hologram_frame
    >>> from libertem_holo.base.utils import HoloParams
    >>> holo = hologram_frame(np.ones((64, 64)), np.zeros((64, 64)))
    >>> params = HoloParams.from_hologram(holo, out_shape=(32, 32))
    >>> plan = ReconstructionPlan.from_params(params)
    >>> wave, bf = plan.wave_and_bf(holo)
    >>> wave.shape, wave.dtype, bf.shape, bf.dtype
    ((32, 32), dtype('complex128'), (32, 32), dtype('float64'))

    """

    def __init__(  # noqa: D107 (parameters are documented on the class)
        self,
        *,
        sig_shape: tuple[int, int],
        out_shape: tuple[int, int],
        sb_pos: tuple[float, float],
        aperture: np.ndarray,
        aperture_bf: np.ndarray | None = None,
        precision: bool = True,
        xp: XPType = np,
        reuse_buffers: bool = True,
    ) -> None:
        self.xp = xp
        self.sig_shape = (int(sig_shape[0]), int(sig_shape[1]))
        self.out_shape = (int(out_shape[0]), int(out_shape[1]))
        self.precision = precision
        self.real_dtype = real_dtype(precision)
        self.complex_dtype = np.dtype(np.complex128 if precision else np.complex64)
        for o, n in zip(self.out_shape, self.sig_shape, strict=True):
            if not 1 <= o <= n:
                msg = f"out_shape {self.out_shape} exceeds sig_shape {self.sig_shape}"
                raise ValueError(msg)
        if tuple(aperture.shape) != self.out_shape:
            msg = f"aperture shape {aperture.shape} != out_shape {self.out_shape}"
            raise ValueError(msg)
        if aperture_bf is not None and tuple(aperture_bf.shape) != self.out_shape:
            msg = (
                f"aperture_bf shape {aperture_bf.shape} != out_shape {self.out_shape}"
            )
            raise ValueError(msg)
        self.aperture = self._as_aperture(aperture)
        self.aperture_bf = (
            None if aperture_bf is None else self._as_aperture(aperture_bf)
        )

        self._shift = (int(sb_pos[0]), int(sb_pos[1]))
        device = None if xp is np else int(xp.cuda.runtime.getDevice())
        self._idx = _build_indices(
            sig_shape=self.sig_shape,
            out_shape=self.out_shape,
            shift=self._shift,
            with_bf=aperture_bf is not None,
            xp=xp,
            device=device,
        )

        self._buffers = reuse_buffers and xp is np
        self._half: np.ndarray | None = None
        if self._buffers:
            sy, sx = self.sig_shape
            self._half = np.empty((sy, sx // 2 + 1), dtype=self.complex_dtype)

    def _as_aperture(self, aperture: np.ndarray) -> np.ndarray:
        """Move `aperture` to the device; cast only if wider than the precision.

        Multiplying the complex window with a narrower float aperture
        promotes it exactly, so there is no need for a copy per plan.
        """
        aperture = self.xp.asarray(aperture)
        wider = aperture.dtype.itemsize > self.real_dtype.itemsize
        if aperture.dtype.kind != "f" or wider:
            aperture = aperture.astype(self.real_dtype)
        return aperture

    @classmethod
    def from_params(
        cls,
        params: HoloParams,
        *,
        precision: bool = True,
        xp: XPType | None = None,
        reuse_buffers: bool = True,
    ) -> ReconstructionPlan:
        """Build a plan from :class:`libertem_holo.base.utils.HoloParams`.

        The frame shape is `params.orig_shape`; `xp` defaults to `params.xp`.
        """
        return cls(
            sig_shape=params.orig_shape,
            out_shape=params.out_shape,
            sb_pos=params.sb_position,
            aperture=params.aperture,
            aperture_bf=params.aperture_bf,
            precision=precision,
            xp=params.xp if xp is None else xp,
            reuse_buffers=reuse_buffers,
        )

    @property
    def n_columns(self) -> int:
        """Number of spectrum columns that :meth:`spectrum` transforms."""
        return len(self._idx.cols_host)

    def spectrum(self, frame: np.ndarray) -> PartialSpectrum:
        """Transform `frame` as far as needed for :meth:`wave` and :meth:`bf`.

        A real-to-complex FFT along the rows, then a complex FFT along the
        columns, only for the columns in the sideband and central band
        windows.
        """
        xp = self.xp
        frame = prepare_frame(
            frame, sig_shape=self.sig_shape, precision=self.precision, xp=xp,
        )
        if self._half is not None:
            half = np.fft.rfft(frame, axis=1, out=self._half)
            # `np.take(..., out=)` is twice as slow as allocating, so the
            # column block is allocated per frame; the FFT then runs in place:
            cols = np.take(half, self._idx.cols_host, axis=1)
            data = np.fft.fft(cols, axis=0, out=cols)
        else:
            half = xp.fft.rfft(frame, axis=1)
            cols = xp.take(half, self._idx.cols, axis=1)
            data = xp.fft.fft(cols, axis=0)
        return PartialSpectrum(kind="pruned", data=data, sig_shape=self.sig_shape)

    def _gather(
        self, spec: PartialSpectrum, shift: tuple[int, int], aperture: np.ndarray,
    ) -> np.ndarray:
        """Return the window around `shift` of `spec`, multiplied by `aperture`."""
        if tuple(spec.sig_shape) != self.sig_shape:
            msg = (
                f"spectrum is of frames of shape {spec.sig_shape}, "
                f"plan is for {self.sig_shape}"
            )
            raise ValueError(msg)
        if spec.kind == "half":
            col_map = self._idx.col_identity
        elif spec.kind == "pruned":
            col_map = self._idx.col_of_ixp
        else:
            msg = f"unknown spectrum kind {spec.kind!r}"
            raise ValueError(msg)
        xp = self.xp
        out = xp.empty(self.out_shape, dtype=self.complex_dtype)
        args = (spec.data, col_map, aperture, shift[0], shift[1], *self.sig_shape, out)
        if xp is np:
            _gather_cpu(*args)
        else:
            by, bx = _GATHER_BLOCK
            grid = (-(-self.out_shape[0] // by), -(-self.out_shape[1] // bx))
            _gather_gpu[grid, _GATHER_BLOCK](*args)
        return out

    def _ifft2(self, crop: np.ndarray) -> np.ndarray:
        if self.xp is np:
            # in place; numpy's `ifft2` ignores `out=`, `ifftn` honours it:
            return np.fft.ifftn(crop, axes=(-2, -1), out=crop)
        return self.xp.fft.ifft2(crop)

    def wave(self, spec: PartialSpectrum) -> np.ndarray:
        """Reconstruct the complex wave from a :class:`PartialSpectrum`."""
        return self._ifft2(self._gather(spec, self._shift, self.aperture))

    def bf(self, spec: PartialSpectrum) -> np.ndarray:
        """Reconstruct the brightfield image from a :class:`PartialSpectrum`."""
        if self.aperture_bf is None:
            msg = "plan was built without `aperture_bf`"
            raise ValueError(msg)
        return self.xp.abs(self._ifft2(self._gather(spec, (0, 0), self.aperture_bf)))

    def reconstruct(self, frame: np.ndarray) -> np.ndarray:
        """Reconstruct the complex wave of one hologram."""
        return self.wave(self.spectrum(frame))

    def reconstruct_bf(self, frame: np.ndarray) -> np.ndarray:
        """Reconstruct the brightfield image of one hologram."""
        return self.bf(self.spectrum(frame))

    def wave_and_bf(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct both the wave and the brightfield image of one hologram."""
        spec = self.spectrum(frame)
        return self.wave(spec), self.bf(spec)

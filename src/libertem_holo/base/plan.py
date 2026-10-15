"""Hologram reconstruction with real-to-complex and pruned FFTs.

The per-frame reconstruction only ever uses an ``out_shape``-sized window of
the hologram's spectrum, and the hologram is real. :class:`ReconstructionPlan`
precomputes, once per parameter set, which spectrum elements end up in that
window and then reconstructs each frame with

* a real-to-complex FFT along the rows (half the spectrum is redundant for
  real input),
* a complex FFT along the columns only for the columns that the sideband
  and central band windows need,
* a gather of the window directly from the unshifted spectrum, so there are
  no full-size ``roll``, ``fftshift`` or scaling passes,
* reuse of the large intermediate buffer between frames (numpy only).

The results agree with the previous ``roll`` / ``fftshift`` / crop
implementation up to floating point rounding, including its layout for odd
sizes; see ``tests/base/test_plan.py``.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

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
    """
    a = int(n / 2 - o / 2)  # slice start, as in `get_slice_fft`
    j = np.arange(o)
    c = (j - o // 2) % o  # `fftshift` of the window
    return (a + c - n // 2 - s) % n  # `fftshift` of the spectrum and `roll(+s)`


@dataclass(frozen=True)
class _Window:
    """Index arrays of one output window; read-only host (numpy) arrays."""

    #: spectrum row of each output row
    iy: np.ndarray
    #: spectrum column of each output column
    ix: np.ndarray
    #: columns that are only stored as their conjugate mirror in a half spectrum
    mirror: np.ndarray
    #: column of the half spectrum for each output column
    ixp: np.ndarray
    #: row of the half spectrum for each output element, shape `out_shape`
    rows: np.ndarray


@lru_cache(maxsize=8)
def _build_window(
    sig_shape: tuple[int, int], out_shape: tuple[int, int], shift: tuple[int, int],
) -> _Window:
    """Index arrays of the window around `shift`; cached, as the arrays are reused."""
    sy, sx = sig_shape
    iy = gather_idx_1d(sy, out_shape[0], shift[0])
    ix = gather_idx_1d(sx, out_shape[1], shift[1])
    mirror = ix > sx // 2
    ixp = np.where(mirror, sx - ix, ix)
    rows = np.where(mirror[None, :], ((-iy) % sy)[:, None], iy[:, None])
    for arr in (iy, ix, mirror, ixp, rows):
        arr.setflags(write=False)
    return _Window(iy=iy, ix=ix, mirror=mirror, ixp=ixp, rows=rows)


@dataclass
class _DeviceWindow:
    """Index arrays of one output window, on the compute device."""

    #: column of the half spectrum for each output column
    ixp: typing.Any
    #: row of the half spectrum for each output element, shape `out_shape`
    rows: typing.Any
    #: position of each output column in the pruned column set
    colpos: typing.Any
    #: -1 for mirrored columns, +1 otherwise; None if nothing is mirrored
    sign: typing.Any


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
    """Precomputed indices and buffers to reconstruct holograms of one shape.

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
        (like :attr:`libertem_holo.base.utils.HoloParams.aperture`)
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
        if tuple(aperture.shape) != self.out_shape:
            msg = f"aperture shape {aperture.shape} != out_shape {self.out_shape}"
            raise ValueError(msg)
        self.aperture = xp.asarray(aperture, dtype=self.real_dtype)
        self.aperture_bf = (
            None if aperture_bf is None
            else xp.asarray(aperture_bf, dtype=self.real_dtype)
        )

        shift = (int(sb_pos[0]), int(sb_pos[1]))
        sb = _build_window(self.sig_shape, self.out_shape, shift)
        bf = _build_window(self.sig_shape, self.out_shape, (0, 0))
        windows = [sb, bf] if aperture_bf is not None else [sb]
        cols, inv = np.unique(
            np.concatenate([w.ixp for w in windows]), return_inverse=True,
        )
        ox = self.out_shape[1]
        self._cols_host = cols
        self._cols = xp.asarray(cols)
        self._sb = self._to_device(sb, inv[:ox])
        self._bf = self._to_device(bf, inv[ox:]) if aperture_bf is not None else None

        self._buffers = reuse_buffers and xp is np
        self._half: np.ndarray | None = None
        if self._buffers:
            sy, sx = self.sig_shape
            self._half = np.empty((sy, sx // 2 + 1), dtype=self.complex_dtype)

    def _to_device(self, w: _Window, colpos: np.ndarray) -> _DeviceWindow:
        xp = self.xp
        sign = None
        if w.mirror.any():
            sign = xp.asarray(np.where(w.mirror, -1, 1).astype(self.real_dtype))
        return _DeviceWindow(
            ixp=xp.asarray(w.ixp),
            rows=xp.asarray(w.rows),
            colpos=xp.asarray(colpos),
            sign=sign,
        )

    @classmethod
    def from_params(
        cls,
        params: typing.Any,  # noqa: ANN401 (HoloParams; avoids a circular import)
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
        return len(self._cols_host)

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
            cols = np.take(half, self._cols_host, axis=1)
            data = np.fft.fft(cols, axis=0, out=cols)
        else:
            half = xp.fft.rfft(frame, axis=1)
            cols = xp.take(half, self._cols, axis=1)
            data = xp.fft.fft(cols, axis=0)
        return PartialSpectrum(kind="pruned", data=data, sig_shape=self.sig_shape)

    def _gather(self, spec: PartialSpectrum, w: _DeviceWindow) -> np.ndarray:
        if tuple(spec.sig_shape) != self.sig_shape:
            msg = (
                f"spectrum is of frames of shape {spec.sig_shape}, "
                f"plan is for {self.sig_shape}"
            )
            raise ValueError(msg)
        if spec.kind == "half":
            crop = spec.data[w.rows, w.ixp[None, :]]
        elif spec.kind == "pruned":
            crop = spec.data[w.rows, w.colpos[None, :]]
        else:
            msg = f"unknown spectrum kind {spec.kind!r}"
            raise ValueError(msg)
        if w.sign is not None:
            # conjugate the columns that are only stored as their mirror:
            crop.imag *= w.sign
        return crop

    def _ifft2(self, crop: np.ndarray) -> np.ndarray:
        if self._buffers:
            # numpy's `ifft2` ignores `out=`, `ifftn` honours it:
            return np.fft.ifftn(crop, axes=(-2, -1), out=crop)
        return self.xp.fft.ifft2(crop)

    def wave(self, spec: PartialSpectrum) -> np.ndarray:
        """Reconstruct the complex wave from a :class:`PartialSpectrum`."""
        crop = self._gather(spec, self._sb)
        crop *= self.aperture
        return self._ifft2(crop)

    def bf(self, spec: PartialSpectrum) -> np.ndarray:
        """Reconstruct the brightfield image from a :class:`PartialSpectrum`."""
        if self._bf is None or self.aperture_bf is None:
            msg = "plan was built without `aperture_bf`"
            raise ValueError(msg)
        crop = self._gather(spec, self._bf)
        crop *= self.aperture_bf
        return self.xp.abs(self._ifft2(crop))

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

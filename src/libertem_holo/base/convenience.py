"""Convenience functions for reconstruction and visualization."""
from __future__ import annotations

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

from libertem.common.threading import set_num_threads
from scipy.ndimage import gaussian_filter

from libertem_holo.base.align import Correlator, ImageCorrelator
from libertem_holo.base.io import InputData, Results
from libertem_holo.base.plan import ReconstructionPlan
from libertem_holo.base.reconstr import phase_offset_correction
from libertem_holo.base.utils import HoloParams, to_cpu

try:
    import cupy as cp
except ImportError:
    cp = None

import empyre as emp
import numpy as np
from matplotlib.axes import Axes
from skimage.measure import block_reduce as br
from tqdm import tqdm

if TYPE_CHECKING:
    from libertem.api import Context

    from libertem_holo.base.reconstr import XPType


def reconstruct_stack(
    stack: InputData,
    stack_ref: InputData | None = None,
    holoparams: HoloParams | None = None,
    correlator: Correlator | None = None,
    *,
    backend: Literal["cupy", "libertem"] = "cupy",
    ctx: Context | None = None,
    progress: bool = True,
    align_threads: int | None = None,
) -> Results:
    """Reconstruct and align a stack of holograms.

    Optionally use a reference stack for phase correction.

    The processing happens in phases: first, all holograms are reconstructed
    into a stack of complex waves and brightfield images. Then, the waves are
    aligned against the first wave, using the brightfield images for
    registration. Finally, the aligned waves are averaged using phase offset
    correction, and optionally the (averaged) reference wave is divided out.

    Parameters
    ----------
    stack : InputData
        The input stack of holograms to reconstruct.
        Shape should be (N, H, W) or (H, W).
    stack_ref : InputData, optional
        An optional reference stack of holograms for phase correction.
        Shape should be (N, H, W) or (H, W).
    holoparams : HoloParams, optional
        Parameters for hologram reconstruction.
        If None, parameters will be estimated from
        the first hologram in the stack.
    correlator : Correlator, optional
        Parameters for frame by frame alignment.
        If None, phase correlation will be used. Make sure the correlator
        works with the array module matching the `backend` (cupy for
        :code:`"cupy"`, numpy for :code:`"libertem"`).
    backend : "cupy" or "libertem"
        With :code:`"cupy"`, everything runs on the GPU in this process.
        With :code:`"libertem"`, the reconstruction runs as a
        :class:`~libertem_holo.udf.reconstr.HoloReconstructUDF` on the
        given :code:`ctx`, and alignment and averaging run on the CPU
        using numpy.
    ctx : libertem.api.Context, optional
        The LiberTEM :code:`Context` to run the reconstruction on;
        required for the :code:`"libertem"` backend, and ignored by the
        :code:`"cupy"` backend. It is not created here, as starting up a
        :code:`Context` can take a long time, so it should be created once
        and re-used.
    progress : bool
        Show progress bars
    align_threads : int, optional
        Number of threads for registering and shifting the frames; all frames
        are aligned against the first one, so they are independent. Defaults
        to the number of CPUs for numpy, and to 1 for cupy.

    Returns
    -------
    results
        The results, packaged up in a Results instance

    """
    if backend == "cupy":
        if cp is None:
            msg = "Need Cupy for the cupy backend."
            raise RuntimeError(msg)
        xp = cp
        # the cupy backend reconstructs in this process:
        ctx = None
    elif backend == "libertem":
        if ctx is None:
            msg = "The libertem backend needs a `Context` passed in as `ctx`"
            raise ValueError(msg)
        xp = np
    else:
        msg = f"unknown backend: {backend!r}"
        raise ValueError(msg)
    return _reconstruct_stack(
        stack=stack,
        stack_ref=stack_ref,
        holoparams=holoparams,
        correlator=correlator,
        xp=xp,
        ctx=ctx,
        progress=progress,
        align_threads=align_threads,
    )


def _reconstruct_stack(
    stack: InputData,
    stack_ref: InputData | None,
    holoparams: HoloParams | None,
    correlator: Correlator | None,
    *,
    xp: XPType,
    ctx: Context | None,
    progress: bool,
    align_threads: int | None = None,
) -> Results:
    """Implementation of :func:`reconstruct_stack`.

    If `ctx` is given, the reconstruction runs as an UDF, otherwise
    frame-by-frame using the array module `xp`. Alignment and averaging
    always use `xp`.
    """
    sig_shape = stack.data[0].shape
    out_shape = (sig_shape[0]//8, sig_shape[1]//8)
    if holoparams is None:
        holoparams = HoloParams.from_hologram(
            stack.data[0],
            out_shape=out_shape,
            line_filter_width=2,
            line_filter_length=0.9,
            central_band_mask_radius=100,
            xp=xp,
        )

    if correlator is None:
        correlator = ImageCorrelator(
            hanning=True,
            normalization="phase",
            binning=2,
            upsample_factor=10,
            xp=xp,
        )

    waves, bfs = _reconstruct_waves(
        stack, holoparams, with_bf=True, xp=xp, ctx=ctx, progress=progress,
    )
    waves, bfs, drifts = _align_waves(
        waves, bfs, correlator, xp=xp, progress=progress, n_threads=align_threads,
    )

    wave_avg, _, _ = phase_offset_correction(waves, xp=xp)
    bf_avg = xp.mean(bfs, axis=0)

    if stack_ref is None:
        wave_ref = xp.ones_like(wave_avg)
    else:
        waves_ref, _ = _reconstruct_waves(
            stack_ref, holoparams, with_bf=False, xp=xp, ctx=ctx,
            progress=progress,
        )
        wave_ref, _, _ = phase_offset_correction(waves_ref, xp=xp)

    wave = wave_avg / wave_ref
    res = Results(complex_wave=to_cpu(wave), brightfield=to_cpu(bf_avg))
    res.metadata_from_input(stack, holoparams)
    drifts = to_cpu(drifts)
    res.metadata["drifts_x"] = list(drifts[..., 0])
    res.metadata["drifts_y"] = list(drifts[..., 1])
    return res


def _reconstruct_waves(
    stack: InputData,
    holoparams: HoloParams,
    *,
    with_bf: bool = True,
    xp: XPType = np,
    ctx: Context | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Reconstruct each hologram in `stack` into a complex wave.

    Parameters
    ----------
    stack
        The input stack of holograms
    holoparams
        Reconstruction parameters
    with_bf
        Also reconstruct brightfield images from the central band
    xp
        The array module (numpy or cupy) for the results. Also used for
        computation if no `ctx` is given.
    ctx
        If given, run the reconstruction as a
        :class:`~libertem_holo.udf.reconstr.HoloReconstructUDF` on this
        :code:`Context`. Otherwise, reconstruct frame by frame in this
        process.
    progress
        Show a progress bar

    Returns
    -------
    waves
        Complex128 array of shape :code:`(N,) + holoparams.out_shape`
    bfs
        Float32 array of brightfield images, same shape as `waves`,
        or None if `with_bf` is False

    """
    if ctx is not None:
        waves, bfs = _reconstruct_waves_udf(
            stack, holoparams, with_bf=with_bf, ctx=ctx, progress=progress,
        )
        return xp.asarray(waves), None if bfs is None else xp.asarray(bfs)
    return _reconstruct_waves_loop(
        stack, holoparams, with_bf=with_bf, xp=xp, progress=progress,
    )


def _reconstruct_waves_loop(
    stack: InputData,
    holoparams: HoloParams,
    *,
    with_bf: bool,
    xp: XPType,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    stack_out_shape = (len(stack.data),) + holoparams.out_shape
    waves = xp.zeros(stack_out_shape, dtype=np.complex128)
    bfs = xp.zeros(stack_out_shape, dtype=np.float32) if with_bf else None
    # the plan computes the window indices once and reuses its FFT buffer:
    plan = ReconstructionPlan.from_params(holoparams, xp=xp)

    for i in tqdm(range(len(stack.data)), disable=not progress):
        # FFT once, shared between wave and brightfield reconstruction:
        spec = plan.spectrum(stack.data[i])
        waves[i] = plan.wave(spec)
        if bfs is not None:
            bfs[i] = plan.bf(spec)
    return waves, bfs


def _reconstruct_waves_udf(
    stack: InputData,
    holoparams: HoloParams,
    *,
    with_bf: bool,
    ctx: Context,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    from libertem_holo.udf.reconstr import HoloReconstructUDF  # noqa: PLC0415

    udf = HoloReconstructUDF(
        out_shape=holoparams.out_shape,
        sb_position=holoparams.sb_position,
        aperture=to_cpu(holoparams.aperture),
        aperture_bf=to_cpu(holoparams.aperture_bf) if with_bf else None,
    )
    ds = stack.into_libertem_dataset(ctx=ctx)
    res = ctx.run_udf(dataset=ds, udf=udf, progress=progress)
    stack_out_shape = (len(stack.data),) + holoparams.out_shape
    waves = res["wave"].data.reshape(stack_out_shape)
    bfs = res["bf"].data.reshape(stack_out_shape) if with_bf else None
    return waves, bfs


def _align_waves(
    waves: np.ndarray,
    bfs: np.ndarray,
    correlator: Correlator,
    *,
    xp: XPType = np,
    progress: bool = True,
    n_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align a stack of waves against the first wave.

    The shifts are determined by registering the brightfield images `bfs`
    using `correlator`, and then applied to both `waves` and `bfs`.

    The input arrays are modified in place. As all frames are aligned
    against the first one, they are processed independently in a thread
    pool of `n_threads` threads (default: the number of CPUs for numpy, 1
    for cupy). The FFTs and the interpolation release the GIL, so this
    scales with the number of cores; BLAS is limited to one thread while the
    pool runs, as the upsampled DFT would otherwise oversubscribe the cores
    (measured 4x slower). `correlator` is shared between the threads, so it
    must not keep per-call state. On cupy, the threads use the GPU that is
    current in the calling thread.

    Returns
    -------
    waves
        The aligned waves
    bfs
        The aligned (absolute) brightfield images
    drifts
        The shifts that were applied, shape (N, 2)

    """
    if xp is np:
        from scipy.ndimage import shift  # noqa: PLC0415
    else:
        from cupyx.scipy.ndimage import shift  # noqa: PLC0415

    if n_threads is None:
        n_threads = (os.cpu_count() or 1) if xp is np else 1

    drifts = xp.zeros((waves.shape[0], 2))
    # copy, as we update `bfs` in-place and `prepare_input` may return a view:
    ref = correlator.prepare_input(bfs[0].copy())
    # the current GPU is thread-local in cupy, so carry it over to the pool:
    device = contextlib.nullcontext() if xp is np else xp.cuda.Device()

    def _align_one(i: int) -> None:
        with device:
            moving = correlator.prepare_input(bfs[i])
            shifts = correlator.correlate(ref_image=ref, moving_image=moving).shift
            waves[i] = shift(waves[i], shifts)
            bfs[i] = xp.abs(shift(bfs[i], shifts))
            drifts[i] = xp.asarray(shifts)

    with (
        set_num_threads(1 if n_threads > 1 else None),
        ThreadPoolExecutor(max_workers=n_threads) as pool,
    ):
        for _ in tqdm(
            pool.map(_align_one, range(waves.shape[0])),
            total=waves.shape[0],
            disable=not progress,
        ):
            pass
    return waves, bfs, drifts


def plot_mag_induction(
    phase: np.ndarray,
    axis: Axes,
    mask: np.ndarray | None = None,
    clip: float = 1e-3,
    binning: int = 1,
    gain: float = 8,
    smooth: float = 5,
    cmap=None,
    **kwargs,
) -> Axes:
    """Plot a magnetic induction map from a magnetic phase image.

    This function combines a color-encoded visualization of the magnetic
    induction (obtained from the curl of the phase) with cosine contours of the
    smoothed phase image. The induction direction is encoded using
    :func:`colorvec`, while the cosine contours provide a visual representation
    of the phase. Optionally, a mask outline can be overlaid.

    Parameters
    ----------
    phase : ndarray
        Two-dimensional magnetic phase image.
    axis : `matplotlib.axes.Axes`
        Axis on which to draw the visualization.
    mask : ndarray, optional
        Binary mask whose boundary is overlaid as a white contour, by default
        None.
    clip : float, optional
        Maximum induction magnitude to display. Larger values are clipped before
        color encoding, by default 1e-3.
    binning : int, optional
        Spatial binning factor applied to the phase image and mask before
        processing, by default 1.
    gain : float, optional
        Gain factor applied when computing the cosine contours, by default 8.
    smooth : float, optional
        Standard deviation of the Gaussian filter applied to the phase image
        before computing the induction field, by default 5.
    cmap : str or `matplotlib.colors.Colormap`, optional
        Colormap used for the induction map. By default,
        ``emp.vis.colors.cmaps.cyclic_cubehelix`` is used.
    **kwargs
        Additional keyword arguments passed to
        :func:`empyre.vis.colorvec` and
        :func:`empyre.vis.cosine_contours`.

    Returns
    -------
    matplotlib.axes.Axes
        The plotting axis.

    Notes
    -----
    The visualization consists of

    - a color-encoded induction map computed from the curl of the smoothed
      phase image,
    - cosine contours of the smoothed phase image,
    - a color wheel indicating the direction encoding,
    - an optional white contour showing the supplied mask.

    """
    phase_binned = br(phase, binning, np.mean)

    phase_field = emp.fields.Field(
        data=gaussian_filter(phase_binned, sigma=smooth),
        scale=1,
        vector=False,
    )

    if cmap is None:
        cmap = emp.vis.colors.cmaps.cyclic_cubehelix

    emp.vis.colorvec(
        phase_field.curl().clip(vmax=clip),
        cmap=cmap,
        axis=axis,
        origin="upper",
        **kwargs,
    )

    emp.vis.cosine_contours(
        phase_field,
        gain=gain,
        axis=axis,
        origin="upper",
        **kwargs,
    )

    emp.vis.colorwheel(cmap=cmap, axis=axis)

    if mask is not None:
        mask_binned = br(mask, binning, np.mean)
        mask_field = emp.fields.Field(data=mask_binned, scale=1, vector=False)

        emp.vis.contour(
            mask_field[::-1],
            axis=axis,
            origin="upper",
            colors="white",
            linewidths=1.0,
            linestyles="-",
            **kwargs,
        )

    return axis

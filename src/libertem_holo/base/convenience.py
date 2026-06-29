"""Convenience functions for reconstruction and visualization."""
from scipy.ndimage import gaussian_filter

from libertem_holo.base.align import Correlator, ImageCorrelator
from libertem_holo.base.io import InputData, Results
from libertem_holo.base.reconstr import (
    phase_offset_correction,
    reconstruct_bf,
    reconstruct_frame,
)
from libertem_holo.base.utils import HoloParams

try:
    import cupy as cp
    from cupyx.scipy.ndimage import shift as shiftcp
except ImportError:
    cp = None
    shiftcp = None

import empyre as emp
import numpy as np
from matplotlib.axes import Axes
from skimage.measure import block_reduce as br
from tqdm import tqdm


def reconstruct_stack(
    stack: InputData,
    stack_ref: InputData | None = None,
    holoparams: HoloParams | None = None,
    correlator: Correlator | None = None,
) -> Results:
    """Reconstruct and align a stack of holograms.

    Optionally use a reference stack for phase correction.

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
        If None, phase correlation will be used.

    Returns
    -------
    results
        The results, packaged up in a Results instance

    """
    if shiftcp is None or cp is None:
        msg = "Need Cupy for this stack reconstruction function."
        raise RuntimeError(msg)
    xp = cp
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
    stack_out_shape = (len(stack.data),) + holoparams.out_shape
    waves_aligned = xp.zeros(stack_out_shape, dtype=np.complex128)
    bfs_aligned = xp.zeros(stack_out_shape, dtype=np.float32)
    drifts = xp.zeros((len(stack.data), 2))

    for i in tqdm(range(len(stack.data))):
        obj = stack.data[i]

        wave_obj = reconstruct_frame(
            frame=obj,
            sb_pos=holoparams.sb_position,
            aperture=holoparams.aperture,
            slice_fft=holoparams.slice_fft,
            xp=xp,
        )
        if i == 0:
            bf_obj_0 = reconstruct_bf(
                frame=obj,
                aperture=holoparams.aperture,
                slice_fft=holoparams.slice_fft,
                xp=xp,
            )
            f1 = correlator.prepare_input(bf_obj_0)

        bf_obj = reconstruct_bf(
            frame=obj,
            aperture=holoparams.aperture,
            slice_fft=holoparams.slice_fft,
            xp=xp,
        )
        f2 = correlator.prepare_input(bf_obj)

        corr_res = correlator.correlate(ref_image=f1, moving_image=f2)
        shifts = corr_res.shift

        wave_shifted = shiftcp(wave_obj, shifts)
        bf_shifted = shiftcp(bf_obj, shifts)

        waves_aligned[i] = wave_shifted
        bfs_aligned[i] = np.abs(bf_shifted)
        drifts[i] = xp.asarray(shifts)

    wave_avg, _, _ = phase_offset_correction(waves_aligned, xp=xp)
    bf_avg = np.mean(bfs_aligned, axis=0)

    if stack_ref is None:
        wave_ref = np.ones_like(wave_avg)
    else:
        waves_ref = xp.zeros(
            (len(stack_ref.data),) + holoparams.out_shape,
            dtype=np.complex128,
        )

        for i in tqdm(range(len(stack_ref.data))):
            ref = stack_ref.data[i]
            wave_ref = reconstruct_frame(
                frame=ref, sb_pos=holoparams.sb_position,
                aperture=holoparams.aperture, slice_fft=holoparams.slice_fft,
                xp=xp,
            )
            waves_ref[i] = wave_ref

        wave_ref, _, _ = phase_offset_correction(waves_ref, xp=xp)

    wave = wave_avg / wave_ref
    res = Results(complex_wave=wave.get(), brightfield=bf_avg.get())
    res.metadata_from_input(stack, holoparams)
    res.metadata["drifts_x"] = list(drifts[..., 0].get())
    res.metadata["drifts_y"] = list(drifts[..., 1].get())
    return res


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
    phase_binned = br(phase, (binning, binning), np.mean)

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
        mask_binned = br(mask, (binning, binning), np.mean)
        mask_field = emp.fields.Field(data=mask_binned, scale=1, vector=False)

        emp.vis.contour(
            mask_field[::-1],
            axis=axis,
            origin="upper",
            colors="white",
            linewidths=1.0,
            linestyles="-",
        )

    return axis

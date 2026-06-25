from libertem_holo.base.utils import HoloParams
from libertem_holo.base.reconstr import reconstruct_frame, reconstruct_bf
from libertem_holo.base.align import ImageCorrelator, Correlator
from libertem_holo.base.reconstr import phase_offset_correction
from scipy.ndimage import shift, gaussian_filter
from cupyx.scipy.ndimage import shift as shiftcp
from skimage.measure import block_reduce as br
import empyre as emp
import numpy as np
from tqdm import tqdm
import typing
XPType = typing.Any  # Union[Module("numpy"), Module("cupy")]


def reconstruct_stack(
        stack: np.ndarray,
        stack_ref: np.ndarray = None,
        holoparams: HoloParams = None,
        correlator: Correlator = None,
        xp: XPType = np,
    ) -> tuple:
    """Reconstruct a stack of holograms, aligning them and optionally using a reference stack for phase correction.

    Parameters
    ----------
    stack : np.ndarray
        The input stack of holograms to reconstruct.
        Shape should be (N, H, W) or (H, W).
    stack_ref : np.ndarray, optional
        An optional reference stack of holograms for phase correction.
        Shape should be (N, H, W) or (H, W).
    holoparams : HoloParams, optional
        Parameters for hologram reconstruction.
        If None, parameters will be estimated from the first hologram in the stack.
    correlator : Correlator, optional
        Parameters for frame by frame alignment.
        If None, phase correlation will be used.
    xp : module, optional
        The array module to use for computations.
        Can be numpy or cupy. Default is numpy.

    Returns
    -------
    wave : np.ndarray
        The reconstructed complex wave.
    bf_avg : np.ndarray
        The averaged bright-field image.
    holoparams : HoloParams
        The holography reconstruction parameters.
    px_size : float
        The pixel size in the reconstructed wave.

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
    px_size = stack.pixelsize * holoparams.scale_factor
    if correlator is None:
        correlator = ImageCorrelator(
            hanning=True,
            normalization='phase',
            binning=2,
            upsample_factor=10,
            xp=xp,
        )
    waves_aligned = xp.zeros((len(stack.data),) + holoparams.out_shape, dtype=np.complex128)
    bfs_aligned = xp.zeros((len(stack.data),) + holoparams.out_shape, dtype=np.float32)
    drifts = xp.zeros((len(stack.data), 2))
    for i in tqdm(range(len(stack.data))):
        obj = stack.data[i]

        wave_obj = reconstruct_frame(
            frame=obj, sb_pos=holoparams.sb_position,
            aperture=holoparams.aperture, slice_fft=holoparams.slice_fft,
            xp=xp,
        )
        if i == 0:
            bf_obj_0 = reconstruct_bf(frame=obj, aperture=holoparams.aperture, slice_fft=holoparams.slice_fft, xp=xp)
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

        if wave_obj.device == 'cpu':
            shifts = tuple(float(x) for x in corr_res.shift)
            wave_shifted = shift(wave_obj, shifts)
            bf_shifted = shift(bf_obj, shifts)
        else:
            wave_shifted = shiftcp(wave_obj, shifts)
            bf_shifted = shiftcp(bf_obj, shifts)

        waves_aligned[i] = wave_shifted
        bfs_aligned[i] = np.abs(bf_shifted)
        drifts[i] = shifts

    wave_avg, _, _ = phase_offset_correction(waves_aligned, xp=xp)
    bf_avg = np.mean(bfs_aligned, axis=0)


    if stack_ref is None:
        wave_ref = np.ones_like(wave_avg)
    else:
        if holoparams is None:
            holoparams = HoloParams.from_hologram(
                stack_ref.data[0], out_shape=out_shape, xp=xp, line_filter_width=5, line_filter_length=0.9, central_band_mask_radius=100,
            )
        waves_ref = xp.zeros((len(stack_ref.data),) + holoparams.out_shape, dtype=np.complex128)

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
    return wave, bf_avg, holoparams, px_size, drifts


def plot_mag_induction(phase, axis, mask=None, clipper = 1e-3, binning = 1, gain = 8, smooth = 5):
    phase_binned = br(phase, (binning, binning), np.mean)
    vmin, vmax = np.min(phase_binned), np.max(phase_binned)

    # Represent the unwrap phase image with the 'field' class
    phase_field = emp.fields.Field(data=gaussian_filter(phase_binned, sigma=smooth), scale=1, vector=False)

    #Display the curl
    cmap = emp.vis.colors.cmaps.cyclic_cubehelix
    emp.vis.colorvec(phase_field.curl().clip(vmax = clipper), vmin=vmin, vmax=vmax, cmap=cmap, axis=axis, origin='upper')

    #Display the cosine contours
    emp.vis.cosine_contours(phase_field, gain=gain, axis=axis, origin='upper')
    emp.vis.colorwheel(cmap=cmap, axis = axis)

    if mask is not None:
        mask_binned  = br(mask, (binning, binning), np.mean)
        mask_field  = emp.fields.Field(data=mask_binned, scale=1, vector=False)
        emp.vis.contour(mask_field[::-1], axis=axis, origin='upper', colors='white', linewidths=1.0, linestyles='-')

from libertem_holo.base.utils import HoloParams
from libertem_holo.base.reconstr import reconstruct_frame, reconstruct_bf
from libertem_holo.base.align import ImageCorrelator
from libertem_holo.base.reconstr import phase_offset_correction
import numpy as np
import tqdm
import typing
XPType = typing.Any  # Union[Module("numpy"), Module("cupy")]

def reconstruct_stack(
        stack: np.ndarray,
        stack_ref: np.ndarray = None,
        holoparams: HoloParams = None,
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
    xp : module, optional
        The array module to use for computations.
        Can be numpy or cupy. Default is numpy.

    Returns
    -------
    wave : np.ndarray
        The reconstructed complex wave.
    bf_avg : np.ndarray
        The averaged bright-field image.
    px_size : float
        The pixel size in the reconstructed wave.

    """
    try:
        import cupyx.scipy.ndimage as ndimage
    except ImportError:
        import scipy.ndimage as ndimage

    if len(stack.data.shape) == 2:
        stack.data = stack.data.reshape((1,) + stack.data.shape)
    sig_shape = stack.data[0].shape
    out_shape = (sig_shape[0] // 4, sig_shape[1] // 4)
    if holoparams is None:
        holoparams = HoloParams.from_hologram(
            stack.data[0],
            out_shape=out_shape,
            xp=xp,
            line_filter_width=5,
            line_filter_length=0.9,
            central_band_mask_radius=100,
        )
    px_size = stack.pixelsize * holoparams.scale_factor
    correlator = ImageCorrelator(
        hanning=True,
        normalization='phase',
        binning=2,
        upsample_factor=10,
        xp=xp,
    )
    waves_aligned = []
    bfs_aligned = []
    for i in tqdm.tqdm(range(len(stack.data))):
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

        corr_res = correlator.correlate(
            ref_image=f1,
            moving_image=f2,
        )
        shifts = tuple(float(x) for x in corr_res.shift)
        wave_shifted = ndimage.shift(wave_obj, shifts)
        bf_shifted = ndimage.shift(bf_obj, shifts)

        waves_aligned.append(wave_shifted)
        bfs_aligned.append(bf_shifted)

    waves_aligned = np.stack(waves_aligned)
    bfs_aligned = np.stack(bfs_aligned)

    wave_avg, _, _ = phase_offset_correction(waves_aligned.get())
    bf_avg, _, _ = phase_offset_correction(bfs_aligned.get())

    if stack_ref is None:
        wave_ref = np.ones_like(wave_avg)
    else:
        if holoparams is None:
            holoparams = HoloParams.from_hologram(
                stack_ref.data[0],
                out_shape=out_shape,
                xp=xp,
                line_filter_width=5,
                line_filter_length=0.9,
                central_band_mask_radius=100,
            )
        waves_ref = []

        for i in tqdm.tqdm(range(len(stack_ref.data))):
            ref = stack_ref.data[i]
            wave_ref = reconstruct_frame(
                frame=ref,
                sb_pos=holoparams.sb_position,
                aperture=holoparams.aperture,
                slice_fft=holoparams.slice_fft,
                xp=xp,
            )

            waves_ref.append(wave_ref)

        waves_ref = np.stack(waves_ref)
        wave_ref, _, _ = phase_offset_correction(waves_ref.get())

    wave = wave_avg / wave_ref

    return wave, bf_avg, px_size

[Feature] Pass HoloParams to reconstruct_frame
==============================================

 * :func:`~libertem_holo.base.reconstr.reconstruct_frame` accepts the
   reconstruction parameters as a
   :class:`~libertem_holo.base.utils.HoloParams` instance in the new
   :code:`holoparams` argument. Explicitly passed :code:`sb_pos`,
   :code:`aperture` and :code:`slice_fft` take precedence over it. Without
   any parameters, they are estimated using
   :meth:`~libertem_holo.base.utils.HoloParams.from_hologram`. Existing calls
   with :code:`sb_pos`, :code:`aperture` and :code:`slice_fft` work as before
   (:issue:`176`).

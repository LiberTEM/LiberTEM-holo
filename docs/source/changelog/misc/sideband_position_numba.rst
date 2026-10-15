[Misc] Faster sideband position estimation
==========================================

 * :func:`~libertem_holo.base.utils.estimate_sideband_position`, and with it
   :meth:`~libertem_holo.base.utils.HoloParams.from_hologram`, now searches the
   spectrum for the sideband with a numba kernel (CPU and GPU) instead of
   building masks and intermediate arrays. The sideband position found is
   the same as before (except for the fix for odd shapes below).
 * An invalid :code:`sb` argument now raises a :code:`ValueError` instead of
   returning :code:`(0.0, 0.0)`.

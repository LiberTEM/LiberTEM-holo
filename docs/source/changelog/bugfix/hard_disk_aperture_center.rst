[Bugfix] Central band mask off by one pixel for odd shapes
==========================================================

 * When estimating the sideband position, the mask removing the central band
   was shifted by one pixel for holograms with an odd number of rows or
   columns. Affects
   :func:`~libertem_holo.base.utils.estimate_sideband_position` and
   :meth:`~libertem_holo.base.utils.HoloParams.from_hologram`.

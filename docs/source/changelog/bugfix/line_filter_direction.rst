[Bugfix] Direction of the butterworth line filter
=================================================

 * :func:`libertem_holo.base.filters.butterworth_line` now always points from
   the center towards :code:`sb_position`. Before, the direction was reversed
   when the slope of the line was between 0 and 1, so that in the aperture from
   :meth:`libertem_holo.base.utils.HoloParams.from_hologram` the line went
   through the center of the sideband (and, in :code:`aperture_bf`, through the
   central band). For sideband positions that were not affected, the
   aperture is unchanged.
 * Sideband positions on the horizontal or vertical axis now get a line
   filter; before, the line was missing.
 * :func:`~libertem_holo.base.filters.butterworth_line` raises a
   :code:`ValueError` if :code:`sb_position` is at the center, instead of
   returning NaNs.

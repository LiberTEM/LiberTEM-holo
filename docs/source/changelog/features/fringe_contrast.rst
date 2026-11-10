[Feature] Fringe contrast
=========================

 * New :func:`~libertem_holo.base.utils.fringe_contrast` to measure the
   fringe contrast (visibility) of a hologram in a region, either from the
   sideband and centerband amplitudes (:code:`method="sideband"`) or from a
   line profile across the fringes (:code:`method="profile"`), and
   :func:`~libertem_holo.base.utils.fringe_contrast_map` for a map of the
   contrast. :func:`~libertem_holo.base.utils.fringe_period` gives the
   direction and period of the fringes from the sideband position.

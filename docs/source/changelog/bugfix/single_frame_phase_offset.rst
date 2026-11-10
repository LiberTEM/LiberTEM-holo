[Bugfix] Phase offset correction of a single image
==================================================

 * :func:`~libertem_holo.base.reconstr.phase_offset_correction` now passes a
   single image through unchanged, given either as a stack of shape
   :code:`(1, sy, sx)` or as a 2D array of shape :code:`(sy, sx)`. Before, it
   failed with :code:`ArpackError: Starting vector is zero`. This also fixes
   :func:`~libertem_holo.base.convenience.reconstruct_stack` with a single
   hologram as :code:`stack_ref` (or :code:`stack`) (:issue:`173`).

[Misc] Gather the reconstruction window with a numba kernel
===========================================================

 * :class:`~libertem_holo.base.plan.ReconstructionPlan` gathers the output
   window from the spectrum with a numba kernel (a CUDA kernel on cupy) that
   computes the spectrum index of each element on the fly and applies the
   conjugation of mirrored columns and the aperture in the same pass. This
   replaces the index arrays of the size of the output window and two
   passes over it; the results are unchanged.

[Feature] Phase unwrapping by ADMM, and cupy input for IRLS and ADMM
====================================================================

 * New phase unwrapping by ADMM, with the objective of
   :func:`~libertem_holo.base.unwrap.unwrap_phase_irls` in fewer iterations,
   based on Rouet-Leduc and Hulbert, Remote Sensing 18, 1801 (2026). Use it with
   :func:`~libertem_holo.base.unwrap.unwrap_phase_admm`, or with
   :code:`method="admm"` in :func:`~libertem_holo.base.unwrap.phase_unwrap`.
   Both methods solve on the GPU for cupy input, keeping their buffers in a
   :class:`~libertem_holo.base.workspace.Workspace`, and leave NaN pixels NaN.
 * Experimental, please report any issues you encounter

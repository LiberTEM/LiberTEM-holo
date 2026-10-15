[Feature] Faster hologram reconstruction
========================================

 * Added :class:`~libertem_holo.base.plan.ReconstructionPlan`, which
   precomputes once which part of a hologram's spectrum ends up in the output
   window, and then reconstructs each hologram with a real-to-complex FFT,
   transforming only the needed columns of the spectrum and gathering the
   window directly, without full-size :code:`roll`, :code:`fftshift` or
   scaling passes. Build it with
   :meth:`~libertem_holo.base.plan.ReconstructionPlan.from_params` and reuse
   it for a whole stack. Reconstructing the wave and the brightfield image of
   a 4096x4096 hologram at 1024x1024 takes about a sixth of the time it did.
 * :func:`~libertem_holo.base.reconstr.reconstruct_frame`,
   :func:`~libertem_holo.base.reconstr.reconstruct_bf` and
   :class:`~libertem_holo.base.reconstr.HologramFFT` are now thin wrappers
   around the plan, with the same signatures; results agree with the previous
   implementation up to floating point rounding.
   :func:`~libertem_holo.base.reconstr.reconstruct_bf` gained the
   :code:`precision` parameter.
 * The :code:`precision` parameter now selects the computation dtype
   explicitly: :code:`True` transforms the hologram in float64 (complex128
   results), :code:`False` in float32 (complex64 results). Previously, float32
   holograms were transformed in float32 regardless, and
   :code:`precision=False` still returned complex128.
 * numpy >= 2.0 is now required.

[Feature] Pairwise graph registration of frame stacks
=====================================================

 * :func:`~libertem_holo.base.convenience.reconstruct_stack` takes a
   :code:`registration`: with :code:`"anchored"` (or an
   :class:`~libertem_holo.base.align.AnchoredRegistration` with
   parameters), batches of frames are registered pairwise, each measurement weighted by its
   peak-to-sidelobe ratio (:func:`~libertem_holo.base.align.peak_to_sidelobe_ratio`),
   and solved jointly by a robust least-squares fit
   (:func:`~libertem_holo.base.align.solve_graph`); the per-pair measurements
   are stored in :code:`metadata["registration"]`. The default
   (:code:`"star"`, every frame against the first) is unchanged.
 * New :class:`~libertem_holo.base.align.WaveCorrelator` for registering the
   complex waves, and :attr:`~libertem_holo.base.align.Correlator.correlate_on`
   to tell :func:`~libertem_holo.base.convenience.reconstruct_stack` what a
   correlator registers.
 * :meth:`~libertem_holo.base.align.Correlator.prepare_input` of the
   FFT-based correlators now returns a
   :class:`~libertem_holo.base.align.PreparedImage` holding the spectrum, so
   each image is transformed once however many pairs it takes part in; its
   :code:`image` property recovers the pre-processed image.
   :func:`~libertem_holo.base.align.align_stack` returns it as the reference.

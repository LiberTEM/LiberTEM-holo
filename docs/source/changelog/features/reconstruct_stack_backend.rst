[Feature] CPU backend for :code:`reconstruct_stack`
===================================================

 * :func:`libertem_holo.base.convenience.reconstruct_stack` now takes a
   :code:`backend` parameter. With :code:`backend="libertem"`, the
   reconstruction runs as a :class:`~libertem_holo.udf.reconstr.HoloReconstructUDF`
   on a LiberTEM :code:`Context` passed in as :code:`ctx`, allowing for fast
   reconstruction on the CPU. The default remains :code:`"cupy"`.
 * :meth:`libertem_holo.base.io.InputData.into_libertem_dataset` now also
   supports data created from arrays, as an in-memory dataset.

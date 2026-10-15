[Misc] Threaded frame registration
==================================

 * :func:`~libertem_holo.base.convenience.reconstruct_stack` registers and
   shifts the frames of a stack in a thread pool (:code:`align_threads`,
   default: the number of CPUs), with BLAS limited to one thread meanwhile.
   Aligning 30 frames of 480x464 pixels takes about a tenth of the time it
   did; the results are unchanged.

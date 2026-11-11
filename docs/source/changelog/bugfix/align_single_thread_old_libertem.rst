[Bugfix] Stack alignment with LiberTEM before 0.16
==================================================

 * :func:`~libertem_holo.base.convenience.reconstruct_stack` failed with
   :code:`TypeError: '>' not supported between instances of 'NoneType' and
   'int'` when aligning on a single thread (the default for the
   :code:`"cupy"` backend) with LiberTEM older than 0.16.

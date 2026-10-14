[Feature] InputData from LiberTEM DataSet
=========================================

 * Add :meth:`libertem_holo.base.io.InputData.load_from_libertem_dataset`,
   which wraps a LiberTEM DataSet and lazily reads slices using
   :class:`~libertem.udf.raw.PickUDF`.

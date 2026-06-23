.. testsetup:: *

   import os
   import pathlib
   base_path = os.environ.get('TESTDATA_BASE_PATH')
   if base_path is None:
      path_to_data = None
   else:
      path_to_data = pathlib.Path(base_path) / 'dm' / '3D' / 'alpha-50_obj.dm3'
   glob_base_path = pathlib.Path(base_path) / 'dm'


Input and Output
================

Loading holograms
-----------------

In our :ref:`I/O module <io api>`, we provide helpers for both loading data,
and after reconstruction, saving the results in a simple numpy npz file.

.. testcode:: load_data
   :skipif: path_to_data is None

   from libertem_holo.base.io import InputData

   # insert the full path to a dm3 or dm4 file here:
   input_data = InputData.load_from_dm(path_to_data)


The data itself and the most important metadata is directly available:

.. testcode:: load_data
   :skipif: path_to_data is None

   print(f"shape: {input_data.shape}")
   print(f"shape of the first image: {input_data.data[0].shape}")
   print(f"pixel size: {input_data.pixelsize} m")
   print(f"total expsure time: {input_data.exposure_time} s")

This code will print, for example:

.. testoutput:: load_data

   shape: (20, 3838, 3710)
   shape of the first image: (3838, 3710)
   pixel size: 1.671157330274582e-10 m
   total expsure time: 120.0 s

You can access the holograms one by one. This is a good pattern,
as only memory for a single hologram is used at a time:

.. testcode:: load_data
   :skipif: path_to_data is None

   for idx in range(20):
       print(f"sum of image {idx}: {int(input_data.data[idx].sum())}")

.. testoutput:: load_data

   sum of image 0: 833924288
   sum of image 1: 834060160
   sum of image 2: 833901824
   sum of image 3: 833629056
   sum of image 4: 833900736
   sum of image 5: 833789056
   sum of image 6: 833937536
   sum of image 7: 833730560
   sum of image 8: 833539456
   sum of image 9: 833594048
   sum of image 10: 833556544
   sum of image 11: 833551680
   sum of image 12: 833571072
   sum of image 13: 833774208
   sum of image 14: 833902912
   sum of image 15: 833897024
   sum of image 16: 834322880
   sum of image 17: 834512896
   sum of image 18: 834958976
   sum of image 19: 835722176

Loading stacks from multiple files
----------------------------------

The above example loads a stack from a single file. In many
cases, holograms in a stack are saved to individual files.
When using the :class:`libertem_holo.base.io.InputData` class,
there should be little difference if the data comes from
a single-file or a multi-file stack.
Multi-file data sets can be loaded with the
:func:`libertem_holo.base.io.InputData.load_from_glob` method:

.. testcode:: load_from_glob
   :skipif: path_to_data is None

   from libertem_holo.base.io import InputData

   # insert the full path to a dm3 or dm4 file here:
   input_data_glob = InputData.load_from_glob(
       base_path=glob_base_path,
       pattern="2018-7-17 15_29_*.dm4",
   )
   print(f"stack shape: {input_data_glob.shape}")

.. testoutput:: load_from_glob

   stack shape: (10, 3838, 3710)

The tags for each file can be accessed individually:

.. testcode:: load_from_glob
   :skipif: path_to_data is None

   for idx in range(9):
       tags = input_data_glob.tags_for_slice(idx)
       print(f"timestamp {idx}: {tags['DataBar Acquisition Time (OS)']}")
       print(f"shape of hologram {idx}: {input_data_glob.data[idx].shape}")

.. testoutput:: load_from_glob

    timestamp 0: 1.3176307760097974e+17
    shape of hologram 0: (3838, 3710)
    timestamp 1: 1.31763077622961e+17
    shape of hologram 1: (3838, 3710)
    timestamp 2: 1.3176307764495226e+17
    shape of hologram 2: (3838, 3710)
    timestamp 3: 1.3176307766695352e+17
    shape of hologram 3: (3838, 3710)
    timestamp 4: 1.3176307768895477e+17
    shape of hologram 4: (3838, 3710)
    timestamp 5: 1.3176307771094603e+17
    shape of hologram 5: (3838, 3710)
    timestamp 6: 1.317630777329373e+17
    shape of hologram 6: (3838, 3710)
    timestamp 7: 1.3176307775494854e+17
    shape of hologram 7: (3838, 3710)
    timestamp 8: 1.317630777769398e+17
    shape of hologram 8: (3838, 3710)

The :func:`libertem_holo.base.io.InputData.tags_for_slice` method
can also be used with single-file stacks, but will
give back the same tags for every slice.

Assumptions
-----------

We make a few assumptions:

- Pixel sizes are given in m/px and are the same in X and Y direction
- The data is either a single hologram or a 3D stack of holograms
- The shapes and data types of individual holograms in multi-file data sets are matching

.. note::

   If you are using UDFs for reconstruction, you need to use
   :ref:`the I/O functionality of LiberTEM <libertem:dataset api>` to load data.
   This is especially useful for reconstructing very large stacks.


Saving and loading results
--------------------------

Once you have reconstructed your data, you can save the results in a convenient
format:

.. testsetup:: results

    import numpy as np
    import tempfile
    import pathlib
    from libertem_holo.base.io import InputData, Results
    from libertem_holo.base.utils import HoloParams

    wave = np.random.random((128, 128)) + 1j * np.random.random((128, 128))
    phase = np.random.random((128, 128))
    brightfield = np.random.random((128, 128))
    temp_dir = tempfile.TemporaryDirectory()
    save_path = pathlib.Path(temp_dir.name) / 'test-1.npz'
    holo = np.random.random((256, 256))
    input_data = InputData.from_array(
        data=holo,
        exposure_time=24.6,
        pixelsize=0.1,
    )
    holo_params = HoloParams.from_hologram(
        holo,
        out_shape=(holo.shape[0] // 2, holo.shape[1] // 2),
    )

.. testcode:: results
   :skipif: path_to_data is None

   from libertem_holo.base.io import Results

   res = Results(
       complex_wave=wave,
       unwrapped_phase=phase,
       brightfield=brightfield,
       metadata={"custom": 12.34},
   )
   # save_path is the full or relative path that ends in .npz
   # where your data will be saved:
   print(f"saving to {save_path}")
   res.save(save_path)

.. testoutput:: results
   :options: +ELLIPSIS

   saving to ...

You can also include metadata from the input data and reconstruction parameters
in the results:

.. testcode:: results
   :skipif: path_to_data is None

   res.metadata_from_input(
       input_data=input_data,
       params=holo_params,  # a HoloParams instance
   )
   res.save(save_path)
   print(f"effective pixel size: {res.metadata['effective_pixelsize']} m")
   print(f"stack shape: {res.metadata['stack_shape']}")
   print(f"total exposure time: {res.metadata['exposure_time']} s")
   print(f"a custom value: {res.metadata['custom']}")

This code will print, for example:

.. testoutput:: results

    effective pixel size: 0.2 m
    stack shape: [1, 256, 256]
    total exposure time: 24.6 s
    a custom value: 12.34

The effective pixel size is the pixel size in the resulting phase image,
which is the original pixel size adjusted for the chosen output shape.
In case of stack reconstruction, the exposure time read from the input metadata
is multiplied by the number of frames in the stack.

Everything but the complex wave is optional, so the minimal result
looks like this:

.. testcode:: results
   :skipif: path_to_data is None

   from libertem_holo.base.io import Results

   res = Results(
       complex_wave=wave,
   )
   print(f"saving to {save_path}")
   res.save(save_path)

.. testoutput:: results
   :options: +ELLIPSIS

   saving to ...

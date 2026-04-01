.. testsetup:: *

   import os
   import pathlib
   base_path = os.environ.get('TESTDATA_BASE_PATH')
   path_to_data = pathlib.Path(base_path) / 'dm' / '3D' / 'alpha-50_obj.dm3'


Input and Output
================

Loading holograms
-----------------

In our :ref:`I/O module <io api>`, we provide helpers for both loading data,
and after reconstruction, saving the results in a simple numpy npz file.

.. testcode:: load_data

    from libertem_holo.base.io import InputData

    # insert the full path to a dm3 or dm4 file here:
    input_data = InputData.load_from_dm(path_to_data)


The data itself and the most important metadata is directly available:

.. testcode:: load_data

    print(f"shape: {input_data.data.shape}")
    print(f"pixel size: {input_data.pixelsize} nm")
    print(f"total expsure time: {input_data.exposure_time} s")

This code will print, for example:

.. testoutput:: load_data

    shape: (20, 3838, 3710)
    pixel size: 0.1671157330274582 nm
    total expsure time: 120.0 s

We make a few assumptions:

- Pixel sizes are given in nm/px and are the same in X and Y direction
- The data is either a single hologram or a 3D stack of holograms

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
    input_data = InputData(
        data=holo,
        exposure_time=24.6,
        pixelsize=0.1,
    )
    holo_params = HoloParams.from_hologram(
        holo,
        out_shape=(holo.shape[0] // 2, holo.shape[1] // 2),
    )

.. testcode:: results

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

    res.metadata_from_input(
        input_data=input_data,
        params=holo_params,  # a HoloParams instance
    )
    res.save(save_path)
    print(f"effective pixel size: {res.metadata['effective_pixelsize']} nm")
    print(f"stack shape: {res.metadata['stack_shape']}")
    print(f"total exposure time: {res.metadata['exposure_time']} s")
    print(f"a custom value: {res.metadata['custom']}")

This code will print, for example:

.. testoutput:: results

    effective pixel size: 0.2 nm
    stack shape: [256, 256]
    total exposure time: 24.6 s
    a custom value: 12.34

The effective pixel size is the pixel size in the resulting phase image,
which is the original pixel size adjusted for the chosen output shape.
In case of stack reconstruction, the exposure time read from the input metadata
is multiplied by the number of frames in the stack.


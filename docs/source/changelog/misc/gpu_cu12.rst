[Misc] GPU support for CUDA 12 drivers
=======================================

 * New extra :code:`gpu-cu12` for machines with a CUDA 12 driver: install
   :code:`libertem-holo[gpu-cu12]` with :code:`cupy-cuda12x` and a
   :code:`cuda-toolkit` no newer than the driver, for example
   :code:`cuda-toolkit==12.4.*` for a CUDA 12.4 driver. The tox environments
   :code:`py3XX-cupy12` test this setup.

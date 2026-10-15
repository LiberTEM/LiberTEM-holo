[Misc] Compile the CUDA kernels in CI, without a GPU
====================================================

 * New tox environment :code:`cuda-compile`, which compiles the CUDA kernels
   to PTX for the lowest and highest compute capability supported by the CUDA
   toolkit. It runs in GitHub Actions and needs no GPU.

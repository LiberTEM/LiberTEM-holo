LiberTEM-holo
=============

LiberTEM-holo is a Python project for holography reconstruction using LiberTEM. Electron Holograms are captured by interfering two parts of a coherent electron beam passing through (1) vacuum and (2) a sample. An electron biprism is used to interfere the two parts of this electron beam. The electron phase shift measured in The program is generally meant to reconstruct electron holograms collected from a magnetic material to construct the magnetic signal from the material. More information can be found in `wiki <https://en.wikipedia.org/wiki/Electron_holography>`_ or `Handbook of Microscopy <https://doi.org/10.1007/978-3-030-00069-1_16>`_. The program focuses on the steps required to reconstruct the electron phase from the electron hologram. In general, the steps required to reconstruct the electron holograms to phase images,

(1) Refining Complex image estimation from electron holograms - Implementation allows for automatic detection of the sideband from the fft of the electron hologram . Two customizable filters - circle and line `butterworths filters <https://en.wikipedia.org/wiki/Butterworth_filter>`_ are available to select the side band and filter out the fresnal noise in the image. The complex image can be calculated from the filtered fft. The code allows for the reconstruction from a single hologram or for a hologram and a reference image to compensate for detection noise.

(2) Stack alignment - If an image stack was acquired for improving the signal, the sample drift and biprism drift can be compensated using the code. The sample drift is calculated using cross-correlation. The `cross correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ is usually performed on the real part of the image and the drift is applied on the complex stack. The biprism drift appears as a phase drift in the complex image. This phase compensation using the `angular synchronization <https://arxiv.org/pdf/2005.02032>`_ method by `Filbir et al. <https://doi.org/10.1007/s00041-021-09834-1>`_. After drift correction and phase compensation, the complex stacks are summed up to create a single complex image. The phase of this complex image is the phase difference induced by the sample - this signal could be due to electric or magnetic field (as per `Aharonov Bohm effect <https://en.wikipedia.org/wiki/Aharonov%E2%80%93Bohm_effect>`_).

(3) Phase image manipulation and plotting - Any residual phase ramps on the images can be removed at this stage and the phase and the contour plots can be plot from the phase image from the curl of the phase image.

A sample jupyter notebook demonstrating the processing of electron holograms are available here (need to add the page). The data collected from a magnetic TEM lamella used for the data is available `here <10.5281/zenodo.15222399.>`_.

Not released yet - please install via git!


Installation
-------
.. code-block:: shell

  $conda create -n LiberTEM_holo python=3.12
  $conda activate LiberTEM_holo
  $pip install git+https://github.com/LiberTEM/LiberTEM-holo.git

Input File formats and i/o operations
-------
LiberTEM-holo was designed specfically for inputting larger stacks of images without loading them into memory. The implementation is borrowed from LiberTEM.

(1) Gatan Digital Micrograph (.dm3, .dm4)
(2) Other file formats (Need to check with Alex)



Associated functions
-------
(0) HoloReconstructUDF
(1) clipped
(2) disk_aperture
(3) butterworth_disk
(4) butterworth_line
(5) estimate_sideband_position
(6) estimate_sideband_size
(7) freq_array
(8) HoloParams
(9) get_slice_fft
(10) fft_shift_coords
(11) remove_phase_ramp
(12) phase_offset_correction
(13) align_stack
(14) ImageCorrelator


License
-------

LiberTEM-holo is licensed under GPLv3.

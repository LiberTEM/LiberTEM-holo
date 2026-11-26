Changelog
=========

.. _continuous:

1.2.0.dev0 (continuous)
#######################

.. toctree::
   :glob:

   changelog/*/*

.. _latest:

1.1.0 / (2026-11-26)
####################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.24510929.svg
  :target: https://doi.org/10.5281/zenodo.24510929

This release makes hologram reconstruction considerably faster, adds a CPU
backend and pairwise graph registration to
:func:`~libertem_holo.base.convenience.reconstruct_stack`, and adds two
experimental phase unwrapping methods (IRLS, ADMM) that also run on the GPU.
numpy >= 2.0 is now required.

A new convenience functions is available:
:func:`~libertem_holobase.convenience.reconstruct_hologram` for
easier reconstruction of a single hologram.

Utility functions for fringe analysis are now available:
:func:`~libertem_holo.base.utils.fringe_contrast` for measuring fringe constrast,
and :func:`~libertem_holo.base.utils.fringe_period`.

Features
--------

* Added :class:`~libertem_holo.base.plan.ReconstructionPlan`, which
  precomputes once which part of a hologram's spectrum ends up in the output
  window, and then reconstructs each hologram with a real-to-complex FFT,
  transforming only the needed columns of the spectrum and gathering the
  window directly, without full-size :code:`roll`, :code:`fftshift` or
  scaling passes. Build it with
  :meth:`~libertem_holo.base.plan.ReconstructionPlan.from_params` and reuse
  it for a whole stack. Reconstructing the wave and the brightfield image of
  a 4096x4096 hologram at 1024x1024 takes about a sixth of the time it did.
* :func:`~libertem_holo.base.reconstr.reconstruct_frame`,
  :func:`~libertem_holo.base.reconstr.reconstruct_bf` and
  :class:`~libertem_holo.base.reconstr.HologramFFT` are now thin wrappers
  around the plan, with the same signatures.
  :func:`~libertem_holo.base.reconstr.reconstruct_bf` gained the
  :code:`precision` parameter. The index arrays of the plan are cached per
  geometry and array module, on the GPU for cupy, so calling the wrappers
  once per frame does not transfer anything per call. An :code:`out_shape`
  larger than the hologram, or a :code:`slice_fft` that is not the centered
  slice of :func:`~libertem_holo.base.utils.get_slice_fft`, now raise
  :code:`ValueError` instead of producing wrapped-around results.
* The :code:`precision` parameter of these functions now selects the
  computation dtype explicitly: :code:`True` transforms the hologram in
  float64 (complex128 results), :code:`False` in float32 (complex64 results).
  Previously, float32 holograms were transformed in float32 regardless, and
  :code:`precision=False` still returned complex128.
* numpy >= 2.0 is now required.
* Added :class:`~libertem_holo.base.reconstr.HologramFFT`, which computes the
  real-to-complex FFT of a hologram once and keeps it as
  :attr:`~libertem_holo.base.reconstr.HologramFFT.half_spectrum`. It can be
  passed to :func:`~libertem_holo.base.reconstr.reconstruct_frame` and
  :func:`~libertem_holo.base.reconstr.reconstruct_bf` in place of the
  hologram, so that reconstructing both the complex wave and the brightfield
  image only needs a single FFT.
  :func:`~libertem_holo.base.convenience.reconstruct_stack` and
  :class:`~libertem_holo.udf.reconstr.HoloReconstructUDF` share the FFT
  between the wave and the brightfield image of each hologram.
* :func:`~libertem_holo.base.convenience.reconstruct_stack` now takes a
  :code:`backend` parameter. With :code:`backend="libertem"`, the
  reconstruction runs as a :class:`~libertem_holo.udf.reconstr.HoloReconstructUDF`
  on a LiberTEM :code:`Context` passed in as :code:`ctx`, allowing for fast
  reconstruction on the CPU. The default remains :code:`"cupy"`.
  :meth:`~libertem_holo.base.io.InputData.into_libertem_dataset` now also
  supports data created from arrays, as an in-memory dataset.
* :func:`~libertem_holo.base.convenience.reconstruct_stack` takes a
  :code:`registration`: with :code:`"anchored"` (or an
  :class:`~libertem_holo.base.align.AnchoredRegistration` with
  parameters), batches of frames are registered pairwise, each measurement
  weighted by its peak-to-sidelobe ratio
  (:func:`~libertem_holo.base.align.peak_to_sidelobe_ratio`), and solved
  jointly by a robust least-squares fit
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
* New :func:`~libertem_holo.base.convenience.reconstruct_hologram`
  reconstructs the complex wave from a single hologram, with the parameters
  given as a :class:`~libertem_holo.base.utils.HoloParams` instance, or
  estimated from the hologram with
  :meth:`~libertem_holo.base.utils.HoloParams.from_hologram` if not given
  (:issue:`176`).
* Added :meth:`~libertem_holo.base.io.InputData.load_from_libertem_dataset`,
  which wraps a LiberTEM DataSet and lazily reads slices using
  :class:`~libertem.udf.raw.PickUDF`.
* New :func:`~libertem_holo.base.utils.fringe_contrast` to measure the
  fringe contrast (visibility) of a hologram in a region, either from the
  sideband and centerband amplitudes (:code:`method="sideband"`) or from a
  line profile across the fringes (:code:`method="profile"`), and
  :func:`~libertem_holo.base.utils.fringe_contrast_map` for a map of the
  contrast. :func:`~libertem_holo.base.utils.fringe_period` gives the
  direction and period of the fringes from the sideband position.
* New phase unwrapping by iteratively reweighted least squares (IRLS), based
  on Dubois-Taine, Akiki and d'Aspremont, arXiv:2401.09961 (2024). It keeps
  discontinuities local, and the result differs from the wrapped phase by
  whole multiples of :math:`2\pi`. Use it with
  :func:`~libertem_holo.base.unwrap.unwrap_phase_irls`, or with
  :code:`method="irls"` in :func:`~libertem_holo.base.unwrap.phase_unwrap`.
  Experimental, please report any issues you encounter.
* New phase unwrapping by ADMM, with the objective of
  :func:`~libertem_holo.base.unwrap.unwrap_phase_irls` in fewer iterations,
  based on Rouet-Leduc and Hulbert, Remote Sensing 18, 1801 (2026). Use it
  with :func:`~libertem_holo.base.unwrap.unwrap_phase_admm`, or with
  :code:`method="admm"` in :func:`~libertem_holo.base.unwrap.phase_unwrap`.
  Both IRLS and ADMM solve on the GPU for cupy input, keeping their buffers
  in a :class:`~libertem_holo.base.workspace.Workspace`, and leave NaN pixels
  NaN. Experimental, please report any issues you encounter.

Bugfixes
--------

* :func:`~libertem_holo.base.filters.butterworth_line` now always points
  from the center towards :code:`sb_position`. Before, the direction was
  reversed when the slope of the line was between 0 and 1, so that in the
  aperture from :meth:`~libertem_holo.base.utils.HoloParams.from_hologram`
  the line went through the center of the sideband (and, in
  :code:`aperture_bf`, through the central band). For sideband positions that
  were not affected, the aperture is unchanged.
* Sideband positions on the horizontal or vertical axis now get a line
  filter; before, the line was missing.
* :func:`~libertem_holo.base.filters.butterworth_line` raises a
  :code:`ValueError` if :code:`sb_position` is at the center, instead of
  returning NaNs.
* When estimating the sideband position, the mask removing the central band
  was shifted by one pixel for holograms with an odd number of rows or
  columns. Affects
  :func:`~libertem_holo.base.utils.estimate_sideband_position` and
  :meth:`~libertem_holo.base.utils.HoloParams.from_hologram`.
* :func:`~libertem_holo.base.reconstr.phase_offset_correction` now passes a
  single image through unchanged, given either as a stack of shape
  :code:`(1, sy, sx)` or as a 2D array of shape :code:`(sy, sx)`. Before, it
  failed with :code:`ArpackError: Starting vector is zero`. This also fixes
  :func:`~libertem_holo.base.convenience.reconstruct_stack` with a single
  hologram as :code:`stack_ref` (or :code:`stack`) (:issue:`173`).
* :func:`~libertem_holo.base.convenience.reconstruct_stack` failed with
  :code:`TypeError: '>' not supported between instances of 'NoneType' and
  'int'` when aligning on a single thread (the default for the
  :code:`"cupy"` backend) with LiberTEM older than 0.16.

Obsolescence
------------

* :func:`~libertem_holo.base.filters.line_filter`,
  :func:`~libertem_holo.base.utils.draw_lf_rect` and
  :func:`~libertem_holo.base.utils.line_filter_coords` are deprecated; use
  :func:`~libertem_holo.base.filters.butterworth_line` instead.

Miscellaneous
-------------

* :func:`~libertem_holo.base.utils.estimate_sideband_position`, and with it
  :meth:`~libertem_holo.base.utils.HoloParams.from_hologram`, now searches
  the spectrum for the sideband with a numba kernel (CPU and GPU) instead of
  building masks and intermediate arrays, and uses the threaded FFT of
  :mod:`scipy.fft` on the CPU, which makes the one-off search about six times
  faster on a 4096x4096 hologram. The sideband position found is the same as
  before, except for the fix for odd shapes (see Bugfixes). An invalid
  :code:`sb` argument now raises a :code:`ValueError` instead of returning
  :code:`(0.0, 0.0)`.
* :class:`~libertem_holo.base.plan.ReconstructionPlan` gathers the output
  window from the spectrum with a numba kernel (a CUDA kernel on cupy) that
  computes the spectrum index of each element on the fly and applies the
  conjugation of mirrored columns and the aperture in the same pass. This
  replaces the index arrays of the size of the output window and two
  passes over it; the results are unchanged.
* :func:`~libertem_holo.base.convenience.reconstruct_stack` registers and
  shifts the frames of a stack in a thread pool (:code:`align_threads`,
  default: the number of CPUs for numpy, 1 for cupy; the threads use the
  caller's current GPU), with BLAS limited to one thread meanwhile.
  Aligning 30 frames of 480x464 pixels takes about a tenth of the time it
  did; the results are unchanged.
* New extra :code:`gpu-cu12` for machines with a CUDA 12 driver: install
  :code:`libertem-holo[gpu-cu12]` with :code:`cupy-cuda12x` and a
  :code:`cuda-toolkit` no newer than the driver, for example
  :code:`cuda-toolkit==12.4.*` for a CUDA 12.4 driver. The tox environments
  :code:`py3XX-cupy12` test this setup.
* New tox environment :code:`cuda-compile`, which compiles the CUDA kernels
  to PTX for the lowest and highest compute capability supported by the CUDA
  toolkit. It runs in GitHub Actions and needs no GPU.
* New dev container configuration in :code:`.devcontainer/`, for VS Code and
  the devcontainer CLI: an editable install with cupy, the Nsight Systems and
  Nsight Compute profilers, neovim and Claude Code. See
  :code:`.devcontainer/README.md`.
* Release and release candidate tags now upload a draft to Zenodo. The
  record is found by its concept ID, as in LiberTEM (secret
  :code:`LT_RELEASE_UPLOAD_ZENODO_CONCEPT` replaces
  :code:`LT_RELEASE_UPLOAD_ZENODO_PARENT`), and a failed upload fails the CI
  release job. New script :code:`scripts/zenodo_add_funding` adds the funding
  entries from :code:`packaging/zenodo-funding.json` (AIDAS) to the draft,
  which Zenodo's deposit API can't set.
* Updated the Zenodo metadata: license GPL-3.0-or-later, grants, keywords,
  creators and contributors. The package keywords in :code:`pyproject.toml`
  now match the Zenodo keywords. Pinned :code:`bibtexparser<2` for the
  release scripts; they use the 1.x API.

1.0.0 / (2026-07-08)
####################

First stable release. Features include:

- Reconstruction functions for single holograms
- Stack reconstruction functions including alignment and phase offset correction
- Image correlation class for various alignment strategies
- Three phase unwrapping algorithms
- Convenience functions for stack reconstruction and magnetic phase plots
- Bright field reconstruction
- GPU-Accelerated reconstruction and filtering
- Input/Output helpers for ...
    - loading dm3/dm4 stacks and single holograms
    - saving typical stack reconstruction results
- Integration with LiberTEM: UDF reconstruction for efficient reconstruction
  of large stacks, using CPU or GPU parallelization

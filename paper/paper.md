---
title: 'LiberTEM-holo: a Python package for electron holography'
tags:
  - Python
  - TEM
  - holography
  - magnetic induction
  - electrostatic potential
authors:
  - name: Malika Khelfallah
    orcid: 0000-0001-6940-5792
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Alexander Clausen
    orcid: 0000-0002-9555-7455
    equal-contrib: true
    affiliation: 1
  - name: David Landers
    orcid: 0000-0002-3295-9473
    affiliation: 2
  - name: Matthew Bryan
    orcid: 0000-0001-9134-384X
    affiliation: 2
  - name: Joseph Vim Vals
    orcid: 0009-0006-3796-9466
    affiliation: 1
  - name: Rafal Dunin-Borkowski
    orcid: 0000-0001-8082-0647
    affiliation: 1
  # TODO: add Patrick Adrian; packaging/creators.json gives "Joseph Vimal Vas"
  # and CEA-IRIG (Landers) / CEA-Leti (Bryan) as affiliations.
affiliations:
 - name: Ernst Ruska-Centre, Forschungszentrum Jülich, Germany
   index: 1
   ror: 02nv7yv05
 - name: LETI, Grenoble, France
   index: 2
date: 20 November 2026
bibliography: paper.bib
---

# Summary

LiberTEM-holo is an open-source Python package designed for the reconstruction of off-axis
electron holography data [@Lichte2008; @DuninBorkowski2019]. It offers a complete workflow,
from raw holograms to reconstructed electron phase images. This includes Fourier filtering,
sideband detection, drift correction, phase alignment, phase unwrapping, and magnetic
induction mapping. With GPU acceleration through CuPy [@Okuta2017], it can efficiently
reconstruct hologram stacks containing tens of frames and several gigabytes of data while
maintaining compatibility with CPU-only systems. The project is released under the GPLv3
license and is distributed through PyPI and GitHub.

# Statement of need

Electron holography reconstruction is often performed using proprietary software, which
limits automation, extensibility, and reproducibility. Modern experiments increasingly
generate stacks of holograms that require efficient batch processing and alignment.
LiberTEM-holo addresses these challenges by offering a free, open-source Python library with
GPU acceleration and a programmable interface for automated reconstruction pipelines. This
package allows researchers to efficiently process large datasets while seamlessly
integrating into scientific Python workflows.

The package is written for microscopists who acquire off-axis electron holograms, for
example to map electrostatic potentials or magnetic fields in materials, and who need to
process hologram stacks reproducibly. It is also meant as a platform for developers of new
reconstruction and phase-processing methods, who can replace single steps of the workflow
while reusing the rest.

# State of the field

The proprietary DigitalMicrograph plugin HoloWorks is a widely used solution for electron
holography reconstruction. Although it is effective for interactive reconstruction,
HoloWorks is not open source and offers limited automation capabilities. Additionally, it
does not exploit GPU acceleration for large datasets.

The main open-source alternative is HoloSpy [@holospy], which was split out of HyperSpy in
2023. HoloSpy reconstructs individual holograms by sideband filtering, optionally normalized
by a vacuum reference hologram, computes fringe statistics, and unwraps the phase through
HyperSpy. Its documentation does not describe GPU acceleration, the registration of hologram
stacks, or the correction of the phase drift caused by biprism drift, which are needed to
average long hologram stacks.

LiberTEM-holo complements the existing ecosystem by providing a scriptable, open-source
alternative with a focus on scalable reconstruction, stack processing, and advanced phase
analysis. This includes multiple phase-unwrapping methods and image registration tools. We
built it on LiberTEM [@Clausen2020] instead of extending HoloSpy because stack
reconstruction depends on what LiberTEM already provides: lazy, chunked reading of large
datasets, readers for many detector and microscope file formats, and parallel execution on
CPUs. Building on LiberTEM also lets the same reconstruction code run inside LiberTEM
processing pipelines.

# Software design

LiberTEM-holo is implemented in Python and builds upon NumPy [@Harris2020], Numba
[@Lam2015], scikit-image [@vanderWalt2014] and LiberTEM, together with CuPy for GPU
computing. The package has two layers. The base layer (`libertem_holo.base`) consists of
functions that take NumPy or CuPy arrays and an array module argument, so that the same code
runs on CPU and GPU. The second layer (`libertem_holo.udf`) wraps the reconstruction as a
LiberTEM user-defined function (UDF), which LiberTEM executes in parallel on the partitions
of a dataset.

To reconstruct long stacks quickly, a `ReconstructionPlan` is
computed once per frame shape and sideband position: it determines which part of the
hologram's spectrum contributes to the output, so that each hologram is transformed with a
real-to-complex FFT in which only the required columns are computed, and the sideband window
is gathered by a Numba kernel (a CUDA kernel on the GPU). The complex wave and the
brightfield image, used for registration, are reconstructed from the same transform. The
trade-off is that a plan is specific to one acquisition geometry, which fits hologram stacks
recorded with fixed settings.

`reconstruct_stack` processes a stack in three phases: all holograms are reconstructed,
the frames are registered, and the aligned waves are averaged. The reconstruction runs either
on the GPU with CuPy, or on the CPU as a LiberTEM UDF, which avoids the GPU memory limit and
uses all CPU cores. Registration is pluggable: correlator classes register the frames on the
amplitude, brightfield, phase, phase gradient or complex wave, using subpixel cross-correlation
[@GuizarSicairos2008]. By default every frame is registered against the first one; for noisy
or strongly drifting stacks, frames can instead be registered pairwise in batches, and the
drifts are found by a robust least-squares fit weighted by the quality of each correlation
peak. Before averaging, the phase offsets between frames, caused by biprism drift, are
corrected by angular synchronization [@Filbir2021], which solves for all offsets at once
from an eigenvector problem instead of iterating.

For phase unwrapping, users can choose between the scikit-image algorithm, a quality-guided
algorithm based on the phase derivative variance, and a fast Laplacian method
[@Schofield2003] that runs on the GPU and is meant for quick previews. Input data are read
lazily from DigitalMicrograph files, sets of files, or any LiberTEM dataset, preserving
metadata such as pixel size, and results are saved together with metadata from the input
and the reconstruction parameters. A hologram simulator generates synthetic test
data, which the automated tests use on Linux, Windows and macOS for all supported Python versions; tests on GPUs and on
experimental datasets run on dedicated machines. Documentation and example notebooks
facilitate adoption and reproducible analysis.

# Research impact statement

LiberTEM-holo is used in the ER-C 1 group at the Ernst Ruska-Centre, where at least four
researchers use it to process off-axis electron holography datasets, and by researchers at
CEA Grenoble for hologram reconstruction and magnetic imaging analysis. Two experimental
datasets, hologram stacks of iron oxide nanoparticles and of a magnetic yttrium iron garnet
lamella, are publicly available on Zenodo (records 21108532 and 15222400) together with
example notebooks that reconstruct them, so that the complete workflow can be reproduced.
<!-- TODO: add publications that use LiberTEM-holo (published, in review or submitted). -->

# AI usage disclosure

<!-- TODO: confirm that this list is complete, including AI use not recorded in commits. -->
Generative AI tools were used during the development of the software, its documentation and
this manuscript. Claude (Anthropic), used through Claude Code with the models Claude Fable
5.1 and Claude Opus 5.5, assisted with code, tests and documentation changes since October
2026; these commits are marked with a co-author line in the version history. Claude Opus 5.5
was also used to help draft and refine the wording of this manuscript. The authors reviewed, edited, and
validated all AI-assisted code and text, made the design decisions, and accept full
responsibility for the software and the final manuscript.

# Acknowledgements

<!-- TODO: confirm which grants funded LiberTEM-holo. packaging/zenodo-upload.json lists
European Union Horizon 2020 grants 780487, 686053 and 823717; libertem.github.io names them
as ERC Proof-of-Concept VIDEO, CritCat and ESTEEM3, but does not say which LiberTEM project
each funded. -->
We thank Thibaud Denneulin and András Kovács for discussions, data and feedback.
TODO: funding statement, and whether the funders had any role in the design of the software
or in the writing of this manuscript.

# References

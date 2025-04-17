|docs|_ |gitter|_ |github|_ |precommit|_ |pypi|_

.. |docs| image:: https://img.shields.io/badge/%F0%9F%95%AE-docs-green.svg
.. _docs: https://libertem.github.io/LiberTEM-holo/

.. |gitter| image:: https://badges.gitter.im/join_chat.svg
.. _gitter: https://gitter.im/LiberTEM/Lobby

.. |github| image:: https://img.shields.io/badge/GitHub-GPLv3-informational
.. _github: https://github.com/LiberTEM/LiberTEM-holo/

.. |precommit| image:: https://results.pre-commit.ci/badge/github/LiberTEM/LiberTEM-holo/master.svg
.. _precommit: https://results.pre-commit.ci/latest/github/LiberTEM/LiberTEM-holo/master

.. |pypi| image:: https://badge.fury.io/py/libertem-holo.svg
.. _pypi: https://pypi.org/project/libertem-holo/

LiberTEM-holo
=============

LiberTEM-holo is a Python project for holography reconstruction using LiberTEM.
Electron Holograms are captured by interfering two parts of a coherent electron
beam passing through (1) vacuum and (2) a sample. An electron biprism is used
to interfere the two parts of this electron beam. The interference pattern is
recorded, and the complex waveform can be reconstructed from it using this
software. From the phase of the complex waveform one can then for example
determine the magnetic signal from the sample.

More information can be found on `Wikipedia 
<https://en.wikipedia.org/wiki/Electron_holography>`_ or the `Handbook of
Microscopy <https://doi.org/10.1007/978-3-030-00069-1_16>`_.

This software focuses on the steps required to reconstruct the electron phase
from stacks of electron holograms. In general, the steps required to
reconstruct the electron holograms to phase images are:

(1) Complex image reconstruction from electron holograms - most parameters can
    be determined automatically. Two customizable filters - circle and line
    `butterworths filters <https://en.wikipedia.org/wiki/Butterworth_filter>`_ are
    available to select the side band and filter out the fresnel fringes in the
    image. The complex image can be calculated from the filtered fft.

(2) Stack alignment - if an image stack was acquired for improving the signal,
    the sample drift and biprism drift can be compensated. The
    sample drift is calculated using cross-correlation. The `cross correlation
    <https://en.wikipedia.org/wiki/Cross-correlation>`_ is usually performed on the
    real part of the image and the drift is applied on the complex stack. The
    biprism drift appears as a phase drift in the complex image. This phase drift
    is compensated using the `angular synchronization
    <https://arxiv.org/pdf/2005.02032>`_ method by `Filbir et al.
    <https://doi.org/10.1007/s00041-021-09834-1>`_. After drift correction and
    phase compensation, the complex stacks are summed up to create a single complex
    image. The phase of this complex image is the phase difference induced by the
    sample - this signal could be due to electric or magnetic field (as per
    `Aharonov Bohm effect <https://en.wikipedia.org/wiki/Aharonov%E2%80%93Bohm_effect>`_).

(3) Phase image processing and visualization - residual phase ramps
    can be removed at this stage and the phase and the contour can be plotted.

A sample jupyter notebook is available `in the GitHub repository
<https://github.com/LiberTEM/LiberTEM-holo/tree/master/notebooks>`_. The data
collected from a magnetic TEM lamella used for the example is available
`on Zenodo <https://zenodo.org/records/15222400>`_.

Not released yet - please install via git!

Installation
------------
.. code-block:: shell

  $ conda create -n holo python=3.12
  $ conda activate holo
  $ pip install git+https://github.com/LiberTEM/LiberTEM-holo.git

Input File formats
------------------
LiberTEM-holo was designed specfically for working on larger stacks of images
without loading them into memory. In addition to files created by Gatan Digital
Micrograph (.dm3, .dm4), any file format
`supported by LiberTEM <https://github.com/LiberTEM/LiberTEM>`_ can be loaded.

License
-------

LiberTEM-holo is licensed under GPLv3.

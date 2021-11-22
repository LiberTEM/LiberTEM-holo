[Bugfix] Correct Poisson noise as a function of counts
======================================================

* Factor the counts into the Poisson noise calculation correctly.
  Previously, :code:`counts` behaved as the detector gain and the
  :code:`poisson_noise` parameter as the gain divided by the actual
  counts. (`LiberTEM/LiberTEM#1156 <https://github.com/LiberTEM/LiberTEM/issues/1156>`_, :pr:`24`)

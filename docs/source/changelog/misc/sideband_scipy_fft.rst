[Misc] Faster sideband search
=============================

 * :func:`~libertem_holo.base.utils.estimate_sideband_position` and therefore
   :meth:`~libertem_holo.base.utils.HoloParams.from_hologram` use the threaded
   FFT of :mod:`scipy.fft` on the CPU, which makes the one-off search about
   six times faster on a 4096x4096 hologram.

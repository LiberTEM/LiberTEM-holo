import numpy as np

from libertem.udf import UDF

from libertem_holo.base.reconstr import get_aperture


class HoloReconstructUDF(UDF):
    """
    Reconstruct off-axis electron holograms using a Fourier-based method.

    Running :meth:`~libertem.api.Context.run_udf` on an instance of this class
    will reconstruct a complex electron wave. Use the :code:`wave` key to access
    the raw data in the result.

    See :ref:`holography app` for detailed application example

    .. versionadded:: 0.3.0

    Examples
    --------
    >>> shape = tuple(dataset.shape.sig)
    >>> sb_position = [2, 3]
    >>> sb_size = 4.4
    >>> holo_udf = HoloReconstructUDF(out_shape=shape,
    ...                               sb_position=sb_position,
    ...                               sb_size=sb_size)
    >>> wave = ctx.run_udf(dataset=dataset, udf=holo_udf)['wave'].data
    """

    def __init__(self,
                 out_shape,
                 sb_position,
                 sb_size,
                 sb_smoothness=.05,
                 precision=True):
        """
        out_shape : (int, int)
            Shape of the returned complex wave image. Note that the result should fit into the
            main memory.
            See :ref:`holography app` for more details

        sb_position : tuple, or vector
            Coordinates of sideband position with respect to non-shifted FFT of a hologram

        sb_size : float
            Radius of side band filter in pixels

        sb_smoothness : float, optional (Default: 0.05)
            Fraction of `sb_size` over which the edge of the filter aperture to be smoothed

        precision : bool, optional, (Default: True)
            Defines precision of the reconstruction, True for complex128 for the resulting
            complex wave, otherwise results will be complex64
        """
        super().__init__(out_shape=out_shape,
                         sb_position=sb_position,
                         sb_size=sb_size,
                         sb_smoothness=sb_smoothness,
                         precision=precision)

    def get_result_buffers(self):
        """
        Initializes :class:`~libertem.common.buffers.BufferWrapper` objects for reconstructed
        wave function

        Returns
        -------
        A dictionary that maps 'wave' to the corresponding
        :class:`~libertem.common.buffers.BufferWrapper` objects
        """
        extra_shape = self.params.out_shape
        if not self.params.precision:
            dtype = np.complex64
        else:
            dtype = np.complex128
        return {
            "wave": self.buffer(kind="nav", dtype=dtype, extra_shape=extra_shape)
        }

    def get_task_data(self):
        """
        Updates `task_data`

        Returns
        -------
        kwargs : dict
        A dictionary with the following keys:
            kwargs['aperture'] : array-like
            Side band filter aperture (mask)
            kwargs['slice'] : slice
            Slice for slicing FFT of the hologram
        """

        slice_fft, aperture = get_aperture(
            out_shape=self.params.out_shape,
            sb_size=self.params.sb_size,
            sb_smoothness=self.params.sb_smoothness,
            sig_shape=self.meta.partition_shape.sig,
        )

        kwargs = {
            'aperture': self.xp.array(aperture),
            'slice': slice_fft
        }
        return kwargs

    def process_frame(self, frame):
        """
        Reconstructs holograms outputting results into 'wave'

        Parameters
        ----------
        frame
           single frame (hologram) of the data
        """

        wav = reconstruct_frame(
            frame,
            sb_pos=self.params.sb_position,
            aperture=self.task_data.aperture,
            slice_fft=self.task_data.slice,
            precision=self.params.precision
        )

        self.results.wave[:] = wav

    def get_backends(self):
        # CuPy support deactivated due to https://github.com/LiberTEM/LiberTEM/issues/815
        return ('numpy',)
        # return ('numpy', 'cupy')

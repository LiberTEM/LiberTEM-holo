import pytest
import numpy as np
from libertem_holo.base.io import InputData, Results
from libertem_holo.base.utils import remove_phase_ramp
from libertem_holo.base.unwrap import phase_unwrap
from libertem_holo.base.convenience import reconstruct_stack
import cupy as cp


def test_reconstruct_stack():
    path = "/storage/er-c-data/adhoc/libertem/libertem-test-data/dm/holo/"
    stack_obj = InputData.load_from_dm(path+"reconstr/stack_obj_minus.dm4")
    stack_ref = InputData.load_from_dm(path+"reconstr/stack_ref_minus.dm4")
    wave, bf_avg, holoparams, px_size, drifts = reconstruct_stack(
        stack=stack_obj, stack_ref=stack_ref, xp=cp
    )
    phase = phase_unwrap(np.angle(wave.get()))
    roi=np.s_[50:150, 250:400]
    phase, _, _  = remove_phase_ramp(phase, roi=roi)
    phase -= np.mean(phase[roi])
    #arrays= {'drift': drifts.get()}
    #np.savez("/home/mkhelfallah/minus_drifts.npz", **arrays, allow_pickle=False)
    #res = Results(complex_wave=wave, unwrapped_phase=phase, brightfield=bf_avg)
    #res.save("/home/mkhelfallah/minus.npz")
    expected = Results.load(path + "align/minus.npz")
    expected_drifts = np.load(path + "align/minus_drifts.npz")['drift']
    np.testing.assert_allclose(phase, expected.unwrapped_phase)
    np.testing.assert_allclose(drifts.get(), expected_drifts, atol=0.1)

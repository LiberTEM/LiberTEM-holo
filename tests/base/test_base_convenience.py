import pytest
import numpy as np
from libertem_holo.base.io import InputData, Results
from libertem_holo.base.utils import remove_phase_ramp
from libertem_holo.base.unwrap import phase_unwrap
from libertem_holo.base.convenience import reconstruct_stack

def test_function():
    path = "/storage/er-c-data/adhoc/libertem/libertem-test-data/dm/holo/"
    stack_obj = InputData.load_from_dm(path+"reconstr/stack_obj_minus.dm4")
    stack_ref = InputData.load_from_dm(path+"reconstr/stack_ref_minus.dm4")
    wave, bf_avg, holoparams, px_size, drifts = reconstruct_stack(
        stack=stack_obj, stack_ref=stack_ref
    )
    phase = phase_unwrap(np.angle(wave))
    phase, _, _  = remove_phase_ramp(phase, roi=np.s_[50:150, 250:400])
    #res = Results(complex_wave=wave, unwrapped_phase=phase, brightfield=bf_avg)
    data = Results.load(path + "align/minus.npz")
    assert np.allclose(phase, data.unwrapped_phase)
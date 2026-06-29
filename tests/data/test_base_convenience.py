import pathlib
import numpy as np
from libertem_holo.base.io import InputData, Results
from libertem_holo.base.utils import remove_phase_ramp
from libertem_holo.base.unwrap import phase_unwrap
from libertem_holo.base.convenience import reconstruct_stack
import cupy as cp


def test_reconstruct_stack(dm_testdata_path: pathlib.Path):    
    path_obj = dm_testdata_path /  "holo/reconstr/stack_obj_minus.dm4"
    path_ref = dm_testdata_path /  "holo/reconstr/stack_ref_minus.dm4"
    stack_obj = InputData.load_from_dm(path_obj)
    stack_ref = InputData.load_from_dm(path_ref)
    res = reconstruct_stack(
        stack=stack_obj, stack_ref=stack_ref, xp=cp
    )
    wave = res.complex_wave
    drifts = np.vstack([res.metadata['drifts_x'], res.metadata['drifts_y']]).T
    phase = phase_unwrap(np.angle(wave))
    roi=np.s_[50:150, 250:400]
    phase, _, _  = remove_phase_ramp(phase, roi=roi)
    phase -= np.mean(phase[roi])
    #arrays= {'drift': drifts.get()}
    #np.savez("/home/mkhelfallah/minus_drifts.npz", **arrays, allow_pickle=False)
    #res = Results(complex_wave=wave, unwrapped_phase=phase, brightfield=bf_avg)
    #res.save("/home/mkhelfallah/minus.npz")
    expected = Results.load(dm_testdata_path / "holo/align/minus.npz")
    expected_drifts = np.load(dm_testdata_path / "holo/align/minus_drifts.npz")['drift']
    np.testing.assert_allclose(phase, expected.unwrapped_phase)
    np.testing.assert_allclose(drifts, expected_drifts, atol=0.1)

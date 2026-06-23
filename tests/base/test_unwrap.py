import pytest
import numpy as np
from libertem_holo.base.unwrap import derivative_variance, quality_unwrap


@pytest.mark.with_numba
def test_quality_unwrap_smoketest():
    phase = np.random.random(128*128).reshape(128, 128)
    dv = derivative_variance(phase)
    _ = quality_unwrap(phase, dv)

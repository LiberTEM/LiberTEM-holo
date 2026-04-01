import os
import glob

import pytest


@pytest.fixture
def dm_default(dm_testdata_path):
    pass


def test_load_smoke(dm_testdata_path):
    dm_testdata_path / '3D'

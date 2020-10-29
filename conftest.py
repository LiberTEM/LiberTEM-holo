"""
Global test configuration. Use this file to define fixtures to use
in both doctests and regular tests.
"""
import pytest

from libertem.api import Context
from libertem.executor.inline import InlineJobExecutor


@pytest.fixture
def lt_ctx():
    return Context(executor=InlineJobExecutor())

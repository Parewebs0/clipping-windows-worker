import os

import pytest


@pytest.fixture(autouse=True)
def _no_real_buffer(monkeypatch):
    """A machine with BUFFER_API_KEY set must never reach Buffer from tests."""
    for name in list(os.environ):
        if name.startswith("BUFFER_"):
            monkeypatch.delenv(name)

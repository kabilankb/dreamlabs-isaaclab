"""Tests must never touch the real Dream credentials (~/.dreamlabs): a test once overwrote the real API key.

Every test gets ``launcher.CREDENTIALS`` pointed at a temporary file, and the session fails if the real file or its
backup changed.
"""

import hashlib
from pathlib import Path

import pytest

from dreamlabs_isaaclab import launcher

REAL = [Path.home() / ".dreamlabs/credentials", Path.home() / ".dreamlabs/credentials.bak"]


def _fingerprint():
    return [hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL]


@pytest.fixture(autouse=True)
def _isolated_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CREDENTIALS", tmp_path / "isolated-home/.dreamlabs/credentials")


@pytest.fixture(scope="session", autouse=True)
def _real_credentials_untouched():
    before = _fingerprint()
    yield
    assert _fingerprint() == before, "a test changed the real ~/.dreamlabs credentials"

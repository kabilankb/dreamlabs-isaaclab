"""Tests must never touch the real Dream credentials or usage records (~/.dreamlabs, ~/.dreamlabs-isaaclab).

A test once overwrote the real API key. So every test gets ``launcher.CREDENTIALS`` and every usage path pointed at a temporary
location, and the session fails if any real file changed.
"""

import hashlib
from pathlib import Path

import pytest

from dreamlabs_isaaclab import launcher, usage

HOME = Path.home()
REAL = [HOME / ".dreamlabs/credentials", HOME / ".dreamlabs/credentials.bak", HOME / ".dreamlabs-isaaclab/usage.jsonl",
        HOME / ".dreamlabs-isaaclab/pricing.json", HOME / ".dreamlabs-isaaclab/logo.png"]
REAL_DIRS = [HOME / ".dreamlabs-isaaclab/dream-usage"]


def _fingerprint():
    files = list(REAL) + [p for d in REAL_DIRS if d.is_dir() for p in sorted(d.iterdir())]
    return [(str(p), hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None) for p in files]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CREDENTIALS", tmp_path / "isolated-home/.dreamlabs/credentials")
    monkeypatch.setattr(usage, "LEDGER", tmp_path / "isolated-home/usage.jsonl")
    monkeypatch.setattr(usage, "PRICING", tmp_path / "isolated-home/pricing.json")
    monkeypatch.setattr(usage, "RUNS", tmp_path / "isolated-home/runs")
    monkeypatch.setattr(usage, "EXPORTS", tmp_path / "isolated-home/dream-usage")
    for name in ("DREAMLABS_RATE_PER_REQUEST", "DREAMLABS_RATE_PER_SESSION", "DREAMLABS_CURRENCY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _real_files_untouched():
    before = _fingerprint()
    yield
    assert _fingerprint() == before, "a test changed a real file under ~/.dreamlabs or ~/.dreamlabs-isaaclab"

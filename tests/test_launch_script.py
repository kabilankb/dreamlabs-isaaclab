"""launch.sh: its refusals and its --check mode, in a throwaway HOME so nothing real is read or changed."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "launch.sh"


def run(tmp_path, *args, **env):
    full = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "DREAMLABS_ISAAC_PYTHON": str(tmp_path / "no-isaac/python"),
            "DREAMLABS_DREAM_ENV": str(tmp_path / "no-dream-env"), "DREAMLABS_YAMLAB": str(tmp_path / "no-yamlab"),
            "DREAMLABS_ASSETS_DIR": str(tmp_path / "no-assets"), **env}
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True, env=full, timeout=60, check=False)


def snapshot(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_the_script_is_valid_bash_and_executable():
    assert os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


def test_help_describes_every_option(tmp_path):
    result = run(tmp_path, "--help")
    assert result.returncode == 0
    for option in ("--check", "--install-only", "--no-browser", "--port"):
        assert option in result.stdout


@pytest.mark.parametrize("args", [("--bogus",), ("--port",), ("--port", "abc")])
def test_bad_options_are_refused(tmp_path, args):
    assert run(tmp_path, *args).returncode == 2


def test_check_reports_what_is_missing_and_changes_nothing(tmp_path):
    before = snapshot(tmp_path)
    result = run(tmp_path, "--check")
    assert result.returncode == 1
    assert "no Isaac Sim Python" in result.stdout
    assert "would: create it and install dreamlabs" in result.stdout
    assert snapshot(tmp_path) == before          # nothing was created: no env, no assets, no credentials


def test_without_isaac_it_installs_nothing_and_does_not_launch(tmp_path):
    before = snapshot(tmp_path)
    result = run(tmp_path, "--install-only")
    assert result.returncode == 1
    assert "nothing was installed" in result.stderr
    assert "Starting the launcher" not in result.stdout
    assert snapshot(tmp_path) == before          # in particular, no Dream environment was created or downloaded


def test_the_api_key_is_never_printed(tmp_path):
    credentials = tmp_path / ".dreamlabs/credentials"
    credentials.parent.mkdir()
    credentials.write_text("DREAM_API_KEY=dl_live_SECRETSECRETSECRET\n")
    result = run(tmp_path, "--check")
    assert "SECRETSECRET" not in result.stdout + result.stderr
    assert "saved in" in result.stdout


def test_a_missing_course_repo_is_a_note_not_a_failure(tmp_path):
    result = run(tmp_path, "--check", DREAMLABS_COURSE_REPO=str(tmp_path / "nope"))
    assert "6. reBot apple task" in result.stdout
    assert result.returncode == 1                       # because of Isaac / Dream here, not because of step 6
    assert "rebot-duo-course" in result.stdout

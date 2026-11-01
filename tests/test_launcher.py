import http.client
import json
import os
import stat
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from dreamlabs_isaaclab import launch_spec, launcher

TOKEN = "t" * 32
JSON = {"Content-Type": "application/json", "X-Token": TOKEN}


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CREDENTIALS", tmp_path / "home/.dreamlabs/credentials")
    paths = launch_spec.Paths(isaac_python=tmp_path / "py", dream_python=tmp_path / "dream",
                              rebot_sim_pipeline=tmp_path / "rsp", yamlab=tmp_path / "y", runs=tmp_path / "runs")
    app = launcher.Launcher(paths)
    probe = ThreadingHTTPServer(("127.0.0.1", 0), launcher.BaseHTTPRequestHandler)
    port = probe.server_address[1]
    probe.server_close()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), launcher.make_handler(app, TOKEN, port))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app, port
    httpd.shutdown()
    httpd.server_close()


def call(port, method, path, body=None, headers=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    base = {"Host": host or f"127.0.0.1:{port}"}
    base.update(headers or {})
    conn.request(method, path, body=None if body is None else json.dumps(body), headers=base)
    response = conn.getresponse()
    return response.status, response.read()


def test_the_page_needs_the_token(server):
    _, port = server
    assert call(port, "GET", "/")[0] == 403
    assert call(port, "GET", "/?token=wrong")[0] == 403
    status, body = call(port, "GET", f"/?token={TOKEN}")
    assert status == 200 and b"<title>dreamlabs-isaaclab" in body


def test_the_api_needs_the_token_and_the_right_host(server):
    _, port = server
    assert call(port, "GET", "/api/state")[0] == 403
    assert call(port, "GET", "/api/state", headers={"X-Token": "x" * 32})[0] == 403
    assert call(port, "GET", "/api/state", headers={"X-Token": TOKEN})[0] == 200
    # DNS rebinding: a page on another name that resolves to 127.0.0.1 sends its own Host header
    assert call(port, "GET", "/api/state", headers={"X-Token": TOKEN}, host="evil.example")[0] == 403
    assert call(port, "GET", f"/?token={TOKEN}", host="evil.example")[0] == 403


def test_posts_must_be_json_from_our_own_origin(server):
    _, port = server
    assert call(port, "POST", "/api/stop", {}, {"X-Token": TOKEN, "Content-Type": "text/plain"})[0] == 415
    assert call(port, "POST", "/api/stop", {}, {**JSON, "Origin": "https://evil.example"})[0] == 403
    assert call(port, "POST", "/api/stop", {}, {**JSON, "Origin": f"http://127.0.0.1:{port}"})[0] == 200
    assert call(port, "POST", "/api/stop", {}, {"Content-Type": "application/json"})[0] == 403


def test_state_lists_the_robots_and_never_a_key(server):
    _, port = server
    launcher.save_key("dl_live_" + "A" * 30 + "WXYZ")
    status, body = call(port, "GET", "/api/state", headers={"X-Token": TOKEN})
    state = json.loads(body)
    assert status == 200 and {r["id"] for r in state["robots"]} == set(launch_spec.ROBOTS)
    assert state["key"]["present"] and state["key"]["masked"].endswith("WXYZ")
    assert "A" * 30 not in body.decode()


def test_launch_is_refused_without_a_ready_robot_and_a_key(server):
    _, port = server
    status, body = call(port, "POST", "/api/launch", {"robot": "rebot"}, JSON)
    assert status == 400 and b"not ready" in body
    assert call(port, "POST", "/api/launch", {"robot": "nope"}, JSON)[0] == 400


def test_a_bad_key_is_refused_before_any_network_call(server):
    _, port = server
    for bad in ("short", "has space in it 123456789", "dl_live_;echo injected", 5, None):
        assert call(port, "POST", "/api/key", {"key": bad}, JSON)[0] == 400


def test_save_key_writes_mode_600_keeps_a_backup_and_other_settings(tmp_path):
    path = tmp_path / "d/.dreamlabs/credentials"
    launcher.save_key("dl_live_first_key_value_1234", path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    launcher.save_key("dl_live_second_key_value_5678", path)
    values = launcher.read_credentials(path)
    assert values["DREAM_API_KEY"] == "dl_live_second_key_value_5678"
    assert values["DREAM_GATEWAY_URL"].startswith("https://")
    backup = path.with_name("credentials.bak")
    assert "first_key" in backup.read_text() and stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert not path.with_name("credentials.tmp").exists()


def test_mask_shows_only_the_ends():
    assert launcher.mask("") == "" and launcher.mask("dl_live_" + "x" * 30 + "ABCD") == "dl_live_…ABCD"
    assert launcher.mask("short") == "…hort"


def test_a_run_gets_no_key_in_its_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CREDENTIALS", tmp_path / "c/credentials")
    launcher.save_key("dl_live_" + "B" * 30)
    monkeypatch.setenv("DREAM_API_KEY", "should-not-leak")
    fake = tmp_path / "py"
    fake.write_text('#!/bin/sh\nenv > "$(dirname "$0")/seen_env.txt"\n')
    os.chmod(fake, 0o755)
    (tmp_path / "dream").touch()
    (tmp_path / "rsp").mkdir()
    paths = launch_spec.Paths(isaac_python=fake, dream_python=tmp_path / "dream", rebot_sim_pipeline=tmp_path / "rsp",
                              yamlab=tmp_path / "y", runs=tmp_path / "runs")
    app = launcher.Launcher(paths)
    state = app.start({"robot": "rebot", "gui": False})
    assert state["state"] in {"running", "finished"}
    app.run["proc"].wait(timeout=10)
    seen = (tmp_path / "seen_env.txt").read_text()
    assert "DREAM_API_KEY" not in seen and "should-not-leak" not in seen and "BBBB" not in seen
    assert (Path(state["dir"]) / "command.txt").exists()


def test_the_logo_is_served_only_with_the_token_and_only_if_it_is_a_small_png(server, tmp_path, monkeypatch):
    _, port = server
    logo = tmp_path / "logo.png"
    monkeypatch.setattr(launcher, "LOGO", logo)
    assert call(port, "GET", f"/logo.png?token={TOKEN}")[0] == 404           # none supplied
    logo.write_bytes(launcher.PNG_MAGIC + b"\x00" * 50)
    assert call(port, "GET", "/logo.png")[0] == 403 and call(port, "GET", "/logo.png?token=wrong")[0] == 403
    status, body = call(port, "GET", f"/logo.png?token={TOKEN}")
    assert status == 200 and body.startswith(launcher.PNG_MAGIC)
    assert json.loads(call(port, "GET", "/api/state", headers={"X-Token": TOKEN})[1])["logo"] is True
    logo.write_bytes(b"<html>not a png</html>")                              # a file that is not a PNG is never served
    assert call(port, "GET", f"/logo.png?token={TOKEN}")[0] == 404
    logo.write_bytes(launcher.PNG_MAGIC + b"\x00" * launcher.MAX_LOGO_BYTES)   # nor an oversized one
    assert call(port, "GET", f"/logo.png?token={TOKEN}")[0] == 404
    assert call(port, "GET", f"/logo.png?token={TOKEN}", host="evil.example")[0] == 403


def test_the_api_test_needs_a_key_a_valid_request_and_no_run_in_progress(server, monkeypatch):
    app, port = server
    assert call(port, "POST", "/api/test", {"call_model": False}, JSON)[0] == 400            # no key saved
    launcher.save_key("dl_live_" + "C" * 30)
    for bad in ({"call_model": "yes"}, {"call_model": True, "model": "$(echo injected)"}, {"model": 5}):
        assert call(port, "POST", "/api/test", bad, JSON)[0] == 400
    assert call(port, "POST", "/api/test", {}, {"X-Token": TOKEN})[0] == 415                 # JSON only
    app.run = {"proc": type("P", (), {"poll": lambda self: None})(), "dir": Path("."), "launch": {}, "started": 0, "command": ""}
    status, body = call(port, "POST", "/api/test", {"call_model": True}, JSON)
    assert status == 400 and b"A run is going" in body                                       # one client per key at a time
    app.run = None


def test_the_api_test_reports_success_and_failure_without_leaking_anything(server, monkeypatch):
    _, port = server
    launcher.save_key("dl_live_" + "D" * 30)
    seen = []

    def fake(model, call_model, dream_python):
        seen.append((model, call_model))
        return {"models": ["a", "b"], "rows": 48, "width": 7, "ms": 612.0}

    monkeypatch.setattr(launcher, "selftest", fake)
    assert call(port, "POST", "/api/test", {"call_model": True, "model": "dreamlabs/dream-1-beta"}, JSON)[0] == 200
    for _ in range(100):
        state = json.loads(call(port, "GET", "/api/state", headers={"X-Token": TOKEN})[1])["api_test"]
        if state["state"] != "running":
            break
        time.sleep(0.05)
    assert state["state"] == "ok" and "48 rows" in state["line"] and "2 models" in state["line"]
    assert seen == [("dreamlabs/dream-1-beta", True)]

    def broken(model, call_model, dream_python):
        raise RuntimeError("Dream bridge selftest: DreamAuthError: This api key is not valid.")

    monkeypatch.setattr(launcher, "selftest", broken)
    assert call(port, "POST", "/api/test", {}, JSON)[0] == 200
    for _ in range(100):
        state = json.loads(call(port, "GET", "/api/state", headers={"X-Token": TOKEN})[1])["api_test"]
        if state["state"] != "running":
            break
        time.sleep(0.05)
    assert state["state"] == "error" and "not valid" in state["line"] and "D" * 30 not in json.dumps(state)

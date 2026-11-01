"""A local launcher page: choose a robot, choose the Dream API model, and start a run from the browser.

    python -m dreamlabs_isaaclab.launcher            # opens the page; Ctrl-C stops it (and any run it started)

Standard library only, so any Python runs it (no simulator and no Dream library in this process). It listens on
127.0.0.1 only. The page's address carries a random token that every request must present, every state-changing request
must be JSON, and the Host and Origin headers are checked, so another web page cannot make your browser start a
simulator or change your key. Anyone who has the address can start the simulators on this machine: keep it private.

The Dream API key: the page shows only its last four characters. Saving a key checks it with Dream first, then writes
``~/.dreamlabs/credentials`` (mode 600) as Dream's own setup does. The key is never put in a command, an environment
variable or a log; the Dream library reads the file itself.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dreamlabs_isaaclab import launch_spec
from dreamlabs_isaaclab.client import Bridge, selftest

PAGE = Path(__file__).with_name("launcher.html")
#: An optional logo you supply (Dream Labs' own mark, shown by the API section). Not shipped with this package: it stays
#: in your home folder. Served only if it is a small PNG.
LOGO = Path.home() / ".dreamlabs-isaaclab/logo.png"
MAX_LOGO_BYTES = 256 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CREDENTIALS = Path.home() / ".dreamlabs/credentials"
KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{16,200}$")
MAX_BODY = 16 * 1024
MAX_LOG_CHUNK = 64 * 1024


# ---- the Dream API key ----------------------------------------------------------------------------------------------


def read_credentials(path: Path | None = None) -> dict[str, str]:
    path = CREDENTIALS if path is None else path  # looked up per call, so tests can point it elsewhere
    values: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip():
                values[name.strip()] = value.strip()
    except OSError:
        pass
    return values


def mask(key: str) -> str:
    return "" if not key else f"{key[:8]}…{key[-4:]}" if len(key) > 16 else "…" + key[-4:]


def check_key(key: str, api_url: str = "https://platform.dreamlabs.ai") -> tuple[bool, str]:
    """Ask Dream whether a key is accepted. Returns (accepted, one line for the page); the key is never echoed."""
    request = urllib.request.Request(f"{api_url.rstrip('/')}/api/downloads/authorize",
                                     headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=15) as answer:
            return (200 <= answer.status < 300), "Dream accepted the key"
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            return False, f"Dream refused the key (HTTP {error.code})"
        return False, f"Dream answered HTTP {error.code}"
    except (urllib.error.URLError, TimeoutError, OSError):
        return False, "Could not reach platform.dreamlabs.ai"


def save_key(key: str, path: Path | None = None) -> None:
    """Write the credentials file as Dream's setup does, mode 600 from the first byte, keeping a backup of the old one."""
    path = CREDENTIALS if path is None else path  # looked up per call, so tests can point it elsewhere
    old = umask = None
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if path.exists():
        old = path.read_bytes()
    lines = {"DREAM_GATEWAY_URL": "https://api.dreamlabs.ai", "DREAM_API_URL": "https://platform.dreamlabs.ai",
             "DREAM_DATA_URL": "https://data.dreamlabs.ai"}
    lines.update({k: v for k, v in read_credentials(path).items() if k.startswith("DREAM_") and k != "DREAM_API_KEY"})
    lines["DREAM_API_KEY"] = key
    text = "".join(f"{k}={v}\n" for k, v in lines.items())
    umask = os.umask(0o077)
    try:
        if old is not None:
            backup = path.with_name(path.name + ".bak")
            fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(old)
        temp = path.with_name(path.name + ".tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(temp, path)
    finally:
        os.umask(umask)
    os.chmod(path, 0o600)


# ---- the launcher's state -------------------------------------------------------------------------------------------


class Launcher:
    """The paths, the one running simulation, and the model list. Thread safe."""

    def __init__(self, paths: launch_spec.Paths) -> None:
        self.paths = paths
        self.lock = threading.Lock()
        self.run: dict | None = None
        self.models: dict = {"state": "idle", "items": [{"id": m, "owner": "public"} for m in launch_spec.DEFAULT_MODELS],
                             "error": None, "at": None}
        self.key_check: dict = {"state": "unchecked", "line": "Not checked yet"}
        self.api_test: dict = {"state": "idle", "line": ""}

    # -- the key
    def key_status(self) -> dict:
        key = read_credentials().get("DREAM_API_KEY", "")
        return {"present": bool(key), "masked": mask(key), "file": str(CREDENTIALS), **self.key_check}

    def check_saved_key(self) -> dict:
        key = read_credentials().get("DREAM_API_KEY", "")
        if not key:
            self.key_check = {"state": "missing", "line": "No key saved yet"}
        else:
            ok, line = check_key(key, read_credentials().get("DREAM_API_URL", "https://platform.dreamlabs.ai"))
            self.key_check = {"state": "valid" if ok else "invalid", "line": line}
        return self.key_status()

    def set_key(self, key: object) -> dict:
        if not isinstance(key, str) or not KEY_RE.match(key.strip()):
            raise launch_spec.SpecError("that does not look like an API key")
        key = key.strip()
        ok, line = check_key(key)
        if not ok:
            self.key_check = {"state": "invalid", "line": line}
            raise launch_spec.SpecError(line + "; nothing was saved")
        save_key(key)
        self.key_check = {"state": "valid", "line": "Saved, and Dream accepted it"}
        self.refresh_models()
        return self.key_status()

    # -- the connection test
    def start_api_test(self, request: object) -> dict:
        """Test the Dream connection in the background: models always, one real request when ``call_model``."""
        request = request if isinstance(request, dict) else {}
        call_model = request.get("call_model", False)
        model = request.get("model") or launch_spec.DEFAULT_MODELS[0]
        if not isinstance(call_model, bool) or not isinstance(model, str) or not launch_spec.MODEL_RE.match(model):
            raise launch_spec.SpecError("call_model is true or false and model is a model id")
        if not read_credentials().get("DREAM_API_KEY"):
            raise launch_spec.SpecError("Save your Dream API key first")
        with self.lock:
            if self.run is not None and self.run["proc"].poll() is None:
                raise launch_spec.SpecError("A run is going. Two clients on one key replace each other's sessions: "
                                            "stop it first.")
            if self.api_test["state"] == "running":
                raise launch_spec.SpecError("A test is already running")
            self.api_test = {"state": "running", "line": "Sending a test request…" if call_model else "Asking Dream…"}
        threading.Thread(target=self._api_test, args=(model, call_model), daemon=True).start()
        return self.api_test

    def _api_test(self, model: str, call_model: bool) -> None:
        try:
            answer = selftest(model, call_model, self.paths.dream_python)
            line = f"Dream answered: {len(answer['models'])} models on the account"
            if call_model:
                line += f"; {model} returned {answer['rows']} rows × {answer['width']} values in {answer['ms']:.0f} ms"
            state = {"state": "ok", "line": line}
        except Exception as error:  # noqa: BLE001 - shown on the page
            state = {"state": "error", "line": f"{type(error).__name__}: {error}"[:300]}
        with self.lock:
            self.api_test = {**state, "at": time.time()}

    # -- the models
    def refresh_models(self) -> None:
        with self.lock:
            if self.models["state"] == "loading":
                return
            self.models = {**self.models, "state": "loading", "error": None}
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self) -> None:
        bridge = None
        try:
            bridge = Bridge(self.paths.dream_python)
            answer = bridge.call(op="models")
            items = [{"id": m["id"], "owner": m.get("owner", ""), "embodiment": m.get("embodiment", "")}
                     for m in answer["models"] if m.get("servable", True)]
            state = {"state": "ready", "items": items or self.models["items"], "error": None, "at": time.time()}
        except Exception as error:  # noqa: BLE001 - shown on the page; the defaults stay selectable
            state = {**self.models, "state": "error", "error": f"{type(error).__name__}: {error}"[:300]}
        finally:
            if bridge is not None:
                bridge.close()
        with self.lock:
            self.models = state

    # -- the robots
    def robots(self) -> list[dict]:
        out = []
        for robot in launch_spec.ROBOTS.values():
            gaps = launch_spec.missing(robot, self.paths)
            out.append({"id": robot.id, "name": robot.name, "tagline": robot.tagline, "simulator": robot.simulator,
                        "default_instruction": robot.default_instruction, "presets": list(robot.presets),
                        "has_seed": robot.has_seed, "notes": list(robot.notes), "ready": not gaps, "missing": gaps})
        return out

    # -- a run
    def status(self) -> dict:
        with self.lock:
            run = self.run
            if run is None:
                return {"state": "idle"}
            code = run["proc"].poll()
            state = "running" if code is None else ("finished" if code == 0 else "failed")
            results = None
            result_file = run["dir"] / "results.json"
            if result_file.exists():
                try:
                    results = json.loads(result_file.read_text())
                except (OSError, ValueError):
                    results = None
            return {"state": state, "exit_code": code, "launch": run["launch"], "dir": str(run["dir"]),
                    "started": run["started"], "command": run["command"], "results": results}

    def preview(self, request: object) -> dict:
        launch = launch_spec.validate(request)
        command = launch_spec.build(launch, self.paths, self.paths.runs / "<run>")
        return {"command": shlex.join(command.argv), "cwd": str(command.cwd), "launch": launch.public()}

    def start(self, request: object) -> dict:
        launch = launch_spec.validate(request)
        gaps = launch_spec.missing(launch.robot, self.paths)
        if gaps:
            raise launch_spec.SpecError(f"{launch.robot.name} is not ready: missing " + "; ".join(gaps))
        if not read_credentials().get("DREAM_API_KEY"):
            raise launch_spec.SpecError("Save your Dream API key first")
        with self.lock:
            if self.run is not None and self.run["proc"].poll() is None:
                raise launch_spec.SpecError("A run is already going. Two simulators on one key replace each other's "
                                            "sessions: stop it first.")
            run_dir = self.paths.runs / f"{time.strftime('%Y%m%d-%H%M%S')}-{launch.robot.id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = launch_spec.build(launch, self.paths, run_dir)
            (run_dir / "command.txt").write_text(shlex.join(command.argv) + "\n")
            env = {k: v for k, v in os.environ.items() if k != "DREAM_API_KEY"}
            env.update(command.env)
            log = open(run_dir / "run.log", "wb")  # noqa: SIM115 - owned by the child until it exits
            try:
                proc = subprocess.Popen(command.argv, cwd=command.cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                log.close()
            self.run = {"proc": proc, "dir": run_dir, "launch": launch.public(), "started": time.time(),
                        "command": shlex.join(command.argv)}
        return self.status()

    def stop(self) -> dict:
        with self.lock:
            run = self.run
        if run is None or run["proc"].poll() is not None:
            return self.status()
        proc = run["proc"]
        for sig, wait in ((signal.SIGINT, 8), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
            try:
                os.killpg(proc.pid, sig)  # the simulator and the bridge it started are in this group
            except ProcessLookupError:
                break
            deadline = time.time() + wait
            while time.time() < deadline and proc.poll() is None:
                time.sleep(0.2)
            if proc.poll() is not None:
                break
        return self.status()

    def log(self, offset: int) -> dict:
        with self.lock:
            run = self.run
        if run is None:
            return {"offset": 0, "text": ""}
        path = run["dir"] / "run.log"
        try:
            size = path.stat().st_size
            offset = max(0, min(offset, size))
            if size - offset > MAX_LOG_CHUNK:
                offset = size - MAX_LOG_CHUNK
            with open(path, "rb") as handle:
                handle.seek(offset)
                data = handle.read(MAX_LOG_CHUNK)
        except OSError:
            return {"offset": offset, "text": ""}
        text = data.decode("utf-8", "replace")
        lines = [ln for ln in text.splitlines() if launch_spec.parse_progress(ln)]
        return {"offset": offset + len(data), "text": "\n".join(lines)}


def read_logo(path: Path | None = None) -> bytes | None:
    """The user's logo as bytes, or None when there is none or it is not a small PNG."""
    path = LOGO if path is None else path
    try:
        if not path.is_file() or path.stat().st_size > MAX_LOGO_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    return data if data.startswith(PNG_MAGIC) else None


# ---- the HTTP layer -------------------------------------------------------------------------------------------------


def make_handler(launcher: Launcher, token: str, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "dreamlabs-launcher"

        def log_message(self, *_args) -> None:  # the default log would print request lines; keep the terminal clean
            return

        # -- checks
        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            return origin is None or origin in {f"http://{h}" for h in allowed_hosts}

        def _token_ok(self, supplied: str) -> bool:
            return hmac.compare_digest(supplied.encode(), token.encode())

        def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                             "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode())

        def _fail(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        # -- routes
        def do_GET(self) -> None:
            if not self._host_ok():
                return self._fail(403, "unexpected Host header")
            path, _, query = self.path.partition("?")
            if path in ("/", "/logo.png"):
                supplied = dict(p.partition("=")[::2] for p in query.split("&") if p).get("token", "")
                if not self._token_ok(supplied):
                    return self._fail(403, "open the address the launcher printed (it carries the token)")
                if path == "/logo.png":  # an <img> cannot send a header, so this one takes the token in the address
                    logo = read_logo()
                    return self._send(200, logo, "image/png") if logo else self._fail(404, "no logo")
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if not self._token_ok(self.headers.get("X-Token", "")):
                return self._fail(403, "missing or wrong token")
            if path == "/api/state":
                return self._json(200, {"robots": launcher.robots(), "key": launcher.key_status(),
                                        "models": launcher.models, "run": launcher.status(),
                                        "logo": read_logo() is not None, "api_test": launcher.api_test})
            if path == "/api/log":
                offset = dict(p.partition("=")[::2] for p in query.split("&") if p).get("offset", "0")
                return self._json(200, launcher.log(int(offset) if offset.isdigit() else 0))
            self._fail(404, "not found")

        def do_POST(self) -> None:
            if not (self._host_ok() and self._origin_ok()):
                return self._fail(403, "unexpected Host or Origin header")
            if not self._token_ok(self.headers.get("X-Token", "")):
                return self._fail(403, "missing or wrong token")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._fail(415, "send JSON")
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 <= length <= MAX_BODY:
                return self._fail(413, "request too large")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._fail(400, "not JSON")
            try:
                if self.path == "/api/key":
                    return self._json(200, launcher.set_key(body.get("key")))
                if self.path == "/api/key/check":
                    return self._json(200, launcher.check_saved_key())
                if self.path == "/api/test":
                    return self._json(200, launcher.start_api_test(body))
                if self.path == "/api/models/refresh":
                    launcher.refresh_models()
                    return self._json(200, launcher.models)
                if self.path == "/api/preview":
                    return self._json(200, launcher.preview(body))
                if self.path == "/api/launch":
                    return self._json(200, launcher.start(body))
                if self.path == "/api/stop":
                    return self._json(200, launcher.stop())
            except launch_spec.SpecError as error:
                return self._fail(400, str(error))
            self._fail(404, "not found")

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8765, help="Port on 127.0.0.1 (the next free one if taken)")
    parser.add_argument("--no-browser", action="store_true", help="Print the address only")
    args = parser.parse_args(argv)

    launcher = Launcher(launch_spec.Paths.from_env())
    token = secrets.token_urlsafe(24)
    server = None
    for port in range(args.port, args.port + 20):
        try:
            handler = make_handler(launcher, token, port)
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    if server is None:
        print(f"no free port from {args.port} to {args.port + 19}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}/?token={token}"
    print(f"Launcher: {url}\nKeep this address private: it can start the simulators on this machine. Ctrl-C to stop.")
    launcher.refresh_models()
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        launcher.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

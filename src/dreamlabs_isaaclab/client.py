"""The simulator-side half of the Dream bridge: starts ``bridge.py`` in the Dream env and talks to it.

Isaac Sim and the Dream library pin different numpy / jax versions, so they cannot share a process. Isaac's Python
imports this module, which starts ``bridge.py`` with the Dream env's Python and exchanges one JSON object per line.
Nothing here imports Isaac or the Dream library, so it is importable (and testable) anywhere.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
from pathlib import Path

DEFAULT_DREAM_PYTHON = Path.home() / "dream-env/bin/python"
BRIDGE_SCRIPT = Path(__file__).with_name("bridge.py")
CONFIGS = Path(__file__).with_name("configs")


class BridgeError(RuntimeError):
    """The bridge answered ``ok: false`` or exited."""


class Bridge:
    """The bridge subprocess. ``call(op=..., ...)`` sends a request and returns the answer dict."""

    def __init__(self, dream_python: str | Path = DEFAULT_DREAM_PYTHON, script: str | Path = BRIDGE_SCRIPT) -> None:
        # -I: the bridge needs nothing from the current directory, and an Isaac cwd must not shadow its imports.
        self.process = subprocess.Popen([str(dream_python), "-I", str(script)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)

    def call(self, **request) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise BridgeError(f"the Dream bridge exited (code {self.process.poll()}); see its log above")
        answer = json.loads(line)
        if not answer.get("ok"):
            raise BridgeError(f"Dream bridge {request['op']}: {answer.get('error')}")
        return answer

    def close(self) -> None:
        try:
            self.call(op="close")
        except Exception:  # noqa: BLE001, S110 - closing a bridge that already died is not an error
            pass
        self.process.terminate()


def selftest(checkpoint: str, call_model: bool, dream_python: str | Path = DEFAULT_DREAM_PYTHON) -> dict:
    """Ask the bridge for the connection test: the account's models, and (with ``call_model``) one real request.

    The request sends grey camera frames and a neutral pose to Dream and **uses account credits**. Returns the bridge's
    answer (``models``, and ``rows`` / ``width`` / ``ms`` for the request); raises BridgeError when Dream refuses.
    """
    bridge = Bridge(dream_python)
    try:
        return bridge.call(op="selftest", checkpoint=checkpoint, call_model=call_model)
    finally:
        bridge.close()


def jpeg_b64(image, size: tuple[int, int], quality: int = 90) -> str:
    """One camera frame (uint8 HxWx3 array) as the resized base64 JPEG Dream's ``set_observation`` takes."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image[..., :3]).resize(size, Image.BILINEAR).save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def load_embodiment(name_or_path: str | Path) -> dict:
    """An embodiment config: a file path, or the name of one shipped in ``configs/`` (without ``.json``)."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIGS / f"{name_or_path}.json"
    return json.loads(path.read_text())

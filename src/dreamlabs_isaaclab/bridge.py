"""Dream-1 bridge: lets the Isaac Lab sim (its own Python env) use the Dream library (``~/dream-env``).

The two envs pin different numpy / jax versions, so they cannot share a process. The sim client starts this script
as a subprocess and talks to it with one JSON object per line over stdin / stdout:

    {"op": "open",    "checkpoint": ..., "arm": "seeed_faas", "embodiment": {...}}      -> {"ok": true}
    {"op": "observe", "frames": {role: base64 JPEG}, "state": {...}, "prompt": ..., "t": s}  -> {"ok": true}
    {"op": "act",     "commanded": {"q_l_chunk": [...], "q_r_chunk": [...]} | null}    -> {"ok": true, "chunks": ...}
    {"op": "reset"}                                                                      -> {"ok": true}
    {"op": "models"}                                                -> {"ok": true, "models": [{"id": ...}, ...]}
    {"op": "selftest", "checkpoint": ..., "call_model": bool}     -> {"ok": true, "models": [...], "rows", "width", "ms"}
    {"op": "close"}

Everything the libraries print goes to stderr; stdout carries the protocol only. The API key is read from
``~/.dreamlabs/credentials`` by the library; this script never sees it.

    ~/dream-env/bin/python -I src/dreamlabs_isaaclab/bridge.py   (the sim client starts it)
"""

import json
import os
import sys
import time

protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout = sys.stderr  # loguru / prints from the libraries must not corrupt the protocol

import dream
import numpy as np

sessions_replaced = [0]


def reply(**fields) -> None:
    protocol.write(json.dumps(fields) + "\n")


def with_retry(call, tries: int = 30):
    """A model that is starting answers DA-2000 pool_warming; wait as the answer says and call again."""
    for attempt in range(tries):
        try:
            return call()
        except dream.DreamSessionExpired as error:
            # The session idled out or the model restarted; the client already adopted the new one, so asking
            # again continues the episode there (its memory of the episode starts over).
            print(f"[bridge] session replaced ({error}); continuing on the new one", file=sys.stderr, flush=True)
            sessions_replaced[0] += 1
        except dream.DreamServerError as error:
            if not error.retryable or attempt == tries - 1:
                raise
            wait = error.retry_after_s or 2.0
            print(f"[bridge] {error.code} {error.reason}: waiting {wait:g}s", file=sys.stderr, flush=True)
            time.sleep(wait)


def model_call(client, checkpoint: str) -> dict:
    """One request the way yam.py makes it (the YAM station keys), on grey frames; returns the chunk's shape and time."""
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (320, 176), (110, 110, 110)).save(buffer, format="JPEG", quality=85)
    frame = base64.b64encode(buffer.getvalue()).decode("ascii")
    client.policy({"checkpoint": checkpoint, "timeout_ms": 60000, "strict": False, "action_hz": 20, "_record": False})
    state = {"robot.arm": "yam", "robot.exec_head": "joint", "robot.max_joint_vel_rad_s": 0.0, "robot.gripper_open": 1.0}
    for slot in ("arm_l", "arm_r"):
        state[f"state.{slot}.joint_pos"] = np.zeros(6)
        state[f"state.{slot}.gripper_pos"] = np.array([1.0])
    state["prompt"] = "put apple into the bowl"
    client.set_observation({f"image_window.{camera}": frame for camera in ("top", "left", "right")}, state, timestamp=0.0)
    started = time.time()
    action = with_retry(lambda: client.get_action())
    rows = np.asarray(action["q_l_chunk"])
    client.reset_episode()
    return {"rows": int(rows.shape[0]), "width": int(rows.shape[1]), "ms": (time.time() - started) * 1000.0}


def main() -> None:
    client = dream.DreamClient()
    for line in sys.stdin:
        request = json.loads(line)
        op = request["op"]
        try:
            if op == "open":
                # "config": a raw session config (a YAM, which needs no ik); else the custom-arm form below.
                config = request.get("config") or {
                    "checkpoint": request["checkpoint"],
                    "embodiment": request["embodiment"],
                    "ik": {"arm": request["arm"], "head": "eef"},
                    "timeout_ms": 60000,
                }
                client.policy(config)
                reply(ok=True)
            elif op == "observe":
                observation = {k: np.asarray(v, dtype=np.float64) if isinstance(v, list) else v
                               for k, v in request["state"].items()}
                observation["prompt"] = request["prompt"]
                client.set_observation(request["frames"], observation, timestamp=request["t"])
                reply(ok=True)
            elif op == "act":
                commanded = request.get("commanded")
                if commanded:
                    commanded = {k: np.asarray(v, dtype=np.float64) for k, v in commanded.items()}
                started = time.time()
                action = with_retry(lambda commanded=commanded: client.get_action(commanded=commanded or None))
                chunks = {k: np.asarray(action[k]).tolist() for k in ("q_l_chunk", "q_r_chunk")}
                reply(ok=True, chunks=chunks, ms=(time.time() - started) * 1000.0, replaced=sessions_replaced[0])
            elif op == "models":
                # What the account can run: the public base models and its own fine-tunes (for the launcher page).
                models = [{"id": m.id, "owner": m.owner, "embodiment": m.embodiment, "servable": m.servable}
                          for m in client.get_models()]
                reply(ok=True, models=models)
            elif op == "selftest":
                # The connection test of the launcher page and of api_check: your key and models, and (when asked) one
                # real request with grey camera frames on a neutral YAM pose, which uses account credits.
                result = {"models": [m.id for m in client.get_models()]}
                if request.get("call_model"):
                    result.update(model_call(client, request["checkpoint"]))
                reply(ok=True, **result)
            elif op == "reset":
                client.reset_episode()
                reply(ok=True)
            elif op == "close":
                client.close()
                reply(ok=True)
                return
            else:
                reply(ok=False, error=f"unknown op {op!r}")
        except Exception as error:  # noqa: BLE001 - reported to the sim client, which decides; the bridge keeps running
            reply(ok=False, error=f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()

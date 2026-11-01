"""Drive the bimanual reBot Isaac Lab scene with Dream-1, zero-shot or a fine-tune of it.

Run with the Isaac Lab Python (it starts Isaac Sim), with ``rebot-sim-pipeline`` importable::

    python -m dreamlabs_isaaclab.rebot --episodes 3 --seed 1000            # headless
    python -m dreamlabs_isaaclab.rebot --viz kit --realtime                # watch it

The Dream library lives in its own env (``~/dream-env``; it pins other numpy / jax versions than Isaac Sim), so
this script starts ``bridge.py`` there as a subprocess and talks to it with JSON lines (``client.Bridge``). The API
key is read by the library from ``~/.dreamlabs/credentials``; it never passes through this process.

Units: the Dream arm model of the reBot (``seeed_faas``) reads and writes the arm's URDF joints in radians, which are
the simulator's own joint angles, and the gripper as an open fraction 0 (closed) to 1 (open). The 20 Hz action rows
of Dream-1 are played against the simulator's control rate by time.

Exit code 0 only if every episode succeeded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

from dreamlabs_isaaclab import units, usage
from dreamlabs_isaaclab.client import BRIDGE_SCRIPT, DEFAULT_DREAM_PYTHON, Bridge, jpeg_b64, load_embodiment

parser = argparse.ArgumentParser(description="Bimanual reBot sim rollout with Dream-1.")
parser.add_argument("--checkpoint", default="dreamlabs/dream-1-beta", help="A model id from get_models().")
parser.add_argument("--arm", default="seeed_faas", help="Dream's name for the arm's kinematics.")
parser.add_argument("--embodiment", default="rebot_zero_shot_embodiment",
                    help="Session embodiment: a file, or a name in configs/ (camera roles and the state parts the client adds).")
parser.add_argument("--dream-python", default=str(DEFAULT_DREAM_PYTHON))
parser.add_argument("--bridge", default=str(BRIDGE_SCRIPT))
parser.add_argument("--instruction", default=None, help="Default: the task description from the task YAML.")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=1000, help="Layout seed of the first episode.")
parser.add_argument("--execute", type=int, default=30, help="Action rows (20 Hz) executed per chunk.")
parser.add_argument("--action-hz", type=float, default=20.0, help="Rate of Dream-1's action rows.")
parser.add_argument("--max-steps", type=int, default=None, help="Control steps per episode (default: the task's).")
parser.add_argument("--task", default="stationery_sorting",
                    help="stationery_sorting (pens into a cup) or apple_in_bowl (Dream's MuJoCo scene, needs the "
                         "converted assets and the rebot-sim-pipeline of this repository).")
parser.add_argument("--no-grasp-assist", action="store_true",
                    help="Turn the sim's grasp assist (a gripper that closes near an object carries it) off: "
                         "grasps are then pure physics, as on the YAM.")
parser.add_argument("--robot", default="rebot_b601_bimanual")
parser.add_argument("--rebot-project", default=None, help="Single-arm reBot checkout with the assets.")
parser.add_argument("--video", default=None, help="Write the three camera views side by side to this .mp4.")
parser.add_argument("--results", default=None, help="Write per-episode results to this JSON file.")
parser.add_argument("--realtime", action="store_true")
parser.add_argument("--episode-timeout", type=float, default=None, help="Wall-clock limit per episode, seconds.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device=None)
args = parser.parse_args()
args.enable_cameras = True
if args.device is None:
    from rebot_sim_pipeline.lab.config import load_task as _load_task

    args.device = str(_load_task(args.task, mode="evaluation")["sim"]["device"])
app = AppLauncher(args).app

import numpy as np
import torch
from rebot_sim_pipeline.lab.assets import find_workstation, prepare_robot_asset, prepare_stationery_assets
from rebot_sim_pipeline.lab.config import find_rebot_project, load_robot, load_task
from rebot_sim_pipeline.lab.env import ARMS, StationeryBimanualEnv
from rebot_sim_pipeline.lab.env_cfg import build_env_cfg
from rebot_sim_pipeline.lab.expert_driver import BimanualExpertDriver

#: Dream camera role -> camera in the simulator.
ROLES = {"overhead": "top", "wrist_l": "left_wrist", "wrist_r": "right_wrist"}
JPEG_SIZE = (640, 360)  # the model takes 16:9


def read_sim(obs: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Dream's ``state.*`` observation and the camera images (uint8 HxWx3) of env 0."""
    policy = obs["policy"]
    state = {}
    for arm in ARMS:
        side = arm.replace("_arm", "")  # left / right
        state[f"state.{side}_arm"] = policy[f"{side}_joint_pos"][0].cpu().numpy().astype(np.float64).tolist()
        opening = float(policy[f"{side}_gripper"][0, 0])
        state[f"state.{side}_grip"] = [units.rebot_grip_fraction(opening)]
    images = {key[: -len("_rgb")]: value[0].cpu().numpy() for key, value in policy.items() if key.endswith("_rgb")}
    return state, images


def describe(env) -> str:
    """What the success check sees: what each gripper holds, the success timer, and the first object's spot."""
    held = {}
    for arm in ARMS:
        index = int(env.grasp_assist.held[arm][0])
        held[arm.replace("_arm", "")] = env.object_names[index] if index >= 0 else None
    first = env.object_names[0]
    position, _ = env.object_pose(first)
    cup = env.cup_xy.cpu().numpy()
    table_z = float(env.robot_spec["table_z"])
    return (f" | holds {held} | success timer {float(env.hold_time[0]):.2f}s | {first}: "
            f"{np.linalg.norm(position[:2] - cup):.3f} m from the container axis, {position[2] - table_z:.3f} m above the table")


def main() -> int:
    embodiment = load_embodiment(args.embodiment)
    bridge = Bridge(args.dream_python, args.bridge)
    bridge.call(op="open", checkpoint=args.checkpoint, arm=args.arm, embodiment=embodiment)
    print(f"[dream] policy open: {args.checkpoint} on arm model {args.arm}", flush=True)

    task = load_task(args.task, mode="evaluation", overrides={"sim": {"device": args.device}})
    robot = load_robot(args.robot)
    if args.no_grasp_assist:
        robot["grasp_assist"]["enabled"] = False
    instruction = args.instruction or task["task_description"]
    print(f"[dream] instruction: '{instruction}'", flush=True)
    project = find_rebot_project(args.rebot_project)
    assets = prepare_stationery_assets(task, project)
    cfg = build_env_cfg(task, robot, assets, robot_usd=prepare_robot_asset(robot, project),
                        workstation_usd=find_workstation(robot), enable_cameras=True)
    env = StationeryBimanualEnv(cfg, task, robot, seed=args.seed)
    env.terminate_on_success = False
    dt = env.step_dt
    settle_steps = round(float(task["expert"]["settle_s"]) / dt)
    max_steps = args.max_steps or int(env.max_episode_length) - settle_steps - 1
    steps_per_chunk = units.steps_per_chunk(args.execute, dt, args.action_hz)
    obs_every = max(1, round(0.2 / dt))  # the library keeps frames at 5 fps; hand it a tick that often

    writer = None
    if args.video:
        import imageio

        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=round(1.0 / dt), macro_block_size=1)

    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env.base_seed = seed - env.reset_count
        obs, _ = env.reset()
        print(f"[RESET] episode {episode} seed={env.layouts[0]['seed']}", flush=True)
        hold = BimanualExpertDriver(env, task, robot).current_action()
        for _ in range(settle_steps):
            obs, *_ = env.step(hold)
        bridge.call(op="reset")

        chunks, row0, steps, requests, latency, timed_out = None, 0, 0, 0, [], False
        started = time.time()
        last_q = None
        while steps < max_steps and not bool(env.success[0]):
            if args.episode_timeout is not None and time.time() - started > args.episode_timeout:
                timed_out = True
                break
            state, images = read_sim(obs)
            if writer is not None:
                writer.append_data(np.concatenate([images[n] for n in ROLES.values()], axis=1))
            need_chunk = chunks is None or (steps - row0) >= steps_per_chunk
            if need_chunk or steps % obs_every == 0:
                frames = {f"video.{role}": jpeg_b64(images[src], JPEG_SIZE) for role, src in ROLES.items()}
                bridge.call(op="observe", frames=frames, state=state, prompt=instruction, t=steps * dt)
            if need_chunk:
                answer = bridge.call(op="act", commanded=last_q)
                chunks = {k: np.asarray(v) for k, v in answer["chunks"].items()}
                row0, requests = steps, requests + 1
                latency.append(answer["ms"])
                print(f"[dream] chunk {requests} at step {steps}: {answer['ms']:.0f} ms, sessions replaced "
                      f"so far {answer['replaced']}, {time.time() - started:.0f}s elapsed" + describe(env), flush=True)
            index = units.row_index(steps, row0, dt, len(chunks["q_l_chunk"]), args.action_hz)
            left, right = chunks["q_l_chunk"][index], chunks["q_r_chunk"][index]
            last_q = {"q_l_chunk": left[:6].tolist(), "q_r_chunk": right[:6].tolist()}
            obs, *_ = env.step(torch.as_tensor(units.rebot_command(left, right), device=env.device)[None])
            steps += 1
            if args.realtime:
                time.sleep(max(0.0, started + steps * dt - time.time()))
        info = env.episode_info()
        result = {"episode": episode, "seed": seed, "success": info["success"], "steps": steps,
                  "requests": requests, "model_ms_mean": float(np.mean(latency)) if latency else None,
                  "wall_s": round(time.time() - started, 1), "timed_out": timed_out, "objects": info["objects"]}
        results.append(result)
        usage.record({"kind": "episode", "robot": "rebot", "scene": args.task, "instruction": instruction, "model": args.checkpoint,
                      "run": os.environ.get("DREAMLABS_RUN_ID"), "episode": episode, "requests": requests, "sessions": 1,
                      "model_s": round(sum(latency) / 1000.0, 2), "success": bool(info["success"]), "steps": steps})
        if args.results:
            Path(args.results).parent.mkdir(parents=True, exist_ok=True)
            Path(args.results).write_text(json.dumps(results, indent=2), encoding="utf-8")
        ms = "n/a" if result["model_ms_mean"] is None else f"{result['model_ms_mean']:.0f}"
        print(f"[rollout] episode {episode}: success={result['success']} steps={steps} requests={requests} "
              f"model_ms_mean={ms} wall={result['wall_s']}s" + (" TIMED OUT" if timed_out else ""), flush=True)
        for name, item in info["objects"].items():
            print(f"[rollout]   {name}: final={item['final_position']} grasped={item['grasped']}", flush=True)

    if writer is not None:
        writer.close()
    successes = sum(r["success"] for r in results)
    print(f"[rollout] Done: {successes}/{len(results)} successful", flush=True)
    bridge.close()
    env.close()
    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)

"""Run Dream-1 zero-shot on YAMLab's simulated bimanual YAM, Dream's native robot.

The YAM is the arm Dream-1 was trained on, so this separates "the model" from "the reBot adaptation": if Dream-1 does
well here and badly on the reBot, the arm model or the units are the weak part; if it does badly here too, the model
needs the fine-tune.

    cd ~/yamlab6    # YAMLab on Isaac Sim 6 (see its README)
    D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
    # Dream's MuJoCo scene (apple into the bowl; the assets are found automatically):
    #   python -m dreamlabs_isaaclab.yam --task PutAppleInBowl-v0 --enable_gripper_clamp --enable_cameras --viz kit
    PYTHONPATH=~/yamlab6:~/rebot-duo-course/dreamlabs-isaaclab/src ~/miniconda3/envs/env_yamlab6/bin/python -m dreamlabs_isaaclab.yam \
        --task PutPotOnCooktop-v0 --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000 \
        --num_episodes 3 --enable_gripper_clamp --enable_cameras --viz kit

Like ``scripts/eval/eval_act.py`` (whose environment set-up this repeats), but the policy is Dream-1 behind the
``bridge.py`` subprocess, because the Dream library lives in ``~/dream-env`` (other numpy / jax pins than Isaac
Sim). The API key is read by the library from ``~/.dreamlabs/credentials``.

Units: YAMLab's arm joints are radians, as Dream's; its gripper is finger travel (open -0.0475 m, closed 0), which is
normalized to Dream's 0 (closed) .. 1 (open) both ways. Dream's rows are 20 Hz and are played against the
simulator's control rate by time.
"""

import argparse
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

from dreamlabs_isaaclab import units, usage
from dreamlabs_isaaclab.client import BRIDGE_SCRIPT, DEFAULT_DREAM_PYTHON, Bridge, jpeg_b64

parser = argparse.ArgumentParser(description="Dream-1 zero-shot on YAMLab's bimanual YAM")
parser.add_argument("--checkpoint", default="dreamlabs/dream-1-beta")
parser.add_argument("--dream-python", default=str(DEFAULT_DREAM_PYTHON))
parser.add_argument("--bridge", default=str(BRIDGE_SCRIPT))
parser.add_argument("--task", default="PutPotOnCooktop-v0")
parser.add_argument("--instruction", default=None, help="Default: the task's own words, e.g. 'Put the pot on the cooktop'.")
parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--timeout_steps", type=int, default=900, help="Control steps before an episode counts as a failure")
parser.add_argument("--execute", type=int, default=30, help="Dream rows (20 Hz) executed per chunk")
parser.add_argument("--action-hz", type=float, default=20.0)
parser.add_argument("--camera_width", type=int, default=320)
parser.add_argument("--camera_height", type=int, default=240)
parser.add_argument("--enable_gripper_clamp", action="store_true", default=False)
parser.add_argument("--results", default=None, help="Write per-episode results to this JSON file.")
parser.add_argument("--realtime", action="store_true")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device=None)
args = parser.parse_args()

from yamlab.configs import get_objects_randomization, get_task_kwargs, parse_asset_args, resolve_assets, resolve_device

args.device = resolve_device(args.task, "evaluation", args.device)
simulation_app = AppLauncher(args).app

import numpy as np
import torch
from yamlab.robot.spec import get_robot
from yamlab.utils.task_creation import create_task_environment

SPEC = get_robot("yam")
FINGER_OPEN = abs(float(SPEC.finger_open("left")))  # 0.0475: the finger travel of a fully open gripper
#: Dream's camera key suffix per YAMLab camera (the station keys of a model that reads hand poses).
CAMERAS = {"top": "top", "left_wrist": "left", "right_wrist": "right"}
#: The YAM station keys the base model reads; the library adds the hand poses from the joints.
STATION = {"robot.arm": "yam", "robot.exec_head": "joint", "robot.max_joint_vel_rad_s": 0.0, "robot.gripper_open": 1.0}
SLOT = {"left": "arm_l", "right": "arm_r"}
FRAME_SIZE = (320, 176)
DEFAULT_PROMPT = {"PutPotOnCooktop-v0": "Put the pot on the cooktop", "PutAppleInBowl-v0": "put apple into the bowl"}


def read_env(obs_dict: dict) -> tuple[dict, dict[str, np.ndarray], dict[str, list[float]]]:
    """Dream's station-key observation, the camera images, and the last joints, for env 0."""
    policy = obs_dict.get("policy", obs_dict)
    state, joints = dict(STATION), {}
    for side in ("left", "right"):
        q = policy[f"{side}_arm_joint_pos"][0].cpu().numpy().astype(np.float64)
        opening = float(policy[f"{side}_gripper_state"].reshape(-1)[0])
        state[f"state.{SLOT[side]}.joint_pos"] = q.tolist()
        state[f"state.{SLOT[side]}.gripper_pos"] = [units.yam_grip_fraction(opening, FINGER_OPEN)]
        joints[side] = q.tolist()
    images = {}
    for name in CAMERAS:
        image = policy[f"{name}_camera_rgb"][0].cpu().numpy()
        images[name] = (image * 255).astype(np.uint8) if image.dtype != np.uint8 and image.max() <= 1.0 else image.astype(np.uint8)
    return state, images, joints


def main() -> int:
    cli_assets = parse_asset_args(args.asset)
    if not cli_assets and args.task == "PutAppleInBowl-v0":  # Dream's MuJoCo scene, converted by mujoco_scene_assets
        from dreamlabs_isaaclab.mujoco_scene_assets import yam_asset_args

        cli_assets = yam_asset_args()
    assets = resolve_assets(args.task, cli_assets=cli_assets)
    if not assets:
        raise SystemExit("No object assets: pass --asset pot=... --asset cooktop=... (see the docstring).")
    prompt = args.instruction or DEFAULT_PROMPT.get(args.task, "")
    if not prompt:
        raise SystemExit("Pass --instruction for this task.")

    bridge = Bridge(args.dream_python, args.bridge)
    bridge.call(op="open", config={"checkpoint": args.checkpoint, "timeout_ms": 60000, "strict": False,
                                   "action_hz": int(args.action_hz), "_record": False})
    print(f"[dream] policy open: {args.checkpoint} (YAM station keys); instruction: {prompt!r}", flush=True)

    task_kwargs = get_task_kwargs(args.task, "evaluation")
    task_kwargs.update(timeout_steps=args.timeout_steps, enable_gripper_grasp_clamp=args.enable_gripper_clamp)
    env = create_task_environment(
        task_name=args.task, assets_instance_paths=assets, objects_randomization=get_objects_randomization(args.task),
        init_joint_pos_randomization=0.0, mode="evaluation", observation_modalities=["rgb", "proprioception"],
        num_envs=1, device=args.device, camera_width=args.camera_width, camera_height=args.camera_height,
        image_downsample_factor=1, enable_domain_randomization=False, hdris_path=None, materials_path=None,
        use_unseen_materials=False, **task_kwargs)
    dt = float(getattr(env, "step_dt", 1.0 / 30.0))
    steps_per_chunk = units.steps_per_chunk(args.execute, dt, args.action_hz)
    obs_every = max(1, round(0.2 / dt))

    obs, _ = env.reset()
    bridge.call(op="reset")
    results, steps, ep_steps, requests, chunks, row0, last_q, latency = [], 0, 0, 0, None, 0, None, []
    picked = False
    wall = time.time()
    while simulation_app.is_running() and len(results) < args.num_episodes:
        state, images, _ = read_env(obs)
        need_chunk = chunks is None or (ep_steps - row0) >= steps_per_chunk
        if need_chunk or ep_steps % obs_every == 0:
            frames = {f"image_window.{CAMERAS[n]}": jpeg_b64(images[n], FRAME_SIZE, 85) for n in CAMERAS}
            bridge.call(op="observe", frames=frames, state={**state, "prompt": prompt}, prompt=prompt, t=ep_steps * dt)
        if need_chunk:
            answer = bridge.call(op="act", commanded=last_q)
            chunks = {k: np.asarray(v) for k, v in answer["chunks"].items()}
            row0, requests = ep_steps, requests + 1
            latency.append(answer["ms"])
            print(f"[dream] chunk {requests} at step {ep_steps}: {answer['ms']:.0f} ms, sessions replaced "
                  f"{answer['replaced']}", flush=True)
        index = units.row_index(ep_steps, row0, dt, len(chunks["q_l_chunk"]), args.action_hz)
        left, right = chunks["q_l_chunk"][index], chunks["q_r_chunk"][index]
        last_q = {"q_l_chunk": left[:6].tolist(), "q_r_chunk": right[:6].tolist()}
        obs, _, terminated, truncated, _ = env.step(torch.as_tensor(units.yam_action(left, right, FINGER_OPEN), device=env.device)[None])
        steps, ep_steps = steps + 1, ep_steps + 1
        if hasattr(env, "stage1_success"):
            picked |= bool(env.stage1_success[0])
        if args.realtime:
            time.sleep(max(0.0, wall + steps * dt - time.time()))
        if bool(terminated[0] | truncated[0]):
            success = bool(terminated[0]) and not bool(truncated[0])
            results.append({"episode": len(results), "success": success, "pick": picked or success, "steps": ep_steps,
                            "requests": requests, "model_ms_mean": float(np.mean(latency)) if latency else None})
            usage.record({"kind": "episode", "robot": "yam", "scene": args.task, "instruction": prompt, "model": args.checkpoint,
                          "run": os.environ.get("DREAMLABS_RUN_ID"), "episode": len(results) - 1, "requests": requests,
                          "sessions": 1, "model_s": round(sum(latency) / 1000.0, 2), "success": success, "steps": ep_steps})
            print(f"[EVAL] episode {len(results)}/{args.num_episodes}: {'SUCCESS' if success else 'FAIL'} "
                  f"(pick {'yes' if results[-1]['pick'] else 'no'}, {ep_steps} steps)", flush=True)
            if args.results:
                Path(args.results).parent.mkdir(parents=True, exist_ok=True)
                Path(args.results).write_text(json.dumps(results, indent=2), encoding="utf-8")
            ep_steps, requests, chunks, last_q, latency, picked = 0, 0, None, None, [], False
            bridge.call(op="reset")
    n = len(results)
    if n:
        print("=" * 60)
        print(f"[EVAL] {args.task}: {sum(r['success'] for r in results)}/{n} succeeded, "
              f"{sum(r['pick'] for r in results)}/{n} reached pick")
        print("=" * 60)
    bridge.close()
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

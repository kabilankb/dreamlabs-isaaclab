"""Physics check of the converted MuJoCo-scene assets: do they load, collide and settle sensibly in Isaac Lab?

    python -m dreamlabs_isaaclab.verify_assets [--assets DIR] --headless

The apple is dropped above the bowl and must come to rest inside it; the banana, orange, bottle and plate are dropped
on the floor and must come to rest on it (an origin at z ~ 0 means resting on the face the asset was modelled to rest on).
Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from dreamlabs_isaaclab.mujoco_scene_assets import DEFAULT_OUT, FOLDERS

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--assets", type=Path, default=DEFAULT_OUT)
parser.add_argument("--seconds", type=float, default=4.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import json

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

#: Where each object starts (the apple is above the bowl; the rest drop a few mm onto the floor).
START = {"bowl": (0.0, 0.0, 0.0), "apple": (0.0, 0.0, 0.16), "banana": (0.35, 0.0, 0.02), "orange": (0.35, 0.25, 0.02),
         "water_bottle": (-0.35, 0.0, 0.02), "plate": (-0.35, 0.3, 0.02)}


def main() -> int:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device or "cpu", dt=1 / 120))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=1500.0))
    sizes, objects = {}, {}
    for name, folder in FOLDERS.items():
        directory = args.assets / folder / f"{name}_000"
        sizes[name] = json.loads((directory / "asset_size.json").read_text())
        cfg = RigidObjectCfg(
            prim_path=f"/World/{name}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(directory / f"{name}_000.usd"),
                                       rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=START[name]),
        )
        objects[name] = RigidObject(cfg=cfg)
    sim.reset()
    for _ in range(int(args.seconds * 120)):
        for obj in objects.values():
            obj.write_data_to_sim()
        sim.step()
        for obj in objects.values():
            obj.update(sim.get_physics_dt())
    final = {name: [round(float(v), 4) for v in obj.data.root_pos_w.torch[0]] for name, obj in objects.items()}
    print("[verify] final origins (x, y, z):")
    for name, pos in final.items():
        print(f"[verify]   {name:13s} {pos}")

    rim = sizes["bowl"]["max"]["z"]
    apple, bowl = final["apple"], final["bowl"]
    checks = {
        "apple rests inside the bowl (below the rim, near its axis)":
            apple[2] + sizes["apple"]["size"]["z"] / 2 < rim + 0.01 and abs(apple[0] - bowl[0]) < 0.05 and abs(apple[1] - bowl[1]) < 0.05,
        "apple did not fall through the bowl floor": apple[2] > -0.005,
        "bowl stayed where it was": abs(bowl[0]) < 0.02 and abs(bowl[1]) < 0.02 and bowl[2] < 0.01,
        "banana rests on the floor": abs(final["banana"][2]) < 0.01,
        "orange rests on the floor": abs(final["orange"][2]) < 0.01,
        "bottle stands (origin on the floor)": abs(final["water_bottle"][2]) < 0.01,
        "plate rests on the floor": abs(final["plate"][2]) < 0.01,
    }
    for text, ok in checks.items():
        print(f"[verify] {'PASS' if ok else 'FAIL'}  {text}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)

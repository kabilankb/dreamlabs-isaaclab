# Adapted from YAMLab's put_pot_on_cooktop task (https://github.com/ARISE-Initiative/yamlab), MIT License,
# Copyright (c) 2026 Tianyuan Dai: see LICENSE-YAMLab next to this file (yamlab6-task/) and NOTICE.
"""Configuration for the PutAppleInBowl task (Dream's MuJoCo YAM scene) using ManagerBasedRLEnv.

Six rigid objects on the YAMLab table: the apple (picked), the bowl (target) and four distractors. Their default spots
are Dream's yam_mujoco layout mapped into YAMLab's frame: the arms stand at lateral +-0.305 in both scenes, so
(X, Y) = (0.2525 + (0.24 - y_mujoco), x_mujoco). Per-reset ranges come from configs/tasks/put_apple_in_bowl.yaml.
"""

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from yamlab.configs import pose_range_for
from yamlab.envs.yam_bimanual_scene import YamBimanualSceneCfg
from yamlab.robot.yam import ROBOT
from yamlab.utils.task_logic import check_obj_below_table, compute_sparse_success_reward

from .yam_bimanual_env_cfg import BaseEventsCfg, YamBimanualEnvCfg

TASK_NAME = "PutAppleInBowl-v0"

__all__ = ["PutAppleInBowlSceneCfg", "PutAppleInBowlEventsCfg", "PutAppleInBowlRewardsCfg",
           "PutAppleInBowlTerminationsCfg", "PutAppleInBowlManagerEnvCfg"]

#: Object -> (X, Y) default spot on the YAMLab table, from Dream's MuJoCo positions (see the module docstring).
SPOTS = {
    "apple": (0.3875, 0.150),         # mujoco ( .150,  .105)
    "bowl": (0.5325, 0.000),          # mujoco ( .000, -.040)
    "banana": (0.4075, -0.115),       # mujoco (-.115,  .085)
    "orange": (0.3625, 0.000),        # mujoco ( .000,  .130)
    "water_bottle": (0.6025, 0.220),  # mujoco ( .220, -.110)
    "plate": (0.5375, -0.220),        # mujoco (-.220, -.045)
}
#: Default masses (kg), Dream's. The YAML's per-object ``mass`` is the documented value; these seed the spawner.
MASSES = {"apple": 0.12, "bowl": 2.0, "banana": 0.045, "orange": 0.11, "water_bottle": 0.276, "plate": 0.30}
OBJECT_NAMES = tuple(SPOTS)


def _rigid(name: str) -> RigidObjectCfg:
    """One table object. The USD path is set by configure_assets_instance_paths, which also sets the resting Z."""
    x, y = SPOTS[name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASSES[name]),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(x, y, ROBOT.table_position[2] + 1.0),  # Z is replaced from asset_size.json
            rot=(0.0, 0.0, 0.0, 1.0),  # xyzw identity
        ),
    )


@configclass
class PutAppleInBowlSceneCfg(YamBimanualSceneCfg):
    """Bimanual workstation scene with Dream's six tabletop objects."""

    apple = _rigid("apple")
    bowl = _rigid("bowl")
    banana = _rigid("banana")
    orange = _rigid("orange")
    water_bottle = _rigid("water_bottle")
    plate = _rigid("plate")


def _reset_event(name: str) -> EventTermCfg:
    return EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": pose_range_for(TASK_NAME, name), "velocity_range": {},
                "asset_cfg": SceneEntityCfg(name, body_names=".*")},
    )


@configclass
class PutAppleInBowlEventsCfg(BaseEventsCfg):
    """Reset events: every object moves a little around its default spot (ranges in the task YAML)."""

    reset_apple_position = _reset_event("apple")
    reset_bowl_position = _reset_event("bowl")
    reset_banana_position = _reset_event("banana")
    reset_orange_position = _reset_event("orange")
    reset_water_bottle_position = _reset_event("water_bottle")
    reset_plate_position = _reset_event("plate")


@configclass
class PutAppleInBowlRewardsCfg:
    """Reward terms for the PutAppleInBowl task."""

    sparse_success_reward = RewardTermCfg(func=compute_sparse_success_reward, params={}, weight=1.0)


def check_task_success(env):
    """Per-environment task success: both stage flags latched."""
    return torch.logical_and(env.stage1_success, env.stage2_success)


def _below_table(name: str) -> TerminationTermCfg:
    return TerminationTermCfg(
        func=check_obj_below_table,
        params={"asset_cfg": SceneEntityCfg(name), "table_z": ROBOT.table_position[2], "margin": 0.01},
    )


@configclass
class PutAppleInBowlTerminationsCfg:
    """Termination: timeout, task success, or the apple / bowl falling off the table.

    The distractors may fall off without ending the episode (a knocked-over bottle is not a failure of the task).
    """

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    task_success = TerminationTermCfg(func=check_task_success, params={})
    apple_below_table = _below_table("apple")
    bowl_below_table = _below_table("bowl")


@configclass
class PutAppleInBowlManagerEnvCfg(YamBimanualEnvCfg):
    """Full configuration for the PutAppleInBowl ManagerBasedRLEnv (num_envs defaults to 1)."""

    scene: PutAppleInBowlSceneCfg = PutAppleInBowlSceneCfg(num_envs=1, env_spacing=3.0)

    events: PutAppleInBowlEventsCfg = PutAppleInBowlEventsCfg()
    rewards: PutAppleInBowlRewardsCfg = PutAppleInBowlRewardsCfg()
    terminations: PutAppleInBowlTerminationsCfg = PutAppleInBowlTerminationsCfg()

    obj_name_to_event_name: dict[str, str] = {name: f"reset_{name}_position" for name in OBJECT_NAMES}


# Fallback grasp target if the task YAML had no grasp_detect block. Attached outside @configclass so it is not
# treated as a scene asset field.
PutAppleInBowlManagerEnvCfg.CONTACT_OBJECT_NAMES = ["apple"]

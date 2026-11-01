"""What the launcher page may start: the robots, their settings, and the commands built from them.

Pure Python, no simulator and no network, so it is tested. Every command is an argv list built here from validated
values; nothing the page sends is ever put through a shell, and the API key is never part of a command or of the
environment: the Dream library reads it from ``~/.dreamlabs/credentials`` itself.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()
PACKAGE_SRC = Path(__file__).resolve().parents[1]
APPLE_TASK = "rebot-sim-pipeline/src/rebot_sim_pipeline/lab/configs/tasks/apple_in_bowl.yaml"


def find_course_repo() -> Path:
    """The rebot-duo-course checkout whose rebot-sim-pipeline has the reBot apple task.

    ``$DREAMLABS_COURSE_REPO``, then the folder this package sits in (it lives inside the course repository there), then
    ``~/rebot-duo-course``. When none has the task, the package's parent folder is returned and the reBot apple robot
    reports what is missing.
    """
    beside = Path(__file__).resolve().parents[3]
    for candidate in (os.environ.get("DREAMLABS_COURSE_REPO"), beside, HOME / "rebot-duo-course"):
        if candidate and (Path(candidate).expanduser() / APPLE_TASK).is_file():
            return Path(candidate).expanduser()
    return beside


COURSE_REPO = find_course_repo()

#: Model ids the page offers before the account's own list is fetched.
DEFAULT_MODELS = ("dreamlabs/dream-1-beta", "dreamlabs/dream-1-beta-flash")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@-]{0,127}$")
MAX_INSTRUCTION = 300


class SpecError(ValueError):
    """A launch request the page must not honour; ``str(error)`` is the line shown to the person."""


@dataclass(frozen=True)
class Paths:
    """Where things live on this machine. Every one can be overridden with an environment variable."""

    isaac_python: Path = HOME / "miniconda3/envs/env_yamlab6/bin/python"
    dream_python: Path = HOME / "dream-env/bin/python"
    rebot_sim_pipeline: Path = HOME / "rebot-sim-pipeline"
    yamlab: Path = HOME / "yamlab6"
    runs: Path = HOME / ".dreamlabs-isaaclab/runs"

    @classmethod
    def from_env(cls) -> Paths:
        def pick(name: str, default: Path) -> Path:
            return Path(os.environ.get(name, str(default))).expanduser()

        base = cls()
        return cls(
            isaac_python=pick("DREAMLABS_ISAAC_PYTHON", base.isaac_python),
            dream_python=pick("DREAMLABS_DREAM_PYTHON", base.dream_python),
            rebot_sim_pipeline=pick("DREAMLABS_REBOT_SIM_PIPELINE", base.rebot_sim_pipeline),
            yamlab=pick("DREAMLABS_YAMLAB", base.yamlab),
            runs=pick("DREAMLABS_RUNS", base.runs),
        )


@dataclass(frozen=True)
class Robot:
    """One thing the page can launch."""

    id: str
    name: str
    tagline: str
    simulator: str
    default_instruction: str
    presets: tuple[str, ...]
    has_seed: bool
    notes: tuple[str, ...] = ()
    needs: tuple[str, ...] = field(default=())  # Paths attribute names that must exist for the robot to be ready


ROBOTS: dict[str, Robot] = {
    "rebot": Robot(
        id="rebot",
        name="reBot B601 duo",
        tagline="Two Seeed reBot arms, a top camera and two wrist cameras, putting stationery in a cup.",
        simulator="Isaac Lab",
        default_instruction="pick up the red pen and put it in the pencil cup",
        presets=(
            "pick up the red pen and put it in the pencil cup",
            "Put the two pens and the eraser into the pencil cup",
            "pick up the eraser and put it in the pencil cup",
        ),
        has_seed=True,
        notes=("Zero-shot on an arm Dream-1 was not trained on; it grasps but does not yet release reliably.",),
        needs=("isaac_python", "dream_python", "rebot_sim_pipeline"),
    ),
    "rebot-apple": Robot(
        id="rebot-apple",
        name="reBot B601 duo · apple into the bowl",
        tagline="Dream's MuJoCo scene (apple, bowl, banana, orange, bottle, plate) with the two reBot arms.",
        simulator="Isaac Lab",
        default_instruction="put apple into the bowl",
        presets=("put apple into the bowl", "put orange into the bowl", "put banana on the plate"),
        has_seed=True,
        notes=(
            "Needs the converted assets: python -m dreamlabs_isaaclab.mujoco_scene_assets.",
            "Runs from this repository's rebot-sim-pipeline, not from ~/rebot-sim-pipeline.",
        ),
        needs=("isaac_python", "dream_python"),
    ),
    "yam-gear": Robot(
        id="yam-gear",
        name="YAM · gear assembly",
        tagline="YAMLab's bimanual YAM seating a factory gear on its shaft. Dream's native robot, a hard task.",
        simulator="Isaac Lab (YAMLab)",
        default_instruction="Pick up the gear and place it on the middle shaft of the gear base",
        presets=(
            "Pick up the gear and place it on the middle shaft of the gear base",
            "Pick up the gear",
        ),
        has_seed=False,
        notes=("Needs the gear assets: python scripts/download_gear_assets.py in yamlab6.",),
        needs=("isaac_python", "dream_python", "yamlab"),
    ),
    "yam-apple": Robot(
        id="yam-apple",
        name="YAM · apple into the bowl",
        tagline="Dream's MuJoCo scene (apple, bowl, banana, orange, bottle, plate) rebuilt in YAMLab on Isaac Lab.",
        simulator="Isaac Lab (YAMLab)",
        default_instruction="put apple into the bowl",
        presets=("put apple into the bowl", "put orange into the bowl", "put banana on the plate"),
        has_seed=False,
        notes=("Needs the converted assets: python -m dreamlabs_isaaclab.mujoco_scene_assets.",),
        needs=("isaac_python", "dream_python", "yamlab"),
    ),
    "yam-mujoco": Robot(
        id="yam-mujoco",
        name="YAM · Dream's MuJoCo sim",
        tagline="Dream's own simulator for the YAM. The control experiment: the model is known to work here.",
        simulator="MuJoCo",
        default_instruction="put apple into the bowl",
        presets=("put apple into the bowl", "put orange into the bowl", "put banana on the plate"),
        has_seed=False,
        notes=("Runs for a fixed number of seconds; there is no success check, watch the window.",),
        needs=("dream_python",),
    ),
}


@dataclass(frozen=True)
class Launch:
    """A validated launch request, ready to run."""

    robot: Robot
    model: str
    instruction: str
    episodes: int
    seed: int
    gui: bool
    realtime: bool
    seconds: int

    def public(self) -> dict:
        return {"robot": self.robot.id, "model": self.model, "instruction": self.instruction, "episodes": self.episodes,
                "seed": self.seed, "gui": self.gui, "realtime": self.realtime, "seconds": self.seconds}


def _int(request: dict, key: str, default: int, low: int, high: int) -> int:
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise SpecError(f"{key} must be a whole number")
    if not low <= int(value) <= high:
        raise SpecError(f"{key} must be between {low} and {high}")
    return int(value)


def validate(request: object) -> Launch:
    """Check a launch request from the page; raises SpecError with the reason."""
    if not isinstance(request, dict):
        raise SpecError("the request must be an object")
    robot = ROBOTS.get(request.get("robot"))
    if robot is None:
        raise SpecError(f"unknown robot {request.get('robot')!r}; choose one of {sorted(ROBOTS)}")
    model = request.get("model") or DEFAULT_MODELS[0]
    if not isinstance(model, str) or not MODEL_RE.match(model):
        raise SpecError("the model id has characters a model id never has")
    instruction = request.get("instruction")
    instruction = robot.default_instruction if instruction is None else instruction
    if not isinstance(instruction, str) or not instruction.strip():
        raise SpecError("the instruction is empty")
    instruction = " ".join(instruction.split())
    if len(instruction) > MAX_INSTRUCTION:
        raise SpecError(f"the instruction is longer than {MAX_INSTRUCTION} characters")
    gui, realtime = request.get("gui", True), request.get("realtime", False)
    if not isinstance(gui, bool) or not isinstance(realtime, bool):
        raise SpecError("gui and realtime are true or false")
    return Launch(
        robot=robot,
        model=model,
        instruction=instruction,
        episodes=_int(request, "episodes", 1, 1, 50),
        seed=_int(request, "seed", 1000, 0, 1_000_000),
        gui=gui,
        realtime=realtime,
        seconds=_int(request, "seconds", 90, 10, 3600),
    )


def missing(robot: Robot, paths: Paths) -> list[str]:
    """What a robot needs that is not on this machine, as lines for the page; empty when it is ready."""
    labels = {
        "isaac_python": "the Isaac Sim Python",
        "dream_python": "the Dream environment",
        "rebot_sim_pipeline": "rebot-sim-pipeline",
        "yamlab": "YAMLab",
    }
    out = [f"{labels[name]} ({getattr(paths, name)})" for name in robot.needs if not getattr(paths, name).exists()]
    if robot.id == "yam-gear":
        gear = paths.yamlab / "yamlab_datasets/tasks_data/PutPotOnCooktop/objects/Gear/factory_gear_medium"
        if not gear.exists():
            out.append(f"the gear assets ({gear})")
    if robot.id == "rebot-apple":
        course = COURSE_REPO / APPLE_TASK
        for what, path in (("the converted MuJoCo-scene assets", _apple_assets() / "Apple/apple_000/apple_000.usd"),
                           ("the apple_in_bowl task of rebot-sim-pipeline", course)):
            if not path.exists():
                out.append(f"{what} ({path})")
    if robot.id == "yam-apple":
        apple = _apple_assets() / "Apple/apple_000/apple_000.usd"
        task = paths.yamlab / "yamlab/configs/tasks/put_apple_in_bowl.yaml"
        for what, path in (("the converted MuJoCo-scene assets", apple), ("the PutAppleInBowl task in YAMLab", task)):
            if not path.exists():
                out.append(f"{what} ({path})")
    if robot.id == "yam-mujoco":
        control = _mujoco_control(paths)
        if not control.exists():
            out.append(f"Dream's YAM example ({control})")
    return out


def _apple_assets() -> Path:
    return HOME / ".dreamlabs-isaaclab/assets/yam_mujoco_scene"


def _mujoco_control(paths: Paths) -> Path:
    site = paths.dream_python.parent.parent / "lib"
    found = sorted(site.glob("python3.*/site-packages/dream_examples/yam_mujoco/dream_control.py"))
    return found[0] if found else site / "python3.x/site-packages/dream_examples/yam_mujoco/dream_control.py"


@dataclass(frozen=True)
class Command:
    argv: list[str]
    cwd: Path
    env: dict[str, str]


def build(launch: Launch, paths: Paths, run_dir: Path, display: str | None = None) -> Command:
    """The command, working directory and environment of a launch. The key is in none of them."""
    env = {
        "PYTHONUNBUFFERED": "1",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "OPENBLAS_NUM_THREADS": "1",
    }
    if launch.gui:
        env["DISPLAY"] = display or os.environ.get("DISPLAY") or ":1"
    results = str(run_dir / "results.json")
    robot = launch.robot.id
    if robot == "rebot":
        argv = [str(paths.isaac_python), "-m", "dreamlabs_isaaclab.rebot", "--checkpoint", launch.model,
                "--instruction", launch.instruction, "--episodes", str(launch.episodes), "--seed", str(launch.seed),
                "--device", "cpu", "--episode-timeout", "1500", "--results", results,
                "--video", str(run_dir / "rollout.mp4")]
        argv += ["--viz", "kit"] if launch.gui else ["--headless"]
        argv += ["--realtime"] if launch.realtime else []
        env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_SRC), str(paths.rebot_sim_pipeline / "src")])
        env["OPENBLAS_NUM_THREADS"] = "1"
        return Command(argv, run_dir, env)
    if robot == "yam-gear":
        objects = "yamlab_datasets/tasks_data/PutPotOnCooktop/objects"
        argv = [str(paths.isaac_python), "-m", "dreamlabs_isaaclab.yam", "--checkpoint", launch.model,
                "--task", "PutPotOnCooktop-v0",
                "--asset", f"pot={objects}/Gear/factory_gear_medium",
                "--asset", f"cooktop={objects}/GearBase/factory_gear_base",
                "--instruction", launch.instruction, "--num_episodes", str(launch.episodes),
                "--enable_gripper_clamp", "--enable_cameras", "--results", results]
        argv += ["--viz", "kit"] if launch.gui else ["--headless"]
        argv += ["--realtime"] if launch.realtime else []
        env["PYTHONPATH"] = os.pathsep.join([str(paths.yamlab), str(PACKAGE_SRC)])
        return Command(argv, paths.yamlab, env)
    if robot == "rebot-apple":
        argv = [str(paths.isaac_python), "-m", "dreamlabs_isaaclab.rebot", "--checkpoint", launch.model,
                "--task", "apple_in_bowl", "--instruction", launch.instruction, "--episodes", str(launch.episodes),
                "--seed", str(launch.seed), "--device", "cpu", "--episode-timeout", "1500", "--results", results,
                "--video", str(run_dir / "rollout.mp4")]
        argv += ["--viz", "kit"] if launch.gui else ["--headless"]
        argv += ["--realtime"] if launch.realtime else []
        # The vendored pipeline of this repository has the apple task; ~/rebot-sim-pipeline is your own working copy.
        env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_SRC), str(COURSE_REPO / "rebot-sim-pipeline/src")])
        return Command(argv, run_dir, env)
    if robot == "yam-apple":
        argv = [str(paths.isaac_python), "-m", "dreamlabs_isaaclab.yam", "--checkpoint", launch.model,
                "--task", "PutAppleInBowl-v0", "--instruction", launch.instruction,
                "--num_episodes", str(launch.episodes), "--timeout_steps", "1800",
                "--enable_gripper_clamp", "--enable_cameras", "--results", results]
        argv += ["--viz", "kit"] if launch.gui else ["--headless"]
        argv += ["--realtime"] if launch.realtime else []
        env["PYTHONPATH"] = os.pathsep.join([str(paths.yamlab), str(PACKAGE_SRC)])
        return Command(argv, paths.yamlab, env)
    if robot == "yam-mujoco":
        argv = [str(paths.dream_python), str(_mujoco_control(paths)), "--task", launch.instruction,
                "--model", launch.model, "--seconds", str(launch.seconds)]
        argv += [] if launch.gui else ["--headless"]
        return Command(argv, run_dir, env)
    raise SpecError(f"no command for {robot!r}")  # unreachable: validate() only lets known robots through


def parse_progress(line: str) -> dict | None:
    """The few log lines the page shows as results: ``[rollout] episode ...``, ``[EVAL] ...``, ``Done:``."""
    stripped = line.strip()
    if stripped.startswith(("[rollout]", "[EVAL]", "[RESET]")) or "Dream ·" in stripped:
        return {"text": stripped[:300]}
    return None

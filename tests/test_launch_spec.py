from pathlib import Path

import pytest

from dreamlabs_isaaclab import launch_spec as ls

PATHS = ls.Paths(isaac_python=Path("/x/isaac/python"), dream_python=Path("/x/dream/bin/python"),
                 rebot_sim_pipeline=Path("/x/rsp"), yamlab=Path("/x/yamlab6"), runs=Path("/x/runs"))


def test_a_minimal_request_gets_the_defaults():
    launch = ls.validate({"robot": "rebot"})
    assert launch.model == ls.DEFAULT_MODELS[0]
    assert launch.instruction == ls.ROBOTS["rebot"].default_instruction
    assert (launch.episodes, launch.seed, launch.gui, launch.realtime) == (1, 1000, True, False)


@pytest.mark.parametrize("bad", [
    None, [], "rebot", {}, {"robot": "nope"}, {"robot": "rebot", "episodes": 0}, {"robot": "rebot", "episodes": 51},
    {"robot": "rebot", "episodes": 1.5}, {"robot": "rebot", "episodes": True}, {"robot": "rebot", "seed": -1},
    {"robot": "rebot", "gui": "yes"}, {"robot": "rebot", "realtime": 1}, {"robot": "rebot", "instruction": "  "},
    {"robot": "rebot", "instruction": "x" * 301}, {"robot": "rebot", "model": "a b"},
    {"robot": "rebot", "model": "$(echo injected)"}, {"robot": "rebot", "model": "-flag"},
    {"robot": "yam-mujoco", "seconds": 5},
])
def test_bad_requests_are_refused(bad):
    with pytest.raises(ls.SpecError):
        ls.validate(bad)


def test_the_instruction_is_one_argv_element_and_whitespace_is_collapsed():
    launch = ls.validate({"robot": "rebot", "instruction": "put  the pen;  echo injected && true \n in the cup"})
    command = ls.build(launch, PATHS, Path("/x/runs/r1"))
    assert launch.instruction == "put the pen; echo injected && true in the cup"
    assert command.argv[command.argv.index("--instruction") + 1] == launch.instruction  # no shell: stays one argument


def test_no_command_or_environment_carries_a_key():
    for robot in ls.ROBOTS:
        command = ls.build(ls.validate({"robot": robot}), PATHS, Path("/x/runs/r1"))
        text = " ".join(command.argv) + " ".join(command.env.values())
        assert "dl_live" not in text and "KEY" not in " ".join(command.env)


def test_rebot_command():
    launch = ls.validate({"robot": "rebot", "episodes": 3, "seed": 1005, "gui": False, "model": "dream-1-ft:abc-x"})
    command = ls.build(launch, PATHS, Path("/x/runs/r1"))
    argv = command.argv
    assert argv[:3] == ["/x/isaac/python", "-m", "dreamlabs_isaaclab.rebot"]
    assert argv[argv.index("--seed") + 1] == "1005" and argv[argv.index("--episodes") + 1] == "3"
    assert argv[argv.index("--checkpoint") + 1] == "dream-1-ft:abc-x"
    assert "--headless" in argv and "--viz" not in argv and "DISPLAY" not in command.env
    assert "/x/rsp/src" in command.env["PYTHONPATH"]


def test_gui_sets_a_display_and_viz():
    command = ls.build(ls.validate({"robot": "rebot", "gui": True, "realtime": True}), PATHS, Path("/x/r"), display=":7")
    assert command.env["DISPLAY"] == ":7"
    assert command.argv[command.argv.index("--viz") + 1] == "kit" and "--realtime" in command.argv


def test_yam_gear_command_uses_the_gear_assets_from_yamlab():
    command = ls.build(ls.validate({"robot": "yam-gear", "gui": False}), PATHS, Path("/x/r"))
    assert command.cwd == Path("/x/yamlab6")
    assert "pot=yamlab_datasets/tasks_data/PutPotOnCooktop/objects/Gear/factory_gear_medium" in command.argv
    assert command.argv[1:3] == ["-m", "dreamlabs_isaaclab.yam"] and "--seed" not in command.argv


def test_rebot_apple_command_uses_the_vendored_pipeline_not_the_working_copy():
    command = ls.build(ls.validate({"robot": "rebot-apple", "seed": 1003, "gui": False}), PATHS, Path("/x/r"))
    assert command.argv[command.argv.index("--task") + 1] == "apple_in_bowl"
    assert command.argv[command.argv.index("--seed") + 1] == "1003" and "--headless" in command.argv
    assert str(ls.COURSE_REPO / "rebot-sim-pipeline/src") in command.env["PYTHONPATH"]
    assert "/x/rsp" not in command.env["PYTHONPATH"]          # PATHS.rebot_sim_pipeline, the working copy
    assert ls.ROBOTS["rebot-apple"].has_seed and ls.ROBOTS["rebot-apple"].default_instruction == "put apple into the bowl"


def test_yam_apple_command_runs_the_new_yamlab_task():
    command = ls.build(ls.validate({"robot": "yam-apple", "episodes": 2, "gui": False}), PATHS, Path("/x/r"))
    assert command.cwd == Path("/x/yamlab6") and command.argv[1:3] == ["-m", "dreamlabs_isaaclab.yam"]
    assert command.argv[command.argv.index("--task") + 1] == "PutAppleInBowl-v0"
    assert command.argv[command.argv.index("--num_episodes") + 1] == "2" and "--asset" not in command.argv  # found by the runner
    assert ls.ROBOTS["yam-apple"].default_instruction == "put apple into the bowl"


def test_mujoco_command_runs_dreams_example_with_seconds():
    command = ls.build(ls.validate({"robot": "yam-mujoco", "seconds": 120, "gui": False}), PATHS, Path("/x/r"))
    assert command.argv[0] == "/x/dream/bin/python" and command.argv[1].endswith("dream_control.py")
    assert command.argv[command.argv.index("--seconds") + 1] == "120" and "--headless" in command.argv


def test_missing_reports_what_the_robot_needs(tmp_path):
    paths = ls.Paths(isaac_python=tmp_path / "py", dream_python=tmp_path / "dream", rebot_sim_pipeline=tmp_path / "rsp",
                     yamlab=tmp_path / "y", runs=tmp_path / "runs")
    gaps = ls.missing(ls.ROBOTS["rebot"], paths)
    assert len(gaps) == 3 and any("rebot-sim-pipeline" in g for g in gaps)
    (tmp_path / "py").touch()
    (tmp_path / "dream").touch()
    (tmp_path / "rsp").mkdir()
    assert ls.missing(ls.ROBOTS["rebot"], paths) == []
    with_yamlab = ls.Paths(isaac_python=tmp_path / "py", dream_python=tmp_path / "dream", rebot_sim_pipeline=tmp_path / "rsp",
                           yamlab=tmp_path / "rsp", runs=tmp_path / "runs")
    assert any("gear assets" in g for g in ls.missing(ls.ROBOTS["yam-gear"], with_yamlab))


def test_progress_lines_are_filtered():
    assert ls.parse_progress("[rollout] episode 0: success=False steps=100") is not None
    assert ls.parse_progress("[EVAL] 0/1 succeeded") is not None
    assert ls.parse_progress("2026-11-01 [Warning] [omni.rtx] Skipping unsupported GPU") is None


def test_the_course_repo_is_found_through_the_environment_or_reported_missing(tmp_path, monkeypatch):
    marker = tmp_path / "course" / ls.APPLE_TASK
    marker.parent.mkdir(parents=True)
    marker.write_text("task_name: x\n")
    monkeypatch.setenv("DREAMLABS_COURSE_REPO", str(tmp_path / "course"))
    assert ls.find_course_repo() == tmp_path / "course"
    monkeypatch.setenv("DREAMLABS_COURSE_REPO", str(tmp_path / "elsewhere"))      # no task there: falls back, never raises
    assert isinstance(ls.find_course_repo(), Path)

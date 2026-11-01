import itertools
import json

import numpy as np
import pytest

from dreamlabs_isaaclab import client, units


def test_grip_fraction_ignores_sign_and_clips():
    assert units.yam_grip_fraction(-0.0475) == pytest.approx(1.0)  # YAM open is negative travel
    assert units.yam_grip_fraction(0.0) == 0.0
    assert units.yam_grip_fraction(-0.1) == 1.0
    assert units.rebot_grip_fraction(0.025) == pytest.approx(0.5)
    assert units.rebot_grip_fraction(0.2) == 1.0


def test_rebot_command_passes_joints_and_scales_gripper():
    left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    right = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.0]
    out = units.rebot_command(left, right)
    assert out.shape == (14,) and out.dtype == np.float32
    np.testing.assert_allclose(out[:6], left[:6], atol=1e-6)
    assert out[6] == pytest.approx(units.REBOT_GRIPPER_OPEN_M)
    np.testing.assert_allclose(out[7:13], right[:6], atol=1e-6)
    assert out[13] == 0.0


def test_gripper_round_trip_for_both_scenes():
    for fraction in (0.0, 0.25, 0.5, 1.0):
        row = [0.0] * 6 + [fraction]
        rebot = units.rebot_command(row, row)
        assert units.rebot_grip_fraction(rebot[6]) == pytest.approx(fraction, abs=1e-6)
        yam = units.yam_action(row, row)
        assert yam[6] <= 0.0  # YAMLab's open finger target is negative
        assert units.yam_grip_fraction(yam[6]) == pytest.approx(fraction, abs=1e-6)


def test_gripper_command_is_clipped_to_the_open_range():
    row = [0.0] * 6 + [1.7]
    assert units.rebot_command(row, row)[6] == pytest.approx(units.REBOT_GRIPPER_OPEN_M)
    assert units.yam_action(row, row)[6] == pytest.approx(-units.YAM_FINGER_OPEN_M)


def test_a_malformed_row_is_refused():
    with pytest.raises(ValueError):
        units.rebot_command([0.0] * 6, [0.0] * 7)
    with pytest.raises(ValueError):
        units.yam_action([0.0] * 7, [0.0] * 8)


def test_row_index_plays_rows_against_simulated_time():
    # A 30 Hz scene plays a 20 Hz chunk: some rows repeat; the index never leaves the chunk.
    dt = 1 / 30
    indices = [units.row_index(step, 0, dt, 48) for step in range(45)]
    assert indices[0] == 0 and indices == sorted(indices) and indices[-1] == 29  # step 44 = 1.467 s = row 29.3
    assert units.row_index(10_000, 0, dt, 48) == 47
    # A 20 Hz scene plays each row once.
    assert [units.row_index(s, 5, 1 / 20, 48) for s in range(5, 10)] == [0, 1, 2, 3, 4]


def test_steps_per_chunk():
    assert units.steps_per_chunk(30, 1 / 30) == 45  # 30 rows = 1.5 s = 45 steps at 30 Hz
    assert units.steps_per_chunk(30, 1 / 20) == 30
    assert units.steps_per_chunk(0, 1 / 30) == 1


def test_shipped_embodiments_are_wellformed_and_tile_their_spans():
    for name in ("rebot_bimanual_embodiment", "rebot_zero_shot_embodiment"):
        config = client.load_embodiment(name)
        assert set(config) >= {"state", "action", "video"}
        for vector in ("state", "action"):
            spans = sorted(config[vector].values())
            assert spans[0][0] == 0
            assert all(a[1] == b[0] for a, b in itertools.pairwise(spans)), f"{name}.{vector} spans leave a gap or overlap"
        assert set(config["video"]) == {"overhead", "wrist_l", "wrist_r"}
        json.dumps(config)


def test_the_bridge_script_ships_with_the_package():
    assert client.BRIDGE_SCRIPT.is_file()

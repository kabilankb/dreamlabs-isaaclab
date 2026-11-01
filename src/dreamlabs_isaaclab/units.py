"""Unit conversions between Isaac Lab scenes and Dream-1, with no simulator imports (so they can be tested).

Dream-1 speaks arm joints in radians and a gripper as an open fraction, 0 (closed) to 1 (open). Each scene has its own
gripper unit, so each gets one pair of functions here: scene -> Dream (``*_grip_fraction``) and Dream -> scene
(``*_command`` / ``yam_action``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: reBot B601: per-finger opening of a fully open gripper, metres (``rebot_sim_pipeline.conventions.GRIPPER_OPEN_M``).
REBOT_GRIPPER_OPEN_M = 0.05
#: YAMLab's YAM: finger travel of a fully open gripper, metres. Open is -0.0475, closed 0 (the sign is the joint's).
YAM_FINGER_OPEN_M = 0.0475
#: Rows per second of Dream-1's action chunks.
DREAM_ACTION_HZ = 20.0


def grip_fraction(opening_m: float, fully_open_m: float) -> float:
    """A gripper's opening as Dream's 0 (closed) .. 1 (open); the sign of ``opening_m`` is ignored."""
    return float(np.clip(abs(opening_m) / fully_open_m, 0.0, 1.0))


def rebot_grip_fraction(finger_mean_m: float) -> float:
    return grip_fraction(finger_mean_m, REBOT_GRIPPER_OPEN_M)


def yam_grip_fraction(finger_pos_m: float, finger_open_m: float = YAM_FINGER_OPEN_M) -> float:
    return grip_fraction(finger_pos_m, finger_open_m)


def rebot_command(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    """Two Dream rows (6 joints in rad, grip 0..1) -> the reBot env action ``[left q6, gripper m, right q6, gripper m]``.

    The reBot URDF joints of Dream's ``seeed_faas`` arm model are the simulator's own joint angles, so the joints pass
    through; only the gripper changes unit (fraction -> metres).
    """
    out: list[float] = []
    for row in (left, right):
        row = np.asarray(row, dtype=np.float64)
        if row.shape != (7,):
            raise ValueError(f"a Dream row has 7 values (6 joints + grip), got {row.shape}")
        out += [*row[:6], float(np.clip(row[6], 0.0, 1.0)) * REBOT_GRIPPER_OPEN_M]
    return np.asarray(out, dtype=np.float32)


def yam_action(left: Sequence[float], right: Sequence[float], finger_open_m: float = YAM_FINGER_OPEN_M) -> np.ndarray:
    """Two Dream rows -> YAMLab's action ``[left q6, left gripper, right q6, right gripper]`` (finger target, negative)."""
    out: list[float] = []
    for row in (left, right):
        row = np.asarray(row, dtype=np.float64)
        if row.shape != (7,):
            raise ValueError(f"a Dream row has 7 values (6 joints + grip), got {row.shape}")
        out += [*row[:6], -finger_open_m * float(np.clip(row[6], 0.0, 1.0))]
    return np.asarray(out, dtype=np.float32)


def row_index(step: int, first_step: int, step_dt: float, n_rows: int, action_hz: float = DREAM_ACTION_HZ) -> int:
    """Which row of a 20 Hz chunk to play at control step ``step``, for a chunk installed at ``first_step``.

    Rows are played against simulated time, so a 30 Hz scene repeats some rows and a 20 Hz scene plays each once.
    """
    return int(min(round((step - first_step) * step_dt * action_hz), n_rows - 1))


def steps_per_chunk(execute_rows: int, step_dt: float, action_hz: float = DREAM_ACTION_HZ) -> int:
    """Control steps that ``execute_rows`` rows of a chunk last; at least 1."""
    return max(1, int(execute_rows / action_hz / step_dt))

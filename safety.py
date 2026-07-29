"""Safety checks for bimanual Piper deployment.

Call SafetyChecker.check() before every env.step().
On violation it raises SafetyViolation; caller must call env.emergency_stop().
"""

import time
import numpy as np
from piper_env import JOINT_LIMITS_RAD, GRIPPER_RANGE_M

# Maximum allowed joint delta per step (radians).
# At 50 Hz: 0.3 rad/step ≈ 15 rad/s ≈ 860 °/s — already conservative.
MAX_JOINT_DELTA_RAD = 0.3
MAX_GRIPPER_DELTA_M = 0.02   # 20 mm per step

STALE_IMAGE_THRESHOLD_S = 0.5
COMM_WATCHDOG_S = 1.0   # if no valid qpos read in this window, stop


class SafetyViolation(RuntimeError):
    pass


class SafetyChecker:
    def __init__(self):
        self._last_qpos_time: float = 0.0
        self._last_qpos: np.ndarray | None = None

    def record_qpos(self, qpos: np.ndarray):
        self._last_qpos = qpos.copy()
        self._last_qpos_time = time.time()

    def check(
        self,
        current_qpos: np.ndarray,
        proposed_action: np.ndarray,
        image_timestamps: dict,
    ):
        """Raise SafetyViolation if any check fails."""
        self._check_joint_limits(proposed_action)
        if self._last_qpos is not None:
            self._check_delta(current_qpos, proposed_action)
        self._check_stale_images(image_timestamps)
        self._check_comm_watchdog()

    # ---- individual checks ----

    def _check_joint_limits(self, action: np.ndarray):
        # action layout: [left×7, right×7]
        for side_offset in (0, 7):
            joints = action[side_offset : side_offset + 6]
            gripper = action[side_offset + 6]
            for i, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
                if not (lo <= joints[i] <= hi):
                    raise SafetyViolation(
                        f"joint{i+1} action {joints[i]:.4f} rad out of limits [{lo}, {hi}]"
                        f" (side={'left' if side_offset==0 else 'right'})"
                    )
            g_lo, g_hi = GRIPPER_RANGE_M
            if not (g_lo <= gripper <= g_hi):
                raise SafetyViolation(
                    f"gripper action {gripper:.4f} m out of range [{g_lo}, {g_hi}]"
                    f" (side={'left' if side_offset==0 else 'right'})"
                )

    def _check_delta(self, current: np.ndarray, action: np.ndarray):
        for side_offset in (0, 7):
            j_delta = np.abs(action[side_offset:side_offset+6] - current[side_offset:side_offset+6])
            if j_delta.max() > MAX_JOINT_DELTA_RAD:
                worst = j_delta.argmax()
                raise SafetyViolation(
                    f"joint{worst+1} delta {j_delta[worst]:.4f} rad exceeds limit {MAX_JOINT_DELTA_RAD}"
                    f" (side={'left' if side_offset==0 else 'right'})"
                )
            g_delta = abs(action[side_offset+6] - current[side_offset+6])
            if g_delta > MAX_GRIPPER_DELTA_M:
                raise SafetyViolation(
                    f"gripper delta {g_delta:.4f} m exceeds limit {MAX_GRIPPER_DELTA_M}"
                    f" (side={'left' if side_offset==0 else 'right'})"
                )

    def _check_stale_images(self, timestamps: dict):
        now = time.time()
        for key, t in timestamps.items():
            age = now - t
            if age > STALE_IMAGE_THRESHOLD_S:
                raise SafetyViolation(f"stale image: {key} is {age*1000:.0f} ms old")

    def _check_comm_watchdog(self):
        if self._last_qpos_time == 0.0:
            return
        age = time.time() - self._last_qpos_time
        if age > COMM_WATCHDOG_S:
            raise SafetyViolation(f"no qpos update for {age:.2f} s (watchdog limit {COMM_WATCHDOG_S} s)")

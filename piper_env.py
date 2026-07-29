"""Bimanual Piper environment using piper-sdk.

State/action layout (14D):
  [left_j1..j6 (rad), left_gripper (m), right_j1..j6 (rad), right_gripper (m)]

CAN port assignment is UNKNOWN until hardware is connected.
Default: left_can="can0", right_can="can1". Verify with `ip link show` or `ls /sys/class/net/`.
"""

import math
import time
import numpy as np
from piper_sdk import C_PiperInterface_V2

# --- unit conversion constants ---
_RAD_TO_MDEG = 1000.0 * 180.0 / math.pi   # rad  →  0.001 °
_M_TO_MM01   = 1_000_000.0                  # m    →  0.001 mm

# Joint limits from piper_sdk C_PiperParamManager (radians)
JOINT_LIMITS_RAD = np.array([
    (-2.6179,  2.6179),
    ( 0.0000,  3.1400),
    (-2.9670,  0.0000),
    (-1.7450,  1.7450),
    (-1.2200,  1.2200),
    (-2.0944,  2.0944),
], dtype=np.float64)

GRIPPER_RANGE_M = (0.0, 0.07)   # fully closed → fully open


def _joints_mdeg_to_rad(joint_state) -> np.ndarray:
    return np.array([
        joint_state.joint_1,
        joint_state.joint_2,
        joint_state.joint_3,
        joint_state.joint_4,
        joint_state.joint_5,
        joint_state.joint_6,
    ], dtype=np.float32) / _RAD_TO_MDEG


def _gripper_mm01_to_m(gripper_state) -> float:
    return float(gripper_state.grippers_angle) / _M_TO_MM01


class SingleArm:
    """Wraps one C_PiperInterface_V2 instance."""

    def __init__(self, can_name: str, speed_pct: int = 30):
        # judge_flag=False: skip CAN port validation (needed when can0/can1 not yet up)
        # can_auto_init=False: defer CAN init to connect()
        self._arm = C_PiperInterface_V2(can_name, judge_flag=False, can_auto_init=False)
        self._speed_pct = speed_pct

    def connect(self):
        self._arm.ConnectPort(can_init=True, piper_init=True)
        time.sleep(0.5)
        deadline = time.time() + 5.0
        while not self._arm.EnablePiper():
            if time.time() > deadline:
                raise RuntimeError(f"EnablePiper timed out on {self._arm.GetCanName()}")
            time.sleep(0.05)
        # MOVE J mode, CAN control
        self._arm.ModeCtrl(0x01, 0x01, self._speed_pct, 0x00)
        time.sleep(0.05)

    def disconnect(self):
        try:
            self._arm.DisablePiper()
        finally:
            self._arm.DisconnectPort()

    def emergency_stop(self):
        self._arm.EmergencyStop(0x01)

    def recover(self):
        """Reset e-stop and re-enable after EmergencyStop."""
        self._arm.EmergencyStop(0x02)
        time.sleep(0.3)
        self._arm.EnablePiper()
        self._arm.ModeCtrl(0x01, 0x01, self._speed_pct, 0x00)

    def read(self) -> np.ndarray:
        """Return 7D: [j1..j6 (rad), gripper (m)]."""
        j = _joints_mdeg_to_rad(self._arm.GetArmJointMsgs().joint_state)
        g = _gripper_mm01_to_m(self._arm.GetArmGripperMsgs().gripper_state)
        return np.append(j, g)

    def send(self, joints_rad: np.ndarray, gripper_m: float):
        """Send joint angles (6×rad) and gripper position (m)."""
        j_mdeg = [round(float(v) * _RAD_TO_MDEG) for v in joints_rad]
        self._arm.JointCtrl(*j_mdeg)
        gripper_mm01 = round(abs(gripper_m) * _M_TO_MM01)
        # gripper_code=0x01 → enable
        self._arm.GripperCtrl(gripper_mm01, 1000, 0x01, 0)

    def go_home(self):
        """Send all joints to 0 rad, gripper to closed (0 m)."""
        self.send(np.zeros(6), GRIPPER_RANGE_M[0])

    def set_speed_pct(self, pct: int):
        self._speed_pct = pct
        self._arm.ModeCtrl(0x01, 0x01, pct, 0x00)

    @property
    def raw(self) -> C_PiperInterface_V2:
        return self._arm


class PiperBimanualEnv:
    """Bimanual Piper environment.

    Usage:
        env = PiperBimanualEnv()
        env.connect()
        qpos = env.get_qpos()    # 14D ndarray
        env.step(action_14d)
        env.disconnect()
    """

    def __init__(
        self,
        left_can: str = "can0",    # UNKNOWN — verify with `ip link show`
        right_can: str = "can1",   # UNKNOWN — verify with `ip link show`
        speed_pct: int = 30,
    ):
        self.left  = SingleArm(left_can,  speed_pct)
        self.right = SingleArm(right_can, speed_pct)

    def connect(self):
        self.left.connect()
        self.right.connect()

    def disconnect(self):
        self.left.disconnect()
        self.right.disconnect()

    def emergency_stop(self):
        self.left.emergency_stop()
        self.right.emergency_stop()

    def recover(self):
        self.left.recover()
        self.right.recover()

    def get_qpos(self) -> np.ndarray:
        """Read 14D state: [left×7, right×7]."""
        return np.concatenate([self.left.read(), self.right.read()])

    def step(self, action: np.ndarray):
        """Send 14D action: [left_j1..6, left_gripper, right_j1..6, right_gripper]."""
        self.left.send(action[0:6], float(action[6]))
        self.right.send(action[7:13], float(action[13]))

    def go_home(self):
        self.left.go_home()
        self.right.go_home()

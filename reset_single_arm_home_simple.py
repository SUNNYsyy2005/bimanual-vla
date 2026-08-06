#!/usr/bin/env python3
"""Minimal Piper SDK MOVE J home command based on piper_ctrl_go_zero.py."""

import time

from piper_sdk import C_PiperInterface_V2


CAN_NAME = "can0"
TARGET_JOINTS = (90000, 0, 0, 0, 0, 0)  # Piper unit: 0.001 degree
TARGET_GRIPPER = 0
SPEED_PERCENT = 10
COMMAND_HZ = 200
TIMEOUT_S = 30.0


def main() -> None:
    piper = C_PiperInterface_V2(CAN_NAME)
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)

    steps = int(COMMAND_HZ * TIMEOUT_S)
    for step in range(steps):
        piper.ModeCtrl(0x01, 0x01, SPEED_PERCENT, 0x00)
        piper.JointCtrl(*TARGET_JOINTS)
        piper.GripperCtrl(TARGET_GRIPPER, 1000, 0x01, 0)

        if step % COMMAND_HZ == 0:
            joints = piper.GetArmJointMsgs().joint_state
            current = (
                joints.joint_1,
                joints.joint_2,
                joints.joint_3,
                joints.joint_4,
                joints.joint_5,
                joints.joint_6,
            )
            current_deg = [round(value / 1000.0, 3) for value in current]
            max_error = max(
                abs(value - target) for value, target in zip(current, TARGET_JOINTS)
            )
            print(f"joints={current_deg} max_error={max_error / 1000.0:.3f}deg")
            if max_error <= 1000:
                print("Piper home reached: [90, 0, 0, 0, 0, 0] deg, gripper closed")
                return

        time.sleep(1.0 / COMMAND_HZ)

    raise RuntimeError("Piper did not reach the home target within 30 seconds")


if __name__ == "__main__":
    main()

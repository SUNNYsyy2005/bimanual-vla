"""Collect measured Piper output-arm feedback and synchronized RGB cameras.

Supported raw contracts:

* single + delivery: 10D state / 10D absolute EEF fallback target / 2 cameras;
* single + joint: 7D state / 7D next measured joint target / 2 cameras;
* bimanual + delivery: 20D state / 20D absolute EEF fallback target / 3 cameras;
* bimanual + joint: 14D state / 14D next measured joint target / 3 cameras.

This output-only collector cannot see the teleoperator's commanded target, so
joint actions are deliberately labelled ``next_measured_joint_fallback``.  For preferred
same-step master-arm targets use ``teleop_single.py`` or ``teleop.py``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from camera import CameraCapture
from piper_data_contract import (
    BIMANUAL,
    DEFAULT_FPS,
    DELIVERY_SCHEMA,
    IMAGE_HW,
    JOINT_SCHEMA,
    JOINT_MEASURED_ACTION_SOURCE,
    SINGLE_ARM,
    EpisodeBuffer,
    EpisodeContract,
    DEFAULT_ACTION_HORIZON,
    DELIVERY_MEASURED_ACTION_SOURCE,
    build_actions as _build_actions,
    build_delivery_state,
    gripper_opening_fraction,
    gripper_opening_m,
)
try:
    from teleop import KeyListener
except ModuleNotFoundError:
    class KeyListener:  # pragma: no cover - only used without the hardware stack
        def __init__(self):
            raise RuntimeError("teleop/piper_sdk is not installed in this environment")

try:
    from piper_sdk import C_PiperInterface_V2
except ModuleNotFoundError:  # Allows schema/tests to run without the hardware SDK.
    C_PiperInterface_V2 = None  # type: ignore[assignment]


RAD_FACTOR = 57295.7795  # Piper unit: 0.001 degree -> rad
GRIPPER_FACTOR = 1_000_000.0  # Piper unit: 0.001 mm -> metre
DEFAULT_CAN = "can0"
DEFAULT_LEFT_CAN = "can0"
DEFAULT_RIGHT_CAN = "can1"
DEFAULT_HIGH_DEVICE = "auto"
DEFAULT_WRIST_DEVICE = "auto"
# The rig has two D405 wrist cameras. Bind them to their USB topology rather
# than model-only discovery, which cannot distinguish identical D405 devices
# and previously selected the ASUS/internal webcam for the left wrist.
DEFAULT_LEFT_WRIST_DEVICE = "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:7.2:1.0-video-index4"
DEFAULT_RIGHT_WRIST_DEVICE = "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:4.2:1.0-video-index4"
DEFAULT_CAMERA_FPS = 30
CAMERA_SOURCE_HW = (240, 424)
PIPER_FEEDBACK_MAX_AGE_S = 0.5
CAN_INTERFACE_UP_FLAG = 0x1


class PiperFeedbackStaleError(RuntimeError):
    """Raised when Piper SDK getters only contain old cached CAN feedback."""


def require_can_interface_up(
    can_name: str,
    *,
    sysfs_root: pathlib.Path = pathlib.Path("/sys/class/net"),
) -> None:
    """Fail early when a SocketCAN interface exists but is not operationally up.

    ``python-can`` can successfully create a socket for a DOWN interface.  The
    first read then fails with ``Network is down``, while Piper's SDK may only
    expose that as stale cached feedback.  Checking the Linux interface flags
    before constructing the SDK object gives the GUI an actionable error.
    """
    name = can_name.strip()
    if not name:
        raise RuntimeError("CAN interface name must not be empty")
    interface_dir = sysfs_root / name
    if not interface_dir.is_dir():
        raise RuntimeError(
            f"CAN interface {name!r} does not exist. Connect the USB-CAN adapter first."
        )
    try:
        flags = int((interface_dir / "flags").read_text().strip(), 0)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read Linux link flags for CAN interface {name!r}: {exc}") from exc
    if not flags & CAN_INTERFACE_UP_FLAG:
        raise RuntimeError(
            f"CAN interface {name!r} exists but is DOWN. Activate it at 1000000 bit/s "
            "and confirm that 'candump' receives live Piper frames before connecting the GUI."
        )


def _require_fresh_feedback(
    messages: dict[str, object],
    *,
    max_age_s: float = PIPER_FEEDBACK_MAX_AGE_S,
) -> None:
    """Reject missing/stale SDK cache values before recording an episode."""
    now = time.time()
    failures = []
    for name, message in messages.items():
        timestamp = float(getattr(message, "time_stamp", 0.0) or 0.0)
        hz = float(getattr(message, "Hz", 0.0) or 0.0)
        age_s = now - timestamp if timestamp > 0 else float("inf")
        if timestamp <= 0 or age_s > max_age_s or age_s < -1.0:
            failures.append(f"{name}: age={age_s:.3f}s Hz={hz:.1f}")
    if failures:
        raise PiperFeedbackStaleError(
            "Piper CAN feedback is missing or stale; reconnect before collecting: "
            + "; ".join(failures)
        )


def _qpos_from_feedback(joints_message: Any, gripper_message: Any) -> np.ndarray:
    joints = joints_message.joint_state
    gripper = gripper_message.gripper_state
    values = np.array(
        [
            joints.joint_1,
            joints.joint_2,
            joints.joint_3,
            joints.joint_4,
            joints.joint_5,
            joints.joint_6,
        ],
        dtype=np.float32,
    ) / RAD_FACTOR
    opening_m = abs(float(gripper.grippers_angle)) / GRIPPER_FACTOR
    return np.append(values, gripper_opening_fraction(opening_m)).astype(np.float32)


def read_output_qpos(piper: Any) -> np.ndarray:
    """Read fresh measured joint feedback and gripper opening for one arm."""
    joints_message = piper.GetArmJointMsgs()
    gripper_message = piper.GetArmGripperMsgs()
    _require_fresh_feedback({"joint": joints_message, "gripper": gripper_message})
    return _qpos_from_feedback(joints_message, gripper_message)


def read_output_gripper_command_sample(
    piper: Any,
    *,
    max_age_s: float = PIPER_FEEDBACK_MAX_AGE_S,
) -> tuple[float, float] | None:
    """Read the latest opening-fraction target and its SDK/CAN timestamp."""
    getter = getattr(piper, "GetArmGripperCtrl", None)
    if not callable(getter):
        return None
    try:
        message = getter()
        timestamp = float(getattr(message, "time_stamp", 0.0) or 0.0)
        age_s = time.time() - timestamp if timestamp > 0 else float("inf")
        if timestamp <= 0 or age_s > max_age_s or age_s < -1.0:
            return None
        command = getattr(message, "gripper_ctrl", None)
        opening_m = abs(float(getattr(command, "grippers_angle"))) / GRIPPER_FACTOR
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(opening_m):
        return None
    return gripper_opening_fraction(opening_m), timestamp


def read_output_gripper_command_target(
    piper: Any,
    *,
    max_age_s: float = PIPER_FEEDBACK_MAX_AGE_S,
) -> float | None:
    """Backward-compatible value-only view of the latest gripper command."""
    sample = read_output_gripper_command_sample(piper, max_age_s=max_age_s)
    return None if sample is None else sample[0]

def read_robot_gripper_command_targets(
    robot: Any,
    *,
    arm_mode: str,
) -> np.ndarray | None:
    """Read normalized absolute gripper targets for one or two output arms.

    A bimanual result may contain one missing arm as ``NaN``; ``None`` is
    returned only when no arm has a usable command target.
    """
    if arm_mode == SINGLE_ARM:
        value = read_output_gripper_command_target(robot)
        return None if value is None else np.asarray([value], dtype=np.float32)
    if arm_mode != BIMANUAL or not isinstance(robot, dict):
        raise ValueError("arm_mode must be single or bimanual with the matching robot object")
    values = np.asarray(
        [
            read_output_gripper_command_target(robot["left"]),
            read_output_gripper_command_target(robot["right"]),
        ],
        dtype=np.float32,
    )
    return values if np.isfinite(values).any() else None


def read_robot_gripper_command_samples(
    robot: Any,
    *,
    arm_mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return per-arm command targets and original command timestamps."""
    if arm_mode == SINGLE_ARM:
        arms = [robot]
    elif arm_mode == BIMANUAL and isinstance(robot, dict) and set(robot) == {"left", "right"}:
        arms = [robot["left"], robot["right"]]
    else:
        raise ValueError("arm_mode must match the robot object")
    samples = [read_output_gripper_command_sample(arm) for arm in arms]
    if not any(sample is not None for sample in samples):
        return None, None
    targets = np.asarray([np.nan if sample is None else sample[0] for sample in samples], dtype=np.float32)
    timestamps = np.asarray([np.nan if sample is None else sample[1] for sample in samples], dtype=np.float64)
    return targets, timestamps


def read_output_delivery_state(piper: Any, qpos: np.ndarray | None = None) -> np.ndarray:
    if qpos is None:
        joints_message = piper.GetArmJointMsgs()
        gripper_message = piper.GetArmGripperMsgs()
        pose_message = piper.GetArmEndPoseMsgs()
        _require_fresh_feedback(
            {
                "joint": joints_message,
                "gripper": gripper_message,
                "end_pose": pose_message,
            }
        )
        qpos = _qpos_from_feedback(joints_message, gripper_message)
    else:
        qpos = np.asarray(qpos, dtype=np.float32)
        pose_message = piper.GetArmEndPoseMsgs()
        _require_fresh_feedback({"end_pose": pose_message})
    pose = pose_message.end_pose
    xyz_m = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / 1_000_000.0
    rpy_rad = np.deg2rad(
        np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64) / 1000.0
    )
    rotation = Rotation.from_euler("xyz", rpy_rad).as_matrix()
    return build_delivery_state(xyz_m, rotation, gripper_opening_m(float(qpos[6])))


def read_output_state(piper: Any) -> tuple[np.ndarray, np.ndarray]:
    """Backwards-compatible return of single-arm 10D delivery state + 7D qpos."""
    qpos = read_output_qpos(piper)
    return read_output_delivery_state(piper, qpos), qpos


def read_robot_state(
    robot: Any,
    *,
    schema: str,
    arm_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a schema-specific state and diagnostic joint vector."""
    if arm_mode == SINGLE_ARM:
        qpos = read_output_qpos(robot)
        state = qpos if schema == JOINT_SCHEMA else read_output_delivery_state(robot, qpos)
        return np.asarray(state, dtype=np.float32), qpos

    if not isinstance(robot, dict) or set(robot) != {"left", "right"}:
        raise ValueError("bimanual robot must be a {'left': arm, 'right': arm} mapping")
    qpos_by_side = {side: read_output_qpos(robot[side]) for side in ("left", "right")}
    qpos = np.concatenate((qpos_by_side["left"], qpos_by_side["right"]))
    if schema == JOINT_SCHEMA:
        state = qpos
    else:
        state = np.concatenate(
            tuple(
                read_output_delivery_state(robot[side], qpos_by_side[side])
                for side in ("left", "right")
            )
        )
    return np.asarray(state, dtype=np.float32), np.asarray(qpos, dtype=np.float32)


def send_output_qpos(piper: Any, qpos: np.ndarray) -> None:
    """Send one joint/gripper target to one output arm."""
    joints = [round(float(value) * RAD_FACTOR) for value in qpos[:6]]
    piper.JointCtrl(*joints)
    piper.GripperCtrl(round(gripper_opening_m(float(qpos[6])) * GRIPPER_FACTOR), 1000, 0x01, 0)


def reset_output_arm(
    piper: Any,
    duration_s: float = 4.0,
    hz: int = 20,
    speed_pct: int = 10,
) -> None:
    """Smoothly move one output arm to the all-zero joint pose."""
    current = read_output_qpos(piper)
    target = np.zeros(7, dtype=np.float32)
    piper.ModeCtrl(0x01, 0x01, speed_pct, 0x00)
    steps = max(1, round(duration_s * hz))
    for step in range(1, steps + 1):
        alpha = step / steps
        send_output_qpos(piper, current + alpha * (target - current))
        time.sleep(1.0 / hz)


def reset_robot_arms(
    robot: Any,
    *,
    arm_mode: str,
    duration_s: float = 4.0,
    hz: int = 20,
    speed_pct: int = 10,
) -> None:
    """Reset one arm or both arms concurrently to the all-zero joint pose."""
    if arm_mode == SINGLE_ARM:
        reset_output_arm(robot, duration_s=duration_s, hz=hz, speed_pct=speed_pct)
        return
    if arm_mode != BIMANUAL or not isinstance(robot, dict) or set(robot) != {"left", "right"}:
        raise ValueError("bimanual reset requires a {'left': arm, 'right': arm} mapping")

    failures: list[BaseException] = []

    def reset_side(side: str) -> None:
        try:
            reset_output_arm(
                robot[side],
                duration_s=duration_s,
                hz=hz,
                speed_pct=speed_pct,
            )
        except BaseException as exc:  # Propagate worker failures to the caller.
            failures.append(exc)

    workers = [threading.Thread(target=reset_side, args=(side,)) for side in ("left", "right")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        raise RuntimeError(f"failed to reset bimanual output arms: {failures[0]}") from failures[0]


def connect(can_name: str) -> Any:
    if C_PiperInterface_V2 is None:
        raise RuntimeError("piper_sdk is not installed; run this collector in the Piper hardware environment")
    require_can_interface_up(can_name)
    piper = C_PiperInterface_V2(can_name, judge_flag=False, can_auto_init=False)
    piper.CreateCanBus(
        can_name=can_name,
        bustype="socketcan",
        expected_bitrate=1_000_000,
        judge_flag=False,
    )
    piper.ConnectPort(can_init=True, piper_init=True)
    time.sleep(0.5)
    return piper


def next_episode_index(out_dir: pathlib.Path) -> int:
    """Return the next unused episode number in an existing output folder."""
    indices = []
    for path in out_dir.glob("ep_*.npz"):
        try:
            indices.append(int(path.stem.removeprefix("ep_")))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def verify_camera_streams(
    cameras: CameraCapture,
    expected_fps: int,
) -> dict[str, dict]:
    checks = cameras.verify()
    failures = []
    for key, info in checks.items():
        if not info["ok"]:
            failures.append(f"{key}: frame read failed")
            continue
        actual_fps = float(info["fps"])
        if (
            not np.isfinite(actual_fps)
            or abs(actual_fps - expected_fps) / expected_fps > 0.05
        ):
            failures.append(
                f"{key}: requested {expected_fps} FPS but negotiated {actual_fps:.3f} FPS"
            )
    if failures:
        raise RuntimeError("Camera verification failed: " + "; ".join(failures))
    return checks


def _connect_robot(args) -> tuple[Any, list[Any]]:
    connected: list[Any] = []
    try:
        if args.arm_mode == SINGLE_ARM:
            robot = connect(args.can)
            connected.append(robot)
            return robot, connected
        left = connect(args.left_can)
        connected.append(left)
        right = connect(args.right_can)
        connected.append(right)
        return {"left": left, "right": right}, connected
    except Exception:
        for piper in reversed(connected):
            piper.DisconnectPort()
        raise


def _camera_devices(args, contract: EpisodeContract) -> dict[str, str]:
    if contract.arm_mode == BIMANUAL:
        return {
            "cam_high": args.cam_high_device,
            "cam_left_wrist": args.cam_left_wrist_device,
            "cam_right_wrist": args.cam_right_wrist_device,
        }
    wrist_key = contract.camera_keys[1]
    return {"cam_high": args.cam_high_device, wrist_key: args.cam_wrist_device}


def run(args) -> None:
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.camera_fps <= 0:
        raise ValueError("camera-fps must be positive")
    if args.fps > args.camera_fps:
        raise ValueError("dataset fps cannot exceed camera-fps")

    contract = EpisodeContract(
        schema=args.schema,
        arm_mode=args.arm_mode,
        arm_side=args.arm_side,
        action_source=(
            JOINT_MEASURED_ACTION_SOURCE
            if args.schema == JOINT_SCHEMA
            else DELIVERY_MEASURED_ACTION_SOURCE
        ),
        action_alignment="next_observation",
    )
    can_summary = args.can if args.arm_mode == SINGLE_ARM else f"{args.left_can},{args.right_can}"
    print(
        f"Connecting {contract.robot_type} output arm(s) on {can_summary}; "
        f"schema={contract.schema} state={contract.state_dim} action={contract.action_dim} ..."
    )
    robot, connected = _connect_robot(args)
    cameras = CameraCapture(
        cam_ids=_camera_devices(args, contract),
        fps=args.camera_fps,
        image_hw=IMAGE_HW,
        capture_hw=CAMERA_SOURCE_HW,
        parallel_reads=True,
    )
    try:
        cameras.open()
        checks = verify_camera_streams(cameras, args.camera_fps)
    except Exception:
        cameras.close()
        for piper in reversed(connected):
            piper.DisconnectPort()
        raise
    for key, info in checks.items():
        print(
            f"  {key}: OK {info['shape']} @ {info['fps']:.1f} FPS, "
            f"read {info['latency_ms']} ms"
        )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = EpisodeBuffer(
        fps=args.fps,
        schema=contract.schema,
        arm_mode=contract.arm_mode,
        arm_side=contract.arm_side,
        camera_keys=contract.camera_keys,
        action_source=contract.action_source,
        action_alignment=contract.action_alignment,
        action_horizon=args.action_horizon,
    )
    keys = KeyListener()
    episode_index = next_episode_index(out_dir)
    if episode_index:
        print(f"Continuing from episode {episode_index:04d} in {out_dir}")
    dt = 1.0 / args.fps
    instruction = args.instruction or args.task_name.replace("_", " ")

    print("\n[COLLECT] SPACE=end episode, S=success, F=failure, D=discard, Q=quit\n")
    try:
        while not keys.quit:
            t0 = time.time()
            state, qpos = read_robot_state(
                robot,
                schema=contract.schema,
                arm_mode=contract.arm_mode,
            )
            state_timestamp = time.time()
            images, image_ts = cameras.read()
            if contract.schema == DELIVERY_SCHEMA:
                gripper_targets, gripper_command_timestamps = read_robot_gripper_command_samples(
                    robot, arm_mode=contract.arm_mode
                )
            else:
                gripper_targets, gripper_command_timestamps = None, None
            buffer.add(
                state,
                images,
                image_ts,
                qpos=qpos,
                gripper_targets=gripper_targets,
                gripper_command_timestamps=gripper_command_timestamps,
                state_timestamp=state_timestamp,
            )
            grippers = np.asarray([gripper_opening_m(value) for value in qpos[6::7]]) * 1000.0
            display = ",".join(f"{value:.1f}" for value in grippers)
            sys.stdout.write(
                f"\r[ep {episode_index:04d}] step {len(buffer):04d} "
                f"state_dim={len(state)} gripper_mm=[{display}]   "
            )
            sys.stdout.flush()

            if keys.end_episode:
                keys.end_episode = False
                print()
                choice = keys.wait_for("sfd", "S=save-success F=save-fail D=discard: ")
                if choice in ("s", "f") and len(buffer):
                    path = out_dir / f"ep_{episode_index:04d}.npz"
                    buffer.save(path, args.task_name, instruction, success=(choice == "s"))
                    episode_index += 1
                else:
                    print("Discarded.")
                buffer.start()

            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        if len(buffer):
            print("\nUnsaved episode remains in memory and was discarded.")
        cameras.close()
        for piper in reversed(connected):
            piper.DisconnectPort()
        print(f"\nDone. {episode_index} episodes saved to {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", choices=(DELIVERY_SCHEMA, JOINT_SCHEMA), default=DELIVERY_SCHEMA)
    ap.add_argument("--arm-mode", choices=(SINGLE_ARM, BIMANUAL), default=SINGLE_ARM)
    ap.add_argument("--arm-side", choices=("left", "right"), default="right")
    ap.add_argument("--can", default=DEFAULT_CAN, help="single-arm output CAN")
    ap.add_argument("--left-can", default=DEFAULT_LEFT_CAN, help="bimanual left output CAN")
    ap.add_argument("--right-can", default=DEFAULT_RIGHT_CAN, help="bimanual right output CAN")
    ap.add_argument("--cam-high-device", default=DEFAULT_HIGH_DEVICE)
    ap.add_argument("--cam-wrist-device", default=DEFAULT_WRIST_DEVICE)
    ap.add_argument("--cam-left-wrist-device", default=DEFAULT_LEFT_WRIST_DEVICE)
    ap.add_argument("--cam-right-wrist-device", default=DEFAULT_RIGHT_WRIST_DEVICE)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    ap.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS)
    ap.add_argument("--out-dir", default="episodes_piper_v21")
    ap.add_argument("--task-name", default="output_arm_task")
    ap.add_argument("--instruction", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

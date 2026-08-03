"""Collect output-arm feedback and two RGB camera streams.

This collector deliberately talks to only one Piper interface (can0).  It
records the execution/output arm's measured joint angles and gripper position;
it does not read master-arm data or control-command frames.

Each saved episode contains the Piper delivery schema:
  state: [EEF xyz, rotation 6D, gripper fraction], shape (T, 10)
  actions: base-frame delta action, shape (T, 7)
  image / wrist_image: RGB HWC frames, shape (T, 256, 256, 3)
  joint_qpos: optional diagnostic joint feedback, shape (T, 7)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from camera import CameraCapture
from piper_data_contract import (
    DEFAULT_FPS,
    IMAGE_HW,
    EpisodeBuffer,
    build_actions as _build_actions,
    build_delivery_state,
    gripper_closed_fraction,
)
from piper_sdk import C_PiperInterface_V2
from teleop import KeyListener


RAD_FACTOR = 57295.7795  # Piper unit: 0.001 degree -> rad
GRIPPER_FACTOR = 1_000_000.0  # Piper unit: 0.001 mm -> metre
DEFAULT_CAN = "can0"
DEFAULT_HIGH_DEVICE = "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:4:1.3-video-index0"
DEFAULT_WRIST_DEVICE = "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:5.2:1.0-video-index4"
DEFAULT_CAMERA_FPS = 30
CAMERA_SOURCE_HW = (240, 424)
PIPER_FEEDBACK_MAX_AGE_S = 0.5


class PiperFeedbackStaleError(RuntimeError):
    """Raised when Piper SDK getters only contain old cached CAN feedback."""


def _require_fresh_feedback(
    messages: dict[str, object],
    *,
    max_age_s: float = PIPER_FEEDBACK_MAX_AGE_S,
) -> None:
    """Fail when SDK feedback timestamps have stopped advancing.

    Piper SDK getters keep returning the last decoded values after CAN traffic
    stops. Their wrapper timestamps originate from SocketCAN receive frames,
    so an age check prevents frozen robot state from being recorded alongside
    live camera images.
    """
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


def _qpos_from_feedback(joints_message, gripper_message) -> np.ndarray:
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
    return np.append(values, float(gripper.grippers_angle) / GRIPPER_FACTOR)


def read_output_qpos(piper: C_PiperInterface_V2) -> np.ndarray:
    """Read measured output-arm joint feedback and gripper position only."""
    joints_message = piper.GetArmJointMsgs()
    gripper_message = piper.GetArmGripperMsgs()
    _require_fresh_feedback({"joint": joints_message, "gripper": gripper_message})
    return _qpos_from_feedback(joints_message, gripper_message)


def read_output_state(piper: C_PiperInterface_V2) -> tuple[np.ndarray, np.ndarray]:
    """Return delivery state (10D) and diagnostic joint qpos (7D)."""
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
    pose = pose_message.end_pose
    xyz_m = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / 1_000_000.0
    rpy_rad = np.deg2rad(np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64) / 1000.0)
    rotation = Rotation.from_euler("xyz", rpy_rad).as_matrix()
    state = build_delivery_state(xyz_m, rotation, float(qpos[6]))
    return state, qpos.astype(np.float32)


def send_output_qpos(piper: C_PiperInterface_V2, qpos: np.ndarray):
    """Send one joint/gripper target to the output arm."""
    joints = [round(float(value) * RAD_FACTOR) for value in qpos[:6]]
    piper.JointCtrl(*joints)
    piper.GripperCtrl(round(abs(float(qpos[6])) * GRIPPER_FACTOR), 1000, 0x01, 0)


def reset_output_arm(
    piper: C_PiperInterface_V2,
    duration_s: float = 4.0,
    hz: int = 20,
    speed_pct: int = 10,
):
    """Smoothly move the output arm to the all-zero joint pose."""
    current = read_output_qpos(piper)
    target = np.zeros(7, dtype=np.float32)
    piper.ModeCtrl(0x01, 0x01, speed_pct, 0x00)
    steps = max(1, round(duration_s * hz))
    for step in range(1, steps + 1):
        alpha = step / steps
        send_output_qpos(piper, current + alpha * (target - current))
        time.sleep(1.0 / hz)


def connect(can_name: str) -> C_PiperInterface_V2:
    piper = C_PiperInterface_V2(can_name, judge_flag=False, can_auto_init=False)
    # With can_auto_init=False, Piper SDK requires explicit CAN-bus creation
    # before ConnectPort(). The Linux SocketCAN interface is already brought
    # up by can_activate.sh; CreateCanBus only binds the SDK to that interface.
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
                f"{key}: requested {expected_fps} FPS but negotiated "
                f"{actual_fps:.3f} FPS"
            )
    if failures:
        raise RuntimeError("Camera verification failed: " + "; ".join(failures))
    return checks


def run(args):
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.camera_fps <= 0:
        raise ValueError("camera-fps must be positive")
    if args.fps > args.camera_fps:
        raise ValueError("dataset fps cannot exceed camera-fps")
    print(f"Connecting to output arm on {args.can} ...")
    piper = connect(args.can)
    cameras = CameraCapture(
        cam_ids={
            "cam_high": args.cam_high_device,
            "cam_wrist": args.cam_wrist_device,
        },
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
        piper.DisconnectPort()
        raise
    for key, info in checks.items():
        print(
            f"  {key}: OK {info['shape']} @ {info['fps']:.1f} FPS, "
            f"read {info['latency_ms']} ms"
        )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = EpisodeBuffer(fps=args.fps)
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
            state, qpos = read_output_state(piper)
            state_timestamp = time.time()
            images, image_ts = cameras.read()
            buffer.add(
                state,
                images,
                image_ts,
                qpos=qpos,
                state_timestamp=state_timestamp,
            )
            sys.stdout.write(
                f"\r[ep {episode_index:04d}] step {len(buffer):04d} "
                f"eef=({state[:3].round(3)}) g={qpos[6] * 1000:.1f}mm   "
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
        piper.DisconnectPort()
        print(f"\nDone. {episode_index} episodes saved to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can", default=DEFAULT_CAN)
    ap.add_argument("--cam-high-device", default=DEFAULT_HIGH_DEVICE)
    ap.add_argument("--cam-wrist-device", default=DEFAULT_WRIST_DEVICE)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS)
    ap.add_argument("--out-dir", default="episodes_piper_v21")
    ap.add_argument("--task-name", default="output_arm_task")
    ap.add_argument("--instruction", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

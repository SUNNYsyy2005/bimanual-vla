"""Collect output-arm feedback and two RGB camera streams.

This collector deliberately talks to only one Piper interface (can0).  It
records the execution/output arm's measured joint angles and gripper position;
it does not read master-arm data or control-command frames.

Each saved episode contains:
  qpos: [j1..j6, gripper] in rad/metres, shape (T, 7)
  timestamps: host-clock timestamps, shape (T,)
  images_cam_high: third-person RGB frames, shape (T, 3, 224, 224)
  images_cam_wrist: wrist RGB frames, shape (T, 3, 224, 224)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

from camera import CameraCapture
from piper_sdk import C_PiperInterface_V2
from teleop import KeyListener


RAD_FACTOR = 57295.7795  # Piper unit: 0.001 degree -> rad
GRIPPER_FACTOR = 1_000_000.0  # Piper unit: 0.001 mm -> metre
DEFAULT_CAN = "can0"
DEFAULT_HIGH_DEVICE = "/dev/video12"
DEFAULT_WRIST_DEVICE = "/dev/video4"


def read_output_qpos(piper: C_PiperInterface_V2) -> np.ndarray:
    """Read measured output-arm joint feedback and gripper position only."""
    joints = piper.GetArmJointMsgs().joint_state
    gripper = piper.GetArmGripperMsgs().gripper_state
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


class EpisodeBuffer:
    def __init__(self):
        self.start()

    def start(self):
        self.qpos: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.images: dict[str, list[np.ndarray]] = {}
        self.image_timestamps: dict[str, list[float]] = {}

    def add(self, qpos: np.ndarray, images: dict[str, np.ndarray], image_ts: dict[str, float]):
        now = time.time()
        self.qpos.append(np.asarray(qpos, dtype=np.float32).copy())
        self.timestamps.append(now)
        for key, image in images.items():
            self.images.setdefault(key, []).append(np.asarray(image, dtype=np.uint8).copy())
            self.image_timestamps.setdefault(key, []).append(float(image_ts.get(key, now)))

    def __len__(self):
        return len(self.qpos)

    def save(self, path: pathlib.Path, task_name: str, instruction: str, success: bool):
        payload: dict[str, np.ndarray] = {
            "qpos": np.asarray(self.qpos, dtype=np.float32),
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
            "task_name": np.asarray(task_name),
            "instruction": np.asarray(instruction),
            "success": np.asarray(bool(success), dtype=np.bool_),
        }
        for key, frames in self.images.items():
            payload[f"images_{key}"] = np.asarray(frames, dtype=np.uint8)
        for key, timestamps in self.image_timestamps.items():
            payload[f"image_timestamps_{key}"] = np.asarray(timestamps, dtype=np.float64)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        print(f"Saved {len(self)} steps -> {path}")


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


def run(args):
    print(f"Connecting to output arm on {args.can} ...")
    piper = connect(args.can)
    cameras = CameraCapture(
        cam_ids={
            "cam_high": args.cam_high_device,
            "cam_wrist": args.cam_wrist_device,
        },
        fps=args.fps,
    )
    cameras.open()
    for key, info in cameras.verify().items():
        print(f"  {key}: {'OK' if info['ok'] else 'FAIL'} {info['shape']} {info['latency_ms']} ms")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = EpisodeBuffer()
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
            qpos = read_output_qpos(piper)
            images, image_ts = cameras.read()
            buffer.add(qpos, images, image_ts)
            sys.stdout.write(
                f"\r[ep {episode_index:04d}] step {len(buffer):04d} "
                f"q=({qpos[:3].round(2)}) g={qpos[6] * 1000:.1f}mm   "
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
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out-dir", default="episodes_output_arm")
    ap.add_argument("--task-name", default="output_arm_task")
    ap.add_argument("--instruction", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

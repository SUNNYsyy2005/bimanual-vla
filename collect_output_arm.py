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
from piper_sdk import C_PiperInterface_V2
from teleop import KeyListener


RAD_FACTOR = 57295.7795  # Piper unit: 0.001 degree -> rad
GRIPPER_FACTOR = 1_000_000.0  # Piper unit: 0.001 mm -> metre
DEFAULT_CAN = "can0"
DEFAULT_HIGH_DEVICE = "/dev/video8"
DEFAULT_WRIST_DEVICE = "/dev/video16"
DEFAULT_FPS = 20
IMAGE_HW = (256, 256)
GRIPPER_MAX_M = 0.07


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


def gripper_closed_fraction(gripper_m: float) -> float:
    """Map Piper gripper position to 0=open, 1=closed."""
    return float(np.clip(1.0 - gripper_m / GRIPPER_MAX_M, 0.0, 1.0))


def read_output_state(piper: C_PiperInterface_V2) -> tuple[np.ndarray, np.ndarray]:
    """Return delivery state (10D) and diagnostic joint qpos (7D)."""
    qpos = read_output_qpos(piper)
    pose = piper.GetArmEndPoseMsgs().end_pose
    xyz_m = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / 1_000_000.0
    rpy_rad = np.deg2rad(np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64) / 1000.0)
    rotation = Rotation.from_euler("xyz", rpy_rad).as_matrix()
    rotation6d = rotation[:, :2].T.reshape(-1)
    state = np.concatenate(([xyz_m[0], xyz_m[1], xyz_m[2]], rotation6d, [gripper_closed_fraction(float(qpos[6]))]))
    return state.astype(np.float32), qpos.astype(np.float32)


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
    def __init__(self, fps: int = DEFAULT_FPS):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        self.start()

    def start(self):
        self.states: list[np.ndarray] = []
        self.joint_qpos: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.images: dict[str, list[np.ndarray]] = {}
        self.image_timestamps: dict[str, list[float]] = {}

    def add(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        image_ts: dict[str, float],
        qpos: np.ndarray | None = None,
        state_timestamp: float | None = None,
    ):
        now = time.time() if state_timestamp is None else float(state_timestamp)
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        if qpos is not None:
            self.joint_qpos.append(np.asarray(qpos, dtype=np.float32).copy())
        self.timestamps.append(now)
        for key, image in images.items():
            # CameraCapture returns CHW; delivery format is RGB HWC.
            frame = np.asarray(image, dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
            self.images.setdefault(key, []).append(frame.copy())
            self.image_timestamps.setdefault(key, []).append(float(image_ts.get(key, now)))

    def __len__(self):
        return len(self.states)

    def save(self, path: pathlib.Path, task_name: str, instruction: str, success: bool):
        if not self.states:
            raise ValueError("cannot save an empty episode")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction must not be empty")
        states_real = np.asarray(self.states, dtype=np.float32)
        # Add one terminal observation. Its action becomes the required no-op.
        states = np.concatenate((states_real, states_real[-1:]), axis=0)
        actions = _build_actions(states)
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        terminal_dt = 1.0 / self.fps
        timestamps = np.concatenate((timestamps, [timestamps[-1] + terminal_dt]))
        payload: dict[str, np.ndarray] = {
            "state": states,
            "actions": actions,
            "timestamps": timestamps,
            "task": np.asarray(task_name),
            "instruction": np.asarray(instruction),
            "success": np.asarray(bool(success), dtype=np.bool_),
        }
        if self.joint_qpos:
            qpos = np.asarray(self.joint_qpos, dtype=np.float32)
            payload["joint_qpos"] = np.concatenate((qpos, qpos[-1:]), axis=0)
        high = np.asarray(self.images["cam_high"], dtype=np.uint8)
        wrist = np.asarray(self.images["cam_wrist"], dtype=np.uint8)
        payload["image"] = np.concatenate((high, high[-1:]), axis=0)
        payload["wrist_image"] = np.concatenate((wrist, wrist[-1:]), axis=0)
        for key, timestamps in self.image_timestamps.items():
            image_ts = np.asarray(timestamps, dtype=np.float64)
            payload[f"image_timestamps_{key}"] = np.concatenate((image_ts, [image_ts[-1]]))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        print(f"Saved {len(self)} steps -> {path}")


def _rotation_from_state(state: np.ndarray) -> np.ndarray:
    c0 = np.asarray(state[3:6], dtype=np.float64)
    c1 = np.asarray(state[6:9], dtype=np.float64)
    c0 /= max(np.linalg.norm(c0), 1e-12)
    c1 = c1 - c0 * np.dot(c0, c1)
    c1 /= max(np.linalg.norm(c1), 1e-12)
    c2 = np.cross(c0, c1)
    return np.column_stack((c0, c1, c2))


def _build_actions(states: np.ndarray) -> np.ndarray:
    """Build 7D base-frame delta actions plus a terminal no-op."""
    count = len(states)
    actions = np.zeros((count, 7), dtype=np.float32)
    for i in range(max(0, count - 1)):
        r_t = _rotation_from_state(states[i])
        r_next = _rotation_from_state(states[i + 1])
        delta_rotvec = Rotation.from_matrix(r_next @ r_t.T).as_rotvec()
        actions[i, :3] = states[i + 1, :3] - states[i, :3]
        actions[i, 3:6] = delta_rotvec.astype(np.float32)
        actions[i, 6] = states[i + 1, 9]
    if count:
        actions[-1, 6] = states[-1, 9]
    return actions


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
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    print(f"Connecting to output arm on {args.can} ...")
    piper = connect(args.can)
    cameras = CameraCapture(
        cam_ids={
            "cam_high": args.cam_high_device,
            "cam_wrist": args.cam_wrist_device,
        },
        fps=args.fps,
        image_hw=IMAGE_HW,
        parallel_reads=True,
    )
    cameras.open()
    for key, info in cameras.verify().items():
        print(f"  {key}: {'OK' if info['ok'] else 'FAIL'} {info['shape']} {info['latency_ms']} ms")

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
    ap.add_argument("--out-dir", default="episodes_piper_v21")
    ap.add_argument("--task-name", default="output_arm_task")
    ap.add_argument("--instruction", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

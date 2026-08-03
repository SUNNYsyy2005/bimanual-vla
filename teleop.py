"""Bimanual Piper master/slave teleoperation and v3 raw-data collection."""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import termios
    import tty
    _UNIX = True
except ImportError:  # pragma: no cover
    import msvcrt
    _UNIX = False

from camera import CameraCapture
from piper_data_contract import (
    BIMANUAL,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_FPS,
    DELIVERY_MEASURED_ACTION_SOURCE,
    DELIVERY_MIXED_ACTION_SOURCE,
    DELIVERY_SCHEMA,
    JOINT_MAPPED_ACTION_SOURCE,
    JOINT_SCHEMA,
    EpisodeContract,
    MASTER_GRIPPER_FEEDBACK_ACTION_SOURCE,
    build_delivery_actions_with_gripper_targets,
    build_delivery_state,
    gripper_opening_fraction,
    gripper_opening_m,
    next_observation_timestamps,
)
try:
    from piper_sdk import C_PiperInterface_V2
except ModuleNotFoundError:  # Allow contract tooling/tests off the robot host.
    C_PiperInterface_V2 = None  # type: ignore[assignment]
from trajectory import TrajectoryRecorder

DEFAULT_LEFT_MASTER = "can0"
DEFAULT_LEFT_SLAVE = "can1"
DEFAULT_RIGHT_MASTER = "can2"
DEFAULT_RIGHT_SLAVE = "can3"

# v3 collection default. CLI --fps may override it and metadata follows it.
RECORD_HZ = DEFAULT_FPS
RESET_SPEED_PCT = 20
RESET_HZ = 20
RESET_DURATION_S = 4.0
COUNTDOWN_S = 3
START_POSE_FILE = "start_pose.npy"

_RAD_FACTOR = 57295.7795
_M_FACTOR = 1_000_000.0


def _read_7d(arm: C_PiperInterface_V2) -> np.ndarray:
    """Read six measured joint radians plus v3 opening fraction."""
    joints_message = arm.GetArmJointMsgs().joint_state
    gripper_message = arm.GetArmGripperMsgs().gripper_state
    joints = np.array(
        [
            joints_message.joint_1,
            joints_message.joint_2,
            joints_message.joint_3,
            joints_message.joint_4,
            joints_message.joint_5,
            joints_message.joint_6,
        ],
        dtype=np.float64,
    ) / _RAD_FACTOR
    opening_m = abs(float(gripper_message.grippers_angle)) / _M_FACTOR
    return np.append(joints, gripper_opening_fraction(opening_m)).astype(np.float32)


def _read_eef_10d(arm: C_PiperInterface_V2) -> tuple[np.ndarray, np.ndarray]:
    """Read measured absolute EEF state and diagnostic 7D joint state."""
    qpos = _read_7d(arm)
    pose = arm.GetArmEndPoseMsgs().end_pose
    xyz_m = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / 1_000_000.0
    rpy_rad = np.deg2rad(
        np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64) / 1000.0
    )
    rotation = Rotation.from_euler("xyz", rpy_rad).as_matrix()
    return build_delivery_state(xyz_m, rotation, gripper_opening_m(qpos[6])), qpos


def _send_7d(arm: C_PiperInterface_V2, target: np.ndarray):
    """Send v3 absolute joint target with opening fraction gripper."""
    values = np.asarray(target, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError(f"joint target must have shape (7,), got {values.shape}")
    joints = [round(float(value) * _RAD_FACTOR) for value in values[:6]]
    arm.JointCtrl(*joints)
    opening_m = gripper_opening_m(values[6])
    arm.GripperCtrl(round(opening_m * _M_FACTOR), 1000, 0x01, 0)


class KeyListener:
    def __init__(self):
        self.end_episode = False
        self.estop = False
        self.last_key = None
        self.quit = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        if _UNIX:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self.quit:
                    ch = sys.stdin.read(1)
                    self.last_key = ch.lower()
                    if ch == " ":
                        self.end_episode = True
                    elif ch.lower() == "e":
                        self.estop = True
                    elif ch.lower() == "q":
                        self.quit = True
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        else:  # pragma: no cover
            while not self.quit:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    self.last_key = ch.lower()
                    if ch == b" ":
                        self.end_episode = True
                    elif ch.lower() == b"e":
                        self.estop = True
                    elif ch.lower() == b"q":
                        self.quit = True
                time.sleep(0.02)

    def wait_for(self, options: str, prompt: str) -> str:
        self.last_key = None
        sys.stdout.write(prompt)
        sys.stdout.flush()
        while not self.quit:
            key = self.last_key
            if key and key in options:
                self.last_key = None
                return key
            time.sleep(0.02)
        return "q"


def _reset_one_arm(arm: C_PiperInterface_V2, current: np.ndarray, target: np.ndarray):
    steps = max(1, int(RESET_DURATION_S * RESET_HZ))
    arm.ModeCtrl(0x01, 0x01, RESET_SPEED_PCT, 0x00)
    time.sleep(0.05)
    for index in range(1, steps + 1):
        alpha = index / steps
        _send_7d(arm, current + alpha * (target - current))
        time.sleep(1.0 / RESET_HZ)


def _countdown(seconds: int, keys: KeyListener):
    for value in range(seconds, 0, -1):
        if keys.quit or keys.estop:
            return
        sys.stdout.write(f"\r[RESET] Starting in {value}s ...  ")
        sys.stdout.flush()
        time.sleep(1.0)
    print("\r[RECORD] GO                   ")


def estop_all(lm, ls, rm, rs):
    for arm in (lm, ls, rm, rs):
        arm.EmergencyStop(0x01)
    print("\n\033[91m[E-STOP] ALL ARMS STOPPED. Press R to recover.\033[0m")


def recover_all(lm, ls, rm, rs):
    for arm in (lm, ls, rm, rs):
        arm.EmergencyStop(0x02)
    time.sleep(0.3)
    for arm in (lm, ls, rm, rs):
        arm.EnablePiper()
    time.sleep(0.3)
    setup_master_slave(lm, ls, rm, rs)


def handle_estop(lm, ls, rm, rs, start_14d, recorder, keys):
    keys.estop = False
    estop_all(lm, ls, rm, rs)
    recorder.start()
    keys.wait_for("r", "  Press R to recover: ")
    print()
    if keys.quit:
        return
    recover_all(lm, ls, rm, rs)
    auto_reset_all(lm, ls, rm, rs, start_14d)


def auto_reset_all(lm, ls, rm, rs, start_14d: np.ndarray):
    target_left, target_right = start_14d[:7], start_14d[7:14]
    current = (_read_7d(lm), _read_7d(ls), _read_7d(rm), _read_7d(rs))
    targets = (target_left, target_left, target_right, target_right)
    arms = (lm, ls, rm, rs)
    sys.stdout.write("\n[RESET] All 4 arms → start pose...")
    sys.stdout.flush()
    workers = [threading.Thread(target=_reset_one_arm, args=(arm, now, target)) for arm, now, target in zip(arms, current, targets)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    setup_master_slave(lm, ls, rm, rs)
    time.sleep(0.3)
    print(" done.")


def connect_arm(can_name: str, role: str) -> C_PiperInterface_V2:
    if C_PiperInterface_V2 is None:
        raise RuntimeError("piper_sdk is not installed; run teleop on the Piper host")
    arm = C_PiperInterface_V2(can_name, judge_flag=False, can_auto_init=False)
    arm.ConnectPort(can_init=True, piper_init=True)
    time.sleep(0.3)
    print(f"  {role} ({can_name}) connected.")
    return arm


def setup_master_slave(lm, ls, rm, rs):
    lm.MasterSlaveConfig(0xFA, 0, 0, 0)
    rm.MasterSlaveConfig(0xFA, 0, 0, 0)
    ls.MasterSlaveConfig(0xFC, 0, 0, 0)
    rs.MasterSlaveConfig(0xFC, 0, 0, 0)
    print("Master-slave configured.")


def teardown_master_slave(lm, rm):
    for arm in (lm, rm):
        arm.MasterSlaveConfig(0x00, 0, 0, 0)
    print("Master-slave disabled. Slave arms need reboot to resume direct CAN control.")


def _default_instruction(task_name: str) -> str:
    return ((task_name or "teleop_task").strip()).replace("_", " ")


def _episode_contract(args) -> EpisodeContract:
    delivery = args.schema == DELIVERY_SCHEMA
    return EpisodeContract(
        schema=args.schema,
        arm_mode=BIMANUAL,
        arm_side="both",
        camera_keys=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        action_source=(DELIVERY_MIXED_ACTION_SOURCE if delivery else JOINT_MAPPED_ACTION_SOURCE),
        action_alignment=("next_observation_pose_same_step_gripper" if delivery else "same_step_command"),
        fps=args.fps,
        action_horizon=args.action_horizon,
        coordinate_frame="slave_base",
        source_frame=("slave_base_pose_plus_master_gripper_feedback" if delivery else "master_joint_identity_mapping"),
    )


def _prepare_delivery_episode(episode: dict[str, np.ndarray], contract: EpisodeContract) -> dict[str, np.ndarray]:
    """Use next slave EEF pose and same-step master gripper without EEF calibration."""
    if contract.schema != DELIVERY_SCHEMA:
        return dict(episode)
    prepared = dict(episode)
    provisional = np.asarray(prepared["actions"], dtype=np.float32)
    gripper_targets = provisional[:, list(contract.gripper_action_indices)]
    master_timestamps = np.asarray(prepared["action_timestamp"], dtype=np.float64)
    prepared["actions"] = build_delivery_actions_with_gripper_targets(
        prepared["qpos"], gripper_targets, arm_count=contract.arm_count
    )
    prepared["action_timestamp"] = next_observation_timestamps(
        prepared["state_timestamp"], fps=contract.fps
    )
    prepared["gripper_command_target"] = gripper_targets
    prepared["gripper_command_timestamp"] = np.repeat(
        master_timestamps[:, None], contract.arm_count, axis=1
    )
    prepared["gripper_command_present"] = np.ones(
        gripper_targets.shape, dtype=np.bool_
    )
    return prepared


def _maybe_make_pi0_writer(args, contract: EpisodeContract):
    if not args.record or args.no_pi0_export:
        return None
    from pi0_dataset import Pi0LeRobotDatasetWriter
    return Pi0LeRobotDatasetWriter(
        args.dataset_root,
        fps=args.fps,
        robot_type=args.robot_type,
        state_names=list(contract.state_names),
        action_names=list(contract.action_names),
        camera_keys=list(contract.camera_keys),
        image_hw=(224, 224),
        schema=contract.schema,
        arm_mode=contract.arm_mode,
        arm_side=contract.arm_side,
        action_semantics=contract.action_semantics,
        action_source=contract.action_source,
        action_alignment=contract.action_alignment,
    )


def _save_episode(recorder, args, contract, ep_dir, ep_idx, pi0_writer=None, success=True):
    if len(recorder) == 0:
        print("  (empty episode, skipping)")
        return ep_idx
    instruction = args.instruction or _default_instruction(args.task_name)
    episode = _prepare_delivery_episode(recorder.to_numpy_dict(), contract)
    extras = {
        "state": episode["qpos"],
        "task_name": args.task_name,
        "instruction": instruction,
        "success": np.asarray(bool(success), dtype=np.bool_),
        **contract.metadata_payload(),
        "terminal_padding": np.asarray(False, dtype=np.bool_),
    }
    overrides = None
    if contract.schema == DELIVERY_SCHEMA:
        extras.update(
            {
                "pose_action_source": np.asarray(DELIVERY_MEASURED_ACTION_SOURCE),
                "pose_action_alignment": np.asarray("next_observation"),
                "gripper_action_source": np.asarray(MASTER_GRIPPER_FEEDBACK_ACTION_SOURCE),
                "gripper_action_alignment": np.asarray("same_step_command"),
            }
        )
        overrides = {
            key: episode[key]
            for key in (
                "actions",
                "action_timestamp",
                "gripper_command_target",
                "gripper_command_timestamp",
                "gripper_command_present",
            )
        }
    raw_path = ep_dir / f"ep_{ep_idx:04d}.npz"
    recorder.save(raw_path, extras=extras, overrides=overrides)
    if pi0_writer is not None:
        images = {key.removeprefix("images_"): value for key, value in episode.items() if key.startswith("images_")}
        pi0_writer.append_episode(
            states=episode["qpos"],
            actions=episode["actions"],
            timestamps=episode["state_timestamp"],
            images=images,
            task_name=args.task_name,
            instruction=instruction,
            success=success,
            metadata={"source_raw_episode": raw_path.name, "contract_version": contract.version},
        )
        print(f"  pi0 dataset updated at {args.dataset_root}")
    recorder.start()
    return ep_idx + 1


def run(args):
    if args.fps <= 0 or args.action_horizon <= 0:
        raise ValueError("--fps and --action-horizon must be positive")
    contract = _episode_contract(args)
    print("Connecting arms...")
    lm = connect_arm(args.left_master, "left-master")
    ls = connect_arm(args.left_slave, "left-slave")
    rm = connect_arm(args.right_master, "right-master")
    rs = connect_arm(args.right_slave, "right-slave")
    setup_master_slave(lm, ls, rm, rs)
    time.sleep(0.5)

    pose_file = pathlib.Path(args.start_pose)
    if args.capture_start:
        start_14d = np.concatenate([_read_7d(ls), _read_7d(rs)])
        np.save(str(pose_file), start_14d)
    elif pose_file.exists():
        start_14d = np.asarray(np.load(str(pose_file)), dtype=np.float32)
        if start_14d.shape != (14,):
            raise ValueError(f"start pose must be 14D, got {start_14d.shape}")
    else:
        start_14d = np.zeros(14, dtype=np.float32)

    cameras = None
    if args.record:
        cameras = CameraCapture(
            cam_ids={"cam_high": args.cam_high_id, "cam_left_wrist": args.cam_left_wrist_id, "cam_right_wrist": args.cam_right_wrist_id},
            fps=args.fps,
        )
        cameras.open()
        for key, info in cameras.verify().items():
            print(f"  {key}: {'OK' if info['ok'] else 'FAIL'}  {info['latency_ms']} ms")

    recorder = TrajectoryRecorder()
    recorder.start()
    ep_dir = pathlib.Path(args.out_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_idx = 0
    keys = KeyListener()
    dt = 1.0 / args.fps
    pi0_writer = _maybe_make_pi0_writer(args, contract)

    auto_reset_all(lm, ls, rm, rs, start_14d)
    print(f"\n{'[RECORD]' if args.record else '[DRY RUN]'} schema={args.schema} fps={args.fps} SPACE=end E=e-stop q=quit\n")
    try:
        while not keys.quit:
            started = time.time()
            if keys.estop:
                handle_estop(lm, ls, rm, rs, start_14d, recorder, keys)
                if not keys.quit:
                    _countdown(COUNTDOWN_S, keys)
                continue

            if args.schema == JOINT_SCHEMA:
                left_state, right_state = _read_7d(ls), _read_7d(rs)
                state_timestamp = time.time()
                left_action, right_action = _read_7d(lm), _read_7d(rm)
                action_timestamp = time.time()
                state = np.concatenate((left_state, right_state))
                action = np.concatenate((left_action, right_action))
                joint_qpos = state.copy()
            else:
                left_state, left_qpos = _read_eef_10d(ls)
                right_state, right_qpos = _read_eef_10d(rs)
                state_timestamp = time.time()
                left_master_qpos = _read_7d(lm)
                right_master_qpos = _read_7d(rm)
                action_timestamp = time.time()
                state = np.concatenate((left_state, right_state))
                # Pose is derived from the next measured slave observation at
                # save time. These provisional rows carry only the same-step
                # master gripper targets in the canonical 10D slots.
                action = state.copy()
                action[9] = left_master_qpos[6]
                action[19] = right_master_qpos[6]
                joint_qpos = np.concatenate((left_qpos, right_qpos))

            if cameras is not None:
                images, image_ts = cameras.read()
                recorder.add(
                    state,
                    action,
                    images,
                    image_ts,
                    state_timestamp=state_timestamp,
                    action_timestamp=action_timestamp,
                    joint_qpos=joint_qpos,
                )
            if args.record:
                grippers = joint_qpos[6::7]
                sys.stdout.write(
                    f"\r[ep {ep_idx:03d}] step {len(recorder):04d} "
                    f"L_g={grippers[0]:.2f} R_g={grippers[1]:.2f} dt={int((time.time()-started)*1000)}ms   "
                )
                sys.stdout.flush()

            if keys.end_episode:
                keys.end_episode = False
                print()
                choice = keys.wait_for("sfd", f"  {len(recorder)} steps — S=save-success F=save-fail D=discard: ")
                print()
                if choice in {"s", "f"}:
                    ep_idx = _save_episode(recorder, args, contract, ep_dir, ep_idx, pi0_writer, success=choice == "s")
                else:
                    recorder.start()
                if not keys.quit and not keys.estop:
                    auto_reset_all(lm, ls, rm, rs, start_14d)
                    _countdown(COUNTDOWN_S, keys)
            sleep = dt - (time.time() - started)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        print("\nShutting down...")
        if args.record and len(recorder) > 0:
            ep_idx = _save_episode(recorder, args, contract, ep_dir, ep_idx, pi0_writer, success=True)
        if cameras:
            cameras.close()
        teardown_master_slave(lm, rm)
        for arm in (lm, ls, rm, rs):
            arm.DisconnectPort()
        print(f"Done. {ep_idx} episodes saved to {ep_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-master", default=DEFAULT_LEFT_MASTER)
    parser.add_argument("--left-slave", default=DEFAULT_LEFT_SLAVE)
    parser.add_argument("--right-master", default=DEFAULT_RIGHT_MASTER)
    parser.add_argument("--right-slave", default=DEFAULT_RIGHT_SLAVE)
    parser.add_argument("--schema", choices=(JOINT_SCHEMA, DELIVERY_SCHEMA), default=JOINT_SCHEMA)
    parser.add_argument("--eef-calibration", default=None, help="deprecated compatibility option; delivery pose now uses next measured slave EEF")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--capture-start", action="store_true")
    parser.add_argument("--start-pose", default=START_POSE_FILE)
    parser.add_argument("--out-dir", default="episodes")
    parser.add_argument("--task-name", default="teleop_task")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--dataset-root", default="pi0_dataset_bimanual")
    parser.add_argument("--robot-type", default="piper_bimanual")
    parser.add_argument("--no-pi0-export", action="store_true")
    parser.add_argument("--cam-high-id", type=int, default=0)
    parser.add_argument("--cam-left-wrist-id", dest="cam_left_wrist_id", type=int, default=2)
    parser.add_argument("--cam-right-wrist-id", dest="cam_right_wrist_id", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

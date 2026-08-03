"""Single-arm Piper master/slave teleoperation and v3 raw-data collection."""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

import numpy as np

from camera import CameraCapture
from piper_data_contract import (
    DEFAULT_ACTION_HORIZON,
    DEFAULT_FPS,
    DELIVERY_MEASURED_ACTION_SOURCE,
    DELIVERY_MIXED_ACTION_SOURCE,
    DELIVERY_SCHEMA,
    JOINT_MAPPED_ACTION_SOURCE,
    JOINT_SCHEMA,
    SINGLE_ARM,
    EpisodeContract,
    MASTER_GRIPPER_FEEDBACK_ACTION_SOURCE,
)
from teleop import (
    KeyListener,
    _countdown,
    _prepare_delivery_episode,
    _read_7d,
    _read_eef_10d,
    _reset_one_arm,
    connect_arm,
)
from trajectory import TrajectoryRecorder

DEFAULT_MASTER = "can0"
DEFAULT_SLAVE = "can1"
RECORD_HZ = DEFAULT_FPS
COUNTDOWN_S = 3
START_POSE_FILE = "start_pose_single.npy"


def setup_master_slave(master, slave):
    master.MasterSlaveConfig(0xFA, 0, 0, 0)
    slave.MasterSlaveConfig(0xFC, 0, 0, 0)
    print("Master-slave configured.")


def teardown_master_slave(master):
    master.MasterSlaveConfig(0x00, 0, 0, 0)
    print("Master-slave disabled. Slave arm may need reboot to resume direct CAN control.")


def auto_reset_pair(master, slave, start_7d: np.ndarray):
    current_master = _read_7d(master)
    current_slave = _read_7d(slave)
    sys.stdout.write("\n[RESET] Master + slave → start pose...")
    sys.stdout.flush()
    workers = [
        threading.Thread(target=_reset_one_arm, args=(master, current_master, start_7d)),
        threading.Thread(target=_reset_one_arm, args=(slave, current_slave, start_7d)),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    setup_master_slave(master, slave)
    time.sleep(0.3)
    print(" done.")


def estop_pair(master, slave):
    for arm in (master, slave):
        arm.EmergencyStop(0x01)
    print("\n\033[91m[E-STOP] SINGLE ARM STOPPED. Press R to recover.\033[0m")


def recover_pair(master, slave):
    for arm in (master, slave):
        arm.EmergencyStop(0x02)
    time.sleep(0.3)
    for arm in (master, slave):
        arm.EnablePiper()
    time.sleep(0.3)
    setup_master_slave(master, slave)


def _default_instruction(task_name: str) -> str:
    return (task_name or "single_arm_task").replace("_", " ")


def _episode_contract(args) -> EpisodeContract:
    delivery = args.schema == DELIVERY_SCHEMA
    return EpisodeContract(
        schema=args.schema,
        arm_mode=SINGLE_ARM,
        arm_side=args.arm_side,
        camera_keys=("cam_high", f"cam_{args.arm_side}_wrist"),
        action_source=(DELIVERY_MIXED_ACTION_SOURCE if delivery else JOINT_MAPPED_ACTION_SOURCE),
        action_alignment=("next_observation_pose_same_step_gripper" if delivery else "same_step_command"),
        fps=args.fps,
        action_horizon=args.action_horizon,
        coordinate_frame="slave_base",
        source_frame=("slave_base_pose_plus_master_gripper_feedback" if delivery else "master_joint_identity_mapping"),
    )


def _maybe_make_pi0_writer(args, contract: EpisodeContract):
    if not args.record or args.no_pi0_export:
        return None
    from pi0_dataset import Pi0LeRobotDatasetWriter
    return Pi0LeRobotDatasetWriter(
        args.dataset_root,
        fps=args.fps,
        robot_type=contract.robot_type,
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
            metadata={"source_raw_episode": raw_path.name, "arm_side": args.arm_side, "contract_version": contract.version},
        )
        print(f"  pi0 dataset updated at {args.dataset_root}")
    recorder.start()
    return ep_idx + 1


def run(args):
    if args.fps <= 0 or args.action_horizon <= 0:
        raise ValueError("--fps and --action-horizon must be positive")
    contract = _episode_contract(args)

    print("Connecting arms...")
    master = connect_arm(args.master, "master")
    slave = connect_arm(args.slave, "slave")
    setup_master_slave(master, slave)
    time.sleep(0.5)

    pose_file = pathlib.Path(args.start_pose)
    if args.capture_start:
        start_7d = _read_7d(slave)
        np.save(str(pose_file), start_7d)
    elif pose_file.exists():
        start_7d = np.asarray(np.load(str(pose_file)), dtype=np.float32)
        if start_7d.shape != (7,):
            raise ValueError(f"start pose must be 7D, got {start_7d.shape}")
    else:
        start_7d = np.zeros(7, dtype=np.float32)

    wrist_key = f"cam_{args.arm_side}_wrist"
    cameras = None
    if args.record:
        cameras = CameraCapture(cam_ids={"cam_high": args.cam_high_id, wrist_key: args.cam_wrist_id}, fps=args.fps)
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

    auto_reset_pair(master, slave, start_7d)
    print(f"\n{'[RECORD]' if args.record else '[DRY RUN]'} schema={args.schema} fps={args.fps} SPACE=end E=e-stop q=quit\n")
    try:
        while not keys.quit:
            started = time.time()
            if keys.estop:
                keys.estop = False
                estop_pair(master, slave)
                recorder.start()
                keys.wait_for("r", "  Press R to recover: ")
                print()
                if keys.quit:
                    break
                recover_pair(master, slave)
                auto_reset_pair(master, slave, start_7d)
                _countdown(COUNTDOWN_S, keys)
                continue

            if args.schema == JOINT_SCHEMA:
                state = _read_7d(slave)
                state_timestamp = time.time()
                action = _read_7d(master)
                action_timestamp = time.time()
                joint_qpos = state.copy()
            else:
                state, joint_qpos = _read_eef_10d(slave)
                state_timestamp = time.time()
                master_qpos = _read_7d(master)
                action_timestamp = time.time()
                # Pose is replaced by the next measured slave EEF at save time;
                # the master contributes only its same-step gripper opening.
                action = state.copy()
                action[9] = master_qpos[6]

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
                sys.stdout.write(
                    f"\r[ep {ep_idx:03d}] step {len(recorder):04d} G_open={joint_qpos[6]:.2f} "
                    f"dt={int((time.time()-started)*1000)}ms   "
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
                    auto_reset_pair(master, slave, start_7d)
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
        teardown_master_slave(master)
        for arm in (master, slave):
            arm.DisconnectPort()
        print(f"Done. {ep_idx} episodes saved to {ep_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--slave", default=DEFAULT_SLAVE)
    parser.add_argument("--arm-side", choices=("left", "right"), default="right")
    parser.add_argument("--schema", choices=(JOINT_SCHEMA, DELIVERY_SCHEMA), default=JOINT_SCHEMA)
    parser.add_argument("--eef-calibration", default=None, help="deprecated compatibility option; delivery pose now uses next measured slave EEF")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--capture-start", action="store_true")
    parser.add_argument("--start-pose", default=START_POSE_FILE)
    parser.add_argument("--out-dir", default="episodes_single")
    parser.add_argument("--task-name", default="single_arm_task")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--dataset-root", default="pi0_dataset_single")
    parser.add_argument("--no-pi0-export", action="store_true")
    parser.add_argument("--cam-high-id", type=int, default=0)
    parser.add_argument("--cam-wrist-id", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

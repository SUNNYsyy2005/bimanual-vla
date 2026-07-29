"""Single-arm master-slave teleoperation with pi0.5/LeRobot dataset export."""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

import numpy as np

from camera import CameraCapture
from piper_sdk import C_PiperInterface_V2
from teleop import KeyListener, _countdown, _read_7d, _reset_one_arm, _send_7d, connect_arm
from trajectory import TrajectoryRecorder

DEFAULT_MASTER = "can0"
DEFAULT_SLAVE = "can1"
RECORD_HZ = 30
RESET_HZ = 20
RESET_DURATION_S = 4.0
RESET_SPEED_PCT = 20
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
    cur_master = _read_7d(master)
    cur_slave = _read_7d(slave)
    sys.stdout.write("\n[RESET] Master + slave → start pose...")
    sys.stdout.flush()
    threads = [
        threading.Thread(target=_reset_one_arm, args=(master, cur_master, start_7d)),
        threading.Thread(target=_reset_one_arm, args=(slave, cur_slave, start_7d)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
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


def _maybe_make_pi0_writer(args):
    if not args.record or args.no_pi0_export:
        return None
    from pi0_dataset import Pi0LeRobotDatasetWriter, single_arm_joint_names
    wrist_key = f"cam_{args.arm_side}_wrist"
    return Pi0LeRobotDatasetWriter(
        args.dataset_root,
        fps=RECORD_HZ,
        robot_type=f"piper_single_arm_{args.arm_side}",
        state_names=single_arm_joint_names(args.arm_side),
        action_names=single_arm_joint_names(args.arm_side),
        camera_keys=["cam_high", wrist_key],
        image_hw=(224, 224),
    )


def _save_episode(recorder: TrajectoryRecorder, args, ep_dir: pathlib.Path, ep_idx: int, pi0_writer=None, success: bool = True) -> int:
    if len(recorder) == 0:
        print("  (empty episode, skipping)")
        return ep_idx
    instruction = args.instruction or _default_instruction(args.task_name)
    extras = {
        "task_name": args.task_name,
        "instruction": instruction,
        "success": np.array(bool(success), dtype=np.bool_),
        "arm_side": np.array(args.arm_side),
    }
    raw_path = ep_dir / f"ep_{ep_idx:04d}.npz"
    recorder.save(raw_path, extras=extras)
    if pi0_writer is not None:
        episode = recorder.to_numpy_dict()
        images = {key.removeprefix("images_"): value for key, value in episode.items() if key.startswith("images_")}
        pi0_writer.append_episode(
            states=episode["qpos"],
            actions=episode["actions"],
            timestamps=episode["timestamps"],
            images=images,
            task_name=args.task_name,
            instruction=instruction,
            success=success,
            metadata={"source_raw_episode": str(raw_path.name), "arm_side": args.arm_side},
        )
        print(f"  pi0 dataset updated at {args.dataset_root}")
    recorder.start()
    return ep_idx + 1


def run(args):
    print("Connecting arms...")
    master = connect_arm(args.master, "master")
    slave = connect_arm(args.slave, "slave")
    setup_master_slave(master, slave)
    time.sleep(0.5)

    pose_file = pathlib.Path(args.start_pose)
    if args.capture_start:
        start_7d = _read_7d(slave)
        np.save(str(pose_file), start_7d)
        print(f"Start pose captured from slave arm and saved to {pose_file}")
    elif pose_file.exists():
        start_7d = np.load(str(pose_file))
        print(f"Loaded start pose from {pose_file}")
    else:
        start_7d = np.zeros(7, dtype=np.float64)
        print(f"No start pose file found ({pose_file}), using all-zeros.")

    wrist_key = f"cam_{args.arm_side}_wrist"
    cameras = None
    if args.record:
        cameras = CameraCapture(
            cam_ids={"cam_high": args.cam_high_id, wrist_key: args.cam_wrist_id},
            fps=RECORD_HZ,
        )
        cameras.open()
        for k, info in cameras.verify().items():
            print(f"  {k}: {'OK' if info['ok'] else 'FAIL'}  {info['latency_ms']} ms")

    recorder = TrajectoryRecorder()
    recorder.start()
    ep_dir = pathlib.Path(args.out_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_idx = 0
    keys = KeyListener()
    dt = 1.0 / RECORD_HZ
    pi0_writer = _maybe_make_pi0_writer(args)

    print("[RESET] Moving master + slave to start pose before first episode...")
    auto_reset_pair(master, slave, start_7d)
    print(f"\n{'[RECORD]' if args.record else '[DRY RUN]'} SPACE=end episode  E=e-stop  q=quit\n")

    try:
        while not keys.quit:
            t0 = time.time()
            if keys.estop:
                keys.estop = False
                estop_pair(master, slave)
                recorder.start()
                keys.wait_for('r', "  Press R to recover: ")
                print()
                if keys.quit:
                    break
                recover_pair(master, slave)
                auto_reset_pair(master, slave, start_7d)
                _countdown(COUNTDOWN_S, keys)
                continue

            qpos = _read_7d(slave)
            action = _read_7d(master)
            if cameras is not None:
                images, image_ts = cameras.read()
                recorder.add(qpos, action, images, image_ts)

            if args.record:
                sys.stdout.write(
                    f"\r[ep {ep_idx:03d}] step {len(recorder):04d}  "
                    f"G={qpos[6] * 1000:.0f}mm  dt={int((time.time() - t0) * 1000)}ms   "
                )
                sys.stdout.flush()

            if keys.end_episode:
                keys.end_episode = False
                print()
                n_steps = len(recorder)
                choice = keys.wait_for('sfd', f"  {n_steps} steps — S=save-success  F=save-fail  D=discard: ")
                print()
                if choice == 's':
                    ep_idx = _save_episode(recorder, args, ep_dir, ep_idx, pi0_writer, success=True)
                elif choice == 'f':
                    ep_idx = _save_episode(recorder, args, ep_dir, ep_idx, pi0_writer, success=False)
                else:
                    recorder.start()
                    print("  Discarded.")
                if not keys.quit and not keys.estop:
                    auto_reset_pair(master, slave, start_7d)
                    _countdown(COUNTDOWN_S, keys)

            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        print("\nShutting down...")
        if args.record and len(recorder) > 0:
            ep_idx = _save_episode(recorder, args, ep_dir, ep_idx, pi0_writer, success=True)
        if cameras:
            cameras.close()
        teardown_master_slave(master)
        for arm in (master, slave):
            arm.DisconnectPort()
        print(f"Done. {ep_idx} episodes saved to {ep_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--slave", default=DEFAULT_SLAVE)
    ap.add_argument("--arm-side", choices=["left", "right"], default="right")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--capture-start", action="store_true")
    ap.add_argument("--start-pose", default=START_POSE_FILE)
    ap.add_argument("--out-dir", default="episodes_single")
    ap.add_argument("--task-name", default="single_arm_task")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--dataset-root", default="pi0_dataset_single")
    ap.add_argument("--no-pi0-export", action="store_true")
    ap.add_argument("--cam-high-id", type=int, default=0)
    ap.add_argument("--cam-wrist-id", type=int, default=2)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

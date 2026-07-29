"""Bimanual master-slave teleoperation with pi0.5/LeRobot dataset export."""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

import numpy as np

try:
    import termios
    import tty
    _UNIX = True
except ImportError:
    import msvcrt
    _UNIX = False

from camera import CameraCapture
from piper_sdk import C_PiperInterface_V2
from trajectory import TrajectoryRecorder

DEFAULT_LEFT_MASTER = "can0"
DEFAULT_LEFT_SLAVE = "can1"
DEFAULT_RIGHT_MASTER = "can2"
DEFAULT_RIGHT_SLAVE = "can3"

RECORD_HZ = 30
RESET_SPEED_PCT = 20
RESET_HZ = 20
RESET_DURATION_S = 4.0
COUNTDOWN_S = 3
START_POSE_FILE = "start_pose.npy"

_RAD_FACTOR = 57295.7795
_M_FACTOR = 1_000_000.0


def _read_7d(arm: C_PiperInterface_V2) -> np.ndarray:
    j = arm.GetArmJointMsgs().joint_state
    g = arm.GetArmGripperMsgs().gripper_state
    joints = np.array([
        j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6,
    ], dtype=np.float64) / _RAD_FACTOR
    gripper = float(g.grippers_angle) / _M_FACTOR
    return np.append(joints, gripper)


def _send_7d(arm: C_PiperInterface_V2, target: np.ndarray):
    joints = [round(float(v) * _RAD_FACTOR) for v in target[:6]]
    arm.JointCtrl(*joints)
    arm.GripperCtrl(round(abs(float(target[6])) * _M_FACTOR), 1000, 0x01, 0)


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
                    if ch == ' ':
                        self.end_episode = True
                    elif ch.lower() == 'e':
                        self.estop = True
                    elif ch.lower() == 'q':
                        self.quit = True
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        else:
            while not self.quit:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    self.last_key = ch.lower()
                    if ch == b' ':
                        self.end_episode = True
                    elif ch.lower() in (b'e', b'E'):
                        self.estop = True
                    elif ch.lower() in (b'q', b'Q'):
                        self.quit = True
                time.sleep(0.02)

    def wait_for(self, options: str, prompt: str) -> str:
        self.last_key = None
        sys.stdout.write(prompt)
        sys.stdout.flush()
        while not self.quit:
            k = self.last_key
            if k and k in options:
                self.last_key = None
                return k
            time.sleep(0.02)
        return 'q'


def _reset_one_arm(arm: C_PiperInterface_V2, current: np.ndarray, target: np.ndarray):
    n_steps = max(1, int(RESET_DURATION_S * RESET_HZ))
    arm.ModeCtrl(0x01, 0x01, RESET_SPEED_PCT, 0x00)
    time.sleep(0.05)
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        _send_7d(arm, current + alpha * (target - current))
        time.sleep(1.0 / RESET_HZ)


def _countdown(seconds: int, keys: KeyListener):
    for i in range(seconds, 0, -1):
        if keys.quit or keys.estop:
            return
        sys.stdout.write(f"\r[RESET] Starting in {i}s ...  ")
        sys.stdout.flush()
        time.sleep(1.0)
    print("\r[RECORD] GO                   ")


def estop_all(lm, ls, rm, rs):
    for arm in (lm, ls, rm, rs):
        arm.EmergencyStop(0x01)
    print("\n\033[91m[E-STOP] ALL ARMS STOPPED. Press R to recover.\033[0m")


def recover_all(lm, ls, rm, rs):
    print("[RECOVER] Releasing e-stop...")
    for arm in (lm, ls, rm, rs):
        arm.EmergencyStop(0x02)
    time.sleep(0.3)
    for arm in (lm, ls, rm, rs):
        arm.EnablePiper()
    time.sleep(0.3)
    setup_master_slave(lm, ls, rm, rs)
    print("[RECOVER] Done. Resetting to start pose...")


def handle_estop(lm, ls, rm, rs, start_14d: np.ndarray, recorder: TrajectoryRecorder, keys: KeyListener):
    keys.estop = False
    estop_all(lm, ls, rm, rs)
    recorder.start()
    keys.wait_for('r', "  Press R to recover: ")
    print()
    if keys.quit:
        return
    recover_all(lm, ls, rm, rs)
    auto_reset_all(lm, ls, rm, rs, start_14d)


def auto_reset_all(lm, ls, rm, rs, start_14d: np.ndarray):
    tgt_l = start_14d[0:7]
    tgt_r = start_14d[7:14]
    cur_lm = _read_7d(lm)
    cur_ls = _read_7d(ls)
    cur_rm = _read_7d(rm)
    cur_rs = _read_7d(rs)
    sys.stdout.write("\n[RESET] All 4 arms → start pose...")
    sys.stdout.flush()
    threads = [
        threading.Thread(target=_reset_one_arm, args=(lm, cur_lm, tgt_l)),
        threading.Thread(target=_reset_one_arm, args=(ls, cur_ls, tgt_l)),
        threading.Thread(target=_reset_one_arm, args=(rm, cur_rm, tgt_r)),
        threading.Thread(target=_reset_one_arm, args=(rs, cur_rs, tgt_r)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    setup_master_slave(lm, ls, rm, rs)
    time.sleep(0.3)
    print(" done.")


def connect_arm(can_name: str, role: str) -> C_PiperInterface_V2:
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
    task_name = (task_name or "teleop_task").strip()
    return task_name.replace("_", " ")


def _maybe_make_pi0_writer(args):
    if not args.record or args.no_pi0_export:
        return None
    from pi0_dataset import BIMANUAL_JOINT_NAMES, Pi0LeRobotDatasetWriter
    return Pi0LeRobotDatasetWriter(
        args.dataset_root,
        fps=RECORD_HZ,
        robot_type=args.robot_type,
        state_names=BIMANUAL_JOINT_NAMES,
        action_names=BIMANUAL_JOINT_NAMES,
        camera_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
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
    }
    raw_path = ep_dir / f"ep_{ep_idx:04d}.npz"
    recorder.save(raw_path, extras=extras)
    if pi0_writer is not None:
        episode = recorder.to_numpy_dict()
        images = {
            key.removeprefix("images_"): value
            for key, value in episode.items()
            if key.startswith("images_")
        }
        pi0_writer.append_episode(
            states=episode["qpos"],
            actions=episode["actions"],
            timestamps=episode["timestamps"],
            images=images,
            task_name=args.task_name,
            instruction=instruction,
            success=success,
            metadata={"source_raw_episode": str(raw_path.name)},
        )
        print(f"  pi0 dataset updated at {args.dataset_root}")
    recorder.start()
    return ep_idx + 1


def run(args):
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
        print(f"Start pose captured from slave arms and saved to {pose_file}")
    elif pose_file.exists():
        start_14d = np.load(str(pose_file))
        print(f"Loaded start pose from {pose_file}")
    else:
        start_14d = np.zeros(14, dtype=np.float64)
        print(f"No start pose file found ({pose_file}), using all-zeros.")

    cameras = None
    if args.record:
        print("Opening cameras...")
        cameras = CameraCapture(
            cam_ids={
                "cam_high": args.cam_high_id,
                "cam_left_wrist": args.cam_left_wrist_id,
                "cam_right_wrist": args.cam_right_wrist_id,
            },
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

    print("[RESET] Moving all 4 arms to start pose before first episode...")
    auto_reset_all(lm, ls, rm, rs, start_14d)
    print(f"\n{'[RECORD]' if args.record else '[DRY RUN]'} SPACE=end episode  E=e-stop  q=quit\n")

    try:
        while not keys.quit:
            t0 = time.time()
            if keys.estop:
                handle_estop(lm, ls, rm, rs, start_14d, recorder, keys)
                if not keys.quit:
                    _countdown(COUNTDOWN_S, keys)
                continue

            left_state = _read_7d(ls)
            right_state = _read_7d(rs)
            qpos = np.concatenate([left_state, right_state])
            left_action = _read_7d(lm)
            right_action = _read_7d(rm)
            action = np.concatenate([left_action, right_action])

            if cameras is not None:
                images, image_ts = cameras.read()
                recorder.add(qpos, action, images, image_ts)

            if args.record:
                sys.stdout.write(
                    f"\r[ep {ep_idx:03d}] step {len(recorder):04d}  "
                    f"L_g={left_state[6] * 1000:.0f}mm  "
                    f"R_g={right_state[6] * 1000:.0f}mm  "
                    f"dt={int((time.time() - t0) * 1000)}ms   "
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
                    auto_reset_all(lm, ls, rm, rs, start_14d)
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
        teardown_master_slave(lm, rm)
        for arm in (lm, ls, rm, rs):
            arm.DisconnectPort()
        print(f"Done. {ep_idx} episodes saved to {ep_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-master", default=DEFAULT_LEFT_MASTER)
    ap.add_argument("--left-slave", default=DEFAULT_LEFT_SLAVE)
    ap.add_argument("--right-master", default=DEFAULT_RIGHT_MASTER)
    ap.add_argument("--right-slave", default=DEFAULT_RIGHT_SLAVE)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--capture-start", action="store_true", help="Read slave-arm current pose as start pose and save to --start-pose")
    ap.add_argument("--start-pose", default=START_POSE_FILE)
    ap.add_argument("--out-dir", default="episodes")
    ap.add_argument("--task-name", default="teleop_task")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--dataset-root", default="pi0_dataset_bimanual")
    ap.add_argument("--robot-type", default="piper_bimanual")
    ap.add_argument("--no-pi0-export", action="store_true")
    ap.add_argument("--cam-high-id", type=int, default=0)
    ap.add_argument("--cam_left_wrist_id", type=int, default=2)
    ap.add_argument("--cam-right-wrist-id", dest="cam_right_wrist_id", type=int, default=4)
    ap.add_argument("--cam-left-wrist-id", dest="cam_left_wrist_id", type=int, default=2)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

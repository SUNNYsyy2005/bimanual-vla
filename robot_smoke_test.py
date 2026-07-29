"""Hardware smoke test for bimanual Piper + cameras.

Purpose:
- verify camera devices can open/read
- verify both controlled arms can connect/read qpos
- optionally verify the command path by resending current qpos (near-zero motion)

Examples:
    # safest: read-only smoke test
    python robot_smoke_test.py

    # also test command path by resending current pose
    python robot_smoke_test.py --send-current
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from camera import CameraCapture
from piper_env import GRIPPER_RANGE_M, JOINT_LIMITS_RAD, PiperBimanualEnv


@dataclass
class Summary:
    camera_ok: bool = False
    arm_ok: bool = False
    command_ok: bool = False
    steps: int = 0


def _check_qpos_ranges(qpos: np.ndarray):
    assert qpos.shape == (14,), f"expected qpos shape (14,), got {qpos.shape}"
    for side_name, offset in (("left", 0), ("right", 7)):
        joints = qpos[offset:offset + 6]
        grip = qpos[offset + 6]
        for i, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
            if not (lo <= joints[i] <= hi):
                raise RuntimeError(
                    f"{side_name} joint{i+1}={joints[i]:.4f} rad out of limit [{lo:.4f}, {hi:.4f}]"
                )
        g_lo, g_hi = GRIPPER_RANGE_M
        if not (g_lo <= grip <= g_hi):
            raise RuntimeError(
                f"{side_name} gripper={grip:.4f} m out of range [{g_lo:.4f}, {g_hi:.4f}]"
            )


def _make_cameras(args) -> CameraCapture:
    return CameraCapture(
        cam_ids={
            "cam_high": args.cam_high_id,
            "cam_left_wrist": args.cam_left_wrist_id,
            "cam_right_wrist": args.cam_right_wrist_id,
        },
        fps=args.camera_fps,
    )


def run(args):
    summary = Summary()
    env = None
    cameras = None

    try:
        if not args.skip_cameras:
            print("[1/3] Opening cameras...")
            cameras = _make_cameras(args)
            cameras.open()
            verify = cameras.verify()
            for key, info in verify.items():
                status = "OK" if info["ok"] else "FAIL"
                print(f"  {key}: {status} shape={info['shape']} latency={info['latency_ms']} ms")
                if not info["ok"]:
                    raise RuntimeError(f"camera {key} failed smoke test")

            images, timestamps = cameras.read()
            stale = cameras.check_stale(timestamps)
            if stale:
                raise RuntimeError(f"stale camera frames detected: {stale}")
            for key, img in images.items():
                print(f"  {key}: frame shape={img.shape} dtype={img.dtype}")
            summary.camera_ok = True
        else:
            print("[1/3] Skip cameras (--skip-cameras)")

        if not args.skip_arms:
            print("[2/3] Connecting arms...")
            env = PiperBimanualEnv(left_can=args.left_can, right_can=args.right_can, speed_pct=args.speed_pct)
            env.connect()

            qposes = []
            for i in range(args.steps):
                t0 = time.time()
                qpos = env.get_qpos().astype(np.float32)
                _check_qpos_ranges(qpos)
                qposes.append(qpos)
                print(
                    f"  step {i:02d}: qpos ok | "
                    f"left_g={qpos[6]*1000:.1f}mm right_g={qpos[13]*1000:.1f}mm "
                    f"read={(time.time()-t0)*1000:.1f}ms"
                )
                time.sleep(1.0 / args.hz)

            qposes = np.stack(qposes, axis=0)
            print(
                f"  qpos stats: left joints [{qposes[:, :6].min():.3f}, {qposes[:, :6].max():.3f}] "
                f"right joints [{qposes[:, 7:13].min():.3f}, {qposes[:, 7:13].max():.3f}]"
            )
            summary.arm_ok = True
            summary.steps = len(qposes)

            if args.send_current:
                print("[3/3] Command-path test: resending current qpos...")
                hold = env.get_qpos().astype(np.float32)
                for i in range(args.command_repeats):
                    env.step(hold)
                    print(f"  command {i+1}/{args.command_repeats}: current pose resent")
                    time.sleep(args.command_interval_s)
                summary.command_ok = True
            else:
                print("[3/3] Skip command-path test (use --send-current to enable)")
        else:
            print("[2/3] Skip arms (--skip-arms)")
            print("[3/3] Skip command-path test because arms are skipped")

    finally:
        if cameras is not None:
            cameras.close()
        if env is not None:
            env.disconnect()

    print("\n=== Smoke Test Summary ===")
    print(f"camera_ok={summary.camera_ok}")
    print(f"arm_ok={summary.arm_ok}")
    print(f"command_ok={summary.command_ok}")
    print(f"steps={summary.steps}")
    print("✅ Smoke test completed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-can", default="can0")
    ap.add_argument("--right-can", default="can1")
    ap.add_argument("--steps", type=int, default=5, help="number of qpos read cycles")
    ap.add_argument("--hz", type=float, default=2.0, help="qpos read rate")
    ap.add_argument("--speed-pct", type=int, default=10)
    ap.add_argument("--send-current", action="store_true", help="resend current qpos to verify command path")
    ap.add_argument("--command-repeats", type=int, default=3)
    ap.add_argument("--command-interval-s", type=float, default=0.5)
    ap.add_argument("--skip-cameras", action="store_true")
    ap.add_argument("--skip-arms", action="store_true")
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--cam-high-id", type=int, default=0)
    ap.add_argument("--cam-left-wrist-id", type=int, default=2)
    ap.add_argument("--cam-right-wrist-id", type=int, default=4)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

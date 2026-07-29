"""End-to-end smoke test for camera/robot + remote pi0.5 policy server.

Checks:
- camera capture works
- optional real-arm state read works
- websocket policy server reachable
- raw chunk inference shape/range looks valid
- action broker loop works for N steps
- optional real action execution through SafetyChecker

Examples:
    # safest: real cameras + remote server, but no arm motion
    python policy_server_smoke_test.py --shadow --steps 10

    # use real arm state while still not moving
    python policy_server_smoke_test.py --shadow --read-arms-in-shadow --steps 10

    # only after shadow passes
    python policy_server_smoke_test.py --steps 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from camera import CameraCapture
from piper_env import PiperBimanualEnv
from run import ActionChunkBroker, WebsocketClientPolicy, build_obs
from safety import SafetyChecker, SafetyViolation
from trajectory import TrajectoryRecorder

EXPECTED_ACTION_DIM = 14


@dataclass
class TestStats:
    raw_chunk_shape: tuple[int, ...] | None = None
    raw_roundtrip_ms: float | None = None
    loop_steps: int = 0
    avg_loop_ms: float | None = None


def _validate_action_vector(action: np.ndarray):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (EXPECTED_ACTION_DIM,):
        raise RuntimeError(f"expected action shape (14,), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise RuntimeError("non-finite action values detected")


def _validate_action_chunk(raw_actions: np.ndarray):
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if raw_actions.ndim == 1:
        _validate_action_vector(raw_actions)
    elif raw_actions.ndim == 2:
        if raw_actions.shape[1] != EXPECTED_ACTION_DIM:
            raise RuntimeError(f"expected raw chunk shape (T,14), got {raw_actions.shape}")
        if not np.all(np.isfinite(raw_actions)):
            raise RuntimeError("non-finite values in raw action chunk")
    else:
        raise RuntimeError(f"unexpected raw action rank: {raw_actions.shape}")
    return raw_actions


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
    stats = TestStats()
    env = None
    cameras = None
    recorder = TrajectoryRecorder()
    safety = SafetyChecker()

    try:
        print("[1/5] Opening cameras...")
        cameras = _make_cameras(args)
        cameras.open()
        for key, info in cameras.verify().items():
            status = "OK" if info["ok"] else "FAIL"
            print(f"  {key}: {status} shape={info['shape']} latency={info['latency_ms']} ms")
            if not info["ok"]:
                raise RuntimeError(f"camera {key} failed verify()")

        read_real_state = (not args.shadow) or args.read_arms_in_shadow
        if read_real_state:
            print("[2/5] Connecting arms for state read...")
            env = PiperBimanualEnv(left_can=args.left_can, right_can=args.right_can, speed_pct=args.speed_pct)
            env.connect()
        else:
            print("[2/5] Shadow mode without arm reads: will use zero state")

        print("[3/5] Connecting to policy server...")
        policy = WebsocketClientPolicy(host=args.server, port=args.port)
        broker = ActionChunkBroker(policy, action_horizon=args.action_horizon)

        if read_real_state:
            qpos = env.get_qpos().astype(np.float32)
            safety.record_qpos(qpos)
        else:
            qpos = np.zeros(14, dtype=np.float32)

        images, image_ts = cameras.read()
        obs = build_obs(qpos, images, prompt=args.instruction)

        print("[4/5] Raw server inference smoke test...")
        t0 = time.time()
        raw_result = policy.infer(obs)
        stats.raw_roundtrip_ms = (time.time() - t0) * 1000.0
        if "actions" not in raw_result:
            raise RuntimeError(f"server result missing 'actions' key: {list(raw_result.keys())}")
        raw_actions = _validate_action_chunk(raw_result["actions"])
        stats.raw_chunk_shape = tuple(raw_actions.shape)
        print(f"  raw chunk shape={stats.raw_chunk_shape} roundtrip={stats.raw_roundtrip_ms:.1f}ms")
        if raw_actions.ndim == 2:
            print(
                f"  raw chunk stats: min={raw_actions.min():.4f} max={raw_actions.max():.4f} "
                f"first_left={raw_actions[0,:6].round(3)} first_right={raw_actions[0,7:13].round(3)}"
            )
        else:
            print(f"  raw action: left={raw_actions[:6].round(3)} right={raw_actions[7:13].round(3)}")

        print("[5/5] Broker loop smoke test...")
        recorder.start()
        loop_ms = []
        for step in range(args.steps):
            t0 = time.time()
            if read_real_state:
                qpos = env.get_qpos().astype(np.float32)
                safety.record_qpos(qpos)
            else:
                qpos = np.zeros(14, dtype=np.float32)

            images, image_ts = cameras.read()
            obs = build_obs(qpos, images, prompt=args.instruction)
            result = broker.infer(obs)
            action = np.asarray(result["actions"], dtype=np.float32)
            _validate_action_vector(action)

            if args.shadow:
                print(
                    f"  step {step:02d}: "
                    f"left={action[:6].round(3)} g={action[6]:.3f} | "
                    f"right={action[7:13].round(3)} g={action[13]:.3f}"
                )
            else:
                try:
                    safety.check(qpos, action, image_ts)
                    env.step(action)
                    print(f"  step {step:02d}: action applied")
                except SafetyViolation as e:
                    env.emergency_stop()
                    raise RuntimeError(f"safety violation during integration test: {e}") from e

            recorder.add(qpos, action, images, image_ts)
            dt_ms = (time.time() - t0) * 1000.0
            loop_ms.append(dt_ms)
            time.sleep(max(0.0, (1.0 / args.hz) - (dt_ms / 1000.0)))

        stats.loop_steps = len(loop_ms)
        stats.avg_loop_ms = float(np.mean(loop_ms)) if loop_ms else None

        if args.save_traj and len(recorder) > 0:
            path = f"episodes/integration_{int(time.time())}.npz"
            recorder.save(path, extras={"instruction": args.instruction})
            print(f"  saved trajectory -> {path}")

    finally:
        if cameras is not None:
            cameras.close()
        if env is not None:
            env.disconnect()

    print("\n=== Policy Server Smoke Summary ===")
    print(f"raw_chunk_shape={stats.raw_chunk_shape}")
    print(f"raw_roundtrip_ms={None if stats.raw_roundtrip_ms is None else round(stats.raw_roundtrip_ms, 1)}")
    print(f"loop_steps={stats.loop_steps}")
    print(f"avg_loop_ms={None if stats.avg_loop_ms is None else round(stats.avg_loop_ms, 1)}")
    print("✅ Policy server smoke test completed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="192.168.101.9")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--left-can", default="can0")
    ap.add_argument("--right-can", default="can1")
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--cam-high-id", type=int, default=0)
    ap.add_argument("--cam-left-wrist-id", type=int, default=2)
    ap.add_argument("--cam-right-wrist-id", type=int, default=4)
    ap.add_argument("--hz", type=float, default=3.0, help="broker-loop target rate")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--shadow", action="store_true", help="infer only; do not send actions to arms")
    ap.add_argument("--read-arms-in-shadow", action="store_true", help="still connect arms and use real qpos in shadow mode")
    ap.add_argument("--save-traj", action="store_true")
    ap.add_argument("--speed-pct", type=int, default=10)
    ap.add_argument("--action-horizon", type=int, default=25)
    ap.add_argument("--instruction", default="pick up the red cube and place it in the box")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

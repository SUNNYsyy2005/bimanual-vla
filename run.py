"""Main inference loop for bimanual Piper + π0.5.

Shadow mode (--shadow):
  Policy runs, predicted actions are logged but NOT sent to arms.
  Use this to verify policy output before enabling real motion.

Real mode (default):
  Actions pass through SafetyChecker, then are sent to arms.

Example:
  # shadow inference on single left arm first:
  python run.py --shadow --server 192.168.101.9 --port 8000

  # real mode:
  python run.py --server 192.168.101.9 --port 8000
"""

import argparse
import signal
import sys
import time
import numpy as np

from piper_env import PiperBimanualEnv
from camera import CameraCapture
from safety import SafetyChecker, SafetyViolation
from trajectory import TrajectoryRecorder

try:
    from openpi_client import websocket_client_policy as _wcp
    from openpi_client import action_chunk_broker as _acb
except ImportError:
    raise SystemExit(
        "openpi_client not installed. Run: pip install -e /path/to/openpi-client"
    )

WebsocketClientPolicy = _wcp.WebsocketClientPolicy
ActionChunkBroker = _acb.ActionChunkBroker

TASK_INSTRUCTION = "pick up the red cube and place it in the box"
ACTION_HORIZON = 25   # re-query policy every 25 steps
TARGET_HZ = 10        # control loop frequency (conservative for first deployment)


def build_obs(qpos: np.ndarray, images: dict) -> dict:
    """Build observation dict matching π0.5 AlohaInputs format.

    AlohaInputs expects images keyed as cam_high / cam_left_wrist / cam_right_wrist,
    each (C, H, W) uint8.  camera.py already returns them in that format.
    """
    return {
        "state":  qpos.astype(np.float32),
        "images": {
            "cam_high":        images["cam_high"],
            "cam_left_wrist":  images["cam_left_wrist"],
            "cam_right_wrist": images["cam_right_wrist"],
        },
        "prompt": TASK_INSTRUCTION,
    }


def run(args):
    env     = PiperBimanualEnv(left_can=args.left_can, right_can=args.right_can)
    cameras = CameraCapture()
    safety  = SafetyChecker()
    recorder = TrajectoryRecorder()

    _stop = False

    def _sigint(sig, frame):
        nonlocal _stop
        print("\n[run] SIGINT — stopping after current step")
        _stop = True

    signal.signal(signal.SIGINT, _sigint)

    print("Opening cameras...")
    cameras.open()
    verify = cameras.verify()
    for key, info in verify.items():
        status = "OK" if info["ok"] else "FAIL"
        print(f"  {key}: {status}  {info['shape']}  latency={info['latency_ms']} ms")

    print("Connecting to policy server...")
    policy = WebsocketClientPolicy(host=args.server, port=args.port)
    broker = ActionChunkBroker(policy, action_horizon=ACTION_HORIZON)

    if not args.shadow:
        print("Connecting arms...")
        env.connect()

    recorder.start()
    step = 0
    dt = 1.0 / TARGET_HZ

    print(f"\n{'[SHADOW MODE]' if args.shadow else '[REAL MODE]'} Starting loop at {TARGET_HZ} Hz. Ctrl-C to stop.\n")

    try:
        while not _stop:
            t0 = time.time()

            # --- observe ---
            if args.shadow:
                qpos = np.zeros(14, dtype=np.float32)   # dummy state in shadow
            else:
                qpos = env.get_qpos()
                safety.record_qpos(qpos)

            images, timestamps = cameras.read()

            stale = cameras.check_stale(timestamps)
            if stale:
                print(f"[warn] stale cameras: {stale}")

            # --- infer ---
            obs = build_obs(qpos, images)
            result = broker.infer(obs)          # {"actions": (14,), ...} after AlohaOutputs
            action = np.asarray(result["actions"], dtype=np.float32)

            # --- act ---
            if args.shadow:
                print(f"step {step:04d} | left_j={action[:6].round(3)} g={action[6]:.3f} "
                      f"| right_j={action[7:13].round(3)} g={action[13]:.3f}")
            else:
                try:
                    safety.check(qpos, action, timestamps)
                    env.step(action)
                except SafetyViolation as e:
                    print(f"[SAFETY] {e}")
                    env.emergency_stop()
                    break

            recorder.add(qpos, action, images)
            step += 1

            # --- pace ---
            elapsed = time.time() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)
            elif step % 50 == 0:
                print(f"[warn] loop overrun by {-sleep*1000:.0f} ms")

    finally:
        if not args.shadow:
            env.disconnect()
        cameras.close()
        if step > 0 and args.save_traj:
            path = f"episodes/ep_{int(time.time())}.npz"
            recorder.save(path)
        print(f"Loop ended after {step} steps.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server",    default="192.168.101.9")
    ap.add_argument("--port",      type=int, default=8000)
    ap.add_argument("--left-can",  default="can0")
    ap.add_argument("--right-can", default="can1")
    ap.add_argument("--shadow",    action="store_true",
                    help="Run policy but do not send actions to arms")
    ap.add_argument("--save-traj", action="store_true",
                    help="Save trajectory to episodes/ when loop ends")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

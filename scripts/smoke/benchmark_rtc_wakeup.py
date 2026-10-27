"""Loopback-only latency benchmark; no robot, GPU, model weights or remote host.

Run with the robot Conda environment:
  python -m scripts.smoke.benchmark_rtc_wakeup --samples 30

The simulated policy sleeps for the measured historical model processing time.
This validates scheduling/transport overhead, NOT real-model latency attainment.
"""
import argparse
import asyncio
import json
import threading
import time

import numpy as np
from websockets.asyncio.server import serve

from bimanual_vla.deployment.client import AsyncPolicyInference, InferenceLaunch, InferenceWorkerResult
from bimanual_vla.deployment.servo import ServoConfig, ServoLoop
from bimanual_vla.deployment.transport import TimedWebsocketClientPolicy, serve_policy_requests


def benchmark(samples, model_ms):
    ready = threading.Event()
    connection = {}
    actions = np.full((50, 14), 0.01, dtype=np.float32)

    class SimulatedPolicy:
        def infer(self, observation):
            started = time.monotonic()
            time.sleep(model_ms / 1000)
            return {"actions": actions, "transport_timing": {
                "model_inference_ms": (time.monotonic() - started) * 1000}}

    async def server_main():
        stop = asyncio.Event()
        async def handler(socket):
            await serve_policy_requests(SimulatedPolicy(), {}, socket)
        async with serve(handler, "127.0.0.1", 0, compression=None, max_size=None) as server:
            connection.update(port=server.sockets[0].getsockname()[1],
                              loop=asyncio.get_running_loop(), stop=stop)
            ready.set()
            await stop.wait()

    thread = threading.Thread(target=lambda: asyncio.run(server_main()), daemon=True)
    thread.start()
    if not ready.wait(5):
        raise RuntimeError("loopback server failed to start")
    client = TimedWebsocketClientPolicy(host="127.0.0.1", port=connection["port"])
    worker = AsyncPolicyInference()
    feedback = np.zeros(14)
    servo = ServoLoop(feedback.copy(), -np.ones(14), np.ones(14), ServoConfig(),
                      lambda: feedback.copy(), lambda q: (feedback.__setitem__(slice(None), q) or {}))
    observations = {"state": np.zeros(14, dtype=np.float32), "images": {
        key: np.zeros((224, 224, 3), dtype=np.uint8)
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}
    rows = {"poll_20hz": [], "completion_event": []}
    try:
        # Alternate modes to avoid attributing startup/temperature drift to one.
        for index in range(samples + 1):
            for mode in rows:
                started = time.monotonic()
                generation = index * 2 + (mode == "completion_event")
                launch = InferenceLaunch(generation, time.time(), started, time.time(), started,
                                         np.zeros(0), feedback.copy(), {})
                def infer():
                    result = client.infer({**observations, "client_metadata": {
                        "inference_generation": generation}})
                    return InferenceWorkerResult(result, {}, {
                        "response_received_monotonic": time.monotonic()})
                worker.launch_callable(infer, launch)
                results = []
                next_tick = started + 0.05
                while not results:
                    if mode == "completion_event":
                        def accept(result):
                            results.append(result)
                            return True
                        worker.wait_until(next_tick, accept)
                    else:
                        time.sleep(max(0, next_tick - time.monotonic()))
                        result = worker.poll()
                        if result is not None:
                            results.append(result)
                    next_tick += 0.05
                completion = results[0]
                if completion.error:
                    raise completion.error
                # Only the in-memory servo callback receives this command.
                sample = servo.submit(completion.result["actions"][0])
                if index:  # Warm both modes once.
                    rows[mode].append({
                        "capture_to_admission_ms": (completion.arrived_monotonic - started) * 1000,
                        "capture_to_servo_publish_ms": (sample["monotonic_timestamp"] - started) * 1000,
                        "receive_to_admission_ms": completion.result["_client_transport_timing"]["response_to_admission_ms"],
                        **completion.result["client_transport_timing"],
                    })
    finally:
        worker.shutdown()
        servo.close()
        client._ws.close()
        connection["loop"].call_soon_threadsafe(connection["stop"].set)
        thread.join(5)
    report = {"kind": "synthetic_loopback_no_hardware", "samples_per_mode": samples,
              "simulated_model_ms": model_ms, "target_ms": 150, "modes": {}}
    for mode, values in rows.items():
        report["modes"][mode] = {key: dict(zip(("p50", "p95", "p99"),
            np.percentile([r[key] for r in values], [50, 95, 99]).round(3).tolist()))
            for key in values[0] if key.endswith("_ms")}
    report["synthetic_target_met"] = report["modes"]["completion_event"]["capture_to_servo_publish_ms"]["p95"] <= 150
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--model-ms", type=float, default=109.5)
    args = parser.parse_args()
    if args.samples < 1 or not np.isfinite(args.model_ms) or args.model_ms < 0:
        parser.error("samples must be positive and model-ms finite and nonnegative")
    report = benchmark(args.samples, args.model_ms)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["synthetic_target_met"] else 1)

"""Evaluate the resumed run once its final checkpoint and an idle GPU exist."""

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
RESPONSE = json.loads((ROOT / "train_final_response.json").read_text())
TRAIN_EXIT = ROOT / "resume_from_5000_exit.json"
TRAIN_RUNTIME = ROOT / "resume_from_5000_runtime.json"
STATUS = ROOT / "resume_noise_followup_status.json"
CHECKPOINT = Path(RESPONSE["metadata"]["checkpoint_dir"]) / "20000"
OUTPUT = ROOT / "final_noise_evaluation"
MIDPOINT_OUTPUT = ROOT / "midpoint_10000_noise_evaluation"
MIDPOINT_EXIT = ROOT / "midpoint_10000_waiter_exit.json"
ENV_PYTHON = "/home/sunny/miniconda3/envs/openpi/bin/python"


def record(**kwargs) -> None:
    STATUS.write_text(json.dumps(kwargs, indent=2))


def complete_checkpoint() -> bool:
    return all((CHECKPOINT / name).is_file() for name in
               ("_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA"))


def worker_alive(runtime: dict) -> bool:
    pid = int(runtime["worker_pid"])
    try:
        current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return False
    return current == str(runtime["worker_start_ticks"])


def idle_gpu() -> int | None:
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    for line in inventory.splitlines():
        fields = line.split(",")
        if len(fields) == 3 and float(fields[1]) >= 23000 and float(fields[2]) < 5:
            return int(fields[0])
    return None


def main() -> int:
    deadline = time.monotonic() + 72 * 3600
    missing_since = None
    while time.monotonic() < deadline:
        if not TRAIN_EXIT.is_file():
            runtime = None
            if TRAIN_RUNTIME.is_file():
                runtime = json.loads(TRAIN_RUNTIME.read_text())
                if not worker_alive(runtime):
                    missing_since = missing_since or time.monotonic()
                    if time.monotonic() - missing_since >= 60:
                        record(state="aborted", reason="training process disappeared without exit record",
                               train_exit=str(TRAIN_EXIT), worker_pid=runtime["worker_pid"])
                        return 5
                else:
                    missing_since = None
            record(state="waiting_training", train_exit=str(TRAIN_EXIT),
                   worker_pid=runtime["worker_pid"] if runtime else None)
            time.sleep(30)
            continue
        train = json.loads(TRAIN_EXIT.read_text())
        if train.get("returncode") != 0:
            record(state="aborted", reason="resume training failed", train_returncode=train.get("returncode"))
            return 2
        if not complete_checkpoint():
            record(state="aborted", reason="final checkpoint incomplete despite zero training exit")
            return 3
        if (OUTPUT / "summary.json").is_file():
            record(state="completed", checkpoint=str(CHECKPOINT), output=str(OUTPUT), reused=True)
            return 0
        # If the first idle GPU arrives only after training ends, let the
        # user-requested 10000-step comparison take that window first.
        if not (MIDPOINT_OUTPUT / "summary.json").is_file() and not MIDPOINT_EXIT.is_file():
            record(state="waiting_midpoint_evaluation", checkpoint=str(CHECKPOINT),
                   midpoint_output=str(MIDPOINT_OUTPUT))
            time.sleep(30)
            continue
        try:
            gpu = idle_gpu()
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            print("GPU_INVENTORY_WAIT", type(error).__name__, str(error), flush=True)
            time.sleep(30)
            continue
        if gpu is None:
            record(state="waiting_idle_gpu", checkpoint=str(CHECKPOINT))
            time.sleep(30)
            continue
        env = {k: v for k, v in os.environ.items() if not k.startswith("BIMANUAL_VLA_SERVER_")}
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.75",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
                   PYTHONPATH="/home/sunny/bimanual-vla:/home/sunny/robotwin_ws/RoboTwin/policy/pi05/src")
        record(state="evaluating", checkpoint=str(CHECKPOINT), gpu=gpu)
        with (ROOT / "final_noise_evaluation.log").open("w") as log:
            result = subprocess.run(
                [ENV_PYTHON, "-u", str(ROOT / "evaluate_noise.py"),
                 "--checkpoint", str(CHECKPOINT), "--output", str(OUTPUT)],
                cwd="/home/sunny/bimanual-vla", env=env, stdout=log, stderr=subprocess.STDOUT)
        record(state="completed" if result.returncode == 0 else "failed",
               checkpoint=str(CHECKPOINT), gpu=gpu, returncode=result.returncode, output=str(OUTPUT))
        return result.returncode
    record(state="timeout", train_exit=str(TRAIN_EXIT))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())

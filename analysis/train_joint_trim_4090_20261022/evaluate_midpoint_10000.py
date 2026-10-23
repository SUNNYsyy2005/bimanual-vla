"""Run the 10000-step noise check after a safe GPU and memory window opens."""

import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parent
RESPONSE = json.loads((ROOT / "train_final_response.json").read_text())
CHECKPOINT = Path(RESPONSE["metadata"]["checkpoint_dir"]) / "10000"
OUTPUT = ROOT / "midpoint_10000_noise_evaluation"
STATUS = ROOT / "midpoint_10000_status.json"
LOG = ROOT / "resume_from_5000.log"
TRAIN_RUNTIME = ROOT / "resume_from_5000_runtime.json"
FINAL_STATUS = ROOT / "resume_noise_followup_status.json"
PYTHON = "/home/sunny/miniconda3/envs/openpi/bin/python"


def record(state: str, **details) -> None:
    payload = dict(state=state, checkpoint=str(CHECKPOINT), output=str(OUTPUT),
                   observed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(temporary, STATUS)


def checkpoint_complete() -> bool:
    return all((CHECKPOINT / name).is_file() for name in
               ("_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA"))


def save_in_progress() -> bool:
    # The training run logs both save starts and Orbax's finalized-directory event.
    data = LOG.read_bytes()[-2_000_000:].decode("utf-8", "replace")
    started = re.findall(r"Saving checkpoint at step (\d+)", data)
    finalized = re.findall(r"Finished saving checkpoint \(finalized tmp dir\) to `[^`]+/(\d+)`", data)
    return bool(started and (not finalized or int(started[-1]) > int(finalized[-1])))


def memory_available_gib() -> float:
    match = re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
    return int(match.group(1)) / 1024**2


def idle_gpu() -> int | None:
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    for line in inventory.splitlines():
        index, free, utilization = (float(field.strip()) for field in line.split(","))
        if int(index) == 3 and free >= 17000 and utilization < 5:
            return int(index)
    return None


def training_alive() -> bool:
    runtime = json.loads(TRAIN_RUNTIME.read_text())
    pid = int(runtime["worker_pid"])
    try:
        ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return False
    return ticks == str(runtime["worker_start_ticks"])


def main() -> int:
    deadline = time.monotonic() + 72 * 3600
    missing_since = None
    finalized_since = None
    while time.monotonic() < deadline:
        if (OUTPUT / "summary.json").is_file():
            record("completed", reused=True)
            return 0
        complete = checkpoint_complete()
        saving = save_in_progress()
        if not complete or saving:
            finalized_since = None
            if not training_alive():
                missing_since = missing_since or time.monotonic()
                if time.monotonic() - missing_since >= 60:
                    record("aborted", reason="training stopped before a finalized 10000 checkpoint")
                    return 5
            else:
                missing_since = None
            record("waiting_checkpoint", complete=complete, save_in_progress=saving)
            time.sleep(30)
            continue
        finalized_since = finalized_since or time.monotonic()
        # Let the finalization and cache-release activity settle before loading weights.
        if time.monotonic() - finalized_since < 60:
            record("settling_checkpoint")
            time.sleep(30)
            continue
        if FINAL_STATUS.is_file() and json.loads(FINAL_STATUS.read_text()).get("state") == "evaluating":
            record("waiting_final_evaluation")
            time.sleep(30)
            continue
        available = memory_available_gib()
        if available < 48:
            record("waiting_memory", available_gib=round(available, 1), required_gib=48)
            time.sleep(30)
            continue
        try:
            gpu = idle_gpu()
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            record("waiting_gpu_inventory", reason=repr(error))
            time.sleep(30)
            continue
        if gpu is None:
            record("waiting_idle_gpu", available_gib=round(available, 1),
                   required_free_mib=17000, requested_gpu=3)
            time.sleep(30)
            continue
        # Recheck the save boundary immediately before inference starts.
        if save_in_progress() or not checkpoint_complete() or memory_available_gib() < 48:
            time.sleep(30)
            continue
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("BIMANUAL_VLA_SERVER_")}
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.65",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
                   PYTHONPATH="/home/sunny/bimanual-vla:/home/sunny/robotwin_ws/RoboTwin/policy/pi05/src")
        record("evaluating", gpu=gpu, available_gib=round(available, 1))
        with (ROOT / "midpoint_10000_evaluation.log").open("w") as log:
            result = subprocess.run(
                [PYTHON, "-u", str(ROOT / "evaluate_noise.py"),
                 "--checkpoint", str(CHECKPOINT), "--output", str(OUTPUT)],
                cwd="/home/sunny/bimanual-vla", env=env,
                stdout=log, stderr=subprocess.STDOUT)
        summary = OUTPUT / "summary.json"
        good = result.returncode == 0 and summary.is_file()
        record("completed" if good else "failed", gpu=gpu,
               returncode=result.returncode, summary_present=summary.is_file())
        return 0 if good else result.returncode or 6
    record("timeout", reason="no safe evaluation window within 72 hours")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())

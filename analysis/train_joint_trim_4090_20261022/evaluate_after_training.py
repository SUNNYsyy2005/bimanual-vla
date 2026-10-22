"""Run one offline noise audit after this training succeeds and a GPU is idle."""
import json
import os
from pathlib import Path
import subprocess
import time

from dashboard import api


def main():
    root = Path(__file__).parent
    response = json.loads((root / "train_final_response.json").read_text())
    task_id = response["id"]
    target = Path(response["metadata"]["checkpoint_dir"])
    status_file = root / "noise_followup_status.json"
    deadline = time.monotonic()+72*3600
    while time.monotonic() < deadline:
        try:
            task = api("/api/tasks/"+task_id+"/status")["task"]
            if task["state"] in {"failed", "stopped", "lost", "interrupted"}:
                status_file.write_text(json.dumps(dict(state="aborted", train_task_id=task_id,
                    reason="training ended as "+task["state"]), indent=2))
                return
            if task["state"] != "completed":
                status_file.write_text(json.dumps(dict(state="waiting_training", train_task_id=task_id), indent=2))
                time.sleep(30)
                continue
            checkpoints = [p for p in target.iterdir() if p.name.isdigit() and
                (p/"_CHECKPOINT_METADATA").is_file() and (p/"params/_METADATA").is_file()]
            checkpoint = max(checkpoints, key=lambda p: int(p.name))
            assert int(checkpoint.name) >= int(response["metadata"]["steps"])-1
            output = root / "final_noise_evaluation"
            if (output/"summary.json").is_file():
                status_file.write_text(json.dumps(dict(state="completed", output=str(output)), indent=2))
                return
            inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                                 "--format=csv,noheader,nounits"], text=True)
            # Require a genuinely idle card; never evict or co-locate with services.
            idle = [int(row[0]) for line in inventory.splitlines() if
                    len(row := line.split(",")) == 3 and float(row[1]) >= 23000 and float(row[2]) < 5]
            if not idle:
                status_file.write_text(json.dumps(dict(state="waiting_idle_gpu", train_task_id=task_id), indent=2))
                time.sleep(30)
                continue
            env = {k: v for k, v in os.environ.items() if not k.startswith("BIMANUAL_VLA_SERVER_")}
            env.update(CUDA_VISIBLE_DEVICES=str(idle[0]), XLA_PYTHON_CLIENT_MEM_FRACTION="0.75",
                OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
                PYTHONPATH="/home/sunny/bimanual-vla:/home/sunny/robotwin_ws/RoboTwin/policy/pi05/src")
            status_file.write_text(json.dumps(dict(state="evaluating", train_task_id=task_id,
                checkpoint=str(checkpoint), gpu=idle[0]), indent=2))
            with (root / "final_noise_evaluation.log").open("w") as log:
                result = subprocess.run(["/home/sunny/miniconda3/envs/openpi/bin/python", "-u",
                    str(root/"evaluate_noise.py"), "--checkpoint", str(checkpoint), "--output", str(output)],
                    cwd="/home/sunny/bimanual-vla", env=env, stdout=log, stderr=subprocess.STDOUT)
            status_file.write_text(json.dumps(dict(state="completed" if result.returncode == 0 else "failed",
                train_task_id=task_id, checkpoint=str(checkpoint), returncode=result.returncode,
                output=str(output)), indent=2))
            return
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            print("WAIT", type(error).__name__, str(error), flush=True)
            time.sleep(30)
    status_file.write_text(json.dumps(dict(state="timeout", train_task_id=task_id), indent=2))


if __name__ == "__main__":
    main()

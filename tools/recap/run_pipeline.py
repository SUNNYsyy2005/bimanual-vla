"""Run official RECAP stages and retain a truthful per-stage status record."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import socket

root = Path(os.environ["TASK_ROOT"])
repo = Path(os.environ["REPO_PATH"])
records = root / "records"
config = root / "configs"
state = {"synthetic": True, "status": "running", "pid": os.getpid(), "host": socket.gethostname(),
         "job_id": os.environ.get("SLURM_JOB_ID"), "stages": []}

def save_state():
    temporary = records / f".pipeline-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(records / "pipeline.json")

def stage(name, args):
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log = records / f"{name}.log"
    entry = {"stage": name, "status": "running", "started": started, "log": str(log)}
    state["stages"].append(entry)
    save_state()
    print(f"Starting {name}", flush=True)
    with log.open("w") as output:
        result = subprocess.run([sys.executable, *map(str, args)], cwd=repo, stdout=output, stderr=subprocess.STDOUT)
    entry.update(status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode,
                 finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if result.returncode:
        state["status"] = "failed"
    save_state()
    print(f"{name}: {entry['status']}", flush=True)
    if result.returncode:
        print(log.read_text(errors="replace")[-6000:], flush=True)
        raise SystemExit(result.returncode)

stage("source_fixes", [root / "apply_recap_fixes.py"])
stage("return_contract_tests", [root / "test_returns_contract.py"])
stage("preflight", [root / "preflight.py"])
if not (records / "datasets.json").exists():
    stage("synthetic_data", [root / "create_synthetic_dataset.py"])
stage("policy_assets", [root / "prepare_policy_assets.py"])
stage("model_assets", [root / "check_model_assets.py"])
stage("configs", [root / "generate_configs.py"])
stage("returns", [repo / "examples/offline_rl/advantage_labeling/recap/process/compute_returns.py",
                  "--config-path", config, "--config-name", "returns"])
stage("value_train", [repo / "examples/offline_rl/advantage_labeling/recap/train_value.py",
                      "--config-path", config, "--config-name", "value"])
stage("value_resume", [repo / "examples/offline_rl/advantage_labeling/recap/train_value.py",
                       "--config-path", config, "--config-name", "value_resume"])
stage("advantages", [repo / "examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py",
                     "--config-path", config, "--config-name", "advantages"])
stage("cfg_train", [repo / "examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py",
                    "--config-path", config, "--config-name", "cfg"])
stage("cfg_resume", [repo / "examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py",
                     "--config-path", config, "--config-name", "cfg_resume"])
stage("verify", [root / "verify_pipeline.py"])
state["status"] = "passed"
save_state()

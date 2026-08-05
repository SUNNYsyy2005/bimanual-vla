#!/usr/bin/env python3
"""Submit and monitor a Slurm job from the Dashboard.

This helper intentionally does only lightweight orchestration on the 4x4090
Dashboard host: write an sbatch script through SSH, submit it, poll Slurm, and
mirror state to stdout. The actual GPU workload runs under Slurm on the target.
"""
from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


def b64_json(value: str) -> Any:
    return json.loads(base64.urlsafe_b64decode(value.encode()).decode())


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def run_ssh(host: str, command: str, *, input_text: str | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    ssh_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        host,
        command,
    ]
    return subprocess.run(
        ssh_cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def slurm_header(target: dict[str, Any], job_name: str, log_dir: str) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH -p {target['partition']}",
    ]
    node = target.get("node")
    if node:
        lines.append(f"#SBATCH -w {node}")
    gpu_type = target.get("gpu_type")
    gpu_count = int(target.get("gpu_count", 1))
    if gpu_count > 0:
        gres = f"gpu:{gpu_type}:{gpu_count}" if gpu_type else f"gpu:{gpu_count}"
        lines.append(f"#SBATCH --gres={gres}")
    lines += [
        f"#SBATCH --cpus-per-task={int(target.get('cpus_per_task', 8))}",
        f"#SBATCH --mem={target.get('mem', '64G')}",
        f"#SBATCH --time={target.get('time', '24:00:00')}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
        "",
        "set -euo pipefail",
    ]
    return "\n".join(lines)


def build_script(target: dict[str, Any], job_name: str, commands: list[list[str]], command_labels: list[str]) -> str:
    workdir = target["workdir"]
    log_dir = target.get("log_dir", f"{workdir.rstrip('/')}/logs/dashboard_slurm")
    cache_root = target.get("cache_root", f"/DATA/sync/$USER/.cache")
    conda_sh = target.get("conda_sh") or target.get("conda_init")
    conda_env = target.get("conda_env")
    prelude = [
        slurm_header(target, job_name, log_dir),
        f"mkdir -p {shlex.quote(log_dir)}",
        "echo '[dashboard] host='$(hostname)' pwd='$(pwd)' date='$(date -Is)",
        "command -v resources >/dev/null 2>&1 && resources || true",
        "command -v myquota >/dev/null 2>&1 && myquota || true",
        f"cd {shlex.quote(workdir)}",
        f"export XDG_CACHE_HOME={shlex.quote(target.get('xdg_cache_home', cache_root))}",
        f"export PIP_CACHE_DIR={shlex.quote(target.get('pip_cache_dir', cache_root + '/pip/cache'))}",
        f"export TMPDIR={shlex.quote(target.get('tmpdir', cache_root + '/pip/tmp'))}",
        f"export HF_HOME={shlex.quote(target.get('hf_home', cache_root + '/huggingface'))}",
        f"export HF_LEROBOT_HOME={shlex.quote(target['dataset_root'])}",
        f"export PYTHONUNBUFFERED=1",
        f"export TOKENIZERS_PARALLELISM=false",
        f"export XLA_PYTHON_CLIENT_MEM_FRACTION={target.get('xla_memory_fraction', 0.90)}",
    ]
    if conda_sh:
        prelude.append(f"source {shlex.quote(conda_sh)}")
    if conda_env:
        prelude.append(f"conda activate {shlex.quote(conda_env)}")
    body = []
    for idx, command in enumerate(commands):
        label = command_labels[idx] if idx < len(command_labels) else f"command_{idx + 1}"
        body += [
            f"echo '[dashboard] start {label}: '$(date -Is)",
            "srun " + shell_join(command),
            f"echo '[dashboard] done {label}: '$(date -Is)",
        ]
    return "\n".join(prelude + [""] + body + [""])


def parse_job_id(output: str) -> str:
    for token in output.replace(".", " ").split():
        if token.isdigit():
            return token
    raise RuntimeError(f"could not parse sbatch job id from: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-json", required=True, help="urlsafe-base64 encoded target config JSON")
    parser.add_argument("--commands-json", required=True, help="urlsafe-base64 encoded list[list[str]]")
    parser.add_argument("--command-labels-json", default="", help="urlsafe-base64 encoded labels")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--poll-interval", type=float, default=20.0)
    args = parser.parse_args()

    target = b64_json(args.target_json)
    commands = b64_json(args.commands_json)
    labels = b64_json(args.command_labels_json) if args.command_labels_json else []
    host = target["submit_host"]
    remote_job_dir = target.get("remote_job_dir", f"{target['workdir'].rstrip('/')}/logs/dashboard_slurm")
    script = build_script(target, args.job_name, commands, labels)
    remote_script = f"{remote_job_dir.rstrip('/')}/{args.job_name}_{int(time.time())}.sbatch"

    print(f"[dashboard] submitting Slurm job on {host}: {args.job_name}", flush=True)
    print(f"[dashboard] remote script: {remote_script}", flush=True)
    mkdir = run_ssh(host, f"mkdir -p {shlex.quote(remote_job_dir)}", timeout=60)
    if mkdir.returncode:
        print(mkdir.stdout, flush=True)
        return mkdir.returncode
    upload = run_ssh(host, f"cat > {shlex.quote(remote_script)}", input_text=script, timeout=60)
    if upload.returncode:
        print(upload.stdout, flush=True)
        return upload.returncode
    submit = run_ssh(host, f"sbatch {shlex.quote(remote_script)}", timeout=60)
    print(submit.stdout, end="", flush=True)
    if submit.returncode:
        return submit.returncode
    job_id = parse_job_id(submit.stdout)
    print(f"[dashboard] slurm_job_id={job_id}", flush=True)

    last_state = None
    while True:
        q = run_ssh(host, f"squeue -h -j {shlex.quote(job_id)} -o '%T|%M|%R'", timeout=60)
        qout = q.stdout.strip()
        if q.returncode == 0 and qout:
            if qout != last_state:
                print(f"[dashboard] squeue {job_id}: {qout}", flush=True)
                last_state = qout
            time.sleep(max(5.0, args.poll_interval))
            continue
        acct = run_ssh(
            host,
            f"sacct -j {shlex.quote(job_id)} --format=JobID,State,ExitCode,Elapsed -n -P",
            timeout=60,
        )
        print(acct.stdout, end="", flush=True)
        state_lines = [line.split("|") for line in acct.stdout.splitlines() if line.strip()]
        batch_line = next((parts for parts in state_lines if parts and parts[0].endswith(".batch")), None)
        main_line = next((parts for parts in state_lines if parts and parts[0] == job_id), None)
        chosen = batch_line or main_line
        state = chosen[1] if chosen and len(chosen) > 1 else "UNKNOWN"
        exit_code = chosen[2] if chosen and len(chosen) > 2 else ""
        print(f"[dashboard] final_state={state} exit_code={exit_code}", flush=True)
        return 0 if state.startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())

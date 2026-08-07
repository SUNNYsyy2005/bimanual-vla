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
import os
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


def clean_transport_env() -> dict[str, str]:
    # The Dashboard process uses the OpenPI conda environment for Python/JAX,
    # but system OpenSSH must load the system OpenSSL ABI on login-server.
    env = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "OPENSSL_CONF", "OPENSSL_MODULES"):
        env.pop(key, None)
    return env


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
    try:
        return subprocess.run(
            ssh_cmd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=clean_transport_env(),
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or exc.stderr or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return subprocess.CompletedProcess(ssh_cmd, 124, str(output))


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
    openpi_src = target.get("openpi_src", workdir.rstrip("/") + "/src")
    openpi_client_src = target.get(
        "openpi_client_src", workdir.rstrip("/") + "/packages/openpi-client/src"
    )
    # The H100 checkout keeps LeRobot as a sibling source checkout under
    # <root>/src/lerobot_src/<revision>.  Add the conventional location as a
    # fallback; nonexistent PYTHONPATH entries are harmless and this keeps
    # the target config backward compatible.
    inferred_lerobot_src = str(Path(workdir).parent / "lerobot_src/a445d9c")
    lerobot_src = target.get("lerobot_src", inferred_lerobot_src)
    prelude = [
        slurm_header(target, job_name, log_dir),
        f"mkdir -p {shlex.quote(log_dir)}",
        "echo '[dashboard] host='$(hostname)' pwd='$(pwd)' date='$(date -Is)",
        "command -v resources >/dev/null 2>&1 && resources || true",
        "command -v myquota >/dev/null 2>&1 && myquota || true",
        f"cd {shlex.quote(workdir)}",
        # The H100 checkout is a src-layout project and is not necessarily
        # installed editable in the compute-node environment.  Ensure the
        # remote helper can import openpi, openpi_client, and LeRobot after
        # `cd` without relying on a login-shell activation.
        "export PYTHONPATH="
        f"{shlex.quote(str(openpi_src))}:{shlex.quote(str(openpi_client_src))}:{shlex.quote(str(lerobot_src))}"
        "${PYTHONPATH:+:$PYTHONPATH}",
        f"export XDG_CACHE_HOME={shlex.quote(target.get('xdg_cache_home', cache_root))}",
        f"export PIP_CACHE_DIR={shlex.quote(target.get('pip_cache_dir', cache_root + '/pip/cache'))}",
        f"export TMPDIR={shlex.quote(target.get('tmpdir', cache_root + '/pip/tmp'))}",
        f"export HF_HOME={shlex.quote(target.get('hf_home', cache_root + '/huggingface'))}",
        f"export HF_LEROBOT_HOME={shlex.quote(target['dataset_root'])}",
        f"export PYTHONUNBUFFERED=1",
        f"export TOKENIZERS_PARALLELISM=false",
        f"export XLA_PYTHON_CLIENT_MEM_FRACTION={target.get('xla_memory_fraction', 0.90)}",
    ]
    # Some cluster compute nodes expose only the configured environment
    # executable (for example /home/sunny/miniconda3/envs/openpi/bin/python)
    # and do not mount the Conda root's profile.d/conda.sh.  The workload
    # commands already use target["openpi_python"] directly, so activation is
    # optional; never let a missing init script abort an otherwise valid job.
    if conda_sh:
        quoted_conda_sh = shlex.quote(conda_sh)
        prelude.append(
            f"if [ -f {quoted_conda_sh} ]; then source {quoted_conda_sh}; "
            "else echo '[dashboard] conda init script not present; using configured Python'; fi"
        )
    if conda_env:
        prelude.append(
            "if command -v conda >/dev/null 2>&1; then "
            f"conda activate {shlex.quote(conda_env)}; "
            "else echo '[dashboard] conda command unavailable; using configured Python'; fi"
        )
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
    transport_failures = 0
    while True:
        q = run_ssh(host, f"squeue -h -j {shlex.quote(job_id)} -o '%T|%M|%R'", timeout=60)
        qout = q.stdout.strip()
        if q.returncode == 0 and qout:
            transport_failures = 0
            if qout != last_state:
                print(f"[dashboard] squeue {job_id}: {qout}", flush=True)
                last_state = qout
            time.sleep(max(5.0, args.poll_interval))
            continue

        # A job disappearing from squeue is normally terminal, but the SSH
        # connection can transiently fail exactly at that boundary.  Query
        # sacct and retry instead of converting one transport failure into an
        # UNKNOWN/failed task.  This is especially important for long H100
        # jobs, where login-server may briefly refuse a new SSH session.
        if q.returncode != 0:
            transport_failures += 1
            print(
                f"[dashboard] squeue status unavailable for {job_id} "
                f"(attempt {transport_failures}): {qout[-500:]}",
                flush=True,
            )

        acct = run_ssh(
            host,
            f"sacct -j {shlex.quote(job_id)} --format=JobID,State,ExitCode,Elapsed -n -P",
            timeout=60,
        )
        if acct.stdout:
            print(acct.stdout, end="", flush=True)
        state_lines = [line.split("|") for line in acct.stdout.splitlines() if line.strip()]
        batch_line = next((parts for parts in state_lines if parts and parts[0].endswith(".batch")), None)
        main_line = next((parts for parts in state_lines if parts and parts[0] == job_id), None)
        chosen = batch_line or main_line
        if chosen and len(chosen) > 1:
            # sacct can report RUNNING/PENDING while squeue is temporarily
            # unavailable; keep monitoring those states instead of exiting.
            state = chosen[1]
            exit_code = chosen[2] if len(chosen) > 2 else ""
            transport_failures = 0
            print(f"[dashboard] sacct {job_id}: state={state} exit_code={exit_code}", flush=True)
            if state.startswith("COMPLETED") or state not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
                print(f"[dashboard] final_state={state} exit_code={exit_code}", flush=True)
                return 0 if state.startswith("COMPLETED") else 1
            time.sleep(max(5.0, args.poll_interval))
            continue

        # No reliable answer yet.  Keep the runner alive through a bounded
        # outage; only report UNKNOWN after repeated failed queries.
        if transport_failures < 12:
            print(f"[dashboard] Slurm status temporarily unavailable for {job_id}; retrying", flush=True)
            time.sleep(max(5.0, args.poll_interval))
            continue
        print("[dashboard] final_state=UNKNOWN exit_code=", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

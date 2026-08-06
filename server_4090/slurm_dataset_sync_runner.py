#!/usr/bin/env python3
"""Sync one LeRobot dataset when one side is Slurm-only.

The Dashboard host never opens a port on H100/H200 and never bypasses Slurm.
For Slurm-only nodes this helper stages through NAS, then submits a CPU-only
Slurm copy job on the target/source node.  Direct SSH transfers are still
handled by dataset_transfer_runner.py.
"""
from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import PurePosixPath
from typing import Any

APP_DIR = __import__("pathlib").Path(__file__).resolve().parent


def b64_json(value: str) -> Any:
    return json.loads(base64.urlsafe_b64decode(value.encode()).decode())


def json_arg(value: Any) -> str:
    return base64.urlsafe_b64encode(json.dumps(value, ensure_ascii=False).encode()).decode()


def q(value: Any) -> str:
    return shlex.quote(str(value))


def dataset_path(location: dict[str, Any], dataset_id: str) -> str:
    return str(PurePosixPath(str(location["dataset_root"]).rstrip("/")) / dataset_id)


def staging_root(source: dict[str, Any], target: dict[str, Any], fallback: str) -> str:
    for item in (target, source):
        value = item.get("nas_dataset_staging_root") or item.get("dataset_staging_root")
        if value:
            return str(value).rstrip("/")
    return fallback.rstrip("/")


def run(command: list[str], *, label: str) -> None:
    print(f"[dashboard] start {label}: {' '.join(shlex.quote(str(x)) for x in command)}", flush=True)
    result = subprocess.run(command, text=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(f"[dashboard] done {label}", flush=True)


def transfer(source: dict[str, Any], target: dict[str, Any], dataset_id: str, *, overwrite: bool, parallelism: int, label: str, skip_existing: bool = False) -> None:
    cmd = [
        sys.executable,
        str(APP_DIR / "dataset_transfer_runner.py"),
        "--dataset-id", dataset_id,
        "--source-json", json_arg(source),
        "--target-json", json_arg(target),
        "--parallelism", str(parallelism),
    ]
    if overwrite:
        cmd.append("--overwrite")
    if skip_existing:
        cmd.append("--skip-existing")
    run(cmd, label=label)


def inventory_refresh_snippet(target: dict[str, Any]) -> str:
    cache_path = target.get("inventory_source_path") or target.get("inventory_cache_path")
    if not cache_path:
        return "echo '[dashboard] no inventory cache path configured; skip inventory refresh'"
    root = str(target["dataset_root"])
    return textwrap.dedent(f"""
    python3 - <<'PY'
    import json, os, time
    from pathlib import Path
    root = Path({root!r})
    out = Path({str(cache_path)!r})
    def read_json(path):
        try:
            with open(path, encoding='utf-8') as f: return json.load(f)
        except Exception: return None
    def origin_for(dataset_id, info, marker):
        if isinstance(marker, dict) and marker.get('origin') in {{'real','simulation','unknown'}}: return marker.get('origin')
        for key in ('dataset_origin','data_origin','source_domain'):
            value = str((info or {{}}).get(key, '')).lower()
            if value in {{'real','robot','real_robot','physical'}}: return 'real'
            if value in {{'simulation','sim','synthetic','synthetic_sim'}}: return 'simulation'
        name = dataset_id.lower(); robot_type = str((info or {{}}).get('robot_type') or '').lower()
        if robot_type == 'piper': return 'real'
        if __import__('re').search(r'(?:^|[._-])real(?:[._-]|$)', name) or name == 'my_dataset': return 'real'
        simulation_name = any(token in name for token in ('sim','synth','synthetic','smoke','robottwin'))
        if simulation_name or robot_type in {{'aloha','sim','simulation'}} or (robot_type.startswith('piper_single_arm') and bool((info or {{}}).get('video_path'))): return 'simulation'
        if 'piper' in robot_type: return 'real'
        return 'unknown'
    rows=[]
    if root.exists():
        for d in sorted(root.iterdir(), key=lambda p:p.name):
            if not d.is_dir() or d.name.startswith('.'): continue
            info = read_json(d/'meta'/'info.json')
            if not isinstance(info, dict): continue
            marker = read_json(d/'meta'/'dashboard_dataset_origin.json')
            st = d.stat()
            rows.append({{'id':d.name,'origin':origin_for(d.name, info, marker),'path':str(d),'episodes':info.get('total_episodes'),'frames':info.get('total_frames'),'fps':info.get('fps'),'robot_type':info.get('robot_type'),'mtime':st.st_mtime,'marker':marker if isinstance(marker, dict) else None}})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    tmp.write_text(json.dumps({{'generated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'roots':[str(root)],'datasets':rows}}, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, out)
    print(f'[dashboard] wrote inventory {{out}} datasets={{len(rows)}}')
    PY
    """).strip()


def slurm_copy_job(target: dict[str, Any], commands: list[list[str]], labels: list[str], dataset_id: str, direction: str) -> None:
    job_target = dict(target)
    job_target["gpu_count"] = 0
    # Dataset staging is a CPU / storage operation; do not require a Conda env
    # to exist on the target node just to copy files.
    job_target.pop("conda_sh", None)
    job_target.pop("conda_env", None)
    job_target["time"] = job_target.get("sync_time", "04:00:00")
    job_target["mem"] = job_target.get("sync_mem", "64G")
    job_target["cpus_per_task"] = int(job_target.get("sync_cpus_per_task", 4))
    job_name = f"dsync_{direction}_{dataset_id[:32]}_{int(time.time())}"
    cmd = [
        sys.executable,
        str(APP_DIR / "slurm_job_runner.py"),
        "--target-json", json_arg(job_target),
        "--commands-json", json_arg(commands),
        "--command-labels-json", json_arg(labels),
        "--job-name", job_name,
        "--poll-interval", "10",
    ]
    run(cmd, label=f"slurm {direction} copy")


def copy_from_staging_command(staging: str, target: dict[str, Any], dataset_id: str, overwrite: bool, skip_existing: bool = False) -> str:
    src = str(PurePosixPath(staging) / dataset_id)
    root = str(target["dataset_root"]).rstrip("/")
    dst = str(PurePosixPath(root) / dataset_id)
    tmp = f"{dst}.incoming-$$"
    if overwrite:
        overwrite_logic = f"rm -rf {q(dst)}"
    elif skip_existing:
        overwrite_logic = f"if test -e {q(dst)}; then echo '[dashboard] target already exists; skip import: {dst}'; exit 0; fi"
    else:
        overwrite_logic = f"test ! -e {q(dst)}"
    return "\n".join([
        "set -euo pipefail",
        f"test -d {q(src)}",
        f"mkdir -p {q(root)}",
        overwrite_logic,
        f"rm -rf {q(tmp)}",
        f"mkdir -p {q(tmp)}",
        f"tar -C {q(src)} -cf - . | tar -C {q(tmp)} -xf -",
        f"mv {q(tmp)} {q(dst)}",
        f"echo '[dashboard] installed dataset to {dst}'",
        inventory_refresh_snippet(target),
    ])


def copy_to_staging_command(source: dict[str, Any], staging: str, dataset_id: str, overwrite: bool) -> str:
    src = dataset_path(source, dataset_id)
    root = staging.rstrip("/")
    dst = str(PurePosixPath(root) / dataset_id)
    tmp = f"{dst}.incoming-$$"
    overwrite_logic = f"rm -rf {q(dst)}" if overwrite else f"test ! -e {q(dst)}"
    return "\n".join([
        "set -euo pipefail",
        f"test -d {q(src)}",
        f"mkdir -p {q(root)}",
        overwrite_logic,
        f"rm -rf {q(tmp)}",
        f"mkdir -p {q(tmp)}",
        f"tar -C {q(src)} -cf - . | tar -C {q(tmp)} -xf -",
        f"mv {q(tmp)} {q(dst)}",
        f"echo '[dashboard] exported dataset to {dst}'",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--target-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Return success without copying when target already exists")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--nas-staging-root", default="/DATA/NAS/GPUServer/sunny/dashboard_dataset_sync")
    args = parser.parse_args()

    source = b64_json(args.source_json)
    target = b64_json(args.target_json)
    source_slurm = source.get("kind") == "slurm_only" or source.get("access_mode") == "slurm_only"
    target_slurm = target.get("kind") == "slurm_only" or target.get("access_mode") == "slurm_only"
    if not (source_slurm or target_slurm):
        transfer(source, target, args.dataset_id, overwrite=args.overwrite, parallelism=args.parallelism, label="direct transfer", skip_existing=args.skip_existing)
        return 0

    staging = staging_root(source, target, args.nas_staging_root)
    staging_location = {
        "name": "nas_staging",
        "kind": "ssh",
        "host": target.get("submit_host") or source.get("submit_host"),
        "dataset_root": staging,
    }
    if not staging_location["host"]:
        raise SystemExit("slurm dataset sync requires submit_host for NAS staging")

    print(f"[dashboard] slurm-aware dataset sync: {source.get('name')} -> {target.get('name')}", flush=True)
    print(f"[dashboard] staging={staging}/{args.dataset_id}", flush=True)

    if source_slurm:
        slurm_copy_job(
            source,
            [["bash", "-lc", copy_to_staging_command(source, staging, args.dataset_id, True)]],
            ["export_to_nas"],
            args.dataset_id,
            "export",
        )
    else:
        transfer(source, staging_location, args.dataset_id, overwrite=True, parallelism=args.parallelism, label="stage to NAS")

    if target_slurm:
        slurm_copy_job(
            target,
            [["bash", "-lc", copy_from_staging_command(staging, target, args.dataset_id, args.overwrite, args.skip_existing)]],
            ["import_from_nas"],
            args.dataset_id,
            "import",
        )
    else:
        transfer(staging_location, target, args.dataset_id, overwrite=args.overwrite, parallelism=args.parallelism, label="fetch from NAS", skip_existing=args.skip_existing)

    print("[dashboard] slurm-aware dataset sync complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

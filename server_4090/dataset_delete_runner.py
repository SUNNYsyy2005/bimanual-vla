#!/usr/bin/env python3
"""Delete one configured dataset copy, using CPU-only Slurm for cluster storage."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from .slurm_dataset_sync_runner import inventory_refresh_snippet
except ImportError:
    from slurm_dataset_sync_runner import inventory_refresh_snippet


DELETE_SCRIPT = r'''
import json,os,pathlib,re,shutil,subprocess,sys,uuid
root=pathlib.Path(sys.argv[1]).expanduser().resolve(); dataset_id=sys.argv[2]
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',dataset_id): raise ValueError('invalid dataset id')
if root==pathlib.Path(root.anchor): raise ValueError('dataset root cannot be filesystem root')
target=root/dataset_id
if target.is_symlink() or target.resolve().parent!=root: raise ValueError('dataset path escapes configured root or is a symlink')
if not target.is_dir() or not (target/'meta/info.json').is_file(): raise FileNotFoundError('dataset copy not found: '+str(target))
if hasattr(os,'getuid'):
  uid=os.getuid()
  for current,dirs,files in os.walk(target,followlinks=False):
    for path in [pathlib.Path(current),*[pathlib.Path(current)/name for name in dirs+files]]:
      if path.lstat().st_uid!=uid: raise PermissionError('dataset contains files owned by another user: '+str(path))
# On compute nodes, inspect this user's other active batch scripts before deleting.
if os.environ.get('SLURM_JOB_ID'):
  queue=subprocess.run(['squeue','-h','-u',str(os.getuid()),'-o','%i'],capture_output=True,text=True,check=True)
  pattern=re.compile(r'(?<![A-Za-z0-9_.-])'+re.escape(dataset_id)+r'(?![A-Za-z0-9_.-])')
  for job in queue.stdout.split():
    if job==os.environ['SLURM_JOB_ID']: continue
    script=subprocess.run(['scontrol','write','batch_script',job,'-'],capture_output=True,text=True)
    if script.returncode: raise RuntimeError('could not verify dataset usage by Slurm job '+job)
    if pattern.search(script.stdout): raise ValueError('dataset is referenced by active Slurm job '+job)
trash=root/('.'+dataset_id+'.deleting-'+uuid.uuid4().hex)
os.replace(target,trash)
try: shutil.rmtree(trash)
except BaseException:
  if trash.exists() and not target.exists(): os.replace(trash,target)
  raise
print(json.dumps({'dataset_deleted':True,'dataset_id':dataset_id,'path':str(target)}),flush=True)
'''


def encoded(value):
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def deletion_command(location, dataset_id):
    python = location.get("openpi_python") or (sys.executable if location.get("kind") in {"local", "local_archive"} else None)
    if not python:
        raise ValueError("remote deletion requires a configured Conda Python")
    return [str(python), "-c", DELETE_SCRIPT, str(location["dataset_root"]), dataset_id]


def slurm_deletion_command(location, dataset_id):
    target = {key: value for key, value in location.items() if value is not None}
    target.update(gpu_count=0, cpus_per_task=2, mem="8G", time="02:00:00", workdir=str(location["dataset_root"]))
    target.setdefault("remote_job_dir", str(__import__("pathlib").PurePosixPath(location["dataset_root"]).parent / ".dashboard_delete_jobs"))
    target.setdefault("log_dir", target["remote_job_dir"])
    target.pop("conda_env", None)
    target.pop("conda_sh", None)
    refresh = inventory_refresh_snippet(location).replace("python3 - <<'PY'", shlex.quote(str(location["openpi_python"])) + " - <<'PY'")
    command = "set -euo pipefail\n" + shlex.join(deletion_command(location, dataset_id)) + "\n(\n" + refresh + "\n) || echo '[dashboard] node inventory refresh failed; local cache will be updated'"
    return [sys.executable, str(Path(__file__).with_name("slurm_job_runner.py")),
            "--target-json", encoded(target), "--commands-json", encoded([["bash", "-lc", command]]),
            "--command-labels-json", encoded(["delete_dataset_copy"]),
            "--job-name", "ddelete_" + dataset_id[:24] + "_" + __import__("uuid").uuid4().hex[:8], "--poll-interval", "10"]


def refresh_local_cache(location, dataset_id):
    cache = location.get("inventory_cache_path")
    if not cache or location.get("inventory_cache_host") not in {None, "local", "local_4090", "4x4090"}:
        return
    path = Path(cache).expanduser()
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["datasets"] = [row for row in payload.get("datasets", []) if row.get("id") != dataset_id]
    elif isinstance(payload, list):
        payload = [row for row in payload if row.get("id") != dataset_id]
    else:
        return
    temporary = path.with_suffix(path.suffix + ".delete.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--location-json", required=True)
    args = parser.parse_args()
    location = json.loads(base64.urlsafe_b64decode(args.location_json))
    if location.get("kind") in {"local", "local_archive"}:
        command = deletion_command(location, args.dataset_id)
    else:
        if not location.get("submit_host") or not location.get("partition") or not location.get("node"):
            raise ValueError("cluster deletion requires a configured Slurm target")
        command = slurm_deletion_command(location, args.dataset_id)
    status = subprocess.call(command)
    if status == 0:
        refresh_local_cache(location, args.dataset_id)
        print("[dashboard] dataset copy deleted: " + args.dataset_id + " @ " + location["name"], flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

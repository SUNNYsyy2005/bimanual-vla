#!/usr/bin/env python3
"""Create reviewed per-host plans from authenticated Dashboard inventories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

try:
    from . import eval_video_catalog as catalog
except ImportError:
    import eval_video_catalog as catalog


def inferred_task(text: str, explicit: str | None = None) -> str:
    if explicit:
        return catalog.normalize_task(explicit)
    match = re.search(r"((?:tsa|fshoa)[_-](?:moka|microwave|basket))(?:[-_/]|$)", text)
    if match:
        return catalog.normalize_task(match[1])
    if "handover_mic" in text or "handover-mic" in text:
        return "handover-mic"
    if "put_bottles_dustbin" in text:
        return "put-bottles-dustbin"
    return "unknown"


def storage_root(path: str) -> str:
    parts = Path(path).parts
    if len(parts) > 3 and parts[1] in {"mnt", "DATA"}:
        return str(Path(*parts[:4]) / "simulation-artifacts")
    return str(Path.home() / "simulation-artifacts")


def prepare(config: dict, rlinf: dict, status: dict) -> dict[str, list]:
    plans: dict[str, list] = {"local": []}
    sources = {s["id"]: s for s in config.get("rlinf_dashboard", {}).get("sources", [])}
    for run in rlinf.get("runs", []):
        if not run.get("path") or run.get("state") == "unavailable":
            continue
        source = sources.get(run["source_id"], {})
        host = source.get("ssh_host") or "local"
        path = run["path"]
        token = hashlib.sha256(path.encode()).hexdigest()[:12]
        experiment = catalog.slug(run["source_id"])
        if run["source_id"] == "h100_safe_rl":
            relative = Path(path).relative_to(source["root"])
            experiment = catalog.slug(relative.parts[0])
        plans.setdefault(host, []).append({
            "path": path, "artifact_root": storage_root(path), "backend": "rlinf", "kind": "rl",
            "video_source": "libero-safety", "task": inferred_task(path, run.get("task_name")),
            "experiment": experiment, "experiment_label": run.get("source_label"),
            "run_id": catalog.slug(Path(path).name)[:90] + "--" + token,
            "state": run["state"], "job_id": run.get("job_id"), "dashboard_id": run["id"],
            "storage_source": "rlinf-" + run["source_id"],
            "scope": "mirror_only" if "moka-h200-dashboard" in path else "full_run",
        })
    # Failed tasks and duplicate resumes are retained in the task history;
    # physical checkpoint directories are migrated once per location.
    seen = set()
    for task in status.get("tasks", []):
        meta = task.get("metadata", {})
        if task.get("type") != "train" or meta.get("framework") == "RLinf":
            continue
        candidates = [("local", meta.get("checkpoint_dir"))]
        cluster = config.get("cluster_targets", {}).get(meta.get("cluster_target"), {})
        if meta.get("remote_checkpoint_dir") and cluster.get("access_mode") != "slurm_only":
            candidates.append((cluster.get("submit_host", "login-server"), meta["remote_checkpoint_dir"]))
        for host, path in candidates:
            if not path or (host, path) in seen:
                continue
            seen.add((host, path))
            token = hashlib.sha256(path.encode()).hexdigest()[:12]
            plans.setdefault(host, []).append({
                "path": path, "artifact_root": storage_root(path), "backend": "openpi", "kind": "sft",
                "video_source": "robotwin" if inferred_task(path) != "unknown" else "unknown",
                "task": inferred_task(path), "experiment": catalog.slug(meta.get("exp_name", Path(path).name)),
                "run_id": catalog.slug(task["id"])[:90] + "--" + token, "state": task["state"],
                "dashboard_id": task["id"], "storage_source": host,
                "task_log": task.get("log_path"), "config_name": Path(path).parent.name,
            })
    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rlinf", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(os.path.expandvars(args.config.read_text()))
    plans = prepare(config, json.loads(args.rlinf.read_text()), json.loads(args.status.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for host, plan in plans.items():
        catalog.atomic_write(args.output_dir / (host + ".json"), plan)
    print(json.dumps({host: len(plan) for host, plan in plans.items()}))


if __name__ == "__main__":
    main()

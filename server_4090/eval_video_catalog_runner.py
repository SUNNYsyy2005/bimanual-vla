#!/usr/bin/env python3
"""Refresh video classification, retaining original files and event annotations."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

try:
    from . import eval_video_catalog as catalog
    from .eval_video_browser import publish_index
    from .dataset_transfer_runner import clean_transport_env
except ImportError:
    import eval_video_catalog as catalog
    from eval_video_browser import publish_index
    from dataset_transfer_runner import clean_transport_env


def organize_row(row: dict, destination: Path, registry: dict) -> str:
    original = Path(row["root"]) / row["relative_path"]
    key = str(original.absolute())
    previous = registry.get(key)
    target = destination / row["catalog_relative_path"]
    if previous and not (destination / previous["relative_path"]).exists():
        # An explicit Dashboard deletion stays deleted on future refreshes.
        return "deleted"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(original.absolute())
    elif not target.is_symlink() or target.resolve() != original.resolve():
        raise ValueError(f"catalog destination collision: {target}")
    original_override = original.parent / "_event_overrides" / (original.stem + ".json")
    target_override = target.parent / "_event_overrides" / (target.stem + ".json")
    if original_override.is_file() and not target_override.exists():
        target_override.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_override, target_override)
    if previous and previous["relative_path"] != row["catalog_relative_path"]:
        old = destination / previous["relative_path"]
        old_override = old.parent / "_event_overrides" / (old.stem + ".json")
        if old_override.is_file():
            target_override.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_override, target_override)
        old.unlink(missing_ok=True)
        catalog.metadata_path(old).unlink(missing_ok=True)
        old_override.unlink(missing_ok=True)
    catalog.atomic_write(catalog.metadata_path(target), row)
    registry[key] = {"relative_path": row["catalog_relative_path"], "size_bytes": row["size_bytes"]}
    return "indexed"


def refresh(config: dict, *, apply: bool, local_only: bool = False) -> dict:
    destination = catalog.catalog_root(config).expanduser().absolute()
    rows, errors = [], {}
    for root in dict.fromkeys(config.get("eval_video_roots", [])):
        root = Path(os.path.expandvars(root)).expanduser().absolute()
        if root == destination or destination in root.parents:
            continue
        rows.extend(catalog.scan_source({"root": str(root), "storage_source": "local_4090"}))
    sources = catalog.configured_sources(config)
    for source in sources:
        if not source.get("ssh_host"):
            rows.extend(catalog.scan_source(source))
    summary = {"local_videos": len(rows), "by_source": {}, "outcomes": {}, "unknown_tasks": 0}
    for row in rows:
        for field, key in (("video_source", "by_source"), ("success", "outcomes")):
            summary[key][row[field]] = summary[key].get(row[field], 0) + 1
        summary["unknown_tasks"] += row["task"] == "unknown"
    if not apply:
        return {**summary, "plan": [{k: r.get(k) for k in ("root", "relative_path", "catalog_relative_path", "video_source", "task", "experiment", "run_id", "phase", "success", "success_evidence")} for r in rows]}
    destination.mkdir(parents=True, exist_ok=True)
    registry = catalog.read_object(destination / "registry.json", max_bytes=128 * 1024 * 1024)
    indexed_rows = []
    for row in rows:
        if organize_row(row, destination, registry) == "indexed":
            indexed_rows.append({**row, "root": str(destination), "relative_path": row["catalog_relative_path"]})
    catalog.atomic_write(destination / "registry.json", registry)
    inventory_path = destination / "remote-inventory.json"
    old_inventory = catalog.read_object(inventory_path, max_bytes=128 * 1024 * 1024)
    by_source = old_inventory.get("sources", {})
    if not local_only:
        script = Path(catalog.__file__).read_text(encoding="utf-8")
        for source in sources:
            if not source.get("ssh_host"):
                continue
            # Only configured submit hosts; never SSH into a compute node.
            if source.get("access_mode") == "slurm_only":
                errors[source["storage_source"]] = "Slurm-only source requires staged artifacts"
                continue
            command = [str(source.get("python") or sys.executable), "-c", script, json.dumps(source)]
            try:
                result = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                         source["ssh_host"], shlex.join(command)], capture_output=True, text=True, timeout=150, env=clean_transport_env())
                if result.returncode:
                    raise RuntimeError(result.stderr[-1200:])
                remote = json.loads(result.stdout)
                for row in remote:
                    row["storage_source"] = source["storage_source"]
                by_source[source["storage_source"]] = {"updated_at": time.time(), "videos": remote}
            except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
                errors[source["storage_source"]] = str(exc)[-1200:]
        enabled = {s["storage_source"] for s in sources if s.get("ssh_host")}
        by_source = {k: v for k, v in by_source.items() if k in enabled}
        catalog.atomic_write(inventory_path, {"sources": by_source, "errors": errors, "updated_at": time.time()})
    publish_index(destination, indexed_rows, by_source)
    summary.update(remote_videos=sum(len(v["videos"]) for v in by_source.values()), errors=errors, updated_at=time.time())
    catalog.atomic_write(destination / "status.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(os.path.expandvars(args.config.read_text(encoding="utf-8")))
    lock_path = Path(config["workspace_root"]) / "eval_video_catalog.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("video catalog refresh already running", flush=True)
            return
        result = refresh(config, apply=args.apply, local_only=args.local_only)
        if args.output:
            catalog.atomic_write(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "plan"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

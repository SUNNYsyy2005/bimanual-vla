"""Backend-independent experiment manifests and reversible history migration.

No model imports, data conversion, decompression or weight copying is involved.
Physical migrations use same-filesystem rename and leave compatibility links.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import zipfile
from typing import Any

try:
    from . import eval_video_catalog as catalog
except ImportError:
    import eval_video_catalog as catalog

SCHEMA = "experiment_run.v1"
MIGRATION_REVISION = 3
ACTIVE_STATES = {"running", "starting", "waiting_gpu", "waiting_norm", "stopping", "pending", "unavailable"}


def identity(source: str, task: str, experiment: str, run: str) -> dict:
    values = {"video_source": source, "task": task, "experiment": experiment, "run_id": run}
    for key, value in values.items():
        if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,119}", value):
            raise ValueError(f"invalid artifact identity field: {key}")
    return values


def run_directory(root: Path | str, fields: dict) -> Path:
    checked = identity(*(fields[k] for k in ("video_source", "task", "experiment", "run_id")))
    return Path(root).joinpath(*(checked[k] for k in ("video_source", "task", "experiment", "run_id")))


def manifest(fields: dict, *, backend: str, kind: str, state: str, origin: dict | None = None) -> dict:
    identity(*(fields[k] for k in ("video_source", "task", "experiment", "run_id")))
    return {"schema": SCHEMA, **fields, "source": fields["video_source"], "backend": backend, "kind": kind, "state": state,
            "updated_at": time.time(), "origin": origin or {},
            "artifacts": {"metadata": "metadata", "logs": "logs", "metrics": "metrics",
                          "checkpoints": "checkpoints", "train": "train", "eval": "eval"}}


def record_task(config: dict, task: dict) -> dict:
    """Every trainer gets one common manifest, including failed/empty runs."""
    metadata = task["metadata"]
    supplied = metadata.get("artifact_identity") or {}
    if not isinstance(supplied, dict):
        raise ValueError("artifact_identity must be an object")
    source = supplied.get("video_source", "libero-safety" if metadata.get("framework") == "RLinf" else "unknown")
    task_name = supplied.get("task", "unknown")
    experiment = supplied.get("experiment", catalog.slug(metadata.get("exp_name") or metadata.get("run_name") or task["id"]))
    fields = identity(source, task_name, experiment, supplied.get("run_id", task["id"]))
    root = Path(config.get("experiment_artifact_root") or Path(config["workspace_root"]) / "experiment_artifacts")
    directory = run_directory(root, fields)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        allocated = catalog.read_object(directory / "run.json")
        if allocated.get("state") != "created" or allocated.get("origin", {}).get("task_id"):
            raise ValueError("artifact run already exists; use a new run_id")
        if allocated.get("backend") != ("rlinf" if metadata.get("framework") == "RLinf" else "openpi"):
            raise ValueError("allocated artifact run belongs to a different backend")
    try:
        with (directory / ".dashboard-task").open("x") as claim:
            claim.write(task["id"])
    except FileExistsError:
        raise ValueError("artifact run is already attached to a Dashboard task") from None
    result = manifest(fields, backend="rlinf" if metadata.get("framework") == "RLinf" else "openpi",
                      kind="rl" if metadata.get("framework") == "RLinf" else "sft", state=task["state"],
                      origin={"task_id": task["id"], "legacy_checkpoint_dir": metadata.get("checkpoint_dir"),
                              "config_name": metadata.get("config_name"), "model_variant": metadata.get("model_variant"),
                              "remote_checkpoint_dir": metadata.get("remote_checkpoint_dir"),
                              "rlinf_output_dir": metadata.get("rlinf_output_dir"),
                              "execution_target": metadata.get("execution_target")})
    result["task_scene"] = metadata.get("task_scene")
    result["action_contract"] = metadata.get("action_contract")
    result["dataset_ids"] = [metadata["dataset_id"]] if metadata.get("dataset_id") else []
    catalog.atomic_write(directory / "run.json", result)
    catalog.atomic_write(directory / "metadata" / "submission.json", metadata)
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / "dashboard.log"
    if not log.exists() and not log.is_symlink():
        log.symlink_to(task["log_path"])
    metadata.update(artifact_run_dir=str(directory), artifact_manifest=str(directory / "run.json"), artifact_identity=fields)
    return result


def move_with_alias(source: Path, target: Path) -> None:
    if source.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise ValueError(f"cross-filesystem migration requires a staged copy job: {source}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(str(target))
    os.rename(source, target)
    try:
        source.symlink_to(target.absolute(), target_is_directory=target.is_dir())
    except Exception:
        os.rename(target, source)
        raise


def preserve_action_contract(source: Path, target: Path, entry: dict) -> None:
    if entry["backend"] != "openpi":
        return
    destination = target / "metadata" / "action-contract.json"
    if destination.is_file():
        return
    candidates = [source.parent / ".policy_action_conventions" / f"{source.name}.json"]
    original = catalog.read_object(target / "run.json").get("origin", {})
    for key in ("physical_legacy_path", "legacy_path"):
        if original.get(key):
            path = Path(original[key])
            candidates.append(path.parent / ".policy_action_conventions" / f"{path.name}.json")
    for candidate in candidates:
        payload = catalog.read_object(candidate)
        if payload:
            catalog.atomic_write(destination, payload)
            return


def normalize_contents(directory: Path, fields: dict, backend: str) -> dict:
    """Move payloads, retaining every old relative path as an alias."""
    moved = {"episodes": 0, "checkpoints": 0, "metadata": 0}
    reserved = {"run.json", "_video_context.json", "metadata", "logs", "metrics", "checkpoints", "train", "eval", "artifacts"}
    for child in list(directory.iterdir()):
        if child.name in reserved or child.is_symlink():
            continue
        name = child.name
        if name.isdigit() or name.startswith("global_step_") or re.match(r"actor-round\d+\.pt$", name):
            target = directory / "checkpoints" / name
            moved["checkpoints"] += 1
        elif "curve" in name or name in {"metrics.jsonl", "metrics.log", "phase_events.jsonl", "sampling_timings.jsonl"}:
            target = directory / "metrics" / name
        elif child.suffix in {".log", ".out", ".err"}:
            target = directory / "logs" / name
        elif child.is_dir():
            target = directory / "artifacts" / name
        else:
            target = directory / "metadata" / name
            moved["metadata"] += 1
        move_with_alias(child, target)
    context = {**fields, "experiment_label": fields.get("experiment_label", fields["experiment"])}
    # Locate complete per-episode directories without opening trajectory/weight files.
    artifacts = directory / "artifacts"
    episodes = []
    if artifacts.exists():
        for current, dirs, files in os.walk(artifacts, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink() and d not in catalog.PRUNED]
            if "review.mp4" in files and "summary.json" in files:
                episodes.append(Path(current))
                dirs[:] = []
    for episode in episodes:
        rel = (episode / "review.mp4").relative_to(directory).as_posix()
        row = catalog.classify(directory, rel, context)
        stem = catalog.slug(episode.name)[:65]
        token = hashlib.sha256(rel.encode()).hexdigest()[:16]
        episode_id = f"{stem}--{token}"
        target = directory / row["phase"] / row["checkpoint_id"] / row["success"] / episode_id
        move_with_alias(episode, target)
        # The episode directory contains summary, metadata, trajectory refs,
        # videos and annotations together. Never detach video from evidence.
        row["provenance"] = {"storage_source": fields.get("storage_source", "local_4090"),
                             "root": str(directory), "relative_path": (target / "review.mp4").relative_to(directory).as_posix()}
        row["catalog_relative_path"] = catalog.catalog_relative_path(row, "review.mp4")
        catalog.atomic_write(catalog.metadata_path(target / "review.mp4"), row)
        catalog.atomic_write(target / "episode.json", {"schema": "rollout_episode.v1", **{k: v for k, v in row.items() if k not in {"episode_record", "schema"}}, "episode_id": episode_id})
        moved["episodes"] += 1
    checkpoints = []
    checkpoint_dir = directory / "checkpoints"
    if checkpoint_dir.exists():
        for child in checkpoint_dir.iterdir():
            if child.name == "index.json":
                continue
            orbax = backend == "openpi"
            weights = child if child.suffix == ".pt" else child / "actor/model_state_dict/full_weights.pt"
            complete = (child / "_CHECKPOINT_METADATA").is_file() if orbax else (
                weights.is_file() and zipfile.is_zipfile(weights))
            checkpoints.append({"id": child.name, "path": child.relative_to(directory).as_posix(),
                                "format": "orbax" if orbax else "pytorch", "complete": complete,
                                "restore_capabilities": [], "restore_capabilities_verified": False,
                                "completeness_evidence": "orbax_marker" if orbax else "torch_archive_header",
                                "integrity_verification": "structure_only"})
    catalog.atomic_write(checkpoint_dir / "index.json", {"schema": "checkpoint_index.v1", "checkpoints": checkpoints})
    return moved


def migrate(entry: dict, *, apply: bool) -> dict:
    source = Path(entry["path"]).absolute()
    fields = identity(entry["video_source"], entry["task"], entry["experiment"], entry["run_id"])
    target = run_directory(entry["artifact_root"], fields).absolute()
    result = {"source": str(source), "target": str(target), "state": entry.get("state"), "backend": entry["backend"]}
    if entry.get("state") in ACTIVE_STATES:
        return {**result, "action": "deferred_active"}
    if not source.exists():
        return {**result, "action": "missing"}
    if source.stat().st_uid != os.getuid():
        raise PermissionError(f"not owned by current user: {source}")
    if source.is_symlink() and source.resolve() == target.resolve():
        data = catalog.read_object(target / "run.json")
        if data.get("migration_revision", 0) >= MIGRATION_REVISION and data.get("schema") == SCHEMA:
            return {**result, "action": "already_migrated"}
        if apply:
            if data.get("schema") != SCHEMA:
                data = manifest(fields, backend=entry["backend"], kind=entry.get("kind", "rl"), state=entry.get("state", "unknown"),
                                origin={"legacy_path": str(source), "dashboard_id": entry.get("dashboard_id"), "scope": entry.get("scope", "full_run")})
            preserve_action_contract(source, target, entry)
            counts = normalize_contents(target, {**fields, "storage_source": entry.get("storage_source", "local_4090")}, entry["backend"])
            catalog.atomic_write(target / "run.json", {**data, "migration_complete": True, "migration_revision": MIGRATION_REVISION})
            return {**result, "action": "resumed", **counts}
        return {**result, "action": "resume"}
    # Resolve pre-existing storage links and replace that physical directory with
    # an alias too. Logical aliases elsewhere keep working without weight copies.
    physical = source.resolve()
    probe = target.parent
    while not probe.exists():
        probe = probe.parent
    if physical.stat().st_dev != probe.stat().st_dev:
        return {**result, "action": "deferred_cross_filesystem"}
    if not apply:
        return {**result, "action": "move"}
    target.parent.mkdir(parents=True, exist_ok=True)
    move_with_alias(physical, target)
    fields.update(experiment_label=entry.get("experiment_label", entry["experiment"]), storage_source=entry.get("storage_source", "local_4090"))
    data = manifest(fields, backend=entry["backend"], kind=entry.get("kind", "rl"), state=entry.get("state", "unknown"),
                    origin={"legacy_path": str(source), "physical_legacy_path": str(physical), "dashboard_id": entry.get("dashboard_id"),
                            "config_name": entry.get("config_name"),
                            "scope": entry.get("scope", "full_run"), "job_id": entry.get("job_id")})
    catalog.atomic_write(target / "run.json", data)
    preserve_action_contract(source, target, entry)
    catalog.atomic_write(target / "_video_context.json", {k: fields[k] for k in ("video_source", "task", "experiment", "run_id", "experiment_label", "storage_source")})
    counts = normalize_contents(target, fields, entry["backend"])
    catalog.atomic_write(target / "run.json", {**data, "migration_complete": True, "migration_revision": MIGRATION_REVISION})
    return {**result, "action": "moved", **counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # Slurm allocations can expose more CPUs than requested; protect against
    # accidental duplicate srun ranks and separate invocations of the same plan.
    lock = args.plan.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    plan = json.loads(args.plan.read_text())
    report = []
    for entry in plan:
        try:
            report.append(migrate(entry, apply=args.apply))
        except Exception as exc:
            report.append({"source": entry["path"], "action": "error", "error": str(exc)})
        # Persist after each run so interruption never loses the migration audit.
        catalog.atomic_write(args.report, {"updated_at": time.time(), "runs": report})
    counts = {}
    for item in report:
        counts[item["action"]] = counts.get(item["action"], 0) + 1
    print(json.dumps(counts), flush=True)
    if counts.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

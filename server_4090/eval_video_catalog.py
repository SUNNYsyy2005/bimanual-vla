"""Shared video taxonomy and lossless indexing of simulation rollout artifacts.

This module uses only the standard library so the same scanner can run through
the submit host's Conda Python without importing any simulation dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
from typing import Any

SCHEMA = "evaluation_video.v1"
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}
PRUNED = {".git", "__pycache__", "checkpoints", "tensorboard", "wandb", "canonical_policy_origins"}
TASK_ALIASES = {
    ("obstacle_avoidance", 1, 2): "tsa-moka",
    ("obstacle_avoidance", 1, 3): "tsa-basket",
    ("obstacle_avoidance", 1, 4): "tsa-microwave",
    ("obstacle_avoidance_human", 1, 4): "fshoa-microwave",
}


def read_object(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> dict:
    try:
        if path.stat().st_size > max_bytes:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def slug(value: Any, default: str = "unknown") -> str:
    text = re.sub(r"[^a-z0-9._-]+", "-", str(value or "").lower()).strip(".-_")
    return text[:120] or default


def normalize_task(value: Any) -> str:
    return slug(str(value or "").replace("_", "-"))


def outcome(value: Any) -> str:
    if value is True or (type(value) in (int, float) and value == 1):
        return "success"
    if value is False or (type(value) in (int, float) and value == 0):
        return "failed"
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "success", "succeeded"}:
            return "success"
        if value in {"false", "failure", "failed", "policy_error", "expert_failed", "expert_invalid", "expert_unstable"}:
            return "failed"
    return "unknown"


def episode_record(path: Path) -> tuple[dict, str | None]:
    """Match one episode, never broadcast a run-level score to every video."""
    match = re.fullmatch(r"episode[-_]?(\d+)\.(?:mp4|webm|mov|mkv|avi|gif)", path.name, re.I)
    if not match:
        return {}, None
    candidates = [path.parent / "_episode_results.jsonl", *path.parent.glob("_episode_results.rescore*.jsonl")]
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    fallback = ({}, None)
    for candidate in candidates:
        found = {}
        try:
            with candidate.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if int(row.get("episode_index", -1)) == int(match[1]) and row.get("counted_in_success_rate", True):
                            found = row
                    except (ValueError, TypeError, AttributeError):
                        continue
        except OSError:
            continue
        if found:
            if not fallback[0]:
                fallback = found, candidate.name
            if found.get("event_timeline") or "current_event" in found or "max_event_reached" in found:
                return found, candidate.name
    return fallback


def metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def classify(root: Path | str, relative_path: str, context: dict | None = None) -> dict:
    context = context or {}
    root = Path(root)
    path = root / relative_path
    launch_args = {}
    for parent in (path.parent, *path.parents):
        if parent != root and root not in parent.parents:
            break
        inherited = read_object(parent / "_video_context.json")
        if inherited:
            context = {**context, **inherited}
            break
        launch = read_object(parent / "_launch_context.json").get("args", {})
        if isinstance(launch, dict) and launch:
            launch_args = launch
            if launch.get("task_name"):
                context = {**context, "task": launch["task_name"]}
            break
    stored = read_object(metadata_path(path))
    if stored.get("schema") == SCHEMA:
        if stored.get("task") in {"unknown", "put-both-moka-pots-on-the-stove"}:
            original = read_object(path.parent / "metadata.json")
            key = (original.get("suite"), original.get("level"), original.get("task"))
            alias = TASK_ALIASES.get(key) if all(isinstance(v, (str, int, type(None))) for v in key) else None
            if alias:
                stored = {**stored, "task": alias}
                stored["catalog_relative_path"] = catalog_relative_path(stored, stored.get("name", path.name))
        return stored
    parts = PurePosixPath(relative_path).parts
    text = "/".join((str(root), relative_path)).lower()
    row, sidecar = episode_record(path)
    summary = read_object(path.parent / "summary.json") if path.name == "review.mp4" else {}
    detail = read_object(path.parent / "metadata.json") if summary or path.name == "review.mp4" else {}
    # summary.json is per-episode only for the native review.mp4 layout.
    item = {**{k: launch_args[k] for k in ("model_name", "checkpoint_id", "task_name") if k in launch_args}, **detail, **summary, **row}
    text += " " + " ".join(str(item.get(k, "")) for k in ("checkpoint_id", "model_name", "instruction"))
    video_source = slug(context.get("video_source"))
    if video_source == "unknown":
        if summary or detail.get("bddl") or "libero" in text:
            video_source = "libero-safety"
        elif row or "robotwin" in text or any(t in text for t in ("handover_mic", "put_bottles", "tsq")):
            video_source = "robotwin"
    raw_task = item.get("task_name") or item.get("task")
    task = normalize_task(context.get("task"))
    if task == "unknown":
        key = (item.get("suite"), item.get("level"), item.get("task"))
        registered = TASK_ALIASES.get(key) if all(isinstance(v, (str, int, type(None))) for v in key) else None
        alias = re.search(r"(?:^|/)((?:tsa|fshoa|tsd|fsha)[_-](?:moka|microwave|basket))(?:[-_/]|$)", text)
        if registered:
            task = registered
        elif alias:
            task = normalize_task(alias[1])
        elif isinstance(raw_task, str) and raw_task:
            task = normalize_task(raw_task)
        else:
            for known in ("handover_mic", "put_bottles_dustbin", "put_bottles"):
                if known in text or known.replace("_", "-") in text:
                    task = normalize_task(known)
                    break
    phase_match = next((p for p in parts[:-1] if re.match(r"(?:train|eval)[-_]round\d+", p)), None)
    phase = context.get("phase")
    if phase not in {"train", "eval", "unknown"}:
        if phase_match:
            phase = "train" if phase_match.startswith("train") else "eval"
        elif str(item.get("split", "")).lower() in {"train", "training"} or "training_rollout" in text:
            phase = "train"
        elif video_source == "robotwin" or any("eval" in p.lower() for p in parts[:-1]):
            phase = "eval"
        else:
            phase = "unknown"
    experiment = context.get("experiment")
    run_id = context.get("run_id")
    if not experiment:
        if parts and parts[0] == "dashboard_online_eval" and len(parts) > 2:
            experiment = parts[1]
            run_id = run_id or parts[2]
        elif len(parts) >= 4 and parts[1] in {"pi05", "pi0", "act", "diffusion_policy"}:
            experiment = parts[2]
        else:
            experiment = parts[0] if len(parts) > 1 else root.name
    if not run_id:
        if phase_match:
            before = parts[:parts.index(phase_match)]
            run_id = before[-1] if before else root.name
        else:
            timestamps = [p for p in parts[:-1] if re.match(r"\d{4}[-_]?(?:\d{2}|\d{2}-\d{2})", p)]
            run_id = timestamps[-1] if timestamps else experiment
    success = outcome(item.get("success"))
    evidence = sidecar if row and success != "unknown" else "summary.json" if summary and success != "unknown" else None
    if success == "unknown" and row:
        success = outcome(row.get("status"))
        if success != "unknown":
            evidence = sidecar
    # Only whole path components count, never an experiment named 'success-test'.
    if success == "unknown" and path.parent.name.lower() in {"success", "failed", "failure"}:
        success = outcome(path.parent.name.lower())
        evidence = "outcome-directory"
    checkpoint = item.get("checkpoint_id") or context.get("checkpoint_id")
    if not checkpoint:
        checkpoint_match = re.search(r"(?:ckpt|cp|checkpoint|global_step)[-_]?(\d+)", text)
        checkpoint = f"step-{int(checkpoint_match[1])}" if checkpoint_match else "unknown"
    round_match = re.search(r"(?:train|eval)[-_]round(\d+)", phase_match or "")
    provenance = {"root": str(root), "relative_path": relative_path, "storage_source": context.get("storage_source", "local_4090")}
    result = {
        "schema": SCHEMA, "video_source": video_source, "task": task,
        "raw_task": raw_task, "experiment": slug(experiment),
        "experiment_label": context.get("experiment_label") or str(experiment),
        "run_id": slug(run_id), "phase": phase, "checkpoint_id": slug(checkpoint),
        "rollout_round": int(round_match[1]) if round_match else None,
        "success": success, "success_evidence": evidence,
        "safe_success": outcome(item.get("safe_success")),
        "episode_status": item.get("status"), "episode_seed": item.get("seed"),
        "instruction": item.get("instruction"), "model_name": item.get("model_name"),
        "video_layout": item.get("video_layout"), "video_views": item.get("video_views"),
        "score": item.get("stage_reward"), "episode_record": row,
        "has_events": bool(row.get("event_timeline") or "current_event" in row or "max_event_reached" in row),
        "episode_scope": "episode" if row or summary else "unverified",
        "episode_sidecar": sidecar, "provenance": provenance,
        "display_name": path.name,
    }
    result["catalog_relative_path"] = catalog_relative_path(result, path.name)
    return result


def catalog_relative_path(metadata: dict, name: str) -> str:
    levels = [metadata.get(k) for k in ("video_source", "task", "experiment", "run_id", "phase", "checkpoint_id", "success")]
    provenance = metadata.get("provenance", {})
    identity = "\0".join(str(provenance.get(k, "")) for k in ("storage_source", "root", "relative_path"))
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    filename = slug(Path(name).stem)[:70] + "--" + suffix + Path(name).suffix.lower()
    return PurePosixPath(*(slug(value) for value in levels), filename).as_posix()


def walk_videos(root: Path, *, max_depth: int = 18):
    seen = set()
    for current, dirs, files in os.walk(root, followlinks=True):
        resolved = Path(current).resolve()
        if resolved in seen:
            dirs[:] = []
            continue
        seen.add(resolved)
        relative = Path(current).relative_to(root)
        dirs[:] = [d for d in dirs if d not in PRUNED and not d.startswith(".")]
        dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink() or
                   read_object(Path(current) / d / "run.json").get("schema") == "experiment_run.v1"]
        if len(relative.parts) >= max_depth:
            dirs[:] = []
        for name in sorted(files):
            path = Path(current) / name
            if path.suffix.lower() in VIDEO_SUFFIXES:
                try:
                    if path.is_file() and path.stat().st_size:
                        yield path
                except OSError:
                    continue


def scan_source(source: dict) -> list[dict]:
    root = Path(source["root"]).expanduser()
    rows = []
    for path in walk_videos(root, max_depth=int(source.get("video_max_depth", 18))):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        if source.get("stable_seconds", 5) and __import__("time").time() - stat.st_mtime < source.get("stable_seconds", 5):
            continue
        metadata = classify(root, rel, source)
        rows.append({**metadata, "name": path.name, "root": str(root), "relative_path": rel,
                     "size_mib": round(stat.st_size / 1024**2, 2), "size_bytes": stat.st_size,
                     "mtime": stat.st_mtime,
                     "file_identity": dict(host=socket.gethostname(), device=stat.st_dev, inode=stat.st_ino,
                                           size=stat.st_size, mtime_ns=stat.st_mtime_ns)})
    return rows


def configured_sources(config: dict) -> list[dict]:
    result = []
    for item in config.get("rlinf_dashboard", {}).get("sources", []):
        if item.get("enabled", True) and item.get("root"):
            result.append({**item, "video_source": item.get("video_source", "libero-safety"),
                           "experiment": item.get("experiment", item["id"]),
                           "experiment_label": item.get("label", item["id"]),
                           "storage_source": "rlinf-" + item["id"]})
    return result


def catalog_root(config: dict) -> Path:
    return Path(config.get("eval_video_catalog_root") or Path(config["workspace_root"]) / "eval_video_catalog")


if __name__ == "__main__":
    import sys
    # Also used as the stdin script on login-server. No simulation imports/jobs.
    print(json.dumps(scan_source(json.loads(sys.argv[1])), ensure_ascii=False))

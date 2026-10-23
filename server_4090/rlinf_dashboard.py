#!/usr/bin/env python3
"""Run discovery, metric normalization, and safe launch planning for RLinf jobs."""

from __future__ import annotations

import base64
import copy
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


def native_profile_fields() -> dict[str, Any]:
    return {
        "updates": {"type": "int", "label": "训练更新步数", "default": 1, "min": 1, "max": 1000000},
        "learning_rate": {"type": "float", "label": "Actor 学习率", "default": 5e-6, "min": 1e-9, "max": .01},
        "save_interval": {"type": "int", "label": "模型保存间隔（step）", "default": 1, "min": 1, "max": 1000000},
        "eval_interval": {"type": "int", "label": "训练中评测间隔（step；-1 关闭）", "default": -1, "min": -1, "max": 1000000},
        "group_size": {"type": "int", "label": "每组采样个数", "default": 2, "min": 2, "max": 32},
        "num_groups": {"type": "int", "label": "并行采样组数", "default": 1, "min": 1, "max": 64},
        "update_epoch": {"type": "int", "label": "每批更新轮数", "default": 1, "min": 1, "max": 32},
        "micro_batch_size": {"type": "int", "label": "Micro batch size", "default": 1, "min": 1, "max": 64},
        "seed": {"type": "int", "label": "随机种子", "default": 43, "min": 0, "max": 2147483647},
        "rollout_steps": {"type": "int", "label": "Rollout 最大环境步数", "default": 20, "min": 1, "max": 2000},
        "risk_budget": {"type": "float", "label": "GRPO-Lag 风险预算", "default": .05, "min": 0, "max": 1},
        "dual_lr": {"type": "float", "label": "GRPO-Lag 对偶学习率", "default": .01, "min": 1e-9, "max": 1},
    }


NATIVE_TRAIN_OVERRIDES = [
    "runner.max_epochs={updates}", "runner.max_steps={updates}", "actor.optim.lr={learning_rate}",
    "runner.save_interval={save_interval}", "runner.val_check_interval={eval_interval}",
    "algorithm.group_size={group_size}", "env.train.total_num_envs={total_envs}",
    "actor.global_batch_size={total_envs}", "actor.micro_batch_size={micro_batch_size}",
    "algorithm.update_epoch={update_epoch}", "algorithm.safety.enabled={lag_enabled}",
    "algorithm.safety.budget={risk_budget}", "algorithm.safety.dual_lr={dual_lr}",
    "env.train.seed={seed}", "env.eval.seed={seed}",
    "env.train.max_episode_steps={rollout_steps}", "env.train.max_steps_per_rollout_epoch={rollout_steps}",
]


def native_gpu_placement(gpu_ids: list[int], *, evaluation: bool = False) -> list[str]:
    if not gpu_ids:
        raise ValueError("native RLInf requires at least one selected GPU")
    ranks = ",".join(str(int(value)) for value in gpu_ids)
    fields = ["env", "rollout"] if evaluation else ["actor", "env", "rollout"]
    placement = "{" + ",".join(f"{field}:'{ranks}'" for field in fields) + "}"
    return ["~cluster.component_placement", "+cluster.component_placement=" + placement]


def upgrade_native_profiles(config: dict[str, Any]) -> dict[str, Any]:
    """Extend existing native Hydra profiles without changing their paths/environment."""
    result = copy.deepcopy(config)
    for target in result.get("rlinf_dashboard", {}).get("targets", {}).values():
        for profile in target.get("profiles", {}).values():
            if profile.get("backend") != "rlinf" and "train_embodied_agent.py" not in " ".join(profile.get("command_template", [])):
                continue
            profile["backend"] = "rlinf"
            fields = native_profile_fields()
            fields.update(profile.get("overrides", {}))
            profile["overrides"] = fields
            profile.setdefault("evaluation", {"backend": "rlinf", "config_name": "libero_safety_grpo_pi05_smoke"})
            template = profile.get("command_template", [])
            # Existing deployments used a bash command; append only missing fields.
            if len(template) == 3 and template[:2] == ["bash", "-lc"]:
                for override in NATIVE_TRAIN_OVERRIDES:
                    name = override.split("=", 1)[0]
                    if name + "=" not in template[2]:
                        template[2] += " " + override
            elif template:
                for override in NATIVE_TRAIN_OVERRIDES:
                    name = override.split("=", 1)[0]
                    if not any(str(item).startswith(name + "=") for item in template):
                        template.append(override)
    return result


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CURVE_NAMES = {"training_curve.json", "evaluation_curve.json"}
REMOTE_SCAN = r'''import json, os, pathlib, sys, subprocess
root=pathlib.Path(sys.argv[1]); limit=int(sys.argv[2]); parse_events=len(sys.argv)>3 and sys.argv[3]=='1'
curves={}; events={}; tracked={}; statuses={}; metadata={}; job_states={}
tracked_suffixes=('.log','.out','.err')
tracked_names={'training_curve.json','evaluation_curve.json','process.json','resource_samples.jsonl','metrics.log','protocol.json','completion.json','monitor_phase.json'}
if root.is_dir():
  for current, dirs, files in os.walk(root):
    rel=pathlib.Path(current).relative_to(root)
    if len(rel.parts)>=limit: dirs[:]=[]
    dirs[:]=[d for d in dirs if not d.startswith('.')]
    meta={}
    for name in ('protocol.json','completion.json','config.json','monitor_phase.json'):
      if name not in files: continue
      try:
        path=pathlib.Path(current)/name
        if path.stat().st_size<=8*1024*1024:
          value=json.loads(path.read_text(encoding='utf-8'))
          if isinstance(value,dict): meta[name]=value
      except (OSError,ValueError): pass
    if meta: metadata[rel.as_posix()]=meta
    for name in files:
      if name not in tracked_names and not name.endswith(tracked_suffixes): continue
      path=pathlib.Path(current)/name
      try: tracked[str(path.relative_to(root))]=path.stat().st_mtime
      except OSError: pass
    for name in ('training_curve.json','evaluation_curve.json'):
      if name not in files: continue
      path=pathlib.Path(current)/name
      try:
        if path.stat().st_size>8*1024*1024: continue
        curves[str(path.relative_to(root))]=json.loads(path.read_text(encoding='utf-8'))
      except Exception: pass
    if 'process.json' in files:
      path=pathlib.Path(current)/'process.json'
      try:
        if path.stat().st_size<=8*1024*1024:
          process=json.loads(path.read_text(encoding='utf-8'))
          statuses[rel.as_posix()]={k:process.get(k) for k in ('status','job_id','exported_unix','phase','completed_unix')}
          source_times=[p.get('mtime') for p in process.get('provenance',[]) if isinstance(p,dict) and isinstance(p.get('mtime'),(int,float))]
          statuses[rel.as_posix()]['source_updated_unix']=process.get('source_updated_unix') or max(source_times,default=None)
          for field, name in (('training_curve','training_curve.json'),('evaluation_curve','evaluation_curve.json')):
            if isinstance(process.get(field), list):
              curves.setdefault((pathlib.Path(current).relative_to(root)/name).as_posix(), {'points':process[field]})
      except Exception: pass
    event_files=[pathlib.Path(current)/name for name in files if name.startswith('events.out.tfevents.')]
    if event_files:
      run_rel=rel.parent if rel.name=='tensorboard' else rel
      event_key=run_rel.as_posix()
      events[event_key]={'files':[str(path.relative_to(root)) for path in event_files],
                         'updated_at':max(path.stat().st_mtime for path in event_files)}
    if event_files and parse_events:
      try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        points={}; eval_points={}
        for event_file in sorted(event_files, key=lambda item: item.stat().st_mtime):
          accumulator=EventAccumulator(str(event_file), size_guidance={'scalars': 0})
          accumulator.Reload()
          for tag in accumulator.Tags().get('scalars', []):
            destination=eval_points if tag.startswith('eval/') else points
            for scalar in accumulator.Scalars(tag):
              point=destination.setdefault(int(scalar.step), {'actor_updates': int(scalar.step)})
              point[tag[5:] if tag.startswith('eval/') else tag]=float(scalar.value)
        if points:
          key=(run_rel/'training_curve.json').as_posix()
          curves.setdefault(key, {'points':[points[step] for step in sorted(points)]})
        if eval_points:
          key=(run_rel/'evaluation_curve.json').as_posix()
          curves.setdefault(key, {'points':[eval_points[step] for step in sorted(eval_points)]})
      except Exception: pass
if len(sys.argv)>4 and sys.argv[4]=='1':
  ids={str(m.get('protocol.json',{}).get('job_id','')) for m in metadata.values()}
  ids={v for v in ids if v and all(c in '0123456789_' for c in v)}
  if ids:
    try:
      # Query the user's queue: one expired ID must not invalidate all active IDs.
      import getpass
      result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%A|%i|%T'],capture_output=True,text=True,timeout=8)
      if result.returncode==0:
        job_states={job:'NOT_IN_QUEUE' for job in ids}
        for line in result.stdout.splitlines():
          fields=line.strip().split('|')
          if len(fields)==3:
            for job in fields[:2]:
              if job in ids: job_states[job]=fields[2]
    except (OSError,subprocess.TimeoutExpired): pass
    missing=ids-{j for j,s in job_states.items() if s!='NOT_IN_QUEUE'}
    if missing:
      try:
        result=subprocess.run(['sacct','-X','-n','-P','-S','1970-01-01','-j',','.join(sorted(missing)),
                               '-o','JobIDRaw,JobID,State'],capture_output=True,text=True,timeout=8)
        if result.returncode==0:
          for line in result.stdout.splitlines():
            fields=line.strip().split('|')
            if len(fields)>=3 and '.' not in fields[0]:
              for job in fields[:2]:
                if job in missing: job_states[job]=fields[2].split()[0].rstrip('+')
      except (OSError,subprocess.TimeoutExpired): pass
print(json.dumps({'curves':curves,'events':events,'tracked':tracked,'statuses':statuses,'metadata':metadata,'job_states':job_states},separators=(',',':')))
'''


def _local_metadata(root: Path, max_depth: int) -> dict[str, Any]:
    result = {}
    for current, dirs, files in os.walk(root):
        rel = Path(current).relative_to(root)
        if len(rel.parts) >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        meta = {}
        for name in ('protocol.json', 'completion.json', 'config.json', 'monitor_phase.json'):
            if name not in files:
                continue
            path = Path(current) / name
            try:
                if path.stat().st_size <= 8*1024*1024:
                    value = json.loads(path.read_text(encoding='utf-8'))
                    if isinstance(value, dict):
                        meta[name] = value
            except (OSError, ValueError):
                pass
        if meta:
            result[rel.as_posix()] = meta
    return result


def _curve_points(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("points", value.get("records", value.get("curve", [])))
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        # Expose small numerical diagnostics (e.g. PPO critic) without groups/actions.
        expanded = dict(item)
        for namespace in ("critic", "ppo_advantage"):
            nested = item.get(namespace)
            if isinstance(nested, dict):
                expanded.update({f"{namespace}.{key}": value for key, value in nested.items()})
        for key, raw in expanded.items():
            if key == "groups" or isinstance(raw, (dict, list, tuple)):
                continue
            if isinstance(raw, bool) or raw is None:
                continue
            if isinstance(raw, (int, float)):
                number = float(raw)
                if math.isfinite(number):
                    row[str(key)] = number
            elif isinstance(raw, str) and key in {"sampler", "method", "algorithm", "cohort"}:
                row[str(key)] = raw
        if row:
            rows.append(row)
    return rows


def _local_curves(root: Path, max_depth: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not root.is_dir():
        return result
    for current, dirs, files in os.walk(root):
        rel = Path(current).relative_to(root)
        if len(rel.parts) >= max_depth:
            dirs[:] = []
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in CURVE_NAMES:
            if name not in files:
                continue
            path = Path(current) / name
            try:
                if path.stat().st_size > 8 * 1024 * 1024:
                    continue
                result[path.relative_to(root).as_posix()] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        if "process.json" in files:
            path = Path(current) / "process.json"
            try:
                if path.stat().st_size <= 8 * 1024 * 1024:
                    process = json.loads(path.read_text(encoding="utf-8"))
                    relative = path.parent.relative_to(root)
                    for field, name in (("training_curve", "training_curve.json"),
                                        ("evaluation_curve", "evaluation_curve.json")):
                        if isinstance(process.get(field), list):
                            key = (relative / name).as_posix()
                            result.setdefault(key, {"points": process[field]})
            except (OSError, ValueError):
                continue
    return result


def _local_tracked_mtimes(root: Path, max_depth: int) -> dict[str, float]:
    """Return mtimes for files that indicate an RLInf run is still changing."""
    result: dict[str, float] = {}
    if not root.is_dir():
        return result
    tracked_names = {"training_curve.json", "evaluation_curve.json", "process.json",
                     "resource_samples.jsonl", "metrics.log", "protocol.json", "completion.json", "monitor_phase.json"}
    for current, dirs, files in os.walk(root):
        rel = Path(current).relative_to(root)
        if len(rel.parts) >= max_depth:
            dirs[:] = []
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in files:
            if name not in tracked_names and not name.endswith((".log", ".out", ".err")):
                continue
            path = Path(current) / name
            try:
                result[path.relative_to(root).as_posix()] = path.stat().st_mtime
            except OSError:
                continue
    return result


def _run_updated_at(relative: str, tracked: dict[str, Any]) -> float | None:
    """Find the newest relevant file, including the sibling ``<run>.log``."""
    relative = relative.strip("/")
    if not relative or relative == ".":
        candidates = list(tracked.values())
    else:
        prefix = relative + "/"
        parent = str(Path(relative).parent).replace("\\", "/")
        leaf = Path(relative).name
        sibling_prefix = (parent + "/") if parent not in ("", ".") else ""
        candidates = [value for path, value in tracked.items()
                      if path.startswith(prefix)
                      or path in {f"{sibling_prefix}{leaf}.log",
                                  f"{sibling_prefix}{leaf}.out",
                                  f"{sibling_prefix}{leaf}.err"}]
    finite = [float(value) for value in candidates if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return max(finite, default=None)


def _scan_root(source: dict[str, Any], root: str, max_depth: int, *, parse_events: bool = False) -> dict[str, Any]:
    python = str(source.get("python") or "python3")
    query_slurm = source.get("query_slurm", bool(source.get("ssh_host")) and str(source.get("target", "")).lower() in {"h100", "h200"})
    flags = ["1" if parse_events else "0", "1" if query_slurm else "0"]
    command = shlex.join([python, "-c", REMOTE_SCAN, root, str(max_depth), *flags])
    if source.get("ssh_host"):
        host = str(source.get("ssh_host") or "").strip()
        if not host:
            raise ValueError("remote RLinf metric source is missing ssh_host")
        result = subprocess.run(
            ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
            capture_output=True,
            text=True,
            timeout=35,
            check=False,
        )
    else:
        result = subprocess.run(
            [python, "-c", REMOTE_SCAN, root, str(max_depth), *flags],
            capture_output=True, text=True, timeout=35, check=False,
        )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "remote metric scan failed").strip()[-1200:])
    value = json.loads(result.stdout)
    return value if isinstance(value, dict) else {"curves": {}, "events": {}}


def _sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    block = config.get("rlinf_dashboard", {})
    return [item for item in block.get("sources", []) if isinstance(item, dict) and item.get("enabled", True)]


def scan_runs(config: dict[str, Any], tasks: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    launched = tasks or []
    active_by_output = {
        str(task.get("metadata", {}).get("rlinf_output_dir", "")): task
        for task in launched if task.get("type") == "train" and task.get("metadata", {}).get("rlinf_output_dir")
    }
    runs: list[dict[str, Any]] = []
    for source in _sources(config):
        source_id = str(source.get("id") or "")
        root = str(source.get("root") or "")
        if not SAFE_ID.fullmatch(source_id) or not root:
            continue
        depth = max(1, min(8, int(source.get("max_depth", 5))))
        statuses, metadata, job_states = {}, {}, {}
        try:
            if source.get("ssh_host") or source.get("python"):
                scan = _scan_root(source, root, depth, parse_events=False)
                curves = scan.get("curves", {})
                event_runs = scan.get("events", {})
                tracked_mtimes = scan.get("tracked", {})
                statuses = scan.get("statuses", {})
                metadata = scan.get("metadata", {})
                job_states = scan.get("job_states", {})
            else:
                curves = _local_curves(Path(root), depth)
                event_runs = {}
                tracked_mtimes = _local_tracked_mtimes(Path(root), depth)
                metadata = _local_metadata(Path(root), depth)
            if not source.get("ssh_host") and not Path(root).is_dir():
                raise FileNotFoundError(f"configured curve root does not exist: {root}")
            source_error = None
        except Exception as exc:
            curves, event_runs, tracked_mtimes, source_error = {}, {}, {}, str(exc)
        grouped: dict[str, dict[str, Any]] = {}
        for relative, raw in curves.items():
            path = Path(relative)
            kind = "training" if path.name == "training_curve.json" else "evaluation"
            key = path.parent.as_posix()
            entry = grouped.setdefault(key, {"training": [], "evaluation": [], "files": []})
            points = _curve_points(raw)
            if kind == "evaluation":
                entry["evaluation_all"] = points
            cohort = source.get("evaluation_cohort")
            if kind == "evaluation" and cohort:
                # Never let extra-development points overwrite fixed VAL at the same x.
                points = [point for point in points if point.get("cohort") == cohort]
            entry[kind] = points
            entry["files"].append(relative)
        for relative, meta in metadata.items():
            proto = meta.get('protocol.json', {})
            candidate = meta.get('config.json') or proto.get('config') or {}
            if proto and (proto.get('algorithm') or (isinstance(candidate, dict) and isinstance(candidate.get('algorithm'), str))):
                grouped.setdefault(relative, {"training": [], "evaluation": [], "files": []})
        for relative, event_info in event_runs.items():
            entry = grouped.setdefault(relative or ".", {"training": [], "evaluation": [], "files": []})
            entry["event_files"] = event_info.get("files", [])
            entry["event_updated_at"] = event_info.get("updated_at")
        # Include submitted/failed jobs even before their first metric is written.
        for output, task in active_by_output.items():
            if output.startswith(root.rstrip("/") + "/"):
                relative = output[len(root.rstrip("/")) + 1:]
                if not any(item == relative or item.startswith(relative + "/") for item in grouped):
                    grouped.setdefault(relative, {"training": [], "evaluation": [], "files": []})
        for relative, curves_for_run in grouped.items():
            run_path = root.rstrip("/") + ("/" + relative if relative != "." else "")
            task = next((item for output, item in active_by_output.items() if run_path == output or run_path.startswith(output.rstrip("/") + "/")), None)
            meta = metadata.get(relative, {})
            protocol = meta.get('protocol.json', {})
            cfg = meta.get('config.json') or protocol.get('config') or {}
            if not isinstance(cfg, dict):
                cfg = {}
            completion = meta.get('completion.json', {})
            phase = meta.get('monitor_phase.json') or metadata.get(Path(relative).parent.as_posix(), {}).get('monitor_phase.json', {})
            if phase.get('output') and phase['output'].replace('\\', '/').rstrip('/') != run_path.replace('\\', '/').rstrip('/'):
                phase = {}
            if not curves_for_run["training"] and not curves_for_run["evaluation"] and not curves_for_run.get("event_files") and not task and not protocol:
                continue
            stat_updated = _run_updated_at(relative, tracked_mtimes)
            event_updated = curves_for_run.get("event_updated_at")
            if event_updated is not None:
                stat_updated = max(float(event_updated), float(stat_updated or 0))
            all_eval_points = curves_for_run.get("evaluation_all", curves_for_run["evaluation"])
            cohorts = sorted({point.get('cohort', '') for point in all_eval_points})
            cohort = source.get('evaluation_cohort') or ('fixed_validation' if 'fixed_validation' in cohorts else (cohorts[0] if cohorts else None))
            eval_points = [point for point in all_eval_points if not cohort or point.get('cohort') == cohort]
            eval_series: dict[str, list[dict[str, Any]]] = {}
            for point in eval_points:
                sampler = point.get("sampler")
                for key, value in point.items():
                    if key in {"round", "actor_updates", "sampler", "cohort"}:
                        continue
                    series_key = f"eval.{key}@{sampler}" if sampler else f"eval.{key}"
                    eval_series.setdefault(series_key, []).append(point)
            train_series: dict[str, list[dict[str, Any]]] = {}
            for point in curves_for_run["training"]:
                for key, value in point.items():
                    if key in {"round", "actor_updates", "sampler", "cohort"}:
                        continue
                    train_series.setdefault(f"train.{key}", []).append(point)
            metric_keys = sorted(set(train_series) | set(eval_series))
            leaf = Path(relative).name
            run_id = base64.urlsafe_b64encode(f"{source_id}:{relative}".encode()).decode().rstrip("=")
            label = str(source.get("label", source_id))
            name = label if relative in ("", ".") else f"{label} / {relative}"
            algorithm = next((v.lower() for v in (cfg.get('algorithm'), protocol.get('algorithm'), source.get('method'), leaf) if isinstance(v, str) and v), leaf)
            if cfg.get('task_name'):
                name = f"{label} / {cfg['task_name']}"
            state = task.get("state") if task else str(source.get("state", "observed"))
            status_reason = source.get("status_reason")
            synced_at, finished_at = None, None
            if source.get("status_from_process") and relative in statuses and not task:
                status = statuses[relative]
                state = {"PASS": "completed", "COMPLETE": "completed", "COMPLETED": "completed", "FAILED": "failed", "INTERRUPTED": "stopped",
                         "RUNNING": "running"}.get(status.get("status"), "observed")
                age = time.time() - float(status.get("exported_unix") or 0)
                phase = status.get("phase") or {}
                synced_at = status.get('exported_unix')
                finished_at = status.get('completed_unix')
                # Mirror writes are transport activity, not experiment updates.
                stat_updated = status.get('source_updated_unix') or phase.get('unix_time')
                status_reason = f"Slurm {status.get('job_id')}; {phase.get('name', '')}; mirror age {int(age)}s"
                if age > 180 and state == "running":
                    state = "observed"
                    status_reason += "; status stale, check Slurm"
            if protocol and not task and not source.get('state'):
                status = completion.get('status') or protocol.get('status')
                state = {'COMPLETE': 'completed', 'COMPLETED': 'completed', 'FAILED': 'failed', 'INTERRUPTED': 'stopped',
                         'STARTING': 'running', 'RUNNING': 'running'}.get(str(status).upper(), 'observed')
                job_state = job_states.get(str(protocol.get('job_id')))
                if state == 'running':
                    # A protocol is a startup record, not a liveness heartbeat.
                    state = {'RUNNING': 'running', 'COMPLETING': 'running', 'PENDING': 'queued',
                             'CONFIGURING': 'queued', 'SUSPENDED': 'stopped',
                             'COMPLETED': 'completed', 'FAILED': 'failed', 'OUT_OF_MEMORY': 'failed',
                             'TIMEOUT': 'stopped', 'CANCELLED': 'stopped', 'PREEMPTED': 'stopped',
                             'NODE_FAIL': 'failed', 'BOOT_FAIL': 'failed'}.get(job_state, 'observed')
                status_reason = f"Slurm {protocol.get('job_id', '-')}; {job_state or status}; {phase.get('name', '')}"
                if state == 'observed' and str(status).upper() in {'RUNNING', 'STARTING'}:
                    status_reason += '; protocol is stale or execution could not be verified'
                if completion.get('error'):
                    status_reason += '; ' + str(completion['error'])[:300]
            runs.append({
                "id": run_id,
                "source_id": source_id,
                "source_label": source.get("label", source_id),
                "target": source.get("target", "unknown"),
                "path": run_path,
                "name": name,
                "method": algorithm,
                "algorithm": algorithm,
                "task_name": cfg.get('task_name'),
                "job_id": protocol.get('job_id'),
                "phase": phase.get('name'),
                "current_round": phase.get('round'),
                "planned_rounds": cfg.get('rounds'),
                "actor_updates": max((p.get('actor_updates', 0) for p in curves_for_run['training']), default=0),
                "next_evaluation_round": next((r for r in sorted(cfg.get('evaluation_rounds', [])) if r > max((p.get('round', 0) for p in eval_points), default=-1)), None),
                "initialization_risk_excluded": cfg.get('initialization_risk_failure_scored') is False,
                "evaluation_cohorts": cohorts,
                "status_reason": status_reason,
                "evaluation_cohort": cohort,
                "completed_training_round": max((point.get("round", point.get("actor_updates", 0)) for point in curves_for_run["training"]), default=0),
                "completed_evaluation_round": max((point.get("round", point.get("actor_updates", 0)) for point in eval_points), default=0),
                "state": state,
                "task_id": task.get("id") if task else None,
                "launch_profile": task.get("metadata", {}).get("launch_profile") if task else None,
                "execution_target": task.get("metadata", {}).get("execution_target") if task else source.get("target"),
                "updated_at": stat_updated,
                "synced_at": synced_at,
                "finished_at": finished_at,
                "training_points": len(curves_for_run["training"]),
                "evaluation_points": len(curves_for_run["evaluation"]),
                "metrics": metric_keys,
                "latest": {
                    key: series[-1].get(key.removeprefix("train."), series[-1].get(key.removeprefix("eval.").split("@")[0]))
                    for key, series in (train_series | eval_series).items() if series
                },
                "_training": curves_for_run["training"],
                "_evaluation": all_eval_points,
                "_event_files": curves_for_run.get("event_files", []),
                "_source": source,
                "_relative": relative,
            })
        if source_error:
            runs.append({"id": f"source-error-{source_id}", "source_id": source_id,
                         "source_label": source.get("label", source_id), "target": source.get("target", "unknown"),
                         "name": source.get("label", source_id), "state": "unavailable", "error": source_error,
                         "metrics": [], "latest": {}, "training_points": 0, "evaluation_points": 0})
    runs.sort(key=lambda item: (item.get("state") == "unavailable", -float(item.get("updated_at") or 0), item.get("name", "")))
    return runs


def public_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in run.items() if not key.startswith("_")} for run in runs]


CHECKPOINT_SCAN = r'''import base64,json,os,pathlib,re,sys,zipfile
root=pathlib.Path(sys.argv[1]).resolve(); rows=[]
if root.is_dir():
  for current,dirs,files in os.walk(root):
    rel=pathlib.Path(current).relative_to(root)
    if len(rel.parts)>4: dirs[:]=[]; continue
    dirs[:]=[d for d in dirs if not d.startswith('.') and d not in ('video','tensorboard')]
    match=re.fullmatch(r'global_step_(\d+)',pathlib.Path(current).name)
    if not match: continue
    weight=pathlib.Path(current)/'actor/model_state_dict/full_weights.pt'
    if not weight.is_file() or weight.stat().st_size==0 or not zipfile.is_zipfile(weight): continue
    resolved=weight.resolve()
    if root not in resolved.parents: continue
    relative=weight.relative_to(root).as_posix()
    rows.append({'id':base64.urlsafe_b64encode(relative.encode()).decode().rstrip('='),'step':int(match[1]),'path':str(weight),'relative_path':relative,'size_bytes':weight.stat().st_size})
print(json.dumps(sorted(rows,key=lambda row:row['step'],reverse=True)))
'''


def checkpoints_for_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    source = run.get("_source", {})
    python = str(source.get("python") or sys.executable)
    args = [python, "-c", CHECKPOINT_SCAN, run["path"]]
    if source.get("ssh_host"):
        args = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", str(source["ssh_host"]), shlex.join(args)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=35, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or "checkpoint scan failed")[-1200:])
    return json.loads(result.stdout)


NATIVE_LOG_READ = r'''import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(); candidates=[]
if root.is_dir():
  for current,dirs,files in os.walk(root):
    rel=pathlib.Path(current).relative_to(root)
    if len(rel.parts)>=3: dirs[:]=[]
    dirs[:]=[name for name in dirs if not name.startswith('.') and name not in ('checkpoints','video','tensorboard')]
    for name in files:
      path=pathlib.Path(current)/name
      if path.suffix.lower() not in ('.log','.out','.err') or path.is_symlink() or root not in path.resolve().parents: continue
      candidates.append(path)
files=sorted(candidates,key=lambda path:path.stat().st_mtime,reverse=True)[:5]; chunks=[]
for path in files:
  with path.open('rb') as stream:
    stream.seek(max(0,path.stat().st_size-131072//max(1,len(files))))
    chunks.append('=== '+path.relative_to(root).as_posix()+' ===\n'+stream.read().decode('utf-8',errors='replace'))
if not chunks and (root/'process.json').is_file() and (root/'process.json').stat().st_size<=8*1024*1024:
  process=json.loads((root/'process.json').read_text())
  if process.get('latest_log_tail'):
    chunks.append('Mirrored Slurm '+str(process.get('job_id'))+' / '+str(process.get('status'))+'\n'+str(process['latest_log_tail'])[-131072:])
print(json.dumps({'log':'\n\n'.join(chunks),'files':[path.relative_to(root).as_posix() for path in files]}))
'''


def native_log_for_run(run: dict[str, Any]) -> dict[str, Any]:
    source = run.get("_source", {})
    args = [str(source.get("python") or sys.executable), "-c", NATIVE_LOG_READ, run["path"]]
    if source.get("ssh_host"):
        args = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", str(source["ssh_host"]), shlex.join(args)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=35, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or "native RLInf log read failed")[-1200:])
    return json.loads(result.stdout)


def evaluation_profile(config: dict[str, Any], run: dict[str, Any]) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    target_id = str(run.get("execution_target") or run.get("target") or "").lower()
    target = config.get("rlinf_dashboard", {}).get("targets", {}).get(target_id)
    if not target or not target.get("enabled"):
        raise ValueError("该运行没有可用的 RLInf 评测目标")
    profiles = target.get("profiles", {})
    profile_id = run.get("launch_profile")
    if not profile_id:
        profile_id = next((key for key, item in profiles.items() if item.get("evaluation") and item.get("enabled", True)), None)
    profile = profiles.get(profile_id, {})
    if not profile.get("enabled", True) or not profile.get("evaluation"):
        raise ValueError("该运行没有配置原生 RLInf rollout 评测入口")
    return target_id, target, str(profile_id), profile


def metrics_for_run(runs: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    run = next((item for item in runs if item.get("id") == run_id), None)
    if run is None:
        raise FileNotFoundError(run_id)
    if run.get("state") == "unavailable":
        raise RuntimeError(run.get("error", "metric source unavailable"))
    training = run.get("_training", [])
    evaluation = run.get("_evaluation", [])
    if run.get("_event_files"):
        source = run.get("_source", {})
        root = str(source.get("root", "")).rstrip("/")
        relative = str(run.get("_relative", "")).strip("/")
        run_root = root + ("/" + relative if relative and relative != "." else "")
        depth = max(2, min(8, int(source.get("max_depth", 5))))
        parsed = _scan_root(source, run_root, depth, parse_events=True).get("curves", {})
        if not training:
            training = _curve_points(parsed.get("training_curve.json", []))
        if not evaluation:
            evaluation = _curve_points(parsed.get("evaluation_curve.json", []))
        run = dict(run)
        run["training_points"] = len(training)
        run["evaluation_points"] = len(evaluation)
        run["_training"] = training
        run["_evaluation"] = evaluation
        latest = {}
        for prefix, points in (("train", training), ("eval", evaluation)):
            for point in points:
                for key, value in point.items():
                    if key not in {"round", "actor_updates", "sampler", "cohort"}:
                        latest[f"{prefix}.{key}"] = value
        run["latest"] = latest
    return {"run": public_runs([run])[0], "training_points": run.get("_training", []),
            "evaluation_points": run.get("_evaluation", [])}


def public_launch_config(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("rlinf_dashboard", {})
    targets = []
    for target_id, target in block.get("targets", {}).items():
        if not isinstance(target, dict) or not target.get("enabled", False):
            continue
        profiles = []
        for profile_id, profile in target.get("profiles", {}).items():
            if not isinstance(profile, dict) or not profile.get("enabled", True):
                continue
            profiles.append({"id": profile_id, "label": profile.get("label", profile_id),
                             "algorithms": profile.get("algorithms", ["grpo"]),
                             "defaults": profile.get("defaults", {}),
                             "overrides": profile.get("overrides", {}),
                             "requirements": profile.get("requirements", "")})
            profiles[-1]["backend"] = profile.get("backend", "custom")
            profiles[-1]["evaluation_enabled"] = bool(profile.get("evaluation"))
        targets.append({"id": target_id, "label": target.get("label", target_id),
                        "kind": target.get("kind", "local"), "profiles": profiles,
                        "default_gpu_ids": target.get("default_gpu_ids", []),
                        "requirements": target.get("requirements", "")})
    return {"enabled": bool(block.get("enabled", False)), "targets": targets,
            "poll_seconds": max(5, min(120, int(block.get("poll_seconds", 20))))}


def build_launch_spec(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    block = config.get("rlinf_dashboard", {})
    target_id = str(payload.get("target", ""))
    profile_id = str(payload.get("profile", ""))
    target = block.get("targets", {}).get(target_id)
    if not block.get("enabled") or not isinstance(target, dict) or not target.get("enabled"):
        raise ValueError("RLinf launch target is not enabled in Dashboard configuration")
    profile = target.get("profiles", {}).get(profile_id)
    if not isinstance(profile, dict) or not profile.get("enabled", True):
        raise ValueError("unknown or disabled RLinf launch profile")
    algorithm = str(payload.get("algorithm", ""))
    if not SAFE_ID.fullmatch(algorithm) or algorithm not in profile.get("algorithms", []):
        raise ValueError("algorithm is not allowed by this launch profile")
    run_name = str(payload.get("run_name", "")).strip()
    if not SAFE_ID.fullmatch(run_name):
        raise ValueError("run_name must use 1-64 letters, numbers, dot, underscore, or dash")
    run_root = str(profile.get("run_root") or "").rstrip("/")
    project_root = str(target.get("project_root") or "").rstrip("/")
    if not run_root or not project_root:
        raise ValueError("launch profile requires run_root and target project_root")
    output_dir = f"{run_root}/{run_name}/{algorithm}"
    values: dict[str, Any] = {"python": target.get("python", "python3"),
                              "project_root": project_root, "run_name": run_name,
                              "run_root": run_root, "run_dir": output_dir,
                              "algorithm": algorithm}
    algorithm_values = profile.get("algorithm_values", {}).get(algorithm, {})
    if isinstance(algorithm_values, dict):
        values.update({str(key): value for key, value in algorithm_values.items()})
    for key, spec in profile.get("overrides", {}).items():
        if not isinstance(spec, dict):
            continue
        value = payload.get(key, spec.get("default", profile.get("defaults", {}).get(key)))
        kind = spec.get("type", "int")
        if kind == "int":
            try:
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer") from exc
            if value < int(spec.get("min", 0)) or value > int(spec.get("max", 1_000_000)):
                raise ValueError(f"{key} must be in [{spec.get('min', 0)}, {spec.get('max', 1000000)}]")
        elif kind == "float":
            value = float(value)
            if not math.isfinite(value) or value < float(spec.get("min", 0)) or value > float(spec.get("max", 1e9)):
                raise ValueError(f"{key} is outside its configured range")
        elif kind == "enum":
            value = str(value)
            if value not in spec.get("choices", []):
                raise ValueError(f"invalid {key}")
        values[key] = value
    if "group_size" in values and "num_groups" in values:
        values["total_envs"] = int(values["group_size"]) * int(values["num_groups"])
        if values["total_envs"] % int(values.get("micro_batch_size", 1)):
            raise ValueError("每组个数 × 组数必须能被 micro batch size 整除")
    if values.get("eval_interval") == 0:
        raise ValueError("eval_interval must be -1 (disabled) or a positive step interval")
    config_source = profile.get("config_path")
    if config_source:
        config_source = str(config_source)
        values["config"] = config_source
    command_template = profile.get("command_template")
    if not isinstance(command_template, list) or not command_template:
        raise ValueError("launch profile has no command_template")
    values["pair_dir"] = str(profile.get("pair_dir", ""))
    command: list[str] = []
    for token in command_template:
        try:
            command.append(str(token).format_map(values))
        except KeyError as exc:
            raise ValueError(f"launch template uses unconfigured field: {exc.args[0]}") from exc
    env = {str(key): str(value).format_map(values) for key, value in profile.get("environment", {}).items()}
    return {"target_id": target_id, "target": target, "profile_id": profile_id,
            "profile": profile, "algorithm": algorithm, "run_name": run_name,
            "run_root": run_root, "project_root": project_root, "output_dir": output_dir,
            "command": command, "environment": env, "values": values}

#!/usr/bin/env python3
"""Read small run manifests across configured storage locations into one index."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

try:
    from . import eval_video_catalog as catalog
    from . import experiment_artifacts as artifacts
    from .dataset_transfer_runner import clean_transport_env
except ImportError:
    import eval_video_catalog as catalog
    import experiment_artifacts as artifacts
    from dataset_transfer_runner import clean_transport_env


def refresh(config: dict) -> dict:
    rows, errors = [], {}
    for source in config.get("artifact_history_sources", []):
        plan = json.loads(Path(source["plan"]).read_text())
        paths = [str(artifacts.run_directory(entry["artifact_root"], entry) / "run.json") for entry in plan]
        if source.get("ssh_host"):
            code = '''import json,sys
from pathlib import Path
result=[]
for name in json.loads(sys.argv[1]):
 try: result.append(json.loads(Path(name).read_text()))
 except (OSError,ValueError): result.append({})
print(json.dumps(result))
'''
            try:
                command = shlex.join([source["python"], "-c", code, json.dumps(paths)])
                result = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", source["ssh_host"], command], capture_output=True, text=True, timeout=45, env=clean_transport_env())
                if result.returncode:
                    raise RuntimeError(result.stderr[-1000:])
                manifests = json.loads(result.stdout)
            except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
                errors[source["ssh_host"]] = str(exc)
                continue
        else:
            manifests = [catalog.read_object(Path(path)) for path in paths]
        for entry, path, found in zip(plan, paths, manifests):
            fields = {k: entry[k] for k in ("video_source", "task", "experiment", "run_id")}
            valid = found.get("schema") == artifacts.SCHEMA
            row = found if valid else artifacts.manifest(fields, backend=entry["backend"], kind=entry["kind"], state=entry["state"],
                                               origin={"legacy_path": entry["path"], "scope": entry.get("scope", "full_run"), "dashboard_id": entry.get("dashboard_id")})
            row.update(path=str(Path(path).parent), storage_host=source.get("ssh_host", "local_4090"),
                       available=valid or entry["state"] in artifacts.ACTIVE_STATES,
                       migration_action="migrated" if valid else "deferred_active" if entry["state"] in artifacts.ACTIVE_STATES else "missing")
            rows.append(row)
    data = {"schema": artifacts.SCHEMA, "runs": rows, "errors": errors, "updated_at": time.time()}
    catalog.atomic_write(Path(config["workspace_root"]) / "artifact-run-inventory.json", data)
    return {"runs": len(rows), "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(refresh(json.loads(os.path.expandvars(args.config.read_text())))), flush=True)

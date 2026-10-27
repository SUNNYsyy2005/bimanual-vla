#!/usr/bin/env python3
"""Print non-secret Dashboard status fields used for deployment verification."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path


ENV_FILE = Path("/home/sunny/.config/bimanual-vla-sim-dashboard/server.env")


def token() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("export BIMANUAL_VLA_SERVER_TOKEN="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("Dashboard token is unavailable")


request = urllib.request.Request(
    "http://127.0.0.1:8091/api/status",
    headers={"Authorization": f"Bearer {token()}"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    status = json.load(response)

print(
    json.dumps(
        {
            "dashboard_build": status.get("config", {}).get("dashboard_build"),
            "base_checkpoint": status.get("config", {}).get("base_checkpoint"),
            "base_models": [
                {
                    key: model.get(key)
                    for key in (
                        "name",
                        "path",
                        "model_variant",
                        "foundation",
                        "source",
                        "default",
                    )
                }
                for model in status.get("base_models", [])
            ],
            "new_training_tasks": [
                {
                    "id": task.get("id"),
                    "state": task.get("state"),
                    "pid": task.get("pid"),
                    "error": task.get("error"),
                    "exp_name": task.get("metadata", {}).get("exp_name"),
                    "gpu_ids": task.get("metadata", {}).get("gpu_ids"),
                    "allow_busy_gpus": task.get("metadata", {}).get("allow_busy_gpus"),
                    "minimum_free_gpu_mib": task.get("metadata", {}).get("minimum_free_gpu_mib"),
                    "xla_memory_fraction": task.get("metadata", {}).get("xla_memory_fraction"),
                }
                for task in status.get("tasks", [])
                if "_20k_b8_2gpu_20260910"
                in str(task.get("metadata", {}).get("exp_name", ""))
            ],
            "active_gpu_tasks": [
                {
                    "id": task.get("id"),
                    "type": task.get("type"),
                    "state": task.get("state"),
                    "gpu_ids": task.get("metadata", {}).get("gpu_ids"),
                    "exp_name": task.get("metadata", {}).get("exp_name"),
                }
                for task in status.get("tasks", [])
                if task.get("state")
                in {"starting", "running", "stopping", "waiting_norm", "waiting_gpu"}
                and task.get("metadata", {}).get("gpu_ids")
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
)

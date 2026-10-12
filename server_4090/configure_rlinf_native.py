#!/usr/bin/env python3
"""Migrate the simulation dashboard's 4090 profile to native RLInf explicitly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

try:
    from .rlinf_dashboard import NATIVE_TRAIN_OVERRIDES, native_profile_fields, upgrade_native_profiles
except ImportError:
    from rlinf_dashboard import NATIVE_TRAIN_OVERRIDES, native_profile_fields, upgrade_native_profiles


def configure_native(config, *, project, python, model, libero_config, libero_source, transformer_root, run_root):
    project = str(project)
    block = config.setdefault("rlinf_dashboard", {})
    block["enabled"] = True
    local_env = {
        "EMBODIED_PATH": project + "/third_party/wovr/examples/embodiment",
        "ROBOT_PLATFORM": "LIBERO", "LIBERO_TYPE": "standard", "LIBERO_CONFIG_PATH": str(libero_config),
        "PYTHONPATH": ":".join([str(transformer_root), project + "/src", project + "/third_party/wovr", str(libero_source)]),
        "MUJOCO_GL": "egl", "RAY_ADDRESS": "local", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    magick_runtime = Path(transformer_root).parent / "risk-magick-runtime"
    if (magick_runtime / "lib").is_dir():
        local_env.update({"MAGICK_HOME": str(magick_runtime),
                          "LD_LIBRARY_PATH": str(magick_runtime / "lib") + ":" + str(Path(python).parent.parent / "lib"),
                          "LD_PRELOAD": str(magick_runtime / "lib/libjpeg.so.8")})
    model_overrides = [f"actor.model.model_path={model}", f"rollout.model.model_path={model}"]
    profile = {
        "enabled": True, "backend": "rlinf", "label": "原生 RLInf · LIBERO-Safety π0.5",
        "algorithms": ["grpo", "grpo-lag"],
        "algorithm_values": {"grpo": {"lag_enabled": "false"}, "grpo-lag": {"lag_enabled": "true"}},
        "run_root": str(run_root), "overrides": native_profile_fields(), "environment": local_env,
        "command_template": [str(python), project + "/third_party/wovr/examples/embodiment/train_embodied_agent.py",
            "--config-path", project + "/configs/rlinf", "--config-name", "libero_safety_grpo_pi05_smoke",
            "runner.logger.log_path={run_dir}", "runner.logger.experiment_name={run_name}", *model_overrides, *NATIVE_TRAIN_OVERRIDES],
        "evaluation": {"backend": "rlinf", "config_name": "libero_safety_grpo_pi05_smoke", "overrides": model_overrides},
        "requirements": "原生 RLInf actor/rollout/env；请使用空闲 GPU。GRPO-Lag 使用 RLInf 安全成本和对偶更新。",
    }
    block.setdefault("targets", {})["4x4090"] = {
        "enabled": True, "kind": "local", "label": "4×4090 · 原生 RLInf", "project_root": project,
        "python": str(python), "minimum_free_gpu_mib": 18000, "allow_busy_gpus": False,
        "default_gpu_ids": [], "profiles": {"libero_safety_pi05": profile},
    }
    block["sources"] = [item for item in block.get("sources", []) if item.get("target") != "4x4090"]
    block["sources"].insert(0, {"id": "4090_rlinf", "label": "4×4090 RLInf", "target": "4x4090", "root": str(run_root), "python": str(python), "max_depth": 5})
    return upgrade_native_profiles(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    for field in ["project", "python", "model", "libero-config", "libero-source", "transformer-root", "run-root"]:
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    values = vars(args).copy()
    path = values.pop("config")
    # Some asset snapshots contain a namespace-only outer libero directory.
    # Give this checkout its own package root so the stock installed LIBERO
    # package cannot shadow LIBERO-Safety; keep the large assets in place.
    inner_libero = Path(args.libero_source) / "libero" / "libero"
    if inner_libero.is_dir() and not (inner_libero.parent / "__init__.py").exists():
        overlay = Path(args.project) / "dashboard_runtime"
        outer = overlay / "libero"
        outer.mkdir(parents=True, exist_ok=True)
        (outer / "__init__.py").touch()
        link = outer / "libero"
        if not link.exists():
            link.symlink_to(inner_libero, target_is_directory=True)
        values["libero_source"] = str(overlay)
    config = configure_native(json.loads(path.read_text()), **values)
    Path(args.run_root).mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".before-native-rlinf")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print("Native RLInf profile configured; previous config backed up.")


if __name__ == "__main__":
    main()

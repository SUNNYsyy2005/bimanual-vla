"""Reuse pretrained weights without changing their shared directory."""
import os
from pathlib import Path
import shutil

root = Path(os.environ["TASK_ROOT"])
target = root / "models/pi05_libero"
target.mkdir(exist_ok=True)
base = Path(os.environ.get("RECAP_PI05_BASE", "/DATA/disk0/sunny/models/pi05_libero_official_rlinf"))
for name in ["config.json", "model.safetensors"]:
    link = target / name
    if not link.is_symlink() and not link.exists():
        link.symlink_to(base / name)
stats = root / "datasets/SYNTHETIC_recap_train/meta/norm_stats.json"
for parent in [target / "assets/physical-intelligence/libero", target / "physical-intelligence/libero"]:
    parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stats, parent / "norm_stats.json")

"""Hash the finalized 20k inference files for an external deployment transfer."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    if source.name != "20000" or not all((source / name).is_file() for name in
            ("_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA")):
        raise ValueError("source must be the finalized 20000-step checkpoint")
    reference = json.loads(args.reference_provenance.read_text())
    contract = reference["action_contract"]
    if source.parent.name != reference["experiment"]:
        raise ValueError("checkpoint experiment differs from 15k deployment")
    if not (contract["arm_mode"] == "single" and contract["arm_side"] == "right"
            and contract["state_dim"] == contract["raw_action_dim"] == 7
            and reference["deployment_arm_side"] == "left"):
        raise ValueError("15k physical-left/trained-right contract changed")
    paths = sorted(path for part in ("assets", "params")
                   for path in (source / part).rglob("*") if path.is_file())
    if not paths:
        raise ValueError("inference files are missing")
    if not any(path.name == "norm_stats.json" for path in paths):
        raise ValueError("normalization asset is missing")
    files = []
    for path in paths:
        relative = path.relative_to(source).as_posix()
        item = {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}
        files.append(item)
        print(relative, item["bytes"], flush=True)
    output = {
        "status": "source_verified_awaiting_target_transfer",
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_host": "4x4090",
        "source_checkpoint": str(source),
        "checkpoint_step": 20000,
        "experiment": reference["experiment"],
        "source_file_count": len(files),
        "source_total_bytes": sum(item["bytes"] for item in files),
        "action_contract": contract,
        "deployment_arm_side": "left",
        "external_wrist_camera": "cam_left_wrist",
        "training_wrist_image_slot": "right_wrist_0_rgb",
        "left_wrist_image_mask": False,
        "joint_and_norm_values_unchanged": True,
        "target_transfer_verified": False,
        "target_conversion_verified": False,
        "target_smoke_verified": False,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()

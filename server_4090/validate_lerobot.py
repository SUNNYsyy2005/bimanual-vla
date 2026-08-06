#!/usr/bin/env python3
"""Validate an installed LeRobot dataset and exercise the loader.

This accepts both canonical Piper delivery (10D/10D or 20D/20D absolute EEF)
and the observed metadata-free/marked ``legacy_v2`` layout (10D/7D or
20D/14D step delta). The dimensions, contract marker, timestamps and video
checks are performed before sampling the LeRobot loader.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def _install_torchvision_stub_if_broken() -> None:
    """Install a tiny torchvision.transforms fallback for the JAX OpenPI env.

    The 4x4090 `openpi` env is JAX-first and may contain torch/torchvision
    wheels that are ABI-incompatible (`torch._custom_ops` missing). LeRobot's
    dataset loader only needs basic `torchvision.transforms.ToTensor` for image
    loading, so this avoids reinstalling PyTorch just to validate uploads.
    """
    try:
        import torchvision  # noqa: F401
        return
    except Exception:
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)

    import importlib.machinery
    import types

    class _InterpolationMode:
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest_exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    class _ToTensor:
        def __call__(self, image):
            import torch

            array = np.array(image)
            if array.ndim == 2:
                array = array[:, :, None]
            tensor = torch.from_numpy(array.transpose((2, 0, 1))).contiguous()
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(dtype=torch.float32).div(255.0)
            else:
                tensor = tensor.to(dtype=torch.float32)
            return tensor

    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
    transforms_module.ToTensor = _ToTensor
    transforms_module.InterpolationMode = _InterpolationMode

    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None, is_package=True)
    torchvision_module.__path__ = []
    torchvision_module.transforms = transforms_module
    torchvision_module.__version__ = "0.0-dashboard-stub"

    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module


_install_torchvision_stub_if_broken()

from check_pi05_dataset import _dataset_contract, check_dataset
from lerobot.common.datasets import lerobot_dataset as _lerobot_dataset_module
from lerobot.common.datasets import video_utils as _video_utils_module
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _decode_video_frames_cv2(video_path, timestamps, tolerance_s, backend=None):
    """Decode LeRobot validation frames without torchcodec/torchvision video IO.

    The 4x4090 OpenPI environment can have a usable JAX/OpenPI stack while its
    PyTorch ecosystem is intentionally minimal or ABI-mismatched.  LeRobot's
    default decoder prefers torchcodec when the package is merely importable,
    but the installed torchcodec requires newer ``torch.library`` symbols.  For
    upload validation we only need to sample a couple of frames, so OpenCV is a
    smaller and more robust dependency than changing the training environment.
    """

    import cv2
    import torch

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        if fps <= 0:
            raise RuntimeError(f"cannot determine fps for video {video_path}")
        frames = []
        for timestamp in timestamps:
            frame_index = int(round(float(timestamp) * fps))
            if frame_count > 0:
                frame_index = max(0, min(frame_count - 1, frame_index))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"cannot decode frame {frame_index} from {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1).copy()).to(dtype=torch.float32).div(255.0)
            frames.append(tensor)
        if not frames:
            raise RuntimeError(f"no timestamps requested for {video_path}")
        return torch.stack(frames, dim=0)
    finally:
        cap.release()


def _patch_lerobot_video_decoder() -> None:
    _video_utils_module.decode_video_frames = _decode_video_frames_cv2
    _lerobot_dataset_module.decode_video_frames = _decode_video_frames_cv2


_patch_lerobot_video_decoder()


def _dataset_root(dataset_id: str) -> Path:
    root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
    return root / dataset_id


def _dataset_info(dataset_id: str) -> dict:
    path = _dataset_root(dataset_id) / "meta" / "info.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _repair_episode_stats_for_lerobot(root: Path, info: dict) -> list[str]:
    """Make dashboard-generated stats acceptable to LeRobot v2.1.

    LeRobot v2.1's ``aggregate_stats`` uses a substring heuristic:
    any stats key containing the word ``image`` is required to have image-like
    ``(3, 1, 1)`` mean/std/min/max arrays.  Our datasets intentionally store
    scalar camera timestamp columns named ``image_timestamp.<camera>``.  Older
    exporter versions also wrote scalar stats for these timestamp columns, which
    are mathematically correct but trip LeRobot's image-shape assertion during
    loader validation.

    Stats are optional for timestamp columns and are not used by OpenPI training,
    so the safe compatibility repair is to remove non-visual stats keys whose
    names contain ``image``.  The structural checker has already validated the
    underlying timestamp columns before this repair runs.
    """

    features = info.get("features", {}) if isinstance(info, dict) else {}
    stats_path = root / "meta" / "episodes_stats.jsonl"
    try:
        lines = stats_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []

    rows: list[dict] = []
    removed: set[str] = set()
    changed = False
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        stats = row.get("stats")
        if isinstance(stats, dict):
            for key in list(stats):
                feature = features.get(key)
                feature_dtype = feature.get("dtype") if isinstance(feature, dict) else None
                if "image" in key and feature_dtype not in {"image", "video"}:
                    stats.pop(key, None)
                    removed.add(key)
                    changed = True
        rows.append(row)

    if changed:
        _atomic_jsonl(stats_path, rows)
    return sorted(removed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    args = parser.parse_args()
    root = _dataset_root(args.dataset_id)
    info = _dataset_info(args.dataset_id)
    errors = check_dataset(root)
    if errors:
        raise ValueError("dataset contract validation failed:\n  - " + "\n  - ".join(errors))
    contract = _dataset_contract(info)

    metadata_repairs = _repair_episode_stats_for_lerobot(root, info)

    metadata = LeRobotDatasetMetadata(args.dataset_id)
    dataset = LeRobotDataset(args.dataset_id)
    if len(dataset) <= 0:
        raise ValueError("dataset is empty")

    features = info.get("features", {})
    state_key, action_key = contract["state_key"], contract["action_key"]
    camera_fields = contract["camera_features"]
    indexes = sorted({0, len(dataset) - 1})
    samples = []
    for index in indexes:
        sample = dataset[index]
        missing = sorted(key for key in {state_key, action_key, *camera_fields} if key not in sample)
        if missing:
            raise ValueError(f"sample {index}: missing fields {missing}")
        state = np.asarray(sample[state_key])
        action = np.asarray(sample[action_key])
        if state.shape[-1] != contract["state_dim"] or action.shape[-1] != contract["raw_action_dim"]:
            raise ValueError(
                f"sample {index}: state={state.shape}, action={action.shape}, "
                f"expected last dims {contract['state_dim']}/{contract['raw_action_dim']}"
            )
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"sample {index}: state/action contains NaN/Inf")
        samples.append(
            {
                "index": index,
                "state_shape": list(state.shape),
                "action_shape": list(action.shape),
                "camera_fields": camera_fields,
            }
        )
    print(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "schema": contract["schema"],
                "arm_mode": contract["arm_mode"],
                "arm_side": contract["arm_side"],
                "state_dim": contract["state_dim"],
                "raw_action_dim": contract["raw_action_dim"],
                "model_action_dim": contract["model_action_dim"],
                "contract_format": contract["contract_format"],
                "legacy_format": contract["legacy_format"],
                "camera_keys": contract["camera_keys"],
                "layout": contract["column_layout"],
                "frames": len(dataset),
                "fps": metadata.fps,
                "metadata_repairs": metadata_repairs,
                "samples": samples,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

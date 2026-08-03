#!/usr/bin/env python3
"""Prepare and resumably upload a dataset to the 4x4090 dashboard.

Input may be either a canonical LeRobot v2.1 directory or a GUI collection
directory containing ``ep_*.npz``. Raw GUI episodes are validated and exported
to a signature-keyed LeRobot cache before the normal resumable upload path.
The archive is intentionally uncompressed because videos are already compressed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_SERVER = "http://192.168.101.9:8090"
PRINT_LOCK = threading.Lock()


def safe_dataset_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not value or len(value) > 128 or value[0] not in allowed or any(ch not in allowed for ch in value):
        raise ValueError("dataset name may only contain letters, numbers, dot, underscore, and dash")
    if value in {".", ".."} or ".." in value:
        raise ValueError("unsafe dataset name")
    return value


def source_signature(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"dataset cannot contain symlinks: {path}")
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        kind = "d" if path.is_dir() else "f" if path.is_file() else "x"
        if kind == "x":
            raise ValueError(f"unsupported dataset entry: {path}")
        digest.update(f"{kind}\0{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def build_archive(dataset_root: Path, dataset_name: str, cache_dir: Path, rebuild: bool) -> tuple[Path, str, int]:
    signature = source_signature(dataset_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"{dataset_name}-{signature[:16]}.tar"
    sidecar = archive.with_suffix(".json")
    if archive.exists() and sidecar.exists() and not rebuild:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("source_signature") == signature and metadata.get("size") == archive.stat().st_size:
            print(f"Reusing cached archive: {archive}")
            return archive, metadata["sha256"], int(metadata["size"])

    temp = archive.with_suffix(".tar.building")
    print(f"Building uncompressed tar: {archive}", flush=True)
    with tarfile.open(temp, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(dataset_root.rglob("*")):
            relative = path.relative_to(dataset_root).as_posix()
            info = tar.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            if path.is_dir():
                tar.addfile(info)
            elif path.is_file():
                with path.open("rb") as source:
                    tar.addfile(info, source)
            else:
                raise ValueError(f"unsupported dataset entry: {path}")
    os.replace(temp, archive)
    sha256 = sha256_file(archive)
    metadata = {
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_name,
        "source_signature": signature,
        "size": archive.stat().st_size,
        "sha256": sha256,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return archive, sha256, archive.stat().st_size


RAW_EXPORT_CACHE_VERSION = 2
LEROBOT_NORMALIZATION_VERSION = 2


def classify_dataset_source(source: Path) -> str:
    """Return ``lerobot`` or ``raw_npz`` for a supported directory."""
    if not source.is_dir():
        raise ValueError(f"dataset directory does not exist: {source}")
    if (source / "meta" / "info.json").is_file():
        return "lerobot"
    if any(source.glob("ep_*.npz")) or any(source.glob("episode_*.npz")):
        return "raw_npz"
    raise ValueError(
        "dataset must be either a LeRobot directory containing meta/info.json "
        "or a GUI collection directory containing ep_*.npz"
    )


def _raw_export_key(
    source: Path,
    *,
    fps: int,
    allow_incomplete_gripper_coverage: bool,
) -> tuple[str, str]:
    source_hash = source_signature(source)
    options = (
        f"version={RAW_EXPORT_CACHE_VERSION}\n"
        f"source={source_hash}\n"
        f"fps={fps}\n"
        f"allow_incomplete_gripper_coverage={int(allow_incomplete_gripper_coverage)}\n"
    )
    return source_hash, hashlib.sha256(options.encode()).hexdigest()


def prepare_raw_npz_dataset(
    source: Path,
    dataset_name: str,
    cache_dir: Path,
    *,
    fps: int,
    allow_incomplete_gripper_coverage: bool,
    rebuild: bool,
) -> Path:
    """Export raw GUI episodes to a reusable, atomically published cache."""
    source_hash, export_key = _raw_export_key(
        source,
        fps=fps,
        allow_incomplete_gripper_coverage=allow_incomplete_gripper_coverage,
    )
    export_cache = cache_dir / "exports"
    export_cache.mkdir(parents=True, exist_ok=True)
    output_root = export_cache / f"{dataset_name}-{export_key[:16]}"
    marker = output_root.parent / f"{output_root.name}.json"
    expected_marker = {
        "cache_version": RAW_EXPORT_CACHE_VERSION,
        "source_root": str(source),
        "source_signature": source_hash,
        "export_key": export_key,
        "fps": fps,
        "allow_incomplete_gripper_coverage": allow_incomplete_gripper_coverage,
    }
    if output_root.is_dir() and marker.is_file() and not rebuild:
        try:
            cached = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if (
            all(cached.get(key) == value for key, value in expected_marker.items())
            and (output_root / "meta" / "info.json").is_file()
        ):
            print(f"Detected GUI NPZ directory; reusing cached LeRobot export: {output_root}")
            return output_root

    temp_root = output_root.with_name(output_root.name + ".building")
    shutil.rmtree(temp_root, ignore_errors=True)
    print(
        f"Detected GUI NPZ directory: {source}\n"
        f"Validating and exporting to LeRobot cache: {output_root}",
        flush=True,
    )
    try:
        from export_lerobot import export_dataset

        exported = export_dataset(
            source,
            temp_root,
            fps=fps,
            allow_incomplete_gripper_coverage=allow_incomplete_gripper_coverage,
        )
        if exported != temp_root or not (temp_root / "meta" / "info.json").is_file():
            raise RuntimeError("raw NPZ export did not produce a valid LeRobot meta/info.json")
        shutil.rmtree(output_root, ignore_errors=True)
        os.replace(temp_root, output_root)
        marker_temp = marker.parent / f"{marker.name}.building"
        marker_temp.write_text(
            json.dumps(
                {
                    **expected_marker,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(marker_temp, marker)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return output_root


def _feature_dim(info: dict[str, Any], key: str) -> int | None:
    feature = info.get("features", {}).get(key, {})
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None
    return None


def classify_lerobot_contract(info: dict[str, Any]) -> dict[str, Any]:
    """Classify a LeRobot info.json, including metadata-free 8_3_64eps."""
    from pi0_dataset import classify_contract_dimensions

    features = info.get("features", {})
    if "observation.state" in features and "action" in features:
        state_key, action_key, layout = "observation.state", "action", "canonical_columns"
    elif "state" in features and "actions" in features:
        state_key, action_key, layout = "state", "actions", "legacy_columns"
    else:
        raise ValueError("cannot locate LeRobot state/action features")
    state_dim = _feature_dim(info, state_key)
    action_dim = _feature_dim(info, action_key)
    if state_dim is None or action_dim is None:
        raise ValueError("state/action features must have one-dimensional shapes")
    dimensions = classify_contract_dimensions(
        state_dim,
        action_dim,
        schema=info.get("schema"),
        legacy_format=info.get("legacy_format") or info.get("contract_format"),
    )
    return {
        **dimensions,
        "state_key": state_key,
        "action_key": action_key,
        "column_layout": layout,
    }


def _legacy_next_measured_matches(root: Path, contract: dict[str, Any]) -> bool:
    """Verify the observed 8_3_64eps step-delta construction when possible."""
    try:
        import numpy as np
        import pyarrow.parquet as pq
        import piper_data_contract as contract_module

        paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
        if not paths:
            return False
        for path in paths:
            table = pq.read_table(path, columns=[contract["state_key"], contract["action_key"]])
            states = np.asarray(table[contract["state_key"]].to_pylist(), dtype=np.float32)
            actions = np.asarray(table[contract["action_key"]].to_pylist(), dtype=np.float32)
            builder = getattr(
                contract_module,
                "build_legacy_delivery_step_actions",
                contract_module.build_delivery_actions,
            )
            expected = builder(states, arm_count=contract["arm_count"])
            if actions.shape != expected.shape or not np.allclose(actions, expected, atol=1e-5, rtol=1e-5):
                return False
        return True
    except Exception:
        return False


def _legacy_camera_keys(info: dict[str, Any], arm_mode: str) -> list[str]:
    features = info.get("features", {})
    if arm_mode == "bimanual":
        return ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    if "image" in features or "wrist_image" in features:
        return ["cam_high", "cam_wrist"]
    keys = [
        key.removeprefix("observation.images.")
        for key, value in features.items()
        if key.startswith("observation.images.")
        and isinstance(value, dict)
        and value.get("dtype") in {"image", "video"}
    ]
    return sorted(keys)


def _legacy_stats_current(root: Path, contract: dict[str, Any]) -> bool:
    """Return whether numeric episode stats already match parquet payloads."""
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        stats_rows = {}
        stats_path = root / "meta" / "episodes_stats.jsonl"
        if stats_path.is_file():
            for line in stats_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    stats_rows[int(row["episode_index"])] = row.get("stats", {})
        parquets = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
        if len(stats_rows) != len(parquets):
            return False
        for path in parquets:
            episode_index = int(path.stem.removeprefix("episode_"))
            table = pq.read_table(path)
            row_stats = stats_rows.get(episode_index, {})
            for name in table.column_names:
                if name not in row_stats:
                    continue
                field_type = table.schema.field(name).type
                if pa.types.is_list(field_type) or pa.types.is_fixed_size_list(field_type):
                    values = np.asarray(table[name].to_pylist(), dtype=np.float64)
                elif pa.types.is_integer(field_type) or pa.types.is_floating(field_type) or pa.types.is_boolean(field_type):
                    values = np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float64)
                else:
                    continue
                if values.ndim == 1:
                    values = values[:, None]
                actual = row_stats[name]
                checks = {
                    "min": np.min(values, axis=0),
                    "max": np.max(values, axis=0),
                    "mean": np.mean(values, axis=0),
                    "std": np.std(values, axis=0),
                }
                for stat_name, expected in checks.items():
                    if stat_name in actual and not np.allclose(
                        np.asarray(actual[stat_name], dtype=np.float64), expected, atol=1e-5, rtol=1e-5
                    ):
                        return False
                if "count" in actual and int(np.asarray(actual["count"]).reshape(-1)[0]) != len(values):
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _repair_legacy_statistics(root: Path) -> None:
    """Rebuild numeric episodes_stats while retaining image statistics."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    old_rows: dict[int, dict[str, Any]] = {}
    stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                old_rows[int(row["episode_index"])] = row
    new_rows: list[dict[str, Any]] = []
    for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        index = int(path.stem.removeprefix("episode_"))
        table = pq.read_table(path)
        stats = dict(old_rows.get(index, {}).get("stats", {}))
        for name in table.column_names:
            field_type = table.schema.field(name).type
            try:
                if pa.types.is_list(field_type) or pa.types.is_fixed_size_list(field_type):
                    values = np.asarray(table[name].to_pylist(), dtype=np.float64)
                elif pa.types.is_integer(field_type) or pa.types.is_floating(field_type) or pa.types.is_boolean(field_type):
                    values = np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float64)
                else:
                    continue
            except (TypeError, ValueError):
                continue
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
                continue
            stats[name] = {
                "min": np.min(values, axis=0).tolist(),
                "max": np.max(values, axis=0).tolist(),
                "mean": np.mean(values, axis=0).tolist(),
                "std": np.std(values, axis=0).tolist(),
                "count": [len(values)],
            }
        new_rows.append({"episode_index": index, "stats": stats})
    if stats_path.exists():
        stats_path.unlink()
    stats_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in new_rows),
        encoding="utf-8",
    )


def _normalized_legacy_metadata(root: Path, info: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    from pi0_dataset import (
        DEFAULT_ACTION_HORIZON,
        DELIVERY_LEGACY_ACTION_FORMAT,
        DELIVERY_LEGACY_ACTION_SEMANTICS,
        GRIPPER_CLOSED_FRACTION_LEGACY,
        LEGACY_ROTATION_SEMANTICS,
        LEGACY_V2,
        default_eef_names,
    )

    arm_mode = contract["arm_mode"]
    arm_side = str(info.get("arm_side") or ("both" if arm_mode == "bimanual" else "right"))
    if arm_mode == "bimanual":
        arm_side = "both"
    fallback_state, fallback_action = default_eef_names(
        arm_mode=arm_mode, arm_side=arm_side, legacy=True
    )
    state_feature = info["features"][contract["state_key"]]
    action_feature = info["features"][contract["action_key"]]
    state_names = state_feature.get("names") or fallback_state
    action_names = action_feature.get("names") or fallback_action
    measured_verified = _legacy_next_measured_matches(root, contract)
    return {
        "contract_version": 2,
        "schema": "delivery",
        "arm_mode": arm_mode,
        "arm_side": arm_side,
        "state_dim": contract["state_dim"],
        "action_dim": contract["raw_action_dim"],
        "raw_action_dim": contract["raw_action_dim"],
        "model_action_dim": contract["model_action_dim"],
        "state_names": list(state_names),
        "action_names": list(action_names),
        "camera_keys": _legacy_camera_keys(info, arm_mode),
        "contract_format": LEGACY_V2,
        "legacy": True,
        "legacy_format": LEGACY_V2,
        "legacy_delivery_v2": True,
        "delivery_action_format": DELIVERY_LEGACY_ACTION_FORMAT,
        "action_semantics": DELIVERY_LEGACY_ACTION_SEMANTICS,
        "action_source": "next_measured_eef" if measured_verified else "legacy_recorded_eef_delta",
        "action_alignment": "next_observation",
        "action_offset": 1,
        "action_horizon": int(info.get("action_horizon", DEFAULT_ACTION_HORIZON)),
        "gripper_semantics": GRIPPER_CLOSED_FRACTION_LEGACY,
        "rotation_semantics": LEGACY_ROTATION_SEMANTICS,
        "coordinate_frame": "slave_base",
        "legacy_next_measured_verified": measured_verified,
    }


def _link_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst)


def prepare_lerobot_dataset(
    source: Path,
    dataset_name: str,
    cache_dir: Path,
    *,
    rebuild: bool,
) -> Path:
    """Annotate metadata-free 10D+7D/20D+14D delivery as legacy_v2."""
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    try:
        contract = classify_lerobot_contract(info)
    except ValueError:
        # Keep generic LeRobot passthrough behavior; server validation will
        # report unsupported datasets with its normal diagnostics.
        return source
    if not contract["legacy"]:
        return source

    metadata = _normalized_legacy_metadata(source, info, contract)
    if all(info.get(key) == value for key, value in metadata.items()) and _legacy_stats_current(source, contract):
        print(f"Detected explicitly marked {metadata['legacy_format']} LeRobot dataset: {source}")
        return source

    signature = source_signature(source)
    key = hashlib.sha256(
        f"normalize={LEROBOT_NORMALIZATION_VERSION}\nsource={signature}\n".encode()
    ).hexdigest()
    normalized = cache_dir / "normalized" / f"{dataset_name}-{key[:16]}"
    marker = normalized.with_name(normalized.name + ".json")
    if normalized.is_dir() and marker.is_file() and not rebuild:
        cached = json.loads(marker.read_text(encoding="utf-8"))
        if cached.get("source_signature") == signature:
            print(f"Reusing legacy_v2 metadata-normalized dataset: {normalized}")
            return normalized

    temporary = normalized.with_name(normalized.name + ".building")
    shutil.rmtree(temporary, ignore_errors=True)
    normalized.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary, copy_function=_link_copy)
    normalized_info_path = temporary / "meta" / "info.json"
    normalized_info = json.loads(normalized_info_path.read_text(encoding="utf-8"))
    normalized_info.update(metadata)
    normalized_info["features"][contract["state_key"]]["names"] = metadata["state_names"]
    normalized_info["features"][contract["action_key"]]["names"] = metadata["action_names"]
    # copytree may have hard-linked metadata; unlink before writing so the
    # source dataset is never modified by normalization.
    normalized_info_path.unlink()
    normalized_info_path.write_text(
        json.dumps(normalized_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    policy_contract = {
        "version": normalized_info.get("contract_version", 2),
        "robot_type": normalized_info.get("robot_type", "piper"),
        **metadata,
    }
    policy_path = temporary / "meta" / "policy_contract.json"
    policy_path.unlink(missing_ok=True)
    policy_path.write_text(
        json.dumps(policy_contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _repair_legacy_statistics(temporary)
    shutil.rmtree(normalized, ignore_errors=True)
    os.replace(temporary, normalized)
    marker.write_text(
        json.dumps(
            {
                "normalization_version": LEROBOT_NORMALIZATION_VERSION,
                "source": str(source),
                "source_signature": signature,
                "contract_format": metadata["contract_format"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Marked metadata-free delivery dataset as legacy_v2: {normalized}")
    return normalized


def prepare_dataset_directory(
    source: Path,
    dataset_name: str,
    cache_dir: Path,
    *,
    fps: int,
    allow_incomplete_gripper_coverage: bool,
    rebuild: bool,
) -> tuple[Path, str]:
    """Resolve a LeRobot input directly or auto-export a GUI NPZ directory."""
    kind = classify_dataset_source(source)
    if kind == "lerobot":
        prepared = prepare_lerobot_dataset(
            source, dataset_name, cache_dir, rebuild=rebuild
        )
        print(f"Detected LeRobot dataset directory: {source}")
        return prepared, kind
    exported = prepare_raw_npz_dataset(
            source,
            dataset_name,
            cache_dir,
            fps=fps,
            allow_incomplete_gripper_coverage=allow_incomplete_gripper_coverage,
            rebuild=rebuild,
    )
    return prepare_lerobot_dataset(
        exported, dataset_name, cache_dir, rebuild=rebuild
    ), kind


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
            done += len(block)
            if total >= 1024**3 and done % (512 * 1024**2) < len(block):
                print(f"Hashing: {done / total:.1%}", flush=True)
    return digest.hexdigest()


class Client:
    def __init__(self, server: str, token: str, timeout: int):
        self.server = server.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, *, body: bytes | None = None, headers: dict[str, str] | None = None) -> dict:
        request_headers = {"Authorization": f"Bearer {self.token}", **(headers or {})}
        request = Request(self.server + path, data=body, headers=request_headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"cannot reach {self.server}: {exc}") from exc

    def json(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        return self.request(method, path, body=body, headers={"Content-Type": "application/json"})


def upload_one(
    client: Client,
    archive: Path,
    upload_id: str,
    index: int,
    chunk_size: int,
    total_size: int,
    attempts: int,
) -> tuple[int, int]:
    offset = index * chunk_size
    expected = min(chunk_size, total_size - offset)
    with archive.open("rb") as source:
        source.seek(offset)
        body = source.read(expected)
    if len(body) != expected:
        raise RuntimeError(f"short read for chunk {index}: {len(body)} != {expected}")
    chunk_sha = hashlib.sha256(body).hexdigest()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            client.request(
                "PUT",
                f"/api/uploads/{upload_id}/chunks/{index}",
                body=body,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(expected),
                    "X-Chunk-SHA256": chunk_sha,
                },
            )
            return index, expected
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(30, 2 ** (attempt - 1)))
    raise RuntimeError(f"chunk {index} failed after {attempts} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="LeRobot directory, GUI ep_*.npz directory, or .tar with --archive")
    parser.add_argument("--name", default=None, help="server-side LeRobot repo/directory name")
    parser.add_argument("--server", default=os.environ.get("BIMANUAL_VLA_SERVER", DEFAULT_SERVER))
    parser.add_argument("--token", default=os.environ.get("BIMANUAL_VLA_SERVER_TOKEN"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "bimanual-vla" / "uploads")
    parser.add_argument("--archive", action="store_true", help="input is an existing uncompressed .tar")
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="expected/exported FPS for a raw GUI NPZ directory (default: 20)",
    )
    parser.add_argument(
        "--allow-incomplete-gripper-coverage",
        action="store_true",
        help="allow raw GUI export without both fully-open and fully-closed gripper samples",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild cached raw export and upload archive",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate/export a GUI NPZ directory to LeRobot and stop before uploading",
    )
    install_mode = parser.add_mutually_exclusive_group()
    install_mode.add_argument(
        "--merge",
        action="store_true",
        help="append uploaded episodes to an existing compatible dataset; install normally if it does not exist",
    )
    install_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing server dataset after validation",
    )
    args = parser.parse_args()

    if args.prepare_only and args.archive:
        parser.error("--prepare-only requires a dataset directory, not --archive")
    if not args.prepare_only and (not args.token or len(args.token) < 20):
        parser.error("provide --token or BIMANUAL_VLA_SERVER_TOKEN (at least 20 characters)")
    if args.workers <= 0 or args.chunk_mib <= 0 or args.attempts <= 0 or args.fps <= 0:
        parser.error("workers, chunk-mib, attempts, and fps must be positive")
    source = args.dataset.expanduser().resolve()
    dataset_name = safe_dataset_name(args.name or source.stem if args.archive else args.name or source.name)

    if args.archive:
        if not source.is_file() or source.suffix != ".tar":
            parser.error("--archive requires an existing uncompressed .tar file")
        archive = source
        size = archive.stat().st_size
        archive_sha = sha256_file(archive)
    else:
        cache_dir = args.cache_dir.expanduser().resolve()
        try:
            dataset_root, _ = prepare_dataset_directory(
                source,
                dataset_name,
                cache_dir,
                fps=args.fps,
                allow_incomplete_gripper_coverage=args.allow_incomplete_gripper_coverage,
                rebuild=args.rebuild,
            )
        except (OSError, RuntimeError, ValueError, SystemExit) as exc:
            parser.error(str(exc))
        if args.prepare_only:
            print(f"PREPARED_LEROBOT_PATH={dataset_root}")
            print(f"LeRobot preparation complete: {dataset_root}")
            return 0
        archive, archive_sha, size = build_archive(
            dataset_root,
            dataset_name,
            cache_dir,
            args.rebuild,
        )

    chunk_size = args.chunk_mib * 1024 * 1024
    client = Client(args.server, args.token, args.timeout)
    initialized = client.json(
        "POST",
        "/api/uploads/init",
        {
            "dataset_name": dataset_name,
            "size": size,
            "sha256": archive_sha,
            "chunk_size": chunk_size,
            "overwrite": args.overwrite,
            "merge": args.merge,
        },
    )
    upload_id = initialized["id"]
    received = set(map(int, initialized.get("received", [])))
    chunk_count = int(initialized["chunk_count"])
    missing = [index for index in range(chunk_count) if index not in received]
    completed_bytes = sum(min(chunk_size, size - index * chunk_size) for index in received)
    print(
        f"Upload {upload_id}: {len(received)}/{chunk_count} chunks already present, "
        f"remaining={len(missing)}, archive={size / 1024**3:.2f} GiB",
        flush=True,
    )

    failures = []
    done_chunks = len(received)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                upload_one, client, archive, upload_id, index, chunk_size, size, args.attempts
            ): index
            for index in missing
        }
        for future in concurrent.futures.as_completed(future_map):
            index = future_map[future]
            try:
                _, uploaded = future.result()
                completed_bytes += uploaded
                done_chunks += 1
                with PRINT_LOCK:
                    print(
                        f"[{done_chunks}/{chunk_count}] chunk {index} OK · "
                        f"{completed_bytes / size:.1%}",
                        flush=True,
                    )
            except Exception as exc:
                failures.append((index, str(exc)))
                print(f"[FAIL] chunk {index}: {exc}", file=sys.stderr, flush=True)
    if failures:
        print("Upload incomplete. Re-run the same command to resume.", file=sys.stderr)
        return 1

    print("All chunks uploaded; server is assembling, validating, and atomically installing...", flush=True)
    result = client.json("POST", f"/api/uploads/{upload_id}/complete", {})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    operation = result.get("operation", "install")
    print(f"Dataset {operation} complete: {dataset_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

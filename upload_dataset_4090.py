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


RAW_EXPORT_CACHE_VERSION = 1


def classify_dataset_source(source: Path) -> str:
    """Return ``lerobot`` or ``raw_npz`` for a supported directory."""
    if not source.is_dir():
        raise ValueError(f"dataset directory does not exist: {source}")
    if (source / "meta" / "info.json").is_file():
        return "lerobot"
    if any(source.glob("ep_*.npz")):
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
        print(f"Detected LeRobot dataset directory: {source}")
        return source, kind
    return (
        prepare_raw_npz_dataset(
            source,
            dataset_name,
            cache_dir,
            fps=fps,
            allow_incomplete_gripper_coverage=allow_incomplete_gripper_coverage,
            rebuild=rebuild,
        ),
        kind,
    )


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

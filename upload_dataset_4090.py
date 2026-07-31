#!/usr/bin/env python3
"""Pack and resumably upload a LeRobot dataset to the 4x4090 dashboard.

The archive is intentionally uncompressed: videos are already compressed, and this
avoids wasting CPU during collection and server-side installation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
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
    parser.add_argument("dataset", type=Path, help="LeRobot dataset directory, or .tar with --archive")
    parser.add_argument("--name", default=None, help="server-side LeRobot repo/directory name")
    parser.add_argument("--server", default=os.environ.get("BIMANUAL_VLA_SERVER", DEFAULT_SERVER))
    parser.add_argument("--token", default=os.environ.get("BIMANUAL_VLA_SERVER_TOKEN"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "bimanual-vla" / "uploads")
    parser.add_argument("--archive", action="store_true", help="input is an existing uncompressed .tar")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing server dataset after validation")
    args = parser.parse_args()

    if not args.token or len(args.token) < 20:
        parser.error("provide --token or BIMANUAL_VLA_SERVER_TOKEN (at least 20 characters)")
    if args.workers <= 0 or args.chunk_mib <= 0 or args.attempts <= 0:
        parser.error("workers, chunk-mib, and attempts must be positive")
    source = args.dataset.expanduser().resolve()
    dataset_name = safe_dataset_name(args.name or source.stem if args.archive else args.name or source.name)

    if args.archive:
        if not source.is_file() or source.suffix != ".tar":
            parser.error("--archive requires an existing uncompressed .tar file")
        archive = source
        size = archive.stat().st_size
        archive_sha = sha256_file(archive)
    else:
        if not source.is_dir() or not (source / "meta" / "info.json").exists():
            parser.error("dataset must be a LeRobot directory containing meta/info.json")
        archive, archive_sha, size = build_archive(source, dataset_name, args.cache_dir.expanduser(), args.rebuild)

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
    print(f"Installed dataset id: {dataset_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

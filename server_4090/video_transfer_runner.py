#!/usr/bin/env python3
"""Parallel-copy a configured remote evaluation video back to the local Dashboard host."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import shlex
import subprocess
import time
from pathlib import Path, PurePosixPath

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}
SSH_OPTS = [
    "ssh",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
]
CHUNK_BYTES = 64 * 1024 * 1024


def q(value: str) -> str:
    return shlex.quote(str(value))


def ssh_capture(host: str, inner: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*SSH_OPTS, host, inner], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def remote_size(host: str, path: str) -> int:
    result = ssh_capture(host, f"test -f {q(path)} && stat -c %s {q(path)}")
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:] or f"remote video not found: {path}")
    return int(result.stdout.strip())


def copy_chunk(host: str, remote_path: str, offset: int, length: int, part_path: Path, index: int, count: int) -> str:
    started = time.time()
    # GNU dd on the cluster supports byte-based skip/count.  Keep each chunk as
    # an independent SSH stream so large videos are transferred concurrently.
    inner = (
        f"dd if={q(remote_path)} bs=4M iflag=skip_bytes,count_bytes "
        f"skip={offset} count={length} status=none"
    )
    with part_path.open("wb") as output:
        result = subprocess.run([*SSH_OPTS, host, inner], stdout=output, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(f"chunk {index + 1}/{count} failed: {stderr}")
    actual = part_path.stat().st_size
    if actual != length:
        raise RuntimeError(f"chunk {index + 1}/{count} size mismatch: got {actual}, expected {length}")
    return f"[dashboard] video shard {index + 1}/{count} bytes={length} elapsed_s={time.time() - started:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parallelism", type=int, default=4)
    args = parser.parse_args()

    rel = PurePosixPath(args.relative_path)
    if rel.is_absolute() or ".." in rel.parts or not rel.name:
        raise SystemExit("invalid relative path")
    if Path(rel.name).suffix.lower() not in VIDEO_SUFFIXES:
        raise SystemExit("unsupported video suffix")

    source_path = str(PurePosixPath(args.source_root) / rel)
    target_root = Path(args.target_root).expanduser().resolve() / args.source_name
    target_path = target_root / Path(*rel.parts)
    if target_path.exists() and not args.overwrite:
        raise SystemExit(f"target exists; use overwrite: {target_path}")

    started = time.time()
    size = remote_size(args.source_host, source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if args.overwrite and target_path.exists():
        target_path.unlink()

    chunks: list[tuple[int, int]] = []
    for offset in range(0, size, CHUNK_BYTES):
        chunks.append((offset, min(CHUNK_BYTES, size - offset)))
    parallelism = max(1, min(16, int(args.parallelism), max(1, len(chunks))))
    print(
        f"[dashboard] video transfer start: {args.source_name}:{rel} -> {target_path} bytes={size} parallelism={parallelism}",
        flush=True,
    )

    part_paths = [target_path.with_name(f"{target_path.name}.part{idx:04d}") for idx in range(len(chunks))]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [
                executor.submit(
                    copy_chunk,
                    args.source_host,
                    source_path,
                    offset,
                    length,
                    part_paths[idx],
                    idx,
                    len(chunks),
                )
                for idx, (offset, length) in enumerate(chunks)
            ]
            for future in concurrent.futures.as_completed(futures):
                print(future.result(), flush=True)
        with tmp_path.open("wb") as output:
            for part in part_paths:
                with part.open("rb") as input_fh:
                    while True:
                        block = input_fh.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
        if tmp_path.stat().st_size != size:
            raise RuntimeError(f"assembled video size mismatch: got {tmp_path.stat().st_size}, expected {size}")
        os.replace(tmp_path, target_path)
    finally:
        for part in part_paths:
            try:
                part.unlink()
            except FileNotFoundError:
                pass
        if tmp_path.exists() and not target_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    print(
        f"[dashboard] video transfer finished rc=0 elapsed_s={time.time() - started:.1f} target={target_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

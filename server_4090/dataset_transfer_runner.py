#!/usr/bin/env python3
"""Parallel stream-copy one LeRobot dataset between Dashboard locations."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import PurePosixPath
from typing import Any

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SSH_OPTS = [
    "ssh",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
]


def b64_json(value: str) -> Any:
    return json.loads(base64.urlsafe_b64decode(value.encode()).decode())


def q(value: str) -> str:
    return shlex.quote(str(value))


def location_root(location: dict[str, Any]) -> str:
    root = location.get("dataset_root")
    if not root:
        raise ValueError(f"location {location.get('name')} has no dataset_root")
    return str(root).rstrip("/")


def location_host(location: dict[str, Any]) -> str | None:
    host = location.get("host") or location.get("submit_host")
    return str(host) if host else None


def dataset_path(location: dict[str, Any], dataset_id: str) -> str:
    return str(PurePosixPath(location_root(location)) / dataset_id)


def shell_args(location: dict[str, Any], inner: str) -> list[str]:
    host = location_host(location)
    if host:
        return [*SSH_OPTS, host, inner]
    return ["bash", "-lc", inner]


def shell_command(location: dict[str, Any], inner: str) -> str:
    host = location_host(location)
    if host:
        return " ".join(q(part) for part in [*SSH_OPTS, host, inner])
    return inner


def run_shell(location: dict[str, Any], inner: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(shell_args(location, inner), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def list_source_files(location: dict[str, Any], dataset_id: str) -> list[tuple[str, int]]:
    root = location_root(location)
    dataset = str(PurePosixPath(root) / dataset_id)
    # Use NUL output so spaces in filenames remain safe.  A LeRobot dataset is
    # copied as regular files; empty dirs are not meaningful for training/eval.
    inner = f"test -d {q(dataset)} && cd {q(dataset)} && find . -type f -printf '%s\\t%P\\0' | sort -z"
    result = subprocess.run(shell_args(location, inner), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(f"failed to list source dataset files: {stderr}")
    rows: list[tuple[str, int]] = []
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        size_text, sep, rel_bytes = entry.partition(b"\t")
        if not sep or not rel_bytes:
            continue
        rel = rel_bytes.decode(errors="surrogateescape")
        if PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
            raise RuntimeError(f"unsafe source path: {rel!r}")
        try:
            size = int(size_text.decode())
        except ValueError:
            size = 0
        rows.append((rel, max(0, size)))
    if not rows:
        raise RuntimeError(f"source dataset has no regular files: {dataset}")
    return rows


def shard_files(files: list[tuple[str, int]], parallelism: int) -> list[list[tuple[str, int]]]:
    parallelism = max(1, min(int(parallelism), len(files)))
    shards: list[list[tuple[str, int]]] = [[] for _ in range(parallelism)]
    shard_sizes = [0 for _ in range(parallelism)]
    for rel, size in sorted(files, key=lambda item: item[1], reverse=True):
        idx = min(range(parallelism), key=lambda i: shard_sizes[i])
        shards[idx].append((rel, size))
        shard_sizes[idx] += size
    return [shard for shard in shards if shard]


def prepare_target(location: dict[str, Any], dataset_id: str, overwrite: bool) -> None:
    root = location_root(location)
    target = str(PurePosixPath(root) / dataset_id)
    if overwrite:
        inner = f"rm -rf {q(target)} && mkdir -p {q(target)}"
    else:
        inner = f"mkdir -p {q(root)} && test ! -e {q(target)} && mkdir -p {q(target)}"
    result = run_shell(location, inner)
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-4000:] or f"failed to prepare target {target}")


def write_shard_list(tmpdir: str, dataset_id: str, shard_index: int, shard: list[tuple[str, int]]) -> str:
    path = os.path.join(tmpdir, f"shard_{shard_index:03d}.list0")
    with open(path, "wb") as fh:
        for rel, _size in shard:
            fh.write(f"{dataset_id}/{rel}".encode(errors="surrogateescape") + b"\0")
    return path


def run_shard(
    *,
    shard_index: int,
    shard_count: int,
    list_path: str,
    source: dict[str, Any],
    target: dict[str, Any],
) -> tuple[int, str]:
    source_inner = f"tar -C {q(location_root(source))} -cf - --null -T -"
    target_inner = f"tar -C {q(location_root(target))} -xf -"
    pipeline = (
        "set -euo pipefail; "
        f"cat {q(list_path)} | "
        f"{shell_command(source, source_inner)} | "
        f"{shell_command(target, target_inner)}"
    )
    started = time.time()
    result = subprocess.run(["bash", "-lc", pipeline], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    summary = (
        f"[dashboard] shard {shard_index + 1}/{shard_count} "
        f"rc={result.returncode} elapsed_s={elapsed:.1f}\n{result.stdout[-4000:]}"
    )
    return result.returncode, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--target-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parallelism", type=int, default=4)
    args = parser.parse_args()

    if not SAFE_NAME.match(args.dataset_id):
        raise SystemExit("invalid dataset id")
    source = b64_json(args.source_json)
    target = b64_json(args.target_json)
    source_name = source.get("name", "source")
    target_name = target.get("name", "target")
    parallelism = max(1, min(16, int(args.parallelism)))
    print(
        f"[dashboard] dataset transfer start: {args.dataset_id} {source_name} -> {target_name} parallelism={parallelism}",
        flush=True,
    )
    print(f"[dashboard] source_path={dataset_path(source, args.dataset_id)}", flush=True)
    print(f"[dashboard] target_path={dataset_path(target, args.dataset_id)}", flush=True)
    started = time.time()
    files = list_source_files(source, args.dataset_id)
    total_bytes = sum(size for _rel, size in files)
    shards = shard_files(files, parallelism)
    print(
        f"[dashboard] files={len(files)} bytes={total_bytes} shards={len(shards)}",
        flush=True,
    )
    prepare_target(target, args.dataset_id, args.overwrite)
    with tempfile.TemporaryDirectory(prefix="dashboard_dataset_transfer_") as tmpdir:
        list_paths = [write_shard_list(tmpdir, args.dataset_id, idx, shard) for idx, shard in enumerate(shards)]
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(list_paths)) as executor:
            futures = [
                executor.submit(
                    run_shard,
                    shard_index=idx,
                    shard_count=len(list_paths),
                    list_path=list_path,
                    source=source,
                    target=target,
                )
                for idx, list_path in enumerate(list_paths)
            ]
            for future in concurrent.futures.as_completed(futures):
                rc, summary = future.result()
                print(summary, flush=True)
                if rc != 0:
                    failures.append(rc)
    elapsed = time.time() - started
    final_rc = failures[0] if failures else 0
    print(f"[dashboard] dataset transfer finished rc={final_rc} elapsed_s={elapsed:.1f}", flush=True)
    return final_rc


if __name__ == "__main__":
    raise SystemExit(main())

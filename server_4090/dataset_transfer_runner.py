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
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SSH_OPTS = [
    "ssh",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
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


def is_login_shared_slurm(location: dict[str, Any]) -> bool:
    """Whether a location is H100-style shared storage behind login-server.

    H100 cannot accept a Dashboard port, but its /DATA filesystem is visible
    from login-server.  Use rsync over SSH for this case instead of piping a
    tar archive into login-server, so login-server only forwards/writes files
    and never performs a large archive extraction.
    """
    return str(location.get("access_mode", "")).strip().lower() == "login_shared_slurm"


def shell_args(
    location: dict[str, Any],
    inner: str,
    *,
    control_path: str | None = None,
) -> list[str]:
    host = location_host(location)
    if host:
        options = list(SSH_OPTS)
        if control_path:
            options.extend(["-S", control_path, "-o", "ControlMaster=no"])
        return [*options, host, inner]
    return ["bash", "-lc", inner]


def shell_command(location: dict[str, Any], inner: str) -> str:
    host = location_host(location)
    if host:
        return " ".join(q(part) for part in [*SSH_OPTS, host, inner])
    return inner


def start_ssh_control_master(location: dict[str, Any], control_path: str) -> str | None:
    """Open one multiplexed SSH connection for parallel rsync lanes.

    login-server applies connection/session limits.  Starting one master TCP
    connection and multiplexing the rsync channels through it avoids the
    failure mode where several independent SSH handshakes are accepted and
    then one is dropped with ``kex_exchange_identification``.  The returned
    host is used to close the master after the transfer.
    """
    host = location_host(location)
    if not host:
        return None
    command = [
        *SSH_OPTS,
        "-M",
        "-N",
        "-f",
        "-S",
        control_path,
        "-o",
        "ControlMaster=yes",
        "-o",
        "ControlPersist=600",
        host,
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=clean_transport_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to open multiplexed SSH master to {host}: "
            f"{result.stdout[-4000:]}"
        )
    return host


def stop_ssh_control_master(host: str | None, control_path: str | None) -> None:
    if not host or not control_path:
        return
    subprocess.run(
        [*SSH_OPTS, "-S", control_path, "-O", "exit", host],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=clean_transport_env(),
    )


def clean_transport_env() -> dict[str, str]:
    """Avoid leaking the OpenPI conda OpenSSL into system ssh/rsync.

    Dashboard tasks intentionally run Python from the OpenPI environment, but
    that environment's ``libssl`` can be newer than the system OpenSSH binary
    on login-server.  Keeping LD_LIBRARY_PATH/LD_PRELOAD for transport caused
    ``OpenSSL version mismatch`` before any bytes were copied.
    """
    env = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "OPENSSL_CONF", "OPENSSL_MODULES"):
        env.pop(key, None)
    return env


def run_shell(location: dict[str, Any], inner: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        shell_args(location, inner),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=clean_transport_env(),
    )


def sync_root_relative_file(
    source: dict[str, Any],
    target: dict[str, Any],
    relative_path: str,
) -> tuple[int, str]:
    """Copy a small file that lives beside a transferred dataset/checkpoint.

    Checkpoint action-contract markers intentionally live outside the numeric
    Orbax step directory, under ``.policy_action_conventions``.  The normal
    manifest transfer only walks ``<root>/<dataset_id>`` and therefore cannot
    carry the marker along.  This helper handles the marker separately while
    preserving the same SSH-only transport rules as the main transfer.
    """
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        return 2, f"invalid marker relative path: {relative_path}"
    source_path = str(PurePosixPath(location_root(source)) / relative)
    target_path = str(PurePosixPath(location_root(target)) / relative)

    if location_host(source):
        source_result = run_shell(source, f"cat {q(source_path)}")
        if source_result.returncode:
            return source_result.returncode, (
                f"source marker is unavailable: {source_path}\n{source_result.stdout[-2000:]}"
            )
        contents = source_result.stdout
    else:
        try:
            contents = Path(source_path).read_text(encoding="utf-8")
        except OSError as exc:
            return 2, f"source marker is unavailable: {source_path}: {exc}"

    if location_host(target):
        parent = str(PurePosixPath(target_path).parent)
        target_result = subprocess.run(
            shell_args(target, f"mkdir -p {q(parent)} && cat > {q(target_path)}"),
            input=contents,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=clean_transport_env(),
        )
        if target_result.returncode:
            return target_result.returncode, (
                f"target marker write failed: {target_path}\n{target_result.stdout[-2000:]}"
            )
    else:
        try:
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            Path(target_path).write_text(contents, encoding="utf-8")
        except OSError as exc:
            return 2, f"target marker write failed: {target_path}: {exc}"
    return 0, f"marker synchronized: {source_path} -> {target_path}"


def list_source_files(
    location: dict[str, Any],
    dataset_id: str,
    *,
    control_path: str | None = None,
) -> list[tuple[str, int]]:
    root = location_root(location)
    dataset = str(PurePosixPath(root) / dataset_id)
    # Use NUL output so spaces in filenames remain safe.  A LeRobot dataset is
    # copied as regular files; empty dirs are not meaningful for training/eval.
    inner = f"test -d {q(dataset)} && cd {q(dataset)} && find . -type f -printf '%s\\t%P\\0' | sort -z"
    result = subprocess.run(
        shell_args(location, inner, control_path=control_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=clean_transport_env(),
    )
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


class TransferProgress:
    """Write a small, atomically replaced progress sidecar for the Dashboard.

    The transfer itself remains unchanged: the reporter only performs a cheap
    periodic target manifest scan and never participates in the data path.
    ``rsync`` may keep an in-progress file under a temporary name, so the
    displayed byte count advances at completed-file boundaries.  This is
    intentionally conservative rather than claiming bytes that are not yet
    visible at the target path.
    """

    def __init__(
        self,
        path: str | None,
        *,
        dataset_id: str,
        source: dict[str, Any],
        target: dict[str, Any],
        parallelism: int,
    ) -> None:
        self.path = Path(path).expanduser() if path else None
        self.dataset_id = dataset_id
        self.source = source
        self.target = target
        self.parallelism = parallelism
        self.files: list[tuple[str, int]] = []
        self.shard_count = 0
        self.completed_shards = 0
        self.total_bytes = 0
        self.completed_bytes = 0
        self.completed_files = 0
        self.speed_bytes_per_sec = 0.0
        self.eta_seconds: float | None = None
        self.started_at = time.time()
        self.last_sample_at = self.started_at
        self.last_sample_bytes = 0
        self.last_scan_error: str | None = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.finalized = False

    def _write(self, *, state: str, error: str | None = None) -> None:
        if self.path is None:
            return
        with self.lock:
            total = max(0, int(self.total_bytes))
            completed = max(0, min(int(self.completed_bytes), total or int(self.completed_bytes)))
            progress = (completed / total) if total else 0.0
            payload: dict[str, Any] = {
                "state": state,
                "progress": round(progress, 6),
                "completed_bytes": completed,
                "total_bytes": total,
                "completed_files": int(self.completed_files),
                "total_files": len(self.files),
                "completed_shards": int(self.completed_shards),
                "total_shards": int(self.shard_count),
                "parallelism": int(self.parallelism),
                "speed_bytes_per_sec": round(max(0.0, self.speed_bytes_per_sec), 2),
                "eta_seconds": None if self.eta_seconds is None else round(max(0.0, self.eta_seconds), 1),
                "dataset_id": self.dataset_id,
                "source": self.source.get("name"),
                "target": self.target.get("name"),
                "source_path": dataset_path(self.source, self.dataset_id),
                "target_path": dataset_path(self.target, self.dataset_id),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started_at)),
            }
            if self.last_scan_error:
                payload["scan_error"] = self.last_scan_error
            if error:
                payload["error"] = str(error)[-4000:]
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)

    def configure(self, files: list[tuple[str, int]], shard_count: int) -> None:
        with self.lock:
            self.files = list(files)
            self.shard_count = int(shard_count)
            self.total_bytes = sum(max(0, int(size)) for _rel, size in self.files)
        self._write(state="preparing")

    def start(self, *, control_path: str | None = None) -> None:
        self._write(state="running")
        self.thread = threading.Thread(
            target=self._scan_loop,
            args=(control_path,),
            name="dataset-transfer-progress",
            daemon=True,
        )
        self.thread.start()

    def _scan_once(self, control_path: str | None, *, allow_final: bool = False) -> None:
        try:
            target_files = dict(list_source_files(self.target, self.dataset_id, control_path=control_path))
            with self.lock:
                completed = 0
                files = 0
                for rel, expected in self.files:
                    expected = max(0, int(expected))
                    actual = max(0, int(target_files.get(rel, 0)))
                    completed += min(actual, expected)
                    if expected == 0 or actual >= expected:
                        files += 1
                now = time.time()
                elapsed = max(0.001, now - self.last_sample_at)
                delta = max(0, completed - self.last_sample_bytes)
                instant = delta / elapsed
                if instant > 0:
                    self.speed_bytes_per_sec = instant if not self.speed_bytes_per_sec else (0.7 * self.speed_bytes_per_sec + 0.3 * instant)
                self.completed_bytes = completed
                self.completed_files = files
                self.last_sample_at = now
                self.last_sample_bytes = completed
                remaining = max(0, self.total_bytes - completed)
                self.eta_seconds = remaining / self.speed_bytes_per_sec if self.speed_bytes_per_sec > 0 and remaining else (0.0 if remaining == 0 else None)
                self.last_scan_error = None
                if self.finalized and not allow_final:
                    return
            self._write(state="running")
        except Exception as exc:  # progress must never interrupt a transfer
            with self.lock:
                self.last_scan_error = str(exc)[-1000:]
            self._write(state="running")

    def _scan_loop(self, control_path: str | None) -> None:
        while not self.stop_event.is_set():
            self._scan_once(control_path)
            self.stop_event.wait(2.0)

    def shard_completed(self) -> None:
        with self.lock:
            if self.finalized:
                return
            self.completed_shards += 1
        self._write(state="running")

    def finish(self, *, state: str, returncode: int | None = None, error: str | None = None) -> None:
        with self.lock:
            self.finalized = True
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        if state == "completed":
            self._scan_once(None, allow_final=True)
            with self.lock:
                self.completed_bytes = self.total_bytes
                self.completed_files = len(self.files)
                self.completed_shards = self.shard_count
                self.speed_bytes_per_sec = max(self.speed_bytes_per_sec, self.total_bytes / max(0.001, time.time() - self.started_at))
                self.eta_seconds = 0.0
        message = error
        if returncode not in (None, 0):
            message = f"returncode={returncode}" + (f"; {error}" if error else "")
        self._write(state=state, error=message)


def shard_files(files: list[tuple[str, int]], parallelism: int) -> list[list[tuple[str, int]]]:
    parallelism = max(1, min(int(parallelism), len(files)))
    shards: list[list[tuple[str, int]]] = [[] for _ in range(parallelism)]
    shard_sizes = [0 for _ in range(parallelism)]
    for rel, size in sorted(files, key=lambda item: item[1], reverse=True):
        idx = min(range(parallelism), key=lambda i: shard_sizes[i])
        shards[idx].append((rel, size))
        shard_sizes[idx] += size
    return [shard for shard in shards if shard]


def target_exists(location: dict[str, Any], dataset_id: str) -> bool:
    target = str(PurePosixPath(location_root(location)) / dataset_id)
    result = run_shell(location, f"test -e {q(target)}")
    return result.returncode == 0


def prepare_target(
    location: dict[str, Any],
    dataset_id: str,
    overwrite: bool,
    resume_existing: bool = False,
) -> None:
    root = location_root(location)
    target = str(PurePosixPath(root) / dataset_id)
    if overwrite:
        inner = f"rm -rf {q(target)} && mkdir -p {q(target)}"
    elif resume_existing:
        # Keep files from an interrupted transfer.  rsync's
        # --partial/--append-verify will fill them in and the final manifest
        # check rejects any incomplete or stale target.
        inner = f"mkdir -p {q(root)} {q(target)}"
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
    control_path: str | None = None,
) -> tuple[int, str]:
    started = time.time()
    if is_login_shared_slurm(source) or is_login_shared_slurm(target):
        # H100 and login-server share the target filesystem.  Copy the exact
        # file shard with rsync; unlike the old tar pipeline this does not
        # decompress a multi-GB archive on login-server.
        ssh_options = list(SSH_OPTS)
        if control_path:
            ssh_options.extend(
                [
                    "-S",
                    control_path,
                    "-o",
                    "ControlMaster=no",
                ]
            )
        ssh_rsh = shlex.join(ssh_options)
        rsync = [
            "rsync", "-a", "--partial", "--append-verify",
            "--timeout=1200", "--from0",
            f"--files-from={list_path}",
            "-e", ssh_rsh,
        ]
        if is_login_shared_slurm(target) and not is_login_shared_slurm(source):
            # 4x4090 -> H100 shared /DATA via login-server.
            rsync += [
                f"{location_root(source)}/",
                f"{location_host(target)}:{location_root(target)}/",
            ]
        elif is_login_shared_slurm(source) and not is_login_shared_slurm(target):
            # H100 shared /DATA via login-server -> 4x4090.
            rsync += [
                f"{location_host(source)}:{location_root(source)}/",
                f"{location_root(target)}/",
            ]
        else:
            return 2, "[dashboard] invalid shared-slurm transfer: both endpoints use login_shared_slurm"
        # A long rsync stream can be interrupted by the login-server/SSH
        # session limit.  Retry the same shard; --partial/--append-verify
        # resumes already transferred bytes instead of restarting the shard.
        result = None
        attempts = 3
        for attempt in range(1, attempts + 1):
            result = subprocess.run(
                rsync,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=clean_transport_env(),
            )
            if result.returncode == 0 or attempt == attempts:
                break
            time.sleep(min(10, 2 ** attempt))
        assert result is not None
    else:
        source_inner = f"tar -C {q(location_root(source))} -cf - --null -T -"
        target_inner = f"tar -C {q(location_root(target))} -xf -"
        pipeline = (
            "set -euo pipefail; "
            f"cat {q(list_path)} | "
            f"{shell_command(source, source_inner)} | "
            f"{shell_command(target, target_inner)}"
        )
        result = subprocess.run(
            ["bash", "-lc", pipeline],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=clean_transport_env(),
        )
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
    parser.add_argument("--skip-existing", action="store_true", help="Return success without copying when target already exists")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--progress-json", default=None, help="Dashboard progress sidecar path")
    parser.add_argument(
        "--marker-name",
        default=None,
        help="Checkpoint experiment name whose action-contract marker is beside the transferred directory",
    )
    args = parser.parse_args()

    if not SAFE_NAME.match(args.dataset_id):
        raise SystemExit("invalid dataset id")
    source = b64_json(args.source_json)
    target = b64_json(args.target_json)
    source_name = source.get("name", "source")
    target_name = target.get("name", "target")
    parallelism = max(1, min(16, int(args.parallelism)))
    progress = TransferProgress(
        args.progress_json or os.environ.get("DASHBOARD_TASK_PROGRESS_PATH"),
        dataset_id=args.dataset_id,
        source=source,
        target=target,
        parallelism=parallelism,
    )
    progress._write(state="starting")
    print(
        f"[dashboard] dataset transfer start: {args.dataset_id} {source_name} -> {target_name} parallelism={parallelism}",
        flush=True,
    )
    print(f"[dashboard] source_path={dataset_path(source, args.dataset_id)}", flush=True)
    print(f"[dashboard] target_path={dataset_path(target, args.dataset_id)}", flush=True)
    started = time.time()
    files = list_source_files(source, args.dataset_id)
    effective_overwrite = bool(args.overwrite)
    resume_existing = False
    if args.skip_existing and not args.overwrite and target_exists(target, args.dataset_id):
        target_files = list_source_files(target, args.dataset_id)
        if target_files == files:
            marker_rc = 0
            if args.marker_name:
                marker_relative_path = f".policy_action_conventions/{args.marker_name}.json"
                marker_rc, marker_summary = sync_root_relative_file(source, target, marker_relative_path)
                print(f"[dashboard] {marker_summary}", flush=True)
            progress.configure(files, 0)
            progress.finish(
                state="completed" if marker_rc == 0 else "failed",
                returncode=marker_rc,
            )
            print(
                f"[dashboard] target already exists and manifest matches; skip transfer: "
                f"{dataset_path(target, args.dataset_id)}",
                flush=True,
            )
            return marker_rc
        # A directory left by an interrupted transfer is resumed in place.
        # This avoids throwing away already copied multi-GB parquet files.
        print(
            f"[dashboard] target exists but manifest differs "
            f"(source_files={len(files)} target_files={len(target_files)}); resuming partial target",
            flush=True,
        )
        resume_existing = True
    elif target_exists(target, args.dataset_id) and not args.overwrite:
        raise RuntimeError(
            f"target already exists and does not match overwrite policy: {dataset_path(target, args.dataset_id)}"
        )
    total_bytes = sum(size for _rel, size in files)
    shards = shard_files(files, parallelism)
    print(
        f"[dashboard] files={len(files)} bytes={total_bytes} shards={len(shards)}",
        flush=True,
    )
    progress.configure(files, len(shards))
    prepare_target(target, args.dataset_id, effective_overwrite, resume_existing=resume_existing)
    with tempfile.TemporaryDirectory(prefix="dashboard_dataset_transfer_") as tmpdir:
        list_paths = [write_shard_list(tmpdir, args.dataset_id, idx, shard) for idx, shard in enumerate(shards)]
        failures = []
        control_path = None
        control_host = None
        shared_slurm_transfer = is_login_shared_slurm(source) or is_login_shared_slurm(target)
        try:
            if shared_slurm_transfer and len(list_paths) > 1:
                control_path = os.path.join(tmpdir, "ssh-control")
                control_host = location_host(target if is_login_shared_slurm(target) else source)
                start_ssh_control_master(
                    target if is_login_shared_slurm(target) else source,
                    control_path,
                )
                print(
                    f"[dashboard] using one multiplexed SSH master for {len(list_paths)} parallel rsync lanes "
                    f"via {control_host}",
                    flush=True,
                )
            progress.start(control_path=control_path)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(list_paths)) as executor:
                futures = [
                    executor.submit(
                        run_shard,
                        shard_index=idx,
                        shard_count=len(list_paths),
                        list_path=list_path,
                        source=source,
                        target=target,
                        control_path=control_path,
                    )
                    for idx, list_path in enumerate(list_paths)
                ]
                for future in concurrent.futures.as_completed(futures):
                    rc, summary = future.result()
                    print(summary, flush=True)
                    if rc == 0:
                        progress.shard_completed()
                    if rc != 0:
                        failures.append(rc)
        finally:
            stop_ssh_control_master(control_host, control_path)
    final_rc = failures[0] if failures else 0
    if final_rc == 0:
        try:
            target_files = list_source_files(target, args.dataset_id)
            if target_files != files:
                final_rc = 2
                print(
                    f"[dashboard] manifest validation failed: source_files={len(files)} "
                    f"target_files={len(target_files)}",
                    flush=True,
                )
            else:
                print(f"[dashboard] manifest validation passed: files={len(files)}", flush=True)
        except Exception as exc:
            final_rc = 2
            print(f"[dashboard] manifest validation failed: {exc}", flush=True)
    if final_rc == 0 and args.marker_name:
        marker_relative_path = f".policy_action_conventions/{args.marker_name}.json"
        marker_rc, marker_summary = sync_root_relative_file(source, target, marker_relative_path)
        print(f"[dashboard] {marker_summary}", flush=True)
        if marker_rc:
            final_rc = marker_rc
    elapsed = time.time() - started
    progress.finish(
        state="completed" if final_rc == 0 else "failed",
        returncode=final_rc,
        error=None if final_rc == 0 else "manifest validation or shard transfer failed",
    )
    print(f"[dashboard] dataset transfer finished rc={final_rc} elapsed_s={elapsed:.1f}", flush=True)
    return final_rc


if __name__ == "__main__":
    raise SystemExit(main())

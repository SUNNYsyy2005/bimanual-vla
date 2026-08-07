#!/usr/bin/env python3
"""Detached task runner used by the Dashboard.

The Dashboard may restart while long-running training/norm/eval/policy commands are
still active.  This small wrapper keeps a durable exit-status file so the next
Dashboard process can classify the task as completed/failed instead of "lost".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-json", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command after --")

    exit_path = Path(args.exit_json).expanduser().resolve()
    cwd = str(Path(args.cwd).expanduser().resolve())
    started_at = now_iso()
    child: subprocess.Popen[Any] | None = None

    def forward(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    rc = 127
    error: str | None = None
    child_pid: int | None = None
    try:
        child = subprocess.Popen(command, cwd=cwd)
        child_pid = int(child.pid)
        rc = int(child.wait())
    except BaseException as exc:  # keep a durable reason even for launch errors
        error = repr(exc)
        rc = 128 if isinstance(exc, KeyboardInterrupt) else 127
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        payload: dict[str, Any] = {
            "returncode": rc,
            "started_at": started_at,
            "finished_at": now_iso(),
            "runner_pid": os.getpid(),
            "child_pid": child_pid,
            "command": command,
        }
        if error:
            payload["error"] = error
        try:
            atomic_json(exit_path, payload)
        except Exception as write_exc:  # noqa: BLE001 - last-resort stderr
            print(f"failed to write task exit status {exit_path}: {write_exc}", file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

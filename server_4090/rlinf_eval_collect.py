#!/usr/bin/env python3
"""Monitor a Slurm rollout and collect its JSON/video artifacts from shared storage."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile


def collect_artifacts(host: str, remote_output: str, local_output: Path) -> None:
    local_output.mkdir(parents=True, exist_ok=True)
    root = local_output.resolve()
    # The generated evaluation directory is server-owned, never a client path.
    command = "tar -C " + shlex.quote(remote_output) + " -cf - ."
    with subprocess.Popen(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command], stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
            for item in archive:
                target = (root / item.name).resolve()
                if root not in target.parents or not item.isfile() or target.suffix.lower() not in {".json", ".jsonl", ".mp4", ".webm", ".gif"}:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        if process.wait() != 0:
            raise RuntimeError("RLInf evaluation artifact collection failed")
    if not (root / "result.json").is_file():
        raise RuntimeError("rollout completed without result.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-json", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-output", required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    args = parser.parse_args()
    command = json.loads(base64.urlsafe_b64decode(args.command_json))
    status = subprocess.call(command)
    if status:
        return status
    collect_artifacts(args.host, args.remote_output, args.local_output)
    print("[dashboard] RLInf rollout results and videos collected", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

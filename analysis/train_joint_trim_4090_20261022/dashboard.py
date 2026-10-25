"""Authenticated Dashboard requests; token is supplied only through the environment."""
import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def api(path, body=None):
    request = urllib.request.Request("http://127.0.0.1:8090"+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer "+os.environ["BIMANUAL_VLA_SERVER_TOKEN"],
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Dashboard HTTP {error.code}: {error.read().decode()}") from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/api/status")
    parser.add_argument("--body", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = api(args.path, json.loads(args.body.read_text()) if args.body else None)
    if args.path == "/api/status":
        result = dict(config={k: result.get("config", {}).get(k) for k in
            ("dataset_root", "checkpoint_base_dir", "base_checkpoint", "openpi_python", "assets_base_dir")},
            datasets=[{k: d.get(k) for k in ("dataset_id", "id", "total_episodes", "total_frames", "arm_mode")}
                for d in result.get("datasets", []) if "joint" in str(d.get("dataset_id", d.get("id", "")))],
            tasks=[{k: t.get(k) for k in ("id", "kind", "status", "dataset_id", "exp_name")}
                for t in result.get("tasks", []) if t.get("status") in ("running", "queued", "pending")])
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

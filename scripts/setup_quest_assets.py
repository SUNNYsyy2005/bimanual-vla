#!/usr/bin/env python
"""Download pinned upstream URDF/meshes/APK only. Does not run ROS or touch CAN."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gzip
import json
from pathlib import Path
import time
import urllib.request

from bimanual_vla.collection.quest import DEFAULT_MODEL_DIR, SOURCE_COMMIT

REPO = "https://api.github.com/repos/agilexrobotics/questVR_ws"
MODEL_PREFIX = "src/Piper_ros/src/piper_description/"


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={
            "User-Agent": "bimanual-vla-quest-setup", "Accept-Encoding": "gzip"}), timeout=30) as response:
        data = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_MODEL_DIR.parents[3])
    args = parser.parse_args()
    root = args.destination.expanduser().resolve()
    tree = get_json(f"{REPO}/git/trees/{SOURCE_COMMIT}?recursive=1")
    wanted = [entry for entry in tree["tree"] if entry["type"] == "blob" and (
        entry["path"] == MODEL_PREFIX + "urdf/piper_description.urdf"
        or entry["path"].startswith(MODEL_PREFIX + "meshes/")
        or entry["path"] in ("src/Piper_ros/LICENSE", "src/oculus_reader/APK/teleop-debug.apk", "README.md"))]

    def download(entry):
        path = root / entry["path"]
        for attempt in range(3):
            try:
                if path.exists():
                    data = path.read_bytes()
                else:
                    url = f"https://raw.githubusercontent.com/agilexrobotics/questVR_ws/{SOURCE_COMMIT}/{entry['path']}"
                    try:
                        with urllib.request.urlopen(url, timeout=20) as response:
                            data = response.read()
                    except Exception:
                        blob = get_json(f"{REPO}/git/blobs/{entry['sha']}")
                        data = base64.b64decode(blob["content"])
                digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if digest != entry["sha"]:
                    raise RuntimeError(f"asset differs from pinned upstream: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                print(f"Verified {entry['path']}", flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, wanted))
    (root / ".quest-source.json").write_text(json.dumps({"commit": SOURCE_COMMIT, "files": wanted}, indent=2))
    print(f"Model: {root / MODEL_PREFIX}")
    print(f"APK: {root / 'src/oculus_reader/APK/teleop-debug.apk'}")


if __name__ == "__main__":
    main()

"""Verify a copied 20k inference checkpoint against the 4090 source manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["checkpoint_step"] != 20000 or manifest["status"] != "source_verified_awaiting_target_transfer":
        raise ValueError("unexpected source manifest")
    root = args.directory.resolve()
    expected = {item["path"]: item for item in manifest["files"]}
    actual = {path.relative_to(root).as_posix(): path for directory in (root / "params", root / "assets")
              for path in directory.rglob("*") if path.is_file()}
    if set(actual) != set(expected):
        raise ValueError(f"file set mismatch: missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}")
    for relative, item in expected.items():
        path = actual[relative]
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"file mismatch: {relative}")
    print(json.dumps({"verified": True, "checkpoint_step": 20000,
                      "files": len(expected), "bytes": sum(item["bytes"] for item in expected)}))


if __name__ == "__main__":
    main()

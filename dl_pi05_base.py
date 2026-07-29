#!/usr/bin/env python3
"""Parallel downloader for gs://openpi-assets/checkpoints/pi05_base/params
via the public HTTPS endpoint (storage.googleapis.com). gcsfs recursive get
stalls on the ocdbt dir, so we curl each object directly and place it in the
openpi cache so maybe_download() short-circuits."""
import concurrent.futures as cf
import json
import pathlib
import subprocess
import sys

CACHE_ROOT = pathlib.Path.home() / ".cache" / "openpi"
FILELIST = pathlib.Path.home() / "pi05_base_filelist.json"


def download_one(gcs_path: str, size: int) -> tuple[str, bool, str]:
    url = f"https://storage.googleapis.com/{gcs_path}"
    dest = CACHE_ROOT / gcs_path  # gcs_path starts with "openpi-assets/..."
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Skip if already fully downloaded.
    if dest.exists() and dest.stat().st_size == size:
        return (gcs_path, True, "cached")
    cmd = [
        "curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "3",
        "--connect-timeout", "30", "-o", str(dest), url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return (gcs_path, False, f"curl rc={r.returncode}: {r.stderr[:200]}")
    actual = dest.stat().st_size if dest.exists() else -1
    if actual != size:
        return (gcs_path, False, f"size mismatch got={actual} want={size}")
    return (gcs_path, True, f"{actual/1e6:.1f}MB")


def main():
    info = json.loads(FILELIST.read_text())
    total = sum(s for _, s in info)
    print(f"Downloading {len(info)} files, {total/1e9:.2f} GB -> {CACHE_ROOT}")
    ok = 0
    fail = []
    # 6 parallel: matches the number of large blobs.
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(download_one, p, s): p for p, s in info}
        for fut in cf.as_completed(futs):
            path, success, msg = fut.result()
            short = path.split("/params/")[-1]
            if success:
                ok += 1
                print(f"  [OK {ok}/{len(info)}] {short}  ({msg})")
            else:
                fail.append((path, msg))
                print(f"  [FAIL] {short}  {msg}")
    print(f"\nDone: {ok}/{len(info)} ok, {len(fail)} failed")
    if fail:
        for p, m in fail:
            print("  FAILED:", p, m)
        sys.exit(1)


if __name__ == "__main__":
    main()

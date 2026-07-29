#!/usr/bin/env python3
"""Resume pi05_base params from the HF mirror using HTTP range requests.

This complements dl_chunks.py: it reuses the existing GCS cache/chunk files under
~/.cache/openpi/openpi-assets/checkpoints/pi05_base/params, but downloads the
missing byte ranges from hf-mirror.com, which is much faster from the 4x4090 box
than storage.googleapis.com after GCS anonymous throttling kicks in.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import pathlib
import subprocess
import sys
from collections import defaultdict

CACHE_ROOT = pathlib.Path.home() / ".cache" / "openpi"
FILELIST = pathlib.Path.home() / "pi05_base_filelist.json"
HF_BASE = "https://hf-mirror.com/robotgeneralist/openpi_checkpoint_mirrors2/resolve/main"
CHUNKS_PER_FILE = 16
MAX_PARALLEL = 8


def hf_url(gcs_path: str) -> str:
    # GCS: openpi-assets/checkpoints/pi05_base/params/...
    # HF : pi05_base/params/...
    rel = gcs_path.removeprefix("openpi-assets/checkpoints/")
    return f"{HF_BASE}/{rel}"


def run_curl(url: str, start: int, end: int, out: pathlib.Path) -> tuple[bool, str]:
    """Append bytes [start,end] from url to out.tmp, then append to out."""
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.unlink(missing_ok=True)
    cmd = [
        "curl", "-L", "-sS", "--fail", "--retry", "10", "--retry-delay", "2",
        "--connect-timeout", "20", "--max-time", "1800",
        "--range", f"{start}-{end}", "-o", str(tmp), url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False, f"curl rc={r.returncode}: {r.stderr[:200]}"
    want = end - start + 1
    got = tmp.stat().st_size if tmp.exists() else -1
    if got != want:
        tmp.unlink(missing_ok=True)
        return False, f"part size {got}!={want}"
    with open(out, "ab") as fout, open(tmp, "rb") as fin:
        fout.write(fin.read())
    tmp.unlink(missing_ok=True)
    return True, "ok"


def fetch_chunk(url: str, chunk_start: int, chunk_end: int, cpath: pathlib.Path) -> tuple[bool, str]:
    want = chunk_end - chunk_start + 1
    cpath.parent.mkdir(parents=True, exist_ok=True)
    have = cpath.stat().st_size if cpath.exists() else 0
    if have == want:
        return True, "cached"
    if have > want:
        cpath.unlink()
        have = 0
    # Resume the existing partial chunk instead of redownloading it.
    return run_curl(url, chunk_start + have, chunk_end, cpath)


def main() -> int:
    info = dict(json.loads(FILELIST.read_text()))
    todo_files = []
    jobs = []
    for gcs_path, size in info.items():
        size = int(size)
        dest = CACHE_ROOT / gcs_path
        if dest.exists() and dest.stat().st_size == size:
            continue
        todo_files.append((gcs_path, size, dest))
        url = hf_url(gcs_path)
        chunk_size = (size + CHUNKS_PER_FILE - 1) // CHUNKS_PER_FILE
        for i, start in enumerate(range(0, size, chunk_size)):
            end = min(start + chunk_size - 1, size - 1)
            cpath = dest.with_suffix(dest.suffix + f".chunk{i:03d}")
            want = end - start + 1
            have = cpath.stat().st_size if cpath.exists() else 0
            if have == want:
                continue
            jobs.append((url, start, end, cpath, want, have))

    if not todo_files:
        print("All final files are already complete.", flush=True)
        return 0
    print(f"Incomplete final files: {len(todo_files)}", flush=True)
    for gcs_path, size, dest in todo_files:
        print(f"  {dest.name}: final={dest.stat().st_size if dest.exists() else 0}/{size}", flush=True)
    remain = sum(max(0, want - min(have, want)) for *_prefix, want, have in jobs)
    print(f"Range jobs to fetch: {len(jobs)}, missing bytes in chunks: {remain} ({remain/1024**3:.2f} GiB), parallel={MAX_PARALLEL}", flush=True)

    failed = []
    done = 0
    by_file = defaultdict(int)
    with cf.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = {ex.submit(fetch_chunk, url, start, end, cpath): (url, start, end, cpath) for url, start, end, cpath, _want, _have in jobs}
        for fut in cf.as_completed(futs):
            ok, msg = fut.result()
            done += 1
            cpath = futs[fut][3]
            by_file[cpath.name.split('.chunk', 1)[0]] += 1
            if not ok:
                failed.append((futs[fut], msg))
                print(f"[FAIL {done}/{len(jobs)}] {cpath.name}: {msg}", flush=True)
            elif done % 4 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] chunks completed", flush=True)
    if failed:
        print(f"FAILED chunks: {len(failed)}; not assembling.", flush=True)
        for job, msg in failed[:10]:
            print("  ", job[3], msg, flush=True)
        return 1

    print("Assembling final files...", flush=True)
    for gcs_path, size, dest in todo_files:
        size = int(size)
        chunk_size = (size + CHUNKS_PER_FILE - 1) // CHUNKS_PER_FILE
        starts = list(range(0, size, chunk_size))
        tmp = dest.with_suffix(dest.suffix + ".assembling")
        with open(tmp, "wb") as fout:
            for i, start in enumerate(starts):
                end = min(start + chunk_size - 1, size - 1)
                cpath = dest.with_suffix(dest.suffix + f".chunk{i:03d}")
                want = end - start + 1
                if not cpath.exists() or cpath.stat().st_size != want:
                    raise RuntimeError(f"bad chunk {cpath}: {cpath.stat().st_size if cpath.exists() else 'missing'} != {want}")
                with open(cpath, "rb") as fin:
                    fout.write(fin.read())
        actual = tmp.stat().st_size
        if actual != size:
            print(f"ASSEMBLY MISMATCH {dest.name}: {actual}!={size}", flush=True)
            return 1
        tmp.replace(dest)
        for i in range(len(starts)):
            dest.with_suffix(dest.suffix + f".chunk{i:03d}").unlink(missing_ok=True)
        print(f"  assembled {dest.name}: {actual/1024**2:.1f} MiB OK", flush=True)
    print("All files complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

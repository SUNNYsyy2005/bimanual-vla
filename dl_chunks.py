#!/usr/bin/env python3
"""Chunked parallel downloader using HTTP byte-range requests.
GCS anonymous throttles single connections, but allows many parallel ranges.
Re-downloads only the incomplete files from the filelist, in N ranges each,
then assembles. Verifies final size."""
import concurrent.futures as cf
import json
import pathlib
import subprocess
import sys

CACHE_ROOT = pathlib.Path.home() / ".cache" / "openpi"
FILELIST = pathlib.Path.home() / "pi05_base_filelist.json"
BASE_URL = "https://storage.googleapis.com"
CHUNKS_PER_FILE = 16
MAX_PARALLEL = 24  # total concurrent curl range requests across all files


def fetch_range(url: str, start: int, end: int, out: pathlib.Path) -> tuple[bool, str]:
    """Download bytes [start, end] inclusive to `out`. Skip if already correct size."""
    want = end - start + 1
    if out.exists() and out.stat().st_size == want:
        return (True, "cached")
    cmd = [
        "curl", "-sS", "--fail", "--retry", "8", "--retry-delay", "3",
        "--connect-timeout", "30", "--max-time", "1800",
        "--range", f"{start}-{end}", "-o", str(out), url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return (False, f"rc={r.returncode}: {r.stderr[:150]}")
    got = out.stat().st_size if out.exists() else -1
    if got != want:
        return (False, f"size {got}!={want}")
    return (True, "ok")


def main():
    info = dict(json.loads(FILELIST.read_text()))
    # Find incomplete files.
    todo = []
    for gcs_path, size in info.items():
        dest = CACHE_ROOT / gcs_path
        have = dest.stat().st_size if dest.exists() else 0
        if have != size:
            todo.append((gcs_path, size, dest))
    if not todo:
        print("Nothing to do — all files complete.")
        return
    print(f"Incomplete files: {len(todo)}")

    # Build all range jobs across all files.
    jobs = []  # (url, start, end, chunk_path, dest, size, chunk_idx, n_chunks)
    for gcs_path, size, dest in todo:
        url = f"{BASE_URL}/{gcs_path}"
        chunk_size = (size + CHUNKS_PER_FILE - 1) // CHUNKS_PER_FILE
        n = 0
        starts = list(range(0, size, chunk_size))
        for i, start in enumerate(starts):
            end = min(start + chunk_size - 1, size - 1)
            cpath = dest.with_suffix(dest.suffix + f".chunk{i:03d}")
            jobs.append((url, start, end, cpath))
            n += 1
        print(f"  {gcs_path.split('/params/')[-1][-24:]}: {size/1e6:.0f} MB -> {n} chunks")

    print(f"Total range jobs: {len(jobs)}, parallelism: {MAX_PARALLEL}")
    failed = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = {ex.submit(fetch_range, u, s, e, c): (u, s, e, c) for (u, s, e, c) in jobs}
        for fut in cf.as_completed(futs):
            ok, msg = fut.result()
            done += 1
            if not ok:
                failed.append((futs[fut], msg))
                print(f"  [FAIL {done}/{len(jobs)}] {msg}")
            elif done % 8 == 0:
                print(f"  [{done}/{len(jobs)}] ...")
    if failed:
        print(f"\n{len(failed)} range jobs FAILED — aborting assembly")
        for (job, msg) in failed[:10]:
            print("  ", job[3].name, msg)
        sys.exit(1)

    # Assemble each file from its chunks.
    print("\nAssembling files...")
    for gcs_path, size, dest in todo:
        chunk_size = (size + CHUNKS_PER_FILE - 1) // CHUNKS_PER_FILE
        starts = list(range(0, size, chunk_size))
        tmp = dest.with_suffix(dest.suffix + ".assembling")
        with open(tmp, "wb") as fout:
            for i in range(len(starts)):
                cpath = dest.with_suffix(dest.suffix + f".chunk{i:03d}")
                with open(cpath, "rb") as fin:
                    fout.write(fin.read())
        actual = tmp.stat().st_size
        if actual != size:
            print(f"  ASSEMBLY MISMATCH {dest.name}: {actual}!={size}")
            sys.exit(1)
        tmp.replace(dest)
        # Clean up chunks.
        for i in range(len(starts)):
            dest.with_suffix(dest.suffix + f".chunk{i:03d}").unlink(missing_ok=True)
        print(f"  assembled {dest.name[-24:]}: {actual/1e6:.0f} MB OK")

    print("\nAll files complete.")


if __name__ == "__main__":
    main()

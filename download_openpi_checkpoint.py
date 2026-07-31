#!/usr/bin/env python3
"""Parallel, resumable downloader for public OpenPI checkpoints.

Features:
- recursively lists a gs:// checkpoint through the public GCS JSON API;
- downloads different objects and byte ranges concurrently;
- resumes every partially downloaded range after interruption;
- tries the Hugging Face mirror first and falls back to GCS (`--source auto`);
- verifies object sizes and GCS MD5 hashes before atomic installation.

Example:
    python download_openpi_checkpoint.py \
      --checkpoint gs://openpi-assets/checkpoints/pi05_droid \
      --source auto --workers 16 --chunks-per-file 16

Then serve from the printed local directory instead of gs://.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_REPO = "robotgeneralist/openpi_checkpoint_mirrors2"
GCS_JSON_API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
GCS_OBJECT_URL = "https://storage.googleapis.com/{bucket}/{name}"


def parse_checkpoint(value: str) -> tuple[str, str]:
    if value.startswith("gs://"):
        parsed = urlparse(value)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/").rstrip("/")
    else:
        bucket = "openpi-assets"
        name = value.strip("/")
        prefix = name if name.startswith("checkpoints/") else f"checkpoints/{name}"
    if not bucket or not prefix:
        raise ValueError(f"invalid checkpoint path: {value!r}")
    return bucket, prefix


def read_json_url(url: str, *, timeout: int = 60) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "bimanual-vla-checkpoint-downloader/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def list_gcs_objects(bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    page_token = None
    while True:
        query = {"prefix": prefix.rstrip("/") + "/", "fields": "items(name,size,md5Hash,crc32c),nextPageToken"}
        if page_token:
            query["pageToken"] = page_token
        url = GCS_JSON_API.format(bucket=quote(bucket, safe="")) + "?" + urlencode(query)
        payload = read_json_url(url)
        for item in payload.get("items", []):
            size = int(item.get("size", 0))
            if size > 0 and not item["name"].endswith("/"):
                objects.append(
                    {
                        "name": item["name"],
                        "size": size,
                        "md5": item.get("md5Hash"),
                        "crc32c": item.get("crc32c"),
                    }
                )
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    if not objects:
        raise RuntimeError(f"no public GCS objects found under gs://{bucket}/{prefix}")
    return sorted(objects, key=lambda item: item["name"])


def checkpoint_destination(cache_root: Path, bucket: str, prefix: str) -> Path:
    return cache_root / bucket / prefix


def load_or_create_manifest(
    manifest_path: Path,
    *,
    bucket: str,
    prefix: str,
    refresh: bool,
) -> list[dict[str, Any]]:
    if manifest_path.exists() and not refresh:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("bucket") == bucket and payload.get("prefix") == prefix:
            return list(payload["objects"])
    objects = list_gcs_objects(bucket, prefix)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bucket": bucket,
        "prefix": prefix,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "objects": objects,
    }
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    return objects


def md5_base64(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def source_urls(
    source: str,
    *,
    bucket: str,
    object_name: str,
    hf_endpoint: str,
    hf_repo: str,
) -> list[str]:
    gcs_url = GCS_OBJECT_URL.format(bucket=quote(bucket, safe=""), name=quote(object_name, safe="/"))
    rel = object_name.removeprefix("checkpoints/")
    hf_url = f"{hf_endpoint.rstrip('/')}/{hf_repo}/resolve/main/{quote(rel, safe='/')}"
    if source == "gcs":
        return [gcs_url]
    if source == "hf-mirror":
        return [hf_url]
    return [hf_url, gcs_url]


def range_layout(size: int, chunks_per_file: int, min_chunk_bytes: int) -> list[tuple[int, int]]:
    if size <= min_chunk_bytes or chunks_per_file <= 1:
        return [(0, size - 1)]
    chunk_size = max(min_chunk_bytes, (size + chunks_per_file - 1) // chunks_per_file)
    return [(start, min(size - 1, start + chunk_size - 1)) for start in range(0, size, chunk_size)]


def part_path(parts_root: Path, relative: Path, index: int) -> Path:
    return parts_root / relative.parent / f"{relative.name}.part-{index:04d}"


def seed_parts_from_partial(final_path: Path, parts: list[tuple[int, int, Path]]) -> None:
    """Reuse a legacy partial final file as completed/partial range files."""
    if not final_path.exists():
        return
    available = final_path.stat().st_size
    expected = parts[-1][1] + 1
    if available <= 0 or available >= expected:
        return
    print(f"  reusing legacy partial file: {final_path} ({available}/{expected} bytes)", flush=True)
    with final_path.open("rb") as source:
        remaining = available
        for start, end, path in parts:
            if remaining <= 0:
                break
            want = end - start + 1
            take = min(want, remaining)
            path.parent.mkdir(parents=True, exist_ok=True)
            copied = 0
            with path.open("wb") as target:
                while copied < take:
                    block = source.read(min(8 * 1024 * 1024, take - copied))
                    if not block:
                        raise RuntimeError(
                            f"partial file ended early while seeding {path}: "
                            f"copied {copied}/{take} bytes"
                        )
                    target.write(block)
                    copied += len(block)
            remaining -= copied
    final_path.unlink()


def curl_range(url: str, start: int, end: int, output: Path) -> tuple[bool, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    want = end - start + 1
    have = output.stat().st_size if output.exists() else 0
    if have == want:
        return True, "cached"
    if have > want:
        output.unlink()
        have = 0
    temp = output.with_suffix(output.suffix + ".incoming")
    temp.unlink(missing_ok=True)
    request_start = start + have
    command = [
        "curl", "-L", "--fail", "--silent", "--show-error",
        "--retry", "12", "--retry-all-errors", "--retry-delay", "2",
        "--connect-timeout", "20", "--max-time", "3600",
        "--range", f"{request_start}-{end}",
        "--output", str(temp),
        url,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        temp.unlink(missing_ok=True)
        return False, f"curl rc={result.returncode}: {result.stderr.strip()[:240]}"
    expected_new = end - request_start + 1
    actual_new = temp.stat().st_size if temp.exists() else -1
    if actual_new != expected_new:
        temp.unlink(missing_ok=True)
        return False, f"server did not honor byte range: got {actual_new}, expected {expected_new}"
    with output.open("ab") as destination, temp.open("rb") as incoming:
        shutil.copyfileobj(incoming, destination, length=8 * 1024 * 1024)
    temp.unlink(missing_ok=True)
    actual = output.stat().st_size
    if actual != want:
        return False, f"resumed part size {actual}, expected {want}"
    return True, "ok"


def download_part(urls: list[str], start: int, end: int, output: Path) -> tuple[bool, str, str]:
    messages = []
    for url in urls:
        ok, message = curl_range(url, start, end, output)
        if ok:
            return True, message, url
        messages.append(f"{url}: {message}")
    return False, " | ".join(messages), ""


def assemble_object(
    final_path: Path,
    parts: list[tuple[int, int, Path]],
    *,
    expected_size: int,
    expected_md5: str | None,
    keep_parts: bool,
) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp = final_path.with_suffix(final_path.suffix + ".assembling")
    with temp.open("wb") as output:
        for start, end, path in parts:
            want = end - start + 1
            if not path.exists() or path.stat().st_size != want:
                raise RuntimeError(f"bad/missing range {path}: expected {want} bytes")
            with path.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=8 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    if temp.stat().st_size != expected_size:
        raise RuntimeError(f"assembled size mismatch for {final_path}: {temp.stat().st_size} != {expected_size}")
    if expected_md5:
        actual_md5 = md5_base64(temp)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {final_path}: {actual_md5} != {expected_md5}")
    temp.replace(final_path)
    if not keep_parts:
        for _, _, path in parts:
            path.unlink(missing_ok=True)


def acquire_lock(lock_path: Path, force_unlock: bool) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if force_unlock:
        lock_path.unlink(missing_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner = lock_path.read_text(encoding="utf-8", errors="replace") if lock_path.exists() else "unknown"
        raise RuntimeError(f"another downloader may be active; lock={lock_path}, owner={owner!r}") from exc
    os.write(fd, f"pid={os.getpid()} host={os.uname().nodename}\n".encode())
    return fd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_droid")
    ap.add_argument("--cache-root", type=Path, default=Path.home() / ".cache" / "openpi")
    ap.add_argument("--output", type=Path, default=None, help="override local checkpoint directory")
    ap.add_argument("--source", choices=("auto", "hf-mirror", "gcs"), default="auto")
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT))
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunks-per-file", type=int, default=16)
    ap.add_argument("--min-chunk-mib", type=int, default=16)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--refresh-manifest", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--keep-parts", action="store_true")
    ap.add_argument("--force-unlock", action="store_true")
    args = ap.parse_args()

    if args.workers <= 0 or args.chunks_per_file <= 0 or args.min_chunk_mib <= 0:
        ap.error("workers, chunks-per-file, and min-chunk-mib must be positive")
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required")

    bucket, prefix = parse_checkpoint(args.checkpoint)
    checkpoint_root = (args.output or checkpoint_destination(args.cache_root.expanduser(), bucket, prefix)).expanduser()
    manifest_path = args.manifest or checkpoint_root.parent / f".{checkpoint_root.name}.manifest.json"
    objects = load_or_create_manifest(
        manifest_path,
        bucket=bucket,
        prefix=prefix,
        refresh=args.refresh_manifest,
    )
    total_bytes = sum(int(item["size"]) for item in objects)
    print(f"Checkpoint: gs://{bucket}/{prefix}")
    print(f"Objects: {len(objects)}, total={total_bytes / 1024**3:.2f} GiB")
    print(f"Destination: {checkpoint_root}")
    print(f"Manifest: {manifest_path}")
    if args.list_only:
        for item in objects:
            print(f"{int(item['size']):12d}  {item['name']}")
        return 0

    lock_path = checkpoint_root.parent / f".{checkpoint_root.name}.download.lock"
    lock_fd = acquire_lock(lock_path, args.force_unlock)
    parts_root = checkpoint_root.parent / f".{checkpoint_root.name}.download_parts"
    min_chunk_bytes = args.min_chunk_mib * 1024 * 1024
    try:
        object_plans = []
        jobs = []
        completed_bytes = 0
        for item in objects:
            object_name = item["name"]
            size = int(item["size"])
            relative = Path(object_name).relative_to(prefix)
            final_path = checkpoint_root / relative
            if final_path.exists() and final_path.stat().st_size == size:
                if item.get("md5") and md5_base64(final_path) != item["md5"]:
                    print(f"  checksum mismatch, re-downloading: {relative}")
                else:
                    completed_bytes += size
                    continue
            ranges = range_layout(size, args.chunks_per_file, min_chunk_bytes)
            parts = [(start, end, part_path(parts_root, relative, index)) for index, (start, end) in enumerate(ranges)]
            seed_parts_from_partial(final_path, parts)
            urls = source_urls(
                args.source,
                bucket=bucket,
                object_name=object_name,
                hf_endpoint=args.hf_endpoint,
                hf_repo=args.hf_repo,
            )
            object_plans.append((item, relative, final_path, parts))
            for start, end, path in parts:
                want = end - start + 1
                have = path.stat().st_size if path.exists() else 0
                completed_bytes += min(have, want)
                if have != want:
                    jobs.append((urls, start, end, path, relative))

        if not object_plans:
            print("All checkpoint files are already complete and verified.")
            print(f"Use: --policy.dir={checkpoint_root}")
            return 0

        missing_bytes = max(0, total_bytes - completed_bytes)
        print(
            f"Pending objects={len(object_plans)}, range jobs={len(jobs)}, "
            f"remaining≈{missing_bytes / 1024**3:.2f} GiB, workers={args.workers}",
            flush=True,
        )
        failures = []
        done = 0
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(download_part, urls, start, end, path): (relative, start, end, path)
                for urls, start, end, path, relative in jobs
            }
            for future in futures.as_completed(future_map):
                relative, start, end, path = future_map[future]
                done += 1
                try:
                    ok, message, used_url = future.result()
                except Exception as exc:
                    ok, message, used_url = False, repr(exc), ""
                if not ok:
                    failures.append((relative, start, end, message))
                    print(f"[FAIL {done}/{len(jobs)}] {relative} {start}-{end}: {message}", flush=True)
                elif done % 8 == 0 or done == len(jobs):
                    endpoint = "hf" if args.hf_endpoint in used_url else "gcs"
                    print(f"[{done}/{len(jobs)}] ranges complete (last={endpoint})", flush=True)
        if failures:
            print(f"{len(failures)} range(s) failed. Re-run the same command to resume.", file=sys.stderr)
            return 1

        print("Assembling and verifying objects...", flush=True)
        for item, relative, final_path, parts in object_plans:
            assemble_object(
                final_path,
                parts,
                expected_size=int(item["size"]),
                expected_md5=item.get("md5"),
                keep_parts=args.keep_parts,
            )
            print(f"  OK {relative} ({int(item['size']) / 1024**2:.1f} MiB)", flush=True)
        if not args.keep_parts:
            shutil.rmtree(parts_root, ignore_errors=True)
        print("Checkpoint download complete.")
        print(f"Use: uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir={checkpoint_root} --port=8000")
        return 0
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

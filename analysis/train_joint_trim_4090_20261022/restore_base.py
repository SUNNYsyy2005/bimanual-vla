"""Stage and hash-check the user's archived base weights onto NVMe, atomically."""
import hashlib
import json
import os
from pathlib import Path
import uuid

SOURCE = Path("/mnt/sda1_16T/sunny/tsq-archive/checkpoints/base-models/pi05_base")
DESTINATION = Path("/mnt/nvme0n1p4_1T/sunny/tsq-storage/checkpoints/base-models/pi05_base")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(16*1024*1024):
            value.update(data)
    return value.hexdigest()


def main():
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    assert (SOURCE / "params/_METADATA").is_file()
    staging = DESTINATION.with_name(".pi05_base.staging-"+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    records = []
    for source in sorted(SOURCE.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(SOURCE)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = hashlib.sha256()
        with source.open("rb") as reader, target.open("xb") as writer:
            while data := reader.read(16*1024*1024):
                source_hash.update(data)
                writer.write(data)
        expected = source_hash.hexdigest()
        assert digest(target) == expected, relative
        records.append(dict(path=str(relative), bytes=target.stat().st_size, sha256=expected))
        print("VERIFIED", relative, target.stat().st_size, flush=True)
    report = dict(source=str(SOURCE), destination=str(DESTINATION), files=records,
                  total_bytes=sum(r["bytes"] for r in records))
    Path(__file__).with_name("base_restore.json").write_text(json.dumps(report, indent=2))
    os.rename(staging, DESTINATION)
    print("COMPLETED", DESTINATION, report["total_bytes"], flush=True)


if __name__ == "__main__":
    main()

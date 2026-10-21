"""Create a training-ready copy with only leading holds removed.

The Dashboard has per-episode in-place cropping, but no batch crop-to-new-dataset
operation. Reuse its rebuild implementation and validate before installation.
Default is a read-only dry run. Source data and checkpoint files are untouched.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

from bimanual_vla.data.check import check_dataset
from bimanual_vla.data.leading_idle import leading_hold_cut
from server_4090 import dataset_editor as de


class TrainingDatasetEditor(de.DatasetEditor):
    @staticmethod
    def _rewrite_raw(*args, **kwargs):
        # Legacy optional archives can contain pickle-backed metadata. Training
        # reads the checked parquet/video payloads; retain originals only in the
        # source dataset rather than copying stale or unsafe raw archives.
        return


def lossless_video_crop(source, target, start_frame, end_frame, *, fps=20):
    ffmpeg = Path(sys.executable).parent / "ffmpeg"
    if not ffmpeg.is_file():
        raise RuntimeError(f"Conda ffmpeg is unavailable: {ffmpeg}")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source), "-map", "0:v:0", "-an", "-vf",
                    f"trim=start_frame={start_frame}:end_frame={end_frame+1},setpts=PTS-STARTPTS",
                    "-vsync", "0", "-c:v", "libx264", "-preset", "fast", "-crf", "0",
                    "-threads", "1", str(target)], check=True, timeout=180)


def validate(root):
    errors = check_dataset(root)
    if errors:
        raise ValueError("\n".join(errors))
    return "structural validation passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    de._safe_dataset_id(destination.name)
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("destination must be a new dataset outside the source tree")
    info = json.loads((source / "meta/info.json").read_text())
    if (info.get("schema") != "joint" or info.get("contract_version") != 3
            or info.get("action_semantics") != "absolute_joint_position_opening_fraction"):
        raise ValueError("only verified absolute-joint datasets are supported")
    fps = float(info["fps"])
    entries, ranges = [], {}
    for index, path in sorted(de._parquet_paths(source).items()):
        table = pq.read_table(path, columns=["observation.state", "action"])
        q, a = (np.asarray(table[k].to_pylist()) for k in ("observation.state", "action"))
        result = leading_hold_cut(q, a)
        entry = dict(episode_index=index, original_frames=len(q), **result)
        entry.update(removed_seconds=result["start_frame"]/fps,
                     retained_frames=len(q)-result["start_frame"])
        entries.append(entry)
        ranges[index] = (result["start_frame"], len(q)-1)
    report = dict(source=str(source), destination=str(destination), fps=fps,
                  joint_threshold_rad=.001, gripper_threshold_fraction=.002,
                  keep_before_motion_frames=2, episodes=entries,
                  removed_frames=sum(x["start_frame"] for x in entries),
                  original_frames=sum(x["original_frames"] for x in entries),
                  retained_frames=sum(x["retained_frames"] for x in entries),
                  video_encoding="H264 lossless, no resize or padding",
                  legacy_raw_archives="retained in source dataset; omitted from training copy", applied=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    de._atomic_json(args.report, report)
    print(json.dumps(report), flush=True)
    if not args.apply:
        return
    validate(source)
    editor = TrainingDatasetEditor(dataset_root=destination.parent,
                             assets_base_dir=destination.parent / ".unused-trim-assets",
                             validate_staging=validate, validate_installed=lambda _: "unused")
    # This batch process uses lossless encoding instead of the interactive
    # editor's preview-oriented MP4 encoder. It does not modify a running API.
    de._crop_video = lossless_video_crop
    candidate = editor._rebuild_candidate(destination.name, [(source, None)], frame_ranges=ranges)
    try:
        split_path = candidate / "meta/train_test_split.json"
        if split_path.exists():
            split = json.loads(split_path.read_text())
            split["dataset_id"] = destination.name
            de._atomic_json(split_path, split)  # Break any source hardlink atomically.
        for filename in ("policy_contract.json", "info.json"):
            path = candidate / "meta" / filename
            value = json.loads(path.read_text())
            if "dataset_id" in value:
                value["dataset_id"] = destination.name
                de._atomic_json(path, value)
        de._atomic_json(candidate / "meta/leading_idle_trim.json", report)
        validate(candidate)
        for index, oldpath in de._parquet_paths(source).items():
            old = pq.read_table(oldpath)
            new = pq.read_table(de._parquet_paths(candidate)[index])
            start = ranges[index][0]
            # Every payload and acquisition timestamp is the exact source slice.
            for name in ["observation.state", "action"] + [k for k in old.column_names if "timestamp" in k and k != "timestamp"]:
                if not old[name].slice(start).equals(new[name]):
                    raise ValueError(f"payload or sensor timestamp changed: episode {index} {name}")
        # Atomic visibility only after all files and alignments have passed.
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(candidate, destination)
        report["applied"] = True
        de._atomic_json(destination / "meta/leading_idle_trim.json", report)
        de._atomic_json(args.report, report)
        print(f"COMPLETED {destination} frames={report['retained_frames']}", flush=True)
    finally:
        if candidate.exists():
            shutil.rmtree(candidate)


if __name__ == "__main__":
    main()

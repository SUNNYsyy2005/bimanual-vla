"""Create a training-ready copy with boundary holds removed.

The Dashboard has per-episode in-place cropping, but no batch crop-to-new-dataset
operation. Reuse its rebuild implementation and validate before installation.
Default is a read-only dry run. Source data and checkpoint files are untouched.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

from bimanual_vla.data.check import check_dataset
from bimanual_vla.data.leading_idle import leading_hold_cut, boundary_hold_range
from server_4090 import dataset_editor as de


class TrainingDatasetEditor(de.DatasetEditor):
    @staticmethod
    def _rewrite_raw(*args, **kwargs):
        # Legacy optional archives can contain pickle-backed metadata. Training
        # reads the checked parquet/video payloads; retain originals only in the
        # source dataset rather than copying stale or unsafe raw archives.
        return


def lossless_video_crop(source, target, start_frame, end_frame, *, fps=20,
                        remove_known_letterbox=False):
    ffmpeg = Path(sys.executable).parent / "ffmpeg"
    if not ffmpeg.is_file():
        raise RuntimeError(f"Conda ffmpeg is unavailable: {ffmpeg}")
    target.parent.mkdir(parents=True, exist_ok=True)
    filters = f"trim=start_frame={start_frame}:end_frame={end_frame+1},setpts=PTS-STARTPTS"
    if remove_known_letterbox:
        filters += ",crop=224:127:0:48:exact=1,scale=224:224:flags=bilinear"
    subprocess.run([str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source), "-map", "0:v:0", "-an", "-vf",
                    filters,
                    "-vsync", "0", "-c:v", "libx264", "-preset", "fast", "-crf", "0",
                    "-threads", "1", str(target)], check=True, timeout=180)


def verify_known_letterbox(source):
    """Require the historical 224x224 / 127-content-row format in every video."""
    import cv2
    cv2.setNumThreads(1)
    videos = sorted(source.glob("videos/**/*.mp4"))
    if not videos:
        raise ValueError("no videos to verify")
    for path in videos:
        cap = cv2.VideoCapture(str(path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError(f"empty video: {path}")
            for index in sorted(set([0, count//2, count-1])):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame.shape != (224, 224, 3):
                    raise ValueError(f"invalid video: {path} frame {index}")
                bands = np.concatenate([frame[:46].reshape(-1, 3), frame[178:].reshape(-1, 3)])
                if np.percentile(bands, 99.9) > 12 or np.mean(frame[50:173]) < 5:
                    raise ValueError(f"verified historical black bands absent: {path} frame {index}")
        finally:
            cap.release()
    return len(videos)


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
    parser.add_argument("--trim-trailing", action="store_true",
                        help="Also remove trailing holds, retaining two frames of endpoint context")
    parser.add_argument("--remove-known-letterbox", action="store_true",
                        help="Verify and remove historical 48:175 row letterbox; prepare 224x224 in one encode")
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
        if args.trim_trailing:
            result.update(boundary_hold_range(q, a))
        else:
            result.update(end_frame=len(q)-1, removed_leading=result["start_frame"], removed_trailing=0)
        entry = dict(episode_index=index, original_frames=len(q), **result)
        entry.update(removed_seconds=(result["removed_leading"]+result["removed_trailing"])/fps,
                     retained_frames=result["end_frame"]-result["start_frame"]+1)
        entries.append(entry)
        ranges[index] = (result["start_frame"], result["end_frame"])
    verified_videos = verify_known_letterbox(source) if args.remove_known_letterbox else None
    report = dict(source=str(source), destination=str(destination), fps=fps,
                  joint_threshold_rad=.001, gripper_threshold_fraction=.002,
                  keep_before_motion_frames=2, episodes=entries,
                  trim_trailing=args.trim_trailing, remove_known_letterbox=args.remove_known_letterbox,
                  verified_letterbox_videos=verified_videos,
                  removed_frames=sum(x["removed_leading"]+x["removed_trailing"] for x in entries),
                  removed_leading_frames=sum(x["removed_leading"] for x in entries),
                  removed_trailing_frames=sum(x["removed_trailing"] for x in entries),
                  original_frames=sum(x["original_frames"] for x in entries),
                  retained_frames=sum(x["retained_frames"] for x in entries),
                  video_encoding=("H264 lossless; crop rows 48:175, bilinear stretch to 224x224"
                                  if args.remove_known_letterbox else "H264 lossless, no resize or padding"),
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
    de._crop_video = partial(lossless_video_crop, remove_known_letterbox=args.remove_known_letterbox)
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
        if args.remove_known_letterbox:
            de._atomic_json(candidate / "meta/image_preprocessing.json", dict(
                source=str(source), crop_xyxy=[0, 48, 224, 175],
                resize="ffmpeg bilinear stretch to 224x224 during dataset generation",
                verified_videos=verified_videos, image_resize_mode="stretch"))
        validate(candidate)
        for index, oldpath in de._parquet_paths(source).items():
            old = pq.read_table(oldpath)
            new = pq.read_table(de._parquet_paths(candidate)[index])
            start, end = ranges[index]
            # Every payload and acquisition timestamp is the exact source slice.
            for name in ["observation.state", "action"] + [k for k in old.column_names if "timestamp" in k and k != "timestamp"]:
                if not old[name].slice(start, end-start+1).equals(new[name]):
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

"""Read-only audit of joint_dataset boundary holds, action alignment, and images."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from bimanual_vla.data.leading_idle import leading_hold_cut
from bimanual_vla.data.check import check_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    info = json.loads((args.source / "meta/info.json").read_text())
    errors = check_dataset(args.source)
    if errors:
        raise ValueError(errors)
    episodes, videos = [], []
    for path in sorted(args.source.glob("data/chunk-*/episode_*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action"])
        q, a = [np.asarray(table[k].to_pylist()) for k in ("observation.state", "action")]
        leading = leading_hold_cut(q, a)
        trailing = leading_hold_cut(q[::-1], a[::-1])
        start = leading["start_frame"]
        end = max(start, len(q)-1-trailing["start_frame"])
        episodes.append(dict(episode=int(path.stem.split("_")[-1]), frames=len(q),
            start=start, end=end, removed_leading=start, removed_trailing=len(q)-1-end,
            retained=end-start+1, all_idle=leading["all_idle"],
            action_state_mae=float(abs(a-q).mean()),
            action_next_state_mae=float(abs(a[:-1]-q[1:]).mean())))
    contact = []
    for path in sorted(args.source.glob("videos/**/*.mp4")):
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        samples = []
        for frame in sorted(set([0, count//2, count-1])):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"cannot decode {path} frame {frame}")
            gray = image.max(axis=2)
            nonblack_rows = np.flatnonzero(np.quantile(gray, .99, axis=1) > 12)
            nonblack_cols = np.flatnonzero(np.quantile(gray, .99, axis=0) > 12)
            samples.append(dict(frame=frame, shape=list(image.shape),
                nonblack_row_range=[int(nonblack_rows[0]), int(nonblack_rows[-1])],
                nonblack_col_range=[int(nonblack_cols[0]), int(nonblack_cols[-1])]))
            if path.stem in {"episode_000000", "episode_000032", "episode_000064"} and frame == count//2:
                thumbnail = cv2.resize(image, (224, 224))
                cv2.putText(thumbnail, path.parent.name.replace("observation.images.", ""),
                    (3, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
                cv2.putText(thumbnail, path.stem, (3, 31), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
                contact.append(thumbnail)
        capture.release()
        videos.append(dict(path=str(path.relative_to(args.source)), frames=count, samples=samples))
    result = dict(source=str(args.source), source_info=info, episodes=episodes, videos=videos,
        totals={k: sum(r[k] for r in episodes) for k in
                ("frames", "removed_leading", "removed_trailing", "retained")},
        all_idle_episodes=[r["episode"] for r in episodes if r["all_idle"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    if contact:
        cv2.imwrite(str(args.output.with_suffix(".jpg")), np.concatenate(contact, axis=1))
    print(json.dumps({k: result[k] for k in ("totals", "all_idle_episodes")}))
    print("REMOVED", [(r["episode"], r["removed_leading"], r["removed_trailing"], r["retained"]) for r in episodes])
    print("IMAGE_RANGES", sorted(set(tuple(s["nonblack_row_range"]) for r in videos for s in r["samples"])))


if __name__ == "__main__":
    main()

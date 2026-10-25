"""Verify installed training copy, preserving split and exact parquet payload slices."""
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from bimanual_vla.data.check import check_dataset


def main():
    root = Path(__file__).parent
    report = json.loads((root / "trim_report.json").read_text())
    assert report["applied"]
    source, target = (Path(report[k]) for k in ("source", "destination"))
    assert not check_dataset(target)
    old_split = json.loads((source / "meta/train_test_split.json").read_text())
    new_split = json.loads((target / "meta/train_test_split.json").read_text())
    for key in ("train_episodes", "test_episodes", "seed", "test_ratio"):
        assert new_split[key] == old_split[key], key
    assert new_split["dataset_id"] == target.name
    assert not (target / "meta/openpi_norm_stats.json").exists()
    records, contact = [], []
    cv2.setNumThreads(1)
    for episode in report["episodes"]:
        index, start, end = (episode[k] for k in ("episode_index", "start_frame", "end_frame"))
        name = f"episode_{index:06d}"
        old = pq.read_table(source / f"data/chunk-000/{name}.parquet")
        new = pq.read_table(target / f"data/chunk-000/{name}.parquet")
        count = end-start+1
        assert len(new) == count
        for key in ["observation.state", "action"] + [k for k in old.column_names if
                                                     "timestamp" in k and k != "timestamp"]:
            assert old[key].slice(start, count).equals(new[key]), (index, key)
        for camera in ("cam_high", "cam_right_wrist"):
            relative = f"videos/chunk-000/observation.images.{camera}/{name}.mp4"
            capture = cv2.VideoCapture(str(target / relative))
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == count, relative
            assert abs(capture.get(cv2.CAP_PROP_FPS)-20) < .001
            for frame in sorted(set([0, count//2, count-1])):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                assert ok and image.shape == (224, 224, 3), (relative, frame)
                records.append(dict(video=relative, frame=frame, top_mean=float(image[:20].mean()),
                                    bottom_mean=float(image[-20:].mean())))
                if index in (0, 32, 64) and frame == 0:
                    original = cv2.VideoCapture(str(source / relative))
                    original.set(cv2.CAP_PROP_POS_FRAMES, start)
                    source_ok, source_image = original.read()
                    original.release()
                    assert source_ok
                    cv2.putText(image, f"ep{index} {camera} new", (2, 15), cv2.FONT_HERSHEY_SIMPLEX,
                                .4, (0, 255, 0), 1)
                    contact.append(np.concatenate([source_image, image], axis=0))
            capture.release()
    cv2.imwrite(str(root / "prepared_images.jpg"), np.concatenate(contact, axis=1))
    result = dict(validated=True, episodes=len(report["episodes"]), frames=report["retained_frames"],
                  video_files=len(records)//3, sampled_video_frames=len(records),
                  exact_payload_slice_checks=True, preserved_split=new_split,
                  boundary_image_mean_min=min(min(r["top_mean"], r["bottom_mean"]) for r in records),
                  image_samples=records)
    (root / "dataset_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("image_samples", "preserved_split")}))


if __name__ == "__main__":
    main()

"""Replay one collected episode with two camera views and arm state.

Usage:
    python view_episode.py episodes_output_arm/ep_0000.npz

Keys while playing:
    SPACE  pause/resume
    Q/ESC  quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


RAD_TO_DEG = 180.0 / np.pi


def _to_bgr(frame_chw: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    frame = np.asarray(frame_chw).transpose(1, 2, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)


def _put_text(image, text, xy, color=(255, 255, 255), scale=0.65, thickness=2):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _make_panel(qpos: np.ndarray, task: str, instruction: str, high: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    view_w, view_h = 480, 480
    high = _to_bgr(high, (view_w, view_h))
    wrist = _to_bgr(wrist, (view_w, view_h))
    _put_text(high, "cam_high | third-person", (12, 30), (0, 255, 0))
    _put_text(wrist, "cam_wrist | first-person", (12, 30), (0, 255, 0))

    panel = np.zeros((700, view_w * 2, 3), dtype=np.uint8)
    panel[:view_h, :view_w] = high
    panel[:view_h, view_w:] = wrist
    panel[view_h:] = (35, 35, 35)

    _put_text(panel, f"task: {task}", (15, 515), (255, 255, 255), 0.62)
    _put_text(panel, f"instruction: {instruction}", (15, 545), (210, 210, 210), 0.55)
    _put_text(panel, "Output-arm feedback", (15, 580), (0, 220, 255), 0.68)

    labels = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]
    values = list(np.asarray(qpos[:6]) * RAD_TO_DEG) + [float(qpos[6]) * 1000.0]
    units = ["deg"] * 6 + ["mm"]
    x0 = 230
    for i, (label, value, unit) in enumerate(zip(labels, values, units)):
        x = x0 + (i % 4) * 180
        y = 580 + (i // 4) * 45
        _put_text(panel, f"{label}: {value:8.2f} {unit}", (x, y), (255, 255, 255), 0.55)
    return panel


def run(args):
    path = Path(args.episode)
    with np.load(path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        high = data["images_cam_high"]
        wrist = data["images_cam_wrist"]
        task = str(data["task_name"].item())
        instruction = str(data["instruction"].item())

        n = min(len(qpos), len(timestamps), len(high), len(wrist))
        if n == 0:
            raise ValueError("episode is empty")

        writer = None
        if args.save_video:
            writer = cv2.VideoWriter(
                args.save_video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (960, 700),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot open output video: {args.save_video}")

        paused = False
        try:
            for i in range(n):
                panel = _make_panel(qpos[i], task, instruction, high[i], wrist[i])
                if writer is not None:
                    writer.write(panel)

                cv2.imshow("Piper episode viewer", panel)
                if i + 1 < n and not paused:
                    delay = int(np.clip((timestamps[i + 1] - timestamps[i]) * 1000, 1, 200))
                else:
                    delay = 30

                while True:
                    key = cv2.waitKey(delay if not paused else 50) & 0xFF
                    if key in (ord("q"), 27):
                        return
                    if key == ord(" "):
                        paused = not paused
                        break
                    if not paused or key != 255:
                        break
        finally:
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="path to ep_XXXX.npz")
    ap.add_argument("--save-video", default=None, help="optional output MP4 path")
    ap.add_argument("--fps", type=int, default=30, help="FPS when saving MP4")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

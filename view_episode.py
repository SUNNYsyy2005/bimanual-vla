"""Replay one collected single-arm or bimanual episode.

Usage:
    python view_episode.py episodes_piper_v21/dual_arm/ep_0000.npz

Keys while playing:
    SPACE  pause/resume
    Q/ESC  quit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


RAD_TO_DEG = 180.0 / np.pi
SOURCE_ASPECT = 424 / 240
PANEL_WIDTH = 1280
INFO_HEIGHT = 180


def discover_episode_camera_keys(data: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return camera roles in display order for legacy and canonical NPZ files."""
    files = set(data.keys())
    if {"image", "wrist_image"}.issubset(files):
        return ("legacy_high", "legacy_wrist")

    declared: tuple[str, ...] = ()
    if "camera_keys" in files:
        declared = tuple(str(value) for value in np.asarray(data["camera_keys"]).reshape(-1))
    available = {
        key.removeprefix("images_")
        for key in files
        if key.startswith("images_") and not key.startswith("images_timestamp")
    }
    ordered = tuple(key for key in declared if key in available)
    if ordered:
        return ordered
    preferred = ("cam_high", "cam_left_wrist", "cam_right_wrist", "cam_wrist")
    return tuple(key for key in preferred if key in available) + tuple(
        sorted(available.difference(preferred))
    )


def load_episode(path: str | Path) -> dict[str, object]:
    """Load the arrays needed by the viewer without assuming one schema version."""
    episode_path = Path(path)
    with np.load(episode_path, allow_pickle=False) as data:
        camera_keys = discover_episode_camera_keys(data)
        if not camera_keys:
            raise ValueError(f"episode has no supported RGB camera arrays: {episode_path}")
        if camera_keys == ("legacy_high", "legacy_wrist"):
            cameras = {
                "cam_high": np.asarray(data["image"]),
                "cam_wrist": np.asarray(data["wrist_image"]),
            }
        else:
            cameras = {key: np.asarray(data[f"images_{key}"]) for key in camera_keys}
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        states = np.asarray(data["state"], dtype=np.float32) if "state" in data else None
        qpos_key = "joint_qpos" if "joint_qpos" in data else "qpos"
        joint_qpos = np.asarray(data[qpos_key], dtype=np.float32) if qpos_key in data else None
        task_key = "task" if "task" in data else "task_name"
        task = str(data[task_key].item()) if task_key in data else episode_path.stem
        instruction = str(data["instruction"].item()) if "instruction" in data else task
        schema = str(data["schema"].item()) if "schema" in data else "legacy"
        arm_mode = str(data["arm_mode"].item()) if "arm_mode" in data else (
            "bimanual" if joint_qpos is not None and joint_qpos.shape[-1] == 14 else "single"
        )

    lengths = [len(timestamps), *(len(frames) for frames in cameras.values())]
    if states is not None:
        lengths.append(len(states))
    if joint_qpos is not None:
        lengths.append(len(joint_qpos))
    frame_count = min(lengths)
    if frame_count <= 0:
        raise ValueError("episode is empty")
    return {
        "path": episode_path,
        "timestamps": timestamps[:frame_count],
        "states": None if states is None else states[:frame_count],
        "joint_qpos": None if joint_qpos is None else joint_qpos[:frame_count],
        "cameras": {key: frames[:frame_count] for key, frames in cameras.items()},
        "task": task,
        "instruction": instruction,
        "schema": schema,
        "arm_mode": arm_mode,
        "frame_count": frame_count,
    }


def _to_hwc_rgb(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = image.transpose(1, 2, 0)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"camera frame must be HWC/CHW RGB, got {image.shape}")
    return image.astype(np.uint8, copy=False)


def _restore_source_aspect(frame: np.ndarray) -> np.ndarray:
    image = _to_hwc_rgb(frame)
    height, width = image.shape[:2]
    current_aspect = width / height
    if current_aspect < SOURCE_ASPECT:
        crop_height = max(1, round(width / SOURCE_ASPECT))
        top = (height - crop_height) // 2
        image = image[top : top + crop_height]
    elif current_aspect > SOURCE_ASPECT:
        crop_width = max(1, round(height * SOURCE_ASPECT))
        left = (width - crop_width) // 2
        image = image[:, left : left + crop_width]
    return image


def _letterbox_bgr(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    image = _restore_source_aspect(frame)
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    x0 = (width - resized_w) // 2
    y0 = (height - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = cv2.cvtColor(
        resized, cv2.COLOR_RGB2BGR
    )
    return canvas


def _put_text(image, text, xy, color=(255, 255, 255), scale=0.62, thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _camera_title(key: str) -> str:
    return {
        "cam_high": "Overhead camera",
        "cam_wrist": "Wrist camera",
        "cam_left_wrist": "Left wrist camera",
        "cam_right_wrist": "Right wrist camera",
    }.get(key, key.replace("_", " ").title())


def _camera_layout(camera_keys: tuple[str, ...]) -> tuple[int, dict[str, tuple[int, int, int, int]]]:
    if len(camera_keys) >= 3:
        return 840, {
            camera_keys[0]: (0, 0, PANEL_WIDTH, 480),
            camera_keys[1]: (0, 480, PANEL_WIDTH // 2, 360),
            camera_keys[2]: (PANEL_WIDTH // 2, 480, PANEL_WIDTH // 2, 360),
        }
    if len(camera_keys) == 2:
        return 540, {
            camera_keys[0]: (0, 0, PANEL_WIDTH // 2, 540),
            camera_keys[1]: (PANEL_WIDTH // 2, 0, PANEL_WIDTH // 2, 540),
        }
    return 640, {camera_keys[0]: (0, 0, PANEL_WIDTH, 640)}


def _format_arm_rows(qpos: np.ndarray | None) -> list[str]:
    if qpos is None:
        return []
    values = np.asarray(qpos).reshape(-1)
    if values.size not in (7, 14):
        return [f"joint_qpos shape: {values.shape}"]
    sides = ("Arm",) if values.size == 7 else ("Left", "Right")
    rows = []
    for index, side in enumerate(sides):
        block = values[index * 7 : (index + 1) * 7]
        joints = "  ".join(f"J{i + 1} {block[i] * RAD_TO_DEG:+6.1f} deg" for i in range(6))
        rows.append(f"{side}: {joints}  Grip {block[6] * 1000:+6.1f} mm")
    return rows


def make_panel(
    camera_frames: Mapping[str, np.ndarray],
    qpos: np.ndarray | None,
    task: str,
    instruction: str,
    frame_index: int,
    frame_count: int,
) -> np.ndarray:
    camera_keys = tuple(camera_frames)
    camera_height, placements = _camera_layout(camera_keys)
    panel = np.full((camera_height + INFO_HEIGHT, PANEL_WIDTH, 3), (245, 245, 247), dtype=np.uint8)
    for key, (x, y, width, height) in placements.items():
        view = _letterbox_bgr(camera_frames[key], (width, height))
        panel[y : y + height, x : x + width] = view
        _put_text(panel, _camera_title(key), (x + 16, y + 30), (255, 255, 255), 0.65, 2)

    info_y = camera_height
    panel[info_y:] = (247, 247, 249)
    _put_text(panel, f"{task}  |  frame {frame_index + 1}/{frame_count}", (18, info_y + 34), (35, 35, 38), 0.7, 2)
    _put_text(panel, instruction, (18, info_y + 68), (92, 92, 98), 0.58, 1)
    for row, text in enumerate(_format_arm_rows(qpos)):
        _put_text(panel, text, (18, info_y + 108 + row * 34), (35, 35, 38), 0.52, 1)
    return panel


def run(args):
    episode = load_episode(args.episode)
    timestamps = episode["timestamps"]
    cameras = episode["cameras"]
    states = episode["states"]
    joint_qpos = episode["joint_qpos"]
    frame_count = int(episode["frame_count"])

    first_frames = {key: frames[0] for key, frames in cameras.items()}
    first_panel = make_panel(
        first_frames,
        None if joint_qpos is None else joint_qpos[0],
        str(episode["task"]),
        str(episode["instruction"]),
        0,
        frame_count,
    )
    writer = None
    if args.save_video:
        height, width = first_panel.shape[:2]
        writer = cv2.VideoWriter(
            args.save_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open output video: {args.save_video}")

    root = tk.Tk()
    root.title(f"Piper episode viewer - {episode['path'].name}")
    root.configure(bg="#111214")
    image_label = tk.Label(root, bg="#111214", bd=0, highlightthickness=0)
    image_label.pack(fill="both", expand=True)
    playback = {"index": 0, "paused": False, "closed": False, "photo": None}

    max_width = max(640, root.winfo_screenwidth() - 120)
    max_height = max(480, root.winfo_screenheight() - 180)

    def close_viewer(_event=None) -> str:
        if playback["closed"]:
            return "break"
        playback["closed"] = True
        if writer is not None:
            writer.release()
        root.destroy()
        return "break"

    def toggle_pause(_event=None) -> str:
        playback["paused"] = not playback["paused"]
        return "break"

    def show_next() -> None:
        if playback["closed"]:
            return
        if playback["paused"]:
            root.after(40, show_next)
            return
        index = int(playback["index"])
        panel = make_panel(
            {key: frames[index] for key, frames in cameras.items()},
            None if joint_qpos is None else joint_qpos[index],
            str(episode["task"]),
            str(episode["instruction"]),
            index,
            frame_count,
        )
        if writer is not None:
            writer.write(panel)
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        panel_h, panel_w = rgb.shape[:2]
        scale = min(max_width / panel_w, max_height / panel_h, 1.0)
        display_w = max(1, round(panel_w * scale))
        display_h = max(1, round(panel_h * scale))
        image = Image.fromarray(rgb)
        if (display_w, display_h) != (panel_w, panel_h):
            image = image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        playback["photo"] = photo

        if index + 1 >= frame_count:
            playback["paused"] = True
            return
        delay = int(np.clip((timestamps[index + 1] - timestamps[index]) * 1000, 1, 200))
        playback["index"] = index + 1
        root.after(delay, show_next)

    root.bind("<space>", toggle_pause)
    root.bind("q", close_viewer)
    root.bind("Q", close_viewer)
    root.bind("<Escape>", close_viewer)
    root.protocol("WM_DELETE_WINDOW", close_viewer)
    root.after(0, show_next)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", help="path to ep_XXXX.npz")
    parser.add_argument("--save-video", default=None, help="optional output MP4 path")
    parser.add_argument("--fps", type=int, default=20, help="FPS when saving MP4")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

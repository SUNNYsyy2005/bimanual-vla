"""Dashboard-style interactive 3D end-effector trajectory for the Tk analysis page."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np

from bimanual_vla.data.arm_geometry import arm_base_axis_signs, arm_base_origins
from bimanual_vla.data.analysis import end_effector_source_context
from bimanual_vla.data.episode_analysis import _resolve_arm_geometry

if TYPE_CHECKING:
    from bimanual_vla.data.analysis import AnalysisData


TRACK_COLORS = {"left": "#42c7ff", "right": "#ff9f68"}
AXIS_COLORS = ("#ff6f91", "#3ee6a8", "#42c7ff")


def end_effector_frames(data: AnalysisData) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Use the same effective bases and axis signs as the FK calculation."""
    metadata = data.metadata
    width = data.measured.shape[1] if data.measured.ndim == 2 else 0
    bimanual = width in {14, 16, 20}
    robot_type, dataset_origin = end_effector_source_context(data)
    offset, rotations = _resolve_arm_geometry(
        robot_type, metadata.get("arm_base_offset"), metadata.get("arm_base_rotations"),
        bimanual=bimanual, dataset_origin=dataset_origin,
    )
    if bimanual:
        origins = {side: np.asarray(point, dtype=np.float64) for side, point in arm_base_origins(offset).items()}
    else:
        side = str(metadata.get("arm_side") or "right")
        origins = {side: np.zeros(3, dtype=np.float64)}
    signs = arm_base_axis_signs(
        offset, bimanual=bimanual, robot_type=robot_type, dataset_origin=dataset_origin,
    )
    return (
        origins,
        {side: np.asarray(rotation, dtype=np.float64) for side, rotation in (rotations or {}).items()},
        {side: np.asarray(sign, dtype=np.float64) for side, sign in signs.items()},
    )


def project_trajectory_point(
    point: np.ndarray, center: np.ndarray, extent: float, width: int, height: int,
    yaw: float, pitch: float, zoom: float,
) -> tuple[float, float]:
    """Perspective projection matching the dashboard's dataset trajectory view."""
    x, y, z = (np.asarray(point, dtype=np.float64) - center) / extent
    yaw_x = math.cos(yaw) * x - math.sin(yaw) * y
    yaw_y = math.sin(yaw) * x + math.cos(yaw) * y
    screen_y = math.cos(pitch) * yaw_y - math.sin(pitch) * z
    depth = math.sin(pitch) * yaw_y + math.cos(pitch) * z
    perspective = 1.0 / max(0.2, 1.0 + depth * 0.35)
    scale = max(1.0, min(width - 72, height - 72) * 0.82 * zoom)
    return width / 2 + yaw_x * scale * perspective, height / 2 - screen_y * scale * perspective


def _finite_segments(values: np.ndarray, *, max_points: int = 1400) -> list[np.ndarray]:
    """Keep invalid gaps disconnected and bound Canvas work while dragging."""
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        return []
    valid = np.isfinite(values).all(axis=1)
    stride = max(1, math.ceil(len(values) / max_points))
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    result: list[np.ndarray] = []
    for first, last in zip(edges[::2], edges[1::2]):
        indices = np.arange(first, last, stride)
        if len(indices) and indices[-1] != last - 1:
            indices = np.r_[indices, last - 1]
        if len(indices):
            result.append(values[indices])
    return result


def draw_end_effector_trajectory(
    canvas: Any, data: AnalysisData, poses: dict[str, np.ndarray],
    start: int, end: int, selection: str, *,
    width: int, height: int, yaw: float, pitch: float, zoom: float,
    font_name: str,
    chunk_switch_indices: np.ndarray | None = None,
    _viewport_left: int = 0, _split_child: bool = False,
) -> None:
    """Draw selected history, sent chunk changes, bases and world/local axes."""
    canvas.create_rectangle(_viewport_left, 0, _viewport_left + width, height,
                            fill="#07121f", outline="")
    sides = ("left", "right") if selection == "Both arms" else (selection.split()[0].lower(),)
    tracks: dict[str, np.ndarray] = {}
    for side in sides:
        raw = np.asarray(poses.get(f"{side}_measured", np.empty((0, 3))), dtype=np.float64)
        if raw.ndim == 2 and raw.shape[1] == 3:
            tracks[side] = raw[start : end + 1]
    valid_tracks = {side: values[np.isfinite(values).all(axis=1)] for side, values in tracks.items()}
    if not any(len(values) for values in valid_tracks.values()):
        canvas.create_text(_viewport_left + width / 2, height / 2,
                           text="No end-effector trajectory for this selection",
                           fill="#8fa9c3", font=(font_name, 11))
        return

    origins, rotations, signs = end_effector_frames(data)
    calibrated_pair = {"left", "right"}.issubset(origins)
    if selection == "Both arms" and not calibrated_pair and all(len(valid_tracks.get(side, ())) for side in ("left", "right")):
        gap = 8
        left_width = (width - gap) // 2
        for side, viewport_left, viewport_width in (
            ("left", _viewport_left, left_width),
            ("right", _viewport_left + left_width + gap, width - left_width - gap),
        ):
            draw_end_effector_trajectory(
                canvas, data, poses, start, end, f"{side.title()} arm",
                width=viewport_width, height=height, yaw=yaw, pitch=pitch,
                zoom=zoom, font_name=font_name,
                chunk_switch_indices=chunk_switch_indices,
                _viewport_left=viewport_left, _split_child=True,
            )
        divider = _viewport_left + left_width + gap / 2
        canvas.create_line(divider, 0, divider, height, fill="#24415d", width=2)
        canvas.create_text(_viewport_left + width / 2, height - 27,
                           text="Separate aligned arm frames · base offset unavailable",
                           fill="#ffd166", font=(font_name, 9))
        if chunk_switch_indices is not None and len(chunk_switch_indices):
            canvas.create_text(_viewport_left + 10, 32,
                               text="Yellow rings: sent chunk switches",
                               fill="#ffd166", anchor="w", font=(font_name, 8))
        canvas.create_text(_viewport_left + width / 2, height - 10,
                           text=f"Drag to rotate · Wheel to zoom · {zoom:.2f}x",
                           fill="#8fa9c3", font=(font_name, 9))
        return

    display_origins = origins or {
        side: np.zeros(3, dtype=np.float64)
        for side, values in valid_tracks.items() if len(values)
    }
    bounds = [values for values in valid_tracks.values() if len(values)]
    bounds.extend(origin[None, :] for origin in display_origins.values())
    all_points = np.concatenate(bounds)
    minimum, maximum = np.min(all_points, axis=0), np.max(all_points, axis=0)
    extent = max(float(np.max(maximum - minimum)), 0.08)
    center = (minimum + maximum) / 2.0

    def project(point: np.ndarray) -> tuple[float, float]:
        x, y = project_trajectory_point(point, center, extent, width, height, yaw, pitch, zoom)
        return x + _viewport_left, y

    def segment(start_point: np.ndarray, end_point: np.ndarray, color: str, **kwargs: Any) -> None:
        ax, ay = project(start_point)
        bx, by = project(end_point)
        canvas.create_line(ax, ay, bx, by, fill=color, **kwargs)

    grid_z = float(minimum[2])
    grid_size = extent * 0.62
    for step in range(-3, 4):
        offset = grid_size * step / 3.0
        segment(np.array([center[0] - grid_size, center[1] + offset, grid_z]),
                np.array([center[0] + grid_size, center[1] + offset, grid_z]), "#18324b")
        segment(np.array([center[0] + offset, center[1] - grid_size, grid_z]),
                np.array([center[0] + offset, center[1] + grid_size, grid_z]), "#18324b")

    axis_length = extent * 0.32
    if origins:
        world_origin = origins.get("left", next(iter(origins.values()), center))
        frames = [("World", world_origin, np.eye(3), np.ones(3))]
        frames.extend((side.capitalize(), origin, rotations.get(side, np.eye(3)), signs.get(side, np.ones(3)))
                      for side, origin in origins.items())
    else:
        # With no measured base offset, each viewport has its own origin.
        # The right Piper trace has already been rotated into aligned axes.
        frames = [("Aligned", origin, np.eye(3), np.ones(3)) for origin in display_origins.values()]
    for name, origin, rotation, axis_signs in frames:
        for axis, color in enumerate(AXIS_COLORS):
            tip = origin + axis_length * axis_signs[axis] * rotation[:, axis]
            segment(origin, tip, color, width=2)
            tx, ty = project(tip)
            canvas.create_text(tx + 4, ty - 4, text=f"{name} {'XYZ'[axis]}", fill=color,
                               anchor="sw", font=(font_name, 8, "bold"))

    if calibrated_pair:
        segment(origins["left"], origins["right"], "#d9a441", width=2)
    for side, origin in display_origins.items():
        bx, by = project(origin)
        canvas.create_rectangle(bx - 4, by - 4, bx + 4, by + 4,
                                fill=TRACK_COLORS[side], outline="#f7fbff")
        canvas.create_text(bx + 9, by + 4, text=f"{side} {'base' if side in origins else 'origin'}",
                           fill="#f7fbff",
                           anchor="w", font=(font_name, 8))

    for side, values in tracks.items():
        finite = valid_tracks[side]
        if not len(finite):
            continue
        color = TRACK_COLORS[side]
        if side in display_origins:
            segment(display_origins[side], finite[0], color, dash=(3, 3))
        for run in _finite_segments(values):
            if len(run) > 1:
                coordinates = [coordinate for point in run for coordinate in project(point)]
                canvas.create_line(*coordinates, fill=color, width=3, smooth=False)
        latest_x, latest_y = project(finite[-1])
        canvas.create_oval(latest_x - 6, latest_y - 6, latest_x + 6, latest_y + 6,
                           fill=color, outline="#f7fbff", width=2)
        canvas.create_text(latest_x + 10, latest_y - 8, text=f"{side} latest", fill="#f7fbff",
                           anchor="sw", font=(font_name, 9))
        if chunk_switch_indices is not None:
            for row in chunk_switch_indices:
                local_index = int(row) - start
                if not 0 <= local_index < len(values) or not np.isfinite(values[local_index]).all():
                    continue
                switch_x, switch_y = project(values[local_index])
                canvas.create_oval(
                    switch_x - 5, switch_y - 5, switch_x + 5, switch_y + 5,
                    fill="", outline="#ffd166", width=2, dash=(3, 2),
                )

    title = f"{sides[0].title()} · samples {start + 1}–{end + 1} · m" if _split_child else f"Samples {start + 1}–{end + 1} · unit: m"
    canvas.create_text(_viewport_left + 10, 14, text=title, fill="#cfe2f7",
                       anchor="w", font=(font_name, 10))
    if chunk_switch_indices is not None and len(chunk_switch_indices) and not _split_child:
        canvas.create_text(_viewport_left + 10, 32,
                           text="Yellow rings: sent chunk switches",
                           fill="#ffd166", anchor="w", font=(font_name, 8))
    for index, side in enumerate(() if _split_child else sides):
        if side in valid_tracks and len(valid_tracks[side]):
            legend_x = _viewport_left + width - 135
            legend_y = 14 + index * 17
            canvas.create_line(legend_x, legend_y, legend_x + 16, legend_y,
                               fill=TRACK_COLORS[side], width=3)
            canvas.create_text(legend_x + 21, legend_y, text=side, fill="#cfe2f7",
                               anchor="w", font=(font_name, 9))
    if data.measured.shape[1] in {14, 16, 20} and not calibrated_pair and not _split_child:
        canvas.create_text(_viewport_left + 10, height - 27,
                           text="Base offset unavailable: single arm in aligned frame",
                           fill="#ffd166", anchor="w", font=(font_name, 9))
    if not _split_child:
        canvas.create_text(_viewport_left + 10, height - 10,
                           text=f"Drag to rotate · Wheel to zoom · {zoom:.2f}x",
                           fill="#8fa9c3", anchor="w", font=(font_name, 9))

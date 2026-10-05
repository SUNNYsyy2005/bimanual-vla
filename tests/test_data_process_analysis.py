from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from bimanual_vla.data.analysis import (
    compute_metrics,
    compute_end_effector_positions,
    end_effector_source_context,
    load_analysis_data,
    policy_trajectory_series,
    scan_analysis_sources,
    selection_indices,
    sent_chunk_switch_indices,
    trajectory_chunk_switch_times,
    trajectory_motion_series,
)
from bimanual_vla.data.panel import DataProcessPanel, available_plot_groups
from bimanual_vla.data.eef_view import (
    draw_end_effector_trajectory,
    end_effector_frames,
    project_trajectory_point,
)


class DataProcessAnalysisTest(unittest.TestCase):
    def test_motion_curves_and_chunk_switch_markers(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            chunks = root / "model_commands"
            chunks.mkdir(parents=True)
            timestamps = 100.0 + np.arange(11) * 0.05
            positions = np.zeros((11, 6))
            positions[:, 0] = (0, .1, .3, .6, 1.0, 1.1, 1.3, 1.6, 1.6, 2.0, 2.2)
            monotonic = 20.0 + np.arange(11) * 0.05
            monotonic[7] += .05  # Missed control period: break the derivative line.
            np.savez(
                root / "trajectory.npz", timestamp=timestamps,
                qpos=np.zeros((11, 7)), command_action=np.zeros((11, 7)),
                command_sent=np.ones(11, dtype=bool),
                command_hold=np.array([False] * 8 + [True] + [False] * 2),
                command_generation=np.array([1] * 4 + [2] * 7),
                command_queue_index=np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]),
                command_joints_rad=positions,
                command_monotonic_timestamp=monotonic,
            )
            (root / "metadata.json").write_text(json.dumps({"control_hz": 20}), encoding="utf-8")
            records = []
            for generation, arrived, values in ((1, 100.05, (0, 1, 3, 6)), (2, 100.25, (10, 11, 12, 13))):
                actions = np.zeros((4, 7))
                actions[:, 0] = values
                actions[:, 6] = generation * 100  # Gripper is not a joint velocity.
                filename = f"model_commands/{generation}.npz"
                np.savez(root / filename, raw_actions=actions)
                records.append({
                    "accepted": True, "generation": generation,
                    "arrived_at": arrived, "captured_at": arrived - .05,
                    "action_file": filename,
                    "protocol": {"schema": "joint", "arm_mode": "single", "action_hz": 20},
                })
            (root / "model_commands.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records), encoding="utf-8"
            )
            data = load_analysis_data(root)
            x, velocity, switches = trajectory_motion_series(
                data, 0, 10, stream="model_raw", order=1, joint_index=0,
            )
            np.testing.assert_allclose(switches, [.25])
            np.testing.assert_allclose(velocity[np.isfinite(velocity)], [20, 40, 60, 20, 20, 20])
            self.assertEqual(int(np.isnan(velocity).sum()), 3)  # Two chunk starts plus separator.
            self.assertTrue(np.all(np.isfinite(x)))
            _, acceleration, _ = trajectory_motion_series(
                data, 0, 10, stream="model_raw", order=2, joint_index=0,
            )
            np.testing.assert_allclose(acceleration[np.isfinite(acceleration)], [400, 400, 0, 0])
            _, model_norm, _ = trajectory_motion_series(
                data, 0, 10, stream="model_raw", order=1,
            )
            np.testing.assert_allclose(model_norm[np.isfinite(model_norm)],
                                       velocity[np.isfinite(velocity)])

            _, sent_velocity, sent_switches = trajectory_motion_series(
                data, 0, 10, stream="command_sent", order=1, joint_index=0,
            )
            np.testing.assert_allclose(sent_switches, [.2])
            model_marker_times, sent_marker_times = trajectory_chunk_switch_times(data, 0, 10)
            np.testing.assert_allclose(model_marker_times, [.25])
            np.testing.assert_allclose(sent_marker_times, [.2])
            np.testing.assert_array_equal(sent_chunk_switch_indices(data, 0, 10), [4])
            np.testing.assert_array_equal(sent_chunk_switch_indices(data, 4, 10), [4])
            self.assertTrue(np.isnan(sent_velocity[[0, 4, 7, 8, 9]]).all())
            np.testing.assert_allclose(sent_velocity[[1, 2, 3, 5, 6, 10]], [2, 4, 6, 2, 4, 4])
            _, sent_acceleration, _ = trajectory_motion_series(
                data, 0, 10, stream="command_sent", order=2, joint_index=0,
            )
            np.testing.assert_allclose(sent_acceleration[[2, 3, 6]], [40, 40, 40])
            self.assertTrue(np.isnan(sent_acceleration[[4, 5, 7, 8, 9, 10]]).all())

            panel = SimpleNamespace(
                data=data, signal_var=SimpleNamespace(get=lambda: "Joint L2 norm"),
                plot_var=SimpleNamespace(get=lambda: "Velocity"),
                chart_markers=[], chart_series_x=[],
            )
            chart_series, chart_x, unit = DataProcessPanel._make_series(panel, 0, 10)
            self.assertEqual(unit, "rad/s")
            self.assertEqual(
                [item[0] for item in chart_series],
                ["Policy output", "Smoothed policy", "Command sent"],
            )
            np.testing.assert_allclose(panel.chart_markers[0][1], [.25])
            np.testing.assert_allclose(panel.chart_markers[1][1], [.2])
            self.assertEqual(len(chart_series[0][1]), len(panel.chart_series_x[0]))
            self.assertEqual(len(chart_series[1][1]), len(panel.chart_series_x[1]))
            self.assertEqual(len(chart_x), sum(map(len, panel.chart_series_x)))
            np.testing.assert_allclose(chart_series[0][1][np.isfinite(chart_series[0][1])],
                                       model_norm[np.isfinite(model_norm)])
            np.testing.assert_allclose(
                chart_series[1][1][np.isfinite(chart_series[1][1])],
                chart_series[0][1][np.isfinite(chart_series[0][1])],
            )
            np.testing.assert_allclose(chart_series[2][1][np.isfinite(chart_series[2][1])],
                                       sent_velocity[np.isfinite(sent_velocity)])

            class Canvas:
                def __init__(self):
                    self.dashed = []
                    self.lines = []
                    self.ovals = []

                def delete(self, _what):
                    pass

                def winfo_width(self):
                    return 600

                def winfo_height(self):
                    return 300

                def create_line(self, *coords, **kwargs):
                    self.lines.append((coords, kwargs))
                    if kwargs.get("dash"):
                        self.dashed.append((coords, kwargs))

                def create_text(self, *_args, **_kwargs):
                    pass

                def create_rectangle(self, *_args, **_kwargs):
                    pass

                def create_oval(self, *coords, **kwargs):
                    self.ovals.append((coords, kwargs))

            canvas = Canvas()
            panel.chart = canvas
            panel.chart_series = chart_series
            panel.chart_x = chart_x
            panel.chart_y_label = unit
            panel.font_name = "Arial"
            DataProcessPanel._draw_chart(panel)
            switch_lines = [(line, kwargs) for line, kwargs in canvas.dashed if len(line) == 4 and line[1] == 40]
            self.assertEqual(len(switch_lines), 2)
            by_color = {kwargs["fill"]: line for line, kwargs in switch_lines}
            self.assertAlmostEqual(by_color["#1a73e8"][0], 58 + .25 / .5 * (600 - 58 - 18))
            self.assertAlmostEqual(by_color["#e76f51"][0], 58 + .2 / .5 * (600 - 58 - 18))
            self.assertTrue(any(kwargs.get("fill") == "#1a73e8" and len(line) > 4
                                for line, kwargs in canvas.lines))
            self.assertTrue(any(kwargs.get("fill") == "#e76f51" and len(line) > 4
                                for line, kwargs in canvas.lines))

            panel.plot_var = SimpleNamespace(get=lambda: "Acceleration")
            panel.chart_series, panel.chart_x, panel.chart_y_label = DataProcessPanel._make_series(panel, 0, 10)
            np.testing.assert_allclose(panel.chart_markers[0][1], [.25])
            np.testing.assert_allclose(panel.chart_markers[1][1], [.2])
            self.assertEqual(panel.chart_y_label, "rad/s²")
            self.assertEqual(len(panel.chart_series), 3)

            groups = available_plot_groups(data)
            self.assertEqual(groups["Trajectory"], ("Position", "Velocity", "Acceleration", "Tracking error"))
            self.assertEqual(groups["Chunk boundaries"], ("Boundary position jump", "Boundary velocity direction"))
            self.assertEqual(groups["End effector"], ("3D trajectory",))
            panel.signal_var = SimpleNamespace(get=lambda: data.names[0])
            panel.plot_var = SimpleNamespace(get=lambda: "Position")
            positions_series, positions_x, positions_unit = DataProcessPanel._make_series(panel, 0, 10)
            self.assertEqual([name for name, *_ in positions_series[:2]], ["Measured", "Recorded target"])
            self.assertIn("Policy output", [name for name, *_ in positions_series])
            self.assertIn("Smoothed policy", [name for name, *_ in positions_series])
            self.assertEqual([item[0] for item in panel.chart_markers], ["New prediction", "Sent chunk switch"])
            panel.chart_series, panel.chart_x, panel.chart_y_label = positions_series, positions_x, positions_unit
            canvas.dashed.clear()
            DataProcessPanel._draw_chart(panel)
            self.assertEqual(len([line for line, _kwargs in canvas.dashed if len(line) == 4 and line[1] == 40]), 2)
            panel.plot_var = SimpleNamespace(get=lambda: "Tracking error")
            error_series, _, _ = DataProcessPanel._make_series(panel, 0, 10)
            self.assertEqual(len(error_series), 1)
            self.assertEqual(len(panel.chart_markers), 2)

            positions_3d = np.column_stack((np.arange(11) * .01, np.zeros(11), np.zeros(11)))
            panel.plot_var = SimpleNamespace(get=lambda: "3D trajectory")
            panel.signal_var = SimpleNamespace(get=lambda: "Right arm")
            panel.pose_cache = {data.path: {"right_measured": positions_3d}}
            panel._selected_indices = lambda: (0, 10)
            panel.eef_yaw, panel.eef_pitch, panel.eef_zoom = 0, .45, 1
            DataProcessPanel._draw_chart(panel)
            self.assertEqual(sum(bool(kwargs.get("dash")) for _coords, kwargs in canvas.ovals), 1)

    def test_smoothed_policy_curve_reduces_short_horizon_acceleration_spike(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            chunks = root / "model_commands"
            chunks.mkdir(parents=True)
            timestamps = 100.0 + np.arange(8) * 0.05
            np.savez(
                root / "trajectory.npz",
                timestamp=timestamps,
                qpos=np.zeros((8, 7)),
                command_action=np.zeros((8, 7)),
                command_sent=np.ones(8, dtype=bool),
                command_hold=np.zeros(8, dtype=bool),
                command_generation=np.ones(8, dtype=np.int64),
                command_queue_index=np.arange(8),
                command_joints_rad=np.zeros((8, 6)),
                command_monotonic_timestamp=20.0 + np.arange(8) * 0.05,
            )
            (root / "metadata.json").write_text(
                json.dumps({"control_hz": 20, "rtc_execution_horizon": 8}),
                encoding="utf-8",
            )
            actions = np.zeros((8, 7), dtype=np.float32)
            actions[:, 0] = np.arange(8, dtype=np.float32) * 0.01
            actions[3, 0] += 0.025
            filename = "model_commands/1.npz"
            np.savez(root / filename, raw_actions=actions)
            (root / "model_commands.jsonl").write_text(
                json.dumps({
                    "accepted": True,
                    "generation": 1,
                    "arrived_at": 100.0,
                    "captured_at": 99.95,
                    "action_file": filename,
                    "protocol": {"schema": "joint", "arm_mode": "single", "action_hz": 20},
                }),
                encoding="utf-8",
            )
            data = load_analysis_data(root)
            _, raw_accel, _, _ = policy_trajectory_series(
                data, 0, 7, order=2, joint_index=0,
            )
            _, smooth_accel, _, _ = policy_trajectory_series(
                data, 0, 7, order=2, joint_index=0, smoothed=True,
            )
            raw_peak = np.nanmax(np.abs(raw_accel))
            smooth_peak = np.nanmax(np.abs(smooth_accel))
            self.assertLess(smooth_peak, raw_peak)

    def test_loads_episode_and_computes_selection_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            timestamps = np.arange(10, dtype=np.float64) / 20.0
            state = np.zeros((10, 14), dtype=np.float32)
            actions = state.copy()
            actions[:, 0] = np.linspace(0.0, 0.9, 10)
            np.savez(
                root / "ep_0001.npz",
                state=state,
                actions=actions,
                timestamps=timestamps,
                state_names=np.asarray([f"j{i}" for i in range(14)]),
                task=np.asarray("test"),
            )
            data = load_analysis_data(root / "ep_0001.npz")
            start, end = selection_indices(data, 0.1, 0.3)
            metrics = compute_metrics(data, start, end)
            self.assertEqual(data.kind, "episode")
            self.assertEqual(metrics["sample_count"], 5)
            self.assertAlmostEqual(metrics["control_hz"], 20.0)
            self.assertEqual(metrics["model_command_count"], 0)
            self.assertGreater(metrics["action_step_norm"]["p95"], 0.0)
            self.assertEqual(available_plot_groups(data), {
                "Trajectory": ("Position", "Tracking error"),
                "Timing": ("Control interval",),
                "End effector": ("3D trajectory",),
            })

    def test_dashboard_style_3d_eef_view_uses_recorded_20d_positions_and_bases(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep_0001.npz"
            state = np.zeros((4, 20), dtype=np.float64)
            state[:, :3] = [[0.10, 0.10, 0.30], [0.12, 0.12, 0.31],
                             [0.14, 0.14, 0.32], [0.16, 0.16, 0.33]]
            state[:, 10:13] = [[0.10, 0.10, 0.30], [0.12, 0.12, 0.31],
                                [0.14, 0.14, 0.32], [0.16, 0.16, 0.33]]
            names = [f"left_{name}" for name in ("eef_x", "eef_y", "eef_z", *[f"rot{i}" for i in range(6)], "gripper")]
            names += [f"right_{name}" for name in ("eef_x", "eef_y", "eef_z", *[f"rot{i}" for i in range(6)], "gripper")]
            np.savez(path, state=state, actions=state, timestamps=np.arange(4) * .05,
                     state_names=np.asarray(names), robot_type=np.asarray("piper_bimanual"),
                     dataset_origin=np.asarray("real"), arm_base_offset=np.asarray([.8, 0, 0]))
            data = load_analysis_data(path)
            poses = compute_end_effector_positions(data)
            np.testing.assert_allclose(poses["left_measured"][0], [.10, .10, .30])
            np.testing.assert_allclose(poses["right_measured"][0], [.70, -.10, .30])
            origins, _rotations, signs = end_effector_frames(data)
            np.testing.assert_allclose(origins["right"], [.8, 0, 0])
            np.testing.assert_allclose(signs["right"], [-1, -1, 1])

            class Canvas:
                def __init__(self):
                    self.lines = []
                    self.labels = []
                    self.ovals = []
                    self.rectangles = []

                def delete(self, _what):
                    self.lines.clear()
                    self.labels.clear()
                    self.ovals.clear()
                    self.rectangles.clear()

                def winfo_width(self):
                    return 600

                def winfo_height(self):
                    return 360

                def create_line(self, *coords, **kwargs):
                    self.lines.append((coords, kwargs))

                def create_text(self, *coords, **kwargs):
                    self.labels.append(kwargs.get("text"))

                def create_oval(self, *coords, **kwargs):
                    self.ovals.append((coords, kwargs))

                def create_rectangle(self, *coords, **kwargs):
                    self.rectangles.append((coords, kwargs))

            canvas = Canvas()
            draw_end_effector_trajectory(
                canvas, data, poses, 1, 3, "Both arms", width=600, height=360,
                yaw=0, pitch=.45, zoom=1, font_name="Arial",
            )
            self.assertEqual(len(canvas.ovals), 2)
            self.assertIn("left latest", canvas.labels)
            self.assertIn("right latest", canvas.labels)
            self.assertTrue(any(kwargs.get("fill") == "#42c7ff" and kwargs.get("width") == 3
                                for _coords, kwargs in canvas.lines))
            self.assertTrue(any(kwargs.get("fill") == "#ff9f68" and kwargs.get("width") == 3
                                for _coords, kwargs in canvas.lines))
            self.assertTrue(any(kwargs.get("fill") == "#d9a441" for _coords, kwargs in canvas.lines))
            self.assertTrue(any(kwargs.get("fill") == "#07121f" for _coords, kwargs in canvas.rectangles))
            self.assertIn("Samples 2–4 · unit: m", canvas.labels)

            panel_canvas = Canvas()
            panel = SimpleNamespace(
                chart=panel_canvas, data=data, pose_cache={data.path: poses},
                plot_var=SimpleNamespace(get=lambda: "3D trajectory"),
                signal_var=SimpleNamespace(get=lambda: "Both arms"),
                eef_yaw=0.0, eef_pitch=.45, eef_zoom=1.0, font_name="Arial",
                _selected_indices=lambda: (1, 3),
            )
            DataProcessPanel._draw_chart(panel)
            self.assertEqual(len(panel_canvas.ovals), 2)

            right_only = Canvas()
            draw_end_effector_trajectory(
                right_only, data, poses, 1, 3, "Right arm", width=600, height=360,
                yaw=.2, pitch=.45, zoom=1.2, font_name="Arial",
            )
            self.assertEqual(len(right_only.ovals), 1)
            self.assertIn("right latest", right_only.labels)
            self.assertNotIn("left latest", right_only.labels)
            gapped_poses = dict(poses)
            gapped_poses["left_measured"] = poses["left_measured"].copy()
            gapped_poses["left_measured"][2] = np.nan
            gapped_canvas = Canvas()
            draw_end_effector_trajectory(
                gapped_canvas, data, gapped_poses, 1, 3, "Left arm",
                width=600, height=360, yaw=0, pitch=.45, zoom=1, font_name="Arial",
            )
            self.assertFalse(any(len(coords) > 4 and kwargs.get("fill") == "#42c7ff" and kwargs.get("width") == 3
                                 for coords, kwargs in gapped_canvas.lines))
            center = np.zeros(3)
            near = project_trajectory_point(np.array([.2, 0, 0]), center, 1.0, 600, 360, 0, 0, 1)
            zoomed = project_trajectory_point(np.array([.2, 0, 0]), center, 1.0, 600, 360, 0, 0, 2)
            self.assertGreater(zoomed[0], near[0])

            draws = []
            panel = SimpleNamespace(
                plot_var=SimpleNamespace(get=lambda: "3D trajectory"),
                eef_yaw=0.0, eef_pitch=.45, eef_zoom=1.0, eef_drag_at=None,
                _draw_chart=lambda: draws.append(True),
            )
            DataProcessPanel._eef_drag_start(panel, SimpleNamespace(x=10, y=20))
            DataProcessPanel._eef_drag_move(panel, SimpleNamespace(x=20, y=30))
            self.assertAlmostEqual(panel.eef_yaw, .12)
            self.assertAlmostEqual(panel.eef_pitch, .57)
            DataProcessPanel._eef_drag_end(panel, None)
            self.assertIsNone(panel.eef_drag_at)
            DataProcessPanel._eef_mouse_wheel(panel, SimpleNamespace(num=4, delta=0))
            self.assertAlmostEqual(panel.eef_zoom, 1.1)
            self.assertEqual(len(draws), 2)

    def test_uncalibrated_real_piper_run_shows_both_forward_in_separate_frames(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            qpos = np.zeros((2, 14), dtype=np.float64)
            qpos[:, 0] = 1.565
            qpos[:, 7] = -1.565
            qpos[1, 1] = qpos[1, 8] = 1.0
            np.savez(root / "trajectory.npz", timestamp=np.array([1.0, 1.05]),
                     qpos=qpos, command_action=qpos)
            data = load_analysis_data(root)
            self.assertEqual(end_effector_source_context(data), ("piper", "real"))
            poses = compute_end_effector_positions(data)
            left = poses["left_measured"]
            right = poses["right_measured"]
            self.assertGreater(left[1, 1] - left[0, 1], 0.0)
            self.assertGreater(right[1, 1] - right[0, 1], 0.0)
            origins, _rotations, signs = end_effector_frames(data)
            self.assertFalse(origins)
            np.testing.assert_array_equal(signs["right"], [-1, -1, 1])

            class Canvas:
                def __init__(self):
                    self.rectangles = []
                    self.lines = []
                    self.labels = []

                def create_rectangle(self, *coords, **kwargs):
                    self.rectangles.append((coords, kwargs))

                def create_line(self, *coords, **kwargs):
                    self.lines.append((coords, kwargs))

                def create_text(self, *coords, **kwargs):
                    self.labels.append(kwargs.get("text"))

                def create_oval(self, *coords, **kwargs):
                    pass

            canvas = Canvas()
            draw_end_effector_trajectory(canvas, data, poses, 0, 1, "Both arms",
                                         width=600, height=360, yaw=0, pitch=.45,
                                         zoom=1, font_name="Arial")
            self.assertTrue(any("Separate aligned arm frames" in str(label) for label in canvas.labels))
            self.assertFalse(any(kwargs.get("fill") == "#d9a441" for _coords, kwargs in canvas.lines))
            dark_panels = [coords for coords, kwargs in canvas.rectangles if kwargs.get("fill") == "#07121f"]
            self.assertGreaterEqual(len(dark_panels), 2)
            traces = {
                kwargs.get("fill"): coords for coords, kwargs in canvas.lines
                if kwargs.get("width") == 3 and kwargs.get("fill") in {"#42c7ff", "#ff9f68"}
            }
            self.assertEqual(set(traces), {"#42c7ff", "#ff9f68"})
            self.assertEqual(np.sign(traces["#42c7ff"][3] - traces["#42c7ff"][1]),
                             np.sign(traces["#ff9f68"][3] - traces["#ff9f68"][1]))

    def test_loads_deployment_run_and_latency_records(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            (root / "model_commands").mkdir(parents=True)
            timestamps = 100.0 + np.arange(6, dtype=np.float64) / 20.0
            qpos = np.zeros((6, 14), dtype=np.float32)
            desired = np.full((6, 14), np.nan, dtype=np.float32)
            desired[2:] = 0.1
            np.savez(
                root / "trajectory.npz",
                timestamp=timestamps,
                qpos=qpos,
                command_action=desired,
                command_sent=np.array([False, False, True, True, True, True]),
                command_hold=np.array([False, False, False, True, False, False]),
                command_generation=np.arange(6),
                command_queue_index=np.arange(6),
            )
            (root / "metadata.json").write_text(json.dumps({"control_hz": 20}), encoding="utf-8")
            (root / "trajectory.jsonl").write_text(
                "\n".join(json.dumps({"blocked_reason": "", "execution_state": "executing"}) for _ in range(6)),
                encoding="utf-8",
            )
            records = []
            for index in range(3):
                records.append({
                    "captured_at": 100.0 + index * 0.05,
                    "_client_transport_timing": {
                        "model_inference_ms": 140.0,
                        "round_trip_ms": 220.0,
                    },
                })
            (root / "model_commands.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records), encoding="utf-8"
            )
            data = load_analysis_data(root)
            metrics = compute_metrics(data)
            self.assertEqual(data.kind, "deployment")
            self.assertEqual(metrics["hold_count"], 1)
            self.assertEqual(metrics["model_command_count"], 3)
            self.assertEqual(metrics["latency"]["round_trip_ms"]["median"], 220.0)

            sparse = replace(data, command_records=(
                {"captured_at": 100.0, "_client_transport_timing": {
                    "round_trip_ms": 220.0, "model_inference_ms": None,
                    "observation_upload_ms": None,
                }},
                {"captured_at": 100.05, "_client_transport_timing": {
                    "round_trip_ms": None, "model_inference_ms": 135.0,
                }},
                {"captured_at": 100.10, "_client_transport_timing": None},
                {"captured_at": 100.15, "_client_transport_timing": {
                    "round_trip_ms": "invalid", "model_inference_ms": float("inf"),
                }},
            ))
            panel = SimpleNamespace(
                data=sparse,
                signal_var=SimpleNamespace(get=lambda: "—"),
                plot_var=SimpleNamespace(get=lambda: "Inference latency"),
            )
            series, x, unit = DataProcessPanel._make_series(panel, 0, 5)
            self.assertEqual(unit, "milliseconds")
            self.assertEqual(len(x), 4)
            np.testing.assert_allclose(series[0][1], [220.0, np.nan, np.nan, np.nan])
            np.testing.assert_allclose(series[1][1], [np.nan, 135.0, np.nan, np.nan])
            self.assertTrue(np.isnan(series[2][1]).all())

    def test_end_effector_fk_and_discarded_actions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            (root / "model_commands").mkdir(parents=True)
            timestamps = 100.0 + np.arange(3, dtype=np.float64) / 20.0
            qpos = np.zeros((3, 14), dtype=np.float32)
            desired = np.zeros((3, 14), dtype=np.float32)
            np.savez(
                root / "trajectory.npz",
                timestamp=timestamps,
                qpos=qpos,
                command_action=desired,
                command_sent=np.ones(3, dtype=bool),
                command_hold=np.zeros(3, dtype=bool),
                command_generation=np.arange(3),
                command_queue_index=np.arange(3),
            )
            (root / "metadata.json").write_text("{}", encoding="utf-8")
            (root / "trajectory.jsonl").write_text(
                json.dumps({"blocked_reason": "dropped unsafe queued target: test", "execution_state": "executing"}) + "\n"
                + "\n".join(json.dumps({"blocked_reason": "", "execution_state": "executing"}) for _ in range(2)),
                encoding="utf-8",
            )
            (root / "model_commands.jsonl").write_text(
                json.dumps({"captured_at": 100.0, "accepted": False, "action_shape": [4, 14]}) + "\n"
                + json.dumps({"captured_at": 100.05, "accepted": True, "action_shape": [4, 14]}),
                encoding="utf-8",
            )
            data = load_analysis_data(root)
            metrics = compute_metrics(data)
            poses = compute_end_effector_positions(data)
            self.assertEqual(metrics["rejected_action_count"], 1)
            self.assertEqual(metrics["unsafe_drop_count"], 1)
            self.assertEqual(metrics["rejected_action_rows"], 4)
            self.assertEqual(metrics["discarded_action_count"], 5)
            self.assertEqual(poses["left_measured"].shape, (3, 3))
            self.assertTrue(np.isfinite(poses["left_measured"]).all())

    def test_legacy_piper_episode_without_origin_uses_real_arm_geometry(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep_0001.npz"
            state = np.zeros((1, 14), dtype=np.float32)
            np.savez(
                path,
                state=state,
                actions=state,
                timestamps=np.array([0.0]),
                robot_type=np.asarray("piper_bimanual"),
                arm_base_offset=np.asarray([0.8, 0.0, 0.0]),
            )
            data = load_analysis_data(path)
            poses = compute_end_effector_positions(data)
            self.assertLess(float(poses["right_measured"][0, 0]), 0.8)
            self.assertLess(float(poses["right_measured"][0, 1]), 0.0)

    def test_scans_both_source_types_without_model_command_chunks(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run" / "model_commands").mkdir(parents=True)
            np.savez(root / "run" / "trajectory.npz", timestamp=np.array([0.0]), qpos=np.zeros((1, 1)), command_action=np.zeros((1, 1)))
            np.savez(root / "run" / "model_commands" / "command_000001.npz", actions=np.zeros((1, 1)))
            (root / "dataset").mkdir()
            np.savez(root / "dataset" / "ep_0001.npz", state=np.zeros((1, 1)), actions=np.zeros((1, 1)), timestamps=np.array([0.0]))
            sources = scan_analysis_sources([root])
            self.assertEqual(len(sources), 2)
            self.assertTrue(any(path.name == "ep_0001.npz" for path in sources))
            self.assertTrue(any(path.name == "run" for path in sources))


if __name__ == "__main__":
    unittest.main()

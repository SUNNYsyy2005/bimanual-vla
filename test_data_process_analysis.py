from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from data_process_analysis import (
    compute_metrics,
    compute_end_effector_positions,
    load_analysis_data,
    scan_analysis_sources,
    selection_indices,
)


class DataProcessAnalysisTest(unittest.TestCase):
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

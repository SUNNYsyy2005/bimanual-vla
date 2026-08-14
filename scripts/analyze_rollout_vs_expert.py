#!/usr/bin/env python3
"""Compare a RoboTwin π0/π0.5 rollout action trace against a LeRobot expert episode.

This script is meant for debugging BC closed-loop drift.  It reads the JSONL
trace emitted by ``ROBOTWIN_ACTION_TRACE=1`` (preferably with
``ROBOTWIN_ACTION_TRACE_FULL=1``) and compares each executed chunk to the expert
``action`` rows at the same frame index in a LeRobot parquet episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _parse_indices(spec: str) -> np.ndarray:
    out: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            a, b = part.split(':', 1)
            out.extend(range(int(a), int(b)))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"empty index spec: {spec!r}")
    return np.asarray(out, dtype=np.int64)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _as_action_chunk(rec: dict, key_prefix: str) -> np.ndarray | None:
    full_key = f'{key_prefix}_full_exec'
    if full_key in rec:
        arr = np.asarray(rec[full_key], dtype=np.float64)
        return arr if arr.ndim == 2 else None
    # Fallback for older compact traces: compare first/last only.
    block = rec.get(key_prefix)
    if isinstance(block, dict) and 'first' in block and 'last_exec' in block:
        return np.asarray([block['first'], block['last_exec']], dtype=np.float64)
    return None


def _fmt_pair(values: Iterable[float]) -> str:
    return '->'.join(f'{float(v):.3g}' for v in values)


def compare_trace(
    trace_path: Path,
    episode_parquet: Path,
    *,
    action_key: str = 'raw',
    arm_indices: np.ndarray,
    gripper_indices: np.ndarray,
    max_rows: int | None = None,
) -> tuple[list[dict], dict]:
    trace_rows = _read_jsonl(trace_path)
    if max_rows is not None:
        trace_rows = trace_rows[:max_rows]
    df = pd.read_parquet(episode_parquet, columns=['observation.state', 'action'])
    states = np.stack(df['observation.state'].to_numpy()).astype(np.float64)
    actions = np.stack(df['action'].to_numpy()).astype(np.float64)

    out: list[dict] = []
    for rec in trace_rows:
        chunk = _as_action_chunk(rec, action_key)
        if chunk is None or len(chunk) == 0:
            continue
        t = int(rec.get('take_action_cnt', 0))
        n = int(min(len(chunk), max(0, len(actions) - t)))
        if n <= 0:
            break
        pred = np.asarray(chunk[:n], dtype=np.float64)
        expert = actions[t:t + n]
        state = np.asarray(rec.get('state', states[t]), dtype=np.float64).reshape(-1)
        if state.size < max(arm_indices.max(), gripper_indices.max()) + 1:
            state = states[t]

        arm_diff = pred[:, arm_indices] - expert[:, arm_indices]
        grip_diff = pred[:, gripper_indices] - expert[:, gripper_indices]
        state_err = state[arm_indices] - states[t, arm_indices]
        pred_step = np.diff(np.vstack([state[arm_indices], pred[:, arm_indices]]), axis=0)
        expert_step = np.diff(np.vstack([states[t, arm_indices], expert[:, arm_indices]]), axis=0)

        row = {
            'chunk_index': int(rec.get('chunk_index', len(out) + 1)),
            'take_action_cnt': t,
            'n': n,
            'state_err_arm_mean_abs': float(np.mean(np.abs(state_err))),
            'state_err_arm_max_abs': float(np.max(np.abs(state_err))),
            'arm_mse': float(np.mean(arm_diff ** 2)),
            'arm_mae': float(np.mean(np.abs(arm_diff))),
            'arm_max_abs': float(np.max(np.abs(arm_diff))),
            'gripper_mse': float(np.mean(grip_diff ** 2)),
            'gripper_mae': float(np.mean(np.abs(grip_diff))),
            'pred_step_max_abs': float(np.max(np.abs(pred_step))) if pred_step.size else 0.0,
            'expert_step_max_abs': float(np.max(np.abs(expert_step))) if expert_step.size else 0.0,
            'pred_left_gripper': _fmt_pair([pred[0, gripper_indices[0]], pred[-1, gripper_indices[0]]]),
            'pred_right_gripper': _fmt_pair([pred[0, gripper_indices[-1]], pred[-1, gripper_indices[-1]]]),
            'expert_left_gripper': _fmt_pair([expert[0, gripper_indices[0]], expert[-1, gripper_indices[0]]]),
            'expert_right_gripper': _fmt_pair([expert[0, gripper_indices[-1]], expert[-1, gripper_indices[-1]]]),
        }
        out.append(row)

    if out:
        arr = {k: np.asarray([r[k] for r in out], dtype=np.float64) for k in [
            'state_err_arm_mean_abs', 'state_err_arm_max_abs', 'arm_mse', 'arm_mae',
            'arm_max_abs', 'gripper_mse', 'gripper_mae', 'pred_step_max_abs', 'expert_step_max_abs'
        ]}
        summary = {
            'trace': str(trace_path),
            'episode_parquet': str(episode_parquet),
            'chunks': len(out),
            'first_state_err_gt_0p1': next((r['chunk_index'] for r in out if r['state_err_arm_mean_abs'] > 0.1), None),
            'first_arm_mse_gt_0p05': next((r['chunk_index'] for r in out if r['arm_mse'] > 0.05), None),
            'first_gripper_mse_gt_0p05': next((r['chunk_index'] for r in out if r['gripper_mse'] > 0.05), None),
        }
        for key, values in arr.items():
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_max'] = float(np.max(values))
    else:
        summary = {'trace': str(trace_path), 'episode_parquet': str(episode_parquet), 'chunks': 0}
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', required=True, type=Path, help='ROBOTWIN_ACTION_TRACE JSONL file')
    ap.add_argument('--episode-parquet', required=True, type=Path, help='LeRobot episode parquet')
    ap.add_argument('--action-key', default='raw', choices=['raw', 'safe'], help='trace action key to compare')
    ap.add_argument('--arm-indices', default='0:6,7:13')
    ap.add_argument('--gripper-indices', default='6,13')
    ap.add_argument('--max-rows', type=int)
    ap.add_argument('--out-csv', type=Path)
    ap.add_argument('--out-json', type=Path)
    ap.add_argument('--print-rows', type=int, default=40)
    args = ap.parse_args()

    rows, summary = compare_trace(
        args.trace,
        args.episode_parquet,
        action_key=args.action_key,
        arm_indices=_parse_indices(args.arm_indices),
        gripper_indices=_parse_indices(args.gripper_indices),
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.print_rows and rows:
        keys = [
            'chunk_index', 'take_action_cnt', 'state_err_arm_mean_abs', 'arm_mse',
            'arm_max_abs', 'gripper_mse', 'pred_left_gripper', 'pred_right_gripper',
            'expert_left_gripper', 'expert_right_gripper'
        ]
        print('\t'.join(keys))
        for row in rows[:args.print_rows]:
            vals = []
            for key in keys:
                value = row[key]
                vals.append(f'{value:.4g}' if isinstance(value, float) else str(value))
            print('\t'.join(vals))

    if args.out_csv and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Print a compact summary from compare_action_stats_4090.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_KEYS = (
    "checkpoint_step",
    "num_samples",
    "pred_abs_mean",
    "gt_abs_mean",
    "pred_abs_std",
    "gt_abs_std",
    "pred_norm_mean",
    "gt_norm_mean",
    "pred_l2_to_gt_l2_ratio",
    "pred_joint_abs_mean",
    "gt_joint_abs_mean",
    "pred_gripper_abs_mean",
    "gt_gripper_abs_mean",
    "pred_cosine_mean",
    "pred_cosine_std",
    "pred_mse_mean",
    "pred_mse_std",
    "pred_near_zero_frac_0p05",
)


def _head(values: Any, count: int = 5) -> Any:
    return values[:count] if isinstance(values, list) else values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Action-statistics JSON file")
    args = parser.parse_args()

    payload = json.loads(args.input.expanduser().read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints", {})
    if not isinstance(checkpoints, dict):
        raise ValueError("input JSON does not contain a checkpoints object")

    for name, stats in checkpoints.items():
        if not isinstance(stats, dict):
            continue
        print(name)
        for key in SUMMARY_KEYS:
            if key in stats:
                print(f"  {key}: {stats[key]}")
        for key in (
            "pred_step_abs_mean",
            "gt_step_abs_mean",
            "pred_step_l2_mean",
            "gt_step_l2_mean",
        ):
            if key in stats:
                print(f"  {key}_first5: {_head(stats[key])}")
        ranges = stats.get("pred_dim_range")
        if isinstance(ranges, list):
            print(f"  pred_dim_range_first14: {ranges[:14]}")
            grippers = [ranges[index] for index in (6, 13) if index < len(ranges)]
            print(f"  pred_dim_range_grippers: {grippers}")


if __name__ == "__main__":
    main()

"""Read-only jitter probe: demo vs. model-output second differences and action normalization scale.

Run on 4x4090 from any directory with the openpi Conda env. Paths are overridable by env vars.
"""
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

STORAGE = Path(os.environ.get("TSQ_STORAGE", "/mnt/nvme0n1p4_1T/sunny/tsq-storage"))
CKPT = Path(os.environ.get(
    "PROBE_CHECKPOINT",
    STORAGE / "checkpoints/openpi/pi05_piper_bimanual_lora/fix_trimstart_rtc35_h200_20261021/20000"))
DATA = Path(os.environ.get("PROBE_DATASET", STORAGE / "datasets/lerobot/fix_nopad_20261019"))
PRED = Path(os.environ.get(
    "PROBE_PREDICTIONS",
    Path.home() / "bimanual-vla/analysis/rtc_padding_fix_20261021/20000_predictions"))
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
PRED_NAME = re.compile(r"^(?P<obs>.+)_n(?P<steps>\d+)_s(?P<seed>\d+)_(?P<mode>no_prefix|oracle_prefix|self_prefix)$")


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["observation.state", "action"])
    state = np.asarray(table["observation.state"].to_pylist(), np.float64)
    action = np.asarray(table["action"].to_pylist(), np.float64)
    return state, action


def report_norm_stats() -> None:
    stats_files = sorted(CKPT.glob("assets/**/norm_stats.json"))
    if not stats_files:
        raise FileNotFoundError(f"no norm_stats.json under {CKPT}/assets")
    stats = json.loads(stats_files[0].read_text())["norm_stats"]["actions"]
    q01, q99 = np.asarray(stats["q01"])[:14], np.asarray(stats["q99"])[:14]
    print("norm_stats:", stats_files[0])
    print("rad per 1.0 normalized unit (quantile half-range), joints:",
          np.round((q99 - q01)[JOINTS] / 2, 4).tolist())


def report_demos() -> None:
    files = sorted((DATA / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episodes under {DATA}/data")
    d2, d1, lag = [], [], []
    for path in files:
        state, action = load_episode(path)
        d2.append(np.abs(np.diff(action[:, JOINTS], n=2, axis=0)).mean())
        d1.append(np.abs(np.diff(action[:, JOINTS], axis=0)).mean())
        lag.append(np.abs(action[:-1, JOINTS] - state[1:, JOINTS]).mean())
    print(f"episodes={len(files)} demo |d2|={np.mean(d2):.5f} rad |d1|={np.mean(d1):.5f} rad "
          f"|action[t]-state[t+1]|={np.mean(lag):.2e} (0 => labels are next measured state)")


def report_predictions() -> None:
    groups: dict[tuple[str, str, int], list[tuple[float, float, float]]] = {}
    for path in glob.glob(str(PRED / "*.npz")):
        match = PRED_NAME.match(Path(path).stem)
        if match is None:
            continue
        obs = match["obs"]
        split = "real" if obs.startswith("real_") else ("heldout" if obs.startswith(("ep12_", "ep16_")) else "train")
        d2 = np.abs(np.diff(np.load(path)["prediction"][:, JOINTS], n=2, axis=0))
        groups.setdefault((split, match["mode"], int(match["steps"])), []).append(
            (d2.mean(), d2[3:10].mean(), d2[10:].mean()))
    if not groups:
        raise FileNotFoundError(f"no prediction npz matched under {PRED}")
    print("prediction |d2| rad: (split, mode, denoise_steps) n  all / rows3-10 / rows10+")
    for key in sorted(groups):
        values = np.asarray(groups[key])
        print(key, len(values), np.round(values.mean(axis=0), 5).tolist())


def main() -> None:
    report_norm_stats()
    report_demos()
    report_predictions()


if __name__ == "__main__":
    main()

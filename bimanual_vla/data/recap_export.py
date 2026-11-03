"""Convert time-indexed RECAP deployment recordings into canonical LeRobot.

Only explicitly labelled success/failure runs are admitted. Deployment runs
keep their complete timeline for audit, while the exported training episode
removes invalid handoff/hold rows. Source segment boundaries are retained in
the sidecar annotation, so compaction never joins two disconnected control
segments into one policy chunk.
"""

import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .contract import build_delivery_actions_with_gripper_targets
from .lerobot import Pi0LeRobotDatasetWriter, default_eef_names
from .recap import RecapEpisode


def nearest_indices(times, queries):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("source timestamps must be finite and strictly increasing")
    right = np.searchsorted(times, queries).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(times[left] - queries) <= np.abs(times[right] - queries), left, right)


def recorded_control_hz(path):
    value = json.loads((Path(path) / "metadata.json").read_text()).get("control_hz")
    if value is None or not np.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{path}: recording has no valid control_hz")
    return float(value)


def _compact_values(annotation, values):
    """Remove invalid resampled rows while retaining control segment bounds.

    An action at frame *i* targets the next observation. The final valid row
    before an invalid gap therefore cannot be retained: its action would
    otherwise jump over the gap. The source index is used only to create a
    fresh epoch at every removed interval; it is never exposed as a new
    physical timestamp in the exported episode.
    """
    valid = annotation.valid.copy()
    if len(valid) > 1:
        # The exporter marks the terminal observation invalid because it has
        # no following action. It is still a legitimate next-state target for
        # the preceding row, so only real invalid gaps suppress that row.
        next_valid = valid[1:].copy()
        next_valid[-1] = True
        valid[:-1] &= next_valid
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise ValueError("RECAP recording has fewer than two valid contiguous frames")

    owner = annotation.owner[indices]
    source_epoch = annotation.epoch[indices]
    epoch = np.empty(len(indices), dtype=np.int64)
    epoch[0] = int(source_epoch[0])
    for position in range(1, len(indices)):
        previous = position - 1
        boundary = (
            indices[position] != indices[previous] + 1
            or source_epoch[position] != source_epoch[previous]
            or owner[position] != owner[previous]
        )
        epoch[position] = max(
            int(source_epoch[position]),
            int(epoch[previous]) + int(boundary),
        )
    compact_annotation = RecapEpisode(owner, epoch, np.ones(len(indices), dtype=bool), annotation.outcome)

    compact = dict(values)
    for key in ("states", "actions", "timestamps", "state_timestamps", "action_timestamps"):
        compact[key] = np.asarray(values[key])[indices]
    compact["images"] = {key: np.asarray(image)[indices] for key, image in values["images"].items()}
    compact["image_timestamps"] = {
        key: np.asarray(stamps)[indices] for key, stamps in values["image_timestamps"].items()
    }
    compact["allow_repeated_source_timestamps"] = False
    compact["metadata"] = dict(values.get("metadata", {}),
                                recap_compacted=True,
                                recap_source_frame_count=int(len(annotation.owner)),
                                recap_export_frame_count=int(len(indices)),
                                recap_removed_invalid_frames=int(len(annotation.owner) - len(indices)))
    return compact_annotation, compact


def load_run(path, *, fps=None, max_skew_s=.05, compact=False):
    path = Path(path)
    control_hz = recorded_control_hz(path)
    fps = control_hz if fps is None else float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    # Resampling above the control rate repeats source rows, and repeated
    # rows are masked invalid: a 20 Hz run exported at 30 fps had no chunk
    # long enough to train on. Never silently upsample.
    if fps > control_hz * (1 + 1e-6):
        raise ValueError(f"{path}: export fps {fps:g} exceeds the recorded control rate {control_hz:g} Hz")
    metadata = json.loads((path / "metadata.json").read_text())
    outcome = metadata.get("recap", {}).get("outcome")
    if outcome not in ("success", "failure"):
        raise ValueError(f"{path}: RECAP run needs an explicit success/failure label")
    protocol = metadata["policy_protocol"]
    if protocol["schema"] != "delivery" or protocol["contract_version"] != 3:
        raise ValueError("RECAP export requires canonical v3 delivery recordings")
    with np.load(path / "trajectory.npz", allow_pickle=False) as source:
        raw = {k: source[k] for k in source.files}
    times = raw["timestamp"]
    if len(times) < 2:
        raise ValueError("RECAP recording has fewer than two control frames")
    grid = np.arange(times[0], times[-1], 1 / fps)
    if len(grid) < 2:
        raise ValueError("RECAP recording has fewer than two resampled frames")
    indices = nearest_indices(times, grid)
    states = raw["delivery_state"][indices]
    grippers = raw["command_gripper_fraction"][indices].copy()
    valid = raw["recap_valid"][indices].copy()
    valid &= np.abs(times[indices] - grid) <= .75 / fps
    valid[:-1] &= np.diff(times[indices]) > 0
    owner, epoch = raw["recap_owner"][indices], raw["recap_epoch"][indices]
    # Servo commands contain the actual rate-limited gripper command; prefer
    # them over the policy's unexecuted endpoint on policy-owned rows.
    servo_path = path / "servo_commands.jsonl"
    if servo_path.exists():
        servo = [json.loads(line) for line in servo_path.read_text().splitlines() if line.strip()]
        if servo:
            st = np.array([v["timestamp"] for v in servo])
            si = np.searchsorted(st, grid, side="right") - 1
            good = (si >= 0) & (grid - st[si.clip(0)] <= .1) & (owner == "policy")
            commands = np.array([v["command_qpos"][6::7] for v in servo]) / .07
            grippers[good] = commands[si[good]]
    finite = np.isfinite(grippers).all(axis=1)
    valid &= finite
    # Placeholder only on invalid rows; these cannot enter policy loss.
    grippers[~finite] = states[~finite, 9::10]
    valid[:-1] &= epoch[:-1] == epoch[1:]
    valid[-1] = False
    actions = build_delivery_actions_with_gripper_targets(states, grippers, states.shape[1] // 10)
    camera_rows = [json.loads(line) for line in (path / "videos/timestamps.jsonl").read_text().splitlines()]
    images, image_times = {}, {}
    for key in protocol["camera_keys"]:
        rows = [row for row in camera_rows if row["camera"] == key]
        ct = np.array([row["timestamp"] for row in rows])
        ci = nearest_indices(ct, grid)
        valid &= np.abs(ct[ci] - grid) <= max_skew_s
        frames = {}
        captures = {}
        try:
            for index in np.unique(ci):
                row = rows[index]
                storage = (path / row["storage"]).resolve()
                if not storage.is_relative_to(path.resolve()):
                    raise ValueError("video path escapes the recording")
                if storage.suffix == ".mp4":
                    if str(storage) not in captures:
                        captures[str(storage)] = cv2.VideoCapture(str(storage))
                    cap = captures[str(storage)]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame_index"])
                    ok, frame = cap.read()
                else:
                    frame = cv2.imread(str(storage))
                    ok = frame is not None
                if not ok:
                    raise ValueError(f"cannot decode recorded frame: {storage}:{row['frame_index']}")
                frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images[key] = np.stack([frames[i] for i in ci])
        finally:
            for cap in captures.values():
                cap.release()
        image_times[key] = ct[ci]
    annotation = RecapEpisode(owner, epoch, valid, outcome)
    values = dict(states=states, actions=actions, timestamps=grid,
        state_timestamps=times[indices],
        action_timestamps=np.r_[times[indices][1:], times[indices][-1] + 1 / fps],
        allow_repeated_source_timestamps=True,
        images=images, image_timestamps=image_times, task_name=metadata["instruction"],
        instruction=metadata["instruction"], success=outcome == "success",
        metadata={"recap_source_run": str(path.resolve()), "recap_outcome": outcome,
                  "observation_resampling": "nearest; repeated source timestamps retained; invalid gaps masked"})
    if compact:
        annotation, values = _compact_values(annotation, values)
        values["metadata"]["observation_resampling"] = (
            "nearest; invalid handoff/hold rows removed; source segment boundaries retained"
        )
    return metadata, annotation, values


def export_runs(runs, output, *, fps=None, arm_base_offset=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("RECAP export requires a new dataset directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recap-export-", dir=output.parent) as temporary:
        staged = _export_runs(runs, Path(temporary) / output.name, fps=fps, arm_base_offset=arm_base_offset)
        if output.exists():
            raise FileExistsError("RECAP output appeared during conversion")
        staged.rename(output)
    return output


def _export_runs(runs, output, *, fps=None, arm_base_offset=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("RECAP export requires a new dataset directory")
    if fps is None:
        rates = {recorded_control_hz(run) for run in runs}
        if len(rates) != 1:
            raise ValueError(f"RECAP runs have different control rates {sorted(rates)}; pass --fps")
        fps = rates.pop()
    if float(fps) != int(fps):
        raise ValueError(f"dataset fps must be an integer, got {fps:g}")
    fps = int(fps)
    writer = None
    contract = None
    for run in runs:
        metadata, annotation, values = load_run(run, fps=fps, compact=True)
        protocol = metadata["policy_protocol"]
        current_contract = {key: protocol[key] for key in (
            "schema", "contract_version", "arm_mode", "arm_side", "image_resize_mode", "camera_keys")}
        if contract is not None and current_contract != contract:
            raise ValueError("RECAP runs must share arm, camera, and image contracts")
        contract = current_contract
        if writer is None:
            names, actions = default_eef_names(arm_mode=protocol["arm_mode"], arm_side=protocol["arm_side"])
            writer = Pi0LeRobotDatasetWriter(output, fps=fps, robot_type="piper_" + protocol["arm_mode"],
                state_names=names, action_names=actions, camera_keys=protocol["camera_keys"],
                schema="delivery", arm_mode=protocol["arm_mode"], arm_side=protocol["arm_side"],
                arm_base_offset=arm_base_offset,
                image_resize_mode=protocol["image_resize_mode"], action_offset=1,
                action_source="next_measured_eef_fallback_with_same_step_commanded_gripper",
                action_alignment="next_observation_pose_same_step_gripper")
            (output / "meta/recap").mkdir()
        if list(protocol["camera_keys"]) != list(writer.camera_keys) or values["states"].shape[1] != len(writer.state_names):
            raise ValueError("RECAP runs must share arm/camera contracts")
        episode_index = writer.append_episode(**values)
        np.savez_compressed(output / f"meta/recap/episode_{episode_index:06d}.npz", **annotation.to_payload())
    if writer is None:
        raise ValueError("no RECAP recordings supplied")
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["recap"] = {"version": 1, "annotations": "meta/recap/episode_{episode_index:06d}.npz",
                     "outcomes": ["success", "failure"], "policy_chunks_cross_handoffs": False,
                     "invalid_rows_compacted": True,
                     "source_timeline_retained_in_deployment_runs": True}
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=None,
                        help="dataset rate; defaults to the recorded control rate (never above it)")
    parser.add_argument("--arm-base-offset", type=float, nargs=3, default=None,
                        metavar=("X_M", "Y_M", "Z_M"), help="measured right-base minus left-base offset")
    args = parser.parse_args()
    if args.fps is not None and args.fps < 1:
        parser.error("fps must be positive")
    print(export_runs(args.runs, args.output, fps=args.fps, arm_base_offset=args.arm_base_offset))


if __name__ == "__main__":
    main()

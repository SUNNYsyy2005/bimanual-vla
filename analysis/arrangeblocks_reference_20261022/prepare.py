"""Extract a small, evenly sampled reference bundle without altering the episode."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=False) as episode:
        state = episode["state"]
        actions = episode["actions"]
        frames = np.unique(np.linspace(0, len(state) - 51, 24).astype(int))
        bundle = dict(frames=frames, state=state[frames],
                      truth_offset0=np.stack([actions[f:f+50] for f in frames]),
                      truth_offset1=np.stack([actions[f+1:f+51] for f in frames]),
                      instruction=episode["instruction"],
                      fps=episode["fps"], episode_frames=np.asarray(len(state)))
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            images = episode[f"images_{camera}"]
            bundle[f"images_{camera}"] = images[frames].copy()
            del images
            print("EXTRACTED", camera, flush=True)
        metadata = {
            "source": str(args.source.resolve()), "source_bytes": args.source.stat().st_size,
            "frames": frames.tolist(), "episode_frames": len(state),
            "fps": float(episode["fps"]), "instruction": str(episode["instruction"]),
            "actions_equal_states": bool(np.array_equal(actions, state)),
            "collection_metadata": json.loads(str(episode["metadata_json"])),
            "heldout_membership": "unknown",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **bundle)
    metadata["bundle_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

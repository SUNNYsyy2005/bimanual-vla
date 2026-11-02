"""Check sidecars, real optimizer updates, resume steps and saved policy inference."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rlinf.models.embodiment.openpi_cfg import get_model
from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import Observation

root = Path(os.environ["TASK_ROOT"])
if os.environ.get("RECAP_GPU_MEMORY_FRACTION"):
    torch.cuda.set_per_process_memory_fraction(float(os.environ["RECAP_GPU_MEMORY_FRACTION"]))
result = {"synthetic": True, "sidecars": [], "metrics": {}}
for split in ["train", "heldout"]:
    data = root / f"datasets/SYNTHETIC_recap_{split}"
    returns = pq.read_table(data / "meta/returns_synthetic_20261101.parquet").to_pydict()
    advantages = pq.read_table(data / "meta/advantages_synthetic_20261101.parquet").to_pydict()
    return_keys = list(zip(returns["episode_index"], returns["frame_index"]))
    advantage_keys = list(zip(advantages["episode_index"], advantages["frame_index"]))
    assert len(set(return_keys)) == len(return_keys)
    assert len(set(advantage_keys)) == len(advantage_keys)
    assert set(return_keys) == set(advantage_keys)
    assert np.isfinite(returns["return"]).all()
    for episode in sorted(set(returns["episode_index"])):
        indices = [i for i, ep in enumerate(returns["episode_index"]) if ep == episode]
        terminal = max(indices, key=lambda i: returns["frame_index"][i])
        assert returns["reward"][terminal] == (0 if episode % 2 == 0 else -20), "Synthetic outcome mislabeled"
    label_key = "advantage" if "advantage" in advantages else "is_positive"
    positives = int(np.asarray(advantages[label_key], dtype=bool).sum())
    assert 0 < positives < len(advantage_keys), "Need both CFG label classes"
    result["sidecars"].append({"split": split, "frames": len(return_keys), "positive": positives,
                                "negative": len(return_keys) - positives})

for experiment, expected in [("synthetic_value", 12), ("synthetic_cfg", 6)]:
    run = root / "runs" / ("value" if experiment == "synthetic_value" else "cfg")
    assert (run / experiment / f"checkpoints/global_step_{expected}/actor/model_state_dict/full_weights.pt").is_file()
    metric_values = {}
    for event_file in run.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for key in accumulator.Tags()["scalars"]:
            metric_values.setdefault(key, []).extend((e.step, e.value) for e in accumulator.Scalars(key))
    assert metric_values, f"No scalar metrics for {experiment}"
    diagnostic_std_nan = {}
    for key, values in metric_values.items():
        nan_count = sum(math.isnan(v) for _, v in values)
        if nan_count and key.endswith("_std"):
            # The upstream logger uses unbiased std() on a one-item batch.
            # Its diagnostic is NaN; losses and gradients must remain finite.
            diagnostic_std_nan[key] = nan_count
        assert all(math.isfinite(v) or (key.endswith("_std") and math.isnan(v))
                   for _, v in values), f"Non-finite {key}"
    loss_keys = [k for k in metric_values if k.startswith("train/") and "loss" in k]
    assert loss_keys
    max_step = max(s for key in loss_keys for s, _ in metric_values[key])
    assert max_step >= expected - 1, "Resume did not advance training"
    gradients = [v for k, values in metric_values.items() if "grad_norm" in k for _, v in values]
    assert gradients and max(gradients) > 0, "No nonzero training gradients"
    result["metrics"][experiment] = {"last_step": max_step + 1,
        "losses": {k: sorted(metric_values[k]) for k in loss_keys},
        "max_grad_norm": max(gradients), "diagnostic_std_nan": diagnostic_std_nan}

with initialize_config_dir(version_base="1.1", config_dir=str(root / "configs")):
    cfg = compose(config_name="cfg")
model = get_model(cfg.actor.model).cuda().eval()
checkpoint = root / "runs/cfg/synthetic_cfg/checkpoints/global_step_6/actor/model_state_dict/full_weights.pt"
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict(state, strict=True)
base_checkpoint = root / "runs/cfg/synthetic_cfg/checkpoints/global_step_4/actor/model_state_dict/full_weights.pt"
earlier = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
assert any(not torch.equal(state[k], earlier[k]) for k in state if state[k].is_floating_point()), "No parameter update after resume"
del earlier, state
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
data = LeRobotDataset("synthetic/recap_heldout", root=root / "datasets/SYNTHETIC_recap_heldout",
                      download_videos=False)
sample = data[0]
prompt = "SYNTHETIC TEST ONLY: move the colored block"
def image(key):
    value = sample[key].numpy()
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if value.dtype != np.uint8:
        value = (value * 255).clip(0, 255).astype(np.uint8)
    return value[None]
inputs = {"observation/image": image("image"), "observation/wrist_image": image("wrist_image"),
          "observation/state": sample["state"].numpy()[None], "prompt": [prompt],
          "positive_guidance_prompt": [prompt + "\nAdvantage: positive"],
          "negative_guidance_prompt": [prompt + "\nAdvantage: negative"]}
observation = Observation.from_dict(model.precision_processor(model.input_transform(inputs, transpose=False)))
torch.manual_seed(20261101)
with torch.inference_mode():
    actions = model.sample_actions(observation)["actions"]
assert actions.shape == (1, 10, 32), actions.shape
assert torch.isfinite(actions).all()
result["policy_inference"] = {"checkpoint_loaded_strictly": True, "output_shape": list(actions.shape),
                               "all_finite": True, "resume_parameters_changed": True}
(root / "records/verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

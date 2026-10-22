"""Verify numeric norm samples are identical with and without video decoding."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from server_4090 import openpi_single_arm as app


def main():
    root = Path(__file__).parent
    response = json.loads((root / "train_retry_response.json").read_text())
    task = json.loads((Path(response["log_path"]).parent.parent /
                       response["dependency"]["task_id"] / "task.json").read_text())
    sys.argv = task["command"][1:]
    args = app.parse_args()
    config = app.build_config(args)
    fingerprint = app.complete_action_contract_fingerprint(config.policy_metadata)
    split = app._resolve_training_split(args, fingerprint)
    concrete = config.data.create(config.assets_dirs, config.model)
    datasets = [app._create_torch_dataset_for_episodes(concrete, config.model.action_horizon,
        config.model, split.train_episodes, action_offset=1, load_images=load)
        for load in (True, False)]
    transformed = [app.data_loader.TransformedDataset(dataset,
        [*concrete.repack_transforms.inputs, *concrete.data_transforms.inputs, app.RemoveStrings()])
        for dataset in datasets]
    checks, times = [], [0., 0.]
    for index in sorted(set([0, 1, 40, 200, len(datasets[0])//2, len(datasets[0])-1])):
        samples = []
        for i, dataset in enumerate(transformed):
            start = time.monotonic()
            samples.append(dataset[index])
            times[i] += time.monotonic()-start
        for key in ("state", "actions"):
            np.testing.assert_array_equal(samples[0][key], samples[1][key])
        checks.append(dict(index=index, state_shape=list(samples[0]["state"].shape),
                           actions_shape=list(samples[0]["actions"].shape), max_error=0))
    result = dict(checks=checks, numeric_max_error=0, decoded_seconds=times[0],
                  numeric_only_seconds=times[1], same_episode_selection=True)
    (root / "norm_parity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

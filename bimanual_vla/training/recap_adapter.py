"""Piper data adapters for RLinf value learning and OpenPI policy learning.

Value architecture, returns, TD advantages and quantile labels remain upstream
RLinf. Only field mapping and policy sample admission live here.
"""

from pathlib import Path
import json

import numpy as np

from bimanual_vla.data.recap import RecapEpisode


CAMERAS = {"cam_high": "base_0_rgb", "cam_left_wrist": "left_wrist_0_rgb",
           "cam_right_wrist": "right_wrist_0_rgb"}


class PiperValueInputs:
    """Same three RGB views during value training and value inference."""

    def __call__(self, sample):
        images = {}
        for source, destination in CAMERAS.items():
            image = np.asarray(sample["observation.images." + source])
            if image.ndim != 3:
                raise ValueError("Piper value image must be HWC or CHW")
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            if image.dtype != np.uint8:
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError("floating RGB must be in [0,1]")
                image = np.rint(image * 255).astype(np.uint8)
            images[destination] = image
        return {"image": images, "image_mask": {k: np.True_ for k in images},
                "prompt": str(sample["prompt"]), "state": np.asarray(sample["observation.state"])}


def policy_admission(root, episodes, horizon, action_offset):
    if type(horizon) is not int or horizon < 1 or action_offset not in (0, 1):
        raise ValueError("invalid RECAP action horizon/offset")
    admitted = {}
    for episode in episodes:
        with np.load(Path(root) / f"meta/recap/episode_{episode:06d}.npz", allow_pickle=False) as data:
            annotation = RecapEpisode.from_payload(data)
        if not annotation.value_eligible:
            raise ValueError("RECAP training requires success/failure-labelled episodes")
        admitted[episode] = set(annotation.chunk_starts(horizon + 1 - action_offset).tolist())
    return admitted


def advantage_prompt(prompt):
    prompt = str(prompt).strip()
    suffix = "\nAdvantage: positive"
    return prompt if prompt.endswith(suffix) else prompt + suffix


class ConditionedDataset:
    """RLinf positive-only conditioning: negatives stay unconditional.

    This is the JAX-loader adapter, not a replacement value/advantage algorithm.
    Human corrections use the explicit intervention flag as positive examples.
    All admitted autonomous examples remain in the training distribution.
    """

    def __init__(self, dataset, artifact_dir, unconditional_probability):
        if not np.isfinite(unconditional_probability) or not 0 <= unconditional_probability < 1:
            raise ValueError("invalid RECAP condition dropout probability")
        self.dataset = dataset
        self.probability = unconditional_probability
        self.labels = {}
        manifest = json.loads((Path(artifact_dir) / "manifest.json").read_text())
        for episode in manifest["train_episodes"]:
            with np.load(Path(artifact_dir) / f"episode_{episode:06d}.npz", allow_pickle=False) as data:
                self.labels[episode] = data["positive"].copy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        episode = int(np.asarray(sample["episode_index"]).item())
        frame = int(np.asarray(sample["frame_index"]).item())
        if self.labels[episode][frame] and np.random.random() >= self.probability:
            sample["prompt"] = advantage_prompt(sample["prompt"])
        return sample


class PositivePolicy:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, observation):
        observation = dict(observation)
        if "prompt" in observation:
            observation["prompt"] = advantage_prompt(observation["prompt"])
        return self.policy.infer(observation)

    def reset(self):
        if hasattr(self.policy, "reset"):
            return self.policy.reset()

    @property
    def metadata(self):
        return self.policy.metadata

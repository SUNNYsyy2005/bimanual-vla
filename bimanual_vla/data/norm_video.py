"""LeRobot state/action normalization without unused video decoding.

Only run_norm selects this dataset. Training and model evaluation always use
real images. Tiny placeholders satisfy camera repacking but never enter
the state/action statistics or any model.
"""
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class NormOnlyLeRobotDataset(LeRobotDataset):
    def _query_videos(self, query_timestamps, ep_idx):
        return {key: torch.zeros((len(times), 3, 8, 8), dtype=torch.float32).squeeze(0)
                for key, times in query_timestamps.items()}

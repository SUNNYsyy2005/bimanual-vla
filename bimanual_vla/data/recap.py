"""RECAP episode annotations, independent of the robot and training backend.

An ownership transition is not a demonstration. Keep the full timeline for
value learning, but never form a policy action chunk across a handoff or a gap
in valid control. Aborted/unlabelled episodes are not failed demonstrations.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


RECAP_VERSION = 1
OWNERS = ("policy", "human", "transition", "hold")
OUTCOMES = ("success", "failure", "aborted", "unlabelled")


@dataclass(frozen=True)
class RecapEpisode:
    owner: np.ndarray
    epoch: np.ndarray
    valid: np.ndarray
    outcome: str = "unlabelled"

    def __post_init__(self):
        owner = np.asarray(self.owner)
        epoch = np.asarray(self.epoch)
        valid = np.asarray(self.valid)
        if owner.ndim != 1 or not len(owner):
            raise ValueError("RECAP owner must be a nonempty per-frame array")
        if epoch.shape != owner.shape or valid.shape != owner.shape:
            raise ValueError("RECAP annotations must have identical per-frame lengths")
        if not np.isin(owner, OWNERS).all():
            raise ValueError("unknown RECAP command owner")
        if epoch.dtype.kind not in "iu" or np.any(epoch < 0) or np.any(np.diff(epoch.astype(np.int64)) < 0):
            raise ValueError("RECAP epochs must be nonnegative monotonic integers")
        if valid.dtype.kind != "b":
            raise ValueError("RECAP valid must be boolean")
        if np.any((owner[1:] != owner[:-1]) & (epoch[1:] == epoch[:-1])):
            raise ValueError("a RECAP owner change must increment the epoch")
        if self.outcome not in OUTCOMES:
            raise ValueError("unknown RECAP episode outcome")
        for key, value in (("owner", owner), ("epoch", epoch), ("valid", valid)):
            value = value.copy()
            value.flags.writeable = False
            object.__setattr__(self, key, value)

    @property
    def intervention(self):
        return (self.owner == "human") & self.valid

    @property
    def value_eligible(self):
        return self.outcome in ("success", "failure")

    def policy_spans(self, *, human_only=False):
        """Half-open contiguous spans; no deletion/concatenation across gaps."""
        eligible = self.valid & np.isin(self.owner, ("human",) if human_only else ("human", "policy"))
        start = None
        for index in range(len(eligible) + 1):
            boundary = (index == len(eligible) or not eligible[index]
                        or (index > 0 and self.epoch[index] != self.epoch[index - 1]))
            if start is not None and boundary:
                yield start, index
                start = None
            if index < len(eligible) and eligible[index] and start is None:
                start = index

    def chunk_starts(self, horizon, *, human_only=False):
        """Only complete chunks; callers must not pad across ownership changes."""
        if type(horizon) is not int or horizon < 1:
            raise ValueError("action horizon must be a positive integer")
        return np.asarray([i for start, end in self.policy_spans(human_only=human_only)
                           for i in range(start, end - horizon + 1)], dtype=np.int64)

    def to_payload(self):
        return {"recap_version": np.asarray(RECAP_VERSION),
                "recap_owner": self.owner, "recap_epoch": self.epoch,
                "recap_valid": self.valid, "recap_outcome": np.asarray(self.outcome),
                "intervention_mask": self.intervention}

    @classmethod
    def from_payload(cls, data: Mapping):
        if int(np.asarray(data["recap_version"]).item()) != RECAP_VERSION:
            raise ValueError("unsupported RECAP annotation version")
        result = cls(data["recap_owner"], data["recap_epoch"], data["recap_valid"],
                     str(np.asarray(data["recap_outcome"]).item()))
        if "intervention_mask" in data and not np.array_equal(data["intervention_mask"], result.intervention):
            raise ValueError("intervention mask conflicts with RECAP ownership/validity")
        return result

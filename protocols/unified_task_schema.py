"""Schema skeleton for unified manipulation task and evaluation protocols.

The module intentionally contains structure only. Concrete task lists, dataset
paths, and benchmark-specific field constraints should be filled in after
Claude provides verified protocol details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Sequence


class ObservationModality(str, Enum):
    RGB = "rgb"
    DEPTH = "depth"
    POINT_CLOUD = "point_cloud"
    PROPRIOCEPTION = "proprioception"
    LANGUAGE = "language"
    STATE = "state"


class ActionSpaceType(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    END_EFFECTOR_DELTA_POSE = "end_effector_delta_pose"
    END_EFFECTOR_ABSOLUTE_POSE = "end_effector_absolute_pose"


class MetricKind(str, Enum):
    SUCCESS_RATE = "success_rate"
    EPISODE_RETURN = "episode_return"
    COMPLETION_TIME = "completion_time"
    COLLISION_RATE = "collision_rate"
    CUSTOM = "custom"


@dataclass(frozen=True)
class TensorSpec:
    """Shape and dtype contract for an observation or action tensor."""

    name: str
    shape: tuple[int | Literal["variable"], ...]
    dtype: str
    description: str = ""


@dataclass(frozen=True)
class ObservationSpec:
    """One named observation stream used by a task."""

    name: str
    modality: ObservationModality
    tensor: TensorSpec | None = None
    source: str = ""
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSpaceSpec:
    """Action representation expected by policy and environment adapters."""

    action_type: ActionSpaceType
    dimensions: int
    tensor: TensorSpec
    control_hz: float | None = None
    gripper_dimensions: int = 0
    bounds_low: tuple[float, ...] | None = None
    bounds_high: tuple[float, ...] | None = None
    frame: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationMetricSpec:
    """Metric field emitted by evaluators and aggregated by reports."""

    name: str
    kind: MetricKind
    higher_is_better: bool
    unit: str = ""
    description: str = ""
    required: bool = True


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset location and split contract, without cluster-specific paths."""

    name: str
    version: str = ""
    splits: tuple[str, ...] = ()
    path_hint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """Unified task description shared by data conversion and evaluation code."""

    task_id: str
    suite: str
    description: str
    observations: tuple[ObservationSpec, ...]
    action_space: ActionSpaceSpec
    metrics: tuple[EvaluationMetricSpec, ...]
    dataset: DatasetSpec | None = None
    horizon_steps: int | None = None
    language_instruction: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationProtocolSpec:
    """Run-level evaluation contract for one benchmark or benchmark slice."""

    protocol_id: str
    tasks: tuple[TaskSpec, ...]
    num_episodes_per_task: int
    aggregation: Literal["mean", "weighted_mean", "median"] = "mean"
    random_seed: int | None = None
    output_artifacts: tuple[str, ...] = ("metrics.json", "report.md")
    metadata: Mapping[str, Any] = field(default_factory=dict)


def validate_protocol(protocol: EvaluationProtocolSpec) -> None:
    """Perform lightweight structural validation for draft protocol specs."""

    if not protocol.protocol_id:
        raise ValueError("protocol_id must be non-empty")
    if not protocol.tasks:
        raise ValueError("at least one task is required")
    if protocol.num_episodes_per_task <= 0:
        raise ValueError("num_episodes_per_task must be positive")

    seen_task_ids: set[str] = set()
    for task in protocol.tasks:
        if not task.task_id:
            raise ValueError("task_id must be non-empty")
        if task.task_id in seen_task_ids:
            raise ValueError(f"duplicate task_id: {task.task_id}")
        seen_task_ids.add(task.task_id)
        if not task.observations:
            raise ValueError(f"{task.task_id}: at least one observation is required")
        if task.action_space.dimensions <= 0:
            raise ValueError(f"{task.task_id}: action dimensions must be positive")
        if task.action_space.tensor.shape and task.action_space.tensor.shape[-1] != task.action_space.dimensions:
            raise ValueError(
                f"{task.task_id}: final action tensor dimension must match action_space.dimensions"
            )
        if not task.metrics:
            raise ValueError(f"{task.task_id}: at least one evaluation metric is required")


def metric_names(metrics: Sequence[EvaluationMetricSpec]) -> tuple[str, ...]:
    return tuple(metric.name for metric in metrics)

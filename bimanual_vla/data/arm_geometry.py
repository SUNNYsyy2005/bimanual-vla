"""Shared metadata helpers for bimanual arm base geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


ARM_BASE_OFFSET_KEY = "arm_base_offset"
ARM_BASE_ROTATIONS_KEY = "arm_base_rotations"
ARM_BASE_AXIS_CONVENTION = "per_arm_base_frame_explicit_axis_signs"
ARM_AXIS_SIGNS = {
    "left": (1, 1, 1),
    "right": (1, 1, 1),
}

ROBOTWIN_EMBODIMENTS = frozenset(
    {"aloha-agilex", "piper", "arx-x5", "franka-panda"}
)


REAL_PIPER_RIGHT_AXIS_SIGNS = (1, -1, 1)


def normalize_robot_type(value: Any) -> str | None:
    """Normalize common RoboTwin and LeRobot robot type spellings."""

    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if not text:
        return None
    if "franka" in text or "panda" in text:
        return "franka-panda"
    if "aloha" in text or "agilex" in text:
        return "aloha-agilex"
    if text.startswith("arx") or "x5" in text:
        return "arx-x5"
    if "piper" in text:
        return "piper"
    return text


def normalize_arm_base_offset(value: Any, *, required: bool = False) -> tuple[float, float, float] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValueError("arm_base_offset is required for bimanual datasets")
        return None
    candidate = value
    if isinstance(value, Mapping):
        for key in (
            "right_relative_to_left_m",
            "right_relative_to_left",
            "vector_m",
            "vector",
        ):
            if key in value:
                candidate = value[key]
                break
        else:
            if "left" in value and "right" in value:
                candidate = np.asarray(value["right"], dtype=np.float64) - np.asarray(value["left"], dtype=np.float64)
            else:
                candidate = None
    if isinstance(candidate, (str, bytes)):
        raise ValueError("arm_base_offset must contain exactly three numeric metres")
    try:
        vector = np.asarray(candidate, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("arm_base_offset must contain exactly three numeric metres") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("arm_base_offset must contain exactly three finite numeric metres")
    return tuple(float(item) for item in vector)


def arm_base_offset_metadata(value: Any) -> dict[str, Any] | None:
    offset = normalize_arm_base_offset(value)
    if offset is None:
        return None
    return {
        "right_relative_to_left_m": list(offset),
        "convention": "right_base_minus_left_base",
        "frame": "left_arm_base",
        "units": "m",
    }


def normalize_arm_base_rotations(value: Any) -> dict[str, np.ndarray] | None:
    """Normalize optional local-arm-to-common-frame rotation matrices."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    candidate = value
    if isinstance(value, Mapping):
        candidate = value.get("rotations", value)
    if isinstance(candidate, Mapping):
        items = candidate.items()
    else:
        array = np.asarray(candidate, dtype=np.float64)
        if array.shape == (3, 3):
            items = (("right", array),)
        elif array.shape == (2, 3, 3):
            items = (("left", array[0]), ("right", array[1]))
        else:
            raise ValueError("arm_base_rotations must contain 3x3 rotation matrices")
    rotations: dict[str, np.ndarray] = {}
    for side, raw_rotation in items:
        side = str(side).strip().lower()
        if side not in {"left", "right"}:
            continue
        rotation = np.asarray(raw_rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("arm_base_rotations must contain finite 3x3 matrices")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0.0:
            raise ValueError("arm_base_rotations must contain proper rigid rotations")
        rotations[side] = rotation
    return rotations or None


def arm_base_rotation_metadata(value: Any) -> dict[str, list[list[float]]] | None:
    rotations = normalize_arm_base_rotations(value)
    if rotations is None:
        return None
    return {side: rotation.tolist() for side, rotation in rotations.items()}


def arm_base_origins(value: Any) -> dict[str, list[float]]:
    offset = normalize_arm_base_offset(value)
    if offset is None:
        return {}
    return {"left": [0.0, 0.0, 0.0], "right": list(offset)}


def arm_base_axis_signs(
    value: Any,
    *,
    bimanual: bool = False,
    robot_type: Any = None,
    dataset_origin: Any = None,
) -> dict[str, list[int]]:
    if normalize_arm_base_offset(value) is None and not bimanual:
        return {}
    signs = {side: list(axis_signs) for side, axis_signs in ARM_AXIS_SIGNS.items()}
    is_real = str(dataset_origin or "").strip().lower() in {"real", "hardware", "physical"}
    if is_real and normalize_robot_type(robot_type) == "piper" and bimanual:
        signs["right"] = list(REAL_PIPER_RIGHT_AXIS_SIGNS)
    return signs


__all__ = [
    "ARM_BASE_OFFSET_KEY",
    "ARM_BASE_ROTATIONS_KEY",
    "ARM_BASE_AXIS_CONVENTION",
    "ARM_AXIS_SIGNS",
    "REAL_PIPER_RIGHT_AXIS_SIGNS",
    "ROBOTWIN_EMBODIMENTS",
    "arm_base_axis_signs",
    "arm_base_offset_metadata",
    "arm_base_rotation_metadata",
    "arm_base_origins",
    "normalize_arm_base_offset",
    "normalize_arm_base_rotations",
    "normalize_robot_type",
]

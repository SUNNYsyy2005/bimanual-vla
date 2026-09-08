"""Shared metadata helpers for bimanual arm base geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


ARM_BASE_OFFSET_KEY = "arm_base_offset"
ARM_BASE_AXIS_CONVENTION = "left_base_common_frame_right_x_mirrored"
ARM_AXIS_SIGNS = {
    "left": (1, 1, 1),
    "right": (-1, 1, 1),
}


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


def arm_base_origins(value: Any) -> dict[str, list[float]]:
    offset = normalize_arm_base_offset(value)
    if offset is None:
        return {}
    return {"left": [0.0, 0.0, 0.0], "right": list(offset)}


def arm_base_axis_signs(value: Any, *, bimanual: bool = False) -> dict[str, list[int]]:
    if normalize_arm_base_offset(value) is None and not bimanual:
        return {}
    return {side: list(signs) for side, signs in ARM_AXIS_SIGNS.items()}


__all__ = [
    "ARM_BASE_OFFSET_KEY",
    "ARM_BASE_AXIS_CONVENTION",
    "ARM_AXIS_SIGNS",
    "arm_base_axis_signs",
    "arm_base_offset_metadata",
    "arm_base_origins",
    "normalize_arm_base_offset",
]

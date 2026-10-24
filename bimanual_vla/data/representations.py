"""Select a training representation from one dual-recorded raw episode."""
from collections import ChainMap
import numpy as np
from bimanual_vla.data.contract import (
    EpisodeContract, JOINT_SCHEMA, DELIVERY_SCHEMA, JOINT_MEASURED_ACTION_SOURCE,
    infer_episode_contract, derive_joint_actions, next_observation_timestamps,
)


def select_representation(data, target_schema=None):
    if target_schema not in (None, "joint", "delivery"):
        raise ValueError("target_schema must be joint or delivery")
    dual = "recorded_state_schemas" in data
    if not dual:
        if target_schema is not None and infer_episode_contract(data).schema != target_schema:
            raise ValueError("requested representation was not recorded; collect Joint + End-effector first")
        return data
    if set(np.asarray(data["recorded_state_schemas"]).tolist()) != {"joint", "delivery"}:
        raise ValueError("invalid recorded_state_schemas")
    source = infer_episode_contract(data)
    if source.version != 3 or source.schema != DELIVERY_SCHEMA:
        raise ValueError("dual recording requires a canonical v3 delivery primary contract")
    for prefix, dim in (("joint", source.joint_dim), ("eef", source.arm_count * 10)):
        for field in ("state", "actions", "state_timestamp", "action_timestamp"):
            key = f"{prefix}_{field}"
            if key not in data:
                raise ValueError(f"dual recording missing {key}")
            values = np.asarray(data[key])
            expected = (len(data["state"]),) if "timestamp" in field else (len(data["state"]), dim)
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"invalid dual field {key}: expected finite {expected}")
            if "timestamp" in field and np.any(np.diff(values) <= 0):
                raise ValueError(f"{key} must be strictly increasing")
    for key, alias in (("state", "eef_state"), ("actions", "eef_actions"),
                       ("state_timestamp", "eef_state_timestamp"), ("action_timestamp", "eef_action_timestamp"),
                       ("joint_qpos", "joint_state")):
        if not np.array_equal(data[key], data[alias]):
            raise ValueError(f"dual recording {key} disagrees with {alias}")
    if not np.allclose(data["joint_actions"], derive_joint_actions(data["joint_state"]), atol=1e-6):
        raise ValueError("joint_actions must be next measured joint targets")
    expected_ts = next_observation_timestamps(data["joint_state_timestamp"], fps=source.fps)
    if not np.allclose(data["joint_action_timestamp"], expected_ts, rtol=0, atol=1e-6):
        raise ValueError("joint action timestamps disagree with next observation")
    if target_schema in (None, DELIVERY_SCHEMA):
        return data
    contract = EpisodeContract(schema=JOINT_SCHEMA, arm_mode=source.arm_mode,
        arm_side=source.arm_side, camera_keys=source.camera_keys, fps=source.fps,
        action_horizon=source.action_horizon, coordinate_frame=source.coordinate_frame,
        action_source=JOINT_MEASURED_ACTION_SOURCE, action_alignment="next_observation")
    return ChainMap({**contract.metadata_payload(),
        "state": data["joint_state"], "actions": data["joint_actions"],
        "state_timestamp": data["joint_state_timestamp"],
        "timestamps": data["joint_state_timestamp"],
        "action_timestamp": data["joint_action_timestamp"]}, data)

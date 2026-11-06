"""Bridge SDK absolute XYZ/RPY/metre EEF policies to delivery v3."""
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.data.action_conventions import (
    DELIVERY_MODEL_ACTION_SEMANTICS,
    absolute_eef_targets_to_chunk_origin,
    matrix_to_rotation6d,
    rotation6d_to_matrix,
)
from bimanual_vla.data.contract import GRIPPER_MAX_M, GRIPPER_OPENING_SEMANTICS

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
OUTPUT_ORDER = [f"{side}_{name}" for side in ("left", "right")
                for name in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")]
SDK_CONTRACT = dict(
    state_action_same_layout=True, coordinate_frame="each_arm_sdk_base",
    reference_point="sdk_native_tcp", position_semantics="absolute_m",
    orientation_semantics="piper_sdk_rpy_xyz_rad", gripper_semantics="absolute_opening_m",
    gripper_closed_m=0, gripper_open_m=0.07,
)


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite {shape}, got {value.shape}")
    return value


def _opening(value, maximum, name):
    if np.any(value < -1e-6) or np.any(value > maximum + 1e-6):
        raise ValueError(f"{name} outside [0, {maximum}]")
    return np.clip(value, 0, maximum)


@dataclass
class EEFBridge:
    # Each transform is indexed by physical arm: left, right.
    model_base_from_client_base: np.ndarray = field(default_factory=lambda: np.tile(np.eye(4), (2, 1, 1)))
    client_tool_from_model_tool: np.ndarray = field(default_factory=lambda: np.tile(np.eye(4), (2, 1, 1)))
    model_arm_order: tuple = ("left", "right")

    def __post_init__(self):
        if sorted(self.model_arm_order) != ["left", "right"]:
            raise ValueError("model_arm_order must contain left and right exactly once")
        for name in ("model_base_from_client_base", "client_tool_from_model_tool"):
            transforms = _finite(getattr(self, name), (2, 4, 4), name).copy()
            for transform in transforms:
                r = transform[:3, :3]
                if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
                        or not np.allclose(r.T @ r, np.eye(3), atol=1e-8)
                        or not np.isclose(np.linalg.det(r), 1, atol=1e-8)):
                    raise ValueError(f"{name} must contain rigid transforms")
            setattr(self, name, transforms)

    def to_model_state(self, state):
        state = _finite(state, (20,), "delivery state")
        output = np.empty(14, dtype=np.float32)
        for model_arm, side in enumerate(self.model_arm_order):
            arm = ("left", "right").index(side)
            raw = state[arm * 10:arm * 10 + 10]
            pose = np.eye(4)
            pose[:3, :3] = rotation6d_to_matrix(raw[3:9])
            pose[:3, 3] = raw[:3]
            transformed = self.model_base_from_client_base[arm] @ pose @ self.client_tool_from_model_tool[arm]
            offset = model_arm * 7
            output[offset:offset + 3] = transformed[:3, 3]
            output[offset + 3:offset + 6] = Rotation.from_matrix(transformed[:3, :3]).as_euler("xyz")
            output[offset + 6] = _opening(raw[9], 1., "state gripper fraction") * GRIPPER_MAX_M
        return output

    def to_client_targets(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("model actions must be a nonempty finite (T,14) array")
        output = np.empty((len(actions), 20), dtype=np.float32)
        for model_arm, side in enumerate(self.model_arm_order):
            arm = ("left", "right").index(side)
            raw = actions[:, model_arm * 7:model_arm * 7 + 7]
            poses = np.tile(np.eye(4), (len(actions), 1, 1))
            poses[:, :3, :3] = Rotation.from_euler("xyz", raw[:, 3:6]).as_matrix()
            poses[:, :3, 3] = raw[:, :3]
            transformed = np.linalg.inv(self.model_base_from_client_base[arm]) @ poses @ np.linalg.inv(self.client_tool_from_model_tool[arm])
            offset = arm * 10
            output[:, offset:offset + 3] = transformed[:, :3, 3]
            output[:, offset + 3:offset + 9] = matrix_to_rotation6d(transformed[:, :3, :3])
            output[:, offset + 9] = _opening(raw[:, 6], GRIPPER_MAX_M, "model gripper metres") / GRIPPER_MAX_M
        return output


def execution_ready_from_config(settings):
    """Require confirmed SDK semantics and an offline feedback/FK check."""
    if settings.get("validated") is False:
        return False
    if settings.get("validated") is not True:
        raise ValueError("validated must be an explicit boolean")
    output = settings.get("confirmed_output", {})
    if (settings.get("confirmed_sdk_contract") != SDK_CONTRACT
            or not settings.get("confirmation_source")
            or output.get("order") != OUTPUT_ORDER
            or output.get("position_unit") != "metres"
            or output.get("rpy_unit") != "radians"
            or output.get("gripper_unit") != "metres"):
        raise ValueError("execution requires the confirmed SDK input/output contract")
    bridge = EEFBridge(**settings["transforms"])
    if (list(bridge.model_arm_order) != ["left", "right"]
            or not np.allclose(bridge.model_base_from_client_base, np.tile(np.eye(4), (2, 1, 1)), atol=1e-12)
            or not np.allclose(bridge.client_tool_from_model_tool, np.tile(np.eye(4), (2, 1, 1)), atol=1e-12)):
        raise ValueError("confirmed native SDK frames require identity base/tool transforms")
    check = settings.get("kinematics_validation", {})
    position = float(check.get("max_position_error_m", float("inf")))
    rotation = float(check.get("max_rotation_error_rad", float("inf")))
    if (int(check.get("samples", 0)) < 2 or not check.get("source_log")
            or not 0 <= position <= .005 or not 0 <= rotation <= .05):
        raise ValueError("execution requires a passing recorded feedback/FK check")
    return True


def compatibility_metadata(source, horizon=50, *, execution_ready=False):
    return dict(
        transport="openpi_websocket_v1", schema="delivery", arm_mode="bimanual", arm_side="both",
        state_dim=20, action_dim=14, wire_action_dim=14, contract_version=3,
        action_semantics=DELIVERY_MODEL_ACTION_SEMANTICS, delivery_action_convention="chunk_origin",
        gripper_semantics=GRIPPER_OPENING_SEMANTICS, state_gripper_semantics=GRIPPER_OPENING_SEMANTICS,
        camera_keys=list(CAMERAS), action_hz=30, action_horizon=horizon, rtc_supported=False,
        image_resize_mode="pad", image_size=224, image_transport="raw",
        deployment_ready=bool(execution_ready), shadow_compatible=True,
        deployment_reason=("SDK EEF compatibility contract validated; robot commands remain subject to local safety checks"
                           if execution_ready else "Full EEF bridge requires training coordinate/tool-frame and IK calibration validation"),
        compatibility_adapter="full_eef_xyz_rpy_metres_to_delivery_v3", checkpoint_source=source,
    )


class FullEEFPolicy:
    def __init__(self, policy, bridge=None, *, default_prompt="fold the paper box", horizon=50,
                 execution_ready=False):
        self.policy = policy
        self.bridge = bridge or EEFBridge()
        self.default_prompt, self.horizon = default_prompt, horizon
        self.execution_ready = bool(execution_ready)
        self._prompts = OrderedDict()

    def infer(self, observation):
        client = observation.get("client_metadata") or {}
        if client.get("allow_execution") and not self.execution_ready:
            raise ValueError("Full EEF compatibility service is shadow-only; disable allow-execution")
        if (client.get("rtc") or {}).get("enabled") or "action_prefix" in observation:
            raise ValueError("Full EEF checkpoint does not support RTC prefixes")
        state = _finite(observation["state"], (20,), "delivery state")
        inputs = {"state": self.bridge.to_model_state(state), "images": {}}
        for key in CAMERAS:
            image = np.asarray(observation["images"][key])
            if image.shape != (3, 224, 224) or image.dtype != np.uint8:
                raise ValueError(f"{key} must be uint8 CHW 224x224")
            inputs["images"][key] = image
        session = str((client.get("rtc") or {}).get("session_id") or "")
        prompt = observation.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("prompt must be nonempty text")
        if session and prompt is not None:
            self._prompts[session] = prompt
            self._prompts.move_to_end(session)
            if len(self._prompts) > 64:
                self._prompts.popitem(last=False)
        inputs["prompt"] = prompt or self._prompts.get(session, self.default_prompt)
        result = dict(self.policy.infer(inputs))
        raw = _finite(result["actions"], (self.horizon, 14), "model action chunk")
        absolute = self.bridge.to_client_targets(raw)
        result["actions"] = absolute_eef_targets_to_chunk_origin(state, absolute, arm_count=2)
        result["raw_model_actions"] = raw.astype(np.float32)
        return result

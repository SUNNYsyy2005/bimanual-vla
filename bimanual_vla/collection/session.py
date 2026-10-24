"""UI-neutral state machine for Piper data collection."""

from __future__ import annotations

from bimanual_vla.collection.can_identity import resolve_can_pair

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np
import time

from bimanual_vla.collection.camera import CameraCapture
from bimanual_vla.collection.output import (
    CAMERA_SOURCE_HW,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAN,
    DEFAULT_HIGH_DEVICE,
    DEFAULT_LEFT_CAN,
    DEFAULT_LEFT_WRIST_DEVICE,
    DEFAULT_RIGHT_CAN,
    DEFAULT_RIGHT_WRIST_DEVICE,
    DEFAULT_WRIST_DEVICE,
    PiperFeedbackStaleError,
    connect,
    next_episode_index,
    read_robot_gripper_command_samples,
    read_robot_sample,
    RobotStateSample,
    verify_camera_streams,
)
from bimanual_vla.data.contract import (
    BIMANUAL,
    DEFAULT_ACTION_HORIZON,
    DELIVERY_SCHEMA,
    DELIVERY_COMMANDED_GRIPPER_ACTION_SOURCE,
    JOINT_MEASURED_ACTION_SOURCE,
    JOINT_SCHEMA,
    SINGLE_ARM,
    EpisodeBuffer,
    EpisodeContract,
)
from bimanual_vla.data.validate import EpisodeStats, validate_episode


class SessionState(str, Enum):
    DISCONNECTED = "disconnected"
    READY = "ready"
    RECORDING = "recording"
    REVIEW = "review"


@dataclass(frozen=True)
class CollectionConfig:
    can_name: str = DEFAULT_CAN
    cam_high_device: str = DEFAULT_HIGH_DEVICE
    cam_wrist_device: str = DEFAULT_WRIST_DEVICE
    capture_fps: int = DEFAULT_CAMERA_FPS
    action_horizon: int = DEFAULT_ACTION_HORIZON
    camera_fps: int = DEFAULT_CAMERA_FPS
    output_dir: Path = Path("episodes_piper_v21")
    schema: str = DELIVERY_SCHEMA
    record_both: bool = False
    arm_mode: str = SINGLE_ARM
    arm_side: str = "right"
    left_can_name: str = DEFAULT_LEFT_CAN
    right_can_name: str = DEFAULT_RIGHT_CAN
    cam_left_wrist_device: str = DEFAULT_LEFT_WRIST_DEVICE
    cam_right_wrist_device: str = DEFAULT_RIGHT_WRIST_DEVICE

    def __post_init__(self):
        if self.capture_fps <= 0:
            raise ValueError("capture_fps must be positive")
        if self.camera_fps not in (15, 30):
            raise ValueError("camera_fps must be 15 or 30 on this rig")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.capture_fps > self.camera_fps:
            raise ValueError("capture_fps cannot exceed camera_fps")
        if self.record_both and self.schema != DELIVERY_SCHEMA:
            raise ValueError("record_both requires the delivery primary schema")
        EpisodeContract(
            schema=self.schema,
            arm_mode=self.arm_mode,
            arm_side=self.arm_side,
        )
        can_names = (
            (self.can_name,)
            if self.arm_mode == SINGLE_ARM
            else (self.left_can_name, self.right_can_name)
        )
        if any(not name.strip() for name in can_names):
            raise ValueError("CAN interface names must not be empty")
        if len(set(can_names)) != len(can_names):
            raise ValueError("bimanual CAN interface names must be distinct")


@dataclass(frozen=True)
class EpisodeLabel:
    task_name: str
    instruction: str

    def __post_init__(self):
        if not self.task_name.strip():
            raise ValueError("task_name must not be empty")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")


@dataclass(frozen=True)
class CaptureSample:
    state: Any
    joint_qpos: Any
    gripper_command_targets: Any | None
    gripper_command_timestamps: Any | None
    images: dict[str, Any]
    image_timestamps: dict[str, float]
    state_timestamp: float


class CollectionSession:
    """Own devices and one episode while leaving rendering to the UI."""

    def __init__(
        self,
        config: CollectionConfig,
        robot_connect: Callable[[str], Any] = connect,
        camera_factory: Callable[..., CameraCapture] = CameraCapture,
        state_reader: Callable[[Any], RobotStateSample | tuple[Any, Any]] | None = None,
        gripper_command_reader: Callable[[Any], Any | None] | None = None,
        camera_verifier: Callable[[Any, int], dict[str, dict]] = verify_camera_streams,
        episode_validator: Callable[..., EpisodeStats] = validate_episode,
    ):
        self.config = config
        self._robot_connect = robot_connect
        self._camera_factory = camera_factory
        self._state_reader = state_reader or (
            lambda robot: read_robot_sample(
                robot,
                schema=self.config.schema,
                arm_mode=self.config.arm_mode,
                arm_side=self.config.arm_side,
            )
        )
        self._gripper_command_reader = gripper_command_reader or (
            lambda robot: read_robot_gripper_command_samples(
                robot, arm_mode=self.config.arm_mode
            )
        )
        self._camera_verifier = camera_verifier
        self._episode_validator = episode_validator
        self.state = SessionState.DISCONNECTED
        self.piper = None
        self.cameras = None
        self.buffer: EpisodeBuffer | None = None
        self.label: EpisodeLabel | None = None
        self.camera_checks: dict[str, dict] = {}
        self._last_camera_timestamps: dict[str, float] = {}
        self.episode_index = next_episode_index(Path(config.output_dir))

    def _wait_for_robot_feedback(self, robot: Any, timeout_s: float = 3.0) -> None:
        """Wait briefly for Piper's reader thread to populate live CAN feedback."""
        deadline = time.monotonic() + timeout_s
        last_error: PiperFeedbackStaleError | None = None
        while True:
            try:
                self._state_reader(robot)
                return
            except PiperFeedbackStaleError as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "CAN socket opened, but no fresh Piper feedback arrived within "
                    f"{timeout_s:.1f}s. Check that the interface is UP and verify live frames with "
                    "'candump -L <can-interface>'."
                ) from last_error
            time.sleep(0.05)

    @property
    def frame_count(self) -> int:
        return len(self.buffer) if self.buffer is not None else 0

    def connect(self) -> dict[str, dict]:
        if self.state is not SessionState.DISCONNECTED:
            raise RuntimeError(f"cannot connect while session is {self.state.value}")
        contract = EpisodeContract(
            schema=self.config.schema,
            arm_mode=self.config.arm_mode,
            arm_side=self.config.arm_side,
        )
        connected: list[Any] = []
        try:
            if self.config.arm_mode == SINGLE_ARM:
                piper = self._robot_connect(self.config.can_name)
                connected.append(piper)
            else:
                resolve_can_pair(self.config.left_can_name, self.config.right_can_name)
                left = self._robot_connect(self.config.left_can_name)
                connected.append(left)
                right = self._robot_connect(self.config.right_can_name)
                connected.append(right)
                piper = {"left": left, "right": right}
            self._wait_for_robot_feedback(piper)
            if self.config.arm_mode == BIMANUAL:
                camera_ids = {
                    "cam_high": self.config.cam_high_device,
                    "cam_left_wrist": self.config.cam_left_wrist_device,
                    "cam_right_wrist": self.config.cam_right_wrist_device,
                }
            else:
                camera_ids = {
                    "cam_high": self.config.cam_high_device,
                    contract.camera_keys[1]: self.config.cam_wrist_device,
                }
            cameras = self._camera_factory(
                cam_ids=camera_ids,
                fps=self.config.camera_fps,
                image_hw=(224, 224),
                resize_mode="stretch",
                capture_hw=CAMERA_SOURCE_HW,
                parallel_reads=True,
            )
            cameras.open()
            checks = self._camera_verifier(cameras, self.config.camera_fps)
            if hasattr(cameras, "start_background_capture"):
                cameras.start_background_capture()
        except Exception:
            if "cameras" in locals():
                cameras.close()
            for item in reversed(connected):
                item.DisconnectPort()
            raise
        self.piper = piper
        self.cameras = cameras
        self._last_camera_timestamps = {}
        self.camera_checks = checks
        self.state = SessionState.READY
        return checks

    def reconnect_cameras(self) -> dict[str, dict]:
        """Reconnect only cameras while keeping the robot/CAN session alive.

        This is used by the GUI's wrist-camera swap control. A failed new
        camera open rolls back to the previous camera mapping where possible,
        so a temporary V4L2 error does not leave the live preview black.
        """
        if self.state is SessionState.DISCONNECTED or self.piper is None:
            raise RuntimeError("cannot reconnect cameras while devices are disconnected")
        contract = EpisodeContract(
            schema=self.config.schema,
            arm_mode=self.config.arm_mode,
            arm_side=self.config.arm_side,
        )
        camera_ids = {
            "cam_high": self.config.cam_high_device,
        }
        if self.config.arm_mode == BIMANUAL:
            camera_ids.update(
                {
                    "cam_left_wrist": self.config.cam_left_wrist_device,
                    "cam_right_wrist": self.config.cam_right_wrist_device,
                }
            )
        else:
            camera_ids[contract.camera_keys[1]] = self.config.cam_wrist_device

        old_cameras = self.cameras
        if old_cameras is not None:
            old_cameras.close()
        new_cameras = None
        try:
            new_cameras = self._camera_factory(
                cam_ids=camera_ids,
                fps=self.config.camera_fps,
                image_hw=(224, 224),
                resize_mode="stretch",
                capture_hw=CAMERA_SOURCE_HW,
                parallel_reads=True,
            )
            new_cameras.open()
            checks = self._camera_verifier(new_cameras, self.config.camera_fps)
            if hasattr(new_cameras, "start_background_capture"):
                new_cameras.start_background_capture()
        except Exception:
            if new_cameras is not None:
                new_cameras.close()
            if old_cameras is not None:
                try:
                    old_cameras.open()
                    if hasattr(old_cameras, "start_background_capture"):
                        old_cameras.start_background_capture()
                    self.cameras = old_cameras
                except Exception:
                    self.cameras = None
            raise
        self.cameras = new_cameras
        self._last_camera_timestamps = {}
        self.camera_checks = checks
        return checks

    def start_episode(self, task_name: str, instruction: str) -> EpisodeLabel:
        if self.state is not SessionState.READY:
            raise RuntimeError(f"cannot start an episode while session is {self.state.value}")
        self.label = EpisodeLabel(task_name.strip(), instruction.strip())
        self.buffer = EpisodeBuffer(
            self.config.capture_fps,
            schema=self.config.schema,
            arm_mode=self.config.arm_mode,
            arm_side=self.config.arm_side,
            action_source=(
                JOINT_MEASURED_ACTION_SOURCE
                if self.config.schema == JOINT_SCHEMA
                else DELIVERY_COMMANDED_GRIPPER_ACTION_SOURCE
            ),
            action_alignment=("next_observation" if self.config.schema == JOINT_SCHEMA
                              else "next_observation_pose_same_step_gripper"),
            action_horizon=self.config.action_horizon,
            record_both=self.config.record_both,
            image_resize_mode="stretch",
        )
        self.state = SessionState.RECORDING
        return self.label

    def capture_once(self) -> CaptureSample:
        if self.state not in {
            SessionState.READY,
            SessionState.RECORDING,
            SessionState.REVIEW,
        }:
            raise RuntimeError("devices are not connected")
        frame_set = None
        if self.config.capture_fps == self.config.camera_fps and hasattr(self.cameras, "read_next"):
            # At 30/30 Hz the camera stream is the clock. Two independent timers
            # can otherwise select the same frame twice despite a healthy camera.
            frame_set = self.cameras.read_next(self._last_camera_timestamps)
        measured = self._state_reader(self.piper)
        captured_monotonic = time.monotonic()
        if isinstance(measured, RobotStateSample):
            state, joint_qpos = measured.state, measured.joint_qpos
            state_timestamp = measured.state_timestamp
            feedback_timestamps = measured.feedback_timestamps
        else:
            state, joint_qpos = measured  # Compatibility for injected readers.
            state_timestamp = time.time()
            feedback_timestamps = None
        gripper_command_targets = None
        gripper_command_timestamps = None
        if self.config.schema == DELIVERY_SCHEMA:
            command_sample = self._gripper_command_reader(self.piper)
            if isinstance(command_sample, tuple) and len(command_sample) == 2:
                gripper_command_targets, gripper_command_timestamps = command_sample
                if gripper_command_targets is not None and gripper_command_timestamps is not None:
                    gripper_command_targets = np.asarray(gripper_command_targets, dtype=np.float32).copy()
                    gripper_command_timestamps = np.asarray(gripper_command_timestamps, dtype=np.float64).copy()
                    aligned = np.isfinite(gripper_command_timestamps) & (np.abs(gripper_command_timestamps - state_timestamp) <= 0.050)
                    gripper_command_targets[~aligned] = np.nan
                    gripper_command_timestamps[~aligned] = np.nan
            else:
                # Preserve custom value-only readers used by older UIs/tests.
                gripper_command_targets = command_sample
        if frame_set is not None:
            images, image_timestamps = frame_set.images, frame_set.timestamps
        elif hasattr(self.cameras, "read_nearest"):
            frame_set = self.cameras.read_nearest(captured_monotonic)
            images, image_timestamps = frame_set.images, frame_set.timestamps
        else:
            images, image_timestamps = self.cameras.read()
        self._last_camera_timestamps = dict(image_timestamps)
        if self.state is SessionState.RECORDING:
            assert self.buffer is not None
            self.buffer.add(
                state,
                images,
                image_timestamps,
                qpos=joint_qpos,
                gripper_targets=gripper_command_targets,
                gripper_command_timestamps=gripper_command_timestamps,
                state_timestamp=state_timestamp,
                feedback_timestamps=feedback_timestamps,
            )
        return CaptureSample(
            state=state,
            joint_qpos=joint_qpos,
            gripper_command_targets=gripper_command_targets,
            gripper_command_timestamps=gripper_command_timestamps,
            images=images,
            image_timestamps=image_timestamps,
            state_timestamp=state_timestamp,
        )

    def stop_episode(self) -> int:
        if self.state is not SessionState.RECORDING:
            raise RuntimeError(f"cannot stop an episode while session is {self.state.value}")
        self.state = SessionState.REVIEW
        return self.frame_count

    def save_episode(
        self,
        success: bool,
        task_name: str | None = None,
        instruction: str | None = None,
        validate: bool = True,
    ) -> tuple[Path, EpisodeStats | None]:
        if self.state is not SessionState.REVIEW or self.buffer is None or self.label is None:
            raise RuntimeError("there is no stopped episode to save")
        label = EpisodeLabel(
            (task_name if task_name is not None else self.label.task_name).strip(),
            (instruction if instruction is not None else self.label.instruction).strip(),
        )
        self.episode_index = max(
            self.episode_index,
            next_episode_index(Path(self.config.output_dir)),
        )
        path = Path(self.config.output_dir) / f"ep_{self.episode_index:04d}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing episode: {path}")
        stats = None
        if validate:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.stem}.",
                suffix=".npz",
                dir=path.parent,
                delete=False,
            ) as candidate_file:
                candidate = Path(candidate_file.name)
            try:
                self.buffer.save(candidate, label.task_name, label.instruction, success)
                stats = self._episode_validator(
                    candidate,
                    target_fps=self.config.capture_fps,
                )
                candidate.replace(path)
                stats = replace(stats, path=path)
            finally:
                candidate.unlink(missing_ok=True)
        else:
            self.buffer.save(path, label.task_name, label.instruction, success)
        self.episode_index += 1
        self.buffer = None
        self.label = None
        self.state = SessionState.READY
        return path, stats

    def discard_episode(self) -> None:
        if self.state is not SessionState.REVIEW:
            raise RuntimeError("there is no stopped episode to discard")
        self.buffer = None
        self.label = None
        self.state = SessionState.READY

    def disconnect(self, discard_review: bool = False) -> None:
        if self.state is SessionState.RECORDING:
            raise RuntimeError("stop the current episode before disconnecting")
        if self.state is SessionState.REVIEW:
            if not discard_review:
                raise RuntimeError("save or discard the stopped episode before disconnecting")
            self.discard_episode()
        if self.cameras is not None:
            self.cameras.close()
        if isinstance(self.piper, dict):
            for item in reversed(tuple(self.piper.values())):
                item.DisconnectPort()
        elif self.piper is not None:
            self.piper.DisconnectPort()
        self.cameras = None
        self.piper = None
        self.camera_checks = {}
        self.state = SessionState.DISCONNECTED

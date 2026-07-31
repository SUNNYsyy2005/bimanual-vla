"""UI-neutral state machine for Piper data collection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import tempfile
from typing import Any, Callable
import time

from camera import CameraCapture
from collect_output_arm import (
    CAMERA_SOURCE_HW,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAN,
    DEFAULT_HIGH_DEVICE,
    DEFAULT_WRIST_DEVICE,
    connect,
    next_episode_index,
    read_output_state,
    verify_camera_streams,
)
from piper_data_contract import DEFAULT_FPS, IMAGE_HW, EpisodeBuffer
from validate_piper_data import EpisodeStats, validate_episode


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
    capture_fps: int = DEFAULT_FPS
    camera_fps: int = DEFAULT_CAMERA_FPS
    output_dir: Path = Path("episodes_piper_v21")

    def __post_init__(self):
        if self.capture_fps <= 0:
            raise ValueError("capture_fps must be positive")
        if self.camera_fps <= 0:
            raise ValueError("camera_fps must be positive")
        if self.capture_fps > self.camera_fps:
            raise ValueError("capture_fps cannot exceed camera_fps")
        if not self.can_name.strip():
            raise ValueError("can_name must not be empty")


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
        state_reader: Callable[[Any], tuple[Any, Any]] = read_output_state,
        camera_verifier: Callable[[Any, int], dict[str, dict]] = verify_camera_streams,
        episode_validator: Callable[..., EpisodeStats] = validate_episode,
    ):
        self.config = config
        self._robot_connect = robot_connect
        self._camera_factory = camera_factory
        self._state_reader = state_reader
        self._camera_verifier = camera_verifier
        self._episode_validator = episode_validator
        self.state = SessionState.DISCONNECTED
        self.piper = None
        self.cameras = None
        self.buffer: EpisodeBuffer | None = None
        self.label: EpisodeLabel | None = None
        self.camera_checks: dict[str, dict] = {}
        self.episode_index = next_episode_index(Path(config.output_dir))

    @property
    def frame_count(self) -> int:
        return len(self.buffer) if self.buffer is not None else 0

    def connect(self) -> dict[str, dict]:
        if self.state is not SessionState.DISCONNECTED:
            raise RuntimeError(f"cannot connect while session is {self.state.value}")
        piper = self._robot_connect(self.config.can_name)
        cameras = self._camera_factory(
            cam_ids={
                "cam_high": self.config.cam_high_device,
                "cam_wrist": self.config.cam_wrist_device,
            },
            fps=self.config.camera_fps,
            image_hw=IMAGE_HW,
            capture_hw=CAMERA_SOURCE_HW,
            parallel_reads=True,
        )
        try:
            cameras.open()
            checks = self._camera_verifier(cameras, self.config.camera_fps)
        except Exception:
            cameras.close()
            piper.DisconnectPort()
            raise
        self.piper = piper
        self.cameras = cameras
        self.camera_checks = checks
        self.state = SessionState.READY
        return checks

    def start_episode(self, task_name: str, instruction: str) -> EpisodeLabel:
        if self.state is not SessionState.READY:
            raise RuntimeError(f"cannot start an episode while session is {self.state.value}")
        self.label = EpisodeLabel(task_name.strip(), instruction.strip())
        self.buffer = EpisodeBuffer(self.config.capture_fps)
        self.state = SessionState.RECORDING
        return self.label

    def capture_once(self) -> CaptureSample:
        if self.state not in {
            SessionState.READY,
            SessionState.RECORDING,
            SessionState.REVIEW,
        }:
            raise RuntimeError("devices are not connected")
        state, joint_qpos = self._state_reader(self.piper)
        state_timestamp = time.time()
        images, image_timestamps = self.cameras.read()
        if self.state is SessionState.RECORDING:
            assert self.buffer is not None
            self.buffer.add(
                state,
                images,
                image_timestamps,
                qpos=joint_qpos,
                state_timestamp=state_timestamp,
            )
        return CaptureSample(
            state=state,
            joint_qpos=joint_qpos,
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
        if self.piper is not None:
            self.piper.DisconnectPort()
        self.cameras = None
        self.piper = None
        self.camera_checks = {}
        self.state = SessionState.DISCONNECTED

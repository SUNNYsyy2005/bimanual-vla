#!/usr/bin/env python3
"""Single-arm and bimanual Piper π0/π0.5 training, norm-stat, and serving entrypoint.

Run this file with the Python environment of an OpenPI checkout and use the
checkout as the current working directory. It intentionally builds the config
in Python so an uploaded LeRobot repo_id can be selected without editing the
upstream OpenPI config registry.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
from websockets.exceptions import ConnectionClosedError, InvalidMessage

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import data_loader
from openpi.training import weight_loaders

try:
    from .episode_split import EpisodeSplit, resolve_episode_split, write_norm_split
except ImportError:  # openpi_single_arm.py is normally executed directly
    from episode_split import EpisodeSplit, resolve_episode_split, write_norm_split


class _ExpectedWebsocketProbeFilter(logging.Filter):
    """Drop expected health-check and bare-TCP probe noise from websockets."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        # OpenPI serves HTTP /healthz from process_request. websockets reports
        # every successful health response as a rejected WebSocket connection.
        if message == "connection rejected (200 OK)":
            return False
        # Port scanners and TCP readiness probes may connect and close without
        # sending an HTTP Upgrade request. Keep all other handshake errors.
        exception = record.exc_info[1] if record.exc_info else None
        if message == "opening handshake failed":
            if isinstance(exception, ConnectionClosedError):
                return False
            if isinstance(exception, InvalidMessage):
                cause: BaseException | None = exception
                while cause is not None:
                    if isinstance(cause, EOFError):
                        return False
                    cause = cause.__cause__
        return True


def _install_websocket_probe_filter() -> None:
    logger = logging.getLogger("websockets.server")
    if not any(isinstance(item, _ExpectedWebsocketProbeFilter) for item in logger.filters):
        logger.addFilter(_ExpectedWebsocketProbeFilter())


CONFIG_NAMES = {
    ("pi05", "single"): "pi05_piper_single_arm_lora",
    ("pi05", "bimanual"): "pi05_piper_bimanual_lora",
    ("pi0", "single"): "pi0_piper_single_arm_lora",
    ("pi0", "bimanual"): "pi0_piper_bimanual_lora",
}


def config_name(model_variant: str, arm_mode: str) -> str:
    try:
        return CONFIG_NAMES[(model_variant, arm_mode)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported model/arm combination: model_variant={model_variant!r}, arm_mode={arm_mode!r}"
        ) from exc


@dataclasses.dataclass(frozen=True)
class DatasetContract:
    schema: str
    arm_mode: str
    arm_side: str
    layout: str
    state_dim: int
    action_dim: int
    camera_keys: tuple[str, ...]
    action_semantics: str
    action_source: str
    action_alignment: str


def _dataset_root() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))


def _dataset_info(dataset_id: str) -> dict[str, Any]:
    path = _dataset_root() / dataset_id / "meta" / "info.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"invalid LeRobot info.json: {path}")
    return value


def resolve_dataset_contract(args: argparse.Namespace) -> DatasetContract:
    info = _dataset_info(args.dataset_id)
    features = info.get("features", {}) if isinstance(info.get("features", {}), dict) else {}
    if "observation.state" in features and "action" in features:
        layout = "canonical"
        state_key, action_key = "observation.state", "action"
    elif "state" in features and "actions" in features:
        layout = "legacy"
        state_key, action_key = "state", "actions"
    else:
        layout = str(args.dataset_layout)
        if layout == "auto":
            raise ValueError(
                f"cannot infer dataset layout for {args.dataset_id!r}; expected canonical "
                "observation.state/action or legacy state/actions features"
            )
        state_key, action_key = (("observation.state", "action") if layout == "canonical" else ("state", "actions"))

    def last_dim(key: str) -> int | None:
        feature = features.get(key, {})
        shape = feature.get("shape") if isinstance(feature, dict) else None
        return int(shape[-1]) if isinstance(shape, list) and shape else None

    state_dim = last_dim(state_key)
    action_dim = last_dim(action_key)
    by_dims = {
        (7, 7): ("joint", "single"),
        (10, 7): ("delivery", "single"),
        (14, 14): ("joint", "bimanual"),
        (20, 14): ("delivery", "bimanual"),
    }
    inferred = by_dims.get((state_dim, action_dim))
    schema = str(info.get("schema") or (inferred[0] if inferred else args.schema)).lower()
    arm_mode = str(info.get("arm_mode") or (inferred[1] if inferred else args.arm_mode)).lower()
    if args.schema != "auto" and schema != args.schema:
        raise ValueError(f"--schema={args.schema} conflicts with dataset schema={schema}")
    if args.arm_mode != "auto" and arm_mode != args.arm_mode:
        raise ValueError(f"--arm-mode={args.arm_mode} conflicts with dataset arm_mode={arm_mode}")
    if schema not in {"joint", "delivery"} or arm_mode not in {"single", "bimanual"}:
        raise ValueError(f"unsupported dataset contract: schema={schema!r}, arm_mode={arm_mode!r}")
    expected_dims = {
        ("joint", "single"): (7, 7),
        ("delivery", "single"): (10, 7),
        ("joint", "bimanual"): (14, 14),
        ("delivery", "bimanual"): (20, 14),
    }[(schema, arm_mode)]
    if state_dim is not None and (state_dim, action_dim) != expected_dims:
        raise ValueError(
            f"dataset dimensions {(state_dim, action_dim)} disagree with {arm_mode} {schema} "
            f"expected {expected_dims}"
        )
    state_dim, action_dim = expected_dims

    arm_side = "both" if arm_mode == "bimanual" else str(info.get("arm_side") or args.arm_side).lower()
    if arm_mode == "single" and arm_side not in {"left", "right"}:
        raise ValueError("single-arm dataset requires arm_side left or right")
    if arm_mode == "bimanual" and args.arm_side not in {"both", "right"}:
        raise ValueError("bimanual dataset requires --arm-side both")

    media = [
        key.removeprefix("observation.images.")
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
    ]
    if layout == "legacy":
        camera_keys = ("cam_high", "cam_wrist")
    elif arm_mode == "bimanual":
        camera_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    else:
        expected = f"cam_{arm_side}_wrist"
        wrist = expected if expected in media else "cam_wrist" if "cam_wrist" in media else expected
        camera_keys = ("cam_high", wrist)
    missing_media = [
        f"observation.images.{key}" for key in camera_keys
        if layout == "canonical" and f"observation.images.{key}" not in features
    ]
    if missing_media:
        raise ValueError(f"dataset is missing required camera features: {missing_media}")
    if layout == "legacy" and arm_mode != "single":
        raise ValueError("legacy image/wrist_image layout only supports single-arm delivery data")
    if layout == "legacy" and schema != "delivery":
        raise ValueError("legacy state/actions layout only supports delivery schema")

    return DatasetContract(
        schema=schema,
        arm_mode=arm_mode,
        arm_side=arm_side,
        layout=layout,
        state_dim=state_dim,
        action_dim=action_dim,
        camera_keys=camera_keys,
        action_semantics=str(
            info.get("action_semantics")
            or ("absolute_joint_position" if schema == "joint" else "eef_delta_base_xyz_left_rotvec_gripper_target")
        ),
        action_source=str(info.get("action_source") or "unknown"),
        action_alignment=str(info.get("action_alignment") or "unknown"),
    )


def _as_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"image must be rank 3, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"image must have three RGB channels, got {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Map single-arm or bimanual Piper observations into OpenPI inputs."""

    contract: DatasetContract

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != self.contract.state_dim:
            raise ValueError(
                f"Piper {self.contract.arm_mode} {self.contract.schema} state must be "
                f"{self.contract.state_dim}D, got {state.shape}"
            )
        images = data["images"]
        expected = set(self.contract.camera_keys)
        if set(images) != expected:
            raise ValueError(f"camera keys must be {sorted(expected)}, got {sorted(images)}")
        high = _as_hwc_uint8(images["cam_high"])
        if self.contract.arm_mode == "bimanual":
            left = _as_hwc_uint8(images["cam_left_wrist"])
            right = _as_hwc_uint8(images["cam_right_wrist"])
            mapped_images = {
                "base_0_rgb": high,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            }
            image_mask = {key: np.True_ for key in mapped_images}
        else:
            wrist_key = next(key for key in self.contract.camera_keys if "wrist" in key)
            wrist = _as_hwc_uint8(images[wrist_key])
            mapped_images = {
                "base_0_rgb": high,
                "left_wrist_0_rgb": wrist if self.contract.arm_side == "left" else np.zeros_like(wrist),
                "right_wrist_0_rgb": wrist if self.contract.arm_side == "right" else np.zeros_like(wrist),
            }
            image_mask = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.bool_(self.contract.arm_side == "left"),
                "right_wrist_0_rgb": np.bool_(self.contract.arm_side == "right"),
            }
        output = {"image": mapped_images, "image_mask": image_mask, "state": state}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != self.contract.action_dim:
                raise ValueError(
                    f"Piper {self.contract.arm_mode} actions must be "
                    f"{self.contract.action_dim}D, got {actions.shape}"
                )
            output["actions"] = actions
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}


@dataclasses.dataclass(frozen=True)
class RemoveStrings(transforms.DataTransformFn):
    """Drop prompt/string fields before numeric norm-stat accumulation."""

    def __call__(self, sample: dict) -> dict:
        return {
            key: value
            for key, value in sample.items()
            if not np.issubdtype(np.asarray(value).dtype, np.str_)
        }


@dataclasses.dataclass(frozen=True)
class PiperDataConfig(training_config.DataConfigFactory):
    contract: DatasetContract | None = None
    default_prompt: str | None = None

    def create(self, assets_dirs: Path, model_config) -> training_config.DataConfig:
        if self.contract is None:
            raise ValueError("dataset contract is required")
        if self.contract.layout == "legacy":
            repack_mapping = {
                "images": {"cam_high": "image", "cam_wrist": "wrist_image"},
                "state": "state",
                "actions": "actions",
                "prompt": "prompt",
            }
            action_sequence_keys = ("actions",)
        else:
            repack_mapping = {
                "images": {
                    key: f"observation.images.{key}" for key in self.contract.camera_keys
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }
            action_sequence_keys = ("action",)

        repack = transforms.Group(inputs=[transforms.RepackTransform(repack_mapping)])
        robot_transforms = transforms.Group(
            inputs=[PiperInputs(contract=self.contract)],
            outputs=[PiperOutputs(action_dim=self.contract.action_dim)],
        )
        if self.contract.schema == "joint":
            mask = (
                transforms.make_bool_mask(6, -1, 6, -1)
                if self.contract.arm_mode == "bimanual"
                else transforms.make_bool_mask(6, -1)
            )
            robot_transforms = robot_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        model_transforms = training_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=robot_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=action_sequence_keys,
        )


def build_config(args: argparse.Namespace) -> training_config.TrainConfig:
    contract = resolve_dataset_contract(args)
    model_variant = str(getattr(args, "model_variant", "pi05"))
    model = pi0_config.Pi0Config(
        pi05=model_variant == "pi05",
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    params_path = base_checkpoint / "params"
    if args.command == "train" and not params_path.exists():
        raise FileNotFoundError(
            f"base checkpoint params not found: {params_path}. "
            f"Install a compatible {model_variant} checkpoint or choose another base model."
        )
    data_factory = PiperDataConfig(
        repo_id=args.dataset_id,
        contract=contract,
        base_config=training_config.DataConfig(prompt_from_task=True),
    )
    return training_config.TrainConfig(
        name=config_name(model_variant, contract.arm_mode),
        exp_name=getattr(args, "exp_name", "runtime"),
        model=model,
        data=data_factory,
        freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(params_path)),
        assets_base_dir=str(Path(args.assets_base_dir).expanduser().resolve()),
        checkpoint_base_dir=str(Path(args.checkpoint_base_dir).expanduser().resolve()),
        batch_size=getattr(args, "batch_size", 8),
        num_workers=getattr(args, "num_workers", 2),
        num_train_steps=getattr(args, "num_train_steps", 30_000),
        save_interval=getattr(args, "save_interval", 1_000),
        log_interval=getattr(args, "log_interval", 100),
        fsdp_devices=getattr(args, "fsdp_devices", 1),
        resume=getattr(args, "resume", False),
        overwrite=getattr(args, "overwrite", False),
        wandb_enabled=getattr(args, "wandb_enabled", False),
        policy_metadata={
            "robot_type": "piper_bimanual" if contract.arm_mode == "bimanual" else "piper_single_arm",
            "model_variant": model_variant,
            "base_checkpoint": str(base_checkpoint),
            "arm_mode": contract.arm_mode,
            "arm_side": contract.arm_side,
            "schema": contract.schema,
            "dataset_layout": contract.layout,
            "state_dim": contract.state_dim,
            "action_dim": contract.action_dim,
            "camera_keys": list(contract.camera_keys),
            "action_semantics": contract.action_semantics,
            "action_source": contract.action_source,
            "action_alignment": contract.action_alignment,
            "transport": "openpi_websocket_v1",
        },
    )


def _resolve_training_split(args: argparse.Namespace) -> EpisodeSplit:
    split = resolve_episode_split(
        _dataset_root(),
        args.dataset_id,
        test_ratio=float(getattr(args, "test_ratio", 0.1)),
        seed=int(getattr(args, "split_seed", 42)),
    )
    print(
        "Episode split: "
        f"train={len(split.train_episodes)} test={len(split.test_episodes)} "
        f"ratio={split.test_ratio:g} seed={split.seed} "
        f"test_episodes={list(split.test_episodes)}",
        flush=True,
    )
    if not split.test_episodes:
        logging.warning("dataset has fewer than two episodes or test_ratio=0; no held-out test episodes")
    return split


def _create_torch_dataset_for_episodes(
    data_config: training_config.DataConfig,
    action_horizon: int,
    model_config: Any,
    episodes: tuple[int, ...],
):
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return data_loader.FakeDataset(model_config, num_samples=1024)
    dataset_meta = data_loader.lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = data_loader.lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [step / dataset_meta.fps for step in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    # This pinned LeRobot version accepts non-contiguous episode lists but its
    # delta-query indexing still assumes contiguous episode ids. Keep the full
    # dataset's boundary table and select frame indices at the PyTorch layer.
    selected_episodes = set(episodes)
    sample_indices = [
        index
        for index, episode_index in enumerate(dataset.hf_dataset["episode_index"])
        if int(episode_index) in selected_episodes
    ]
    if not sample_indices:
        raise ValueError(f"episode subset for {repo_id!r} contains no frames")
    dataset = data_loader.torch.utils.data.Subset(dataset, sample_indices)
    if data_config.prompt_from_task:
        dataset = data_loader.TransformedDataset(
            dataset,
            [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )
    return dataset


def _install_training_episode_subset(dataset_id: str, episodes: tuple[int, ...]) -> None:
    """Make upstream OpenPI's training loader consume only selected episodes."""
    original = data_loader.create_torch_dataset

    @functools.wraps(original)
    def create_subset(data_config, action_horizon, model_config):
        if data_config.repo_id != dataset_id:
            return original(data_config, action_horizon, model_config)
        return _create_torch_dataset_for_episodes(data_config, action_horizon, model_config, episodes)

    data_loader.create_torch_dataset = create_subset


def _load_upstream_train_main(openpi_root: Path):
    train_path = openpi_root / "scripts" / "train.py"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    spec = importlib.util.spec_from_file_location("openpi_upstream_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def run_train(args: argparse.Namespace) -> None:
    split = _resolve_training_split(args)
    _install_training_episode_subset(args.dataset_id, split.train_episodes)
    config = build_config(args)
    train_main = _load_upstream_train_main(Path.cwd())
    train_main(config)


def run_norm(args: argparse.Namespace) -> None:
    split = _resolve_training_split(args)
    config = build_config(args)
    concrete = config.data.create(config.assets_dirs, config.model)
    dataset = _create_torch_dataset_for_episodes(
        concrete,
        config.model.action_horizon,
        config.model,
        split.train_episodes,
    )

    dataset = data_loader.TransformedDataset(
        dataset,
        [*concrete.repack_transforms.inputs, *concrete.data_transforms.inputs, RemoveStrings()],
    )
    if len(dataset) <= 0:
        raise ValueError(f"dataset {args.dataset_id!r} is empty")
    batch_size = min(args.batch_size, len(dataset))
    if args.max_frames is not None:
        effective_frames = min(args.max_frames, len(dataset))
        num_batches = max(1, effective_frames // batch_size)
        shuffle = effective_frames < len(dataset)
    else:
        num_batches = max(1, len(dataset) // batch_size)
        shuffle = False
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for index, batch in enumerate(loader, start=1):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
        if index % 20 == 0 or index == num_batches:
            print(f"norm stats: {index}/{num_batches} batches", flush=True)
    output = {key: value.get_statistics() for key, value in stats.items()}
    output_path = config.assets_dirs / args.dataset_id
    print(f"Writing stats to: {output_path}", flush=True)
    normalize.save(output_path, output)
    split_path = write_norm_split(output_path, split)
    print(f"Writing episode split manifest to: {split_path}", flush=True)


class PolicyTelemetry:
    """Mirror official WebSocket requests/results for the local dashboard."""

    def __init__(self, root: Path, metadata: dict[str, Any]):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.lock = threading.Lock()
        self.sequence = 0
        self.active_clients = 0
        self.client_addresses: set[str] = set()

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)

    @staticmethod
    def _atomic_image(path: Path, image: np.ndarray) -> list[int]:
        from PIL import Image

        image = _as_hwc_uint8(image)
        temp = path.with_suffix(path.suffix + ".tmp")
        Image.fromarray(image).save(temp, format="JPEG", quality=90)
        os.replace(temp, path)
        return list(image.shape)

    @staticmethod
    def _client_address(remote_address: Any) -> str:
        if isinstance(remote_address, tuple):
            return ":".join(map(str, remote_address))
        return str(remote_address or "unknown")

    def _publish_connections(self, *, event: str, address: str) -> None:
        now = time.time()
        payload = {
            "event": event,
            "address": address,
            "updated_at": now,
            "active_clients": self.active_clients,
            "client_connected": self.active_clients > 0,
            "client_addresses": sorted(self.client_addresses),
        }
        self._atomic_json(self.root / "connections.json", payload)

    def client_opened(self, remote_address: Any) -> None:
        address = self._client_address(remote_address)
        with self.lock:
            self.active_clients += 1
            self.client_addresses.add(address)
            self._publish_connections(event="connected", address=address)

    def client_closed(self, remote_address: Any) -> None:
        address = self._client_address(remote_address)
        with self.lock:
            self.active_clients = max(0, self.active_clients - 1)
            self.client_addresses.discard(address)
            self._publish_connections(event="disconnected", address=address)

    def execution_control(self) -> dict[str, Any]:
        """Read the Dashboard gate; missing, malformed, or expired means shadow."""
        path = self.root / "execution_control.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            value = {}
        if not isinstance(value, dict):
            value = {}
        now = time.time()
        requested_mode = value.get("mode") if value.get("mode") in {"shadow", "execute"} else "shadow"
        expires_at = value.get("expires_at")
        try:
            expires_at = float(expires_at) if expires_at is not None else None
        except (TypeError, ValueError):
            expires_at = None
        expired = requested_mode == "execute" and (expires_at is None or expires_at <= now)
        return {
            "mode": "shadow" if expired else requested_mode,
            "requested_mode": requested_mode,
            "revision": int(value.get("revision", 0)),
            "updated_at": value.get("updated_at"),
            "expires_at": expires_at,
            "expired": expired,
            "task_id": value.get("task_id"),
            "session_id": value.get("session_id", self.root.name),
            "server_time": now,
        }

    def publish(self, observation: dict, result: dict, elapsed_s: float) -> None:
        with self.lock:
            self.sequence += 1
            client = observation.get("client_metadata")
            if not isinstance(client, dict):
                client = {}
            images = observation.get("images", {})
            camera_shapes: dict[str, list[int]] = {}
            for camera_key in self.metadata["camera_keys"]:
                camera_shapes[camera_key] = self._atomic_image(
                    self.root / f"{camera_key}.jpg", images[camera_key]
                )
            if self.metadata["arm_mode"] == "single":
                wrist_key = next(key for key in self.metadata["camera_keys"] if "wrist" in key)
                if wrist_key != "cam_wrist":
                    self._atomic_image(self.root / "cam_wrist.jpg", images[wrist_key])
            actions = np.asarray(result.get("actions"), dtype=np.float32)
            state = np.asarray(observation.get("state"), dtype=np.float32)
            now = time.time()
            payload = {
                "sequence": self.sequence,
                "received_at": now,
                "captured_at": float(client.get("captured_at", now)),
                "source_name": str(client.get("source_name", "official-openpi-client"))[:256],
                "can_name": str(client.get("can_name", ""))[:256],
                "cam_high_device": str(client.get("cam_high_device", ""))[:256],
                "cam_wrist_device": str(client.get("cam_wrist_device", ""))[:256],
                "client_allow_execution": bool(client.get("allow_execution", False)),
                "client_execution_state": str(client.get("execution_state", "unknown"))[:64],
                "client_blocked_reason": str(client.get("blocked_reason", ""))[:500],
                "client_last_command_at": client.get("last_command_at"),
                "client_control_revision": client.get("control_revision"),
                "robot_arm_status": client.get("robot_arm_status"),
                "schema": self.metadata["schema"],
                "arm_mode": self.metadata["arm_mode"],
                "arm_side": self.metadata["arm_side"],
                "transport": "openpi_websocket_v1",
                "state": state.tolist(),
                "state_dim": int(state.shape[-1]),
                "prompt": str(observation.get("prompt", ""))[:500],
                "camera_shapes": camera_shapes,
                "cam_high_shape": camera_shapes.get("cam_high"),
                "cam_wrist_shape": next(
                    (shape for key, shape in camera_shapes.items() if "wrist" in key), None
                ) if self.metadata["arm_mode"] == "single" else None,
                "actions_shape": list(actions.shape),
                "first_action": actions[0].tolist() if actions.ndim > 1 and len(actions) else actions.tolist(),
                "action_min": float(actions.min()) if actions.size else None,
                "action_max": float(actions.max()) if actions.size else None,
                "policy_elapsed_s": elapsed_s,
                "server_timing": result.get("server_timing"),
                "policy_timing": result.get("policy_timing"),
                "execution_control": result.get("execution_control"),
            }
            self._atomic_json(self.root / "latest.json", payload)


class TelemetryWebsocketPolicyServer(websocket_policy_server.WebsocketPolicyServer):
    """Keep the official wire protocol while mirroring client lifecycle events."""

    def __init__(self, *args: Any, telemetry: PolicyTelemetry, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.telemetry = telemetry

    async def _handler(self, websocket: Any) -> None:
        self.telemetry.client_opened(websocket.remote_address)
        try:
            await super()._handler(websocket)
        finally:
            self.telemetry.client_closed(websocket.remote_address)


class TelemetryPolicy:
    def __init__(self, policy: Any, telemetry: PolicyTelemetry):
        self.policy = policy
        self.telemetry = telemetry

    def infer(self, observation: dict) -> dict:
        started = time.monotonic()
        result = dict(self.policy.infer(observation))
        result["execution_control"] = self.telemetry.execution_control()
        try:
            self.telemetry.publish(observation, result, time.monotonic() - started)
        except Exception:
            logging.exception("failed to publish policy telemetry")
        return result

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset()


def run_serve(args: argparse.Namespace) -> None:
    _install_websocket_probe_filter()
    config = build_config(args)
    policy = policy_config.create_trained_policy(
        config,
        Path(args.checkpoint).expanduser().resolve(),
        default_prompt=args.default_prompt,
    )
    if args.telemetry_dir:
        telemetry = PolicyTelemetry(Path(args.telemetry_dir).expanduser().resolve(), config.policy_metadata)
        policy = TelemetryPolicy(policy, telemetry)
        server = TelemetryWebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=config.policy_metadata,
            telemetry=telemetry,
        )
    else:
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=config.policy_metadata,
        )
    server.serve_forever()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--arm-mode", choices=("auto", "single", "bimanual"), default="auto")
    parser.add_argument("--arm-side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--schema", choices=("auto", "delivery", "joint"), default="auto")
    parser.add_argument("--dataset-layout", choices=("auto", "legacy", "canonical"), default="auto")
    parser.add_argument("--model-variant", choices=("pi05", "pi0"), default="pi05")
    parser.add_argument("--assets-base-dir", default="./assets")
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument(
        "--base-checkpoint",
        default=str(Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    norm = subparsers.add_parser("norm")
    add_common(norm)
    norm.add_argument("--batch-size", type=int, default=16)
    norm.add_argument("--num-workers", type=int, default=2)
    norm.add_argument("--max-frames", type=int, default=None)
    norm.add_argument("--test-ratio", type=float, default=0.1)
    norm.add_argument("--split-seed", type=int, default=42)

    train = subparsers.add_parser("train")
    add_common(train)
    train.add_argument("--exp-name", required=True)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--num-train-steps", type=int, default=30_000)
    train.add_argument("--save-interval", type=int, default=1_000)
    train.add_argument("--log-interval", type=int, default=100)
    train.add_argument("--fsdp-devices", type=int, default=1)
    train.add_argument("--test-ratio", type=float, default=0.1)
    train.add_argument("--split-seed", type=int, default=42)
    mode = train.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    train.add_argument("--wandb-enabled", action="store_true")

    serve = subparsers.add_parser("serve")
    add_common(serve)
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--default-prompt", default=None)
    serve.add_argument("--telemetry-dir", default=None)

    args = parser.parse_args()
    if not args.dataset_id or "/" in args.dataset_id or ".." in args.dataset_id:
        parser.error("--dataset-id must be a single safe LeRobot repository directory name")
    for name in ("batch_size", "num_workers", "num_train_steps", "save_interval", "log_interval", "fsdp_devices"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "test_ratio") and not 0.0 <= args.test_ratio < 1.0:
        parser.error("--test-ratio must be in [0, 1)")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    {"norm": run_norm, "train": run_train, "serve": run_serve}[args.command](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Single-arm Piper π0.5 training, norm-stat, and policy serving entrypoint.

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
from typing import Any, ClassVar

import numpy as np

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import data_loader
from openpi.training import weight_loaders


CONFIG_NAME = "pi05_piper_single_arm_lora"


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
class PiperSingleArmInputs(transforms.DataTransformFn):
    """Map a Piper observation and two RGB views into OpenPI inputs."""

    arm_side: str = "right"
    schema: str = "delivery"
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        state_dim = 10 if self.schema == "delivery" else 7
        if state.shape[-1] != state_dim:
            raise ValueError(f"Piper {self.schema} state must be {state_dim}D, got {state.shape}")
        images = data["images"]
        wrist_key = "cam_wrist" if self.schema == "delivery" else f"cam_{self.arm_side}_wrist"
        allowed = {"cam_high", wrist_key}
        unknown = set(images) - allowed
        if unknown:
            raise ValueError(f"unexpected camera keys: {sorted(unknown)}")
        if "cam_high" not in images or wrist_key not in images:
            raise ValueError(f"required cameras are cam_high and {wrist_key}; got {sorted(images)}")

        high = _as_hwc_uint8(images["cam_high"])
        wrist = _as_hwc_uint8(images[wrist_key])
        mapped_images = {
            "base_0_rgb": high,
            "left_wrist_0_rgb": wrist if self.arm_side == "left" else np.zeros_like(wrist),
            "right_wrist_0_rgb": wrist if self.arm_side == "right" else np.zeros_like(wrist),
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.bool_(self.arm_side == "left"),
            "right_wrist_0_rgb": np.bool_(self.arm_side == "right"),
        }
        output = {"image": mapped_images, "image_mask": image_mask, "state": state}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 7:
                raise ValueError(f"Piper single-arm actions must be 7D, got {actions.shape}")
            output["actions"] = actions
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class PiperSingleArmOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}


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
class PiperSingleArmDataConfig(training_config.DataConfigFactory):
    arm_side: str = "right"
    schema: str = "delivery"
    default_prompt: str | None = None

    def create(
        self,
        assets_dirs: Path,
        model_config,
    ) -> training_config.DataConfig:
        if self.schema == "delivery":
            repack_mapping = {
                "images": {"cam_high": "image", "cam_wrist": "wrist_image"},
                "state": "state",
                "actions": "actions",
                "prompt": "prompt",
            }
            action_sequence_keys = ("actions",)
        elif self.schema == "joint":
            wrist_key = f"cam_{self.arm_side}_wrist"
            repack_mapping = {
                "images": {
                    "cam_high": "observation.images.cam_high",
                    wrist_key: f"observation.images.{wrist_key}",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }
            action_sequence_keys = ("action",)
        else:
            raise ValueError(f"unsupported schema: {self.schema}")

        repack = transforms.Group(inputs=[transforms.RepackTransform(repack_mapping)])
        robot_transforms = transforms.Group(
            inputs=[PiperSingleArmInputs(arm_side=self.arm_side, schema=self.schema)],
            outputs=[PiperSingleArmOutputs()],
        )
        if self.schema == "joint":
            mask = transforms.make_bool_mask(6, -1)
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
    model = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    params_path = base_checkpoint / "params"
    if args.command in {"train"} and not params_path.exists():
        raise FileNotFoundError(
            f"base checkpoint params not found: {params_path}. "
            "Download gs://openpi-assets/checkpoints/pi05_base first."
        )
    data_factory = PiperSingleArmDataConfig(
        repo_id=args.dataset_id,
        arm_side=args.arm_side,
        schema=args.schema,
        base_config=training_config.DataConfig(prompt_from_task=True),
    )
    state_dim = 10 if args.schema == "delivery" else 7
    action_semantics = (
        "eef_delta_base_xyz_left_rotvec_gripper_target"
        if args.schema == "delivery"
        else "absolute_joint_position"
    )
    camera_keys = (
        ["cam_high", "cam_wrist"]
        if args.schema == "delivery"
        else ["cam_high", f"cam_{args.arm_side}_wrist"]
    )
    return training_config.TrainConfig(
        name=CONFIG_NAME,
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
            "robot_type": "piper_single_arm",
            "arm_side": args.arm_side,
            "schema": args.schema,
            "state_dim": state_dim,
            "action_dim": 7,
            "camera_keys": camera_keys,
            "action_semantics": action_semantics,
            "transport": "openpi_websocket_v1",
        },
    )


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
    config = build_config(args)
    train_main = _load_upstream_train_main(Path.cwd())
    train_main(config)


def run_norm(args: argparse.Namespace) -> None:
    config = build_config(args)
    concrete = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(concrete, config.model.action_horizon, config.model)

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
            wrist_key = "cam_wrist" if "cam_wrist" in images else f"cam_{self.metadata['arm_side']}_wrist"
            high_shape = self._atomic_image(self.root / "cam_high.jpg", images["cam_high"])
            wrist_shape = self._atomic_image(self.root / "cam_wrist.jpg", images[wrist_key])
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
                "arm_side": self.metadata["arm_side"],
                "transport": "openpi_websocket_v1",
                "state": state.tolist(),
                "state_dim": int(state.shape[-1]),
                "prompt": str(observation.get("prompt", ""))[:500],
                "cam_high_shape": high_shape,
                "cam_wrist_shape": wrist_shape,
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
    parser.add_argument("--arm-side", choices=("left", "right"), default="right")
    parser.add_argument("--schema", choices=("delivery", "joint"), default="delivery")
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

    train = subparsers.add_parser("train")
    add_common(train)
    train.add_argument("--exp-name", required=True)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--num-train-steps", type=int, default=30_000)
    train.add_argument("--save-interval", type=int, default=1_000)
    train.add_argument("--log-interval", type=int, default=100)
    train.add_argument("--fsdp-devices", type=int, default=1)
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
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    {"norm": run_norm, "train": run_train, "serve": run_serve}[args.command](args)


if __name__ == "__main__":
    main()

"""Unified, user-owned hardware identity configuration.

The file deliberately contains identities and selectors only.  It never
guesses a CAN interface, video node, or Quest device when a role is empty.
Command line values remain valid explicit overrides; GUI changes are written
back to this file immediately.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

CONFIG_ENV = "BIMANUAL_VLA_HARDWARE_CONFIG"
DEFAULT_CONFIG_PATH = Path("~/.config/bimanual-vla/hardware.json").expanduser()
SCHEMA_VERSION = 1


def config_path(path: str | os.PathLike[str] | None = None) -> Path:
    value = path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH
    return Path(value).expanduser()


def empty_config() -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "arms": {"left": {}, "right": {}},
        "cameras": {
            "overhead": {},
            "left_wrist": {},
            "right_wrist": {},
        },
        "quest": {},
        "resolver": {},
    }


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("hardware configuration must be a JSON object")
    version = value.get("version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported hardware configuration version: {version!r}")
    result = empty_config()
    for section in ("arms", "cameras", "quest", "resolver"):
        incoming = value.get(section, {})
        if not isinstance(incoming, Mapping):
            raise ValueError(f"hardware configuration section {section!r} must be an object")
        result[section] = _clean(incoming)
    for side in ("left", "right"):
        if not isinstance(result["arms"].get(side, {}), Mapping):
            raise ValueError(f"arms.{side} must be an object")
    for role in ("overhead", "left_wrist", "right_wrist"):
        if not isinstance(result["cameras"].get(role, {}), Mapping):
            raise ValueError(f"cameras.{role} must be an object")
    # Preserve extension metadata while keeping the schema version explicit.
    for key, item in value.items():
        if key not in result and key != "version":
            result[str(key)] = _clean(item)
    return result


def _legacy_migration() -> dict[str, Any] | None:
    """Read old identity files once, without importing GUI credentials."""
    cfg = empty_config()
    found = False
    old_can = Path("~/.config/bimanual-vla/can_bindings.json").expanduser()
    try:
        data = json.loads(old_can.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            for side, key in (("left", "left_can_selector"), ("right", "right_can_selector")):
                selector = str(data.get(key, "")).strip()
                if selector:
                    cfg["arms"][side]["can"] = {"selector": selector}
                    found = True
            selector = str(data.get("can_selector", "")).strip()
            if selector:
                cfg["arms"].setdefault("single", {})["can"] = {"selector": selector}
                found = True
    except (OSError, ValueError, TypeError):
        pass
    old_camera = Path("~/.config/bimanual-vla/camera_serials.json").expanduser()
    try:
        data = json.loads(old_camera.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            for old, role in (("cam_high", "overhead"), ("cam_left_wrist", "left_wrist"),
                              ("cam_right_wrist", "right_wrist")):
                serial = str(data.get(old, "")).strip()
                if serial:
                    cfg["cameras"][role]["serial"] = serial
                    cfg["cameras"][role]["selector"] = f"serial:{serial}"
                    found = True
    except (OSError, ValueError, TypeError):
        pass
    # GUI preferences historically stored reviewed paths and Quest identity
    # alongside credentials.  Copy only hardware keys; never import tokens or
    # passwords into the hardware file.
    old_prefs = Path("~/.config/bimanual-vla/collect_gui_preferences.json").expanduser()
    try:
        data = json.loads(old_prefs.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            for key, side in (("left_can_selector", "left"), ("right_can_selector", "right")):
                selector = str(data.get(key, "")).strip()
                if selector and not cfg["arms"][side].get("can"):
                    cfg["arms"][side]["can"] = {"selector": selector}
                    found = True
            for key, role in (("high_device", "overhead"), ("left_wrist_device", "left_wrist"),
                              ("right_wrist_device", "right_wrist")):
                selector = str(data.get(key, "")).strip()
                if selector and selector.lower() != "auto":
                    cfg["cameras"][role]["selector"] = selector
                    found = True
            for key, target in (("quest_model_dir", "model_dir"), ("quest_adb_serial", "adb_serial")):
                value = str(data.get(key, "")).strip()
                if value:
                    cfg["quest"][target] = value
                    found = True
            resolver = str(data.get("rlsok_resolver", "")).strip()
            if resolver:
                cfg["resolver"]["path"] = resolver
                found = True
    except (OSError, ValueError, TypeError):
        pass
    return cfg if found else None


def load_hardware_config(path: str | os.PathLike[str] | None = None, *, migrate: bool = True) -> dict[str, Any]:
    target = config_path(path)
    try:
        return validate_config(json.loads(target.read_text(encoding="utf-8")))
    except FileNotFoundError:
        migrated = _legacy_migration() if migrate else None
        if migrated is not None:
            save_hardware_config(migrated, target)
            return migrated
        return empty_config()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read hardware configuration {target}: {exc}") from exc


def save_hardware_config(value: Mapping[str, Any], path: str | os.PathLike[str] | None = None) -> Path:
    target = config_path(path)
    data = validate_config(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent), text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        temporary.replace(target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def set_value(config: Mapping[str, Any], *path: str, value: Any) -> dict[str, Any]:
    result = validate_config(config)
    cursor: dict[str, Any] = result
    for key in path[:-1]:
        item = cursor.get(key)
        if not isinstance(item, dict):
            item = {}
            cursor[key] = item
        cursor = item
    cursor[path[-1]] = value
    return result


def role_selector(config: Mapping[str, Any], role: str) -> str:
    """Return an explicit selector for a role, or an empty string."""
    item: Any
    if role in ("left", "right", "single"):
        item = config.get("arms", {}).get(role, {}).get("can", {})
    else:
        item = config.get("cameras", {}).get(role, {})
    if isinstance(item, Mapping):
        selector = str(item.get("selector") or "").strip()
        serial = str(item.get("serial") or "").strip()
        if role not in ("left", "right", "single") and serial:
            return f"serial:{serial}"
        return selector
    return str(item or "").strip()

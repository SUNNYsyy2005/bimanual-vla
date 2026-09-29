"""Small, fail-closed bridge to a project-side RLSOK device resolver.

The resolver owns fresh, read-only discovery and comparison with reviewed
roles.  This module only validates its result and passes resolved locators to
the existing camera/CAN open paths; it never opens a device itself.
Wire contract: docs/rlsok/DEVICE_RESOLVER_GUI_GLUE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import uuid


RESOLVER_ENV = "BIMANUAL_VLA_RLSOK_RESOLVER"
_MAX_INVENTORY_AGE_S = 30.0
_CAN_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
_CAMERA_PATH = re.compile(r"/dev/(?:video[0-9]+|v4l/(?:by-id|by-path)/[^/]+)\Z")


class DeviceCheckRejected(RuntimeError):
    """The requested role mapping was not approved by the fresh resolver."""


@dataclass(frozen=True)
class ResolvedDevices:
    arms: dict[str, str]
    cameras: dict[str, str]
    observed_at: str


def configured_resolver(value: str | None = None) -> str | None:
    """An empty setting preserves the existing workflow until glue is installed."""
    selected = value if value is not None else os.environ.get(RESOLVER_ENV, "")
    return selected.strip() or None


def _role_names(arm_mode: str, arm_side: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if arm_mode == "bimanual" and arm_side == "both":
        return ("left", "right"), ("overhead", "left_wrist", "right_wrist")
    if arm_mode == "single" and arm_side in {"left", "right"}:
        return (arm_side,), ("overhead", f"{arm_side}_wrist")
    raise DeviceCheckRejected("RLSOK: invalid arm mode/side")


def check_devices(
    *,
    purpose: str,
    arm_mode: str,
    arm_side: str,
    arms: dict[str, str],
    cameras: dict[str, str],
    resolver: str | None = None,
) -> ResolvedDevices | None:
    """Ask the configured glue for a fresh role resolution before device open.

    The executable receives one JSON request on stdin and returns one JSON
    response on stdout. No shell is involved. No raw serials enter GUI state.
    """
    executable = configured_resolver(resolver)
    if executable is None:
        return None
    if purpose not in {"collection", "deployment"}:
        raise DeviceCheckRejected("RLSOK: invalid check purpose")
    arm_roles, camera_roles = _role_names(arm_mode, arm_side)
    if set(arms) != set(arm_roles) or set(cameras) != set(camera_roles):
        raise DeviceCheckRejected("RLSOK: requested device roles are incomplete")
    request = {
        "schema_version": 1,
        "request_id": uuid.uuid4().hex,
        "purpose": purpose,
        "arm_mode": arm_mode,
        "arm_side": arm_side,
        "requested": {"arms": arms, "cameras": cameras},
    }
    started_at = datetime.now(timezone.utc)
    try:
        result = subprocess.run(
            [executable],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=8.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeviceCheckRejected(f"RLSOK resolver unavailable: {type(exc).__name__}") from exc
    if result.returncode != 0:
        # Resolver stderr can contain unredacted hardware identifiers.
        raise DeviceCheckRejected(f"RLSOK resolver failed (exit {result.returncode})")
    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ValueError("invalid schema")
        if payload.get("rlsok_version") != "1.5.8":
            raise ValueError("unsupported RLSOK version")
        if payload.get("request_id") != request["request_id"]:
            raise ValueError("response does not match request")
        if payload.get("decision") == "deny":
            if set(payload) != {"schema_version", "rlsok_version", "request_id", "decision", "reason_code"}:
                raise ValueError("unexpected denial fields")
            code = payload.get("reason_code")
            if not isinstance(code, str) or not re.fullmatch(r"[A-Z0-9_]{1,64}", code):
                code = "DEVICE_CHECK_REJECTED"
            raise DeviceCheckRejected(f"RLSOK rejected device check: {code}")
        if payload.get("decision") != "allow":
            raise ValueError("missing allow decision")
        if set(payload) != {
            "schema_version", "rlsok_version", "request_id", "decision",
            "inventory_method", "observed_at", "resolved",
        }:
            raise ValueError("unexpected allow fields")
        if payload.get("inventory_method") != "linux-sysfs-udev":
            raise ValueError("fresh Linux inventory required")
        timestamp = payload["observed_at"]
        observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            raise ValueError("timezone required")
        age = (datetime.now(timezone.utc) - observed).total_seconds()
        if not 0 <= age <= _MAX_INVENTORY_AGE_S or (observed - started_at).total_seconds() < -2.0:
            raise ValueError("stale or future inventory")
        resolved = payload["resolved"]
        if not isinstance(resolved, dict) or set(resolved) != {"arms", "cameras"}:
            raise ValueError("invalid resolved fields")
        resolved_arms = resolved["arms"]
        resolved_cameras = resolved["cameras"]
        if set(resolved_arms) != set(arm_roles) or set(resolved_cameras) != set(camera_roles):
            raise ValueError("resolved roles do not match request")
        for role, value in resolved_arms.items():
            if not isinstance(value, str) or not _CAN_NAME.fullmatch(value):
                raise ValueError(f"invalid CAN locator for {role}")
            requested_name = arms[role].strip()
            if requested_name.lower() != "auto" and value != requested_name:
                raise ValueError(f"explicit CAN selector changed for {role}")
        for role, value in resolved_cameras.items():
            if not isinstance(value, str) or not _CAMERA_PATH.fullmatch(value):
                raise ValueError(f"invalid camera locator for {role}")
        if len(set(resolved_arms.values())) != len(resolved_arms):
            raise ValueError("duplicate CAN locators")
        if len(set(resolved_cameras.values())) != len(resolved_cameras):
            raise ValueError("duplicate camera locators")
    except DeviceCheckRejected:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise DeviceCheckRejected("RLSOK resolver returned invalid or stale evidence") from exc
    return ResolvedDevices(resolved_arms, resolved_cameras, timestamp)


def camera_role_request(
    arm_mode: str,
    arm_side: str,
    *,
    overhead: str,
    wrist: str,
    left_wrist: str,
    right_wrist: str,
) -> dict[str, str]:
    if arm_mode == "bimanual":
        return {"overhead": overhead, "left_wrist": left_wrist, "right_wrist": right_wrist}
    return {"overhead": overhead, f"{arm_side}_wrist": wrist}


def arm_role_request(
    arm_mode: str, arm_side: str, *, single: str, left: str, right: str
) -> dict[str, str]:
    if arm_mode == "bimanual":
        return {"left": left, "right": right}
    return {arm_side: single}

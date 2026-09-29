"""Offline test fixture for the project-side resolver wire contract.

This file never discovers devices or invokes RLSOK. The test suite copies it
to a temporary executable using the active Conda Python interpreter.
"""

from datetime import datetime, timedelta, timezone
import json
import os
import sys


request = json.load(sys.stdin)
mode = os.environ.get("BIMANUAL_VLA_TEST_RLSOK_MODE", "UNCHANGED")
reply = {
    "schema_version": 1,
    "rlsok_version": "1.5.8",
    "request_id": request["request_id"],
    "review_decision": mode if mode in {"NEEDS_MATERIAL", "REVIEW_REQUIRED"} else "UNCHANGED",
    "review_scope": "saved-configuration-only",
    "hardware_dispatch": False,
}
if mode in {"NEEDS_MATERIAL", "REVIEW_REQUIRED"}:
    reply["reason_code"] = (
        "MISSING_CAMERA" if mode == "NEEDS_MATERIAL" else "ROLE_CHANGED"
    )
else:
    camera_locators = {
        "overhead": "/dev/video4",
        "left_wrist": "/dev/video6",
        "right_wrist": "/dev/video8",
    }
    if mode == "ENDPOINT_CHANGED":
        camera_locators["overhead"] = "/dev/video10"
    observed = datetime.now(timezone.utc)
    if mode == "STALE":
        observed -= timedelta(minutes=2)
    reply.update({
        "inventory_method": "linux-sysfs-udev",
        "observed_at": observed.isoformat(),
        "resolved": {
            "arms": {
                role: ("can0" if role == "left" else "can1")
                for role in request["requested"]["arms"]
            },
            "cameras": {
                role: camera_locators[role]
                for role in request["requested"]["cameras"]
            },
        },
    })
    if mode == "BAD_NONCE":
        reply["request_id"] = "wrong-request"

json.dump(reply, sys.stdout)
sys.stdout.write("\n")

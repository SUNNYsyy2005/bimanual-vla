#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/user/miniconda3/envs/dual_arm/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

if [[ ! -d /sys/class/net/can0 ]]; then
    printf '%s\n' \
        "WARNING: SocketCAN interface can0 is not available." \
        "The GUI can still open, but robot connection and data collection will fail until the USB-CAN adapter is connected and can0 is activated." \
        "Activation helper: /home/user/dual_ARM_project/piper_sdk/piper_sdk/can_activate.sh" \
        >&2
fi

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" collect_gui.py "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/user/miniconda3/envs/dual_arm/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

PIPER_CAN_HELPER="/home/user/dual_ARM_project/piper_sdk/piper_sdk/can_activate.sh"

for can_name in can0 can1; do
    if [[ ! -d "/sys/class/net/$can_name" ]]; then
        printf '%s\n' \
            "WARNING: SocketCAN interface $can_name is not available." \
            "Bimanual collection needs two independent gs_usb adapters: can0 for the left arm and can1 for the right arm." \
            "Check 'lsusb -t' and 'journalctl -k -b | grep -E \"gs_usb|USB disconnect\"'." \
            "Activation helper: $PIPER_CAN_HELPER" \
            >&2
        continue
    fi

    flags="$(<"/sys/class/net/$can_name/flags")"
    if (( (flags & 0x1) == 0 )); then
        printf '%s\n' \
            "WARNING: SocketCAN interface $can_name exists but is DOWN." \
            "Activate it at 1000000 bit/s before connecting the GUI." \
            >&2
    fi
done

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" collect_gui.py "$@"

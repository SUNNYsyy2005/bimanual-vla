#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPER_CAN_HELPER="${BIMANUAL_VLA_CAN_ACTIVATE_SCRIPT:-$SCRIPT_DIR/piper_sdk/piper_sdk/can_activate.sh}"

for can_name in can0 can1; do
    if [[ ! -d "/sys/class/net/$can_name" ]]; then
        printf '%s\n' \
            "WARNING: SocketCAN interface $can_name is not available." \
            "Default bimanual settings use can0/can1; a reviewed RLSOK role mapping may select other interfaces." \
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
# RLSOK resolution is checked when Connect devices / Start inference is clicked,
# after GUI role selection and before either CAN or camera handles are opened.
# Checkpoint details: docs/rlsok/START_GUI_PREFLIGHT_GLUE.md.

export BIMANUAL_VLA_CAN_ACTIVATE_SCRIPT="$PIPER_CAN_HELPER"
exec "$SCRIPT_DIR/bin/bimanual-vla" collect-gui "$@"

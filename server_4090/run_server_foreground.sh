#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.json}"
PYTHON_DEFAULT="${BIMANUAL_VLA_BOOTSTRAP_PYTHON:-/home/sunny/miniconda3/envs/openpi/bin/python}"
TOKEN_DIR="${HOME}/.config/bimanual-vla"
TOKEN_FILE="${TOKEN_DIR}/server.env"
STATE_DIR="${HOME}/.local/share/bimanual-vla-server"
PID_FILE="${STATE_DIR}/dashboard.pid"
LOG_FILE="${STATE_DIR}/dashboard.log"

umask 077
mkdir -p "$TOKEN_DIR" "$STATE_DIR"
chmod 700 "$TOKEN_DIR" "$STATE_DIR"

if [[ ! -x "$PYTHON_DEFAULT" ]]; then
  echo "Bootstrap Python is not executable: $PYTHON_DEFAULT" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  cp "$SCRIPT_DIR/config.example.json" "$CONFIG"
fi

if [[ ! -f "$TOKEN_FILE" ]]; then
  TOKEN="$($PYTHON_DEFAULT -c 'import secrets; print(secrets.token_urlsafe(36))')"
  LOGIN_USER="${BIMANUAL_VLA_LOGIN_USER:-${USER:-sunny}}"
  LOGIN_PASSWORD="${BIMANUAL_VLA_LOGIN_PASSWORD:-$($PYTHON_DEFAULT -c 'import secrets; print(secrets.token_urlsafe(24))')}"
  {
    printf 'export BIMANUAL_VLA_SERVER_TOKEN=%q\n' "$TOKEN"
    printf 'export BIMANUAL_VLA_LOGIN_USER=%q\n' "$LOGIN_USER"
    printf 'export BIMANUAL_VLA_LOGIN_PASSWORD=%q\n' "$LOGIN_PASSWORD"
  } > "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"

# server.env intentionally contains shell-escaped export statements rather than
# systemd EnvironmentFile syntax, so source it in this small foreground wrapper.
# shellcheck disable=SC1090
source "$TOKEN_FILE"

# Migrate credentials created by older versions of start_server.sh.
if [[ -z "${BIMANUAL_VLA_SERVER_TOKEN:-}" ]]; then
  echo "BIMANUAL_VLA_SERVER_TOKEN is missing from $TOKEN_FILE" >&2
  exit 1
fi
if [[ -z "${BIMANUAL_VLA_LOGIN_USER:-}" ]]; then
  export BIMANUAL_VLA_LOGIN_USER="${USER:-sunny}"
  printf 'export BIMANUAL_VLA_LOGIN_USER=%q\n' "$BIMANUAL_VLA_LOGIN_USER" >> "$TOKEN_FILE"
fi
if [[ -z "${BIMANUAL_VLA_LOGIN_PASSWORD:-}" ]]; then
  export BIMANUAL_VLA_LOGIN_PASSWORD="$($PYTHON_DEFAULT -c 'import secrets; print(secrets.token_urlsafe(24))')"
  printf 'export BIMANUAL_VLA_LOGIN_PASSWORD=%q\n' "$BIMANUAL_VLA_LOGIN_PASSWORD" >> "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"

PYTHON="$($PYTHON_DEFAULT - "$CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["openpi_python"])
PY
)"
if [[ ! -x "$PYTHON" ]]; then
  echo "Configured OpenPI Python is not executable: $PYTHON" >&2
  exit 1
fi

# Keep compatibility with the manual stop script. systemd removes this file via
# ExecStopPost after a clean stop or a failed start.
echo "$$" > "$PID_FILE"

# Keep the established dashboard.log location while the Python process remains
# in the foreground for systemd supervision.
exec "$PYTHON" "$SCRIPT_DIR/app.py" --config "$CONFIG" >> "$LOG_FILE" 2>&1

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.json}"
PYTHON_DEFAULT="/home/sunny/miniconda3/envs/openpi/bin/python"
TOKEN_DIR="${HOME}/.config/bimanual-vla"
TOKEN_FILE="${TOKEN_DIR}/server.env"
STATE_DIR="${HOME}/.local/share/bimanual-vla-server"
PID_FILE="${STATE_DIR}/dashboard.pid"
LOG_FILE="${STATE_DIR}/dashboard.log"

if [[ ! -f "$CONFIG" ]]; then
  cp "$SCRIPT_DIR/config.example.json" "$CONFIG"
  echo "Created $CONFIG"
fi
mkdir -p "$TOKEN_DIR" "$STATE_DIR"
chmod 700 "$TOKEN_DIR"
if [[ ! -f "$TOKEN_FILE" ]]; then
  TOKEN="$($PYTHON_DEFAULT -c 'import secrets; print(secrets.token_urlsafe(36))')"
  LOGIN_USER="${BIMANUAL_VLA_LOGIN_USER:-${USER:-sunny}}"
  LOGIN_PASSWORD="${BIMANUAL_VLA_LOGIN_PASSWORD:-$($PYTHON_DEFAULT -c 'import secrets; print(secrets.token_urlsafe(24))')}"
  {
    printf 'export BIMANUAL_VLA_SERVER_TOKEN=%q\n' "$TOKEN"
    printf 'export BIMANUAL_VLA_LOGIN_USER=%q\n' "$LOGIN_USER"
    printf 'export BIMANUAL_VLA_LOGIN_PASSWORD=%q\n' "$LOGIN_PASSWORD"
  } > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi
# shellcheck disable=SC1090
source "$TOKEN_FILE"

# Migrate an existing server.env created before username/password login was
# added.  Keep the credentials in the same 0600 file as the API token so the
# dashboard can be restarted without changing the login pair.
if [[ -z "${BIMANUAL_VLA_LOGIN_USER:-}" ]]; then
  export BIMANUAL_VLA_LOGIN_USER="${USER:-sunny}"
  printf 'export BIMANUAL_VLA_LOGIN_USER=%q\n' "$BIMANUAL_VLA_LOGIN_USER" >> "$TOKEN_FILE"
fi
if [[ -z "${BIMANUAL_VLA_LOGIN_PASSWORD:-}" ]]; then
  export BIMANUAL_VLA_LOGIN_PASSWORD="$($PYTHON_DEFAULT -c 'import secrets; print(secrets.token_urlsafe(24))')"
  printf 'export BIMANUAL_VLA_LOGIN_PASSWORD=%q\n' "$BIMANUAL_VLA_LOGIN_PASSWORD" >> "$TOKEN_FILE"
fi

PYTHON="$($PYTHON_DEFAULT - "$CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["openpi_python"])
PY
)"
PORT="$($PYTHON_DEFAULT - "$CONFIG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("port", 8090))
PY
)"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null && tr '\0' ' ' < "/proc/$PID/cmdline" | grep -q 'server_4090/app.py'; then
    echo "Dashboard already running: PID=$PID"
    echo "URL: http://192.168.101.9:${PORT}"
    echo "Login user: ${BIMANUAL_VLA_LOGIN_USER}"
    echo "Login password: ${BIMANUAL_VLA_LOGIN_PASSWORD}"
    echo "Token: ${BIMANUAL_VLA_SERVER_TOKEN}"
    exit 0
  fi
fi

nohup "$PYTHON" "$SCRIPT_DIR/app.py" --config "$CONFIG" >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Dashboard failed to start. Last log lines:" >&2
  tail -n 80 "$LOG_FILE" >&2
  exit 1
fi

echo "Dashboard started: PID=$PID"
echo "URL: http://192.168.101.9:${PORT}"
echo "Login user: ${BIMANUAL_VLA_LOGIN_USER}"
echo "Login password: ${BIMANUAL_VLA_LOGIN_PASSWORD}"
echo "Token: ${BIMANUAL_VLA_SERVER_TOKEN}"
echo "Log: $LOG_FILE"

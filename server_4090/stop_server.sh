#!/usr/bin/env bash
set -euo pipefail
STATE_DIR="${HOME}/.local/share/bimanual-vla-server"
PID_FILE="${STATE_DIR}/dashboard.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "Dashboard PID file does not exist."
  exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  CMD="$(tr '\0' ' ' < "/proc/$PID/cmdline")"
  if [[ "$CMD" != *"server_4090/app.py"* ]]; then
    echo "Refusing to stop PID $PID because it is not the dashboard: $CMD" >&2
    exit 1
  fi
  kill "$PID"
  echo "Stop requested for dashboard PID $PID"
  for _ in $(seq 1 50); do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$PID" 2>/dev/null; then
    CMD="$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || true)"
    if [[ "$CMD" == *"server_4090/app.py"* ]]; then
      echo "Dashboard did not stop in 5 seconds; sending SIGKILL to PID $PID" >&2
      kill -KILL "$PID"
    fi
  fi
else
  echo "Dashboard PID $PID is not running."
fi
rm -f "$PID_FILE"

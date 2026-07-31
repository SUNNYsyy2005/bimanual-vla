#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-4x4090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/sunny/bimanual-vla}"
LOCAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/server_4090/templates'"
rsync -av --relative \
  "$LOCAL_ROOT/./server_4090/app.py" \
  "$LOCAL_ROOT/./server_4090/dataset_editor.py" \
  "$LOCAL_ROOT/./server_4090/openpi_single_arm.py" \
  "$LOCAL_ROOT/./server_4090/validate_lerobot.py" \
  "$LOCAL_ROOT/./server_4090/config.example.json" \
  "$LOCAL_ROOT/./server_4090/start_server.sh" \
  "$LOCAL_ROOT/./server_4090/stop_server.sh" \
  "$LOCAL_ROOT/./server_4090/templates/index.html" \
  "$LOCAL_ROOT/./server_4090/README.md" \
  "$LOCAL_ROOT/./check_pi05_dataset.py" \
  "$LOCAL_ROOT/./download_openpi_checkpoint.py" \
  "$LOCAL_ROOT/./robot_observation_bridge.py" \
  "$LOCAL_ROOT/./piper_data_contract.py" \
  "$LOCAL_ROOT/./camera.py" \
  "$LOCAL_ROOT/./pi0_dataset.py" \
  "$REMOTE_HOST:$REMOTE_ROOT/"

ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && if [[ ! -f server_4090/config.json ]]; then cp server_4090/config.example.json server_4090/config.json; fi && bash server_4090/stop_server.sh && bash server_4090/start_server.sh"

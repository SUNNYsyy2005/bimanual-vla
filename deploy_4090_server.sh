#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-4x4090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/sunny/bimanual-vla}"
LOCAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/server_4090/templates'"
rsync -av --relative \
  "$LOCAL_ROOT/./server_4090/app.py" \
  "$LOCAL_ROOT/./server_4090/dataset_editor.py" \
  "$LOCAL_ROOT/./server_4090/episode_split.py" \
  "$LOCAL_ROOT/./server_4090/openpi_single_arm.py" \
  "$LOCAL_ROOT/./server_4090/eval_heldout_loss.py" \
  "$LOCAL_ROOT/./server_4090/slurm_job_runner.py" \
  "$LOCAL_ROOT/./server_4090/dataset_transfer_runner.py" \
  "$LOCAL_ROOT/./server_4090/video_transfer_runner.py" \
  "$LOCAL_ROOT/./server_4090/validate_lerobot.py" \
  "$LOCAL_ROOT/./server_4090/config.example.json" \
  "$LOCAL_ROOT/./server_4090/config.simulation.example.json" \
  "$LOCAL_ROOT/./server_4090/run_server_foreground.sh" \
  "$LOCAL_ROOT/./server_4090/bimanual-vla-dashboard.service" \
  "$LOCAL_ROOT/./server_4090/bimanual-vla-sim-dashboard.service" \
  "$LOCAL_ROOT/./server_4090/start_server.sh" \
  "$LOCAL_ROOT/./server_4090/stop_server.sh" \
  "$LOCAL_ROOT/./server_4090/templates/index.html" \
  "$LOCAL_ROOT/./server_4090/README.md" \
  "$LOCAL_ROOT/./server_4090/SIMULATION_DASHBOARD.md" \
  "$LOCAL_ROOT/./check_pi05_dataset.py" \
  "$LOCAL_ROOT/./download_openpi_checkpoint.py" \
  "$LOCAL_ROOT/./upload_dataset_4090.py" \
  "$LOCAL_ROOT/./robot_observation_bridge.py" \
  "$LOCAL_ROOT/./piper_action_conventions.py" \
  "$LOCAL_ROOT/./piper_data_contract.py" \
  "$LOCAL_ROOT/./camera.py" \
  "$LOCAL_ROOT/./pi0_dataset.py" \
  "$LOCAL_ROOT/./scripts/query_h100_h200_resources.sh" \
  "$REMOTE_HOST:$REMOTE_ROOT/"

ssh "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' bash -s" <<'REMOTE'
set -euo pipefail

cd "$REMOTE_ROOT"
if [[ ! -f server_4090/config.json ]]; then
  cp server_4090/config.example.json server_4090/config.json
fi

mkdir -p "$HOME/.config/systemd/user" "$HOME/.local/share/bimanual-vla-server"
install -m 0644 \
  server_4090/bimanual-vla-dashboard.service \
  "$HOME/.config/systemd/user/bimanual-vla-dashboard.service"
chmod +x server_4090/slurm_job_runner.py server_4090/dataset_transfer_runner.py server_4090/video_transfer_runner.py server_4090/run_server_foreground.sh scripts/query_h100_h200_resources.sh

systemctl --user daemon-reload
systemctl --user stop bimanual-vla-dashboard.service 2>/dev/null || true
# Stop a legacy nohup-managed Dashboard before systemd binds the same port.
bash server_4090/stop_server.sh || true
systemctl --user enable --now bimanual-vla-dashboard.service

# A user service starts at login by default. Linger makes the user manager (and
# therefore the Dashboard) start during boot even when nobody logs in.
if ! loginctl show-user "$(id -un)" -p Linger --value | grep -qx yes; then
  if ! timeout 10 loginctl enable-linger "$(id -un)"; then
    echo "WARNING: could not enable linger without administrator authorization." >&2
    echo "The service is enabled, but a login may be required after reboot." >&2
  fi
fi

systemctl --user --no-pager --full status bimanual-vla-dashboard.service
REMOTE

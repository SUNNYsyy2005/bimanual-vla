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
  "$LOCAL_ROOT/./server_4090/slurm_dataset_sync_runner.py" \
  "$LOCAL_ROOT/./server_4090/video_transfer_runner.py" \
  "$LOCAL_ROOT/./server_4090/validate_lerobot.py" \
  "$LOCAL_ROOT/./server_4090/config.simulation.example.json" \
  "$LOCAL_ROOT/./server_4090/run_server_foreground.sh" \
  "$LOCAL_ROOT/./server_4090/bimanual-vla-sim-dashboard.service" \
  "$LOCAL_ROOT/./server_4090/templates/index.html" \
  "$LOCAL_ROOT/./server_4090/README.md" \
  "$LOCAL_ROOT/./server_4090/SIMULATION_DASHBOARD.md" \
  "$LOCAL_ROOT/./SERVER_PATHS_ENV_TRAIN_EVAL.md" \
  "$LOCAL_ROOT/./check_pi05_dataset.py" \
  "$LOCAL_ROOT/./download_openpi_checkpoint.py" \
  "$LOCAL_ROOT/./upload_dataset_4090.py" \
  "$LOCAL_ROOT/./piper_action_conventions.py" \
  "$LOCAL_ROOT/./piper_data_contract.py" \
  "$LOCAL_ROOT/./camera.py" \
  "$LOCAL_ROOT/./pi0_dataset.py" \
  "$LOCAL_ROOT/./scripts/query_h100_h200_resources.sh" \
  "$REMOTE_HOST:$REMOTE_ROOT/"

ssh "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' bash -s" <<'REMOTE'
set -euo pipefail
cd "$REMOTE_ROOT"
if [[ ! -f server_4090/config.simulation.json ]]; then
  cp server_4090/config.simulation.example.json server_4090/config.simulation.json
else
  python3 - <<'PY'
import json
from pathlib import Path
example = json.loads(Path('server_4090/config.simulation.example.json').read_text())
path = Path('server_4090/config.simulation.json')
current = json.loads(path.read_text())
for key in (
    'dataset_root', 'assets_base_dir', 'checkpoint_base_dir', 'base_checkpoint',
    'checkpoint_allowed_roots', 'eval_video_roots', 'cluster_targets',
    'transfer_parallelism', 'cluster_resources_script', 'nas_dataset_staging_root',
):
    if key in example:
        current[key] = example[key]
path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n')
PY
fi
mkdir -p \
  "$HOME/.config/systemd/user" \
  "$HOME/.config/bimanual-vla-sim-dashboard" \
  "$HOME/.local/share/bimanual-vla-sim-dashboard/eval_videos" \
  "$HOME/.local/share/bimanual-vla-sim-dashboard/cluster_inventory"
install -m 0644 \
  server_4090/bimanual-vla-sim-dashboard.service \
  "$HOME/.config/systemd/user/bimanual-vla-sim-dashboard.service"
chmod +x server_4090/slurm_job_runner.py server_4090/dataset_transfer_runner.py server_4090/slurm_dataset_sync_runner.py server_4090/video_transfer_runner.py server_4090/run_server_foreground.sh scripts/query_h100_h200_resources.sh
# Best-effort staging for H100/login-server Slurm helpers. H200 remains
# independent and should be prepared via its dedicated setup Slurm jobs.
if command -v rsync >/dev/null 2>&1; then
  timeout 20 ssh -n -o BatchMode=yes -o ConnectTimeout=8 login-server 'mkdir -p /DATA/disk0/sunny/bimanual-vla /DATA/NAS/GPUServer/sunny/dashboard_dataset_sync' 2>/dev/null && \
  timeout 60 rsync -az --delete \
    server_4090 piper_action_conventions.py piper_data_contract.py pi0_dataset.py check_pi05_dataset.py download_openpi_checkpoint.py \
    login-server:/DATA/disk0/sunny/bimanual-vla/ 2>/dev/null || true
fi
# Best-effort mirror of H200 Slurm inventory caches onto 4x4090 so the UI does
# not block on SSH to login-server on every refresh.
for node in h200-ali-01 h200-ali-02; do
  timeout 20 ssh -n -o BatchMode=yes -o ConnectTimeout=8 login-server "test -s /DATA/NAS/GPUServer/sunny/dashboard_probe/${node}_inventory.json && cat /DATA/NAS/GPUServer/sunny/dashboard_probe/${node}_inventory.json" \
    > "$HOME/.local/share/bimanual-vla-sim-dashboard/cluster_inventory/${node}_inventory.json" 2>/dev/null || true
done
systemctl --user daemon-reload
systemctl --user stop bimanual-vla-sim-dashboard.service 2>/dev/null || true
systemctl --user enable --now bimanual-vla-sim-dashboard.service
if ! loginctl show-user "$(id -un)" -p Linger --value | grep -qx yes; then
  timeout 10 loginctl enable-linger "$(id -un)" || true
fi
systemctl --user --no-pager --full status bimanual-vla-sim-dashboard.service
REMOTE

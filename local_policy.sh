#!/usr/bin/env bash
# Default: start the local Piper Pi05 policy in the background.
set -euo pipefail
pi05_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /home/user/miniconda3/envs/pi05_5060/bin/python \
  "$pi05_project_root/scripts/pi05_local/manage_server.py" "$@"

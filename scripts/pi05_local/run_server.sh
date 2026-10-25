#!/usr/bin/env bash
set -euo pipefail
pi05_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$pi05_project_root"
exec /home/user/miniconda3/envs/pi05_5060/bin/python -u scripts/pi05_local/serve.py "$@"

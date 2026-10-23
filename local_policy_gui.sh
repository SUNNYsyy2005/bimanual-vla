#!/usr/bin/env bash
set -euo pipefail
pi05_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$pi05_project_root/local_policy.sh" gui

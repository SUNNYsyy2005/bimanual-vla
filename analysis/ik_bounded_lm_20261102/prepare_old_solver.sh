#!/usr/bin/env bash
# Extract the pre-replacement (minimum-motion) IK solver from commit e324d7d
# into ./old so the replays can compare it with the current solver.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/old"
git -C "$here/../.." show e324d7d:bimanual_vla/deployment/client.py \
  | sed 's/from bimanual_vla.deployment.ik_continuity import/from old_ik_continuity import/' > "$here/old/old_client.py"
git -C "$here/../.." show e324d7d:bimanual_vla/deployment/ik_continuity.py > "$here/old/old_ik_continuity.py"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_PYTHON="${BIMANUAL_VLA_PYTHON:-/home/user/miniconda3/envs/dual_arm/bin/python}"
if [[ ! -x "$TEST_PYTHON" ]]; then
    printf 'Conda Python is unavailable: %s\n' "$TEST_PYTHON" >&2
    exit 2
fi

cd "$PROJECT_DIR"
exec "$TEST_PYTHON" -m unittest -v \
    tests.test_rlsok_offline_integration \
    tests.test_rlsok_device_guard \
    tests.test_deployment_recording

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/run/run_alfworld_ray.py" \
  --config "$PROJECT_ROOT/configs/rl_alf_config.yaml" \
  --ray-config "$PROJECT_ROOT/configs/rl_alf_ray.yml" \
  "$@"

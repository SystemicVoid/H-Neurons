#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU_VENV="${GPU_VENV:-$PROJECT_DIR/.venv-gpu}"
GPU_PYTHONPATH_DEFAULT="$GPU_VENV/lib/python3.10/site-packages"

if [[ -x "$GPU_VENV/bin/python" ]]; then
  export PYTHONPATH="${GPU_PYTHONPATH:-$GPU_PYTHONPATH_DEFAULT}"
  exec "$GPU_VENV/bin/python" "$@"
fi

exec uv run python3 "$@"

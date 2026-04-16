#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="Gold expansion annotation (sequential Codex workers)" \
        -- bash "$0" "$@"
fi

cd "${PROJECT_DIR}"

mkdir -p logs
LOG="logs/annotate_gold_expansion_$(date +%Y%m%d_%H%M%S).log"

echo "Gold expansion annotation log: ${LOG}"
PYTHONUNBUFFERED=1 uv run python scripts/annotate_gold_expansion.py "$@" \
    2>&1 | tee -a "${LOG}"

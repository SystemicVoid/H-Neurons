#!/usr/bin/env bash
# Jailbreak long-budget audit: 1024 tokens, α={0.0, 1.0, 2.0, 3.0}
# Fixes the truncation bias from the 256-token legacy run.
set -euo pipefail

RUN_DIR="data/gemma3_4b/intervention/jailbreak/experiment"

echo "=== Jailbreak 1024-token audit ==="
echo "Alpha grid: 0.0 1.0 2.0 3.0"
echo "Output: ${RUN_DIR}"

PYTHONUNBUFFERED=1 systemd-inhibit --what=idle --why="Jailbreak 1024-tok audit" \
  uv run python scripts/run_intervention.py \
    --benchmark jailbreak \
    --alphas 0.0 1.0 2.0 3.0 \
    --max_new_tokens 1024 \
    --output_dir "${RUN_DIR}" \
    --wandb \
  2>&1 | tee logs/jailbreak_1024tok_intervention.log

echo ""
echo "=== Evaluating jailbreak responses ==="
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
  --input_dir "${RUN_DIR}" \
  2>&1 | tee logs/jailbreak_1024tok_evaluate.log

echo ""
echo "=== Exporting site data ==="
uv run python scripts/export_site_data.py \
  2>&1 | tee logs/jailbreak_1024tok_export.log

echo ""
echo "=== Chain complete ==="

#!/usr/bin/env bash
set -euo pipefail

SEED=42
OUTPUT_DIR="data/gemma3_4b/intervention/jailbreak/experiment"

echo "=== Canonical Jailbreak Rerun: 5000 tokens, sampled, seed=${SEED} ==="
echo "Output: ${OUTPUT_DIR}"
echo "Alphas: 0.0 1.5 3.0 | Samples: 500 | max_new_tokens: 5000"

PYTHONUNBUFFERED=1 systemd-inhibit --what=idle:sleep --who="jailbreak-5000tok" --why="GPU intervention run" \
    uv run python scripts/run_intervention.py \
    --benchmark jailbreak \
    --alphas 0.0 1.5 3.0 \
    --max_new_tokens 5000 \
    --seed ${SEED} \
    --output_dir "${OUTPUT_DIR}" \
    --wandb \
    2>&1 | tee logs/jailbreak_5000tok_canonical.log

echo ""
echo "=== Intervention complete. Running evaluation... ==="

PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --input_dir "${OUTPUT_DIR}" \
    2>&1 | tee logs/jailbreak_5000tok_evaluate.log

echo ""
echo "=== Evaluation complete. Exporting site data... ==="

uv run python scripts/export_site_data.py 2>&1 | tee logs/export.log

echo ""
echo "=== Pipeline complete ==="

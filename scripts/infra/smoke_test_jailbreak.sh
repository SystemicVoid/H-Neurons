#!/usr/bin/env bash
# Smoke test for jailbreak 1024-token rerun.
# Runs 10 samples across canonical alphas into a temp directory.
# Does NOT touch existing data in experiment/.
set -euo pipefail

SMOKE_DIR="data/gemma3_4b/intervention/jailbreak/experiment_smoke_test"

# Safety: refuse to run if output dir already has data
if [ -d "$SMOKE_DIR" ] && [ "$(find "$SMOKE_DIR" -name '*.jsonl' 2>/dev/null | head -1)" ]; then
    echo "ERROR: $SMOKE_DIR already contains JSONL data."
    echo "Remove it first if you want to rerun: rm -rf $SMOKE_DIR"
    exit 1
fi

mkdir -p "$SMOKE_DIR"

echo "=== Smoke Test: Jailbreak 1024-token preflight ==="
echo "Output: $SMOKE_DIR"
echo "Samples: 10 (× 4 alphas)"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --benchmark jailbreak \
    --alphas 0.0 1.0 2.0 3.0 \
    --max_new_tokens 1024 \
    --max_samples 10 \
    --output_dir "$SMOKE_DIR" \
    2>&1 | tee logs/jailbreak_smoke_test.log

echo ""
echo "=== Smoke test complete. Analysing results... ==="
echo ""

uv run python scripts/analyse_smoke_test.py --input_dir "$SMOKE_DIR"

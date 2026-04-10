#!/usr/bin/env bash
# Evaluation-only pipeline for alpha=1.0 baseline (inference already done).
# Binary judge (batch API) → CSV-v2 eval (batch API) → analysis
set -euo pipefail

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

BINARY_JUDGE_LOG="$LOG_DIR/jailbreak_alpha1_binary_judge_${TIMESTAMP}.log"
CSV2_EVAL_LOG="$LOG_DIR/jailbreak_alpha1_csv2_eval_${TIMESTAMP}.log"
CSV2_ANALYSIS_LOG="$LOG_DIR/jailbreak_alpha1_csv2_analysis_${TIMESTAMP}.log"

EXPERIMENT_DIR="data/gemma3_4b/intervention/jailbreak/experiment"
CSV2_DIR="data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
ALL_ALPHAS="0.0 1.0 1.5 3.0"

echo "=========================================="
echo "Jailbreak Alpha=1.0 Evaluation Pipeline"
echo "Started: $(date -Iseconds)"
echo "=========================================="

# ── Pre-flight ─────────────────────────────────────────────────────
if [ ! -f "$EXPERIMENT_DIR/alpha_1.0.jsonl" ]; then
    echo "ERROR: $EXPERIMENT_DIR/alpha_1.0.jsonl not found — run inference first."
    exit 1
fi
echo "Pre-flight: alpha_1.0.jsonl exists ($(wc -l < "$EXPERIMENT_DIR/alpha_1.0.jsonl") records)"
if [ "${CODEX_VERIFY_OPENAI_LIMITS:-1}" = "1" ]; then
    echo "Pre-flight: verifying OpenAI Batch Tier-3 limits via Codex CLI..."
    scripts/infra/check_openai_batch_limits_via_codex.sh
else
    echo "Pre-flight: skipping OpenAI Batch limit check (CODEX_VERIFY_OPENAI_LIMITS=0)"
fi
echo ""

# ── Step 1: Binary Judge (batch API) ──────────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Step 1/3: Binary Judge (batch API)      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark jailbreak \
    --input_dir "$EXPERIMENT_DIR" \
    --alphas $ALL_ALPHAS \
    --api-mode batch \
    2>&1 | tee "$BINARY_JUDGE_LOG"

echo ""
echo "[$(date -Iseconds)] Binary judge complete"
echo ""

# ── Step 2: CSV-v2 Evaluation (batch API) ─────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Step 2/3: CSV-v2 Evaluation (batch API) ║"
echo "╚══════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/evaluate_csv2.py \
    --input_dir "$EXPERIMENT_DIR" \
    --output_dir "$CSV2_DIR" \
    --alphas $ALL_ALPHAS \
    --judge_model gpt-4o \
    --api-mode batch \
    2>&1 | tee "$CSV2_EVAL_LOG"

echo ""
echo "[$(date -Iseconds)] CSV-v2 evaluation complete"
echo ""

# ── Step 3: CSV-v2 Analysis ───────────────────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Step 3/3: CSV-v2 Analysis               ║"
echo "╚══════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/analyze_csv2.py \
    --experiment_dir "$CSV2_DIR" \
    --alphas $ALL_ALPHAS \
    2>&1 | tee "$CSV2_ANALYSIS_LOG"

echo ""
echo "=========================================="
echo "Pipeline finished — $(date -Iseconds)"
echo "=========================================="
echo "Binary judge log: $BINARY_JUDGE_LOG"
echo "CSV-v2 eval log:  $CSV2_EVAL_LOG"
echo "CSV-v2 analysis:  $CSV2_ANALYSIS_LOG"
echo ""
echo "Done!"

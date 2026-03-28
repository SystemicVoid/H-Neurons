#!/usr/bin/env bash
# Pipeline: generate jailbreak alpha=1.0 baseline + binary judge + CSV-v2 evaluation
# Alpha 1.0 = default activation (no scaling), the missing baseline point.
#
# Chain: GPU inference → binary judge (batch API) → CSV-v2 eval (batch API) → analysis
set -euo pipefail

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

INFERENCE_LOG="$LOG_DIR/jailbreak_alpha1_inference_${TIMESTAMP}.log"
BINARY_JUDGE_LOG="$LOG_DIR/jailbreak_alpha1_binary_judge_${TIMESTAMP}.log"
CSV2_EVAL_LOG="$LOG_DIR/jailbreak_alpha1_csv2_eval_${TIMESTAMP}.log"
CSV2_ANALYSIS_LOG="$LOG_DIR/jailbreak_alpha1_csv2_analysis_${TIMESTAMP}.log"

EXPERIMENT_DIR="data/gemma3_4b/intervention/jailbreak/experiment"
CSV2_DIR="data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
ALL_ALPHAS="0.0 1.0 1.5 3.0"

echo "=========================================="
echo "Jailbreak Alpha=1.0 Baseline Pipeline"
echo "Started: $(date -Iseconds)"
echo "=========================================="
echo "Experiment dir: $EXPERIMENT_DIR"
echo "CSV-v2 dir:     $CSV2_DIR"
echo "New alpha:      1.0"
echo "All alphas:     $ALL_ALPHAS"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────
if [ -f "$EXPERIMENT_DIR/alpha_1.0.jsonl" ]; then
    echo "ERROR: $EXPERIMENT_DIR/alpha_1.0.jsonl already exists!"
    echo "       Remove or archive it before running this pipeline."
    exit 1
fi

if [ -f "$CSV2_DIR/alpha_1.0.jsonl" ]; then
    echo "ERROR: $CSV2_DIR/alpha_1.0.jsonl already exists!"
    echo "       Remove or archive it before running this pipeline."
    exit 1
fi

echo "Pre-flight: no existing alpha_1.0 files — safe to proceed"
if [ "${CODEX_VERIFY_OPENAI_LIMITS:-1}" = "1" ]; then
    echo "Pre-flight: verifying OpenAI Batch Tier-2 limits via Codex CLI..."
    scripts/infra/check_openai_batch_limits_via_codex.sh
else
    echo "Pre-flight: skipping OpenAI Batch limit check (CODEX_VERIFY_OPENAI_LIMITS=0)"
fi
echo ""

# ── Step 1: GPU Inference ─────────────────────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Step 1/4: GPU Inference (alpha=1.0)     ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Estimated time: ~1.5-2 hours (500 samples × 5000 max tokens)"
echo ""

PYTHONUNBUFFERED=1 systemd-inhibit --what=idle --why="jailbreak alpha=1.0 inference" \
    uv run python scripts/run_intervention.py \
    --benchmark jailbreak \
    --alphas 1.0 \
    --max_new_tokens 5000 \
    --seed 42 \
    --output_dir "$EXPERIMENT_DIR" \
    2>&1 | tee "$INFERENCE_LOG"

ALPHA1_COUNT=$(wc -l < "$EXPERIMENT_DIR/alpha_1.0.jsonl")
echo ""
echo "[$(date -Iseconds)] Inference complete: $ALPHA1_COUNT records written"
echo ""

# ── Step 2: Binary Judge (batch API) ──────────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Step 2/4: Binary Judge (batch API)      ║"
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

# ── Step 3: CSV-v2 Evaluation (batch API) ─────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Step 3/4: CSV-v2 Evaluation (batch API) ║"
echo "╚══════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/evaluate_csv2.py \
    --input_dir "$EXPERIMENT_DIR" \
    --output_dir "$CSV2_DIR" \
    --alphas $ALL_ALPHAS \
    --judge_model gpt-4o \
    --api_mode batch \
    2>&1 | tee "$CSV2_EVAL_LOG"

echo ""
echo "[$(date -Iseconds)] CSV-v2 evaluation complete"
echo ""

# ── Step 4: CSV-v2 Analysis ───────────────────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Step 4/4: CSV-v2 Analysis               ║"
echo "╚══════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/analyze_csv2.py \
    --experiment_dir "$CSV2_DIR" \
    --alphas $ALL_ALPHAS \
    2>&1 | tee "$CSV2_ANALYSIS_LOG"

echo ""
echo "[$(date -Iseconds)] Analysis complete"

# ── Log to backlog ────────────────────────────────────────────────
cat >> notes/runs_to_analyse.md <<EOF

## $(date -Iseconds) | $EXPERIMENT_DIR (alpha=1.0 addition)
What: jailbreak + h-neuron scaling, alpha=1.0 baseline (5000 tok, seed=42, jailbreakbench)
Key files: alpha_1.0.jsonl, csv2_evaluation/alpha_1.0.jsonl, results.json
Status: awaiting analysis
EOF

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Pipeline finished — $(date -Iseconds)"
echo "=========================================="
echo "Inference log:    $INFERENCE_LOG"
echo "Binary judge log: $BINARY_JUDGE_LOG"
echo "CSV-v2 eval log:  $CSV2_EVAL_LOG"
echo "CSV-v2 analysis:  $CSV2_ANALYSIS_LOG"
echo ""
echo "New files:"
echo "  $EXPERIMENT_DIR/alpha_1.0.jsonl"
echo "  $CSV2_DIR/alpha_1.0.jsonl"
echo ""
echo "Done!"

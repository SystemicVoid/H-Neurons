#!/usr/bin/env bash
# Jailbreak CSV-v2 Negative Control Pipeline
# 3 random-neuron seeds × 4 alphas → CSV-v2 scoring → comparative analysis
set -euo pipefail

# Re-launch under systemd-inhibit for sleep protection
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle --why="jailbreak-csv2-negative-control" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

CONTROL_BASE="data/gemma3_4b/intervention/jailbreak/control"
H_EXPERIMENT_DIR="data/gemma3_4b/intervention/jailbreak/experiment"
H_CSV2_DIR="data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
ALPHAS="0.0 1.0 1.5 3.0"
SEEDS=(0 1 2)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GEN_LOG="$LOG_DIR/jailbreak_control_gen_${TIMESTAMP}.log"
CSV2_LOG="$LOG_DIR/jailbreak_control_csv2_${TIMESTAMP}.log"
ANALYSIS_LOG="$LOG_DIR/jailbreak_control_analysis_${TIMESTAMP}.log"

echo "=========================================="
echo "Jailbreak CSV-v2 Negative Control Pipeline"
echo "$(date -Iseconds)"
echo "=========================================="
echo "Control base:   $CONTROL_BASE"
echo "H-neuron CSV-v2: $H_CSV2_DIR"
echo "Alphas:         $ALPHAS"
echo "Seeds:          ${SEEDS[*]}"
echo "Gen log:        $GEN_LOG"
echo "CSV-v2 log:     $CSV2_LOG"
echo "Analysis log:   $ANALYSIS_LOG"
echo ""

# ── Step 1: GPU Generation ────────────────────────────────────────
echo "╔══════════════════════════════════════════════════╗"
echo "║  Step 1/4: Generation (3 seeds × 4 alphas)      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/run_negative_control.py \
    --benchmark jailbreak \
    --jailbreak_batch_size 1 \
    2>&1 | tee -a "$GEN_LOG"

echo ""
echo "[$(date -Iseconds)] Generation complete"

# ── Step 2: Score H-neuron requested alphas (incremental) ─────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Step 2/4: Score H-neuron requested alphas      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_csv2.py \
    --input_dir "$H_EXPERIMENT_DIR" \
    --output_dir "$H_CSV2_DIR" \
    --alphas $ALPHAS \
    --judge_model gpt-4o \
    --api-mode batch \
    2>&1 | tee -a "$CSV2_LOG"

echo ""
echo "[$(date -Iseconds)] H-neuron scoring done"

# ── Step 3: CSV-v2 Scoring (control seeds) ────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Step 3/4: CSV-v2 Batch Scoring (control seeds)  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${CONTROL_BASE}/seed_${SEED}_unconstrained"
    echo "--- Scoring seed_${SEED} ---"

    MAX_RETRIES=3
    ATTEMPT=0
    EVAL_SUCCESS=false

    while [ $ATTEMPT -lt $MAX_RETRIES ]; do
        ATTEMPT=$((ATTEMPT + 1))
        if PYTHONUNBUFFERED=1 uv run python scripts/evaluate_csv2.py \
            --input_dir "$SEED_DIR" \
            --output_dir "$SEED_DIR" \
            --alphas $ALPHAS \
            --judge_model gpt-4o \
            --api-mode batch \
            2>&1 | tee -a "$CSV2_LOG"; then
            EVAL_SUCCESS=true
            echo "[$(date -Iseconds)] seed_${SEED} scored successfully"
            break
        else
            echo "[$(date -Iseconds)] seed_${SEED} scoring failed (attempt ${ATTEMPT})"
            if [ $ATTEMPT -lt $MAX_RETRIES ]; then
                WAIT=$((30 * ATTEMPT))
                echo "  Retrying in ${WAIT}s..."
                sleep $WAIT
            fi
        fi
    done

    if [ "$EVAL_SUCCESS" != "true" ]; then
        echo "FATAL: seed_${SEED} CSV-v2 scoring failed after ${MAX_RETRIES} attempts"
        exit 1
    fi
    echo ""
done

echo "[$(date -Iseconds)] All seeds scored"

# ── Step 4: Comparative Analysis ──────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Step 4/4: Comparative Analysis                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/analyze_csv2_control.py \
    --control_base "$CONTROL_BASE" \
    --experiment_dir "$H_CSV2_DIR" \
    --alphas $ALPHAS \
    2>&1 | tee -a "$ANALYSIS_LOG"

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Pipeline complete — $(date -Iseconds)"
echo "=========================================="
echo "Gen log:      $GEN_LOG"
echo "CSV-v2 log:   $CSV2_LOG"
echo "Analysis log: $ANALYSIS_LOG"
echo "Output:       $CONTROL_BASE"
echo ""
echo "Key outputs:"
echo "  $CONTROL_BASE/comparison_csv2_summary.json"
echo "  $CONTROL_BASE/negative_control_csv2_comparison.png"

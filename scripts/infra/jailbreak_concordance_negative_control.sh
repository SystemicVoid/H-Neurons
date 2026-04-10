#!/usr/bin/env bash
# Three-Judge Concordance Negative Control Pipeline
#
# Produces paired (binary, csv2-v2, csv2-v3) annotations on the same jailbreak
# responses for both H-neuron and random-neuron control conditions.
#
# 8 steps:
#   1. GPU generation       (3 random seeds × 4 alphas)
#   2. Binary judge         (evaluate_intervention.py on controls)
#   3. CSV2 v3 on H-neurons (evaluate_csv2.py → csv2_v3_evaluation/)
#   4. CSV2 v2 on controls  (evaluate_csv2_v2.py → seed_X_csv2_v2/)
#   5. CSV2 v3 on controls  (evaluate_csv2.py → seed_X_csv2_v3/)
#   6. Neg-control analysis (v3-consistent)
#   7. Neg-control analysis (v2-consistent)
#   8. Concordance analysis (three-judge comparison)
set -euo pipefail

# Re-launch under systemd-inhibit for sleep protection
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle --why="jailbreak-concordance-pipeline" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

# ── Paths ─────────────────────────────────────────────────────────
CONTROL_BASE="data/gemma3_4b/intervention/jailbreak/control"
H_EXPERIMENT_DIR="data/gemma3_4b/intervention/jailbreak/experiment"
H_CSV2_V2_DIR="data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
H_CSV2_V3_DIR="data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation"
CONCORDANCE_DIR="${CONTROL_BASE}/concordance"
ALPHAS="0.0 1.0 1.5 3.0"
SEEDS=(0 1 2)

# ── Logging ───────────────────────────────────────────────────────
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GEN_LOG="$LOG_DIR/concordance_gen_${TIMESTAMP}.log"
BINARY_LOG="$LOG_DIR/concordance_binary_${TIMESTAMP}.log"
CSV2_V3_H_LOG="$LOG_DIR/concordance_csv2v3_hneuron_${TIMESTAMP}.log"
CSV2_V2_CTRL_LOG="$LOG_DIR/concordance_csv2v2_ctrl_${TIMESTAMP}.log"
CSV2_V3_CTRL_LOG="$LOG_DIR/concordance_csv2v3_ctrl_${TIMESTAMP}.log"
ANALYSIS_LOG="$LOG_DIR/concordance_analysis_${TIMESTAMP}.log"

echo "=========================================================="
echo "Three-Judge Concordance Negative Control Pipeline"
echo "$(date -Iseconds)"
echo "=========================================================="
echo "Control base:     $CONTROL_BASE"
echo "H-neuron exp:     $H_EXPERIMENT_DIR"
echo "H-neuron CSV2 v2: $H_CSV2_V2_DIR (existing, preserved)"
echo "H-neuron CSV2 v3: $H_CSV2_V3_DIR (new)"
echo "Concordance:      $CONCORDANCE_DIR"
echo "Alphas:           $ALPHAS"
echo "Seeds:            ${SEEDS[*]}"
echo ""

# ── Helpers ───────────────────────────────────────────────────────
retry_eval() {
    local description="$1"
    shift
    local max_retries=3
    local attempt=0
    local success=false

    while [ $attempt -lt $max_retries ]; do
        attempt=$((attempt + 1))
        if PYTHONUNBUFFERED=1 "$@"; then
            success=true
            echo "[$(date -Iseconds)] ${description} succeeded"
            break
        else
            echo "[$(date -Iseconds)] ${description} failed (attempt ${attempt})"
            if [ $attempt -lt $max_retries ]; then
                local wait=$((30 * attempt))
                echo "  Retrying in ${wait}s..."
                sleep $wait
            fi
        fi
    done

    if [ "$success" != "true" ]; then
        echo "FATAL: ${description} failed after ${max_retries} attempts"
        exit 1
    fi
}

verify_field() {
    # verify_field <dir> <field_name> <alphas...>
    local dir="$1"
    local field="$2"
    shift 2
    local alphas_arr=("$@")
    for alpha in "${alphas_arr[@]}"; do
        local path="${dir}/alpha_${alpha}.jsonl"
        if [ ! -f "$path" ]; then
            echo "  VERIFY WARN: $path not found"
            continue
        fi
        local missing
        missing=$(python3 -c "
import json
recs = [json.loads(l) for l in open('${path}') if l.strip()]
missing = sum(1 for r in recs if '${field}' not in r)
print(missing)
")
        if [ "$missing" -gt 0 ]; then
            echo "  VERIFY WARN: ${path}: ${missing} records missing '${field}'"
        fi
    done
}

# ── Step 1/8: GPU Generation ─────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 1/8: Generation (3 seeds × 4 alphas)          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/run_negative_control.py \
    --benchmark jailbreak \
    --jailbreak_batch_size 1 \
    2>&1 | tee -a "$GEN_LOG"

echo ""
echo "[$(date -Iseconds)] Generation complete"

# Verify line counts
for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${CONTROL_BASE}/seed_${SEED}_unconstrained"
    if [ -d "$SEED_DIR" ]; then
        for alpha in $ALPHAS; do
            fpath="${SEED_DIR}/alpha_${alpha}.jsonl"
            if [ -f "$fpath" ]; then
                lines=$(wc -l < "$fpath")
                echo "  seed_${SEED} alpha=${alpha}: ${lines} records"
            else
                echo "  seed_${SEED} alpha=${alpha}: MISSING"
            fi
        done
    fi
done

# ── Step 2/8: Binary Judge on Controls ───────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 2/8: Binary Judge on Controls                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${CONTROL_BASE}/seed_${SEED}_unconstrained"
    echo "--- Binary judge: seed_${SEED} ---"

    retry_eval "binary judge seed_${SEED}" \
        uv run python scripts/evaluate_intervention.py \
            --benchmark jailbreak \
            --input_dir "$SEED_DIR" \
            --alphas $ALPHAS \
            --judge_model gpt-4o \
            --api-mode batch \
        2>&1 | tee -a "$BINARY_LOG"

    # shellcheck disable=SC2086
    verify_field "$SEED_DIR" "compliance" $ALPHAS
    echo ""
done

echo "[$(date -Iseconds)] Binary judging complete"

# ── Step 3/8: CSV2 v3 on H-neurons ──────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 3/8: CSV2 v3 on H-neurons                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

retry_eval "CSV2 v3 H-neurons" \
    uv run python scripts/evaluate_csv2.py \
        --input_dir "$H_EXPERIMENT_DIR" \
        --output_dir "$H_CSV2_V3_DIR" \
        --alphas $ALPHAS \
        --judge_model gpt-4o \
        --api-mode batch \
    2>&1 | tee -a "$CSV2_V3_H_LOG"

# shellcheck disable=SC2086
verify_field "$H_CSV2_V3_DIR" "csv2" $ALPHAS

echo ""
echo "[$(date -Iseconds)] H-neuron v3 scoring done"

# ── Step 4/8: CSV2 v2 on Controls ────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 4/8: CSV2 v2 on Controls                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${CONTROL_BASE}/seed_${SEED}_unconstrained"
    V2_DIR="${CONTROL_BASE}/seed_${SEED}_unconstrained_csv2_v2"
    echo "--- CSV2 v2: seed_${SEED} ---"

    retry_eval "CSV2 v2 seed_${SEED}" \
        uv run python scripts/evaluate_csv2_v2.py \
            --input_dir "$SEED_DIR" \
            --output_dir "$V2_DIR" \
            --alphas $ALPHAS \
            --judge_model gpt-4o \
            --api_mode batch \
        2>&1 | tee -a "$CSV2_V2_CTRL_LOG"

    # shellcheck disable=SC2086
    verify_field "$V2_DIR" "csv2" $ALPHAS
    echo ""
done

echo "[$(date -Iseconds)] Control v2 scoring done"

# ── Step 5/8: CSV2 v3 on Controls ────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 5/8: CSV2 v3 on Controls                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${CONTROL_BASE}/seed_${SEED}_unconstrained"
    V3_DIR="${CONTROL_BASE}/seed_${SEED}_unconstrained_csv2_v3"
    echo "--- CSV2 v3: seed_${SEED} ---"

    retry_eval "CSV2 v3 seed_${SEED}" \
        uv run python scripts/evaluate_csv2.py \
            --input_dir "$SEED_DIR" \
            --output_dir "$V3_DIR" \
            --alphas $ALPHAS \
            --judge_model gpt-4o \
            --api-mode batch \
        2>&1 | tee -a "$CSV2_V3_CTRL_LOG"

    # shellcheck disable=SC2086
    verify_field "$V3_DIR" "csv2" $ALPHAS
    echo ""
done

echo "[$(date -Iseconds)] Control v3 scoring done"

# ── Step 6/8: Negative Control Analysis (v3-consistent) ─────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 6/8: Negative Control Analysis (v3)            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/analyze_csv2_control.py \
    --control_base "$CONTROL_BASE" \
    --experiment_dir "$H_CSV2_V3_DIR" \
    --control_csv2_suffix "_csv2_v3" \
    --alphas $ALPHAS \
    2>&1 | tee -a "$ANALYSIS_LOG"

echo "[$(date -Iseconds)] V3 negative control analysis done"

# ── Step 7/8: Negative Control Analysis (v2-consistent) ─────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 7/8: Negative Control Analysis (v2)            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/analyze_csv2_control.py \
    --control_base "$CONTROL_BASE" \
    --experiment_dir "$H_CSV2_V2_DIR" \
    --control_csv2_suffix "_csv2_v2" \
    --alphas $ALPHAS \
    2>&1 | tee -a "$ANALYSIS_LOG"

echo "[$(date -Iseconds)] V2 negative control analysis done"

# ── Step 8/8: Concordance Analysis ───────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 8/8: Three-Judge Concordance Analysis          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$CONCORDANCE_DIR"

PYTHONUNBUFFERED=1 uv run python scripts/analyze_concordance.py \
    --h_neuron_binary_dir "$H_EXPERIMENT_DIR" \
    --h_neuron_csv2_v2_dir "$H_CSV2_V2_DIR" \
    --h_neuron_csv2_v3_dir "$H_CSV2_V3_DIR" \
    --control_base "$CONTROL_BASE" \
    --control_seeds ${SEEDS[*]} \
    --alphas $ALPHAS \
    --gold_labels tests/gold_labels/jailbreak_cross_alpha_gold.jsonl \
    --output_dir "$CONCORDANCE_DIR" \
    2>&1 | tee -a "$ANALYSIS_LOG"

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "Pipeline complete — $(date -Iseconds)"
echo "=========================================================="
echo "Logs:"
echo "  Gen:        $GEN_LOG"
echo "  Binary:     $BINARY_LOG"
echo "  CSV2 v3 H:  $CSV2_V3_H_LOG"
echo "  CSV2 v2 C:  $CSV2_V2_CTRL_LOG"
echo "  CSV2 v3 C:  $CSV2_V3_CTRL_LOG"
echo "  Analysis:   $ANALYSIS_LOG"
echo ""
echo "Key outputs:"
echo "  H-neuron v3:    $H_CSV2_V3_DIR"
echo "  Control v2:     ${CONTROL_BASE}/seed_*_csv2_v2/"
echo "  Control v3:     ${CONTROL_BASE}/seed_*_csv2_v3/"
echo "  Concordance:    $CONCORDANCE_DIR/concordance_summary.json"

#!/usr/bin/env bash
# Gated ITI pipeline — Phase 2: Fold extraction + evaluation + controls
#
# Prerequisite: Gate 1 passed (pipeline_state.json exists with locked K×α)
#
# Runs Phases 2a-c + 3 then STOPS for human review.
# After this script completes, check:
#   - Held-out vs cal-val accuracy (overfit?)
#   - Controls vs real ITI (does the intervention do anything?)
#
# Then run: ./scripts/infra/iti_pipeline_downstream.sh
set -euo pipefail

# Wrap in systemd-inhibit (~3h GPU)
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="ITI pipeline evaluate phase (~3h GPU)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

SEED=42
FAMILY="iti_truthfulqa_paperfaithful"
CAL_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration"
PROD_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production"
MANIFESTS="data/manifests"

# ===================================================================
# Gate check: require pipeline_state.json with gate_1_sweep.locked
# ===================================================================
STATE_FILE="${CAL_DIR}/pipeline_state.json"
if [ ! -f "${STATE_FILE}" ]; then
    echo "FATAL: ${STATE_FILE} not found." >&2
    echo "Run scripts/lock_config.py first to lock K×α after reviewing the sweep." >&2
    exit 1
fi

LOCKED_K=$(jq -r '.gate_1_sweep.locked.K' "${STATE_FILE}")
LOCKED_ALPHA=$(jq -r '.gate_1_sweep.locked.alpha' "${STATE_FILE}")

if [ "${LOCKED_K}" = "null" ] || [ "${LOCKED_ALPHA}" = "null" ]; then
    echo "FATAL: gate_1_sweep.locked not set in ${STATE_FILE}" >&2
    exit 1
fi

echo "Using locked config from pipeline_state.json: K=${LOCKED_K}, α=${LOCKED_ALPHA}"

HUMAN_OVERRIDE=$(jq -r '.gate_1_sweep.human_override // "null"' "${STATE_FILE}")
if [ "${HUMAN_OVERRIDE}" != "null" ]; then
    REASON=$(jq -r '.gate_1_sweep.human_override.reason' "${STATE_FILE}")
    echo "  (human override — reason: ${REASON})"
fi

# ===================================================================
# Phase 2a: Extract per-fold artifacts (final 2-fold CV)
# ===================================================================
echo ""
echo "=== Phase 2a: Extract final fold artifacts ==="
for FOLD in 0 1; do
    FOLD_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold${FOLD}"
    FOLD_ARTIFACT="${FOLD_DIR}/iti_heads.pt"
    if [ -f "${FOLD_ARTIFACT}" ]; then
        echo "--- Fold ${FOLD}: artifact exists, skipping ---"
    else
        echo "--- Fold ${FOLD} ---"
        PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
            --family "${FAMILY}" \
            --seed "${SEED}" \
            --fold_path "${MANIFESTS}/truthfulqa_final_fold${FOLD}_seed${SEED}.json" \
            --output_dir "${FOLD_DIR}" \
            2>&1 | tee "logs/extract_final_fold${FOLD}.log"
    fi
done

# ===================================================================
# Phase 2b: Evaluate MC1/MC2 on each fold with locked config
# ===================================================================
echo ""
echo "=== Phase 2b: Final 2-fold MC evaluation ==="
for FOLD in 0 1; do
    FOLD_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold${FOLD}/iti_heads.pt"
    for VARIANT in mc1 mc2; do
        echo "--- Fold ${FOLD}, ${VARIANT} ---"
        PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
            --intervention_mode iti_head \
            --iti_head_path "${FOLD_ARTIFACT}" \
            --iti_family truthfulqa_paperfaithful \
            --benchmark truthfulqa_mc --truthfulqa_variant "${VARIANT}" \
            --iti_k "${LOCKED_K}" --alphas 0.0 "${LOCKED_ALPHA}" \
            --sample_manifest "${MANIFESTS}/truthfulqa_final_fold${FOLD}_heldout_${VARIANT}_seed${SEED}.json" \
            2>&1 | tee "logs/final_fold${FOLD}_${VARIANT}.log"
    done
done

# ===================================================================
# Phase 2c: Build production artifact (all 817 questions)
# ===================================================================
echo ""
echo "=== Phase 2c: Build production artifact ==="
PROD_ARTIFACT="${PROD_DIR}/iti_heads.pt"
if [ -f "${PROD_ARTIFACT}" ]; then
    echo "Production artifact exists, skipping."
else
    PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
        --family "${FAMILY}" \
        --seed "${SEED}" \
        --fold_path "${MANIFESTS}/truthfulqa_production_fold_seed${SEED}.json" \
        --output_dir "${PROD_DIR}" \
        2>&1 | tee logs/extract_production.log
fi

if [ ! -f "${PROD_ARTIFACT}" ]; then
    echo "FATAL: Production artifact not found at ${PROD_ARTIFACT}" >&2
    exit 1
fi

# ===================================================================
# Phase 3: Controls
# ===================================================================
echo ""
echo "=== Phase 3a: Random-head control (3 seeds) ==="
for CTRL_SEED in 1 2 3; do
    echo "--- Random head seed ${CTRL_SEED} ---"
    PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
        --intervention_mode iti_head \
        --iti_head_path "${PROD_ARTIFACT}" \
        --iti_family truthfulqa_paperfaithful \
        --benchmark truthfulqa_mc --truthfulqa_variant mc1 \
        --iti_k "${LOCKED_K}" --alphas 0.0 "${LOCKED_ALPHA}" \
        --iti_selection_strategy random --iti_random_seed "${CTRL_SEED}" \
        --sample_manifest "${MANIFESTS}/truthfulqa_cal_val_mc1_seed${SEED}.json" \
        2>&1 | tee "logs/control_random_head_seed${CTRL_SEED}.log"
done

echo ""
echo "=== Phase 3b: Random-direction control (3 seeds) ==="
for CTRL_SEED in 1 2 3; do
    echo "--- Random direction seed ${CTRL_SEED} ---"
    PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
        --intervention_mode iti_head \
        --iti_head_path "${PROD_ARTIFACT}" \
        --iti_family truthfulqa_paperfaithful \
        --benchmark truthfulqa_mc --truthfulqa_variant mc1 \
        --iti_k "${LOCKED_K}" --alphas 0.0 "${LOCKED_ALPHA}" \
        --iti_direction_mode random --iti_direction_random_seed "${CTRL_SEED}" \
        --sample_manifest "${MANIFESTS}/truthfulqa_cal_val_mc1_seed${SEED}.json" \
        2>&1 | tee "logs/control_random_dir_seed${CTRL_SEED}.log"
done

echo ""
echo "=== Phase 3c: Benign behavior check ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${PROD_ARTIFACT}" \
    --iti_family truthfulqa_paperfaithful \
    --benchmark jailbreak_benign \
    --iti_k "${LOCKED_K}" --alphas 0.0 "${LOCKED_ALPHA}" \
    --max_samples 100 \
    2>&1 | tee logs/control_benign.log

# ===================================================================
# Update pipeline_state.json with gate_2 info
# ===================================================================
echo ""
echo "=== Updating pipeline state ==="
GATE2_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Use jq to add gate_2_folds to existing state
jq --arg ts "${GATE2_TS}" \
   '.gate_2_folds = { "completed_at": $ts, "status": "awaiting_review" }' \
   "${STATE_FILE}" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "${STATE_FILE}"

# ===================================================================
# GATE 2: Stop for human review
# ===================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  GATE 2: Fold evals + controls complete — review before         ║"
echo "║          running downstream benchmarks                          ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                 ║"
echo "║  Check:                                                         ║"
echo "║  1. Held-out MC1/MC2 vs cal-val: overfit if drop > 3pp          ║"
echo "║  2. Controls vs real ITI: null if within 1pp                    ║"
echo "║  3. Fold 0 vs Fold 1 divergence: unstable if > 5pp              ║"
echo "║  4. Spot-check benign responses for coherence at α=${LOCKED_ALPHA}║"
echo "║                                                                 ║"
echo "║  If satisfied: ./scripts/infra/iti_pipeline_downstream.sh       ║"
echo "║                                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

#!/usr/bin/env bash
# Paper-faithful ITI rerun pipeline for Gemma-3-4B-IT on TruthfulQA.
#
# Full chain: calibration splits → extraction → K×α sweep → lock →
# 2-fold CV extraction + eval → production artifact → controls → downstream.
#
# Prerequisites:
#   - data/benchmarks/TruthfulQA.csv (official 817 questions)
#   - data/benchmarks/simpleqa_verified.csv (DeepMind SimpleQA Verified)
#   - data/manifests/truthfulqa_canonical_questions.json (from build_truthfulqa_splits.py)
#   - GPU free (check nvitop -1)
#
# Usage:
#   ./scripts/infra/iti_paperfaithful_rerun_pipeline.sh
set -euo pipefail

# Wrap in systemd-inhibit for the ~6h GPU chain (per AGENTS.md)
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="ITI paperfaithful rerun pipeline (~6h GPU)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

SEED=42
FAMILY="iti_truthfulqa_paperfaithful"
CAL_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration"
PROD_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production"
MANIFESTS="data/manifests"

# ===================================================================
# Phase 0: Build calibration + final 2-fold + production splits
# ===================================================================
echo "=== Phase 0: Build splits ==="
PYTHONUNBUFFERED=1 uv run python scripts/build_truthfulqa_calibration_splits.py \
    --seed "${SEED}" \
    2>&1 | tee logs/build_calibration_splits.log

# ===================================================================
# Phase 1a: Extract calibration artifact
# ===================================================================
echo ""
echo "=== Phase 1a: Extract calibration artifact ==="
PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
    --family "${FAMILY}" \
    --seed "${SEED}" \
    --fold_path "${MANIFESTS}/truthfulqa_cal_fold_seed${SEED}.json" \
    --output_dir "${CAL_DIR}" \
    2>&1 | tee logs/extract_calibration.log

CAL_ARTIFACT="${CAL_DIR}/iti_heads.pt"
if [ ! -f "${CAL_ARTIFACT}" ]; then
    echo "FATAL: Calibration artifact not found at ${CAL_ARTIFACT}" >&2
    exit 1
fi

# ===================================================================
# Phase 1b: K × α sweep on cal-val
# ===================================================================
echo ""
echo "=== Phase 1b: K × α sweep ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_calibration_sweep.py \
    --artifact_path "${CAL_ARTIFACT}" \
    --cal_val_mc1_manifest "${MANIFESTS}/truthfulqa_cal_val_mc1_seed${SEED}.json" \
    --cal_val_mc2_manifest "${MANIFESTS}/truthfulqa_cal_val_mc2_seed${SEED}.json" \
    --k_values 8 12 16 24 32 40 \
    --alpha_values 0.0 0.5 1.0 2.0 4.0 6.0 8.0 12.0 16.0 \
    --output_dir "${CAL_DIR}" \
    2>&1 | tee logs/calibration_sweep.log

LOCKED_CONFIG="${CAL_DIR}/locked_iti_config.json"
if [ ! -f "${LOCKED_CONFIG}" ]; then
    echo "FATAL: locked_iti_config.json not found" >&2
    exit 1
fi

LOCKED_K=$(jq -r '.K_locked' "${LOCKED_CONFIG}")
LOCKED_ALPHA=$(jq -r '.alpha_locked' "${LOCKED_CONFIG}")
echo "Locked: K=${LOCKED_K}, α=${LOCKED_ALPHA}"

# ===================================================================
# Phase 2a: Extract per-fold artifacts (final 2-fold CV)
# ===================================================================
echo ""
echo "=== Phase 2a: Extract final fold artifacts ==="
for FOLD in 0 1; do
    FOLD_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold${FOLD}"
    echo "--- Fold ${FOLD} ---"
    PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
        --family "${FAMILY}" \
        --seed "${SEED}" \
        --fold_path "${MANIFESTS}/truthfulqa_final_fold${FOLD}_seed${SEED}.json" \
        --output_dir "${FOLD_DIR}" \
        2>&1 | tee "logs/extract_final_fold${FOLD}.log"
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
PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
    --family "${FAMILY}" \
    --seed "${SEED}" \
    --fold_path "${MANIFESTS}/truthfulqa_production_fold_seed${SEED}.json" \
    --output_dir "${PROD_DIR}" \
    2>&1 | tee logs/extract_production.log

PROD_ARTIFACT="${PROD_DIR}/iti_heads.pt"
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
# Phase 4: Downstream evals (frozen production artifact)
# ===================================================================
echo ""
echo "=== Phase 4a: SimpleQA (verified) ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${PROD_ARTIFACT}" \
    --iti_family truthfulqa_paperfaithful \
    --benchmark simpleqa \
    --simpleqa_path data/benchmarks/simpleqa_verified.csv \
    --iti_k "${LOCKED_K}" --alphas 0.0 "${LOCKED_ALPHA}" \
    2>&1 | tee logs/downstream_simpleqa.log

echo ""
echo "=== Phase 4b: FalseQA ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${PROD_ARTIFACT}" \
    --iti_family truthfulqa_paperfaithful \
    --benchmark falseqa \
    --iti_k "${LOCKED_K}" --alphas 0.0 "${LOCKED_ALPHA}" \
    2>&1 | tee logs/downstream_falseqa.log

echo ""
echo "=== Phase 4c: FaithEval standard ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${PROD_ARTIFACT}" \
    --iti_family truthfulqa_paperfaithful \
    --benchmark faitheval --prompt_style standard \
    --iti_k "${LOCKED_K}" --alphas 0.0 "${LOCKED_ALPHA}" \
    2>&1 | tee logs/downstream_faitheval.log

# ===================================================================
# Helper: resolve ITI output dir (mirrors run_intervention.py logic)
# ===================================================================
resolve_iti_output_dir() {
    local HEAD_PATH=$1
    local BENCHMARK_NAME=$2
    uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from run_intervention import build_iti_output_suffix
suffix = build_iti_output_suffix('${HEAD_PATH}', 'truthfulqa_paperfaithful', ${LOCKED_K}, 'ranked', 42)
print(f'data/gemma3_4b/intervention/${BENCHMARK_NAME}_' + suffix + '/experiment')
"
}

# ===================================================================
# Phase 5a: 2-fold TruthfulQA report (MC1 + MC2)
# ===================================================================
echo ""
echo "=== Phase 5a: 2-fold TruthfulQA report ==="
for VARIANT in mc1 mc2; do
    FOLD0_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold0/iti_heads.pt"
    FOLD1_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold1/iti_heads.pt"
    FOLD0_DIR=$(resolve_iti_output_dir "${FOLD0_ARTIFACT}" "truthfulqa_mc_${VARIANT}")
    FOLD1_DIR=$(resolve_iti_output_dir "${FOLD1_ARTIFACT}" "truthfulqa_mc_${VARIANT}")

    echo "--- ${VARIANT}: fold0=${FOLD0_DIR}, fold1=${FOLD1_DIR} ---"
    PYTHONUNBUFFERED=1 uv run python scripts/report_iti_2fold.py \
        --fold0_dir "${FOLD0_DIR}" \
        --fold1_dir "${FOLD1_DIR}" \
        --locked_alpha "${LOCKED_ALPHA}" \
        --locked_k "${LOCKED_K}" \
        --variant "${VARIANT}" \
        --output_dir notes/act3-reports \
        2>&1 | tee "logs/report_2fold_${VARIANT}.log"
done

# ===================================================================
# Phase 5b: SimpleQA judging (batch mode)
# ===================================================================
echo ""
echo "=== Phase 5b: SimpleQA judging ==="
SIMPLEQA_DIR=$(resolve_iti_output_dir "${PROD_ARTIFACT}" "simpleqa")
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark simpleqa \
    --input_dir "${SIMPLEQA_DIR}" \
    --alphas 0.0 "${LOCKED_ALPHA}" \
    --api-mode batch \
    2>&1 | tee logs/judge_simpleqa.log

# ===================================================================
# Phase 5c: FalseQA judging (batch mode)
# ===================================================================
echo ""
echo "=== Phase 5c: FalseQA judging ==="
FALSEQA_DIR=$(resolve_iti_output_dir "${PROD_ARTIFACT}" "falseqa")
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark falseqa \
    --input_dir "${FALSEQA_DIR}" \
    --alphas 0.0 "${LOCKED_ALPHA}" \
    --api-mode batch \
    2>&1 | tee logs/judge_falseqa.log

# ===================================================================
# Append to runs_to_analyse.md
# ===================================================================
echo ""
echo "=== Appending to runs_to_analyse.md ==="
mkdir -p notes
RUN_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat >> notes/runs_to_analyse.md << EOF

## ${RUN_TS} | Paper-faithful ITI rerun (full pipeline)
What: TruthfulQA 2-fold CV (MC1+MC2) + controls (random-head×3, random-dir×3, benign) + downstream (SimpleQA, FalseQA, FaithEval) with locked K=${LOCKED_K}, α=${LOCKED_ALPHA}
Key files: ${LOCKED_CONFIG}, notes/act3-reports/iti_2fold_mc1_report.json, notes/act3-reports/iti_2fold_mc2_report.json
Status: awaiting analysis
EOF

# ===================================================================
# Done
# ===================================================================
echo ""
echo "=== All phases complete ==="
echo "Locked config: K=${LOCKED_K}, α=${LOCKED_ALPHA}"
echo "Production artifact: ${PROD_ARTIFACT}"
echo "Reports: notes/act3-reports/iti_2fold_mc{1,2}_report.json"

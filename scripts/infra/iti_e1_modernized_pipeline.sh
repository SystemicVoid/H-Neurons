#!/usr/bin/env bash
# E1 ITI pipeline (TruthfulQA-modernized extraction).
#
# Chain:
#   splits -> calibration extraction -> Kxalpha sweep -> lock
#   -> final fold extraction + MC eval -> production extraction
#   -> SimpleQA-200 (factual_phrase, first_3_tokens) + batch judge
#   -> 2-fold MC reports.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="E1 ITI modernized pipeline (~3h GPU + judge wait)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

SEED=42
FAMILY="iti_truthfulqa_modernized"
ITI_FAMILY_ARG="truthfulqa_modernized"
CAL_DIR="data/contrastive/truthfulness/iti_truthfulqa_modernized_calibration"
PROD_DIR="data/contrastive/truthfulness/iti_truthfulqa_modernized_production"
FOLD_ROOT="data/contrastive/truthfulness/iti_truthfulqa_modernized"
MANIFESTS="data/manifests"
SIMPLEQA_MANIFEST="data/manifests/simpleqa_verified_control200_seed42.json"
SIMPLEQA_PROMPT_STYLE="factual_phrase"
SIMPLEQA_SCOPE="first_3_tokens"

echo "=== Phase 0: Build splits ==="
PYTHONUNBUFFERED=1 uv run python scripts/build_truthfulqa_calibration_splits.py \
    --seed "${SEED}" \
    2>&1 | tee logs/e1_build_calibration_splits.log

echo ""
echo "=== Phase 1a: Extract E1 calibration artifact ==="
PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
    --family "${FAMILY}" \
    --seed "${SEED}" \
    --fold_path "${MANIFESTS}/truthfulqa_cal_fold_seed${SEED}.json" \
    --output_dir "${CAL_DIR}" \
    2>&1 | tee logs/e1_extract_calibration.log

CAL_ARTIFACT="${CAL_DIR}/iti_heads.pt"
if [ ! -f "${CAL_ARTIFACT}" ]; then
    echo "FATAL: Calibration artifact not found at ${CAL_ARTIFACT}" >&2
    exit 1
fi

echo ""
echo "=== Phase 1b: K x alpha sweep ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_calibration_sweep.py \
    --artifact_path "${CAL_ARTIFACT}" \
    --cal_val_mc1_manifest "${MANIFESTS}/truthfulqa_cal_val_mc1_seed${SEED}.json" \
    --cal_val_mc2_manifest "${MANIFESTS}/truthfulqa_cal_val_mc2_seed${SEED}.json" \
    --k_values 8 12 16 24 32 40 \
    --alpha_values 0.0 0.5 1.0 2.0 4.0 6.0 8.0 12.0 16.0 \
    --output_dir "${CAL_DIR}" \
    2>&1 | tee logs/e1_calibration_sweep.log

LOCKED_CONFIG="${CAL_DIR}/locked_iti_config.json"
if [ ! -f "${LOCKED_CONFIG}" ]; then
    echo "FATAL: locked_iti_config.json not found at ${LOCKED_CONFIG}" >&2
    exit 1
fi

LOCKED_K=$(jq -r '.K_locked' "${LOCKED_CONFIG}")
LOCKED_ALPHA=$(jq -r '.alpha_locked' "${LOCKED_CONFIG}")
echo "Locked: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"

echo ""
echo "=== Phase 2a: Extract final fold artifacts ==="
for FOLD in 0 1; do
    FOLD_DIR="${FOLD_ROOT}/final_fold${FOLD}"
    echo "--- Fold ${FOLD} -> ${FOLD_DIR} ---"
    PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
        --family "${FAMILY}" \
        --seed "${SEED}" \
        --fold_path "${MANIFESTS}/truthfulqa_final_fold${FOLD}_seed${SEED}.json" \
        --output_dir "${FOLD_DIR}" \
        2>&1 | tee "logs/e1_extract_final_fold${FOLD}.log"
done

echo ""
echo "=== Phase 2b: Final 2-fold TruthfulQA MC eval ==="
for FOLD in 0 1; do
    FOLD_ARTIFACT="${FOLD_ROOT}/final_fold${FOLD}/iti_heads.pt"
    FOLD_DEF="${MANIFESTS}/truthfulqa_final_fold${FOLD}_seed${SEED}.json"
    for VARIANT in mc1 mc2; do
        echo "--- Fold ${FOLD}, ${VARIANT} ---"
        PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
            --intervention_mode iti_head \
            --iti_head_path "${FOLD_ARTIFACT}" \
            --iti_family "${ITI_FAMILY_ARG}" \
            --benchmark truthfulqa_mc \
            --truthfulqa_variant "${VARIANT}" \
            --truthfulqa_fold_path "${FOLD_DEF}" \
            --iti_k "${LOCKED_K}" \
            --alphas 0.0 "${LOCKED_ALPHA}" \
            --sample_manifest "${MANIFESTS}/truthfulqa_final_fold${FOLD}_heldout_${VARIANT}_seed${SEED}.json" \
            2>&1 | tee "logs/e1_final_fold${FOLD}_${VARIANT}.log"
    done
done

echo ""
echo "=== Phase 2c: Build E1 production artifact ==="
PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
    --family "${FAMILY}" \
    --seed "${SEED}" \
    --fold_path "${MANIFESTS}/truthfulqa_production_fold_seed${SEED}.json" \
    --output_dir "${PROD_DIR}" \
    2>&1 | tee logs/e1_extract_production.log

PROD_ARTIFACT="${PROD_DIR}/iti_heads.pt"
if [ ! -f "${PROD_ARTIFACT}" ]; then
    echo "FATAL: Production artifact not found at ${PROD_ARTIFACT}" >&2
    exit 1
fi

resolve_iti_output_dir() {
    local HEAD_PATH=$1
    local BENCHMARK_NAME=$2
    local DECODE_SCOPE=${3:-full_decode}
    uv run python - <<PY
import sys
sys.path.insert(0, "scripts")
from run_intervention import build_iti_output_suffix
suffix = build_iti_output_suffix(
    "${HEAD_PATH}",
    "${ITI_FAMILY_ARG}",
    ${LOCKED_K},
    "ranked",
    42,
    "artifact",
    None,
    "${DECODE_SCOPE}",
)
print(f"data/gemma3_4b/intervention/${BENCHMARK_NAME}_{suffix}/experiment")
PY
}

echo ""
echo "=== Phase 3a: 2-fold TruthfulQA report ==="
for VARIANT in mc1 mc2; do
    FOLD0_DIR=$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold0/iti_heads.pt" "truthfulqa_mc_${VARIANT}")
    FOLD1_DIR=$(resolve_iti_output_dir "${FOLD_ROOT}/final_fold1/iti_heads.pt" "truthfulqa_mc_${VARIANT}")
    PYTHONUNBUFFERED=1 uv run python scripts/report_iti_2fold.py \
        --fold0_dir "${FOLD0_DIR}" \
        --fold1_dir "${FOLD1_DIR}" \
        --locked_alpha "${LOCKED_ALPHA}" \
        --locked_k "${LOCKED_K}" \
        --variant "${VARIANT}" \
        --output_dir notes/act3-reports \
        2>&1 | tee "logs/e1_report_2fold_${VARIANT}.log"
done

echo ""
echo "=== Phase 3b: SimpleQA-200 pilot (first_3_tokens) ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${PROD_ARTIFACT}" \
    --iti_family "${ITI_FAMILY_ARG}" \
    --benchmark simpleqa \
    --simpleqa_path data/benchmarks/simpleqa_verified.csv \
    --simpleqa_prompt_style "${SIMPLEQA_PROMPT_STYLE}" \
    --iti_decode_scope "${SIMPLEQA_SCOPE}" \
    --iti_k "${LOCKED_K}" \
    --alphas 0.0 "${LOCKED_ALPHA}" \
    --sample_manifest "${SIMPLEQA_MANIFEST}" \
    2>&1 | tee logs/e1_simpleqa_200.log

echo ""
echo "=== Phase 3c: SimpleQA-200 batch judge ==="
SIMPLEQA_DIR=$(resolve_iti_output_dir "${PROD_ARTIFACT}" "simpleqa_${SIMPLEQA_PROMPT_STYLE}" "${SIMPLEQA_SCOPE}")
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark simpleqa \
    --input_dir "${SIMPLEQA_DIR}" \
    --alphas 0.0 "${LOCKED_ALPHA}" \
    --api-mode batch \
    2>&1 | tee logs/e1_simpleqa_200_judge.log

echo ""
echo "=== E1 pipeline complete ==="
echo "Family: ${FAMILY}"
echo "Locked config: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"
echo "Production artifact: ${PROD_ARTIFACT}"
echo "SimpleQA dir: ${SIMPLEQA_DIR}"

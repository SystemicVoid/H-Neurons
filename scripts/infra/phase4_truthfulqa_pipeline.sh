#!/usr/bin/env bash
# Phase 3+4: Extract paper-faithful TruthfulQA ITI artifact (2-fold CV), then
# evaluate on MC1/MC2.  Chain: extract → MC1 eval (K=16) → MC2 eval (K=16)
#
# Usage:
#   ./scripts/infra/phase4_truthfulqa_pipeline.sh          # both folds
#   FOLD=0 ./scripts/infra/phase4_truthfulqa_pipeline.sh   # fold 0 only
set -euo pipefail

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

# Which folds to run (default: both)
FOLDS="${FOLD:-0 1}"

for FOLD_IDX in ${FOLDS}; do
    FOLD_PATH="data/manifests/truthfulqa_fold${FOLD_IDX}_seed42.json"
    MANIFEST_MC1="data/manifests/truthfulqa_fold${FOLD_IDX}_heldout_mc1_seed42.json"
    MANIFEST_MC2="data/manifests/truthfulqa_fold${FOLD_IDX}_heldout_mc2_seed42.json"
    ARTIFACT_DIR="data/contrastive/truthfulness/iti_truthfulqa_paper"
    ARTIFACT_PATH="${ARTIFACT_DIR}/iti_heads.pt"

    echo "=== Fold ${FOLD_IDX}: Extract paper-faithful TruthfulQA ITI artifact ==="
    PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
        --family iti_truthfulqa_paper --seed 42 \
        --fold_path "${FOLD_PATH}" \
        2>&1 | tee logs/extract_truthfulqa_paper_fold${FOLD_IDX}.log

    if [ ! -f "${ARTIFACT_PATH}" ]; then
        echo "FATAL: Artifact not found at ${ARTIFACT_PATH}" >&2
        exit 1
    fi
    echo "Artifact extracted successfully."

    echo ""
    echo "=== Fold ${FOLD_IDX}: TruthfulQA MC1 — K=16, alpha sweep ==="
    PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
        --intervention_mode iti_head \
        --iti_head_path "${ARTIFACT_PATH}" \
        --iti_family truthfulqa_paper \
        --benchmark truthfulqa_mc --truthfulqa_variant mc1 \
        --iti_k 16 --alphas 0.0 1.0 2.0 4.0 8.0 12.0 16.0 \
        --sample_manifest "${MANIFEST_MC1}" \
        2>&1 | tee logs/truthfulqa_mc1_iti_paper_k16_fold${FOLD_IDX}.log

    echo ""
    echo "=== Fold ${FOLD_IDX}: TruthfulQA MC2 — K=16, alpha sweep ==="
    PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
        --intervention_mode iti_head \
        --iti_head_path "${ARTIFACT_PATH}" \
        --iti_family truthfulqa_paper \
        --benchmark truthfulqa_mc --truthfulqa_variant mc2 \
        --iti_k 16 --alphas 0.0 1.0 2.0 4.0 8.0 12.0 16.0 \
        --sample_manifest "${MANIFEST_MC2}" \
        2>&1 | tee logs/truthfulqa_mc2_iti_paper_k16_fold${FOLD_IDX}.log

    echo ""
done

echo "=== Phase 4 baseline: triviaqa_transfer on MC1 (fold 0 test set) ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt \
    --iti_family triviaqa_transfer \
    --benchmark truthfulqa_mc --truthfulqa_variant mc1 \
    --iti_k 16 --alphas 0.0 1.0 2.0 4.0 8.0 \
    --sample_manifest "data/manifests/truthfulqa_fold0_heldout_mc1_seed42.json" \
    2>&1 | tee logs/truthfulqa_mc1_iti_triviaqa_k16.log

echo ""
echo "=== All Phase 4 K=16 runs complete ==="
echo "Check output dirs under data/gemma3_4b/intervention/truthfulqa_mc_*/"

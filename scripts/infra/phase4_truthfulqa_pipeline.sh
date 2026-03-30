#!/usr/bin/env bash
# Phase 3+4: Extract paper-faithful TruthfulQA ITI artifact, then evaluate on MC1/MC2
# Chain: extract → MC1 eval (K=16) → MC2 eval (K=16)
set -euo pipefail

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

ARTIFACT_DIR="data/contrastive/truthfulness/iti_truthfulqa_paper"
ARTIFACT_PATH="${ARTIFACT_DIR}/iti_heads.pt"
MANIFEST_MC1="data/manifests/truthfulqa_paper_heldout_mc1_ids_seed42.json"
MANIFEST_MC2="data/manifests/truthfulqa_paper_heldout_mc2_ids_seed42.json"

echo "=== Phase 3: Extract paper-faithful TruthfulQA ITI artifact ==="
PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
    --family iti_truthfulqa_paper --seed 42 \
    2>&1 | tee logs/extract_truthfulqa_paper.log

if [ ! -f "${ARTIFACT_PATH}" ]; then
    echo "FATAL: Artifact not found at ${ARTIFACT_PATH}" >&2
    exit 1
fi
echo "Artifact extracted successfully."

echo ""
echo "=== Phase 4a: TruthfulQA MC1 — K=16, alpha sweep ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${ARTIFACT_PATH}" \
    --iti_family truthfulqa_paper \
    --benchmark truthfulqa_mc --truthfulqa_variant mc1 \
    --iti_k 16 --alphas 0.0 1.0 2.0 4.0 8.0 12.0 16.0 \
    --sample_manifest "${MANIFEST_MC1}" \
    2>&1 | tee logs/truthfulqa_mc1_iti_paper_k16.log

echo ""
echo "=== Phase 4b: TruthfulQA MC2 — K=16, alpha sweep ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${ARTIFACT_PATH}" \
    --iti_family truthfulqa_paper \
    --benchmark truthfulqa_mc --truthfulqa_variant mc2 \
    --iti_k 16 --alphas 0.0 1.0 2.0 4.0 8.0 12.0 16.0 \
    --sample_manifest "${MANIFEST_MC2}" \
    2>&1 | tee logs/truthfulqa_mc2_iti_paper_k16.log

echo ""
echo "=== Phase 4 baseline: triviaqa_transfer on MC1 ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt \
    --iti_family triviaqa_transfer \
    --benchmark truthfulqa_mc --truthfulqa_variant mc1 \
    --iti_k 16 --alphas 0.0 1.0 2.0 4.0 8.0 \
    --sample_manifest "${MANIFEST_MC1}" \
    2>&1 | tee logs/truthfulqa_mc1_iti_triviaqa_k16.log

echo ""
echo "=== All Phase 4 K=16 runs complete ==="
echo "Check output dirs under data/gemma3_4b/intervention/truthfulqa_mc_*/"

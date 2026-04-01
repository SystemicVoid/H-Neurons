#!/usr/bin/env bash
# Act 3 priority reruns:
#   1. D1 H-neuron TruthfulQA MC rerun on the final held-out folds
#   2. D4 SimpleQA forced-commitment rerun (escape hatch removed)
#
# Runs as a single staged chain so the GPU job can be queued once in tmux.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="Act 3 priority reruns (D1 TruthfulQA + SimpleQA prompt fix)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

mkdir -p logs notes

echo "=== Preflight: GPU status ==="
nvitop -1 | sed -n '1,80p'

AWAITING_ANALYSIS=$(rg -c "^Status: awaiting analysis" notes/runs_to_analyse.md || true)
echo "Awaiting-analysis backlog: ${AWAITING_ANALYSIS}"

echo ""
echo "=== Phase 1: D1 H-neuron TruthfulQA MC rerun ==="
ALPHAS=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
for FOLD in 0 1; do
    for VARIANT in mc1 mc2; do
        OUTDIR="data/gemma3_4b/intervention/truthfulqa_mc_${VARIANT}_h-neurons_final-fold${FOLD}/experiment"
        MANIFEST="data/manifests/truthfulqa_final_fold${FOLD}_heldout_${VARIANT}_seed42.json"
        LOGFILE="logs/d1_truthfulqa_${VARIANT}_fold${FOLD}.log"
        echo "--- Fold ${FOLD}, ${VARIANT} -> ${OUTDIR} ---"
        PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
            --benchmark truthfulqa_mc \
            --truthfulqa_variant "${VARIANT}" \
            --classifier_path models/gemma3_4b_classifier.pkl \
            --sample_manifest "${MANIFEST}" \
            --output_dir "${OUTDIR}" \
            --alphas "${ALPHAS[@]}" \
            --seed 42 \
            2>&1 | tee "${LOGFILE}"
    done
done

D1_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat >> notes/runs_to_analyse.md << EOF

## ${D1_TS} | data/gemma3_4b/intervention/truthfulqa_mc_{mc1,mc2}_h-neurons_final-fold{0,1}/experiment
What: D1 H-neuron TruthfulQA MC rerun on the final held-out folds, variants={mc1,mc2}, alpha=[0.0,0.5,1.0,1.5,2.0,2.5,3.0]
Key files: results.*.json, alpha_*.jsonl, run_intervention.provenance.*.json
Status: awaiting analysis
EOF

echo ""
echo "=== Phase 2: SimpleQA forced-commitment rerun ==="
SIMPLEQA_PROMPT_STYLE=factual_phrase \
SIMPLEQA_ALPHAS="0.0 4.0 8.0" \
SIMPLEQA_BATCH_MAX_ENQUEUED_TOKENS=1600000 \
bash scripts/infra/simpleqa_standalone.sh

echo ""
echo "=== Act 3 priority reruns complete ==="

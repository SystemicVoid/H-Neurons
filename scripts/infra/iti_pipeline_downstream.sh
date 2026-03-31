#!/usr/bin/env bash
# Gated ITI pipeline — Phase 3: Downstream evals + reports + judging
#
# Prerequisite: Gate 2 passed (pipeline_state.json has gate_2_folds)
#
# Runs Phases 4 + 5: SimpleQA, FalseQA, FaithEval, 2-fold reports, batch judges.
set -euo pipefail

# Wrap in systemd-inhibit (~1.5h GPU + batch judge wait)
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="ITI pipeline downstream phase (~1.5h GPU + judge)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

CAL_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration"
PROD_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production"

# ===================================================================
# Gate check: require pipeline_state.json with gate_2_folds
# ===================================================================
STATE_FILE="${CAL_DIR}/pipeline_state.json"
if [ ! -f "${STATE_FILE}" ]; then
    echo "FATAL: ${STATE_FILE} not found." >&2
    exit 1
fi

HAS_GATE2=$(jq -r '.gate_2_folds // "null"' "${STATE_FILE}")
if [ "${HAS_GATE2}" = "null" ]; then
    echo "FATAL: gate_2_folds not set in ${STATE_FILE}" >&2
    echo "Run iti_pipeline_evaluate.sh first and review results." >&2
    exit 1
fi

LOCKED_K=$(jq -r '.gate_1_sweep.locked.K' "${STATE_FILE}")
LOCKED_ALPHA=$(jq -r '.gate_1_sweep.locked.alpha' "${STATE_FILE}")
PROD_ARTIFACT="${PROD_DIR}/iti_heads.pt"

echo "Using locked config: K=${LOCKED_K}, α=${LOCKED_ALPHA}"
echo "Production artifact: ${PROD_ARTIFACT}"

if [ ! -f "${PROD_ARTIFACT}" ]; then
    echo "FATAL: Production artifact not found at ${PROD_ARTIFACT}" >&2
    exit 1
fi

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
# Phase 4a: SimpleQA (verified)
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

# ===================================================================
# Phase 4b: FalseQA
# ===================================================================
echo ""
echo "=== Phase 4b: FalseQA ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${PROD_ARTIFACT}" \
    --iti_family truthfulqa_paperfaithful \
    --benchmark falseqa \
    --iti_k "${LOCKED_K}" --alphas 0.0 "${LOCKED_ALPHA}" \
    2>&1 | tee logs/downstream_falseqa.log

# ===================================================================
# Phase 4c: FaithEval standard
# ===================================================================
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
# Update pipeline state + runs_to_analyse
# ===================================================================
echo ""
echo "=== Updating pipeline state ==="
GATE3_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
jq --arg ts "${GATE3_TS}" \
   '.gate_3_downstream = { "completed_at": $ts, "status": "awaiting_analysis" }' \
   "${STATE_FILE}" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "${STATE_FILE}"

mkdir -p notes
cat >> notes/runs_to_analyse.md << EOF

## ${GATE3_TS} | Paper-faithful ITI rerun (gated pipeline)
What: TruthfulQA 2-fold CV (MC1+MC2) + controls (random-head×3, random-dir×3, benign) + downstream (SimpleQA, FalseQA, FaithEval) with locked K=${LOCKED_K}, α=${LOCKED_ALPHA}
Key files: ${CAL_DIR}/pipeline_state.json, notes/act3-reports/iti_2fold_mc1_report.json, notes/act3-reports/iti_2fold_mc2_report.json
Status: awaiting analysis
EOF

echo ""
echo "=== All phases complete ==="
echo "Locked config: K=${LOCKED_K}, α=${LOCKED_ALPHA}"
echo "Production artifact: ${PROD_ARTIFACT}"
echo "Reports: notes/act3-reports/iti_2fold_mc{1,2}_report.json"
echo "Pipeline state: ${STATE_FILE}"

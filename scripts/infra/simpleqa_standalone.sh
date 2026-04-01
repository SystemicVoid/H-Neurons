#!/usr/bin/env bash
# Standalone SimpleQA run: inference + batch judging
#
# Reads locked K/alpha from pipeline_state.json (Gate 1 must be passed).
# Skips the codex OpenAI-limit check (no credits); batch token cap is set
# to 1_000_000 — sized for 1000 questions × 2 alphas (~900K tokens total)
# which fits in one gpt-4o Tier-2 batch without chunking.
#
# Usage:
#   bash scripts/infra/simpleqa_standalone.sh
#   SIMPLEQA_PROMPT_STYLE=factual_phrase bash scripts/infra/simpleqa_standalone.sh
set -euo pipefail

# Disable the codex-based OpenAI batch limit pre-flight (no API credits)
export CODEX_VERIFY_OPENAI_LIMITS=0

# Wrap in systemd-inhibit (inference ~20-30 min + batch poll)
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="SimpleQA standalone (inference + batch judge)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

CAL_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration"
PROD_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production"
STATE_FILE="${CAL_DIR}/pipeline_state.json"

# --- Gate check ---
if [ ! -f "${STATE_FILE}" ]; then
    echo "FATAL: ${STATE_FILE} not found." >&2
    exit 1
fi
HAS_GATE2=$(jq -r '.gate_2_folds // "null"' "${STATE_FILE}")
if [ "${HAS_GATE2}" = "null" ]; then
    echo "FATAL: gate_2_folds not in ${STATE_FILE} — run iti_pipeline_evaluate.sh first." >&2
    exit 1
fi

LOCKED_K=$(jq -r '.gate_1_sweep.locked.K' "${STATE_FILE}")
LOCKED_ALPHA=$(jq -r '.gate_1_sweep.locked.alpha' "${STATE_FILE}")
PROD_ARTIFACT="${PROD_DIR}/iti_heads.pt"
SIMPLEQA_PROMPT_STYLE="${SIMPLEQA_PROMPT_STYLE:-escape_hatch}"
SIMPLEQA_ALPHAS="${SIMPLEQA_ALPHAS:-0.0 ${LOCKED_ALPHA}}"
SIMPLEQA_BATCH_MAX_ENQUEUED_TOKENS="${SIMPLEQA_BATCH_MAX_ENQUEUED_TOKENS:-1000000}"

echo "Locked config: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"
echo "Production artifact: ${PROD_ARTIFACT}"
echo "Prompt style: ${SIMPLEQA_PROMPT_STYLE}"
echo "Alpha grid: ${SIMPLEQA_ALPHAS}"

if [ ! -f "${PROD_ARTIFACT}" ]; then
    echo "FATAL: Production artifact not found at ${PROD_ARTIFACT}" >&2
    exit 1
fi

# --- Resolve output dir ---
SIMPLEQA_DIR=$(uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from run_intervention import build_iti_output_suffix
benchmark_name = 'simpleqa'
if '${SIMPLEQA_PROMPT_STYLE}' != 'escape_hatch':
    benchmark_name += '_${SIMPLEQA_PROMPT_STYLE}'
suffix = build_iti_output_suffix('${PROD_ARTIFACT}', 'truthfulqa_paperfaithful', ${LOCKED_K}, 'ranked', 42)
print(f'data/gemma3_4b/intervention/{benchmark_name}_' + suffix + '/experiment')
")
echo "Output dir: ${SIMPLEQA_DIR}"

# ===================================================================
# Phase 1: SimpleQA inference
# ===================================================================
echo ""
echo "=== SimpleQA inference (K=${LOCKED_K}, alphas=${SIMPLEQA_ALPHAS}) ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --intervention_mode iti_head \
    --iti_head_path "${PROD_ARTIFACT}" \
    --iti_family truthfulqa_paperfaithful \
    --benchmark simpleqa \
    --simpleqa_path data/benchmarks/simpleqa_verified.csv \
    --simpleqa_prompt_style "${SIMPLEQA_PROMPT_STYLE}" \
    --iti_k "${LOCKED_K}" --alphas ${SIMPLEQA_ALPHAS} \
    2>&1 | tee logs/simpleqa_standalone_inference.log

# ===================================================================
# Phase 2: SimpleQA batch judging
# 1000 questions × 2 alphas = 2000 requests (~900K tokens, fits one batch)
# ===================================================================
echo ""
echo "=== SimpleQA batch judging (batch-max-enqueued-tokens=${SIMPLEQA_BATCH_MAX_ENQUEUED_TOKENS}) ==="
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark simpleqa \
    --input_dir "${SIMPLEQA_DIR}" \
    --alphas ${SIMPLEQA_ALPHAS} \
    --api-mode batch \
    --batch-max-enqueued-tokens "${SIMPLEQA_BATCH_MAX_ENQUEUED_TOKENS}" \
    2>&1 | tee logs/simpleqa_standalone_judge.log

# ===================================================================
# Phase 3: Export site data
# ===================================================================
echo ""
echo "=== Export site data ==="
uv run python scripts/export_site_data.py 2>&1 | tee logs/simpleqa_standalone_export.log

# ===================================================================
# Log to runs_to_analyse
# ===================================================================
RUN_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
mkdir -p notes
cat >> notes/runs_to_analyse.md << EOF

## ${RUN_TS} | ${SIMPLEQA_DIR}
What: SimpleQA standalone — paper-faithful ITI K=${LOCKED_K}, prompt_style=${SIMPLEQA_PROMPT_STYLE}, alpha=[${SIMPLEQA_ALPHAS// /,}], 1000 verified questions
Key files: results.json, alpha_*.jsonl, run_intervention.provenance.*.json
Status: awaiting analysis
EOF

echo ""
echo "=== Done ==="
echo "Results: ${SIMPLEQA_DIR}"
echo "Logged to notes/runs_to_analyse.md"

#!/usr/bin/env bash
# Standalone SimpleQA run: inference + batch judging
#
# Reads locked K/alpha from pipeline_state.json (Gate 1 must be passed).
# Skips the codex OpenAI-limit check (no credits). Batch token cap defaults to
# 1_000_000, which comfortably fits the full 1000-question 2-alpha run and the
# smaller control pilots used in Act 3.
#
# Usage:
#   bash scripts/infra/simpleqa_standalone.sh
#   SIMPLEQA_PROMPT_STYLE=factual_phrase bash scripts/infra/simpleqa_standalone.sh
#   SIMPLEQA_PROMPT_STYLE=factual_phrase \
#   SIMPLEQA_SAMPLE_MANIFEST=data/manifests/simpleqa_verified_control200_seed42.json \
#   ITI_SELECTION_STRATEGY=random \
#   ITI_RANDOM_SEED=1 \
#   SIMPLEQA_ALPHAS="4.0 8.0" \
#   LOG_STEM=simpleqa_random_head_seed1 \
#   bash scripts/infra/simpleqa_standalone.sh
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
SIMPLEQA_SAMPLE_MANIFEST="${SIMPLEQA_SAMPLE_MANIFEST:-}"
ITI_SELECTION_STRATEGY="${ITI_SELECTION_STRATEGY:-ranked}"
ITI_RANDOM_SEED="${ITI_RANDOM_SEED:-42}"
ITI_DIRECTION_MODE="${ITI_DIRECTION_MODE:-artifact}"
ITI_DIRECTION_RANDOM_SEED="${ITI_DIRECTION_RANDOM_SEED:-}"
LOG_STEM="${LOG_STEM:-simpleqa_standalone}"

echo "Locked config: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"
echo "Production artifact: ${PROD_ARTIFACT}"
echo "Prompt style: ${SIMPLEQA_PROMPT_STYLE}"
echo "Alpha grid: ${SIMPLEQA_ALPHAS}"
echo "Selection strategy: ${ITI_SELECTION_STRATEGY}"
echo "ITI random seed: ${ITI_RANDOM_SEED}"
echo "Direction mode: ${ITI_DIRECTION_MODE}"
if [ -n "${ITI_DIRECTION_RANDOM_SEED}" ]; then
    echo "Direction random seed: ${ITI_DIRECTION_RANDOM_SEED}"
fi
if [ -n "${SIMPLEQA_SAMPLE_MANIFEST}" ]; then
    echo "Sample manifest: ${SIMPLEQA_SAMPLE_MANIFEST}"
fi
echo "Log stem: ${LOG_STEM}"

if [ ! -f "${PROD_ARTIFACT}" ]; then
    echo "FATAL: Production artifact not found at ${PROD_ARTIFACT}" >&2
    exit 1
fi
if [ -n "${SIMPLEQA_SAMPLE_MANIFEST}" ] && [ ! -f "${SIMPLEQA_SAMPLE_MANIFEST}" ]; then
    echo "FATAL: Sample manifest not found at ${SIMPLEQA_SAMPLE_MANIFEST}" >&2
    exit 1
fi

read -r -a SIMPLEQA_ALPHA_ARRAY <<< "${SIMPLEQA_ALPHAS}"

# --- Resolve output dir ---
SIMPLEQA_DIR=$(
    SIMPLEQA_PROMPT_STYLE="${SIMPLEQA_PROMPT_STYLE}" \
    PROD_ARTIFACT="${PROD_ARTIFACT}" \
    LOCKED_K="${LOCKED_K}" \
    ITI_SELECTION_STRATEGY="${ITI_SELECTION_STRATEGY}" \
    ITI_RANDOM_SEED="${ITI_RANDOM_SEED}" \
    ITI_DIRECTION_MODE="${ITI_DIRECTION_MODE}" \
    ITI_DIRECTION_RANDOM_SEED="${ITI_DIRECTION_RANDOM_SEED}" \
    uv run python - <<'PY'
import os
import sys

sys.path.insert(0, "scripts")
from run_intervention import build_iti_output_suffix

benchmark_name = "simpleqa"
prompt_style = os.environ["SIMPLEQA_PROMPT_STYLE"]
if prompt_style != "escape_hatch":
    benchmark_name += f"_{prompt_style}"

direction_random_seed = os.environ.get("ITI_DIRECTION_RANDOM_SEED") or None
suffix = build_iti_output_suffix(
    os.environ["PROD_ARTIFACT"],
    "truthfulqa_paperfaithful",
    int(os.environ["LOCKED_K"]),
    os.environ["ITI_SELECTION_STRATEGY"],
    int(os.environ["ITI_RANDOM_SEED"]),
    os.environ["ITI_DIRECTION_MODE"],
    None if direction_random_seed is None else int(direction_random_seed),
)
print(f"data/gemma3_4b/intervention/{benchmark_name}_{suffix}/experiment")
PY
)
echo "Output dir: ${SIMPLEQA_DIR}"

RUN_ARGS=(
    --intervention_mode iti_head
    --iti_head_path "${PROD_ARTIFACT}"
    --iti_family truthfulqa_paperfaithful
    --benchmark simpleqa
    --simpleqa_path data/benchmarks/simpleqa_verified.csv
    --simpleqa_prompt_style "${SIMPLEQA_PROMPT_STYLE}"
    --iti_k "${LOCKED_K}"
    --iti_selection_strategy "${ITI_SELECTION_STRATEGY}"
    --iti_random_seed "${ITI_RANDOM_SEED}"
    --iti_direction_mode "${ITI_DIRECTION_MODE}"
    --alphas "${SIMPLEQA_ALPHA_ARRAY[@]}"
)
if [ -n "${ITI_DIRECTION_RANDOM_SEED}" ]; then
    RUN_ARGS+=(--iti_direction_random_seed "${ITI_DIRECTION_RANDOM_SEED}")
fi
if [ -n "${SIMPLEQA_SAMPLE_MANIFEST}" ]; then
    RUN_ARGS+=(--sample_manifest "${SIMPLEQA_SAMPLE_MANIFEST}")
fi

JUDGE_ARGS=(
    --benchmark simpleqa
    --input_dir "${SIMPLEQA_DIR}"
    --alphas "${SIMPLEQA_ALPHA_ARRAY[@]}"
    --api-mode batch
    --batch-max-enqueued-tokens "${SIMPLEQA_BATCH_MAX_ENQUEUED_TOKENS}"
)

SIMPLEQA_SAMPLE_COUNT=1000
SIMPLEQA_SAMPLE_DESC="1000 verified questions"
if [ -n "${SIMPLEQA_SAMPLE_MANIFEST}" ]; then
    SIMPLEQA_SAMPLE_COUNT=$(jq 'length' "${SIMPLEQA_SAMPLE_MANIFEST}")
    SIMPLEQA_SAMPLE_DESC="${SIMPLEQA_SAMPLE_COUNT} verified questions via $(basename "${SIMPLEQA_SAMPLE_MANIFEST}")"
fi

# ===================================================================
# Phase 1: SimpleQA inference
# ===================================================================
echo ""
echo "=== SimpleQA inference (K=${LOCKED_K}, alphas=${SIMPLEQA_ALPHAS}) ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    "${RUN_ARGS[@]}" \
    2>&1 | tee "logs/${LOG_STEM}_inference.log"

# ===================================================================
# Phase 2: SimpleQA batch judging
# ===================================================================
echo ""
echo "=== SimpleQA batch judging (batch-max-enqueued-tokens=${SIMPLEQA_BATCH_MAX_ENQUEUED_TOKENS}) ==="
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    "${JUDGE_ARGS[@]}" \
    2>&1 | tee "logs/${LOG_STEM}_judge.log"

# ===================================================================
# Phase 3: Export site data
# ===================================================================
echo ""
echo "=== Export site data ==="
uv run python scripts/export_site_data.py 2>&1 | tee "logs/${LOG_STEM}_export.log"

# ===================================================================
# Log to runs_to_analyse
# ===================================================================
RUN_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
mkdir -p notes
cat >> notes/runs_to_analyse.md << EOF

## ${RUN_TS} | ${SIMPLEQA_DIR}
What: SimpleQA standalone — paper-faithful ITI K=${LOCKED_K}, prompt_style=${SIMPLEQA_PROMPT_STYLE}, selection=${ITI_SELECTION_STRATEGY}, direction_mode=${ITI_DIRECTION_MODE}, seed=${ITI_RANDOM_SEED}, alpha=[${SIMPLEQA_ALPHAS// /,}], ${SIMPLEQA_SAMPLE_DESC}
Key files: results.json, alpha_*.jsonl, run_intervention.provenance.*.json
Status: awaiting analysis
EOF

echo ""
echo "=== Done ==="
echo "Results: ${SIMPLEQA_DIR}"
echo "Logged to notes/runs_to_analyse.md"

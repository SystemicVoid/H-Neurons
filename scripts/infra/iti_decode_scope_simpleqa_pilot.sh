#!/usr/bin/env bash
# Stage 5.2 Gate 2a: 200-ID forced-commitment SimpleQA generation pilot
#
# Runs the three surviving decode scopes on the fixed 200-ID manifest and
# stops at the post-generation review gate before any batch judging.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="Decode-scope SimpleQA 200-ID generation pilot" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

SEED=42
CAL_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration"
PROD_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production"
STATE_FILE="${CAL_DIR}/pipeline_state.json"
MANIFEST="data/manifests/simpleqa_verified_control200_seed${SEED}.json"
PROD_ARTIFACT="${PROD_DIR}/iti_heads.pt"
SCOPES=(full_decode first_3_tokens first_8_tokens)
PROMPT_STYLE="factual_phrase"
LOG_STEM="decode_scope_simpleqa_pilot"

mkdir -p logs notes notes/act3-reports

if [ ! -f "${STATE_FILE}" ]; then
    echo "FATAL: ${STATE_FILE} not found." >&2
    exit 1
fi
if [ ! -f "${MANIFEST}" ]; then
    echo "FATAL: pilot manifest not found at ${MANIFEST}" >&2
    exit 1
fi
if [ ! -f "${PROD_ARTIFACT}" ]; then
    echo "FATAL: production artifact not found at ${PROD_ARTIFACT}" >&2
    exit 1
fi

LOCKED_K=$(jq -r '.gate_1_sweep.locked.K' "${STATE_FILE}")
LOCKED_ALPHA=$(jq -r '.gate_1_sweep.locked.alpha' "${STATE_FILE}")
if [ "${LOCKED_K}" = "null" ] || [ "${LOCKED_ALPHA}" = "null" ]; then
    echo "FATAL: gate_1_sweep.locked missing from ${STATE_FILE}" >&2
    exit 1
fi
if [ "${LOCKED_K}" != "12" ] || [ "${LOCKED_ALPHA}" != "8.0" ]; then
    echo "FATAL: Stage 5.2 is locked to K=12 and alpha=8.0, got K=${LOCKED_K}, alpha=${LOCKED_ALPHA}" >&2
    exit 1
fi

echo "=== Stage 5.2 Gate 2a: Decode-scope SimpleQA generation pilot ==="
echo "Locked config: K=${LOCKED_K}, alpha=${LOCKED_ALPHA}"
echo "Artifact: ${PROD_ARTIFACT}"
echo "Manifest: ${MANIFEST}"
echo "Prompt style: ${PROMPT_STYLE}"
echo "Scopes: ${SCOPES[*]}"

echo ""
echo "=== Local validation ==="
uv run pytest tests/test_truthfulness_iti.py tests/test_utils.py -q \
    2>&1 | tee "logs/${LOG_STEM}_pytest.log"
ruff check scripts/intervene_iti.py scripts/run_intervention.py tests/test_truthfulness_iti.py tests/test_utils.py \
    2>&1 | tee "logs/${LOG_STEM}_ruff.log"
ty check scripts \
    2>&1 | tee "logs/${LOG_STEM}_ty.log"
bash -n scripts/infra/simpleqa_standalone.sh
shellcheck scripts/infra/simpleqa_standalone.sh scripts/infra/iti_decode_scope_simpleqa_pilot.sh \
    2>&1 | tee "logs/${LOG_STEM}_shellcheck.log"

echo ""
echo "=== GPU preflight ==="
if command -v nvitop &>/dev/null; then
    nvitop -1 2>&1 | tee "logs/${LOG_STEM}_nvitop.log"
fi
if command -v nvidia-smi &>/dev/null; then
    PMON_OUTPUT=$(nvidia-smi pmon -c 1 2>&1 || true)
    printf '%s\n' "${PMON_OUTPUT}" | tee "logs/${LOG_STEM}_nvidia_pmon.log"
    UNEXPECTED_COMPUTE=$(
        printf '%s\n' "${PMON_OUTPUT}" \
            | awk 'NR > 2 && $2 ~ /^[0-9]+$/ && $3 ~ /C/ {print}' \
            | rg -v 'xdg-desktop-por|cosmic-edit|cosmic-comp|cosmic-panel|Xwayland|ghostty|renderD|code|spo' \
            || true
    )
    if [ -n "${UNEXPECTED_COMPUTE}" ]; then
        echo "FATAL: unexpected compute-like GPU processes detected:" >&2
        printf '%s\n' "${UNEXPECTED_COMPUTE}" >&2
        exit 1
    fi
fi

echo ""
echo "=== Resolve output dirs / freshness check ==="
mapfile -t PANEL_DIRS < <(
    LOCKED_K="${LOCKED_K}" \
    PROD_ARTIFACT="${PROD_ARTIFACT}" \
    PROMPT_STYLE="${PROMPT_STYLE}" \
    uv run python - <<'PY'
import os
import sys

sys.path.insert(0, "scripts")
from run_intervention import build_iti_output_suffix

scopes = ("full_decode", "first_3_tokens", "first_8_tokens")
benchmark_name = "simpleqa"
prompt_style = os.environ["PROMPT_STYLE"]
if prompt_style != "escape_hatch":
    benchmark_name += f"_{prompt_style}"

for scope in scopes:
    suffix = build_iti_output_suffix(
        os.environ["PROD_ARTIFACT"],
        "truthfulqa_paperfaithful",
        int(os.environ["LOCKED_K"]),
        "ranked",
        42,
        "artifact",
        None,
        scope,
    )
    print(f"data/gemma3_4b/intervention/{benchmark_name}_{suffix}/experiment")
PY
)

for dir in "${PANEL_DIRS[@]}"; do
    if [ -e "${dir}/alpha_0.0.jsonl" ] || compgen -G "${dir}/results.*.json" >/dev/null; then
        echo "FATAL: existing pilot output found at ${dir}" >&2
        echo "Archive or remove the old run before launching a fresh 200-ID scope pilot." >&2
        exit 1
    fi
done

echo ""
echo "=== 200-ID SimpleQA generation pilot ==="
export INHIBIT_WRAPPED=1
for scope in "${SCOPES[@]}"; do
    echo ""
    echo "--- Scope: ${scope} ---"
    SIMPLEQA_PROMPT_STYLE="${PROMPT_STYLE}" \
    SIMPLEQA_SAMPLE_MANIFEST="${MANIFEST}" \
    SIMPLEQA_ALPHAS="0.0 ${LOCKED_ALPHA}" \
    ITI_SELECTION_STRATEGY="ranked" \
    ITI_RANDOM_SEED="${SEED}" \
    ITI_DIRECTION_MODE="artifact" \
    ITI_DECODE_SCOPE="${scope}" \
    SIMPLEQA_RUN_JUDGE=0 \
    SIMPLEQA_EXPORT_SITE_DATA=0 \
    SIMPLEQA_LOG_QUEUE=0 \
    LOG_STEM="${LOG_STEM}_${scope}" \
    INHIBIT_WHY="Decode-scope SimpleQA generation pilot (${scope})" \
    bash scripts/infra/simpleqa_standalone.sh
done

echo ""
echo "=== Log to runs_to_analyse ==="
RUN_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
for idx in "${!SCOPES[@]}"; do
    scope="${SCOPES[$idx]}"
    dir="${PANEL_DIRS[$idx]}"
    cat >> notes/runs_to_analyse.md << EOF

## ${RUN_TS} | ${dir}
What: SimpleQA generation-only pilot — paper-faithful ITI K=${LOCKED_K}, prompt_style=${PROMPT_STYLE}, selection=ranked, direction_mode=artifact, decode_scope=${scope}, seed=${SEED}, alpha=[0.0,${LOCKED_ALPHA}], 200 verified questions via $(basename "${MANIFEST}")
Key files: results.*.json, alpha_*.jsonl, run_intervention.provenance.*.json
Status: awaiting analysis
EOF
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Generation pilot complete.                                 ║"
echo "║  Stop here before any batch judge/API work.                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
for dir in "${PANEL_DIRS[@]}"; do
    echo "Output: ${dir}"
done

#!/usr/bin/env bash
# D7 causal pilot — trimmed continuation.
#
# Runs ONLY causal_locked generation + judge/CSV2 for the 3 conditions
# that matter (baseline_noop, l1_neuron, causal_locked), then produces
# a trimmed paired report.
#
# Preconditions (verified before writing this script):
#   - baseline_noop: alpha_1.0.jsonl complete (500 lines)
#   - l1_neuron:     alpha_3.0.jsonl complete (500 lines)
#   - causal_locked:  empty (no alpha file)
#
# Skipped from original d7_causal_pilot.sh:
#   - probe_locked   (pilot already found probe null)
#   - causal_random × 3 seeds (selector-specificity garnish, not worth ~15h)
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="D7 causal pilot trimmed continuation" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

# ---------- Configuration (mirrored from d7_causal_pilot.sh) ----------

SEED="${D7_SEED:-42}"
MODEL_PATH="${D7_MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${D7_DEVICE_MAP:-cuda:0}"
L1_ALPHA="${D7_L1_ALPHA:-3.0}"

MANIFEST_DIR="data/manifests"
FULL_IDS="${MANIFEST_DIR}/jbb_d7_full_harmful500_seed${SEED}.json"

CAUSAL_ARTIFACT="data/contrastive/refusal/iti_refusal_causal_d7/iti_heads.pt"

RUN_ROOT="data/gemma3_4b/intervention/jailbreak_d7"
PILOT_ROOT="${RUN_ROOT}/pilot100_canonical"
FULL_ROOT="${RUN_ROOT}/full500_canonical"

FULL_BASELINE_DIR="${FULL_ROOT}/baseline_noop/experiment"
FULL_L1_DIR="${FULL_ROOT}/l1_neuron/experiment"
FULL_CAUSAL_DIR="${FULL_ROOT}/causal_locked/experiment"
FULL_REPORT="${FULL_ROOT}/d7_csv2_report.json"

CAUSAL_LOCK_JSON="${PILOT_ROOT}/causal_lock.json"

BASELINE_ALPHA=1.0

RUN_PROFILE_ARGS=(--run_profile canonical)
ITI_DEBUG_ARGS=()
if [[ "${D7_ALLOW_CANONICAL_ITI_DEBUG:-0}" == "1" ]]; then
    echo "WARNING: enabling ITI debug traces with fast profile via D7_ALLOW_CANONICAL_ITI_DEBUG=1"
    RUN_PROFILE_ARGS=(--run_profile fast --jailbreak_batch_size 1)
    ITI_DEBUG_ARGS=(--iti_collect_debug_stats)
fi

PIPELINE="uv run python -m scripts.lib.pipeline"

mkdir -p logs
LOG="logs/d7_causal_pilot_trimmed_$(date +%Y%m%d_%H%M%S).log"

run_cmd() {
    echo "+ $*"
    "$@"
}

run_judge_and_csv2() {
    local dir="$1"
    shift
    local alphas=("$@")
    run_cmd uv run python scripts/evaluate_intervention.py \
        --benchmark jailbreak \
        --input_dir "${dir}" \
        --alphas "${alphas[@]}" \
        --api-mode batch 2>&1 | tee -a "${LOG}"
    run_cmd uv run python scripts/evaluate_csv2.py \
        --input_dir "${dir}" \
        --output_dir "${dir%/experiment}/csv2_evaluation" \
        --alphas "${alphas[@]}" \
        --api-mode batch 2>&1 | tee -a "${LOG}"
}

CAUSAL_LOCKED_ALPHA=$(uv run python - <<'PY' "${CAUSAL_LOCK_JSON}"
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["selected_alpha"])
PY
)

echo "=== D7 causal pilot (trimmed) ===" | tee -a "${LOG}"
echo "seed=${SEED} model=${MODEL_PATH} device_map=${DEVICE_MAP}" | tee -a "${LOG}"
echo "causal_locked_alpha=${CAUSAL_LOCKED_ALPHA}" | tee -a "${LOG}"
echo "log=${LOG}" | tee -a "${LOG}"

${PIPELINE} gpu-preflight 2>&1 | tee -a "${LOG}" || true

echo "Verifying baseline_noop..." | tee -a "${LOG}"
if ! ${PIPELINE} check-stage --output-dir "${FULL_BASELINE_DIR}" --manifest "${FULL_IDS}" --alphas "${BASELINE_ALPHA}"; then
    echo "FATAL: baseline_noop is incomplete — cannot continue trimmed run" >&2
    exit 1
fi
echo "  baseline_noop: OK" | tee -a "${LOG}"

echo "Verifying l1_neuron..." | tee -a "${LOG}"
if ! ${PIPELINE} check-stage --output-dir "${FULL_L1_DIR}" --manifest "${FULL_IDS}" --alphas "${L1_ALPHA}"; then
    echo "FATAL: l1_neuron is incomplete — cannot continue trimmed run" >&2
    exit 1
fi
echo "  l1_neuron: OK" | tee -a "${LOG}"

# ---------- Generation: causal_locked only ----------

mkdir -p "${FULL_CAUSAL_DIR}"

if ! ${PIPELINE} check-stage --output-dir "${FULL_CAUSAL_DIR}" --manifest "${FULL_IDS}" --alphas "${CAUSAL_LOCKED_ALPHA}"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${FULL_CAUSAL_DIR}" \
        --sample_manifest "${FULL_IDS}" \
        --intervention_mode iti_head \
        --iti_head_path "${CAUSAL_ARTIFACT}" \
        --iti_family refusal_causal \
        --iti_k 20 \
        --alphas "${CAUSAL_LOCKED_ALPHA}" \
        --max_new_tokens 5000 \
        "${ITI_DEBUG_ARGS[@]}" \
        "${RUN_PROFILE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
else
    echo "causal_locked generation complete; skipping" | tee -a "${LOG}"
fi

# ---------- Judge + CSV2 for the 3 retained conditions ----------

run_judge_and_csv2 "${FULL_BASELINE_DIR}" "${BASELINE_ALPHA}"
run_judge_and_csv2 "${FULL_L1_DIR}" "${L1_ALPHA}"
run_judge_and_csv2 "${FULL_CAUSAL_DIR}" "${CAUSAL_LOCKED_ALPHA}"

# ---------- Trimmed paired report (baseline vs l1 + causal) ----------

run_cmd uv run python scripts/report_d7_csv2.py \
    --baseline_dir "${FULL_BASELINE_DIR%/experiment}/csv2_evaluation" \
    --baseline_alpha "${BASELINE_ALPHA}" \
    --condition "l1:${FULL_L1_DIR%/experiment}/csv2_evaluation:${L1_ALPHA}" \
    --condition "causal:${FULL_CAUSAL_DIR%/experiment}/csv2_evaluation:${CAUSAL_LOCKED_ALPHA}" \
    --output_path "${FULL_REPORT}" 2>&1 | tee -a "${LOG}"

${PIPELINE} log-run \
    --run-dir "${FULL_ROOT}" \
    --description "D7 full500 trimmed: baseline(1.0) + l1(${L1_ALPHA}) + causal(${CAUSAL_LOCKED_ALPHA})"

echo "=== D7 causal pilot (trimmed) complete ===" | tee -a "${LOG}"
echo "Report: ${FULL_REPORT}" | tee -a "${LOG}"

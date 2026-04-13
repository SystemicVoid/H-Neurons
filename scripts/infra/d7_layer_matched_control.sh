#!/usr/bin/env bash
# Post-hoc D7 layer-matched random-head control runner.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="D7 layer-matched random-head control" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

PIPELINE="uv run python -m scripts.lib.pipeline"
SEED="${D7_SEED:-42}"
MODEL_PATH="${D7_MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${D7_DEVICE_MAP:-cuda:0}"
RESUME="${D7_RESUME:-1}"
AUTO_SEED2="${D7_AUTO_SEED2:-0}"
SEED2_MAX_GENERATE_SECONDS="${D7_SEED2_MAX_GENERATE_SECONDS:-21600}"

MANIFEST_DIR="data/manifests"
FULL_IDS="${MANIFEST_DIR}/jbb_d7_full_harmful500_seed${SEED}.json"
CAUSAL_ARTIFACT="data/contrastive/refusal/iti_refusal_causal_d7/iti_heads.pt"
RUN_ROOT="data/gemma3_4b/intervention/jailbreak_d7/full500_canonical"
RANDOM_ROOT="${RUN_ROOT}/causal_random_head_layer_matched"
REPORT_PATH="${RUN_ROOT}/d7_csv2_report_layer_matched_control.json"
LOG="logs/d7_layer_matched_control_$(date +%Y%m%d_%H%M%S).log"

JAILBREAK_DECODE_ARGS=(
    --jailbreak_do_sample true
    --jailbreak_temperature 0.7
    --jailbreak_top_k 20
    --jailbreak_top_p 0.8
    --max_new_tokens 5000
)
RUN_PROFILE_ARGS=(--run_profile canonical)

run_cmd() {
    echo "+ $*" | tee -a "${LOG}" >&2
    "$@"
}

alpha_label() {
    local alpha="$1"
    uv run python - <<'PY' "${alpha}"
from scripts.utils import format_alpha_label
import sys

print(format_alpha_label(float(sys.argv[1])))
PY
}

seed_result_allows_auto_seed2() {
    local result="$1"
    if [[ "${result}" == "SKIPPED" ]]; then
        return 0
    fi
    if [[ "${result}" =~ ^[0-9]+$ ]]; then
        [[ "${result}" -le "${SEED2_MAX_GENERATE_SECONDS}" ]]
        return
    fi
    echo "FATAL: unexpected seed runtime token ${result@Q}" >&2
    exit 1
}

assert_dir_ready() {
    local dir="$1"
    mkdir -p "${dir}"
    if [[ "${RESUME}" != "1" ]] && compgen -G "${dir}/alpha_*.jsonl" >/dev/null; then
        echo "FATAL: output collision at ${dir}" >&2
        echo "Archive existing run directory before re-running (set D7_RESUME=1 to resume)." >&2
        exit 1
    fi
}

run_judge_and_csv2() {
    local dir="$1"
    local alpha="$2"
    run_cmd uv run python scripts/evaluate_intervention.py \
        --benchmark jailbreak \
        --input_dir "${dir}" \
        --alphas "${alpha}" \
        --api-mode batch 2>&1 | tee -a "${LOG}"
    run_cmd uv run python scripts/evaluate_csv2.py \
        --input_dir "${dir}" \
        --output_dir "${dir%/experiment}/csv2_evaluation" \
        --alphas "${alpha}" \
        --api-mode batch 2>&1 | tee -a "${LOG}"
}

maybe_run_seed() {
    local seed="$1"
    local alpha="$2"
    local random_dir="${RANDOM_ROOT}/seed_${seed}/experiment"
    assert_dir_ready "${random_dir}"
    if ! ${PIPELINE} check-stage --output-dir "${random_dir}" --manifest "${FULL_IDS}" --alphas "${alpha}"; then
        local generation_seconds=0
        SECONDS=0
        run_cmd uv run python scripts/run_intervention.py \
            --benchmark jailbreak \
            --model_path "${MODEL_PATH}" \
            --device_map "${DEVICE_MAP}" \
            --output_dir "${random_dir}" \
            --sample_manifest "${FULL_IDS}" \
            --intervention_mode iti_head \
            --iti_head_path "${CAUSAL_ARTIFACT}" \
            --iti_family refusal_causal \
            --iti_k 20 \
            --iti_selection_strategy layer_matched_random \
            --iti_random_seed "${seed}" \
            --alphas "${alpha}" \
            "${RUN_PROFILE_ARGS[@]}" \
            "${JAILBREAK_DECODE_ARGS[@]}" 2>&1 | tee -a "${LOG}" >&2
        generation_seconds="${SECONDS}"
        echo "seed_${seed}_generation_seconds=${generation_seconds}" | tee -a "${LOG}" >&2
        echo "${generation_seconds}"
    else
        echo "Stage: layer-matched seed=${seed} generation complete; skipping" | tee -a "${LOG}" >&2
        echo "SKIPPED"
    fi
}

echo "=== D7 layer-matched control ===" | tee -a "${LOG}"
echo "seed=${SEED} model=${MODEL_PATH} device_map=${DEVICE_MAP} resume=${RESUME}" | tee -a "${LOG}"
${PIPELINE} gpu-preflight 2>&1 | tee -a "${LOG}" || true

if [[ ! -f "${FULL_IDS}" ]]; then
    run_cmd uv run python scripts/build_d7_jbb_manifests.py \
        --seed "${SEED}" \
        --output_dir "${MANIFEST_DIR}" 2>&1 | tee -a "${LOG}"
fi

CAUSAL_LOCK_JSON="data/gemma3_4b/intervention/jailbreak_d7/pilot100_canonical/causal_lock.json"
CAUSAL_LOCKED_ALPHA=$(uv run python - <<'PY' "${CAUSAL_LOCK_JSON}"
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["selected_alpha"])
PY
)
CAUSAL_LOCKED_ALPHA_LABEL="$(alpha_label "${CAUSAL_LOCKED_ALPHA}")"
echo "Using causal locked alpha=${CAUSAL_LOCKED_ALPHA}" | tee -a "${LOG}"

seed1_generation_result="$(maybe_run_seed 1 "${CAUSAL_LOCKED_ALPHA}")"
run_judge_and_csv2 "${RANDOM_ROOT}/seed_1/experiment" "${CAUSAL_LOCKED_ALPHA}"
${PIPELINE} log-run \
    --run-dir "${RANDOM_ROOT}/seed_1/experiment" \
    --description "D7 layer-matched random-head seed 1 at alpha ${CAUSAL_LOCKED_ALPHA}"

if [[ "${AUTO_SEED2}" == "1" ]]; then
    if seed_result_allows_auto_seed2 "${seed1_generation_result}"; then
        maybe_run_seed 2 "${CAUSAL_LOCKED_ALPHA}" >/dev/null
        run_judge_and_csv2 "${RANDOM_ROOT}/seed_2/experiment" "${CAUSAL_LOCKED_ALPHA}"
        ${PIPELINE} log-run \
            --run-dir "${RANDOM_ROOT}/seed_2/experiment" \
            --description "D7 layer-matched random-head seed 2 at alpha ${CAUSAL_LOCKED_ALPHA}"
    else
        echo "Skipping seed 2 auto-run; seed 1 generation was skipped or exceeded runtime gate" | tee -a "${LOG}"
    fi
fi

condition_args=(
    --condition "l1:${RUN_ROOT}/l1_neuron/csv2_evaluation:3.0"
    --condition "causal:${RUN_ROOT}/causal_locked/csv2_evaluation:${CAUSAL_LOCKED_ALPHA}"
    --condition "random_layer_seed1:${RANDOM_ROOT}/seed_1/csv2_evaluation:${CAUSAL_LOCKED_ALPHA}"
)
if [[ -f "${RANDOM_ROOT}/seed_2/csv2_evaluation/alpha_${CAUSAL_LOCKED_ALPHA_LABEL}.jsonl" ]]; then
    condition_args+=(
        --condition "random_layer_seed2:${RANDOM_ROOT}/seed_2/csv2_evaluation:${CAUSAL_LOCKED_ALPHA}"
    )
fi

run_cmd uv run python scripts/report_d7_csv2.py \
    --baseline_dir "${RUN_ROOT}/baseline_noop/csv2_evaluation" \
    --baseline_alpha 1.0 \
    "${condition_args[@]}" \
    --output_path "${REPORT_PATH}" 2>&1 | tee -a "${LOG}"

echo "=== D7 layer-matched control complete ===" | tee -a "${LOG}"
echo "Report: ${REPORT_PATH}" | tee -a "${LOG}"

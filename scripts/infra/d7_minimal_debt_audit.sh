#!/usr/bin/env bash
# Minimal D7 debt audit: BioASQ slice, benign JBB slice, and harmful structure summary.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="D7 minimal debt audit" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

PIPELINE="uv run python -m scripts.lib.pipeline"
SEED="${D7_SEED:-42}"
MODEL_PATH="${D7_MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${D7_DEVICE_MAP:-cuda:0}"
RESUME="${D7_RESUME:-1}"
BIOASQ_N="${D7_DEBT_BIOASQ_N:-100}"

MANIFEST_DIR="data/manifests"
BIOASQ_MANIFEST="${MANIFEST_DIR}/bioasq_factoid_n${BIOASQ_N}_seed${SEED}.json"
BENIGN_MANIFEST="${MANIFEST_DIR}/jbb_d7_pilot_benign100_seed${SEED}.json"
CAUSAL_ARTIFACT="data/contrastive/refusal/iti_refusal_causal_d7/iti_heads.pt"
RUN_ROOT="data/gemma3_4b/intervention/jailbreak_d7/debt_audit"
BIOASQ_BASELINE_DIR="${RUN_ROOT}/bioasq_baseline/experiment"
BIOASQ_CAUSAL_DIR="${RUN_ROOT}/bioasq_causal_locked/experiment"
BENIGN_BASELINE_DIR="${RUN_ROOT}/jbb_benign_baseline/experiment"
BENIGN_CAUSAL_DIR="${RUN_ROOT}/jbb_benign_causal_locked/experiment"
REPORT_PATH="${RUN_ROOT}/d7_debt_audit_report.json"
LOG="logs/d7_minimal_debt_audit_$(date +%Y%m%d_%H%M%S).log"

run_cmd() {
    echo "+ $*"
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

assert_dir_ready() {
    local dir="$1"
    mkdir -p "${dir}"
    if [[ "${RESUME}" != "1" ]] && compgen -G "${dir}/alpha_*.jsonl" >/dev/null; then
        echo "FATAL: output collision at ${dir}" >&2
        echo "Archive existing run directory before re-running (set D7_RESUME=1 to resume)." >&2
        exit 1
    fi
}

echo "=== D7 minimal debt audit ===" | tee -a "${LOG}"
echo "seed=${SEED} model=${MODEL_PATH} device_map=${DEVICE_MAP} resume=${RESUME}" | tee -a "${LOG}"
${PIPELINE} gpu-preflight 2>&1 | tee -a "${LOG}" || true

if [[ ! -f "${BENIGN_MANIFEST}" ]]; then
    run_cmd uv run python scripts/build_d7_jbb_manifests.py \
        --seed "${SEED}" \
        --output_dir "${MANIFEST_DIR}" 2>&1 | tee -a "${LOG}"
fi

if [[ ! -f "${BIOASQ_MANIFEST}" ]]; then
    run_cmd uv run python scripts/build_bioasq_manifest.py \
        --seed "${SEED}" \
        --n "${BIOASQ_N}" \
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

assert_dir_ready "${BIOASQ_BASELINE_DIR}"
assert_dir_ready "${BIOASQ_CAUSAL_DIR}"
assert_dir_ready "${BENIGN_BASELINE_DIR}"
assert_dir_ready "${BENIGN_CAUSAL_DIR}"

if ! ${PIPELINE} check-stage --output-dir "${BIOASQ_BASELINE_DIR}" --manifest "${BIOASQ_MANIFEST}" --alphas 0.0; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark bioasq \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${BIOASQ_BASELINE_DIR}" \
        --sample_manifest "${BIOASQ_MANIFEST}" \
        --intervention_mode iti_head \
        --iti_head_path "${CAUSAL_ARTIFACT}" \
        --iti_family refusal_causal \
        --iti_k 20 \
        --alphas 0.0 2>&1 | tee -a "${LOG}"
fi

if ! ${PIPELINE} check-stage --output-dir "${BIOASQ_CAUSAL_DIR}" --manifest "${BIOASQ_MANIFEST}" --alphas "${CAUSAL_LOCKED_ALPHA}"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark bioasq \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${BIOASQ_CAUSAL_DIR}" \
        --sample_manifest "${BIOASQ_MANIFEST}" \
        --intervention_mode iti_head \
        --iti_head_path "${CAUSAL_ARTIFACT}" \
        --iti_family refusal_causal \
        --iti_k 20 \
        --alphas "${CAUSAL_LOCKED_ALPHA}" 2>&1 | tee -a "${LOG}"
fi

if ! ${PIPELINE} check-stage --output-dir "${BENIGN_BASELINE_DIR}" --manifest "${BENIGN_MANIFEST}" --alphas 0.0; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak_benign \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${BENIGN_BASELINE_DIR}" \
        --sample_manifest "${BENIGN_MANIFEST}" \
        --intervention_mode iti_head \
        --iti_head_path "${CAUSAL_ARTIFACT}" \
        --iti_family refusal_causal \
        --iti_k 20 \
        --alphas 0.0 \
        --run_profile canonical \
        --jailbreak_do_sample true \
        --jailbreak_temperature 0.7 \
        --jailbreak_top_k 20 \
        --jailbreak_top_p 0.8 \
        --max_new_tokens 5000 2>&1 | tee -a "${LOG}"
fi

if ! ${PIPELINE} check-stage --output-dir "${BENIGN_CAUSAL_DIR}" --manifest "${BENIGN_MANIFEST}" --alphas "${CAUSAL_LOCKED_ALPHA}"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak_benign \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${BENIGN_CAUSAL_DIR}" \
        --sample_manifest "${BENIGN_MANIFEST}" \
        --intervention_mode iti_head \
        --iti_head_path "${CAUSAL_ARTIFACT}" \
        --iti_family refusal_causal \
        --iti_k 20 \
        --alphas "${CAUSAL_LOCKED_ALPHA}" \
        --run_profile canonical \
        --jailbreak_do_sample true \
        --jailbreak_temperature 0.7 \
        --jailbreak_top_k 20 \
        --jailbreak_top_p 0.8 \
        --max_new_tokens 5000 2>&1 | tee -a "${LOG}"
fi

run_cmd uv run python scripts/evaluate_jailbreak_benign.py \
    --input_dir "${BENIGN_BASELINE_DIR}" \
    --alphas 0.0 \
    --api-mode batch 2>&1 | tee -a "${LOG}"

run_cmd uv run python scripts/evaluate_jailbreak_benign.py \
    --input_dir "${BENIGN_CAUSAL_DIR}" \
    --alphas "${CAUSAL_LOCKED_ALPHA}" \
    --api-mode batch 2>&1 | tee -a "${LOG}"

random_dir_arg=()
random_csv2_arg=()
RANDOM_EXPERIMENT_ALPHA="data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_1/experiment/alpha_${CAUSAL_LOCKED_ALPHA_LABEL}.jsonl"
RANDOM_CSV2_ALPHA="data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_1/csv2_evaluation/alpha_${CAUSAL_LOCKED_ALPHA_LABEL}.jsonl"
if [[ -f "${RANDOM_EXPERIMENT_ALPHA}" && -f "${RANDOM_CSV2_ALPHA}" ]]; then
    random_dir_arg=(
        --harmful_random_dir "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_1/experiment"
        --harmful_random_alpha "${CAUSAL_LOCKED_ALPHA}"
    )
    random_csv2_arg=(
        --harmful_random_csv2_dir "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_1/csv2_evaluation"
    )
elif [[ -f "${RANDOM_EXPERIMENT_ALPHA}" || -f "${RANDOM_CSV2_ALPHA}" ]]; then
    echo "Skipping random-control debt-audit panel; seed 1 random control is only partially complete" | tee -a "${LOG}"
fi

run_cmd uv run python scripts/report_d7_debt_audit.py \
    --bioasq_baseline_dir "${BIOASQ_BASELINE_DIR}" \
    --bioasq_baseline_alpha 0.0 \
    --bioasq_causal_dir "${BIOASQ_CAUSAL_DIR}" \
    --bioasq_causal_alpha "${CAUSAL_LOCKED_ALPHA}" \
    --benign_baseline_dir "${BENIGN_BASELINE_DIR}" \
    --benign_baseline_alpha 0.0 \
    --benign_causal_dir "${BENIGN_CAUSAL_DIR}" \
    --benign_causal_alpha "${CAUSAL_LOCKED_ALPHA}" \
    --harmful_baseline_dir "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/baseline_noop/experiment" \
    --harmful_baseline_alpha 1.0 \
    --harmful_causal_dir "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_locked/experiment" \
    --harmful_causal_alpha "${CAUSAL_LOCKED_ALPHA}" \
    --harmful_baseline_csv2_dir "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/baseline_noop/csv2_evaluation" \
    --harmful_causal_csv2_dir "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_locked/csv2_evaluation" \
    "${random_dir_arg[@]}" \
    "${random_csv2_arg[@]}" \
    --output_path "${REPORT_PATH}" 2>&1 | tee -a "${LOG}"

${PIPELINE} log-run --run-dir "${BIOASQ_CAUSAL_DIR}" --description "D7 debt audit BioASQ causal alpha ${CAUSAL_LOCKED_ALPHA}"
${PIPELINE} log-run --run-dir "${BENIGN_CAUSAL_DIR}" --description "D7 debt audit benign JBB causal alpha ${CAUSAL_LOCKED_ALPHA}"

echo "=== D7 minimal debt audit complete ===" | tee -a "${LOG}"
echo "Report: ${REPORT_PATH}" | tee -a "${LOG}"

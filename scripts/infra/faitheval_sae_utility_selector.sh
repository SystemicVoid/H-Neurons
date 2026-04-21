#!/usr/bin/env bash
set -euo pipefail

ROOT="data/gemma3_4b/intervention/faitheval_sae_utility_selector"
SELECTOR_DIR="${ROOT}/selector"
HELDOUT_DIR="${ROOT}/heldout/utility_selected/experiment"
REPORT_DIR="${ROOT}/report"

MODEL_PATH="${MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-models/sae_detector.pkl}"
CLASSIFIER_SUMMARY="${CLASSIFIER_SUMMARY:-data/gemma3_4b/pipeline/classifier_sae_summary.json}"

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

run_inhibited() {
    local why="$1"
    shift
    systemd-inhibit --what=sleep --why="${why}" "$@"
}

results_summary_exists() {
    local dir="$1"
    [[ -f "${dir}/results.json" ]] && return 0
    compgen -G "${dir}/results.*.json" >/dev/null
}

require_file "scripts/select_faitheval_sae_utility_features.py" "selector script"
require_file "scripts/report_faitheval_sae_utility_selector.py" "report script"
require_file "scripts/run_intervention.py" "intervention script"
require_file "${CLASSIFIER_PATH}" "SAE classifier"
require_file "${CLASSIFIER_SUMMARY}" "SAE classifier summary"

echo "Checking GPU state via nvitop..."
nvitop -1

mkdir -p "${ROOT}"

if [[ ! -f "${SELECTOR_DIR}/selector_summary.json" ]]; then
    run_inhibited "FaithEval SAE utility selector" \
        env PYTHONUNBUFFERED=1 uv run python scripts/select_faitheval_sae_utility_features.py \
            --model_path "${MODEL_PATH}" \
            --device_map "${DEVICE_MAP}" \
            --classifier_path "${CLASSIFIER_PATH}" \
            --classifier_summary "${CLASSIFIER_SUMMARY}" \
            --output_dir "${SELECTOR_DIR}"
else
    echo "Skipping selector stage; found ${SELECTOR_DIR}/selector_summary.json"
fi

if ! results_summary_exists "${HELDOUT_DIR}"; then
    run_inhibited "FaithEval SAE utility held-out run" \
        env PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
            --model_path "${MODEL_PATH}" \
            --device_map "${DEVICE_MAP}" \
            --benchmark faitheval \
            --prompt_style anti_compliance \
            --intervention_mode sae \
            --sae_feature_manifest "${SELECTOR_DIR}/utility_selected_features.json" \
            --sae_steering_mode delta_only \
            --alphas 0.0 1.0 \
            --sample_manifest "${SELECTOR_DIR}/test_manifest.json" \
            --output_dir "${HELDOUT_DIR}"
else
    echo "Skipping held-out stage; found existing results summary in ${HELDOUT_DIR}"
fi

if [[ ! -f "${REPORT_DIR}/heldout_summary.json" ]]; then
    env PYTHONUNBUFFERED=1 uv run python scripts/report_faitheval_sae_utility_selector.py \
        --selector_dir "${SELECTOR_DIR}" \
        --heldout_dir "${HELDOUT_DIR}" \
        --output_dir "${ROOT}"
else
    echo "Skipping report stage; found ${REPORT_DIR}/heldout_summary.json"
fi

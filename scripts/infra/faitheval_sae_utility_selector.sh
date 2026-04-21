#!/usr/bin/env bash
set -euo pipefail

ROOT="data/gemma3_4b/intervention/faitheval_sae_utility_selector"
SELECTOR_DIR="${ROOT}/selector"
HELDOUT_ROOT="${ROOT}/heldout"
REPORT_DIR="${ROOT}/report"

MODEL_PATH="${MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-models/sae_detector.pkl}"
CLASSIFIER_SUMMARY="${CLASSIFIER_SUMMARY:-data/gemma3_4b/pipeline/classifier_sae_summary.json}"

BENCHMARKS=(
    "faitheval"
    "faitheval_mc_logprob"
)
FAMILIES=(
    "noop"
    "readout_selected"
    "utility_selected"
    "matched_random_seed_0"
    "matched_random_seed_1"
    "matched_random_seed_2"
)

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

selector_stage_complete() {
    local required=(
        "validation_manifest.json"
        "test_manifest.json"
        "candidate_pool.json"
        "utility_selected_features.json"
        "readout_selected_features.json"
        "matched_random_seed_0_features.json"
        "matched_random_seed_1_features.json"
        "matched_random_seed_2_features.json"
        "selector_summary.json"
    )
    local rel_path
    for rel_path in "${required[@]}"; do
        [[ -f "${SELECTOR_DIR}/${rel_path}" ]] || return 1
    done
    return 0
}

expected_alpha_for_family() {
    local family="$1"
    case "${family}" in
        noop)
            echo "1.0"
            ;;
        *)
            echo "0.0"
            ;;
    esac
}

manifest_for_family() {
    local family="$1"
    case "${family}" in
        noop)
            echo "${SELECTOR_DIR}/candidate_pool.json"
            ;;
        *)
            echo "${SELECTOR_DIR}/${family}_features.json"
            ;;
    esac
}

heldout_dir_for() {
    local benchmark="$1"
    local family="$2"
    echo "${HELDOUT_ROOT}/${benchmark}/${family}/experiment"
}

heldout_stage_complete() {
    local benchmark="$1"
    local family="$2"
    local dir
    dir="$(heldout_dir_for "${benchmark}" "${family}")"
    local alpha
    alpha="$(expected_alpha_for_family "${family}")"
    [[ -f "${dir}/alpha_${alpha}.jsonl" ]] || return 1
    results_summary_exists "${dir}"
}

require_file "scripts/select_faitheval_sae_utility_features.py" "selector script"
require_file "scripts/report_faitheval_sae_utility_selector.py" "report script"
require_file "scripts/run_intervention.py" "intervention script"
require_file "${CLASSIFIER_PATH}" "SAE classifier"
require_file "${CLASSIFIER_SUMMARY}" "SAE classifier summary"

echo "Checking GPU state via nvitop..."
nvitop -1

mkdir -p "${ROOT}"

if ! selector_stage_complete; then
    run_inhibited "FaithEval SAE utility selector" \
        env PYTHONUNBUFFERED=1 uv run python scripts/select_faitheval_sae_utility_features.py \
            --model_path "${MODEL_PATH}" \
            --device_map "${DEVICE_MAP}" \
            --classifier_path "${CLASSIFIER_PATH}" \
            --classifier_summary "${CLASSIFIER_SUMMARY}" \
            --output_dir "${SELECTOR_DIR}"
else
    echo "Skipping selector stage; found complete selector bundle in ${SELECTOR_DIR}"
fi

benchmark=""
family=""
for benchmark in "${BENCHMARKS[@]}"; do
    for family in "${FAMILIES[@]}"; do
        dir="$(heldout_dir_for "${benchmark}" "${family}")"
        alpha="$(expected_alpha_for_family "${family}")"
        manifest_path="$(manifest_for_family "${family}")"
        if ! heldout_stage_complete "${benchmark}" "${family}"; then
            extra_args=()
            if [[ "${benchmark}" == "faitheval" ]]; then
                extra_args+=(--prompt_style "anti_compliance")
            fi
            run_inhibited "FaithEval SAE utility held-out run ${benchmark}/${family}" \
                env PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
                    --model_path "${MODEL_PATH}" \
                    --device_map "${DEVICE_MAP}" \
                    --benchmark "${benchmark}" \
                    "${extra_args[@]}" \
                    --intervention_mode sae \
                    --sae_feature_manifest "${manifest_path}" \
                    --sae_steering_mode delta_only \
                    --alphas "${alpha}" \
                    --sample_manifest "${SELECTOR_DIR}/test_manifest.json" \
                    --output_dir "${dir}"
        else
            echo "Skipping held-out stage; found ${benchmark}/${family} outputs in ${dir}"
        fi
    done
done

if [[ ! -f "${REPORT_DIR}/heldout_summary.json" ]]; then
    env PYTHONUNBUFFERED=1 uv run python scripts/report_faitheval_sae_utility_selector.py \
        --selector_dir "${SELECTOR_DIR}" \
        --heldout_root "${HELDOUT_ROOT}" \
        --output_dir "${ROOT}"
else
    echo "Skipping report stage; found ${REPORT_DIR}/heldout_summary.json"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT="data/gemma3_4b/intervention/faitheval_sae_utility_selector"
SELECTOR_DIR="${ROOT}/selector"
HELDOUT_ROOT="${ROOT}/heldout"
REPORT_DIR="${ROOT}/report_augment"
DID_GENERATE_DATA=0

MODEL_PATH="${MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-models/sae_detector.pkl}"

BENCHMARKS=(
    "faitheval"
    "faitheval_anti_compliance_margin"
)
FAMILIES=(
    "utility_positive_selected"
    "matched_random_positive_seed_0"
    "matched_random_positive_seed_1"
    "matched_random_positive_seed_2"
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

artifact_is_fresh() {
    local artifact="$1"
    shift
    [[ -e "${artifact}" ]] || return 1
    local dep=""
    for dep in "$@"; do
        [[ -e "${dep}" ]] || return 1
        [[ "${dep}" -nt "${artifact}" ]] && return 1
    done
    return 0
}

fresh_results_summary_exists() {
    local dir="$1"
    shift
    local result_path=""
    if [[ -f "${dir}/results.json" ]] && artifact_is_fresh "${dir}/results.json" "$@"; then
        return 0
    fi
    while IFS= read -r result_path; do
        if artifact_is_fresh "${result_path}" "$@"; then
            return 0
        fi
    done < <(compgen -G "${dir}/results.*.json" || true)
    return 1
}

derive_stage_complete() {
    local required=(
        "utility_positive_selected_features.json"
        "matched_random_positive_seed_0_features.json"
        "matched_random_positive_seed_1_features.json"
        "matched_random_positive_seed_2_features.json"
        "utility_positive_summary.json"
    )
    local deps=(
        "scripts/derive_utility_positive_selector.py"
        "${SELECTOR_DIR}/utility_scores.jsonl"
        "${SELECTOR_DIR}/feature_stats.json"
        "${SELECTOR_DIR}/selector_summary.json"
        "${CLASSIFIER_PATH}"
    )
    local rel_path
    for rel_path in "${required[@]}"; do
        local artifact="${SELECTOR_DIR}/${rel_path}"
        [[ -f "${artifact}" ]] || return 1
        artifact_is_fresh "${artifact}" "${deps[@]}" || return 1
    done
    return 0
}

manifest_for_family() {
    local family="$1"
    echo "${SELECTOR_DIR}/${family}_features.json"
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
    local alpha_path="${dir}/alpha_0.0.jsonl"
    local manifest_path
    manifest_path="$(manifest_for_family "${family}")"
    local deps=(
        "scripts/run_intervention.py"
        "${manifest_path}"
        "${SELECTOR_DIR}/test_manifest.json"
    )
    [[ -f "${alpha_path}" ]] || return 1
    artifact_is_fresh "${alpha_path}" "${deps[@]}" || return 1
    env PYTHONUNBUFFERED=1 uv run python -m scripts.lib.pipeline check-stage \
        --output-dir "${dir}" \
        --manifest "${SELECTOR_DIR}/test_manifest.json" \
        --alphas 0.0 >/dev/null || return 1
    fresh_results_summary_exists "${dir}" "${deps[@]}"
}

report_stage_complete() {
    local summary_path="${REPORT_DIR}/augment_heldout_summary.json"
    local audit_path="${REPORT_DIR}/augment_audit_note.md"
    [[ -f "${summary_path}" ]] || return 1
    [[ -f "${audit_path}" ]] || return 1

    local deps=(
        "scripts/report_faitheval_sae_utility_positive_augment.py"
        "${SELECTOR_DIR}/selector_summary.json"
        "${SELECTOR_DIR}/utility_positive_summary.json"
        "${SELECTOR_DIR}/test_manifest.json"
    )
    local benchmark=""
    local family=""
    for benchmark in "${BENCHMARKS[@]}"; do
        for family in "${FAMILIES[@]}"; do
            deps+=("$(heldout_dir_for "${benchmark}" "${family}")/alpha_0.0.jsonl")
        done
        deps+=("${HELDOUT_ROOT}/${benchmark}/noop/experiment/alpha_1.0.jsonl")
    done

    artifact_is_fresh "${summary_path}" "${deps[@]}" || return 1
    artifact_is_fresh "${audit_path}" "${deps[@]}"
}

require_file "scripts/derive_utility_positive_selector.py" "derive script"
require_file "scripts/report_faitheval_sae_utility_positive_augment.py" "augment report script"
require_file "scripts/run_intervention.py" "intervention script"
require_file "scripts/lib/pipeline.py" "pipeline guard library"
require_file "${SELECTOR_DIR}/utility_scores.jsonl" "utility scores (Phase 1 output)"
require_file "${SELECTOR_DIR}/feature_stats.json" "feature stats (Phase 1 output)"
require_file "${SELECTOR_DIR}/selector_summary.json" "selector summary (Phase 1 output)"
require_file "${SELECTOR_DIR}/test_manifest.json" "test manifest (Phase 1 output)"
require_file "${SELECTOR_DIR}/utility_selected_features.json" "utility_selected manifest (Phase 1 output)"
require_file "${CLASSIFIER_PATH}" "SAE classifier"

for benchmark in "${BENCHMARKS[@]}"; do
    require_file "${HELDOUT_ROOT}/${benchmark}/noop/experiment/alpha_1.0.jsonl" \
        "Phase 1 noop α=1.0 reference for ${benchmark}"
done

echo "Checking GPU state via nvitop..."
nvitop -1

mkdir -p "${ROOT}"

if ! derive_stage_complete; then
    run_inhibited "FaithEval SAE utility-positive manifest derivation" \
        env PYTHONUNBUFFERED=1 uv run python scripts/derive_utility_positive_selector.py \
            --selector_dir "${SELECTOR_DIR}" \
            --classifier_path "${CLASSIFIER_PATH}"
    DID_GENERATE_DATA=1
else
    echo "Skipping derive stage; found complete utility-positive manifests in ${SELECTOR_DIR}"
fi

benchmark=""
family=""
for benchmark in "${BENCHMARKS[@]}"; do
    for family in "${FAMILIES[@]}"; do
        dir="$(heldout_dir_for "${benchmark}" "${family}")"
        manifest_path="$(manifest_for_family "${family}")"
        if ! heldout_stage_complete "${benchmark}" "${family}"; then
            extra_args=()
            if [[ "${benchmark}" == "faitheval" || "${benchmark}" == "faitheval_anti_compliance_margin" ]]; then
                extra_args+=(--prompt_style "anti_compliance")
            fi
            run_inhibited "FaithEval SAE utility-positive held-out run ${benchmark}/${family}" \
                env PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
                    --model_path "${MODEL_PATH}" \
                    --device_map "${DEVICE_MAP}" \
                    --benchmark "${benchmark}" \
                    "${extra_args[@]}" \
                    --intervention_mode sae \
                    --sae_feature_manifest "${manifest_path}" \
                    --sae_steering_mode delta_only \
                    --alphas 0.0 \
                    --sample_manifest "${SELECTOR_DIR}/test_manifest.json" \
                    --output_dir "${dir}"
            DID_GENERATE_DATA=1
        else
            echo "Skipping held-out stage; found ${benchmark}/${family} outputs in ${dir}"
        fi
    done
done

if ! report_stage_complete; then
    env PYTHONUNBUFFERED=1 uv run python scripts/report_faitheval_sae_utility_positive_augment.py \
        --selector_dir "${SELECTOR_DIR}" \
        --heldout_root "${HELDOUT_ROOT}" \
        --output_dir "${REPORT_DIR}"
else
    echo "Skipping augment report stage; found fresh outputs in ${REPORT_DIR}"
fi

if [[ "${DID_GENERATE_DATA}" -eq 1 ]]; then
    env PYTHONUNBUFFERED=1 uv run python -m scripts.lib.pipeline log-run \
        --run-dir "${REPORT_DIR}" \
        --description "FaithEval SAE utility-positive augment bundle (k=154 positive-only, 3 layer-matched zero-weight seeds × 2 benchmarks, α=0.0; noop α=1.0 reused from Phase 1)" \
        --key-files "selector/utility_positive_*.json, selector/matched_random_positive_*.json, heldout/*/utility_positive_selected/experiment/alpha_0.0.jsonl, heldout/*/matched_random_positive_*/experiment/alpha_0.0.jsonl, report_augment/augment_heldout_summary.json, report_augment/augment_audit_note.md, *.provenance.json"
fi

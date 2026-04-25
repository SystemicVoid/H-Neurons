#!/usr/bin/env bash
set -euo pipefail

ROOT="data/gemma3_4b/intervention/faitheval_sae_utility_selector"
SELECTOR_DIR="${ROOT}/selector"
HELDOUT_ROOT="${ROOT}/heldout"
REPORT_DIR="${ROOT}/report"
SENTINEL_DIR="${ROOT}/sentinels"
DID_GENERATE_DATA=0

MODEL_PATH="${MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-models/sae_detector.pkl}"
CLASSIFIER_SUMMARY="${CLASSIFIER_SUMMARY:-data/gemma3_4b/pipeline/classifier_sae_summary.json}"
N_RANDOM_SEEDS="${N_RANDOM_SEEDS:-10}"

BENCHMARKS=(
    "faitheval"
    "faitheval_anti_compliance_margin"
    "faitheval_answer_span_margin"
)
ANSWER_SPAN_FAMILIES=(
    "noop"
    "readout_selected"
    "utility_selected"
    "answer_span_selected"
)
EXPANDED_FAMILIES=(
    "noop"
    "readout_selected"
    "utility_selected"
    "answer_span_selected"
)
for ((seed=0; seed<N_RANDOM_SEEDS; seed++)); do
    ANSWER_SPAN_FAMILIES+=("matched_random_answer_span_seed_${seed}")
    EXPANDED_FAMILIES+=("matched_random_seed_${seed}")
    EXPANDED_FAMILIES+=("matched_random_answer_span_seed_${seed}")
done
EXPANDED_FAMILIES+=("matched_zero_dead")

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

selector_stage_complete() {
    local required=(
        "validation_manifest.json"
        "test_manifest.json"
        "candidate_pool.json"
        "feature_stats.json"
        "full_sequence_feature_stats.json"
        "utility_scores.jsonl"
        "answer_span_scores.jsonl"
        "utility_selected_features.json"
        "answer_span_selected_features.json"
        "readout_selected_features.json"
        "matched_zero_dead_features.json"
        "selector_summary.json"
    )
    local seed=""
    for ((seed=0; seed<N_RANDOM_SEEDS; seed++)); do
        required+=("matched_random_seed_${seed}_features.json")
        required+=("matched_random_answer_span_seed_${seed}_features.json")
    done
    local rel_path
    for rel_path in "${required[@]}"; do
        local artifact="${SELECTOR_DIR}/${rel_path}"
        [[ -f "${artifact}" ]] || return 1
    done
    env PYTHONUNBUFFERED=1 uv run python - "${SELECTOR_DIR}" "${N_RANDOM_SEEDS}" <<'PY'
import json
import sys
from pathlib import Path

selector_dir = Path(sys.argv[1])
n_random_seeds = int(sys.argv[2])


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


summary = load_json(selector_dir / "selector_summary.json")
state = load_json(selector_dir / "selector_scoring_state.json")
selector_scoring = summary.get("selector_scoring", {})
if selector_scoring.get("input_hash") != state.get("input_hash"):
    raise SystemExit("selector scoring cache hash mismatch")

candidate_features = load_json(selector_dir / "candidate_pool.json")["features"]
candidate_flat_idxs = {int(feature["flat_idx"]) for feature in candidate_features}
if len(candidate_flat_idxs) != int(summary["candidate_pool"]["n_features"]):
    raise SystemExit("candidate pool count mismatch")

score_requirements = {
    "utility_scores.jsonl": {
        "flat_idx",
        "selector_score",
        "selector_score_std",
        "selector_score_sum",
    },
    "answer_span_scores.jsonl": {
        "flat_idx",
        "selector_score",
        "selector_score_std",
        "selector_score_sum",
        "selector_score_full_span",
        "selector_score_full_span_std",
        "selector_score_full_span_sum",
    },
}
for filename, required_fields in score_requirements.items():
    rows = load_jsonl(selector_dir / filename)
    row_flat_idxs = {int(row["flat_idx"]) for row in rows}
    if row_flat_idxs != candidate_flat_idxs:
        raise SystemExit(f"{filename} does not cover the candidate pool")
    if any(not required_fields.issubset(row.keys()) for row in rows):
        raise SystemExit(f"{filename} rows missing required fields")

families = summary["families"]
for family in ("utility_selected", "answer_span_selected", "readout_selected"):
    manifest = load_json(selector_dir / f"{family}_features.json")
    if len(manifest["features"]) != int(families[family]["k"]):
        raise SystemExit(f"{family} manifest count mismatch")

if len(load_json(selector_dir / "matched_zero_dead_features.json")["features"]) != 1:
    raise SystemExit("matched_zero_dead manifest count mismatch")

random_controls = summary["matched_random_controls"]["seed_families"]
answer_span_random_controls = summary["matched_random_answer_span_controls"][
    "seed_families"
]
for seed in range(n_random_seeds):
    family = f"matched_random_seed_{seed}"
    manifest = load_json(selector_dir / f"{family}_features.json")
    if len(manifest["features"]) != int(random_controls[family]["k"]):
        raise SystemExit(f"{family} manifest count mismatch")

    family = f"matched_random_answer_span_seed_{seed}"
    manifest = load_json(selector_dir / f"{family}_features.json")
    if len(manifest["features"]) != int(answer_span_random_controls[family]["k"]):
        raise SystemExit(f"{family} manifest count mismatch")

for manifest_name in ("validation_manifest.json", "test_manifest.json"):
    manifest = load_json(selector_dir / manifest_name)
    if int(manifest.get("n_ids", 0)) <= 0:
        raise SystemExit(f"{manifest_name} is empty")
PY
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

families_for_benchmark() {
    local benchmark="$1"
    if [[ "${benchmark}" == "faitheval_answer_span_margin" ]]; then
        printf '%s\n' "${ANSWER_SPAN_FAMILIES[@]}"
    else
        printf '%s\n' "${EXPANDED_FAMILIES[@]}"
    fi
}

heldout_stage_complete() {
    local benchmark="$1"
    local family="$2"
    local dir
    dir="$(heldout_dir_for "${benchmark}" "${family}")"
    local alpha
    alpha="$(expected_alpha_for_family "${family}")"
    local alpha_path="${dir}/alpha_${alpha}.jsonl"
    [[ -f "${alpha_path}" ]] || return 1
    env PYTHONUNBUFFERED=1 uv run python -m scripts.lib.pipeline check-stage \
        --output-dir "${dir}" \
        --manifest "${SELECTOR_DIR}/test_manifest.json" \
        --alphas "${alpha}" >/dev/null || return 1
    results_summary_exists "${dir}"
}

report_stage_complete() {
    local summary_path="${REPORT_DIR}/heldout_summary.json"
    local audit_path="${REPORT_DIR}/audit_note.md"
    [[ -f "${summary_path}" ]] || return 1
    [[ -f "${audit_path}" ]] || return 1

    local deps=(
        "scripts/report_faitheval_sae_utility_selector.py"
        "${SELECTOR_DIR}/selector_summary.json"
        "${SELECTOR_DIR}/test_manifest.json"
    )
    local benchmark=""
    local family=""
    for benchmark in "${BENCHMARKS[@]}"; do
        while IFS= read -r family; do
            deps+=("$(heldout_dir_for "${benchmark}" "${family}")/alpha_$(expected_alpha_for_family "${family}").jsonl")
        done < <(families_for_benchmark "${benchmark}")
    done

    artifact_is_fresh "${summary_path}" "${deps[@]}" || return 1
    artifact_is_fresh "${audit_path}" "${deps[@]}"
}

require_file "scripts/select_faitheval_sae_utility_features.py" "selector script"
require_file "scripts/report_faitheval_sae_utility_selector.py" "report script"
require_file "scripts/run_intervention.py" "intervention script"
require_file "scripts/lib/pipeline.py" "pipeline guard library"
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
            --n_random_seeds "${N_RANDOM_SEEDS}" \
            --output_dir "${SELECTOR_DIR}"
    DID_GENERATE_DATA=1
else
    echo "Skipping selector stage; found complete selector bundle in ${SELECTOR_DIR}"
fi

if env PYTHONUNBUFFERED=1 uv run python -m scripts.lib.pipeline check-sentinel \
    --dir "${SENTINEL_DIR}" --name "stop_after_selector"; then
    echo "Sentinel stop_after_selector detected. Stopping cleanly before held-out phase." >&2
    echo "Remove ${SENTINEL_DIR}/stop_after_selector and rerun to resume into held-out." >&2
    exit 0
fi

benchmark=""
family=""
for benchmark in "${BENCHMARKS[@]}"; do
    while IFS= read -r family; do
        dir="$(heldout_dir_for "${benchmark}" "${family}")"
        alpha="$(expected_alpha_for_family "${family}")"
        manifest_path="$(manifest_for_family "${family}")"
        if ! heldout_stage_complete "${benchmark}" "${family}"; then
            extra_args=()
            if [[ "${benchmark}" == "faitheval" || "${benchmark}" == "faitheval_anti_compliance_margin" || "${benchmark}" == "faitheval_answer_span_margin" ]]; then
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
            DID_GENERATE_DATA=1
        else
            echo "Skipping held-out stage; found ${benchmark}/${family} outputs in ${dir}"
        fi
    done < <(families_for_benchmark "${benchmark}")
done

if ! report_stage_complete; then
    env PYTHONUNBUFFERED=1 uv run python scripts/report_faitheval_sae_utility_selector.py \
        --selector_dir "${SELECTOR_DIR}" \
        --heldout_root "${HELDOUT_ROOT}" \
        --output_dir "${ROOT}"
else
    echo "Skipping report stage; found fresh report outputs in ${REPORT_DIR}"
fi

if [[ "${DID_GENERATE_DATA}" -eq 1 ]]; then
    env PYTHONUNBUFFERED=1 uv run python -m scripts.lib.pipeline log-run \
        --run-dir "${ROOT}" \
        --description "FaithEval SAE utility-selector ablation + held-out bundle (anti-compliance, delta-only, validation-selected answer-span-selected/readout-selected/full-sequence-token-activation-weighted matched-random controls)" \
        --key-files "selector/selector_summary.json, selector/full_sequence_feature_stats.json, heldout/*/*/alpha_*.jsonl, report/heldout_summary.json, *.provenance.json"
fi

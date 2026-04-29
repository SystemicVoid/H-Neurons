#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons}"
TMUX_SESSION="${TMUX_SESSION:-mistral24b-replication}"

if [ -z "${TMUX:-}" ] && [ -z "${TMUX_WRAPPED:-}" ] && command -v tmux &>/dev/null; then
    quoted_args=""
    for arg in "$@"; do
        printf -v quoted_args '%s %q' "${quoted_args}" "${arg}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
        "cd ${PROJECT_DIR@Q} && env TMUX_WRAPPED=1 bash ${0@Q}${quoted_args}"
    printf 'Started tmux session %s. Attach with: tmux attach -t %s\n' \
        "${TMUX_SESSION}" "${TMUX_SESSION}"
    exit 0
fi

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle --why="mistral24b h-neuron replication" \
        -- bash "$0" "$@"
fi

cd "${PROJECT_DIR}"

UV_RUN=(uv run --no-sync)
PIPELINE=("${UV_RUN[@]}" python -m scripts.lib.pipeline)
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-logs}"

MODEL_KEY="${MODEL_KEY:-mistral_small_24b_instruct_2501}"
MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-Small-24B-Instruct-2501}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/mistral24b}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-${OUTPUT_ROOT}/preflight}"
MANIFEST_DIR="${MANIFEST_DIR:-data/manifests}"
FAITHEVAL_SAMPLE_MANIFEST="${FAITHEVAL_SAMPLE_MANIFEST:-${MANIFEST_DIR}/faitheval_seed42_n200_mistral24b.lock.json}"
FAITHEVAL_PILOT_MANIFEST="${FAITHEVAL_PILOT_MANIFEST:-${MANIFEST_DIR}/faitheval_iti_calibration_ids_seed42_n20_mistral24b.lock.json}"
FAITHEVAL_PILOT_OUTPUT_ROOT="${FAITHEVAL_PILOT_OUTPUT_ROOT:-${OUTPUT_ROOT}/intervention/faitheval_pilot_smoke}"
TRUTHFULQA_MC1_MANIFEST="${TRUTHFULQA_MC1_MANIFEST:-${MANIFEST_DIR}/truthfulqa_paper_heldout_mc1_ids_seed42_mistral24b.lock.json}"
TRUTHFULQA_MC2_MANIFEST="${TRUTHFULQA_MC2_MANIFEST:-${MANIFEST_DIR}/truthfulqa_paper_heldout_mc2_ids_seed42_mistral24b.lock.json}"
TRIVIAQA_BRIDGE_MANIFEST="${TRIVIAQA_BRIDGE_MANIFEST:-${MANIFEST_DIR}/triviaqa_bridge_test500_seed42_mistral24b.lock.json}"
SIMPLEQA_PILOT_MANIFEST="${SIMPLEQA_PILOT_MANIFEST:-${MANIFEST_DIR}/simpleqa_verified_pilot100_disjoint_seed42_mistral24b.lock.json}"
SIMPLEQA_CONTROL_MANIFEST="${SIMPLEQA_CONTROL_MANIFEST:-${MANIFEST_DIR}/simpleqa_verified_control200_seed42_mistral24b.lock.json}"
JBB_FULL_HARMFUL_MANIFEST="${JBB_FULL_HARMFUL_MANIFEST:-${MANIFEST_DIR}/jbb_d7_full_harmful500_seed42_mistral24b.lock.json}"
ANSWER_TOKENS="${ANSWER_TOKENS:-${OUTPUT_ROOT}/answer_tokens_llm.jsonl}"
TRAIN_IDS="${TRAIN_IDS:-${OUTPUT_ROOT}/pipeline/train_qids_llm.json}"
DEV_IDS="${DEV_IDS:-${OUTPUT_ROOT}/pipeline/dev_qids_llm.json}"
TEST_IDS="${TEST_IDS:-${OUTPUT_ROOT}/pipeline/test_qids_llm.json}"
ACT_ROOT="${ACT_ROOT:-${OUTPUT_ROOT}/pipeline/activations_llm_canonical}"
TOKEN_SPAN_AUDIT="${TOKEN_SPAN_AUDIT:-${PREFLIGHT_DIR}/token_span_audit.jsonl}"
TOKEN_SPAN_AUDIT_SUMMARY="${TOKEN_SPAN_AUDIT_SUMMARY:-${PREFLIGHT_DIR}/token_span_audit_summary.json}"
RENDERED_CHAT_EXAMPLES="${RENDERED_CHAT_EXAMPLES:-${PREFLIGHT_DIR}/rendered_chat_examples.json}"
MODEL_LOAD_SMOKE="${MODEL_LOAD_SMOKE:-${PREFLIGHT_DIR}/model_load_smoke.json}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-models/mistral24b_classifier_canonical.pkl}"
CLASSIFIER_DEV_METRICS="${CLASSIFIER_DEV_METRICS:-${OUTPUT_ROOT}/pipeline/classifier_canonical_dev_metrics.json}"
CLASSIFIER_TEST_METRICS="${CLASSIFIER_TEST_METRICS:-${OUTPUT_ROOT}/pipeline/classifier_canonical_test_metrics.json}"
SPLIT_SAMPLES="${SPLIT_SAMPLES:-360}"
DEV_SAMPLES="${DEV_SAMPLES:-100}"
TEST_SAMPLES="${TEST_SAMPLES:-100}"
TOKEN_SPAN_AUDIT_MAX_SAMPLES="${TOKEN_SPAN_AUDIT_MAX_SAMPLES:-50}"
INTERVENTION_MAX_SAMPLES="${INTERVENTION_MAX_SAMPLES:-100}"
STAGES="${STAGES:-all}"
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_TRANSCRIPT="${DRY_RUN_TRANSCRIPT:-${PREFLIGHT_DIR}/runpod_dry_run_transcript.txt}"
GPU_MIN_MEMORY_GIB="${GPU_MIN_MEMORY_GIB:-75}"
GPU_NAME_PATTERN="${GPU_NAME_PATTERN:-H100|A100}"
CP4_REVIEWED="${CP4_REVIEWED:-0}"
FAITHEVAL_PILOT_ALPHAS=(0.0 1.0 3.0)
read -r -a ALPHAS <<<"${ALPHAS:-0.0 0.5 1.0 1.5 2.0 2.5 3.0}"
read -r -a C_VALUES <<<"${C_VALUES:-0.001 0.005 0.01 0.05 0.1 0.5 1.0}"
FAITHEVAL_PILOT_CONTROL_DIRS=(
    seed_0_unconstrained
    seed_1_unconstrained
    seed_2_unconstrained
    seed_0_layer_matched
)

clean_untracked_uv_lock() {
    if [[ -f uv.lock ]] && ! git ls-files --error-unmatch uv.lock >/dev/null 2>&1; then
        rm -f uv.lock
    fi
}

clean_untracked_uv_lock

mkdir -p "${LOG_DIR}" "${PREFLIGHT_DIR}" "${OUTPUT_ROOT}/pipeline" "${OUTPUT_ROOT}/intervention"

if [[ "${DRY_RUN}" == "1" ]]; then
    LOG="${DRY_RUN_TRANSCRIPT}"
    : >"${LOG}"
else
    LOG="${LOG_DIR}/mistral24b_replication_${RUN_TS}.log"
fi

REQUIRED_LOCK_MANIFESTS=(
    "${FAITHEVAL_SAMPLE_MANIFEST}"
    "${FAITHEVAL_PILOT_MANIFEST}"
    "${TRUTHFULQA_MC1_MANIFEST}"
    "${TRUTHFULQA_MC2_MANIFEST}"
    "${TRIVIAQA_BRIDGE_MANIFEST}"
    "${SIMPLEQA_PILOT_MANIFEST}"
    "${SIMPLEQA_CONTROL_MANIFEST}"
    "${JBB_FULL_HARMFUL_MANIFEST}"
)

GPU_REQUIRED_STAGES=(
    model_smoke
    activations
    faitheval_pilot
    faitheval_pilot_controls
    faitheval
    faitheval_controls
    falseqa
    falseqa_controls
)

PREFLIGHT_VALIDATED_STAGES=(
    activations
    classifier
    faitheval_pilot
    faitheval_pilot_controls
    faitheval
    faitheval_controls
    falseqa
    falseqa_controls
)

require_lock_manifests() {
    if ! "${PIPELINE[@]}" validate-sample-locks "$@" 2>&1 | tee -a "${LOG}"; then
        exit 2
    fi
}

format_command() {
    local formatted=""
    local arg

    for arg in "$@"; do
        printf -v formatted '%s %q' "${formatted}" "${arg}"
    done
    printf '%s' "${formatted# }"
}

should_run_stage() {
    local stage="$1"
    [[ "${STAGES}" == "all" || ",${STAGES}," == *",${stage},"* ]]
}

should_run_any_stage() {
    local stage

    for stage in "$@"; do
        if should_run_stage "${stage}"; then
            return 0
        fi
    done
    return 1
}

run_logged() {
    local command_text
    command_text="$(format_command "$@")"
    printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${command_text}" | tee -a "${LOG}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run] skipped execution\n' | tee -a "${LOG}"
        return 0
    fi
    "$@" 2>&1 | tee -a "${LOG}"
}

run_stage() {
    local stage="$1"
    shift
    if should_run_stage "${stage}"; then
        run_logged "$@"
    else
        local command_text
        command_text="$(format_command "$@")"
        printf '\n[%s] [skip:%s] %s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stage}" "${command_text}" | tee -a "${LOG}"
    fi
}

require_gpu_hardware() {
    run_logged "${PIPELINE[@]}" gpu-hardware-guard \
        --min-memory-gib "${GPU_MIN_MEMORY_GIB}" \
        --name-pattern "${GPU_NAME_PATTERN}"
}

require_cp4_review_for_cp5() {
    if [[ "${CP4_REVIEWED}" == "1" ]]; then
        return 0
    fi
    printf '\n[%s] CP5 review gate blocked full FaithEval stage selection. Set CP4_REVIEWED=1 only after CP4 pilot outputs pass review.\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
    exit 2
}

require_stage_complete() {
    local output_dir="$1"
    local manifest="$2"
    shift 2

    run_logged "${PIPELINE[@]}" check-stage \
        --output-dir "${output_dir}" \
        --manifest "${manifest}" \
        --alphas "$@"
}

require_output_file() {
    local path="$1"

    if [[ "${DRY_RUN}" == "1" ]]; then
        run_logged test -f "${path}"
        return 0
    fi
    if [[ ! -f "${path}" ]]; then
        printf '\n[%s] required output file missing: %s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${path}" | tee -a "${LOG}"
        exit 2
    fi
    printf '[%s] verified output file: %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${path}" | tee -a "${LOG}"
}

require_faitheval_pilot_control_outputs() {
    local control_dir="${FAITHEVAL_PILOT_OUTPUT_ROOT}/control"
    local seed_dir

    require_output_file "${control_dir}/control_run_config.json"
    require_output_file "${control_dir}/comparison_summary.json"
    for seed_dir in "${FAITHEVAL_PILOT_CONTROL_DIRS[@]}"; do
        require_stage_complete \
            "${control_dir}/${seed_dir}" \
            "${FAITHEVAL_PILOT_MANIFEST}" \
            "${FAITHEVAL_PILOT_ALPHAS[@]}"
    done
}

validate_token_span_summary() {
    run_logged "${UV_RUN[@]}" python scripts/audit_token_spans.py \
        --validate_summary "${TOKEN_SPAN_AUDIT_SUMMARY}" \
        --expected_model_key "${MODEL_KEY}" \
        --expected_model_path "${MODEL_PATH}" \
        --expected_input_path "${ANSWER_TOKENS}" \
        --expected_output_path "${TOKEN_SPAN_AUDIT}" \
        --expected_rendered_chat_path "${RENDERED_CHAT_EXAMPLES}"
}

validate_model_load_smoke() {
    run_logged "${UV_RUN[@]}" python scripts/model_load_smoke.py \
        --validate_summary "${MODEL_LOAD_SMOKE}" \
        --expected_model_key "${MODEL_KEY}" \
        --expected_model_path "${MODEL_PATH}" \
        --expected_output_path "${MODEL_LOAD_SMOKE}" \
        --expected_device_map "${DEVICE_MAP}"
}

validate_claim_stage_preflight() {
    validate_token_span_summary
    validate_model_load_smoke
}

capture_versions() {
    {
        date -u
        git rev-parse HEAD || true
        git status --short || true
        uv --version
        "${UV_RUN[@]}" python -V
        "${UV_RUN[@]}" python - <<'PY'
import importlib.metadata as md

for name in ["torch", "transformers", "accelerate", "datasets", "scikit-learn"]:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=<not installed>")
PY
    } >"${OUTPUT_ROOT}/pipeline/environment_${RUN_TS}.txt" 2>&1
}

require_lock_manifests "${REQUIRED_LOCK_MANIFESTS[@]}"
if should_run_any_stage faitheval faitheval_controls; then
    require_cp4_review_for_cp5
fi

run_logged "${PIPELINE[@]}" active-run-status
run_logged "${PIPELINE[@]}" check-active-run-git-guard
run_logged "${PIPELINE[@]}" gpu-preflight || true
if command -v nvitop &>/dev/null; then
    run_logged nvitop -1 || true
fi
if should_run_any_stage "${GPU_REQUIRED_STAGES[@]}"; then
    require_gpu_hardware
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    printf '\n[%s] [dry-run] skipped environment capture\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
else
    capture_versions
fi

run_stage preflight "${UV_RUN[@]}" python scripts/audit_token_spans.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --input_path "${ANSWER_TOKENS}" \
    --output_path "${TOKEN_SPAN_AUDIT}" \
    --summary_path "${TOKEN_SPAN_AUDIT_SUMMARY}" \
    --rendered_chat_path "${RENDERED_CHAT_EXAMPLES}" \
    --max_samples "${TOKEN_SPAN_AUDIT_MAX_SAMPLES}"

run_stage model_smoke "${UV_RUN[@]}" python scripts/model_load_smoke.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --output_path "${MODEL_LOAD_SMOKE}" \
    --max_new_tokens 1

if should_run_stage model_smoke; then
    validate_model_load_smoke
fi

if should_run_any_stage "${PREFLIGHT_VALIDATED_STAGES[@]}"; then
    validate_claim_stage_preflight
fi

run_stage splits "${UV_RUN[@]}" python scripts/sample_balanced_ids.py \
    --input_path "${ANSWER_TOKENS}" \
    --output_path "${TRAIN_IDS}" \
    --num_samples "${SPLIT_SAMPLES}" \
    --seed 42 \
    --strict

run_stage splits "${UV_RUN[@]}" python scripts/sample_balanced_ids.py \
    --input_path "${ANSWER_TOKENS}" \
    --output_path "${DEV_IDS}" \
    --num_samples "${DEV_SAMPLES}" \
    --seed 43 \
    --exclude_path "${TRAIN_IDS}" \
    --strict

run_stage splits "${UV_RUN[@]}" python scripts/sample_balanced_ids.py \
    --input_path "${ANSWER_TOKENS}" \
    --output_path "${TEST_IDS}" \
    --num_samples "${TEST_SAMPLES}" \
    --seed 44 \
    --exclude_path "${TRAIN_IDS}" "${DEV_IDS}" \
    --strict

run_stage activations "${UV_RUN[@]}" python scripts/extract_activations.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --input_path "${ANSWER_TOKENS}" \
    --train_ids_path "${TRAIN_IDS}" \
    --output_root "${ACT_ROOT}/train" \
    --device_map "${DEVICE_MAP}" \
    --locations answer_tokens all_except_answer_tokens

run_stage activations "${UV_RUN[@]}" python scripts/extract_activations.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --input_path "${ANSWER_TOKENS}" \
    --train_ids_path "${DEV_IDS}" \
    --output_root "${ACT_ROOT}/dev" \
    --device_map "${DEVICE_MAP}" \
    --locations answer_tokens

run_stage activations "${UV_RUN[@]}" python scripts/extract_activations.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --input_path "${ANSWER_TOKENS}" \
    --train_ids_path "${TEST_IDS}" \
    --output_root "${ACT_ROOT}/test" \
    --device_map "${DEVICE_MAP}" \
    --locations answer_tokens

run_stage classifier "${UV_RUN[@]}" python scripts/classifier.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --train_ids "${TRAIN_IDS}" \
    --train_ans_acts "${ACT_ROOT}/train/answer_tokens" \
    --train_other_acts "${ACT_ROOT}/train/all_except_answer_tokens" \
    --test_ids "${DEV_IDS}" \
    --test_acts "${ACT_ROOT}/dev/answer_tokens" \
    --train_mode 3-vs-1 \
    --penalty l1 \
    --solver liblinear \
    --selection_metric auroc \
    --c_values "${C_VALUES[@]}" \
    --save_model "${CLASSIFIER_PATH}" \
    --metrics_out "${CLASSIFIER_DEV_METRICS}"

run_stage classifier "${UV_RUN[@]}" python scripts/classifier.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --load_model "${CLASSIFIER_PATH}" \
    --test_ids "${TEST_IDS}" \
    --test_acts "${ACT_ROOT}/test/answer_tokens" \
    --metrics_out "${CLASSIFIER_TEST_METRICS}"

run_stage classifier "${UV_RUN[@]}" python scripts/validate_mistral24b_cp23.py \
    --pipeline-dir "${OUTPUT_ROOT}/pipeline" \
    --activation-root "${ACT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --model-key "${MODEL_KEY}" \
    --classifier-path "${CLASSIFIER_PATH}"

run_stage faitheval_pilot "${UV_RUN[@]}" python scripts/run_intervention.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --benchmark faitheval \
    --prompt_style standard \
    --sample_manifest "${FAITHEVAL_PILOT_MANIFEST}" \
    --alphas "${FAITHEVAL_PILOT_ALPHAS[@]}" \
    --output_dir "${FAITHEVAL_PILOT_OUTPUT_ROOT}/experiment"

if should_run_stage faitheval_pilot; then
    require_stage_complete \
        "${FAITHEVAL_PILOT_OUTPUT_ROOT}/experiment" \
        "${FAITHEVAL_PILOT_MANIFEST}" \
        "${FAITHEVAL_PILOT_ALPHAS[@]}"
fi

if should_run_stage faitheval_pilot_controls && ! should_run_stage faitheval_pilot; then
    require_stage_complete \
        "${FAITHEVAL_PILOT_OUTPUT_ROOT}/experiment" \
        "${FAITHEVAL_PILOT_MANIFEST}" \
        "${FAITHEVAL_PILOT_ALPHAS[@]}"
fi

run_stage faitheval_pilot_controls "${UV_RUN[@]}" python scripts/run_negative_control.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --benchmark faitheval \
    --prompt_style standard \
    --sample_manifest "${FAITHEVAL_PILOT_MANIFEST}" \
    --quick \
    --output_base "${FAITHEVAL_PILOT_OUTPUT_ROOT}/control" \
    --h_neuron_baseline "${FAITHEVAL_PILOT_OUTPUT_ROOT}/experiment"

if should_run_stage faitheval_pilot_controls; then
    require_faitheval_pilot_control_outputs
fi

run_stage faitheval "${UV_RUN[@]}" python scripts/run_intervention.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --benchmark faitheval \
    --prompt_style standard \
    --sample_manifest "${FAITHEVAL_SAMPLE_MANIFEST}" \
    --alphas "${ALPHAS[@]}" \
    --output_dir "${OUTPUT_ROOT}/intervention/faitheval/experiment"

run_stage faitheval_controls "${UV_RUN[@]}" python scripts/run_negative_control.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --benchmark faitheval \
    --prompt_style standard \
    --sample_manifest "${FAITHEVAL_SAMPLE_MANIFEST}" \
    --output_base "${OUTPUT_ROOT}/intervention/faitheval/control" \
    --h_neuron_baseline "${OUTPUT_ROOT}/intervention/faitheval/experiment"

run_stage falseqa "${UV_RUN[@]}" python scripts/run_intervention.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --benchmark falseqa \
    --alphas 0.0 1.0 3.0 \
    --max_samples "${INTERVENTION_MAX_SAMPLES}" \
    --output_dir "${OUTPUT_ROOT}/intervention/falseqa/experiment"

run_stage falseqa_controls "${UV_RUN[@]}" python scripts/run_negative_control.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --benchmark falseqa \
    --quick \
    --max_samples "${INTERVENTION_MAX_SAMPLES}" \
    --output_base "${OUTPUT_ROOT}/intervention/falseqa/control" \
    --h_neuron_baseline "${OUTPUT_ROOT}/intervention/falseqa/experiment"

printf '[%s] Mistral 24B replication wrapper completed.\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"

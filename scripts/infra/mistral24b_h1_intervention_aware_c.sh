#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons}"
TMUX_SESSION="${TMUX_SESSION:-mistral24b-h1-intervention-aware-c}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" != "1" ]] && [ -z "${TMUX:-}" ] && [ -z "${TMUX_WRAPPED:-}" ] && command -v tmux &>/dev/null; then
    quoted_args=""
    for arg in "$@"; do
        printf -v quoted_args '%s %q' "${quoted_args}" "${arg}"
    done
    tmux_env=(
        "TMUX_WRAPPED=1"
        "PROJECT_DIR=${PROJECT_DIR}"
        "TMUX_SESSION=${TMUX_SESSION}"
        "DRY_RUN=${DRY_RUN}"
    )
    tmux_passthrough_vars=(
        INHIBIT_WRAPPED
        LOG
        LOG_DIR
        UV_PROJECT_ENVIRONMENT
        MODEL_KEY
        MODEL_PATH
        DEVICE_MAP
        OUTPUT_ROOT
        MANIFEST_DIR
        TRAIN_IDS
        DEV_IDS
        ACT_ROOT
        TRAIN_PER_CLASS
        DEV_PER_CLASS
        FFN_LAYERS
        FFN_NEURONS
        C_SELECTION_ROOT
        CANDIDATE_MODELS_DIR
        CLASSIFIER_SWEEP_MODEL
        CLASSIFIER_SWEEP_METRICS
        SELECTION_SUMMARY
        SELECTED_CLASSIFIER
        TRIVIAQA_MANIFEST
        TRIVIAQA_PARQUET
        TRIVIAQA_ROOT
        FAITHEVAL_MANIFEST
        FAITHEVAL_OUTPUT_DIR
        STAGES
        H1_REVIEWED
        GPU_MIN_MEMORY_GIB
        GPU_NAME_PATTERN
    )
    for env_name in "${tmux_passthrough_vars[@]}"; do
        if [[ -v "${env_name}" ]]; then
            tmux_env+=("${env_name}=${!env_name}")
        fi
    done
    quoted_env=""
    for env_entry in "${tmux_env[@]}"; do
        printf -v quoted_env '%s %q' "${quoted_env}" "${env_entry}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
        "cd ${PROJECT_DIR@Q} && env${quoted_env} bash ${0@Q}${quoted_args}"
    printf 'Started tmux session %s. Attach with: tmux attach -t %s\n' \
        "${TMUX_SESSION}" "${TMUX_SESSION}"
    exit 0
fi

if [[ "${DRY_RUN}" != "1" ]] && [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="Mistral H1 intervention-aware C selection and FaithEval follow-up" \
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
MANIFEST_DIR="${MANIFEST_DIR:-data/manifests}"
TRAIN_IDS="${TRAIN_IDS:-${OUTPUT_ROOT}/pipeline/train_qids_llm.json}"
DEV_IDS="${DEV_IDS:-${OUTPUT_ROOT}/pipeline/dev_qids_llm.json}"
ACT_ROOT="${ACT_ROOT:-${OUTPUT_ROOT}/pipeline/activations_llm_canonical}"

C_SELECTION_ROOT="${C_SELECTION_ROOT:-${OUTPUT_ROOT}/pipeline/h1_intervention_aware_c}"
CANDIDATE_MODELS_DIR="${CANDIDATE_MODELS_DIR:-${C_SELECTION_ROOT}/candidate_models}"
CLASSIFIER_SWEEP_MODEL="${CLASSIFIER_SWEEP_MODEL:-${C_SELECTION_ROOT}/classifier_dev_accuracy_best.pkl}"
CLASSIFIER_SWEEP_METRICS="${CLASSIFIER_SWEEP_METRICS:-${C_SELECTION_ROOT}/classifier_c_sweep_metrics.json}"
SELECTION_SUMMARY="${SELECTION_SUMMARY:-${C_SELECTION_ROOT}/selection_summary.json}"
SELECTED_CLASSIFIER="${SELECTED_CLASSIFIER:-models/mistral24b_classifier_h1_intervention_aware.pkl}"

TRIVIAQA_MANIFEST="${TRIVIAQA_MANIFEST:-${MANIFEST_DIR}/triviaqa_bridge_reserve200_seed42.json}"
TRIVIAQA_PARQUET="${TRIVIAQA_PARQUET:-data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet}"
TRIVIAQA_ROOT="${TRIVIAQA_ROOT:-${OUTPUT_ROOT}/intervention/triviaqa_bridge_h1_intervention_aware_c}"
FAITHEVAL_MANIFEST="${FAITHEVAL_MANIFEST:-${MANIFEST_DIR}/faitheval_seed42_n200_mistral24b.lock.json}"
FAITHEVAL_OUTPUT_DIR="${FAITHEVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/intervention/faitheval_h1_intervention_aware_c/experiment}"

STAGES="${STAGES:-all}"
H1_REVIEWED="${H1_REVIEWED:-0}"
GPU_MIN_MEMORY_GIB="${GPU_MIN_MEMORY_GIB:-75}"
GPU_NAME_PATTERN="${GPU_NAME_PATTERN:-H100|A100}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-360}"
DEV_PER_CLASS="${DEV_PER_CLASS:-100}"
FFN_LAYERS="${FFN_LAYERS:-40}"
FFN_NEURONS="${FFN_NEURONS:-32768}"

C_VALUES=(0.05 0.1 0.3 0.5 0.75 1.0 1.25 1.5 2.0 3.0)
FAITHEVAL_ALPHAS=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
TRIVIAQA_ALPHAS=(0.0)
GPU_STAGES=(triviaqa_c_sweep faitheval)

mkdir -p "${LOG_DIR}"
if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${C_SELECTION_ROOT}" "${CANDIDATE_MODELS_DIR}" "${TRIVIAQA_ROOT}"
fi
LOG="${LOG:-${LOG_DIR}/mistral24b_h1_intervention_aware_c_${RUN_TS}.log}"

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

require_file() {
    local path="$1"
    if [[ "${DRY_RUN}" == "1" ]]; then
        run_logged test -f "${path}"
        return 0
    fi
    if [[ ! -f "${path}" ]]; then
        printf 'FATAL: missing required file: %s\n' "${path}" | tee -a "${LOG}" >&2
        exit 2
    fi
}

require_h1_review_for_gpu() {
    if [[ "${DRY_RUN}" == "1" || "${H1_REVIEWED}" == "1" ]]; then
        return 0
    fi
    printf 'FATAL: GPU stages require H1_REVIEWED=1 after code review.\n' | tee -a "${LOG}" >&2
    exit 2
}

require_gpu_hardware() {
    run_logged "${PIPELINE[@]}" gpu-hardware-guard \
        --min-memory-gib "${GPU_MIN_MEMORY_GIB}" \
        --name-pattern "${GPU_NAME_PATTERN}"
}

c_label() {
    local normalized
    printf -v normalized '%g' "$1"
    normalized="${normalized/-/neg_}"
    normalized="${normalized//./_}"
    printf '%s' "${normalized}"
}

candidate_model_for_c() {
    printf '%s/classifier_C_%s.pkl' "${CANDIDATE_MODELS_DIR}" "$(c_label "$1")"
}

triviaqa_output_dir_for_c() {
    printf '%s/c_%s/experiment' "${TRIVIAQA_ROOT}" "$(c_label "$1")"
}

triviaqa_summary_complete() {
    local output_dir="$1"
    local had_nullglob
    local summaries=()

    if [[ "${DRY_RUN}" == "1" ]]; then
        run_logged test -f "${output_dir}/results.json"
        return 1
    fi

    if [[ -f "${output_dir}/results.json" ]]; then
        return 0
    fi

    if shopt -q nullglob; then
        had_nullglob=1
    else
        had_nullglob=0
    fi
    shopt -s nullglob
    summaries=("${output_dir}"/results.*.json)
    if [[ "${had_nullglob}" == "0" ]]; then
        shopt -u nullglob
    fi

    if ((${#summaries[@]} > 0)); then
        return 0
    fi

    printf '  missing: %s/results.json or results.*.json\n' "${output_dir}" | tee -a "${LOG}" >&2
    return 1
}

stage_complete_with_manifest_prefix() {
    local output_dir="$1"
    local manifest="$2"
    local manifest_id_prefix="$3"
    shift 3
    local check_stage_cmd=(
        "${PIPELINE[@]}" check-stage
        --output-dir "${output_dir}"
        --manifest "${manifest}"
    )
    if [[ -n "${manifest_id_prefix}" ]]; then
        check_stage_cmd+=(--manifest-id-prefix "${manifest_id_prefix}")
    fi
    check_stage_cmd+=(--alphas "$@")

    if [[ "${DRY_RUN}" == "1" ]]; then
        run_logged "${check_stage_cmd[@]}"
        return 1
    fi
    "${check_stage_cmd[@]}" 2>&1 | tee -a "${LOG}"
}

stage_complete() {
    local output_dir="$1"
    local manifest="$2"
    shift 2

    stage_complete_with_manifest_prefix "${output_dir}" "${manifest}" "" "$@"
}

intervention_contract_complete() {
    local output_dir="$1"
    shift
    local contract_cmd=(
        "${PIPELINE[@]}" check-intervention-contract
        --output-dir "${output_dir}"
        "$@"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        run_logged "${contract_cmd[@]}"
        return 1
    fi
    "${contract_cmd[@]}" 2>&1 | tee -a "${LOG}"
}

validate_classifier_activation_inputs() {
    run_stage classifier_c_sweep env PYTHONUNBUFFERED=1 "${UV_RUN[@]}" python scripts/validate_mistral24b_cp23.py \
        --activation-inputs-only \
        --train-ids-path "${TRAIN_IDS}" \
        --dev-ids-path "${DEV_IDS}" \
        --train-answer-dir "${ACT_ROOT}/train/answer_tokens" \
        --train-other-dir "${ACT_ROOT}/train/all_except_answer_tokens" \
        --dev-answer-dir "${ACT_ROOT}/dev/answer_tokens" \
        --train-per-class "${TRAIN_PER_CLASS}" \
        --dev-per-class "${DEV_PER_CLASS}" \
        --layers "${FFN_LAYERS}" \
        --ffn-neurons "${FFN_NEURONS}"
}

TRIVIAQA_RESULT_ARGS=()
for c_value in "${C_VALUES[@]}"; do
    TRIVIAQA_RESULT_ARGS+=(--triviaqa_result "${c_value}=$(triviaqa_output_dir_for_c "${c_value}")")
done

printf '=== Mistral H1 intervention-aware C selection ===\n' | tee -a "${LOG}"
printf 'dry_run=%s stages=%s h1_reviewed=%s\n' "${DRY_RUN}" "${STAGES}" "${H1_REVIEWED}" | tee -a "${LOG}"
printf 'C grid: %s\n' "${C_VALUES[*]}" | tee -a "${LOG}"
printf 'Classifier metrics: %s\n' "${CLASSIFIER_SWEEP_METRICS}" | tee -a "${LOG}"
printf 'Selection summary: %s\n' "${SELECTION_SUMMARY}" | tee -a "${LOG}"
printf 'Selected classifier: %s\n' "${SELECTED_CLASSIFIER}" | tee -a "${LOG}"

require_file "${TRAIN_IDS}"
require_file "${DEV_IDS}"
require_file "${ACT_ROOT}/train/metadata.json"
require_file "${ACT_ROOT}/train/summary.json"
require_file "${ACT_ROOT}/dev/metadata.json"
require_file "${ACT_ROOT}/dev/summary.json"
require_file "${TRIVIAQA_MANIFEST}"
require_file "${TRIVIAQA_PARQUET}"
require_file "${FAITHEVAL_MANIFEST}"

if [[ "${DRY_RUN}" != "1" ]]; then
    "${PIPELINE[@]}" validate-sample-locks "${FAITHEVAL_MANIFEST}" 2>&1 | tee -a "${LOG}"
else
    run_logged "${PIPELINE[@]}" validate-sample-locks "${FAITHEVAL_MANIFEST}"
fi

run_logged "${PIPELINE[@]}" active-run-status
run_logged "${PIPELINE[@]}" check-active-run-git-guard
run_logged "${PIPELINE[@]}" gpu-preflight || true
if command -v nvitop &>/dev/null; then
    run_logged nvitop -1 || true
fi
if should_run_any_stage "${GPU_STAGES[@]}"; then
    require_h1_review_for_gpu
    require_gpu_hardware
fi

validate_classifier_activation_inputs

run_stage classifier_c_sweep env PYTHONUNBUFFERED=1 "${UV_RUN[@]}" python scripts/classifier.py \
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
    --selection_metric accuracy \
    --c_values "${C_VALUES[@]}" \
    --candidate_models_dir "${CANDIDATE_MODELS_DIR}" \
    --save_model "${CLASSIFIER_SWEEP_MODEL}" \
    --metrics_out "${CLASSIFIER_SWEEP_METRICS}"

for c_value in "${C_VALUES[@]}"; do
    candidate_model="$(candidate_model_for_c "${c_value}")"
    triviaqa_output_dir="$(triviaqa_output_dir_for_c "${c_value}")"
    if should_run_stage triviaqa_c_sweep; then
        require_h1_review_for_gpu
        require_file "${candidate_model}"
        if stage_complete_with_manifest_prefix \
            "${triviaqa_output_dir}" \
            "${TRIVIAQA_MANIFEST}" \
            "tqa_bridge_" \
            "${TRIVIAQA_ALPHAS[@]}" && \
            intervention_contract_complete \
                "${triviaqa_output_dir}" \
                --benchmark triviaqa_bridge \
                --model-key "${MODEL_KEY}" \
                --model-path "${MODEL_PATH}" \
                --classifier-path "${candidate_model}" \
                --alphas "${TRIVIAQA_ALPHAS[@]}" \
                --benchmark-config "triviaqa_bridge_manifest=${TRIVIAQA_MANIFEST}" \
                --benchmark-config "triviaqa_bridge_parquet=${TRIVIAQA_PARQUET}" && \
            triviaqa_summary_complete "${triviaqa_output_dir}"; then
            printf '[%s] TriviaQA C=%s complete; skipping\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${c_value}" | tee -a "${LOG}"
        else
            run_logged env PYTHONUNBUFFERED=1 "${UV_RUN[@]}" python scripts/run_intervention.py \
                --model_key "${MODEL_KEY}" \
                --model_path "${MODEL_PATH}" \
                --device_map "${DEVICE_MAP}" \
                --classifier_path "${candidate_model}" \
                --benchmark triviaqa_bridge \
                --triviaqa_bridge_manifest "${TRIVIAQA_MANIFEST}" \
                --triviaqa_bridge_parquet "${TRIVIAQA_PARQUET}" \
                --alphas "${TRIVIAQA_ALPHAS[@]}" \
                --output_dir "${triviaqa_output_dir}"
        fi
    else
        run_stage triviaqa_c_sweep env PYTHONUNBUFFERED=1 "${UV_RUN[@]}" python scripts/run_intervention.py \
            --model_key "${MODEL_KEY}" \
            --model_path "${MODEL_PATH}" \
            --device_map "${DEVICE_MAP}" \
            --classifier_path "${candidate_model}" \
            --benchmark triviaqa_bridge \
            --triviaqa_bridge_manifest "${TRIVIAQA_MANIFEST}" \
            --triviaqa_bridge_parquet "${TRIVIAQA_PARQUET}" \
            --alphas "${TRIVIAQA_ALPHAS[@]}" \
            --output_dir "${triviaqa_output_dir}"
    fi
done

run_stage select_c env PYTHONUNBUFFERED=1 "${UV_RUN[@]}" python scripts/select_intervention_aware_c.py \
    --classifier_metrics "${CLASSIFIER_SWEEP_METRICS}" \
    "${TRIVIAQA_RESULT_ARGS[@]}" \
    --output_summary "${SELECTION_SUMMARY}" \
    --selected_model_out "${SELECTED_CLASSIFIER}"

if should_run_stage faitheval; then
    require_h1_review_for_gpu
    require_file "${SELECTED_CLASSIFIER}"
    if stage_complete "${FAITHEVAL_OUTPUT_DIR}" "${FAITHEVAL_MANIFEST}" "${FAITHEVAL_ALPHAS[@]}" && \
        intervention_contract_complete \
            "${FAITHEVAL_OUTPUT_DIR}" \
            --benchmark faitheval \
            --model-key "${MODEL_KEY}" \
            --model-path "${MODEL_PATH}" \
            --classifier-path "${SELECTED_CLASSIFIER}" \
            --sample-manifest "${FAITHEVAL_MANIFEST}" \
            --alphas "${FAITHEVAL_ALPHAS[@]}" \
            --benchmark-config "prompt_style=standard"; then
        printf '[%s] FaithEval follow-up complete; skipping\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
    else
        run_logged env PYTHONUNBUFFERED=1 "${UV_RUN[@]}" python scripts/run_intervention.py \
            --model_key "${MODEL_KEY}" \
            --model_path "${MODEL_PATH}" \
            --device_map "${DEVICE_MAP}" \
            --classifier_path "${SELECTED_CLASSIFIER}" \
            --benchmark faitheval \
            --prompt_style standard \
            --sample_manifest "${FAITHEVAL_MANIFEST}" \
            --alphas "${FAITHEVAL_ALPHAS[@]}" \
            --output_dir "${FAITHEVAL_OUTPUT_DIR}"
    fi
else
    run_stage faitheval env PYTHONUNBUFFERED=1 "${UV_RUN[@]}" python scripts/run_intervention.py \
        --model_key "${MODEL_KEY}" \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --classifier_path "${SELECTED_CLASSIFIER}" \
        --benchmark faitheval \
        --prompt_style standard \
        --sample_manifest "${FAITHEVAL_MANIFEST}" \
        --alphas "${FAITHEVAL_ALPHAS[@]}" \
        --output_dir "${FAITHEVAL_OUTPUT_DIR}"
fi

if should_run_stage faitheval; then
    run_logged "${PIPELINE[@]}" log-run \
        --run-dir "${FAITHEVAL_OUTPUT_DIR}" \
        --description "Mistral H1 intervention-aware C FaithEval follow-up"
fi

printf '=== Mistral H1 intervention-aware C pipeline complete ===\n' | tee -a "${LOG}"

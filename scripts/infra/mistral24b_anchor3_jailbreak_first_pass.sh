#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons}"
TMUX_SESSION="${TMUX_SESSION:-mistral24b-anchor3-jailbreak}"
DRY_RUN="${DRY_RUN:-1}"
RUN_APPROVED="${RUN_APPROVED:-0}"
API_APPROVED="${API_APPROVED:-0}"

# Anchor 3 is full-generation-first: generate the locked JBB 500 surface once,
# then spend API budget progressively on evaluator comparisons from those rows.
if [[ "${DRY_RUN}" != "1" ]] && [[ -z "${TMUX:-}" ]] && [[ -z "${TMUX_WRAPPED:-}" ]] && command -v tmux >/dev/null 2>&1; then
    quoted_args=""
    for arg in "$@"; do
        printf -v quoted_args '%s %q' "${quoted_args}" "${arg}"
    done
    tmux_env=(
        "TMUX_WRAPPED=1"
        "PROJECT_DIR=${PROJECT_DIR}"
        "TMUX_SESSION=${TMUX_SESSION}"
        "DRY_RUN=${DRY_RUN}"
        "RUN_APPROVED=${RUN_APPROVED}"
        "API_APPROVED=${API_APPROVED}"
    )
    tmux_passthrough_vars=(
        INHIBIT_WRAPPED
        LOG
        LOG_DIR
        UV_PROJECT_ENVIRONMENT
        MODEL_KEY
        MODEL_PATH
        CLASSIFIER_PATH
        DEVICE_MAP
        SAMPLE_MANIFEST
        MAX_SAMPLES
        OUTPUT_ROOT
        FULL_DIR
        TRUNC_256_DIR
        CSV3_FULL_DIR
        STRONGREJECT_FULL_DIR
        SUMMARY_PATH
        JUDGE_MODEL
        OPENAI_API_KEY
        OPENAI_BATCH_MAX_ENQUEUED_TOKENS
        OPENAI_BATCH_QUEUE_SAFETY_MARGIN
        OPENAI_BATCH_USAGE_TIER
        STAGES
        ALPHAS
        GPU_MIN_MEMORY_GIB
        GPU_NAME_PATTERN
    )
    for env_name in "${tmux_passthrough_vars[@]}"; do
        if [[ -v "${env_name}" ]]; then
            if [[ "${env_name}" == OPENAI_* && -z "${!env_name}" ]]; then
                continue
            fi
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

# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/lib/inhibit_suspend.sh"
inhibit_suspend "Mistral anchor 3 jailbreak measurement first pass" "$@"

cd "${PROJECT_DIR}"

UV_RUN=(uv run --no-sync)
PIPELINE=("${UV_RUN[@]}" python -m scripts.lib.pipeline)
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG:-${LOG_DIR}/mistral24b_anchor3_jailbreak_first_pass_${RUN_TS}.log}"

MODEL_KEY="${MODEL_KEY:-mistral_small_24b_instruct_2501}"
MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-Small-24B-Instruct-2501}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-models/mistral24b_classifier_canonical.pkl}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-data/manifests/jbb_d7_full_harmful500_seed42_mistral24b.lock.json}"
# MAX_SAMPLES is only for tiny technical canaries; leave unset for decision runs.
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/mistral24b/intervention/jailbreak_anchor3_full500}"
FULL_DIR="${FULL_DIR:-${OUTPUT_ROOT}/experiment}"
TRUNC_256_DIR="${TRUNC_256_DIR:-${OUTPUT_ROOT}/truncated_256_eval_input}"
CSV3_FULL_DIR="${CSV3_FULL_DIR:-${OUTPUT_ROOT}/csv2_v3_evaluation}"
STRONGREJECT_FULL_DIR="${STRONGREJECT_FULL_DIR:-${OUTPUT_ROOT}/strongreject_evaluation}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/mistral_anchor3_jailbreak_evaluator_summary.json}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-2024-11-20}"
FIRST_PASS_JUDGE_STAGES="truncate_256,binary_full,binary_256,csv3_full,strongreject_full,summary"
STAGES="${STAGES:-${FIRST_PASS_JUDGE_STAGES}}"
read -r -a ALPHAS <<<"${ALPHAS:-0.0 1.0 1.5 3.0}"
MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
    MAX_SAMPLE_ARGS=(--max_samples "${MAX_SAMPLES}")
fi

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
        printf '\n[%s] [skip:%s] %s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stage}" "$(format_command "$@")" | tee -a "${LOG}"
    fi
}

require_paid_approval() {
    if [[ "${DRY_RUN}" == "1" || "${RUN_APPROVED}" == "1" ]]; then
        return 0
    fi
    printf 'FATAL: paid generation requires RUN_APPROVED=1. Re-run with DRY_RUN=1 to inspect commands.\n' | tee -a "${LOG}" >&2
    exit 2
}

require_api_approval() {
    if [[ "${DRY_RUN}" == "1" || "${API_APPROVED}" == "1" ]]; then
        return 0
    fi
    printf 'FATAL: OpenAI evaluation requires API_APPROVED=1. Re-run with DRY_RUN=1 to inspect commands.\n' | tee -a "${LOG}" >&2
    exit 2
}

print_scaffold_summary() {
    printf '\n[%s] Mistral anchor 3 full-generation-first scaffold\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
    printf '[config] manifest=%s (locked JBB harmful500); alphas=%s; stages=%s\n' \
        "${SAMPLE_MANIFEST}" "${ALPHAS[*]}" "${STAGES}" | tee -a "${LOG}"
    printf '[config] output_root=%s; full_responses=%s; trunc_256=%s\n' \
        "${OUTPUT_ROOT}" "${FULL_DIR}" "${TRUNC_256_DIR}" | tee -a "${LOG}"
    if [[ -n "${MAX_SAMPLES}" ]]; then
        printf '[canary] MAX_SAMPLES=%s is a technical canary override, not decision evidence.\n' \
            "${MAX_SAMPLES}" | tee -a "${LOG}"
    else
        printf '[config] MAX_SAMPLES unset: generation uses the full locked 500-row manifest.\n' | tee -a "${LOG}"
    fi
    printf '[plan] Evaluate existing full responses with STAGES=%s.\n' \
        "${FIRST_PASS_JUDGE_STAGES}" | tee -a "${LOG}"
    printf '[plan] No csv3_256 or strongreject_256 stage is included in this pass.\n' \
        | tee -a "${LOG}"
}

print_scaffold_summary

if should_run_stage generate; then
    require_paid_approval
    run_logged "${PIPELINE[@]}" active-run-status
    run_logged "${PIPELINE[@]}" validate-sample-locks "${SAMPLE_MANIFEST}"
    run_logged "${PIPELINE[@]}" gpu-hardware-guard \
        --min-memory-gib "${GPU_MIN_MEMORY_GIB:-75}" \
        --name-pattern "${GPU_NAME_PATTERN:-H100|A100}"
fi

run_stage generate "${UV_RUN[@]}" python scripts/run_intervention.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --benchmark jailbreak \
    --sample_manifest "${SAMPLE_MANIFEST}" \
    "${MAX_SAMPLE_ARGS[@]}" \
    --alphas "${ALPHAS[@]}" \
    --run_profile canonical \
    --seed 42 \
    --output_dir "${FULL_DIR}"

run_stage truncate_256 "${UV_RUN[@]}" python scripts/materialize_jailbreak_truncation_view.py \
    --input_dir "${FULL_DIR}" \
    --output_dir "${TRUNC_256_DIR}" \
    --alphas "${ALPHAS[@]}" \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --response-token-limit 256

if should_run_stage binary_full || should_run_stage binary_256 || should_run_stage csv3_full || should_run_stage strongreject_full; then
    require_api_approval
fi

run_stage binary_full "${UV_RUN[@]}" python scripts/evaluate_intervention.py \
    --benchmark jailbreak \
    --input_dir "${FULL_DIR}" \
    --alphas "${ALPHAS[@]}" \
    --judge_model "${JUDGE_MODEL}" \
    --api-mode batch

run_stage binary_256 "${UV_RUN[@]}" python scripts/evaluate_intervention.py \
    --benchmark jailbreak \
    --input_dir "${TRUNC_256_DIR}" \
    --alphas "${ALPHAS[@]}" \
    --judge_model "${JUDGE_MODEL}" \
    --api-mode batch

run_stage csv3_full "${UV_RUN[@]}" python scripts/evaluate_csv2.py \
    --input_dir "${FULL_DIR}" \
    --output_dir "${CSV3_FULL_DIR}" \
    --alphas "${ALPHAS[@]}" \
    --judge_model "${JUDGE_MODEL}" \
    --api-mode batch

run_stage strongreject_full "${UV_RUN[@]}" python scripts/evaluate_strongreject.py \
    --input_dir "${FULL_DIR}" \
    --output_dir "${STRONGREJECT_FULL_DIR}" \
    --alphas "${ALPHAS[@]}" \
    --judge_model "${JUDGE_MODEL}" \
    --api-mode batch

run_stage summary "${UV_RUN[@]}" python scripts/analyze_available_jailbreak_evaluators.py \
    --mode mistral-anchor \
    --binary-full-dir "${FULL_DIR}" \
    --binary-256-dir "${TRUNC_256_DIR}" \
    --csv3-full-dir "${CSV3_FULL_DIR}" \
    --strongreject-full-dir "${STRONGREJECT_FULL_DIR}" \
    --alphas "${ALPHAS[@]}" \
    --output-path "${SUMMARY_PATH}"

printf '\n[%s] Scaffold complete. Outputs root: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${OUTPUT_ROOT}" | tee -a "${LOG}"

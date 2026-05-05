#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TMUX_SESSION="${TMUX_SESSION:-mistral24b-anchor2}"
DRY_RUN="${DRY_RUN:-1}"

if [[ "${DRY_RUN}" != "1" && -z "${TMUX:-}" && -z "${TMUX_WRAPPED:-}" ]] && command -v tmux &>/dev/null; then
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

# shellcheck source=scripts/lib/inhibit_suspend.sh
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/lib/inhibit_suspend.sh"
inhibit_suspend "Mistral Anchor 2 ITI MC-to-bridge" "$@"

cd "${PROJECT_DIR}"

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG:-${LOG_DIR}/mistral24b_anchor2_${RUN_TS}.log}"

UV_RUNTIME_MODE="${UV_RUNTIME_MODE:-no-sync}"
case "${UV_RUNTIME_MODE}" in
    baked)
        export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/h-neurons/.venv}"
        UV_RUN=(uv run --no-sync)
        ;;
    requirements)
        UV_RUN=(
            uv run --no-project
            --with-requirements requirements.txt
            --python 3.11
            --index https://download.pytorch.org/whl/cu130
            --index-strategy unsafe-best-match
        )
        ;;
    no-sync)
        UV_RUN=(uv run --no-sync)
        ;;
    *)
        printf 'Unsupported UV_RUNTIME_MODE=%s\n' "${UV_RUNTIME_MODE}" >&2
        exit 2
        ;;
esac
PIPELINE=("${UV_RUN[@]}" python -m scripts.lib.pipeline)

MODEL_KEY="${MODEL_KEY:-mistral_small_24b_instruct_2501}"
MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-Small-24B-Instruct-2501}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
SEED="${SEED:-42}"
MANIFEST_DIR="${MANIFEST_DIR:-data/manifests}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/mistral24b}"
STAGES="${STAGES:-all}"
RUN_APPROVED="${RUN_APPROVED:-0}"
API_APPROVED="${API_APPROVED:-0}"
GPU_MIN_MEMORY_GIB="${GPU_MIN_MEMORY_GIB:-75}"
GPU_NAME_PATTERN="${GPU_NAME_PATTERN:-H100|A100}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-2024-11-20}"
OPENAI_BATCH_MAX_ENQUEUED_TOKENS="${OPENAI_BATCH_MAX_ENQUEUED_TOKENS:-}"

TRUTHFULQA_MC1_MANIFEST="${TRUTHFULQA_MC1_MANIFEST:-${MANIFEST_DIR}/truthfulqa_paper_heldout_mc1_ids_seed42_mistral24b.lock.json}"
TRUTHFULQA_MC2_MANIFEST="${TRUTHFULQA_MC2_MANIFEST:-${MANIFEST_DIR}/truthfulqa_paper_heldout_mc2_ids_seed42_mistral24b.lock.json}"
TRIVIAQA_BRIDGE_MANIFEST="${TRIVIAQA_BRIDGE_MANIFEST:-${MANIFEST_DIR}/triviaqa_bridge_test500_seed42_mistral24b.lock.json}"
ANCHOR2_PREFIX="${ANCHOR2_PREFIX:-truthfulqa_anchor2_mistral24b}"
ANCHOR2_FOLD_PATH="${ANCHOR2_FOLD_PATH:-${MANIFEST_DIR}/${ANCHOR2_PREFIX}_fold_seed${SEED}.json}"
ANCHOR2_CAL_MC1_MANIFEST="${ANCHOR2_CAL_MC1_MANIFEST:-${MANIFEST_DIR}/${ANCHOR2_PREFIX}_cal_val_mc1_seed${SEED}.json}"
ANCHOR2_CAL_MC2_MANIFEST="${ANCHOR2_CAL_MC2_MANIFEST:-${MANIFEST_DIR}/${ANCHOR2_PREFIX}_cal_val_mc2_seed${SEED}.json}"

ITI_ROOT="${ITI_ROOT:-${OUTPUT_ROOT}/contrastive/truthfulness/anchor2_iti_truthfulqa_paperfaithful}"
ITI_ARTIFACT_DIR="${ITI_ARTIFACT_DIR:-${ITI_ROOT}/artifact}"
ITI_ARTIFACT_PATH="${ITI_ARTIFACT_PATH:-${ITI_ARTIFACT_DIR}/iti_heads.pt}"
SWEEP_DIR="${SWEEP_DIR:-${ITI_ROOT}/sweep}"
LOCKED_CONFIG="${LOCKED_CONFIG:-${SWEEP_DIR}/locked_iti_config.json}"

MC1_OUTPUT_DIR="${MC1_OUTPUT_DIR:-${OUTPUT_ROOT}/intervention/anchor2_truthfulqa_mc1_iti_truthfulqa_paperfaithful/experiment}"
MC2_OUTPUT_DIR="${MC2_OUTPUT_DIR:-${OUTPUT_ROOT}/intervention/anchor2_truthfulqa_mc2_iti_truthfulqa_paperfaithful/experiment}"
MC_REPORT_DIR="${MC_REPORT_DIR:-${OUTPUT_ROOT}/intervention/anchor2_truthfulqa_mc/reports}"
MC1_REPORT_PATH="${MC1_REPORT_PATH:-${MC_REPORT_DIR}/mistral_anchor2_heldout_mc1_report.json}"
MC2_REPORT_PATH="${MC2_REPORT_PATH:-${MC_REPORT_DIR}/mistral_anchor2_heldout_mc2_report.json}"
BRIDGE_OUTPUT_DIR="${BRIDGE_OUTPUT_DIR:-${OUTPUT_ROOT}/intervention/triviaqa_bridge_anchor2_iti_truthfulqa_paperfaithful/experiment}"
IRR_DIR="${IRR_DIR:-${OUTPUT_ROOT}/judge_validation/bridge_anchor2_irr}"
ADJUDICATION_RULE_PATH="${ADJUDICATION_RULE_PATH:-data/judge_validation/bridge_irr/adjudication_rule.md}"

read -r -a K_VALUES <<<"${K_VALUES:-8 12 16 24 32 40}"
read -r -a ALPHA_VALUES <<<"${ALPHA_VALUES:-0.0 0.5 1.0 2.0 4.0 6.0 8.0 12.0 16.0}"
GPU_STAGES=(extract sweep truthfulqa_mc bridge_generate)
API_STAGES=(bridge_judge)

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
        printf '\n[%s] [skip:%s] %s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stage}" "$(format_command "$@")" | tee -a "${LOG}"
    fi
}

require_run_approved() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    if [[ "${RUN_APPROVED}" != "1" ]]; then
        printf 'GPU stage blocked: set RUN_APPROVED=1 after remote dry-run review.\n' | tee -a "${LOG}"
        exit 2
    fi
}

require_api_approved() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    if [[ "${API_APPROVED}" != "1" ]]; then
        printf 'API stage blocked: set API_APPROVED=1 after bridge generation review.\n' | tee -a "${LOG}"
        exit 2
    fi
}

read_locked_value() {
    local json_key="$1"
    local env_name="$2"
    local dry_run_fallback="$3"
    local env_value="${!env_name:-}"
    if [[ -n "${env_value}" ]]; then
        printf '%s\n' "${env_value}"
    elif [[ -f "${LOCKED_CONFIG}" ]]; then
        "${UV_RUN[@]}" python -c \
            'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' \
            "${LOCKED_CONFIG}" "${json_key}"
    elif [[ "${DRY_RUN}" == "1" ]]; then
        printf '%s\n' "${dry_run_fallback}"
    else
        printf 'Missing %s and %s is absent. Run sweep or set %s.\n' \
            "${env_name}" "${LOCKED_CONFIG}" "${env_name}" >&2
        exit 2
    fi
}

format_alpha_label() {
    local alpha="$1"
    if [[ "${DRY_RUN}" == "1" ]]; then
        if [[ "${alpha}" =~ ^-?[0-9]+$ ]]; then
            printf '%s.0\n' "${alpha}"
        else
            printf '%s\n' "${alpha}"
        fi
        return 0
    fi
    "${UV_RUN[@]}" python -c \
        'import sys; from utils import format_alpha_label; print(format_alpha_label(float(sys.argv[1])))' \
        "${alpha}"
}

require_mc_gate_passed() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        run_logged test -f "${MC1_REPORT_PATH}"
        return 0
    fi
    "${UV_RUN[@]}" python -c '
import json, sys
path = sys.argv[1]
report = json.load(open(path, encoding="utf-8"))
gate = report.get("gate", {})
if gate.get("mode") != "mc1_positive_ci" or gate.get("passed") is not True:
    raise SystemExit(f"TruthfulQA MC gate failed or missing in {path}: {gate}")
print(
    "TruthfulQA MC gate passed: estimate=%+.1fpp, lower=%+.1fpp"
    % (gate.get("estimate_pp", 0.0), gate.get("ci_lower_pp", 0.0))
)
' "${MC1_REPORT_PATH}" 2>&1 | tee -a "${LOG}"
}

run_truthfulqa_variant() {
    local variant="$1"
    local manifest="$2"
    local output_dir="$3"
    local locked_k="$4"
    local locked_alpha="$5"
    run_stage truthfulqa_mc "${UV_RUN[@]}" python scripts/run_intervention.py \
        --model_key "${MODEL_KEY}" \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --benchmark truthfulqa_mc \
        --truthfulqa_variant "${variant}" \
        --sample_manifest "${manifest}" \
        --truthfulqa_fold_path "${ANCHOR2_FOLD_PATH}" \
        --intervention_mode iti_head \
        --iti_head_path "${ITI_ARTIFACT_PATH}" \
        --iti_k "${locked_k}" \
        --iti_family truthfulqa_paperfaithful \
        --iti_decode_scope first_3_tokens \
        --alphas 0.0 "${locked_alpha}" \
        --output_dir "${output_dir}" \
        --run_profile canonical
}

printf 'Mistral Anchor 2 wrapper log: %s\n' "${LOG}" | tee -a "${LOG}"
printf 'DRY_RUN=%s STAGES=%s UV_RUNTIME_MODE=%s PROJECT_DIR=%s OUTPUT_ROOT=%s\n' \
    "${DRY_RUN}" "${STAGES}" "${UV_RUNTIME_MODE}" "${PROJECT_DIR}" "${OUTPUT_ROOT}" | tee -a "${LOG}"

run_logged "${PIPELINE[@]}" active-run-status
run_logged "${PIPELINE[@]}" validate-sample-locks \
    "${TRUTHFULQA_MC1_MANIFEST}" \
    "${TRUTHFULQA_MC2_MANIFEST}" \
    "${TRIVIAQA_BRIDGE_MANIFEST}"
run_stage preflight "${UV_RUN[@]}" python -c \
    'import torch, transformers; print({"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None})'

if should_run_any_stage "${GPU_STAGES[@]}"; then
    require_run_approved
    run_logged "${PIPELINE[@]}" gpu-hardware-guard \
        --min-memory-gib "${GPU_MIN_MEMORY_GIB}" \
        --name-pattern "${GPU_NAME_PATTERN}"
fi
if should_run_any_stage "${API_STAGES[@]}"; then
    require_api_approved
fi

run_stage build_manifests "${UV_RUN[@]}" python scripts/build_truthfulqa_calibration_splits.py \
    --seed "${SEED}" \
    --output-dir "${MANIFEST_DIR}" \
    --anchor2-heldout-mc1-manifest "${TRUTHFULQA_MC1_MANIFEST}" \
    --anchor2-heldout-mc2-manifest "${TRUTHFULQA_MC2_MANIFEST}" \
    --anchor2-prefix "${ANCHOR2_PREFIX}" \
    --only-anchor2

run_stage extract "${UV_RUN[@]}" python scripts/extract_truthfulness_iti.py \
    --family iti_truthfulqa_paperfaithful \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --fold_path "${ANCHOR2_FOLD_PATH}" \
    --output_dir "${ITI_ARTIFACT_DIR}" \
    --seed "${SEED}"

run_stage sweep "${UV_RUN[@]}" python scripts/run_calibration_sweep.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --artifact_path "${ITI_ARTIFACT_PATH}" \
    --cal_val_mc1_manifest "${ANCHOR2_CAL_MC1_MANIFEST}" \
    --cal_val_mc2_manifest "${ANCHOR2_CAL_MC2_MANIFEST}" \
    --k_values "${K_VALUES[@]}" \
    --alpha_values "${ALPHA_VALUES[@]}" \
    --output_dir "${SWEEP_DIR}"

LOCKED_K_VALUE="$(read_locked_value K_locked LOCKED_K 12)"
LOCKED_ALPHA_VALUE="$(read_locked_value alpha_locked LOCKED_ALPHA 8.0)"
LOCKED_ALPHA_LABEL="$(format_alpha_label "${LOCKED_ALPHA_VALUE}")"

run_truthfulqa_variant mc1 "${TRUTHFULQA_MC1_MANIFEST}" "${MC1_OUTPUT_DIR}" \
    "${LOCKED_K_VALUE}" "${LOCKED_ALPHA_VALUE}"
run_truthfulqa_variant mc2 "${TRUTHFULQA_MC2_MANIFEST}" "${MC2_OUTPUT_DIR}" \
    "${LOCKED_K_VALUE}" "${LOCKED_ALPHA_VALUE}"

run_stage truthfulqa_report "${UV_RUN[@]}" python scripts/report_iti_2fold.py \
    --fold_dirs "${MC1_OUTPUT_DIR}" \
    --locked_alpha "${LOCKED_ALPHA_VALUE}" \
    --locked_k "${LOCKED_K_VALUE}" \
    --variant mc1 \
    --output_path "${MC1_REPORT_PATH}" \
    --gate_mode mc1_positive_ci
run_stage truthfulqa_report "${UV_RUN[@]}" python scripts/report_iti_2fold.py \
    --fold_dirs "${MC2_OUTPUT_DIR}" \
    --locked_alpha "${LOCKED_ALPHA_VALUE}" \
    --locked_k "${LOCKED_K_VALUE}" \
    --variant mc2 \
    --output_path "${MC2_REPORT_PATH}"

if should_run_any_stage bridge_generate bridge_judge bridge_irr_prepare; then
    require_mc_gate_passed
fi

run_stage bridge_generate "${UV_RUN[@]}" python scripts/run_intervention.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --benchmark triviaqa_bridge \
    --triviaqa_bridge_manifest "${TRIVIAQA_BRIDGE_MANIFEST}" \
    --intervention_mode iti_head \
    --iti_head_path "${ITI_ARTIFACT_PATH}" \
    --iti_k "${LOCKED_K_VALUE}" \
    --iti_family truthfulqa_paperfaithful \
    --iti_decode_scope first_3_tokens \
    --alphas 0.0 "${LOCKED_ALPHA_VALUE}" \
    --output_dir "${BRIDGE_OUTPUT_DIR}" \
    --run_profile canonical

BRIDGE_JUDGE_ARGS=(
    "${UV_RUN[@]}" python scripts/evaluate_intervention.py
    --benchmark triviaqa_bridge
    --input_dir "${BRIDGE_OUTPUT_DIR}"
    --alphas 0.0 "${LOCKED_ALPHA_VALUE}"
    --api-mode batch
    --judge_model "${JUDGE_MODEL}"
)
if [[ -n "${OPENAI_BATCH_MAX_ENQUEUED_TOKENS}" ]]; then
    BRIDGE_JUDGE_ARGS+=(--batch-max-enqueued-tokens "${OPENAI_BATCH_MAX_ENQUEUED_TOKENS}")
fi
run_stage bridge_judge "${BRIDGE_JUDGE_ARGS[@]}"

run_stage bridge_irr_prepare "${UV_RUN[@]}" python scripts/prepare_bridge_irr_queue.py \
    --skip_dev \
    --test_baseline "${BRIDGE_OUTPUT_DIR}/alpha_0.0.jsonl" \
    --test_comparison "${BRIDGE_OUTPUT_DIR}/alpha_${LOCKED_ALPHA_LABEL}.jsonl" \
    --output_dir "${IRR_DIR}" \
    --adjudication_rule_path "${ADJUDICATION_RULE_PATH}" \
    --seed "${SEED}"

printf '\nDone. Locked K=%s alpha=%s. MC1 report: %s. Bridge dir: %s\n' \
    "${LOCKED_K_VALUE}" "${LOCKED_ALPHA_VALUE}" "${MC1_REPORT_PATH}" "${BRIDGE_OUTPUT_DIR}" | tee -a "${LOG}"

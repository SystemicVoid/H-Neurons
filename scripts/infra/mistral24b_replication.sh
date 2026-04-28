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

PIPELINE=(uv run python -m scripts.lib.pipeline)
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-logs}"
LOG="${LOG_DIR}/mistral24b_replication_${RUN_TS}.log"

MODEL_KEY="${MODEL_KEY:-mistral_small_24b_instruct_2501}"
MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-Small-24B-Instruct-2501}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/mistral24b}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-${OUTPUT_ROOT}/preflight}"
ANSWER_TOKENS="${ANSWER_TOKENS:-${OUTPUT_ROOT}/answer_tokens_llm.jsonl}"
TRAIN_IDS="${TRAIN_IDS:-${OUTPUT_ROOT}/pipeline/train_qids_llm.json}"
DEV_IDS="${DEV_IDS:-${OUTPUT_ROOT}/pipeline/dev_qids_llm.json}"
TEST_IDS="${TEST_IDS:-${OUTPUT_ROOT}/pipeline/test_qids_llm.json}"
ACT_ROOT="${ACT_ROOT:-${OUTPUT_ROOT}/pipeline/activations_llm_canonical}"
TOKEN_SPAN_AUDIT="${TOKEN_SPAN_AUDIT:-${PREFLIGHT_DIR}/token_span_audit.jsonl}"
TOKEN_SPAN_AUDIT_SUMMARY="${TOKEN_SPAN_AUDIT_SUMMARY:-${PREFLIGHT_DIR}/token_span_audit_summary.json}"
RENDERED_CHAT_EXAMPLES="${RENDERED_CHAT_EXAMPLES:-${PREFLIGHT_DIR}/rendered_chat_examples.json}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-models/mistral24b_classifier_canonical.pkl}"
CLASSIFIER_DEV_METRICS="${CLASSIFIER_DEV_METRICS:-${OUTPUT_ROOT}/pipeline/classifier_canonical_dev_metrics.json}"
CLASSIFIER_TEST_METRICS="${CLASSIFIER_TEST_METRICS:-${OUTPUT_ROOT}/pipeline/classifier_canonical_test_metrics.json}"
SPLIT_SAMPLES="${SPLIT_SAMPLES:-560}"
DEV_SAMPLES="${DEV_SAMPLES:-160}"
TEST_SAMPLES="${TEST_SAMPLES:-160}"
TOKEN_SPAN_AUDIT_MAX_SAMPLES="${TOKEN_SPAN_AUDIT_MAX_SAMPLES:-50}"
INTERVENTION_MAX_SAMPLES="${INTERVENTION_MAX_SAMPLES:-100}"
STAGES="${STAGES:-all}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a ALPHAS <<<"${ALPHAS:-0.0 0.5 1.0 1.5 2.0 2.5 3.0}"
read -r -a C_VALUES <<<"${C_VALUES:-0.001 0.005 0.01 0.05 0.1 0.5 1.0}"

mkdir -p "${LOG_DIR}" "${PREFLIGHT_DIR}" "${OUTPUT_ROOT}/pipeline" "${OUTPUT_ROOT}/intervention"

should_run_stage() {
    local stage="$1"
    [[ "${STAGES}" == "all" || ",${STAGES}," == *",${stage},"* ]]
}

run_logged() {
    printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
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
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${stage}" "$*" | tee -a "${LOG}"
    fi
}

capture_versions() {
    {
        date -u
        git rev-parse HEAD || true
        git status --short || true
        uv --version
        uv run python -V
        uv run python - <<'PY'
import importlib.metadata as md

for name in ["torch", "transformers", "accelerate", "datasets", "scikit-learn"]:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=<not installed>")
PY
    } >"${OUTPUT_ROOT}/pipeline/environment_${RUN_TS}.txt" 2>&1
}

run_logged "${PIPELINE[@]}" active-run-status
run_logged "${PIPELINE[@]}" check-active-run-git-guard
run_logged "${PIPELINE[@]}" gpu-preflight || true
if command -v nvitop &>/dev/null; then
    run_logged nvitop -1 || true
fi
capture_versions

run_stage preflight uv run python scripts/audit_token_spans.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --input_path "${ANSWER_TOKENS}" \
    --output_path "${TOKEN_SPAN_AUDIT}" \
    --summary_path "${TOKEN_SPAN_AUDIT_SUMMARY}" \
    --rendered_chat_path "${RENDERED_CHAT_EXAMPLES}" \
    --max_samples "${TOKEN_SPAN_AUDIT_MAX_SAMPLES}"

run_stage splits uv run python scripts/sample_balanced_ids.py \
    --input_path "${ANSWER_TOKENS}" \
    --output_path "${TRAIN_IDS}" \
    --num_samples "${SPLIT_SAMPLES}" \
    --seed 42

run_stage splits uv run python scripts/sample_balanced_ids.py \
    --input_path "${ANSWER_TOKENS}" \
    --output_path "${DEV_IDS}" \
    --num_samples "${DEV_SAMPLES}" \
    --seed 43 \
    --exclude_path "${TRAIN_IDS}"

run_stage splits uv run python scripts/sample_balanced_ids.py \
    --input_path "${ANSWER_TOKENS}" \
    --output_path "${TEST_IDS}" \
    --num_samples "${TEST_SAMPLES}" \
    --seed 44 \
    --exclude_path "${TRAIN_IDS}" "${DEV_IDS}"

run_stage activations uv run python scripts/extract_activations.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --input_path "${ANSWER_TOKENS}" \
    --train_ids_path "${TRAIN_IDS}" \
    --output_root "${ACT_ROOT}/train" \
    --device_map "${DEVICE_MAP}" \
    --locations answer_tokens all_except_answer_tokens

run_stage activations uv run python scripts/extract_activations.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --input_path "${ANSWER_TOKENS}" \
    --train_ids_path "${DEV_IDS}" \
    --output_root "${ACT_ROOT}/dev" \
    --device_map "${DEVICE_MAP}" \
    --locations answer_tokens

run_stage activations uv run python scripts/extract_activations.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --input_path "${ANSWER_TOKENS}" \
    --train_ids_path "${TEST_IDS}" \
    --output_root "${ACT_ROOT}/test" \
    --device_map "${DEVICE_MAP}" \
    --locations answer_tokens

run_stage classifier uv run python scripts/classifier.py \
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

run_stage classifier uv run python scripts/classifier.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --load_model "${CLASSIFIER_PATH}" \
    --test_ids "${TEST_IDS}" \
    --test_acts "${ACT_ROOT}/test/answer_tokens" \
    --metrics_out "${CLASSIFIER_TEST_METRICS}"

run_stage faitheval uv run python scripts/run_intervention.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --benchmark faitheval \
    --prompt_style standard \
    --alphas "${ALPHAS[@]}" \
    --max_samples "${INTERVENTION_MAX_SAMPLES}" \
    --output_dir "${OUTPUT_ROOT}/intervention/faitheval/experiment"

run_stage faitheval_controls uv run python scripts/run_negative_control.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --benchmark faitheval \
    --prompt_style standard \
    --max_samples "${INTERVENTION_MAX_SAMPLES}" \
    --output_base "${OUTPUT_ROOT}/intervention/faitheval/control" \
    --h_neuron_baseline "${OUTPUT_ROOT}/intervention/faitheval/experiment"

run_stage falseqa uv run python scripts/run_intervention.py \
    --model_key "${MODEL_KEY}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --benchmark falseqa \
    --alphas 0.0 1.0 3.0 \
    --max_samples "${INTERVENTION_MAX_SAMPLES}" \
    --output_dir "${OUTPUT_ROOT}/intervention/falseqa/experiment"

run_stage falseqa_controls uv run python scripts/run_negative_control.py \
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

#!/usr/bin/env bash
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle --why="jailbreak-measurement-cleanup" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

STAGE="${1:-all}"
PIPELINE=(uv run python -m scripts.lib.pipeline)
HELPER=(uv run python scripts/jailbreak_measurement_cleanup.py)

STATE_ROOT="data/judge_validation/jailbreak_measurement_cleanup"
SENTINEL_DIR="${STATE_ROOT}/sentinels"
MANIFEST="data/manifests/jbb_d7_full_harmful500_seed42.json"

H_SOURCE="data/gemma3_4b/intervention/jailbreak/experiment"
SEED1_SOURCE="data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained"

H_CANARY_INPUT="${STATE_ROOT}/canary_inputs/h_neuron"
SEED1_CANARY_INPUT="${STATE_ROOT}/canary_inputs/seed_1_control"
H_CANARY_V3="${STATE_ROOT}/canary_v3/h_neuron"
SEED1_CANARY_V3="${STATE_ROOT}/canary_v3/seed_1_control"

H_V3_DIR="data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation"
SEED1_V3_DIR="data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained_csv2_v3"

STRONGREJECT_OUT="data/judge_validation/strongreject"
STRONGREJECT_GOLD="tests/gold_labels/jailbreak_cross_alpha_gold.jsonl"

ALPHAS=(0.0 1.0 1.5 3.0)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${SENTINEL_DIR}"

PRE_CANARY_LOG="${LOG_DIR}/jailbreak_measurement_cleanup_pre_canary_${TIMESTAMP}.log"
POST_CANARY_LOG="${LOG_DIR}/jailbreak_measurement_cleanup_post_canary_${TIMESTAMP}.log"
STRONGREJECT_LOG="${LOG_DIR}/jailbreak_measurement_cleanup_strongreject_${TIMESTAMP}.log"

run_with_log() {
    local log_file="$1"
    shift
    PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "${log_file}"
}

invalidate_canary_state() {
    rm -rf "${H_CANARY_V3}" "${SEED1_CANARY_V3}"
    rm -f \
        "${STATE_ROOT}/canary_summary.json" \
        "${STATE_ROOT}/canary_report.md"
}

check_source_stage() {
    local dir="$1"
    if ! "${PIPELINE[@]}" check-stage --output-dir "${dir}" --manifest "${MANIFEST}" --alphas "${ALPHAS[@]}"; then
        echo "FATAL: expected complete alpha files in ${dir}" >&2
        exit 1
    fi
}

maybe_stop() {
    local sentinel_name="$1"
    if "${PIPELINE[@]}" check-sentinel --dir "${SENTINEL_DIR}" --name "${sentinel_name}"; then
        echo "Sentinel ${sentinel_name} detected. Stopping cleanly."
        exit 0
    fi
}

run_pre_canary() {
    echo "=== Pre-canary: invalidate prior canary outputs ===" | tee -a "${PRE_CANARY_LOG}"
    invalidate_canary_state

    echo "=== Pre-canary: source preflight ===" | tee -a "${PRE_CANARY_LOG}"
    check_source_stage "${H_SOURCE}"
    check_source_stage "${SEED1_SOURCE}"

    echo "=== Pre-canary: build deterministic 20-row subsets ===" | tee -a "${PRE_CANARY_LOG}"
    run_with_log "${PRE_CANARY_LOG}" "${HELPER[@]}" build-canary \
        --state-root "${STATE_ROOT}" \
        --h-neuron-source "${H_SOURCE}" \
        --seed1-source "${SEED1_SOURCE}" \
        --alphas "${ALPHAS[@]}" \
        --canary-rows 20

    echo "=== Pre-canary: sync v3 on H-neuron subset ===" | tee -a "${PRE_CANARY_LOG}"
    run_with_log "${PRE_CANARY_LOG}" uv run python scripts/evaluate_csv2.py \
        --input_dir "${H_CANARY_INPUT}" \
        --output_dir "${H_CANARY_V3}" \
        --alphas "${ALPHAS[@]}" \
        --judge_model gpt-4o \
        --api-mode fast

    echo "=== Pre-canary: sync v3 on seed-1 control subset ===" | tee -a "${PRE_CANARY_LOG}"
    run_with_log "${PRE_CANARY_LOG}" uv run python scripts/evaluate_csv2.py \
        --input_dir "${SEED1_CANARY_INPUT}" \
        --output_dir "${SEED1_CANARY_V3}" \
        --alphas "${ALPHAS[@]}" \
        --judge_model gpt-4o \
        --api-mode fast

    echo "=== Pre-canary: validate canary ===" | tee -a "${PRE_CANARY_LOG}"
    run_with_log "${PRE_CANARY_LOG}" "${HELPER[@]}" validate-canary \
        --state-root "${STATE_ROOT}" \
        --alphas "${ALPHAS[@]}"
}

run_post_canary() {
    echo "=== Post-canary: require pass artifact ===" | tee -a "${POST_CANARY_LOG}"
    run_with_log "${POST_CANARY_LOG}" "${HELPER[@]}" require-canary-pass --state-root "${STATE_ROOT}"

    if [ "${CODEX_VERIFY_OPENAI_LIMITS:-1}" = "1" ]; then
        echo "=== Post-canary: verify OpenAI Batch limits ===" | tee -a "${POST_CANARY_LOG}"
        scripts/infra/check_openai_batch_limits_via_codex.sh 2>&1 | tee -a "${POST_CANARY_LOG}"
    else
        echo "Skipping OpenAI Batch limit check (CODEX_VERIFY_OPENAI_LIMITS=0)" | tee -a "${POST_CANARY_LOG}"
    fi

    echo "=== Post-canary: full v3 rescoring on H-neuron outputs ===" | tee -a "${POST_CANARY_LOG}"
    run_with_log "${POST_CANARY_LOG}" uv run python scripts/evaluate_csv2.py \
        --input_dir "${H_SOURCE}" \
        --output_dir "${H_V3_DIR}" \
        --alphas "${ALPHAS[@]}" \
        --judge_model gpt-4o \
        --api-mode batch

    echo "=== Post-canary: validate H-neuron v3 outputs ===" | tee -a "${POST_CANARY_LOG}"
    run_with_log "${POST_CANARY_LOG}" "${HELPER[@]}" validate-scored-dir \
        --input-dir "${H_SOURCE}" \
        --output-dir "${H_V3_DIR}" \
        --alphas "${ALPHAS[@]}"

    echo "=== Post-canary: full v3 rescoring on seed-1 control ===" | tee -a "${POST_CANARY_LOG}"
    run_with_log "${POST_CANARY_LOG}" uv run python scripts/evaluate_csv2.py \
        --input_dir "${SEED1_SOURCE}" \
        --output_dir "${SEED1_V3_DIR}" \
        --alphas "${ALPHAS[@]}" \
        --judge_model gpt-4o \
        --api-mode batch

    echo "=== Post-canary: validate seed-1 control v3 outputs ===" | tee -a "${POST_CANARY_LOG}"
    run_with_log "${POST_CANARY_LOG}" "${HELPER[@]}" validate-scored-dir \
        --input-dir "${SEED1_SOURCE}" \
        --output-dir "${SEED1_V3_DIR}" \
        --alphas "${ALPHAS[@]}"

    "${PIPELINE[@]}" log-run \
        --run-dir "${H_V3_DIR}" \
        --description "jailbreak h-neuron CSV2 v3 rescore, alphas 0.0/1.0/1.5/3.0, gpt-4o batch"
    "${PIPELINE[@]}" log-run \
        --run-dir "${SEED1_V3_DIR}" \
        --description "jailbreak seed-1 random control CSV2 v3 rescore, alphas 0.0/1.0/1.5/3.0, gpt-4o batch"
}

run_strongreject() {
    echo "=== StrongREJECT: gold rerun on gpt-4o ===" | tee -a "${STRONGREJECT_LOG}"
    run_with_log "${STRONGREJECT_LOG}" uv run python scripts/evaluate_strongreject.py \
        --gold_path "${STRONGREJECT_GOLD}" \
        --output_dir "${STRONGREJECT_OUT}" \
        --judge_model gpt-4o \
        --api-mode batch
}

case "${STAGE}" in
    pre-canary)
        run_pre_canary
        ;;
    post-canary)
        run_post_canary
        ;;
    strongreject)
        maybe_stop "stop_before_strongreject"
        run_strongreject
        ;;
    all)
        run_pre_canary
        maybe_stop "stop_after_pre_canary"
        run_post_canary
        maybe_stop "stop_after_post_canary"
        maybe_stop "stop_before_strongreject"
        run_strongreject
        ;;
    *)
        echo "Usage: $0 [pre-canary|post-canary|strongreject|all]" >&2
        exit 2
        ;;
esac

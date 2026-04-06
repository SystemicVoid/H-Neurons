#!/usr/bin/env bash
# D7 causal pilot orchestrator (JBB paired, staged 100 -> 500).
#
# Stages:
#   1) Build deterministic manifests (extraction/pilot/full)
#   2) Extract probe + causal ITI artifacts
#   3) Pilot lock on 100 harmful prompts (alphas 0,1,2,4,8)
#   4) Full 500 evaluation for baseline/L1/probe/causal/random-head controls
#   5) Paired D7 CSV2 report
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="D7 causal pilot staged run" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

SEED="${D7_SEED:-42}"
MODEL_PATH="${D7_MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${D7_DEVICE_MAP:-cuda:0}"
L1_ALPHA="${D7_L1_ALPHA:-3.0}"
RESUME="${D7_RESUME:-1}"

MANIFEST_DIR="data/manifests"
EXTRACTION_PAIRS="${MANIFEST_DIR}/jbb_d7_extraction_pairs_seed${SEED}.jsonl"
PILOT_IDS="${MANIFEST_DIR}/jbb_d7_pilot_harmful100_seed${SEED}.json"
FULL_IDS="${MANIFEST_DIR}/jbb_d7_full_harmful500_seed${SEED}.json"

CAUSAL_ART_DIR="data/contrastive/refusal/iti_refusal_causal_d7"
PROBE_ART_DIR="data/contrastive/refusal/iti_refusal_probe_d7"
CAUSAL_ARTIFACT="${CAUSAL_ART_DIR}/iti_heads.pt"
PROBE_ARTIFACT="${PROBE_ART_DIR}/iti_heads.pt"

RUN_ROOT="data/gemma3_4b/intervention/jailbreak_d7"
PILOT_ROOT="${RUN_ROOT}/pilot100_canonical"
FULL_ROOT="${RUN_ROOT}/full500_canonical"

PILOT_PROBE_DIR="${PILOT_ROOT}/probe/experiment"
PILOT_CAUSAL_DIR="${PILOT_ROOT}/causal/experiment"
PILOT_PROBE_CSV2="${PILOT_ROOT}/probe/csv2_evaluation"
PILOT_CAUSAL_CSV2="${PILOT_ROOT}/causal/csv2_evaluation"
PILOT_ALPHAS=(0.0 1.0 2.0 4.0 8.0)

FULL_BASELINE_DIR="${FULL_ROOT}/baseline_noop/experiment"
FULL_L1_DIR="${FULL_ROOT}/l1_neuron/experiment"
FULL_PROBE_DIR="${FULL_ROOT}/probe_locked/experiment"
FULL_CAUSAL_DIR="${FULL_ROOT}/causal_locked/experiment"
FULL_RANDOM_ROOT="${FULL_ROOT}/causal_random_head"
FULL_REPORT="${FULL_ROOT}/d7_csv2_report.json"

JAILBREAK_DECODE_ARGS=(
    --jailbreak_do_sample true
    --jailbreak_temperature 0.7
    --jailbreak_top_k 20
    --jailbreak_top_p 0.8
    --max_new_tokens 5000
)
RUN_PROFILE_ARGS=(--run_profile canonical)
ITI_DEBUG_ARGS=()
if [[ "${D7_ALLOW_CANONICAL_ITI_DEBUG:-0}" == "1" ]]; then
    echo "WARNING: enabling ITI debug traces with fast profile via D7_ALLOW_CANONICAL_ITI_DEBUG=1"
    # run_profile=canonical rejects --iti_collect_debug_stats; keep canonical decode knobs for diagnostics.
    RUN_PROFILE_ARGS=(--run_profile fast --jailbreak_batch_size 1)
    ITI_DEBUG_ARGS=(--iti_collect_debug_stats)
fi

mkdir -p logs "${PILOT_ROOT}" "${FULL_ROOT}"
LOG="logs/d7_causal_pilot_$(date +%Y%m%d_%H%M%S).log"

run_cmd() {
    echo "+ $*"
    "$@"
}

run_judge_and_csv2() {
    local dir="$1"
    shift
    local alphas=("$@")
    run_cmd uv run python scripts/evaluate_intervention.py \
        --benchmark jailbreak \
        --input_dir "${dir}" \
        --alphas "${alphas[@]}" \
        --api-mode batch 2>&1 | tee -a "${LOG}"
    run_cmd uv run python scripts/evaluate_csv2.py \
        --input_dir "${dir}" \
        --output_dir "${dir%/experiment}/csv2_evaluation" \
        --alphas "${alphas[@]}" \
        --api-mode batch 2>&1 | tee -a "${LOG}"
}

assert_dir_ready() {
    local dir="$1"
    mkdir -p "${dir}"
    if [[ "${RESUME}" != "1" ]] && compgen -G "${dir}/alpha_*.jsonl" >/dev/null; then
        echo "FATAL: output collision at ${dir}" >&2
        echo "Archive existing run directory before re-running (set D7_RESUME=1 to resume)." >&2
        exit 1
    fi
}

ensure_generation_files() {
    local dir="$1"
    shift
    local missing=0
    for alpha in "$@"; do
        if [[ ! -f "${dir}/alpha_${alpha}.jsonl" ]]; then
            missing=1
        fi
    done
    return ${missing}
}

echo "=== D7 causal pilot ==="
echo "seed=${SEED} model=${MODEL_PATH} device_map=${DEVICE_MAP} resume=${RESUME}"
echo "log=${LOG}"

if command -v nvitop &>/dev/null; then
    echo "--- GPU preflight ---" | tee -a "${LOG}"
    nvitop -1 2>&1 | tee -a "${LOG}" || true
fi

# Stage 1: build manifests
if [[ ! -f "${EXTRACTION_PAIRS}" || ! -f "${PILOT_IDS}" || ! -f "${FULL_IDS}" ]]; then
    run_cmd uv run python scripts/build_d7_jbb_manifests.py \
        --seed "${SEED}" \
        --output_dir "${MANIFEST_DIR}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage 1: manifests already present; skipping" | tee -a "${LOG}"
fi

# Stage 2: extract artifacts
if [[ ! -f "${PROBE_ARTIFACT}" ]]; then
    run_cmd uv run python scripts/extract_truthfulness_iti.py \
        --family iti_refusal_probe \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --seed "${SEED}" \
        --d7_manifest_path "${EXTRACTION_PAIRS}" \
        --output_dir "${PROBE_ART_DIR}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage 2: probe artifact exists; skipping" | tee -a "${LOG}"
fi

if [[ ! -f "${CAUSAL_ARTIFACT}" ]]; then
    run_cmd uv run python scripts/extract_truthfulness_iti.py \
        --family iti_refusal_causal \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --seed "${SEED}" \
        --d7_manifest_path "${EXTRACTION_PAIRS}" \
        --output_dir "${CAUSAL_ART_DIR}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage 2: causal artifact exists; skipping" | tee -a "${LOG}"
fi

# Stage 3: pilot lock (probe + causal)
assert_dir_ready "${PILOT_PROBE_DIR}"
assert_dir_ready "${PILOT_CAUSAL_DIR}"

if ! ensure_generation_files "${PILOT_PROBE_DIR}" "0.0" "1.0" "2.0" "4.0" "8.0"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${PILOT_PROBE_DIR}" \
        --sample_manifest "${PILOT_IDS}" \
        --intervention_mode iti_head \
        --iti_head_path "${PROBE_ARTIFACT}" \
        --iti_family refusal_probe \
        --iti_k 20 \
        --alphas "${PILOT_ALPHAS[@]}" \
        "${ITI_DEBUG_ARGS[@]}" \
        "${RUN_PROFILE_ARGS[@]}" \
        "${JAILBREAK_DECODE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage 3: probe pilot generation complete; skipping" | tee -a "${LOG}"
fi

if ! ensure_generation_files "${PILOT_CAUSAL_DIR}" "0.0" "1.0" "2.0" "4.0" "8.0"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${PILOT_CAUSAL_DIR}" \
        --sample_manifest "${PILOT_IDS}" \
        --intervention_mode iti_head \
        --iti_head_path "${CAUSAL_ARTIFACT}" \
        --iti_family refusal_causal \
        --iti_k 20 \
        --alphas "${PILOT_ALPHAS[@]}" \
        "${ITI_DEBUG_ARGS[@]}" \
        "${RUN_PROFILE_ARGS[@]}" \
        "${JAILBREAK_DECODE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage 3: causal pilot generation complete; skipping" | tee -a "${LOG}"
fi

run_cmd uv run python scripts/evaluate_intervention.py \
    --benchmark jailbreak \
    --input_dir "${PILOT_PROBE_DIR}" \
    --alphas "${PILOT_ALPHAS[@]}" \
    --api-mode batch 2>&1 | tee -a "${LOG}"

run_cmd uv run python scripts/evaluate_intervention.py \
    --benchmark jailbreak \
    --input_dir "${PILOT_CAUSAL_DIR}" \
    --alphas "${PILOT_ALPHAS[@]}" \
    --api-mode batch 2>&1 | tee -a "${LOG}"

run_cmd uv run python scripts/evaluate_csv2.py \
    --input_dir "${PILOT_PROBE_DIR}" \
    --output_dir "${PILOT_PROBE_CSV2}" \
    --alphas "${PILOT_ALPHAS[@]}" \
    --api-mode batch 2>&1 | tee -a "${LOG}"

run_cmd uv run python scripts/evaluate_csv2.py \
    --input_dir "${PILOT_CAUSAL_DIR}" \
    --output_dir "${PILOT_CAUSAL_CSV2}" \
    --alphas "${PILOT_ALPHAS[@]}" \
    --api-mode batch 2>&1 | tee -a "${LOG}"

PROBE_LOCK_JSON="${PILOT_ROOT}/probe_lock.json"
CAUSAL_LOCK_JSON="${PILOT_ROOT}/causal_lock.json"
run_cmd uv run python scripts/lock_d7_alpha.py \
    --csv2_dir "${PILOT_PROBE_CSV2}" \
    --candidate_alphas 1.0 2.0 4.0 8.0 \
    --output_path "${PROBE_LOCK_JSON}" 2>&1 | tee -a "${LOG}"
run_cmd uv run python scripts/lock_d7_alpha.py \
    --csv2_dir "${PILOT_CAUSAL_CSV2}" \
    --candidate_alphas 1.0 2.0 4.0 8.0 \
    --output_path "${CAUSAL_LOCK_JSON}" 2>&1 | tee -a "${LOG}"

PROBE_LOCKED_ALPHA=$(uv run python - <<'PY' "${PROBE_LOCK_JSON}"
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["selected_alpha"])
PY
)
CAUSAL_LOCKED_ALPHA=$(uv run python - <<'PY' "${CAUSAL_LOCK_JSON}"
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["selected_alpha"])
PY
)

echo "Locked alphas: probe=${PROBE_LOCKED_ALPHA}, causal=${CAUSAL_LOCKED_ALPHA}" | tee -a "${LOG}"

# Stage 4: full 500 conditions
assert_dir_ready "${FULL_BASELINE_DIR}"
assert_dir_ready "${FULL_L1_DIR}"
assert_dir_ready "${FULL_PROBE_DIR}"
assert_dir_ready "${FULL_CAUSAL_DIR}"
mkdir -p "${FULL_RANDOM_ROOT}"

if ! ensure_generation_files "${FULL_BASELINE_DIR}" "1.0"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${FULL_BASELINE_DIR}" \
        --sample_manifest "${FULL_IDS}" \
        --alphas 1.0 \
        "${RUN_PROFILE_ARGS[@]}" \
        "${JAILBREAK_DECODE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage 4: baseline generation complete; skipping" | tee -a "${LOG}"
fi

if ! ensure_generation_files "${FULL_L1_DIR}" "${L1_ALPHA}"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${FULL_L1_DIR}" \
        --sample_manifest "${FULL_IDS}" \
        --alphas "${L1_ALPHA}" \
        "${RUN_PROFILE_ARGS[@]}" \
        "${JAILBREAK_DECODE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage 4: L1 generation complete; skipping" | tee -a "${LOG}"
fi

if ! ensure_generation_files "${FULL_PROBE_DIR}" "${PROBE_LOCKED_ALPHA}"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${FULL_PROBE_DIR}" \
        --sample_manifest "${FULL_IDS}" \
        --intervention_mode iti_head \
        --iti_head_path "${PROBE_ARTIFACT}" \
        --iti_family refusal_probe \
        --iti_k 20 \
        --alphas "${PROBE_LOCKED_ALPHA}" \
        "${ITI_DEBUG_ARGS[@]}" \
        "${RUN_PROFILE_ARGS[@]}" \
        "${JAILBREAK_DECODE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage 4: probe generation complete; skipping" | tee -a "${LOG}"
fi

if ! ensure_generation_files "${FULL_CAUSAL_DIR}" "${CAUSAL_LOCKED_ALPHA}"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${FULL_CAUSAL_DIR}" \
        --sample_manifest "${FULL_IDS}" \
        --intervention_mode iti_head \
        --iti_head_path "${CAUSAL_ARTIFACT}" \
        --iti_family refusal_causal \
        --iti_k 20 \
        --alphas "${CAUSAL_LOCKED_ALPHA}" \
        "${ITI_DEBUG_ARGS[@]}" \
        "${RUN_PROFILE_ARGS[@]}" \
        "${JAILBREAK_DECODE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage 4: causal generation complete; skipping" | tee -a "${LOG}"
fi

for seed in 1 2 3; do
    RANDOM_DIR="${FULL_RANDOM_ROOT}/seed_${seed}/experiment"
    assert_dir_ready "${RANDOM_DIR}"
    if ! ensure_generation_files "${RANDOM_DIR}" "${CAUSAL_LOCKED_ALPHA}"; then
        run_cmd uv run python scripts/run_intervention.py \
            --benchmark jailbreak \
            --model_path "${MODEL_PATH}" \
            --device_map "${DEVICE_MAP}" \
            --output_dir "${RANDOM_DIR}" \
            --sample_manifest "${FULL_IDS}" \
            --intervention_mode iti_head \
            --iti_head_path "${CAUSAL_ARTIFACT}" \
            --iti_family refusal_causal \
            --iti_k 20 \
            --iti_selection_strategy random \
            --iti_random_seed "${seed}" \
            --alphas "${CAUSAL_LOCKED_ALPHA}" \
            "${ITI_DEBUG_ARGS[@]}" \
            "${RUN_PROFILE_ARGS[@]}" \
            "${JAILBREAK_DECODE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
    else
        echo "Stage 4: random seed=${seed} generation complete; skipping" | tee -a "${LOG}"
    fi
done

# Judge + CSV2 for full runs (failure-visible and condition-specific alpha lists)
run_judge_and_csv2 "${FULL_BASELINE_DIR}" "1.0"
run_judge_and_csv2 "${FULL_L1_DIR}" "${L1_ALPHA}"
run_judge_and_csv2 "${FULL_PROBE_DIR}" "${PROBE_LOCKED_ALPHA}"
run_judge_and_csv2 "${FULL_CAUSAL_DIR}" "${CAUSAL_LOCKED_ALPHA}"
run_judge_and_csv2 "${FULL_RANDOM_ROOT}/seed_1/experiment" "${CAUSAL_LOCKED_ALPHA}"
run_judge_and_csv2 "${FULL_RANDOM_ROOT}/seed_2/experiment" "${CAUSAL_LOCKED_ALPHA}"
run_judge_and_csv2 "${FULL_RANDOM_ROOT}/seed_3/experiment" "${CAUSAL_LOCKED_ALPHA}"

# Stage 5: paired report
run_cmd uv run python scripts/report_d7_csv2.py \
    --baseline_dir "${FULL_BASELINE_DIR%/experiment}/csv2_evaluation" \
    --baseline_alpha 1.0 \
    --condition "l1:${FULL_L1_DIR%/experiment}/csv2_evaluation:${L1_ALPHA}" \
    --condition "probe:${FULL_PROBE_DIR%/experiment}/csv2_evaluation:${PROBE_LOCKED_ALPHA}" \
    --condition "causal:${FULL_CAUSAL_DIR%/experiment}/csv2_evaluation:${CAUSAL_LOCKED_ALPHA}" \
    --condition "random_seed1:${FULL_RANDOM_ROOT}/seed_1/csv2_evaluation:${CAUSAL_LOCKED_ALPHA}" \
    --condition "random_seed2:${FULL_RANDOM_ROOT}/seed_2/csv2_evaluation:${CAUSAL_LOCKED_ALPHA}" \
    --condition "random_seed3:${FULL_RANDOM_ROOT}/seed_3/csv2_evaluation:${CAUSAL_LOCKED_ALPHA}" \
    --output_path "${FULL_REPORT}" 2>&1 | tee -a "${LOG}"

echo "=== D7 causal pilot complete ===" | tee -a "${LOG}"
echo "Report: ${FULL_REPORT}" | tee -a "${LOG}"

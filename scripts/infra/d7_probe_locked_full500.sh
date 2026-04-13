#!/usr/bin/env bash
# Resume and score the full-500 D7 probe branch at its locked alpha.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="D7 full500 probe locked resume" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

PIPELINE="uv run python -m scripts.lib.pipeline"
SEED="${D7_SEED:-42}"
MODEL_PATH="${D7_MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${D7_DEVICE_MAP:-cuda:0}"
RESUME="${D7_RESUME:-1}"

MANIFEST_DIR="data/manifests"
FULL_IDS="${MANIFEST_DIR}/jbb_d7_full_harmful500_seed${SEED}.json"
PROBE_ARTIFACT="data/contrastive/refusal/iti_refusal_probe_d7/iti_heads.pt"
RUN_ROOT="data/gemma3_4b/intervention/jailbreak_d7/full500_canonical"
PROBE_DIR="${RUN_ROOT}/probe_locked/experiment"
REPORT_PATH="${RUN_ROOT}/d7_csv2_report_with_probe.json"
LOG="logs/d7_probe_locked_full500_$(date +%Y%m%d_%H%M%S).log"

JAILBREAK_DECODE_ARGS=(
    --jailbreak_do_sample true
    --jailbreak_temperature 0.7
    --jailbreak_top_k 20
    --jailbreak_top_p 0.8
    --max_new_tokens 5000
)
RUN_PROFILE_ARGS=(--run_profile canonical)

run_cmd() {
    echo "+ $*" | tee -a "${LOG}" >&2
    "$@"
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

run_judge_and_csv2() {
    local dir="$1"
    local alpha="$2"
    run_cmd uv run python scripts/evaluate_intervention.py \
        --benchmark jailbreak \
        --input_dir "${dir}" \
        --alphas "${alpha}" \
        --api-mode batch 2>&1 | tee -a "${LOG}"
    run_cmd uv run python scripts/evaluate_csv2.py \
        --input_dir "${dir}" \
        --output_dir "${dir%/experiment}/csv2_evaluation" \
        --alphas "${alpha}" \
        --api-mode batch 2>&1 | tee -a "${LOG}"
}

echo "=== D7 full500 probe locked resume ===" | tee -a "${LOG}"
echo "seed=${SEED} model=${MODEL_PATH} device_map=${DEVICE_MAP} resume=${RESUME}" | tee -a "${LOG}"
${PIPELINE} gpu-preflight 2>&1 | tee -a "${LOG}" || true

if [[ ! -f "${FULL_IDS}" ]]; then
    run_cmd uv run python scripts/build_d7_jbb_manifests.py \
        --seed "${SEED}" \
        --output_dir "${MANIFEST_DIR}" 2>&1 | tee -a "${LOG}"
fi

PROBE_LOCK_JSON="data/gemma3_4b/intervention/jailbreak_d7/pilot100_canonical/probe_lock.json"
PROBE_LOCKED_ALPHA=$(uv run python - <<'PY' "${PROBE_LOCK_JSON}"
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["selected_alpha"])
PY
)
CAUSAL_LOCK_JSON="data/gemma3_4b/intervention/jailbreak_d7/pilot100_canonical/causal_lock.json"
CAUSAL_LOCKED_ALPHA=$(uv run python - <<'PY' "${CAUSAL_LOCK_JSON}"
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["selected_alpha"])
PY
)
echo "Using probe locked alpha=${PROBE_LOCKED_ALPHA}" | tee -a "${LOG}"
echo "Using causal locked alpha=${CAUSAL_LOCKED_ALPHA}" | tee -a "${LOG}"

assert_dir_ready "${PROBE_DIR}"
if ! ${PIPELINE} check-stage --output-dir "${PROBE_DIR}" --manifest "${FULL_IDS}" --alphas "${PROBE_LOCKED_ALPHA}"; then
    run_cmd uv run python scripts/run_intervention.py \
        --benchmark jailbreak \
        --model_path "${MODEL_PATH}" \
        --device_map "${DEVICE_MAP}" \
        --output_dir "${PROBE_DIR}" \
        --sample_manifest "${FULL_IDS}" \
        --intervention_mode iti_head \
        --iti_head_path "${PROBE_ARTIFACT}" \
        --iti_family refusal_probe \
        --iti_k 20 \
        --alphas "${PROBE_LOCKED_ALPHA}" \
        "${RUN_PROFILE_ARGS[@]}" \
        "${JAILBREAK_DECODE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
else
    echo "Stage: probe generation complete; skipping" | tee -a "${LOG}"
fi

run_judge_and_csv2 "${PROBE_DIR}" "${PROBE_LOCKED_ALPHA}"
${PIPELINE} log-run \
    --run-dir "${PROBE_DIR}" \
    --description "D7 full500 probe locked at alpha ${PROBE_LOCKED_ALPHA}"

run_cmd uv run python scripts/report_d7_csv2.py \
    --baseline_dir "${RUN_ROOT}/baseline_noop/csv2_evaluation" \
    --baseline_alpha 1.0 \
    --condition "l1:${RUN_ROOT}/l1_neuron/csv2_evaluation:3.0" \
    --condition "probe:${RUN_ROOT}/probe_locked/csv2_evaluation:${PROBE_LOCKED_ALPHA}" \
    --condition "causal:${RUN_ROOT}/causal_locked/csv2_evaluation:${CAUSAL_LOCKED_ALPHA}" \
    --output_path "${REPORT_PATH}" 2>&1 | tee -a "${LOG}"

echo "=== D7 full500 probe locked resume complete ===" | tee -a "${LOG}"
echo "Report: ${REPORT_PATH}" | tee -a "${LOG}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TMUX_SESSION="${TMUX_SESSION:-simid-remote-shard}"

if [ -z "${TMUX:-}" ] && [ -z "${TMUX_WRAPPED:-}" ] && command -v tmux &>/dev/null; then
    quoted_args=""
    for arg in "$@"; do
        printf -v quoted_args '%s %q' "${quoted_args}" "${arg}"
    done
    tmux_env_args=(-e "TMUX_WRAPPED=1")
    tmux_env_names=(
        PROJECT_DIR
        OUTPUT_DIR
        MANIFEST
        MODEL_PATH
        DEVICE_MAP
        ITI_ARTIFACT_PATH
        ITI_FAMILY
        ITI_K
        DECODE_SCOPE
        TOP_K_FIRST_TOKEN
        MC_MAX_NEW_TOKENS
        OPEN_MAX_NEW_TOKENS
        SEED
        GPU_MIN_MEMORY_GIB
        GPU_NAME_PATTERN
        RUN_DESCRIPTION
        LOG_DIR
        LOG
        ALPHAS
        CONDITIONS
    )
    for name in "${tmux_env_names[@]}"; do
        if [[ -v ${name} ]]; then
            tmux_env_args+=(-e "${name}=${!name}")
        fi
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
        "${tmux_env_args[@]}" \
        "cd ${PROJECT_DIR@Q} && bash ${0@Q}${quoted_args}"
    printf 'Started tmux session %s. Attach with: tmux attach -t %s\n' \
        "${TMUX_SESSION}" "${TMUX_SESSION}"
    exit 0
fi

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle --why="SIMID remote tail shard" \
        -- bash "$0" "$@"
fi

cd "${PROJECT_DIR}"

UV_RUN=(uv run --no-sync)
PIPELINE=("${UV_RUN[@]}" python -m scripts.lib.pipeline)

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-logs}"
LOG="${LOG:-${LOG_DIR}/simid_remote_shard_${RUN_TS}.log}"
mkdir -p "${LOG_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
MANIFEST="${MANIFEST:?MANIFEST required}"
MODEL_PATH="${MODEL_PATH:-google/gemma-3-4b-it}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
ITI_ARTIFACT_PATH="${ITI_ARTIFACT_PATH:?ITI_ARTIFACT_PATH required}"
ITI_FAMILY="${ITI_FAMILY:-truthfulqa_paperfaithful}"
ITI_K="${ITI_K:-12}"
DECODE_SCOPE="${DECODE_SCOPE:-first_3_tokens}"
TOP_K_FIRST_TOKEN="${TOP_K_FIRST_TOKEN:-10}"
MC_MAX_NEW_TOKENS="${MC_MAX_NEW_TOKENS:-4}"
OPEN_MAX_NEW_TOKENS="${OPEN_MAX_NEW_TOKENS:-64}"
SEED="${SEED:-42}"
GPU_MIN_MEMORY_GIB="${GPU_MIN_MEMORY_GIB:-16}"
GPU_NAME_PATTERN="${GPU_NAME_PATTERN:-GeForce RTX 5090|GeForce RTX 4090|H100|A100}"
RUN_DESCRIPTION="${RUN_DESCRIPTION:-SIMID remote tail shard}"

if [[ -z "${ALPHAS:-}" ]]; then
    echo "ALPHAS required (space-separated, e.g. '-8.0 0.0 4.0 8.0')" >&2
    exit 2
fi
if [[ -z "${CONDITIONS:-}" ]]; then
    echo "CONDITIONS required (space-separated condition names)" >&2
    exit 2
fi
read -r -a ALPHAS_ARR <<<"${ALPHAS}"
read -r -a CONDITIONS_ARR <<<"${CONDITIONS}"

clean_untracked_uv_lock() {
    if [[ -f uv.lock ]] && ! git ls-files --error-unmatch uv.lock >/dev/null 2>&1; then
        rm -f uv.lock
    fi
}
clean_untracked_uv_lock

{
    echo "=== simid_remote_shard.sh ${RUN_TS} ==="
    echo "PROJECT_DIR=${PROJECT_DIR}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "MANIFEST=${MANIFEST}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "ITI_ARTIFACT_PATH=${ITI_ARTIFACT_PATH}"
    echo "ITI_FAMILY=${ITI_FAMILY} ITI_K=${ITI_K} DECODE_SCOPE=${DECODE_SCOPE}"
    echo "ALPHAS=${ALPHAS}"
    echo "CONDITIONS=${CONDITIONS}"
    echo "GPU_MIN_MEMORY_GIB=${GPU_MIN_MEMORY_GIB} GPU_NAME_PATTERN=${GPU_NAME_PATTERN}"
} | tee -a "${LOG}"

"${PIPELINE[@]}" gpu-preflight 2>&1 | tee -a "${LOG}"
"${PIPELINE[@]}" gpu-hardware-guard \
    --min-memory-gib "${GPU_MIN_MEMORY_GIB}" \
    --name-pattern "${GPU_NAME_PATTERN}" 2>&1 | tee -a "${LOG}"

"${UV_RUN[@]}" python scripts/run_simid.py \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --iti-artifact-path "${ITI_ARTIFACT_PATH}" \
    --iti-family "${ITI_FAMILY}" \
    --iti-k "${ITI_K}" \
    --decode-scope "${DECODE_SCOPE}" \
    --device-map "${DEVICE_MAP}" \
    --alphas "${ALPHAS_ARR[@]}" \
    --conditions "${CONDITIONS_ARR[@]}" \
    --top-k-first-token "${TOP_K_FIRST_TOKEN}" \
    --mc-max-new-tokens "${MC_MAX_NEW_TOKENS}" \
    --open-max-new-tokens "${OPEN_MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    2>&1 | tee -a "${LOG}"

"${PIPELINE[@]}" log-run \
    --run-dir "${OUTPUT_DIR}" \
    --description "${RUN_DESCRIPTION}" 2>&1 | tee -a "${LOG}"

echo "=== simid_remote_shard.sh complete ===" | tee -a "${LOG}"

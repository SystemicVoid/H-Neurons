#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v codex >/dev/null 2>&1; then
    echo "ERROR: codex CLI not found on PATH."
    exit 1
fi

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/openai_batch_limit_check_${TIMESTAMP}.log"
SUMMARY_FILE="$LOG_DIR/openai_batch_limit_check_${TIMESTAMP}.summary.txt"
TIMEOUT_DURATION="${OPENAI_LIMIT_CHECK_TIMEOUT:-10m}"
CODEX_MODEL="${CODEX_LIMIT_CHECK_CODEX_MODEL:-}"
TARGET_MODELS="${OPENAI_LIMIT_CHECK_MODELS:-gpt-4o gpt-4o-mini gpt-4.1 gpt-5 gpt-5-mini o3 o4-mini}"

echo "=========================================="
echo "OpenAI Batch Limit Check via Codex CLI"
echo "Started: $(date -Iseconds)"
echo "Repo:    $ROOT_DIR"
echo "Models:  $TARGET_MODELS"
echo "Log:     $LOG_FILE"
echo "Summary: $SUMMARY_FILE"
echo "=========================================="
echo ""

CODEX_ARGS=(
    --search
    -a
    never
    --sandbox
    workspace-write
)

if [[ -n "$CODEX_MODEL" ]]; then
    CODEX_ARGS+=(-m "$CODEX_MODEL")
fi

PROMPT=$(
    cat <<EOF
Verify the current OpenAI Tier 3 Batch queue limits for these models: ${TARGET_MODELS}.

Requirements:
- Use official OpenAI sources only: developers.openai.com and platform.openai.com.
- Compare the verified values against the local registry in scripts/openai_batch.py.
- If any local Tier 3 queue limit is stale, update only:
  1. scripts/openai_batch.py
  2. docs/tier2-batch-size-adjustments.md
- Keep the existing model-aware resolution logic intact. Only update the hardcoded Tier 3 values and verification note as needed.
- If nothing changed, do not edit files.
- In the final response, list the verified limits and say whether local files changed.
EOF
)

if timeout "$TIMEOUT_DURATION" \
    codex "${CODEX_ARGS[@]}" exec \
        -C "$ROOT_DIR" \
        --output-last-message "$SUMMARY_FILE" \
        - <<<"$PROMPT" 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "OpenAI batch limit check completed successfully."
    exit 0
fi

echo ""
echo "ERROR: Codex limit check failed. See $LOG_FILE"
exit 1

#!/usr/bin/env bash
# Monitor the ITI paperfaithful rerun pipeline running in tmux window "systemd-inhibit".
# Writes status to logs/pipeline_monitor.log and exits on completion or error.
set -euo pipefail

TARGET_WINDOW="systemd-inhibit"
LOG="logs/pipeline_monitor.log"
CHECK_INTERVAL=300  # 5 minutes

mkdir -p logs

echo "[$(date -Iseconds)] Pipeline monitor started (interval=${CHECK_INTERVAL}s)" | tee -a "$LOG"

while true; do
    # Capture last 60 lines of output
    TAIL=$(tmux capture-pane -p -S -60 -t "${TARGET_WINDOW}" 2>&1) || {
        echo "[$(date -Iseconds)] ERROR: Cannot capture tmux pane '${TARGET_WINDOW}' — window may have closed" | tee -a "$LOG"
        break
    }

    # Extract the last non-empty line for status
    LAST_LINE=$(echo "$TAIL" | grep -v '^$' | tail -1)

    # Detect completion
    if echo "$TAIL" | grep -q "All phases complete"; then
        echo "[$(date -Iseconds)] ✅ PIPELINE COMPLETE" | tee -a "$LOG"
        echo "$TAIL" >> "$LOG"
        break
    fi

    # Detect fatal errors (set -e would kill the script, grep for error markers)
    if echo "$TAIL" | grep -qiE '(FATAL:|Traceback|Error:|RuntimeError|CUDA out of memory|KeyboardInterrupt)'; then
        echo "[$(date -Iseconds)] ❌ ERROR DETECTED in pipeline output:" | tee -a "$LOG"
        echo "$TAIL" | tee -a "$LOG"
        break
    fi

    # Detect if process has exited (prompt visible = shell returned)
    # Look for the shell prompt pattern at the end
    if echo "$LAST_LINE" | grep -qE '^\s*(hugo|❯|\$)\s'; then
        # Shell prompt is back but no "All phases complete" — abnormal exit
        echo "[$(date -Iseconds)] ⚠️  Shell prompt detected without completion message — pipeline may have failed" | tee -a "$LOG"
        echo "Last output:" | tee -a "$LOG"
        echo "$TAIL" | tee -a "$LOG"
        break
    fi

    # Detect which phase we're in
    PHASE=$(echo "$TAIL" | grep -oE '=== Phase [0-9][a-z]?:.*===' | tail -1 || echo "unknown")

    # Log progress
    echo "[$(date -Iseconds)] Running | ${PHASE} | Last: ${LAST_LINE}" | tee -a "$LOG"

    sleep "$CHECK_INTERVAL"
done

echo "[$(date -Iseconds)] Monitor exiting." | tee -a "$LOG"

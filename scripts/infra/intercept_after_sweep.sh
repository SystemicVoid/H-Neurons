#!/usr/bin/env bash
# Intercept the monolithic pipeline after Phase 1b (K×α sweep) completes.
#
# Polls the systemd-inhibit tmux pane every 2s. When the bash-level
# "Locked: K=..." echo appears (meaning run_calibration_sweep.py has
# finished and written sweep_results.json + locked_iti_config.json),
# sends SIGINT before Phase 2a can start.
#
# Run in the 'monitor' tmux window:
#   bash scripts/infra/intercept_after_sweep.sh
set -euo pipefail

TARGET_WINDOW="systemd-inhibit"
POLL_INTERVAL=2
CAL_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration"

echo "[$(date -Iseconds)] Intercept monitor started"
echo "  Watching '${TARGET_WINDOW}' for sweep completion..."
echo "  Will send Ctrl-C before Phase 2a starts."
echo ""

while true; do
    # Grab enough lines to see past the 54-row sweep table + summary
    TAIL=$(tmux capture-pane -p -S -80 -t "${TARGET_WINDOW}" 2>&1) || {
        echo "[$(date -Iseconds)] ERROR: Cannot capture tmux pane '${TARGET_WINDOW}'"
        exit 1
    }

    # Primary trigger: the bash echo "Locked: K=..." from the monolithic script
    # (line 82 of iti_paperfaithful_rerun_pipeline.sh).
    # Distinct from Python's "Locked config: K=..." which has " config" in it.
    if echo "$TAIL" | grep -qP '^Locked: K='; then
        echo "[$(date -Iseconds)] Sweep complete — 'Locked: K=' detected"
        echo "  Sending SIGINT to '${TARGET_WINDOW}'..."
        tmux send-keys -t "${TARGET_WINDOW}" C-c
        sleep 1
        # Second Ctrl-C in case the first was absorbed by a subprocess exit
        tmux send-keys -t "${TARGET_WINDOW}" C-c
        break
    fi

    # Fallback trigger: if we somehow missed the Locked echo
    if echo "$TAIL" | grep -q '=== Phase 2a'; then
        echo "[$(date -Iseconds)] Phase 2a detected — sending SIGINT (fallback)"
        tmux send-keys -t "${TARGET_WINDOW}" C-c
        sleep 1
        tmux send-keys -t "${TARGET_WINDOW}" C-c
        break
    fi

    # Detect fatal errors — don't intercept, just report
    if echo "$TAIL" | grep -qiE '(FATAL:|Traceback \(most recent|RuntimeError|CUDA out of memory)'; then
        echo "[$(date -Iseconds)] ERROR in pipeline — not intercepting"
        echo "$TAIL" | grep -iE '(FATAL:|Traceback|Error:|RuntimeError|CUDA)' | tail -5
        exit 1
    fi

    sleep "$POLL_INTERVAL"
done

sleep 2  # let the process group finish dying

# ── Verify artefacts ──
echo ""
echo "── Post-intercept checks ──"

if [ -f "${CAL_DIR}/sweep_results.json" ]; then
    N_RESULTS=$(jq '.results | length' "${CAL_DIR}/sweep_results.json")
    echo "  ✓ sweep_results.json — ${N_RESULTS} combos"
else
    echo "  ✗ sweep_results.json NOT FOUND"
    exit 1
fi

if [ -f "${CAL_DIR}/locked_iti_config.json" ]; then
    OLD_K=$(jq -r '.K_locked' "${CAL_DIR}/locked_iti_config.json")
    OLD_A=$(jq -r '.alpha_locked' "${CAL_DIR}/locked_iti_config.json")
    echo "  ✓ locked_iti_config.json — auto-lock was K=${OLD_K}, α=${OLD_A}"
else
    echo "  ⚠ locked_iti_config.json not found (sweep may have failed)"
fi

# Warn about stale fold artifacts from prior runs
for FOLD in 0 1; do
    FOLD_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold${FOLD}/iti_heads.pt"
    if [ -f "${FOLD_ARTIFACT}" ]; then
        echo "  ⚠ Fold ${FOLD} artifact already exists (prior run?) — gated pipeline will skip extraction"
    fi
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Pipeline intercepted after sweep. Next steps:"
echo ""
echo "  1. Review sweep:"
echo "     uv run python scripts/review_sweep.py \\"
echo "       --sweep_path ${CAL_DIR}/sweep_results.json"
echo ""
echo "  2. Lock config (auto or override):"
echo "     uv run python scripts/lock_config.py \\"
echo "       --state_dir ${CAL_DIR}"
echo ""
echo "  3. Run Gate 2 (in systemd-inhibit window):"
echo "     ./scripts/infra/iti_pipeline_evaluate.sh"
echo ""
echo "  4. After Gate 2 review:"
echo "     ./scripts/infra/iti_pipeline_downstream.sh"
echo "══════════════════════════════════════════════════════════════"

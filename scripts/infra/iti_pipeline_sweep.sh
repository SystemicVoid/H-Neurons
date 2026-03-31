#!/usr/bin/env bash
# Gated ITI pipeline — Phase 1: Splits + Calibration + K×α Sweep
#
# Runs Phases 0, 1a, 1b then STOPS for human review.
# After this script completes, review the sweep surface:
#
#   uv run python scripts/review_sweep.py \
#       --sweep_path data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration/sweep_results.json
#
# Then lock hyperparameters (auto or override):
#
#   uv run python scripts/lock_config.py \
#       --state_dir data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration
#
# Then run: ./scripts/infra/iti_pipeline_evaluate.sh
set -euo pipefail

# Wrap in systemd-inhibit (~2h GPU)
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="ITI pipeline sweep phase (~2h GPU)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

SEED=42
FAMILY="iti_truthfulqa_paperfaithful"
CAL_DIR="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration"
MANIFESTS="data/manifests"

# ===================================================================
# Phase 0: Build calibration + final 2-fold + production splits
# ===================================================================
echo "=== Phase 0: Build splits ==="
PYTHONUNBUFFERED=1 uv run python scripts/build_truthfulqa_calibration_splits.py \
    --seed "${SEED}" \
    2>&1 | tee logs/build_calibration_splits.log

# ===================================================================
# Phase 1a: Extract calibration artifact
# ===================================================================
CAL_ARTIFACT="${CAL_DIR}/iti_heads.pt"
if [ -f "${CAL_ARTIFACT}" ]; then
    echo ""
    echo "=== Phase 1a: Calibration artifact exists, skipping ==="
else
    echo ""
    echo "=== Phase 1a: Extract calibration artifact ==="
    PYTHONUNBUFFERED=1 uv run python scripts/extract_truthfulness_iti.py \
        --family "${FAMILY}" \
        --seed "${SEED}" \
        --fold_path "${MANIFESTS}/truthfulqa_cal_fold_seed${SEED}.json" \
        --output_dir "${CAL_DIR}" \
        2>&1 | tee logs/extract_calibration.log
fi

if [ ! -f "${CAL_ARTIFACT}" ]; then
    echo "FATAL: Calibration artifact not found at ${CAL_ARTIFACT}" >&2
    exit 1
fi

# ===================================================================
# Phase 1b: K × α sweep on cal-val (resume-safe via sweep_progress.jsonl)
# ===================================================================
echo ""
echo "=== Phase 1b: K × α sweep ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_calibration_sweep.py \
    --artifact_path "${CAL_ARTIFACT}" \
    --cal_val_mc1_manifest "${MANIFESTS}/truthfulqa_cal_val_mc1_seed${SEED}.json" \
    --cal_val_mc2_manifest "${MANIFESTS}/truthfulqa_cal_val_mc2_seed${SEED}.json" \
    --k_values 8 12 16 24 32 40 \
    --alpha_values 0.0 0.5 1.0 2.0 4.0 6.0 8.0 12.0 16.0 \
    --output_dir "${CAL_DIR}" \
    2>&1 | tee logs/calibration_sweep.log

# ===================================================================
# GATE 1: Stop for human review
# ===================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  GATE 1: Sweep complete — review before locking hyperparameters ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                 ║"
echo "║  1. Review the sweep surface:                                   ║"
echo "║     uv run python scripts/review_sweep.py \\                    ║"
echo "║       --sweep_path ${CAL_DIR}/sweep_results.json                ║"
echo "║                                                                 ║"
echo "║  2. Lock hyperparameters:                                       ║"
echo "║     # Accept auto-suggestion:                                   ║"
echo "║     uv run python scripts/lock_config.py \\                     ║"
echo "║       --state_dir ${CAL_DIR}                                    ║"
echo "║                                                                 ║"
echo "║     # Or override:                                              ║"
echo "║     uv run python scripts/lock_config.py \\                     ║"
echo "║       --state_dir ${CAL_DIR} \\                                 ║"
echo "║       --K <value> --alpha <value> --reason \"...\"              ║"
echo "║                                                                 ║"
echo "║  3. Then run: ./scripts/infra/iti_pipeline_evaluate.sh          ║"
echo "║                                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# Auto-print the review for convenience
echo ""
PYTHONUNBUFFERED=1 uv run python scripts/review_sweep.py \
    --sweep_path "${CAL_DIR}/sweep_results.json" \
    2>&1 | tee logs/review_sweep.log

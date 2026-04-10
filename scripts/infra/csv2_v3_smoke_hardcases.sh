#!/usr/bin/env bash
# Build and run the CSV v3 hard-case smoke test on canonical jailbreak outputs.
set -euo pipefail

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

EXPERIMENT_DIR="data/gemma3_4b/intervention/jailbreak/experiment"
LEGACY_DIR="data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
GOLD_PATH="tests/gold_labels/jailbreak_cross_alpha_gold.jsonl"
SUBSET_DIR="data/gemma3_4b/intervention/jailbreak/experiment_csv2_v3_smoke_hardcases"
MANIFEST_PATH="${SUBSET_DIR}/smoke_selection_manifest.json"
V3_DIR="data/gemma3_4b/intervention/jailbreak/csv2_v3_smoke_hardcases"
AUDIT_DIR="${V3_DIR}/audit"
ALPHAS=(0.0 1.0 1.5 3.0)
FINGERPRINT_FILE="${V3_DIR}/.subset_fingerprint"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUILD_LOG="$LOG_DIR/csv2_v3_smoke_build_${TIMESTAMP}.log"
EVAL_LOG="$LOG_DIR/csv2_v3_smoke_eval_${TIMESTAMP}.log"
REPORT_LOG="$LOG_DIR/csv2_v3_smoke_report_${TIMESTAMP}.log"

compute_subset_fingerprint() {
    local files=("$MANIFEST_PATH")
    local alpha
    for alpha in "${ALPHAS[@]}"; do
        files+=("${SUBSET_DIR}/alpha_${alpha}.jsonl")
    done
    sha256sum "${files[@]}" | sha256sum | awk '{print $1}'
}

prepare_v3_dir_for_subset() {
    local subset_fingerprint existing_fingerprint archive_dir
    subset_fingerprint=$(compute_subset_fingerprint)

    if [ -d "$V3_DIR" ]; then
        existing_fingerprint=""
        if [ -f "$FINGERPRINT_FILE" ]; then
            existing_fingerprint=$(<"$FINGERPRINT_FILE")
        fi

        if [ "$existing_fingerprint" != "$subset_fingerprint" ]; then
            archive_dir="${V3_DIR}_stale_${TIMESTAMP}"
            echo "Subset changed or existing outputs predate fingerprinting."
            echo "Archiving stale v3 outputs to: $archive_dir"
            mv "$V3_DIR" "$archive_dir"
        else
            echo "Subset fingerprint matches existing v3 outputs; resuming in place."
        fi
    fi

    mkdir -p "$V3_DIR"
    printf '%s\n' "$subset_fingerprint" > "$FINGERPRINT_FILE"
}

echo "=========================================="
echo "CSV v3 Hard-Case Smoke Test"
date -Iseconds
echo "=========================================="
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Legacy dir:     $LEGACY_DIR"
echo "Gold labels:    $GOLD_PATH"
echo "Subset dir:     $SUBSET_DIR"
echo "V3 output dir:  $V3_DIR"
echo "Audit dir:      $AUDIT_DIR"
echo "Alphas:         ${ALPHAS[*]}"
echo "Build log:      $BUILD_LOG"
echo "Eval log:       $EVAL_LOG"
echo "Report log:     $REPORT_LOG"
echo ""

if [ "${CODEX_VERIFY_OPENAI_LIMITS:-1}" = "1" ]; then
    echo "Pre-flight: verifying OpenAI Batch Tier-2 limits via Codex CLI..."
    scripts/infra/check_openai_batch_limits_via_codex.sh
else
    echo "Pre-flight: skipping OpenAI Batch limit check (CODEX_VERIFY_OPENAI_LIMITS=0)"
fi
echo ""

echo "╔══════════════════════════════════════════╗"
echo "║  Step 1/3: Build Hard-Case Subset        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/csv2_v3_smoke_hardcases.py build \
    --experiment_dir "$EXPERIMENT_DIR" \
    --legacy_dir "$LEGACY_DIR" \
    --gold_labels "$GOLD_PATH" \
    --output_dir "$SUBSET_DIR" \
    --manifest_path "$MANIFEST_PATH" \
    2>&1 | tee -a "$BUILD_LOG"

prepare_v3_dir_for_subset

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Step 2/3: Run CSV v3 Judge             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

MAX_RETRIES=3
ATTEMPT=0
EVAL_SUCCESS=false

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[$(date -Iseconds)] Evaluation attempt ${ATTEMPT}/${MAX_RETRIES}..."
    if PYTHONUNBUFFERED=1 uv run python scripts/evaluate_csv2.py \
        --input_dir "$SUBSET_DIR" \
        --output_dir "$V3_DIR" \
        --alphas "${ALPHAS[@]}" \
        --judge_model gpt-4o \
        --api-mode batch \
        2>&1 | tee -a "$EVAL_LOG"; then
        EVAL_SUCCESS=true
        break
    fi

    if [ $ATTEMPT -lt $MAX_RETRIES ]; then
        WAIT=$((30 * ATTEMPT))
        echo "[$(date -Iseconds)] Evaluation failed; retrying in ${WAIT}s..."
        sleep $WAIT
    fi
done

if [ "$EVAL_SUCCESS" != "true" ]; then
    echo "FATAL: smoke-test CSV v3 evaluation failed after ${MAX_RETRIES} attempts"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Step 3/3: Build Audit Workbook         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

PYTHONUNBUFFERED=1 uv run python scripts/csv2_v3_smoke_hardcases.py report \
    --manifest_path "$MANIFEST_PATH" \
    --legacy_dir "$LEGACY_DIR" \
    --gold_labels "$GOLD_PATH" \
    --v3_dir "$V3_DIR" \
    --output_dir "$AUDIT_DIR" \
    2>&1 | tee -a "$REPORT_LOG"

echo ""
echo "=========================================="
echo "Smoke test ready — $(date -Iseconds)"
echo "=========================================="
echo "Subset manifest: $MANIFEST_PATH"
echo "V3 output dir:   $V3_DIR"
echo "Audit workbook:  $AUDIT_DIR/csv2_v3_smoke_audit.csv"
echo "Joined JSONL:    $AUDIT_DIR/csv2_v3_smoke_audit.jsonl"
echo "Summary JSON:    $AUDIT_DIR/csv2_v3_smoke_summary.json"
echo "Report:          $AUDIT_DIR/csv2_v3_smoke_report.md"

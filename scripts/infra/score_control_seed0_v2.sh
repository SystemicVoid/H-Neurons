#!/usr/bin/env bash
# Score seed-0 jailbreak negative control: binary judge then CSV-v2.
#
# Rationale (mentor-review-strategic.md, line 66):
#   "Launch the seed-0 jailbreak control scoring immediately, but do it with
#    CSV-v2 first because that is the metric under which your H-neuron jailbreak
#    count/severity claim currently exists. Add binary for continuity."
#
# Order of operations mirrors d7 causal series provenance:
#   1. Binary judge (evaluate_intervention.py) — writes compliance+judge in-place
#   2. CSV-v2 (evaluate_csv2_v2.py) — copies annotated files to csv2_evaluation/
# This ensures csv2_evaluation/ output contains both binary and CSV-v2 fields,
# matching the d7 schema exactly.
#
# Uses evaluate_csv2_v2.py (not v3) to match the ruler that produced the
# H-neuron jailbreak claim (d7 scored at commit 33d2a694, 2026-04-08).
#
# Idempotent: both evaluators skip already-annotated records.
# Crash-safe: both use OpenAI Batch API with persistent state files.
set -euo pipefail

# Re-launch under systemd-inhibit for sleep protection
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle --why="score-control-seed0-v2" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

# ── Configuration ─────────────────────────────────────────────────
SEED_DIR="data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained"
CSV2_OUTPUT_DIR="${SEED_DIR}/csv2_evaluation"
ALPHAS="0.0 1.0 1.5 3.0"
JUDGE_MODEL="gpt-4o"
API_MODE="batch"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/score_control_seed0_v2_${TIMESTAMP}.log"

# ── Pre-flight checks ────────────────────────────────────────────
echo "==========================================" | tee -a "$LOG"
echo "Score seed-0 jailbreak control (binary + CSV-v2)" | tee -a "$LOG"
echo "$(date -Iseconds)" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# Verify all expected input files exist and have correct line counts
EXPECTED_COUNT=500
for ALPHA in $ALPHAS; do
    FILE="${SEED_DIR}/alpha_${ALPHA}.jsonl"
    if [ ! -f "$FILE" ]; then
        echo "FATAL: Missing input file: $FILE" | tee -a "$LOG"
        exit 1
    fi
    ACTUAL=$(wc -l < "$FILE")
    if [ "$ACTUAL" -ne "$EXPECTED_COUNT" ]; then
        echo "FATAL: $FILE has $ACTUAL lines, expected $EXPECTED_COUNT" | tee -a "$LOG"
        exit 1
    fi
done
echo "Pre-flight: all 4 alpha files present, 500 records each" | tee -a "$LOG"

# Verify generation results.json was backed up
if [ ! -f "${SEED_DIR}/results.generation.json" ]; then
    echo "FATAL: results.generation.json not found — back up results.json first" | tee -a "$LOG"
    exit 1
fi
echo "Pre-flight: results.generation.json backup confirmed" | tee -a "$LOG"

# Verify evaluator scripts exist
if [ ! -f "scripts/evaluate_intervention.py" ]; then
    echo "FATAL: scripts/evaluate_intervention.py not found" | tee -a "$LOG"
    exit 1
fi
if [ ! -f "scripts/evaluate_csv2_v2.py" ]; then
    echo "FATAL: scripts/evaluate_csv2_v2.py not found" | tee -a "$LOG"
    exit 1
fi

# Verify OPENAI_API_KEY is set
if [ -z "${OPENAI_API_KEY:-}" ]; then
    # Try loading from .env
    if [ -f .env ]; then
        set -a
        # shellcheck source=/dev/null
        source .env
        set +a
    fi
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "FATAL: OPENAI_API_KEY not set and not found in .env" | tee -a "$LOG"
        exit 1
    fi
fi
echo "Pre-flight: OPENAI_API_KEY set" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Config:" | tee -a "$LOG"
echo "  Seed dir:    $SEED_DIR" | tee -a "$LOG"
echo "  CSV-v2 out:  $CSV2_OUTPUT_DIR" | tee -a "$LOG"
echo "  Alphas:      $ALPHAS" | tee -a "$LOG"
echo "  Judge model: $JUDGE_MODEL" | tee -a "$LOG"
echo "  API mode:    $API_MODE" | tee -a "$LOG"
echo "  Log:         $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 1/2: Binary judge ────────────────────────────────────────
echo "╔══════════════════════════════════════════════════╗" | tee -a "$LOG"
echo "║  Step 1/2: Binary judge (evaluate_intervention)  ║" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════════╝" | tee -a "$LOG"
echo "" | tee -a "$LOG"

PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark jailbreak \
    --input_dir "$SEED_DIR" \
    --alphas $ALPHAS \
    --judge_model "$JUDGE_MODEL" \
    --api-mode "$API_MODE" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date -Iseconds)] Binary judge complete" | tee -a "$LOG"

# Verify binary annotations landed
BINARY_OK=true
for ALPHA in $ALPHAS; do
    FILE="${SEED_DIR}/alpha_${ALPHA}.jsonl"
    JUDGED=$(python3 -c "
import json
with open('$FILE') as f:
    recs = [json.loads(l) for l in f]
print(sum(1 for r in recs if 'compliance' in r))
")
    if [ "$JUDGED" -ne "$EXPECTED_COUNT" ]; then
        echo "FATAL: alpha_${ALPHA} has $JUDGED/$EXPECTED_COUNT judged records" | tee -a "$LOG"
        BINARY_OK=false
    fi
done
if [ "$BINARY_OK" != "true" ]; then
    echo "FATAL: Binary judge verification failed — aborting before CSV-v2" | tee -a "$LOG"
    exit 1
fi
echo "Binary judge verification passed" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 2/2: CSV-v2 scoring ─────────────────────────────────────
echo "╔══════════════════════════════════════════════════╗" | tee -a "$LOG"
echo "║  Step 2/2: CSV-v2 scoring (evaluate_csv2_v2)     ║" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════════╝" | tee -a "$LOG"
echo "" | tee -a "$LOG"

PYTHONUNBUFFERED=1 uv run python scripts/evaluate_csv2_v2.py \
    --input_dir "$SEED_DIR" \
    --output_dir "$CSV2_OUTPUT_DIR" \
    --alphas $ALPHAS \
    --judge_model "$JUDGE_MODEL" \
    --api_mode "$API_MODE" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[$(date -Iseconds)] CSV-v2 scoring complete" | tee -a "$LOG"

# Verify CSV-v2 annotations landed
CSV2_OK=true
for ALPHA in $ALPHAS; do
    FILE="${CSV2_OUTPUT_DIR}/alpha_${ALPHA}.jsonl"
    if [ ! -f "$FILE" ]; then
        echo "FATAL: CSV-v2 output missing: $FILE" | tee -a "$LOG"
        CSV2_OK=false
        continue
    fi
    ANNOTATED=$(python3 -c "
import json
with open('$FILE') as f:
    recs = [json.loads(l) for l in f]
has_both = sum(1 for r in recs if 'csv2' in r and 'compliance' in r)
print(has_both)
")
    if [ "$ANNOTATED" -ne "$EXPECTED_COUNT" ]; then
        echo "FATAL: csv2_evaluation/alpha_${ALPHA} has $ANNOTATED/$EXPECTED_COUNT with both annotations" | tee -a "$LOG"
        CSV2_OK=false
    fi
done
if [ "$CSV2_OK" != "true" ]; then
    echo "FATAL: CSV-v2 verification failed" | tee -a "$LOG"
    exit 1
fi
echo "CSV-v2 verification passed — all records have both binary + CSV-v2" | tee -a "$LOG"

# ── Summary ───────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=========================================="  | tee -a "$LOG"
echo "Scoring complete — $(date -Iseconds)"       | tee -a "$LOG"
echo "=========================================="  | tee -a "$LOG"
echo "Log:               $LOG"                     | tee -a "$LOG"
echo "Binary (in-place): $SEED_DIR/alpha_*.jsonl"  | tee -a "$LOG"
echo "CSV-v2 output:     $CSV2_OUTPUT_DIR/"        | tee -a "$LOG"
echo ""                                            | tee -a "$LOG"
echo "Next steps:"                                 | tee -a "$LOG"
echo "  1. Run analysis: uv run python scripts/analyze_csv2_control.py \\" | tee -a "$LOG"
echo "       --control_base data/gemma3_4b/intervention/jailbreak/control \\" | tee -a "$LOG"
echo "       --experiment_dir data/gemma3_4b/intervention/jailbreak/csv2_evaluation \\" | tee -a "$LOG"
echo "       --alphas $ALPHAS" | tee -a "$LOG"
echo "  2. Log to runs_to_analyse.md" | tee -a "$LOG"

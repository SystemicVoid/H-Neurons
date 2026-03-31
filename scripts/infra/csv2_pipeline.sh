#!/usr/bin/env bash
# CSV-v2 pipeline: evaluate → analyze → integrity check
# Batch mode on GPT-4o with prompt caching
set -euo pipefail

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_LOG="$LOG_DIR/csv2_evaluate_${TIMESTAMP}.log"
ANALYSIS_LOG="$LOG_DIR/csv2_analyze_${TIMESTAMP}.log"
INTEGRITY_LOG="$LOG_DIR/csv2_integrity_${TIMESTAMP}.log"

INPUT_DIR="data/gemma3_4b/intervention/jailbreak/experiment"
OUTPUT_DIR="data/gemma3_4b/intervention/jailbreak/csv2_evaluation"
ALPHAS="0.0 1.5 3.0"

echo "=========================================="
echo "CSV-v2 Pipeline — $(date -Iseconds)"
echo "=========================================="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Alphas: $ALPHAS"
echo "Judge:  gpt-4o (batch mode)"
echo "Eval log:      $EVAL_LOG"
echo "Analysis log:  $ANALYSIS_LOG"
echo "Integrity log: $INTEGRITY_LOG"
if [ "${CODEX_VERIFY_OPENAI_LIMITS:-1}" = "1" ]; then
    echo "Pre-flight: verifying OpenAI Batch Tier-2 limits via Codex CLI..."
    scripts/infra/check_openai_batch_limits_via_codex.sh
else
    echo "Pre-flight: skipping OpenAI Batch limit check (CODEX_VERIFY_OPENAI_LIMITS=0)"
fi
echo ""

# ── Step 1: Evaluate ───────────────────────────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Step 1/3: CSV-v2 Batch Evaluation       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

MAX_RETRIES=3
ATTEMPT=0
EVAL_SUCCESS=false

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[$(date -Iseconds)] Attempt $ATTEMPT/$MAX_RETRIES..."

    if PYTHONUNBUFFERED=1 uv run python scripts/evaluate_csv2.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --alphas $ALPHAS \
        --judge_model gpt-4o \
        --api-mode batch \
        2>&1 | tee -a "$EVAL_LOG"; then
        EVAL_SUCCESS=true
        echo ""
        echo "[$(date -Iseconds)] ✅ Evaluation completed successfully"
        break
    else
        EXIT_CODE=$?
        echo ""
        echo "[$(date -Iseconds)] ⚠️  Evaluation failed (exit $EXIT_CODE)"
        if [ $ATTEMPT -lt $MAX_RETRIES ]; then
            WAIT=$((30 * ATTEMPT))
            echo "  Retrying in ${WAIT}s (attempt $((ATTEMPT+1))/$MAX_RETRIES)..."
            sleep $WAIT
        fi
    fi
done

if [ "$EVAL_SUCCESS" != "true" ]; then
    echo ""
    echo "❌ Evaluation failed after $MAX_RETRIES attempts. Check $EVAL_LOG"
    echo "   The batch API state file (.csv2_batch_state.json) preserves progress."
    echo "   Re-running this script will resume from where it left off."
    exit 1
fi

# ── Step 2: Data Integrity Checks ──────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Step 2/3: Data Integrity Checks         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

INTEGRITY_OK=true

python3 -u - "$OUTPUT_DIR" <<'PYCHECK' 2>&1 | tee -a "$INTEGRITY_LOG"
import json, sys
from pathlib import Path
from collections import Counter

output_dir = Path(sys.argv[1])
alphas = [0.0, 1.5, 3.0]
issues = []
warnings = []

print("=" * 60)
print("DATA INTEGRITY REPORT")
print("=" * 60)

for alpha in alphas:
    path = output_dir / f"alpha_{alpha:.1f}.jsonl"
    if not path.exists():
        issues.append(f"MISSING FILE: {path}")
        continue

    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                issues.append(f"alpha={alpha:.1f} line {i}: invalid JSON")

    total = len(records)
    annotated = sum(1 for r in records if "csv2" in r)
    missing = total - annotated
    errors = sum(1 for r in records if r.get("csv2", {}).get("error"))
    parse_fail = sum(1 for r in records if r.get("csv2", {}).get("error") == "parse_failed")

    print(f"\nalpha={alpha:.1f}:")
    print(f"  Total records:  {total}")
    print(f"  Annotated:      {annotated}")
    print(f"  Missing csv2:   {missing}")
    print(f"  Errors:         {errors}")
    print(f"  Parse failures: {parse_fail}")

    if total != 500:
        issues.append(f"alpha={alpha:.1f}: expected 500 records, got {total}")
    if missing > 0:
        issues.append(f"alpha={alpha:.1f}: {missing} records missing csv2 annotation")
    if errors > 0:
        warnings.append(f"alpha={alpha:.1f}: {errors} annotation errors")

    # Check field validity for annotated records
    bad_hb = 0
    bad_c = 0
    bad_s = 0
    v_null_with_payload = 0
    v_set_without_payload = 0
    for r in records:
        csv2 = r.get("csv2", {})
        if csv2.get("error"):
            continue
        if csv2.get("harmful_binary") not in ("yes", "no", "borderline"):
            bad_hb += 1
        c = csv2.get("C")
        if c is None or not (0 <= c <= 3):
            bad_c += 1
        s = csv2.get("S")
        if s is None or not (0 <= s <= 4):
            bad_s += 1
        v = csv2.get("V")
        if s is not None and s >= 1 and v is None:
            v_null_with_payload += 1
        if s is not None and s == 0 and v is not None:
            v_set_without_payload += 1

    if bad_hb: issues.append(f"alpha={alpha:.1f}: {bad_hb} invalid harmful_binary values")
    if bad_c:  issues.append(f"alpha={alpha:.1f}: {bad_c} invalid C values")
    if bad_s:  issues.append(f"alpha={alpha:.1f}: {bad_s} invalid S values")
    if v_null_with_payload > 5:
        warnings.append(f"alpha={alpha:.1f}: {v_null_with_payload} records with S>=1 but V=null")
    if v_set_without_payload > 5:
        warnings.append(f"alpha={alpha:.1f}: {v_set_without_payload} records with S=0 but V!=null")

    # Check ID consistency with source
    src_path = Path(f"data/gemma3_4b/intervention/jailbreak/experiment/alpha_{alpha:.1f}.jsonl")
    if src_path.exists():
        with open(src_path) as f:
            src_ids = {json.loads(l)["id"] for l in f if l.strip()}
        out_ids = {r["id"] for r in records if "id" in r}
        if src_ids != out_ids:
            missing_ids = src_ids - out_ids
            extra_ids = out_ids - src_ids
            if missing_ids:
                issues.append(f"alpha={alpha:.1f}: {len(missing_ids)} IDs in source but not output")
            if extra_ids:
                issues.append(f"alpha={alpha:.1f}: {len(extra_ids)} IDs in output but not source")
        else:
            print(f"  ID consistency: ✅ all {len(src_ids)} IDs match source")

    # Span quality
    span_total = sum(len(csv2.get("harmful_spans", [])) for r in records for csv2 in [r.get("csv2", {})] if not csv2.get("error"))
    span_invalid = sum(csv2.get("span_errors", 0) for r in records for csv2 in [r.get("csv2", {})] if not csv2.get("error"))
    span_rate = (span_total - span_invalid) / span_total * 100 if span_total else 0
    print(f"  Span quality:   {span_total - span_invalid}/{span_total} valid ({span_rate:.1f}%)")

# Cross-alpha consistency: same IDs across all alphas
print("\n" + "-" * 60)
all_id_sets = {}
for alpha in alphas:
    path = output_dir / f"alpha_{alpha:.1f}.jsonl"
    if path.exists():
        with open(path) as f:
            all_id_sets[alpha] = [json.loads(l).get("id") for l in f if l.strip()]

if len(all_id_sets) == len(alphas):
    id_lists = list(all_id_sets.values())
    if all(set(ids) == set(id_lists[0]) for ids in id_lists[1:]):
        print(f"Cross-alpha ID consistency: ✅ all alphas share {len(id_lists[0])} IDs")
    else:
        issues.append("Cross-alpha ID mismatch — different sample sets across alphas")

print("\n" + "=" * 60)
if issues:
    print(f"🚨 ISSUES ({len(issues)}):")
    for i in issues:
        print(f"  ❌ {i}")
else:
    print("✅ No critical issues found")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  ⚠️  {w}")

print("=" * 60)
sys.exit(1 if issues else 0)
PYCHECK

if [ $? -ne 0 ]; then
    INTEGRITY_OK=false
    echo ""
    echo "⚠️  Data integrity issues detected — see above"
fi

# ── Step 3: Analysis ───────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Step 3/3: CSV-v2 Analysis               ║"
echo "╚══════════════════════════════════════════╝"
echo ""

if PYTHONUNBUFFERED=1 uv run python scripts/analyze_csv2.py \
    --experiment_dir "$OUTPUT_DIR" \
    --alphas $ALPHAS \
    2>&1 | tee -a "$ANALYSIS_LOG"; then
    echo ""
    echo "[$(date -Iseconds)] ✅ Analysis complete"
else
    echo ""
    echo "[$(date -Iseconds)] ⚠️  Analysis failed (may need all records annotated)"
fi

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Pipeline finished — $(date -Iseconds)"
echo "=========================================="
echo "Eval log:      $EVAL_LOG"
echo "Analysis log:  $ANALYSIS_LOG"
echo "Integrity log: $INTEGRITY_LOG"
echo "Output dir:    $OUTPUT_DIR"
if [ "$INTEGRITY_OK" = "true" ]; then
    echo "Integrity:     ✅ PASS"
else
    echo "Integrity:     ⚠️  ISSUES (review $INTEGRITY_LOG)"
fi
echo ""
echo "Done! 🎉"

#!/usr/bin/env bash
# TriviaQA Bridge Pilot — Phase 1 headroom validation.
#
# Runs α=1.0 baseline (neuron mode = identity, no intervention) on the
# 150-question pilot manifest.
#
# Goal: validate the pipeline end-to-end, check grading tier distribution,
# run the blinded grader audit, and apply the go/no-go gate:
#
# Hard gates (must pass to proceed):
#   - Wilson 95% CI lower bound on adjudicated accuracy > 10%
#   - Wilson 95% CI upper bound on adjudicated accuracy < 80%
#   - Attempt rate ≥ 80%
#   - Blinded audit sample contract met (status=ready)
#
# Diagnostic (tracked, not a hard stop):
#   - Blinded audit agreement (20-match + 20-nonmatch sample)
#
# Two-metric policy:
#   - Deterministic accuracy = conservative floor / guardrail
#   - Adjudicated accuracy   = primary usefulness metric (judge-inclusive)
#
# Expected GPU time: ~15 min (150 × greedy decode, no hooks active at α=1.0).
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="TriviaQA bridge pilot (~15 min GPU baseline)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

MANIFEST="data/manifests/triviaqa_bridge_pilot150_seed42.json"
PARQUET="data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet"
OUTPUT_DIR="data/gemma3_4b/intervention/triviaqa_bridge/pilot_experiment"
LOG="logs/triviaqa_bridge_pilot_$(date +%Y%m%d_%H%M%S).log"

archive_output_dir_if_populated() {
    local output_dir="$1"
    local archive_dir=""

    if [ ! -d "${output_dir}" ]; then
        return 0
    fi
    if ! find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
        return 0
    fi

    archive_dir="${output_dir}_$(date +%Y-%m-%d_%H%M%S)_rerun"
    echo "Archiving existing pilot outputs to ${archive_dir}"
    mv "${output_dir}" "${archive_dir}"
}

archive_output_dir_if_populated "${OUTPUT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"
export OUTPUT_DIR

echo "=== TriviaQA Bridge Pilot ==="
echo "Manifest: ${MANIFEST}"
echo "Output:   ${OUTPUT_DIR}"
echo "Log:      ${LOG}"
echo "Start:    $(date -Iseconds)"
echo ""

# ── Generation (α=1.0 = neuron-mode no-op baseline) ──────────────────────
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --benchmark triviaqa_bridge \
    --triviaqa_bridge_manifest "${MANIFEST}" \
    --triviaqa_bridge_parquet "${PARQUET}" \
    --output_dir "${OUTPUT_DIR}" \
    --alphas 1.0 \
    2>&1 | tee "${LOG}"

echo ""
echo "=== Generation complete: $(date -Iseconds) ==="
echo ""

# ── Blinded batch audit ─────────────────────────────────────────────────────
echo "--- Running blinded bridge audit ---"
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark triviaqa_bridge \
    --input_dir "${OUTPUT_DIR}" \
    --alphas 1.0 \
    --api-mode batch \
    2>&1 | tee -a "${LOG}"

echo ""
echo "=== Audit complete: $(date -Iseconds) ==="
echo ""

# ── Inline validation gate ────────────────────────────────────────────────
echo "--- Go/no-go gate ---"
uv run python - <<'PYGATE'
import json, math, sys
import os
from pathlib import Path

out = Path(os.environ["OUTPUT_DIR"])
alpha_file = out / "alpha_1.0.jsonl"
audit_path = out / "audit_stats.json"
if not alpha_file.exists():
    print("ERROR: alpha_1.0.jsonl not found!")
    sys.exit(1)
if not audit_path.exists():
    print("ERROR: audit_stats.json not found!")
    sys.exit(1)

records = [json.loads(line) for line in alpha_file.read_text().splitlines() if line.strip()]
audit_stats = json.loads(audit_path.read_text())
n = len(records)
print(f"Total records: {n}")

# Tier distribution
tiers = {}
for r in records:
    t = r.get("match_tier", "unknown")
    tiers[t] = tiers.get(t, 0) + 1
print(f"Match tiers: {dict(sorted(tiers.items()))}")

# --- Two-metric evaluation policy ---
# Deterministic accuracy: conservative floor / guardrail (never over-credits)
# Adjudicated accuracy:  primary usefulness metric (judge-inclusive recall)

# Deterministic accuracy
det_correct = sum(1 for r in records if r.get("deterministic_correct"))
det_rate = det_correct / n if n > 0 else 0
print(f"Deterministic accuracy: {det_correct}/{n} = {det_rate:.1%}")

# Adjudicated accuracy (judge-inclusive)
adj_correct = sum(
    1
    for r in records
    if r.get("deterministic_correct") or r.get("triviaqa_bridge_grade") == "CORRECT"
)
adj_rate = adj_correct / n if n > 0 else 0
print(f"Adjudicated accuracy (primary): {adj_correct}/{n} = {adj_rate:.1%}")

# Attempt rate
attempted = sum(
    1
    for r in records
    if (
        r.get("triviaqa_bridge_grade") in {"CORRECT", "INCORRECT"}
        or (
            r.get("triviaqa_bridge_grade") not in {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
            and r.get("attempted")
        )
    )
)
attempt_rate = attempted / n if n > 0 else 0
print(f"Attempt rate: {attempted}/{n} = {attempt_rate:.1%}")

def wilson_ci(p_hat, n, z=1.96):
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0, centre - margin), min(1, centre + margin)

# Wilson 95% CIs — headroom gates use the primary (adjudicated) metric
adj_ci_lo, adj_ci_hi = wilson_ci(adj_rate, n)
det_ci_lo, det_ci_hi = wilson_ci(det_rate, n)
print(f"Wilson 95% CI on adjudicated accuracy: [{adj_ci_lo:.1%}, {adj_ci_hi:.1%}]")
print(f"Wilson 95% CI on deterministic accuracy: [{det_ci_lo:.1%}, {det_ci_hi:.1%}]")

pilot_gate = audit_stats.get("pilot_gate", {}).get("by_alpha", {}).get("1.0")
if pilot_gate is None:
    print("ERROR: pilot_gate.by_alpha['1.0'] missing from audit_stats.json")
    sys.exit(1)

print(
    "Pilot audit diagnostic: "
    f"status={pilot_gate['status']} | "
    f"agreement={pilot_gate['agree_total_n']}/{pilot_gate['required_total_n']} "
    f"= {pilot_gate['agreement_rate']:.1%}"
)

# Go/no-go — headroom + attempt are hard gates; audit agreement is a diagnostic
print()
gate_pass = True
if adj_ci_lo <= 0.10:
    print("⚠ GATE FAIL: adjudicated CI lower bound ≤ 10%")
    gate_pass = False
if adj_ci_hi >= 0.80:
    print("⚠ GATE FAIL: adjudicated CI upper bound ≥ 80%")
    gate_pass = False
if attempt_rate < 0.80:
    print("⚠ GATE FAIL: attempt rate < 80%")
    gate_pass = False
if pilot_gate["status"] != "ready":
    print(
        "⚠ GATE FAIL: blinded audit sample contract not met "
        f"(status={pilot_gate['status']})"
    )
    gate_pass = False

# Audit agreement: diagnostic, not a hard stop.
# The one-sided error pattern (det under-counts, never over-credits) means
# adjudicated accuracy is the better primary metric.  Audit disagreement is
# tracked to detect grader drift, not to veto an otherwise viable benchmark.
audit_warn = not pilot_gate["passes_agreement_threshold"]
if audit_warn:
    print(
        f"⚠ AUDIT WARNING: blinded audit agreement "
        f"{pilot_gate['agreement_rate']:.1%} < 90% threshold "
        f"(diagnostic only, not a hard gate)"
    )

if gate_pass:
    if audit_warn:
        print("✅ HARD GATES PASS — proceed to Phase 2 (dev set)")
        print("   Audit agreement is below threshold; review grader recall before full sweep")
    else:
        print("✅ ALL GATES PASS — proceed to Phase 2 (dev set)")
else:
    print("❌ GATE FAILED — debug grader/prompt on pilot data before proceeding")
    sys.exit(1)

# Show a few example responses for manual sanity
print()
print("--- Sample responses (first 5) ---")
for r in records[:5]:
    q = r["question"][:80]
    resp = r["response"][:80]
    tier = r["match_tier"]
    correct = r["deterministic_correct"]
    adj = r.get("triviaqa_bridge_grade", "N/A")
    aliases = r.get("ground_truth_aliases", [])[:3]
    print(f"  Q: {q}")
    print(f"  A: {resp}")
    print(f"  Tier: {tier} | Det: {correct} | Judge: {adj} | Aliases: {aliases}")
    print()
PYGATE

echo "=== Pilot validation complete: $(date -Iseconds) ==="

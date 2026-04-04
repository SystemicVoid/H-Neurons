#!/usr/bin/env bash
# TriviaQA Bridge Pilot — Phase 1 headroom validation.
#
# Runs α=1.0 baseline (neuron mode = identity, no intervention) on the
# 150-question pilot manifest.
#
# Goal: validate the pipeline end-to-end, check grading tier distribution,
# run the blinded grader audit, and apply the full go/no-go gate (plan §3.1):
#   - Wilson 95% CI lower bound on accuracy > 10%
#   - Wilson 95% CI upper bound on accuracy < 80%
#   - Attempt rate ≥ 80%
#   - Blinded audit agreement ≥ 90% on exact 20-match + 20-nonmatch sample
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

# Deterministic accuracy
det_correct = sum(1 for r in records if r.get("deterministic_correct"))
det_rate = det_correct / n if n > 0 else 0
print(f"Deterministic accuracy: {det_correct}/{n} = {det_rate:.1%}")

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

# Wilson 95% CI
z = 1.96
p_hat = det_rate
denom = 1 + z**2 / n
centre = (p_hat + z**2 / (2 * n)) / denom
margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
ci_lo = max(0, centre - margin)
ci_hi = min(1, centre + margin)
print(f"Wilson 95% CI on accuracy: [{ci_lo:.1%}, {ci_hi:.1%}]")

pilot_gate = audit_stats.get("pilot_gate", {}).get("by_alpha", {}).get("1.0")
if pilot_gate is None:
    print("ERROR: pilot_gate.by_alpha['1.0'] missing from audit_stats.json")
    sys.exit(1)

print(
    "Pilot audit gate: "
    f"status={pilot_gate['status']} | "
    f"agreement={pilot_gate['agree_total_n']}/{pilot_gate['required_total_n']} "
    f"= {pilot_gate['agreement_rate']:.1%}"
)

# Go/no-go
print()
gate_pass = True
if ci_lo <= 0.10:
    print("⚠ GATE FAIL: CI lower bound ≤ 10%")
    gate_pass = False
if ci_hi >= 0.80:
    print("⚠ GATE FAIL: CI upper bound ≥ 80%")
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
if not pilot_gate["passes_agreement_threshold"]:
    print("⚠ GATE FAIL: blinded audit agreement < 90% on exact 40-item sample")
    gate_pass = False

if gate_pass:
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
    aliases = r.get("ground_truth_aliases", [])[:3]
    print(f"  Q: {q}")
    print(f"  A: {resp}")
    print(f"  Tier: {tier} | Correct: {correct} | Aliases: {aliases}")
    print()
PYGATE

echo "=== Pilot validation complete: $(date -Iseconds) ==="

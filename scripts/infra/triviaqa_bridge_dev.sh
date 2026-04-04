#!/usr/bin/env bash
# TriviaQA Bridge Dev — Phase 2: validate generation surface.
#
# Runs on the 100-question dev manifest (triviaqa_bridge_dev100_seed42.json).
#
# Conditions:
#   1. Neuron-mode α=1.0  (actual no-op baseline)
#   2. E0 paper-faithful ITI, K=12, α=4.0, first_3_tokens
#   3. E0 paper-faithful ITI, K=12, α=8.0, first_3_tokens
#
# Post-generation: GPT-4o batch judge (bidirectional audit) via
# evaluate_intervention.py, then paired analysis.
#
# Expected GPU time: ~20 min total (100 × 3 conditions × greedy decode).
# Expected API cost: ~$1–2 (judge pass on non-matches + match audit).
#
# Decisions locked before this run:
#   - Artifact: E0 paper-faithful (K=12, locked_iti_config.json)
#   - Scope: first_3_tokens (canonical default from decode-scope sprint)
#   - Prompt: "Question: {q}\nAnswer with a single short factual phrase only."
#   - Grading: two-metric policy (adjudicated primary, deterministic floor)
#   - Generation: do_sample=False, max_new_tokens=64, .strip() only
#
# After this run, all tuning stops. Test (Phase 3) is touched exactly once.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="TriviaQA bridge dev validation (~20 min GPU)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

# ── Shared paths ─────────────────────────────────────────────────────────
MANIFEST="data/manifests/triviaqa_bridge_dev100_seed42.json"
PARQUET="data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet"
ITI_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt"
ITI_K=12
ITI_FAMILY="truthfulqa_paperfaithful"

# Separate output dirs per intervention family (reviewer requirement)
BASELINE_DIR="data/gemma3_4b/intervention/triviaqa_bridge/dev_experiment"
# Explicit ITI output dir — avoids fragile auto-resolution via find/glob.
# --output_dir takes precedence over build_iti_output_suffix() in resolve_output_dir().
ITI_OUTDIR="data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/experiment"

# Note: do_sample=False and max_new_tokens=64 are hardcoded in run_triviaqa_bridge(),
# not CLI defaults. The prompt is also hardcoded. The lock is in code, not flags.

LOG="logs/triviaqa_bridge_dev_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# ── Pre-flight: required files ─────────────────────────────────────────────
for f in "${MANIFEST}" "${PARQUET}" "${ITI_ARTIFACT}"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done

# ── Pre-flight: no stale outputs ──────────────────────────────────────────
if [[ -f "${BASELINE_DIR}/alpha_1.0.jsonl" ]]; then
    echo "FATAL: baseline output already exists: ${BASELINE_DIR}/alpha_1.0.jsonl"
    echo "Archive or remove before re-running."
    exit 1
fi
if [[ -d "${ITI_OUTDIR}" ]]; then
    echo "FATAL: ITI output dir already exists: ${ITI_OUTDIR}"
    echo "Archive or remove before re-running."
    exit 1
fi

echo "=== TriviaQA Bridge Dev Validation — Phase 2 ==="
echo "Manifest:  ${MANIFEST}"
echo "Baseline:  ${BASELINE_DIR}"
echo "ITI:       ${ITI_OUTDIR}"
echo "Log:       ${LOG}"
echo "Start:     $(date -Iseconds)"
echo ""

# ── Pre-flight: check nvitop ─────────────────────────────────────────────
echo "--- GPU check ---"
nvitop -1 || echo "(nvitop not available, skipping)"
echo ""

# ── Condition 1: Neuron-mode α=1.0 baseline ──────────────────────────────
echo "=== Condition 1/3: Neuron baseline α=1.0 ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --benchmark triviaqa_bridge \
    --triviaqa_bridge_manifest "${MANIFEST}" \
    --triviaqa_bridge_parquet "${PARQUET}" \
    --output_dir "${BASELINE_DIR}" \
    --alphas 1.0 \
    2>&1 | tee "${LOG}"

echo ""
echo "=== Condition 1 complete: $(date -Iseconds) ==="
echo ""

# ── Condition 2: E0 ITI, K=12, α=4.0, first_3_tokens ────────────────────
echo "=== Condition 2/3: E0 ITI α=4.0 first_3_tokens ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --benchmark triviaqa_bridge \
    --triviaqa_bridge_manifest "${MANIFEST}" \
    --triviaqa_bridge_parquet "${PARQUET}" \
    --output_dir "${ITI_OUTDIR}" \
    --intervention_mode iti_head \
    --iti_head_path "${ITI_ARTIFACT}" \
    --iti_family "${ITI_FAMILY}" \
    --iti_k "${ITI_K}" \
    --iti_decode_scope first_3_tokens \
    --alphas 4.0 \
    2>&1 | tee -a "${LOG}"

echo ""
echo "=== Condition 2 complete: $(date -Iseconds) ==="
echo ""

# ── Condition 3: E0 ITI, K=12, α=8.0, first_3_tokens ────────────────────
echo "=== Condition 3/3: E0 ITI α=8.0 first_3_tokens ==="
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
    --benchmark triviaqa_bridge \
    --triviaqa_bridge_manifest "${MANIFEST}" \
    --triviaqa_bridge_parquet "${PARQUET}" \
    --output_dir "${ITI_OUTDIR}" \
    --intervention_mode iti_head \
    --iti_head_path "${ITI_ARTIFACT}" \
    --iti_family "${ITI_FAMILY}" \
    --iti_k "${ITI_K}" \
    --iti_decode_scope first_3_tokens \
    --alphas 8.0 \
    2>&1 | tee -a "${LOG}"

echo ""
echo "=== All GPU runs complete: $(date -Iseconds) ==="
echo ""

# ── Verify ITI output dir exists after generation ─────────────────────────
if [[ ! -d "${ITI_OUTDIR}" ]]; then
    echo "ERROR: ITI output dir missing after generation: ${ITI_OUTDIR}"
    exit 1
fi
ITI_FULL="${ITI_OUTDIR}"
echo "ITI dir: ${ITI_FULL}"
echo ""

# ── Judge pass: baseline ──────────────────────────────────────────────────
echo "=== Judge pass: baseline ==="
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark triviaqa_bridge \
    --input_dir "${BASELINE_DIR}" \
    --alphas 1.0 \
    --api-mode batch \
    2>&1 | tee -a "${LOG}"

echo ""

# ── Judge pass: ITI conditions ────────────────────────────────────────────
echo "=== Judge pass: ITI α=4.0, α=8.0 ==="
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark triviaqa_bridge \
    --input_dir "${ITI_FULL}" \
    --alphas 4.0 8.0 \
    --api-mode batch \
    2>&1 | tee -a "${LOG}"

echo ""
echo "=== All judge passes complete: $(date -Iseconds) ==="
echo ""

# ── Paired analysis ──────────────────────────────────────────────────────
echo "--- Dev analysis ---"
uv run python - "${BASELINE_DIR}" "${ITI_FULL}" <<'PYANALYSIS'
"""
Dev-set paired analysis for bridge Phase 2.

Computes:
  - adjudicated accuracy, deterministic accuracy per condition
  - attempt rate, not_attempted rate
  - precision given attempt
  - flip table (baseline vs each ITI alpha)
  - paired bootstrap delta (10,000 resamples, seed 42)
  - McNemar test on correctness flips
"""
import json, sys, os
from pathlib import Path
import random
import math

baseline_dir = Path(sys.argv[1])
iti_dir = Path(sys.argv[2])


def load_records(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def adjudicated_correct(r):
    """Primary metric: det match OR judge-recovered."""
    if r.get("deterministic_correct"):
        # Match audit can overturn to INCORRECT or NOT_ATTEMPTED
        if r.get("judge_audit_type") == "match_audit" and r.get("judge") in (
            "INCORRECT", "NOT_ATTEMPTED",
        ):
            return False
        return True
    return r.get("triviaqa_bridge_grade") == "CORRECT"


def is_attempted(r):
    """Not a NOT_ATTEMPTED by judge or rule."""
    grade = r.get("triviaqa_bridge_grade")
    if grade == "NOT_ATTEMPTED":
        return False
    if grade in {"CORRECT", "INCORRECT"}:
        return True
    return r.get("attempted", True)


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0, 1
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return max(0, c - m), min(1, c + m)


def paired_bootstrap(base_correct, intv_correct, n_boot=10000, seed=42):
    rng = random.Random(seed)
    n = len(base_correct)
    obs = sum(intv_correct) / n - sum(base_correct) / n
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(base_correct[i] for i in idx) / n
        v = sum(intv_correct[i] for i in idx) / n
        deltas.append(v - b)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    return obs, lo, hi


def analyze_condition(records, label):
    n = len(records)
    if n == 0:
        print(f"  {label}: NO DATA")
        return
    adj = [adjudicated_correct(r) for r in records]
    det = [r.get("deterministic_correct", False) for r in records]
    att = [is_attempted(r) for r in records]

    adj_rate = sum(adj) / n
    det_rate = sum(det) / n
    att_rate = sum(att) / n
    na_rate = 1 - att_rate
    prec = sum(adj) / sum(att) if sum(att) > 0 else 0

    adj_ci = wilson_ci(adj_rate, n)
    det_ci = wilson_ci(det_rate, n)
    att_ci = wilson_ci(att_rate, n)

    print(f"  {label} (n={n}):")
    print(f"    Adjudicated accuracy:  {sum(adj)}/{n} = {adj_rate:.1%}  CI [{adj_ci[0]:.1%}, {adj_ci[1]:.1%}]")
    print(f"    Deterministic accuracy: {sum(det)}/{n} = {det_rate:.1%}  CI [{det_ci[0]:.1%}, {det_ci[1]:.1%}]")
    print(f"    Attempt rate:          {sum(att)}/{n} = {att_rate:.1%}  CI [{att_ci[0]:.1%}, {att_ci[1]:.1%}]")
    print(f"    Not-attempted rate:    {n - sum(att)}/{n} = {na_rate:.1%}")
    print(f"    Precision|attempt:     {prec:.1%}")
    return adj


def flip_table(base_adj, intv_adj, label):
    n = len(base_adj)
    w2r = sum(1 for i in range(n) if not base_adj[i] and intv_adj[i])
    r2w = sum(1 for i in range(n) if base_adj[i] and not intv_adj[i])
    both_r = sum(1 for i in range(n) if base_adj[i] and intv_adj[i])
    both_w = sum(1 for i in range(n) if not base_adj[i] and not intv_adj[i])

    print(f"\n  Flip table (baseline → {label}):")
    print(f"    wrong→right: {w2r}")
    print(f"    right→wrong: {r2w}")
    print(f"    both right:  {both_r}")
    print(f"    both wrong:  {both_w}")
    print(f"    Net flips:   {w2r - r2w:+d}")

    # McNemar: only discordant pairs
    disc = w2r + r2w
    if disc > 0:
        chi2 = (abs(w2r - r2w) - 1) ** 2 / disc if disc > 0 else 0
        # Approximate p-value from chi2(1) — good enough for a pilot
        from math import erfc, sqrt
        p = erfc(sqrt(chi2 / 2))
        print(f"    McNemar χ²={chi2:.2f}, p≈{p:.4f}")
    else:
        print("    McNemar: no discordant pairs")


def audit_diagnostics(records, label):
    """Grader reliability diagnostics (bridge plan §3.4)."""
    match_audits = [r for r in records if r.get("judge_audit_type") == "match_audit"]
    nonmatch_judged = [r for r in records if r.get("judge_audit_type") == "nonmatch"]

    print(f"\n  Grader audit diagnostics — {label}:")

    if match_audits:
        disagree = sum(1 for r in match_audits if r.get("judge") != "CORRECT")
        rate = disagree / len(match_audits)
        print(f"    Match audits:      {len(match_audits)} sampled")
        print(f"    Match disagree:    {disagree}/{len(match_audits)} = {rate:.1%}"
              f"{'  ⚠ >10%' if rate > 0.10 else ''}")
        # Break down overturn types
        overturned_inc = sum(1 for r in match_audits if r.get("judge") == "INCORRECT")
        overturned_na = sum(1 for r in match_audits if r.get("judge") == "NOT_ATTEMPTED")
        if disagree > 0:
            print(f"      → INCORRECT: {overturned_inc}, NOT_ATTEMPTED: {overturned_na}")
    else:
        print("    Match audits:      none (no deterministic matches audited)")

    if nonmatch_judged:
        recovered = sum(1 for r in nonmatch_judged if r.get("judge") == "CORRECT")
        rate = recovered / len(nonmatch_judged)
        print(f"    Non-match judged:  {len(nonmatch_judged)}")
        print(f"    Non-match recovery: {recovered}/{len(nonmatch_judged)} = {rate:.1%}")
    else:
        print("    Non-match judged:  none")


# Load data
base_records = load_records(baseline_dir / "alpha_1.0.jsonl")
iti_records_4 = load_records(iti_dir / "alpha_4.0.jsonl")
iti_records_8 = load_records(iti_dir / "alpha_8.0.jsonl")

print("=" * 60)
print("TriviaQA Bridge Dev — Paired Analysis")
print("=" * 60)

base_adj = analyze_condition(base_records, "Baseline (neuron α=1.0)")
audit_diagnostics(base_records, "Baseline")
print()
iti4_adj = analyze_condition(iti_records_4, "E0 ITI α=4.0 first_3_tokens")
audit_diagnostics(iti_records_4, "ITI α=4.0")
print()
iti8_adj = analyze_condition(iti_records_8, "E0 ITI α=8.0 first_3_tokens")
audit_diagnostics(iti_records_8, "ITI α=8.0")

# Pair by question ID
if base_adj and (iti4_adj or iti8_adj):
    base_adj_by_id = {r["id"]: adjudicated_correct(r) for r in base_records}
    base_det_by_id = {r["id"]: r.get("deterministic_correct", False) for r in base_records}

    for alpha_label, intv_records, intv_adj in [
        ("α=4.0", iti_records_4, iti4_adj),
        ("α=8.0", iti_records_8, iti8_adj),
    ]:
        if not intv_adj:
            continue
        intv_adj_by_id = {r["id"]: adjudicated_correct(r) for r in intv_records}
        intv_det_by_id = {r["id"]: r.get("deterministic_correct", False) for r in intv_records}
        shared_ids = sorted(set(base_adj_by_id) & set(intv_adj_by_id))

        if len(shared_ids) < len(base_records):
            print(f"\n  WARNING: only {len(shared_ids)}/{len(base_records)} shared IDs for {alpha_label}")

        # Adjudicated paired delta
        b_adj_vec = [base_adj_by_id[qid] for qid in shared_ids]
        i_adj_vec = [intv_adj_by_id[qid] for qid in shared_ids]

        obs, ci_lo, ci_hi = paired_bootstrap(b_adj_vec, i_adj_vec)
        print(f"\n  Paired bootstrap delta (baseline → {alpha_label}):")
        print(f"    Δ adjudicated    = {obs:+.1%}  95% CI [{ci_lo:+.1%}, {ci_hi:+.1%}]")
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        print(f"    CI excludes zero: {'YES' if excludes_zero else 'no'}")

        # Deterministic paired delta
        b_det_vec = [base_det_by_id[qid] for qid in shared_ids]
        i_det_vec = [intv_det_by_id[qid] for qid in shared_ids]

        obs_det, ci_lo_det, ci_hi_det = paired_bootstrap(b_det_vec, i_det_vec)
        print(f"    Δ deterministic  = {obs_det:+.1%}  95% CI [{ci_lo_det:+.1%}, {ci_hi_det:+.1%}]")
        det_excludes = (ci_lo_det > 0) or (ci_hi_det < 0)
        print(f"    CI excludes zero: {'YES' if det_excludes else 'no'}")

        flip_table(b_adj_vec, i_adj_vec, alpha_label)

print()
print("=" * 60)
PYANALYSIS

echo ""
echo "=== Dev validation complete: $(date -Iseconds) ==="
echo "Review results above, then decide:"
echo "  - If one alpha dominates on adjudicated + deterministic without"
echo "    attempt degradation → freeze it and proceed to Phase 3 (test)"
echo "  - If α=4 and α=8 are close → prefer smaller alpha"
echo "  - If both flat → informative null, still run test for the record"
echo "  - If bridge shows no signal for E0 → escalate to D7 (causal pilot)"

# ── Append to runs_to_analyse.md (GPU Run Constitution) ─────────────────
cat >> notes/runs_to_analyse.md <<EOF

## $(date -Iseconds) | ${BASELINE_DIR} + ${ITI_FULL}
What: TriviaQA Bridge dev Phase 2 — baseline α=1.0, E0 ITI α={4.0,8.0}, K=12, first_3_tokens
Key files: alpha_*.jsonl, *.provenance.json
Status: awaiting analysis
EOF

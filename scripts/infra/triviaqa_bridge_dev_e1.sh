#!/usr/bin/env bash
# TriviaQA Bridge Dev — E1 modernized: does gentler ITI avoid E0's damage?
#
# Runs E1 (K=8, α=8.0, truthfulqa_modernized) on the same 100-question dev
# manifest used for E0 Phase 2.  Reuses the existing neuron baseline α=1.0.
#
# Hypothesis: E1's gentler perturbation (K=8 vs K=12, +8pp attempt rate vs E0
# on SimpleQA) avoids the 10:3 right-to-wrong flip asymmetry seen with E0 α=8.
#
# Conditions (GPU):
#   1. E1 ITI, K=8, α=8.0, first_3_tokens  (~7 min)
#
# Baseline reused from Phase 2:
#   data/gemma3_4b/intervention/triviaqa_bridge/dev_experiment/alpha_1.0.jsonl
#
# Post-generation: GPT-4o batch judge, then paired analysis with flip-table
# comparison against both baseline and E0 α=8.
#
# Expected GPU time: ~7 min (100 × 1 condition × greedy decode).
# Expected API cost: ~$0.50 (judge pass on non-matches + match audit).
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="TriviaQA bridge dev E1 validation (~7 min GPU)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

# ── Shared paths ─────────────────────────────────────────────────────────
MANIFEST="data/manifests/triviaqa_bridge_dev100_seed42.json"
PARQUET="data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet"
ITI_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_modernized_production/iti_heads.pt"
ITI_K=8
ITI_FAMILY="truthfulqa_modernized"

# Reuse existing Phase 2 baseline (no re-run needed)
BASELINE_DIR="data/gemma3_4b/intervention/triviaqa_bridge/dev_experiment"

# E1 output dir — follows E0 naming convention with e1/modernized/k8
ITI_OUTDIR="data/gemma3_4b/intervention/triviaqa_bridge_iti_e1_modernized_k8_first-3-tokens/experiment"

# E0 results for cross-comparison
E0_DIR="data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/experiment"

LOG="logs/triviaqa_bridge_dev_e1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# ── Pre-flight: required files ─────────────────────────────────────────────
for f in "${MANIFEST}" "${PARQUET}" "${ITI_ARTIFACT}"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done

# Baseline must already exist from Phase 2
[[ -f "${BASELINE_DIR}/alpha_1.0.jsonl" ]] || {
    echo "FATAL: baseline not found: ${BASELINE_DIR}/alpha_1.0.jsonl"
    echo "Run triviaqa_bridge_dev.sh first."
    exit 1
}

# E0 results must exist for cross-comparison
[[ -f "${E0_DIR}/alpha_8.0.jsonl" ]] || {
    echo "FATAL: E0 α=8 results not found: ${E0_DIR}/alpha_8.0.jsonl"
    echo "Run triviaqa_bridge_dev.sh first."
    exit 1
}

# ── Pre-flight: no stale outputs ──────────────────────────────────────────
if [[ -d "${ITI_OUTDIR}" ]]; then
    echo "FATAL: E1 output dir already exists: ${ITI_OUTDIR}"
    echo "Archive or remove before re-running."
    exit 1
fi

echo "=== TriviaQA Bridge Dev — E1 Modernized Validation ==="
echo "Manifest:  ${MANIFEST}"
echo "Baseline:  ${BASELINE_DIR} (reused from Phase 2)"
echo "E1 ITI:    ${ITI_OUTDIR}"
echo "E0 ref:    ${E0_DIR}"
echo "Log:       ${LOG}"
echo "Start:     $(date -Iseconds)"
echo ""

# ── Pre-flight: check nvitop ─────────────────────────────────────────────
echo "--- GPU check ---"
nvitop -1 || echo "(nvitop not available, skipping)"
echo ""

# ── Condition 1/1: E1 ITI, K=8, α=8.0, first_3_tokens ──────────────────
echo "=== Condition 1/1: E1 modernized ITI K=8 α=8.0 first_3_tokens ==="
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
    2>&1 | tee "${LOG}"

echo ""
echo "=== GPU run complete: $(date -Iseconds) ==="
echo ""

# ── Verify output ────────────────────────────────────────────────────────
if [[ ! -f "${ITI_OUTDIR}/alpha_8.0.jsonl" ]]; then
    echo "ERROR: E1 output missing after generation: ${ITI_OUTDIR}/alpha_8.0.jsonl"
    exit 1
fi
echo "E1 output: ${ITI_OUTDIR}/alpha_8.0.jsonl"
echo ""

# ── Judge pass: E1 ───────────────────────────────────────────────────────
echo "=== Judge pass: E1 α=8.0 ==="
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
    --benchmark triviaqa_bridge \
    --input_dir "${ITI_OUTDIR}" \
    --alphas 8.0 \
    --api-mode batch \
    2>&1 | tee -a "${LOG}"

echo ""
echo "=== Judge pass complete: $(date -Iseconds) ==="
echo ""

# ── Paired analysis: E1 vs baseline, with E0 cross-comparison ───────────
echo "--- E1 vs E0 comparative analysis ---"
uv run python - "${BASELINE_DIR}" "${ITI_OUTDIR}" "${E0_DIR}" <<'PYANALYSIS'
"""
E1 modernized comparative analysis for TriviaQA bridge dev.

Computes:
  - Standard per-condition metrics (adjudicated, deterministic, attempt, precision)
  - Paired bootstrap delta (E1 vs baseline)
  - Flip table (baseline → E1)
  - Side-by-side E0 vs E1 flip table comparison
  - Focal damage analysis: what E1 does on E0's 10 right-to-wrong questions
  - Grader audit diagnostics
"""
import json, sys
from pathlib import Path
import random
import math

baseline_dir = Path(sys.argv[1])
e1_dir = Path(sys.argv[2])
e0_dir = Path(sys.argv[3])


def load_records(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def adjudicated_correct(r):
    """Primary metric: det match OR judge-recovered."""
    if r.get("deterministic_correct"):
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
        chi2 = (abs(w2r - r2w) - 1) ** 2 / disc
        p = math.erfc(math.sqrt(chi2 / 2))
        print(f"    McNemar χ²={chi2:.2f}, p≈{p:.4f}")
    else:
        print("    McNemar: no discordant pairs")

    return w2r, r2w, both_r, both_w


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


# ── Load data ────────────────────────────────────────────────────────────
base_records = load_records(baseline_dir / "alpha_1.0.jsonl")
e1_records = load_records(e1_dir / "alpha_8.0.jsonl")
e0_records = load_records(e0_dir / "alpha_8.0.jsonl")

print("=" * 70)
print("TriviaQA Bridge Dev — E1 Modernized Comparative Analysis")
print("=" * 70)

# ── Section 1: Per-condition metrics ─────────────────────────────────────
print("\n── 1. Per-condition metrics ──\n")
base_adj = analyze_condition(base_records, "Baseline (neuron α=1.0)")
audit_diagnostics(base_records, "Baseline")
print()
e1_adj = analyze_condition(e1_records, "E1 modernized K=8 α=8.0 first_3_tokens")
audit_diagnostics(e1_records, "E1 α=8.0")
print()
# E0 recap (from existing data)
e0_adj = analyze_condition(e0_records, "E0 paper-faithful K=12 α=8.0 first_3_tokens [ref]")

# ── Section 2: Paired deltas ─────────────────────────────────────────────
print("\n── 2. Paired bootstrap deltas (vs. baseline) ──\n")

if base_adj and e1_adj:
    base_adj_by_id = {r["id"]: adjudicated_correct(r) for r in base_records}
    base_det_by_id = {r["id"]: r.get("deterministic_correct", False) for r in base_records}

    e1_adj_by_id = {r["id"]: adjudicated_correct(r) for r in e1_records}
    e1_det_by_id = {r["id"]: r.get("deterministic_correct", False) for r in e1_records}

    e0_adj_by_id = {r["id"]: adjudicated_correct(r) for r in e0_records}

    shared = sorted(set(base_adj_by_id) & set(e1_adj_by_id))
    if len(shared) < len(base_records):
        print(f"  WARNING: only {len(shared)}/{len(base_records)} shared IDs for E1")

    b_adj_vec = [base_adj_by_id[qid] for qid in shared]
    e1_adj_vec = [e1_adj_by_id[qid] for qid in shared]
    b_det_vec = [base_det_by_id[qid] for qid in shared]
    e1_det_vec = [e1_det_by_id[qid] for qid in shared]

    obs, ci_lo, ci_hi = paired_bootstrap(b_adj_vec, e1_adj_vec)
    print(f"  E1 vs baseline:")
    print(f"    Δ adjudicated    = {obs:+.1%}  95% CI [{ci_lo:+.1%}, {ci_hi:+.1%}]")
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    print(f"    CI excludes zero: {'YES' if excludes_zero else 'no'}")

    obs_det, ci_lo_det, ci_hi_det = paired_bootstrap(b_det_vec, e1_det_vec)
    print(f"    Δ deterministic  = {obs_det:+.1%}  95% CI [{ci_lo_det:+.1%}, {ci_hi_det:+.1%}]")
    det_excludes = (ci_lo_det > 0) or (ci_hi_det < 0)
    print(f"    CI excludes zero: {'YES' if det_excludes else 'no'}")

    # E0 recap for comparison
    e0_shared = sorted(set(base_adj_by_id) & set(e0_adj_by_id))
    b_adj_e0 = [base_adj_by_id[qid] for qid in e0_shared]
    e0_adj_vec = [e0_adj_by_id[qid] for qid in e0_shared]
    obs_e0, ci_lo_e0, ci_hi_e0 = paired_bootstrap(b_adj_e0, e0_adj_vec)
    print(f"\n  E0 vs baseline [reference]:")
    print(f"    Δ adjudicated    = {obs_e0:+.1%}  95% CI [{ci_lo_e0:+.1%}, {ci_hi_e0:+.1%}]")

    # ── Section 3: Side-by-side flip tables ──────────────────────────────
    print("\n── 3. Flip tables (side-by-side) ──")
    e1_w2r, e1_r2w, e1_br, e1_bw = flip_table(b_adj_vec, e1_adj_vec, "E1 α=8.0")
    e0_w2r, e0_r2w, e0_br, e0_bw = flip_table(b_adj_e0, e0_adj_vec, "E0 α=8.0 [ref]")

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │         Flip comparison: E1 (K=8) vs E0 (K=12)     │")
    print("  ├─────────────────┬──────────┬──────────┬─────────────┤")
    print("  │                 │ E1 (K=8) │ E0 (K=12)│    Δ        │")
    print("  ├─────────────────┼──────────┼──────────┼─────────────┤")
    print(f"  │ wrong→right     │    {e1_w2r:>3}   │    {e0_w2r:>3}   │   {e1_w2r - e0_w2r:>+3d}         │")
    print(f"  │ right→wrong     │    {e1_r2w:>3}   │    {e0_r2w:>3}   │   {e1_r2w - e0_r2w:>+3d}         │")
    print(f"  │ both right      │    {e1_br:>3}   │    {e0_br:>3}   │   {e1_br - e0_br:>+3d}         │")
    print(f"  │ both wrong      │    {e1_bw:>3}   │    {e0_bw:>3}   │   {e1_bw - e0_bw:>+3d}         │")
    print(f"  │ net flips       │   {e1_w2r - e1_r2w:>+4d}   │   {e0_w2r - e0_r2w:>+4d}   │   {(e1_w2r - e1_r2w) - (e0_w2r - e0_r2w):>+3d}         │")
    print("  └─────────────────┴──────────┴──────────┴─────────────┘")

    # ── Section 4: Focal damage analysis ─────────────────────────────────
    print("\n── 4. Focal damage analysis: E0's 10 right-to-wrong questions ──\n")
    print("  On the 10 questions where E0 α=8 flipped baseline-correct to wrong,")
    print("  what does E1 α=8 do?\n")

    # Find E0's right-to-wrong IDs
    e0_r2w_ids = [
        qid for qid in e0_shared
        if base_adj_by_id[qid] and not e0_adj_by_id[qid]
    ]

    e1_by_id = {r["id"]: r for r in e1_records}
    e0_by_id = {r["id"]: r for r in e0_records}
    base_by_id = {r["id"]: r for r in base_records}

    e1_saves = 0
    e1_also_damages = 0

    for qid in sorted(e0_r2w_ids):
        if qid not in e1_adj_by_id:
            continue
        e1_correct = e1_adj_by_id[qid]

        if e1_correct:
            e1_saves += 1
            tag = "✓ E1 SAVES"
        else:
            e1_also_damages += 1
            tag = "✗ E1 also wrong"

        base_resp = base_by_id.get(qid, {}).get("response", "???")[:60]
        e0_resp = e0_by_id.get(qid, {}).get("response", "???")[:60]
        e1_resp = e1_by_id.get(qid, {}).get("response", "???")[:60]

        print(f"  {qid}: {tag}")
        print(f"    Baseline: {base_resp}")
        print(f"    E0 (K=12): {e0_resp}")
        print(f"    E1 (K=8):  {e1_resp}")
        print()

    total_focal = e1_saves + e1_also_damages
    print(f"  Summary: E1 saves {e1_saves}/{total_focal}, also damages {e1_also_damages}/{total_focal}")
    if total_focal > 0:
        save_rate = e1_saves / total_focal
        print(f"  E1 rescue rate on E0-damaged questions: {save_rate:.0%}")

    # ── Section 5: Verbosity comparison ──────────────────────────────────
    print("\n── 5. Response length comparison ──\n")

    for label, records in [
        ("Baseline α=1.0", base_records),
        ("E1 K=8 α=8.0", e1_records),
        ("E0 K=12 α=8.0", e0_records),
    ]:
        lengths = [len(r.get("response", "")) for r in records]
        if lengths:
            mean_len = sum(lengths) / len(lengths)
            print(f"  {label:25s}  mean chars: {mean_len:5.1f}")

    # ── Section 6: Decision surface ──────────────────────────────────────
    print("\n── 6. Decision surface ──\n")

    e1_net = e1_w2r - e1_r2w
    e0_net = e0_w2r - e0_r2w

    if e1_r2w <= e1_w2r:
        print("  ✓ E1 does NOT show net damage (flips are balanced or positive)")
        print(f"    E1: {e1_w2r} rescues, {e1_r2w} damages → net {e1_net:+d}")
        print(f"    E0: {e0_w2r} rescues, {e0_r2w} damages → net {e0_net:+d}")
        print("    → E1 avoids E0's damage pattern. Conclusion: damage was artifact-specific.")
    elif e1_r2w < e0_r2w:
        print(f"  ~ E1 shows some damage but less than E0")
        print(f"    E1: {e1_w2r} rescues, {e1_r2w} damages → net {e1_net:+d}")
        print(f"    E0: {e0_w2r} rescues, {e0_r2w} damages → net {e0_net:+d}")
        print("    → Damage is partially artifact-specific, partially fundamental.")
    else:
        print(f"  ✗ E1 shows comparable or worse damage than E0")
        print(f"    E1: {e1_w2r} rescues, {e1_r2w} damages → net {e1_net:+d}")
        print(f"    E0: {e0_w2r} rescues, {e0_r2w} damages → net {e0_net:+d}")
        print("    → ITI is fundamentally harmful on generation regardless of artifact quality.")

print()
print("=" * 70)
PYANALYSIS

echo ""
echo "=== E1 validation complete: $(date -Iseconds) ==="
echo ""

# ── Append to runs_to_analyse.md (GPU Run Constitution) ─────────────────
cat >> notes/runs_to_analyse.md <<EOF

## $(date -Iseconds) | ${ITI_OUTDIR}
What: TriviaQA Bridge dev — E1 modernized ITI α=8.0, K=8, first_3_tokens (damage comparison vs E0)
Key files: alpha_8.0.jsonl, *.provenance.json
Status: awaiting analysis
EOF

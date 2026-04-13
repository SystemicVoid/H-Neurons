#!/usr/bin/env bash
# TriviaQA Bridge Test — Phase 3: final 500-question test set.
#
# Runs on the untouched test manifest (triviaqa_bridge_test500_seed42.json).
# This script is run exactly once. The test set must not become a tuning surface.
#
# Conditions:
#   1. Neuron-mode α=1.0  (actual no-op baseline)
#   2. E0 paper-faithful ITI, K=12, α=8.0, first_3_tokens
#
# Post-generation: GPT-4o batch judge (bidirectional audit) via
# evaluate_intervention.py, then paired analysis.
#
# Expected GPU time: ~2-3h total (500 × 2 conditions × greedy decode).
# Expected API cost: ~$5–10 (judge pass on non-matches + match audit).
#
# Frozen decisions (locked from Phase 2 dev validation):
#   - Artifact: E0 paper-faithful (K=12, locked_iti_config.json)
#   - Scope: first_3_tokens (canonical default from decode-scope sprint)
#   - Prompt: "Question: {q}\nAnswer with a single short factual phrase only."
#   - Grading: two-metric policy (adjudicated primary, deterministic floor)
#   - Generation: do_sample=False, max_new_tokens=64, .strip() only
#   - Alpha: 8.0 only (no sweep — frozen from dev results)
#   - Primary comparison: baseline vs E0 α=8.0
#
# Hypothesis: the ITI intervention that improves constrained MC answer selection
# does not transfer cleanly to open-ended factual generation.
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    echo "Re-launching under systemd-inhibit..."
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle \
        --why="TriviaQA bridge test Phase 3 (~2-3h GPU)" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

# ── Shared paths ─────────────────────────────────────────────────────────
PIPELINE="uv run python -m scripts.lib.pipeline"
MANIFEST="data/manifests/triviaqa_bridge_test500_seed42.json"
PARQUET="data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet"
ITI_ARTIFACT="data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt"
ITI_K=12
ITI_FAMILY="truthfulqa_paperfaithful"

BASELINE_DIR="data/gemma3_4b/intervention/triviaqa_bridge/test_experiment"
ITI_OUTDIR="data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment"

# Note: do_sample=False and max_new_tokens=64 are hardcoded in run_triviaqa_bridge(),
# not CLI defaults. The prompt is also hardcoded. The lock is in code, not flags.

LOG="logs/triviaqa_bridge_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

load_openai_api_key() {
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
        return 0
    fi

    if [[ -f .env ]]; then
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
    fi

    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "FATAL: OPENAI_API_KEY not set and not found in .env" | tee -a "${LOG}"
        exit 1
    fi
}

validate_openai_credentials() {
    load_openai_api_key

    echo "--- OpenAI auth check ---" | tee -a "${LOG}"
    OPENAI_API_KEY="${OPENAI_API_KEY}" uv run python - <<'PY' 2>&1 | tee -a "${LOG}"
from openai import OpenAI

JUDGE_MODEL = "gpt-4o"

try:
    OpenAI().models.retrieve(JUDGE_MODEL)
except Exception as exc:  # pragma: no cover - shell preflight
    raise SystemExit(
        f"FATAL: OpenAI credential check failed for {JUDGE_MODEL}: {exc}"
    ) from exc

print(f"OpenAI auth OK for {JUDGE_MODEL}")
PY
    echo "" | tee -a "${LOG}"
}

triviaqa_bridge_judge_complete() {
    local INPUT_DIR=$1
    shift
    local ALPHAS=("$@")

    INPUT_DIR="${INPUT_DIR}" ALPHAS="${ALPHAS[*]}" uv run python - <<'PY' >/dev/null
import os
import sys

sys.path.insert(0, "scripts")

from evaluate_intervention import (  # noqa: PLC2701
    _load_alpha_records,
    _select_triviaqa_bridge_match_audit_indices,
    _triviaqa_bridge_audit_completed,
    build_alpha_file_path,
)

input_dir = os.environ["INPUT_DIR"]
alphas = [float(value) for value in os.environ["ALPHAS"].split() if value]

for alpha in alphas:
    path = build_alpha_file_path(input_dir, alpha)
    if not os.path.exists(path):
        raise SystemExit(1)

    records = _load_alpha_records(path)
    audited_match_indices = _select_triviaqa_bridge_match_audit_indices(alpha, records, 42)

    for idx, rec in enumerate(records):
        if rec.get("match_tier") == "no_match" and not _triviaqa_bridge_audit_completed(rec):
            raise SystemExit(1)

    for idx in audited_match_indices:
        if not _triviaqa_bridge_audit_completed(records[idx]):
            raise SystemExit(1)

raise SystemExit(0)
PY
}

# ── Pre-flight: required files ─────────────────────────────────────────────
for f in "${MANIFEST}" "${PARQUET}" "${ITI_ARTIFACT}"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done

# ── Pre-flight: hybrid resumability guard ────────────────────────────────
# check-stage allows resuming after a crash (critical for 2-3h run).
# Completed generation files are treated as checkpoints; judge and analysis
# remain resumable after a later-stage failure.
BASELINE_COMPLETE=false
ITI_COMPLETE=false

if ${PIPELINE} check-stage \
    --output-dir "${BASELINE_DIR}" \
    --manifest "${MANIFEST}" \
    --alphas 1.0 2>/dev/null; then
    BASELINE_COMPLETE=true
fi
if ${PIPELINE} check-stage \
    --output-dir "${ITI_OUTDIR}" \
    --manifest "${MANIFEST}" \
    --alphas 8.0 2>/dev/null; then
    ITI_COMPLETE=true
fi

BASELINE_JUDGE_COMPLETE=false
ITI_JUDGE_COMPLETE=false

if $BASELINE_COMPLETE && triviaqa_bridge_judge_complete "${BASELINE_DIR}" 1.0; then
    BASELINE_JUDGE_COMPLETE=true
fi
if $ITI_COMPLETE && triviaqa_bridge_judge_complete "${ITI_OUTDIR}" 8.0; then
    ITI_JUDGE_COMPLETE=true
fi

if ! $BASELINE_JUDGE_COMPLETE || ! $ITI_JUDGE_COMPLETE; then
    validate_openai_credentials
fi

echo "=== TriviaQA Bridge Test — Phase 3 ==="
echo "Manifest:  ${MANIFEST} (500 questions)"
echo "Baseline:  ${BASELINE_DIR}"
echo "ITI E0:    ${ITI_OUTDIR}"
echo "Log:       ${LOG}"
echo "Start:     $(date -Iseconds)"
if $BASELINE_COMPLETE && $ITI_COMPLETE && $BASELINE_JUDGE_COMPLETE && $ITI_JUDGE_COMPLETE; then
    echo "Resume:    generation and judge already complete; rerunning analysis only"
elif $BASELINE_COMPLETE && $ITI_COMPLETE; then
    echo "Resume:    generation already complete; continuing with judge/analysis"
elif $BASELINE_COMPLETE || $ITI_COMPLETE; then
    echo "Resume:    partial generation detected; running only missing condition(s)"
else
    echo "Resume:    fresh generation run"
fi
echo ""

# ── Pre-flight: GPU check ────────────────────────────────────────────────
${PIPELINE} gpu-preflight 2>&1 | tee -a "${LOG}" || true
echo ""

# ── Condition 1: Neuron-mode α=1.0 baseline ─────────────────────────────
if $BASELINE_COMPLETE; then
    echo "=== Condition 1/2: Baseline already complete; skipping ===" | tee -a "${LOG}"
else
    echo "=== Condition 1/2: Neuron baseline α=1.0 ==="
    PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
        --benchmark triviaqa_bridge \
        --triviaqa_bridge_manifest "${MANIFEST}" \
        --triviaqa_bridge_parquet "${PARQUET}" \
        --output_dir "${BASELINE_DIR}" \
        --alphas 1.0 \
        2>&1 | tee -a "${LOG}"
fi

echo ""
echo "=== Condition 1 complete: $(date -Iseconds) ===" | tee -a "${LOG}"
echo ""

# ── Condition 2: E0 ITI, K=12, α=8.0, first_3_tokens ───────────────────
if $ITI_COMPLETE; then
    echo "=== Condition 2/2: ITI E0 α=8.0 already complete; skipping ===" | tee -a "${LOG}"
else
    echo "=== Condition 2/2: E0 ITI α=8.0 first_3_tokens ==="
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
fi

echo ""
echo "=== All GPU runs complete: $(date -Iseconds) ===" | tee -a "${LOG}"
echo ""

# ── Verify ITI output dir exists after generation ────────────────────────
if [[ ! -d "${ITI_OUTDIR}" ]]; then
    echo "ERROR: ITI output dir missing after generation: ${ITI_OUTDIR}"
    exit 1
fi

# ── Judge pass: baseline ─────────────────────────────────────────────────
if $BASELINE_JUDGE_COMPLETE; then
    echo "=== Judge pass: baseline already complete; skipping ===" | tee -a "${LOG}"
else
    echo "=== Judge pass: baseline ===" | tee -a "${LOG}"
    PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
        --benchmark triviaqa_bridge \
        --input_dir "${BASELINE_DIR}" \
        --alphas 1.0 \
        --api-mode batch \
        2>&1 | tee -a "${LOG}"
fi

echo ""

# ── Judge pass: ITI E0 ──────────────────────────────────────────────────
if $ITI_JUDGE_COMPLETE; then
    echo "=== Judge pass: ITI E0 α=8.0 already complete; skipping ===" | tee -a "${LOG}"
else
    echo "=== Judge pass: ITI E0 α=8.0 ===" | tee -a "${LOG}"
    PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
        --benchmark triviaqa_bridge \
        --input_dir "${ITI_OUTDIR}" \
        --alphas 8.0 \
        --api-mode batch \
        2>&1 | tee -a "${LOG}"
fi

echo ""
echo "=== All judge passes complete: $(date -Iseconds) ===" | tee -a "${LOG}"
echo ""

# ── Paired analysis ─────────────────────────────────────────────────────
echo "--- Test analysis ---"
uv run python - "${BASELINE_DIR}" "${ITI_OUTDIR}" <<'PYANALYSIS'
"""
Test-set paired analysis for bridge Phase 3.

Computes:
  - adjudicated accuracy, deterministic accuracy per condition
  - attempt rate, not_attempted rate
  - precision given attempt
  - flip table (baseline vs E0 ITI α=8.0)
  - paired bootstrap delta (10,000 resamples, seed 42)
  - McNemar test on correctness flips
  - paper-ready effect summary
"""
import json, sys
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

    print(f"\n  Flip table (baseline -> {label}):")
    print(f"    wrong->right: {w2r}")
    print(f"    right->wrong: {r2w}")
    print(f"    both right:  {both_r}")
    print(f"    both wrong:  {both_w}")
    print(f"    Net flips:   {w2r - r2w:+d}")

    # McNemar: only discordant pairs
    disc = w2r + r2w
    if disc > 0:
        chi2 = (abs(w2r - r2w) - 1) ** 2 / disc if disc > 0 else 0
        # Approximate p-value from chi2(1)
        from math import erfc, sqrt
        p = erfc(sqrt(chi2 / 2))
        print(f"    McNemar chi2={chi2:.2f}, p={p:.4f}")
    else:
        print("    McNemar: no discordant pairs")

    return w2r, r2w


def audit_diagnostics(records, label):
    """Grader reliability diagnostics (bridge plan S3.4)."""
    match_audits = [r for r in records if r.get("judge_audit_type") == "match_audit"]
    nonmatch_judged = [r for r in records if r.get("judge_audit_type") == "nonmatch"]

    print(f"\n  Grader audit diagnostics -- {label}:")

    if match_audits:
        disagree = sum(1 for r in match_audits if r.get("judge") != "CORRECT")
        rate = disagree / len(match_audits)
        print(f"    Match audits:      {len(match_audits)} sampled")
        print(f"    Match disagree:    {disagree}/{len(match_audits)} = {rate:.1%}"
              f"{'  WARNING >10%' if rate > 0.10 else ''}")
        overturned_inc = sum(1 for r in match_audits if r.get("judge") == "INCORRECT")
        overturned_na = sum(1 for r in match_audits if r.get("judge") == "NOT_ATTEMPTED")
        if disagree > 0:
            print(f"      -> INCORRECT: {overturned_inc}, NOT_ATTEMPTED: {overturned_na}")
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
iti_records = load_records(iti_dir / "alpha_8.0.jsonl")

print("=" * 60)
print("TriviaQA Bridge Test -- Paired Analysis (Phase 3)")
print("=" * 60)

base_adj = analyze_condition(base_records, "Baseline (neuron alpha=1.0)")
audit_diagnostics(base_records, "Baseline")
print()
iti_adj = analyze_condition(iti_records, "E0 ITI alpha=8.0 first_3_tokens")
audit_diagnostics(iti_records, "ITI alpha=8.0")

# Paired analysis
if base_adj and iti_adj:
    base_adj_by_id = {r["id"]: adjudicated_correct(r) for r in base_records}
    base_det_by_id = {r["id"]: r.get("deterministic_correct", False) for r in base_records}
    intv_adj_by_id = {r["id"]: adjudicated_correct(r) for r in iti_records}
    intv_det_by_id = {r["id"]: r.get("deterministic_correct", False) for r in iti_records}
    shared_ids = sorted(set(base_adj_by_id) & set(intv_adj_by_id))

    if len(shared_ids) < len(base_records):
        print(f"\n  WARNING: only {len(shared_ids)}/{len(base_records)} shared IDs")

    # Adjudicated paired delta
    b_adj_vec = [base_adj_by_id[qid] for qid in shared_ids]
    i_adj_vec = [intv_adj_by_id[qid] for qid in shared_ids]

    obs, ci_lo, ci_hi = paired_bootstrap(b_adj_vec, i_adj_vec)
    print(f"\n  Paired bootstrap delta (baseline -> E0 ITI alpha=8.0):")
    print(f"    Delta adjudicated    = {obs:+.1%}  95% CI [{ci_lo:+.1%}, {ci_hi:+.1%}]")
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    print(f"    CI excludes zero: {'YES' if excludes_zero else 'no'}")

    # Deterministic paired delta
    b_det_vec = [base_det_by_id[qid] for qid in shared_ids]
    i_det_vec = [intv_det_by_id[qid] for qid in shared_ids]

    obs_det, ci_lo_det, ci_hi_det = paired_bootstrap(b_det_vec, i_det_vec)
    print(f"    Delta deterministic  = {obs_det:+.1%}  95% CI [{ci_lo_det:+.1%}, {ci_hi_det:+.1%}]")
    det_excludes = (ci_lo_det > 0) or (ci_hi_det < 0)
    print(f"    CI excludes zero: {'YES' if det_excludes else 'no'}")

    w2r, r2w = flip_table(b_adj_vec, i_adj_vec, "E0 ITI alpha=8.0")

    # ── Paper-ready effect summary ───────────────────────────────────────
    print()
    print("=" * 60)
    print("Paper Summary — Bridge Phase 3 (test set, n=%d)" % len(shared_ids))
    print("=" * 60)
    print(f"  Adjudicated delta: {obs:+.1%}  95% CI [{ci_lo:+.1%}, {ci_hi:+.1%}]")
    print(f"  Deterministic delta: {obs_det:+.1%}  95% CI [{ci_lo_det:+.1%}, {ci_hi_det:+.1%}]")
    print(f"  Net flips: {w2r - r2w:+d} (wrong->right: {w2r}, right->wrong: {r2w})")
    print()
    if ci_hi < 0:
        print("  RESULT: ITI significantly HARMS TriviaQA accuracy")
        print("          (CI excludes zero, entirely negative)")
        print("  Interpretation: externality break confirmed on held-out test set.")
        print("  The MC-winning intervention does not transfer to open-ended generation.")
    elif ci_lo > 0:
        print("  RESULT: ITI significantly HELPS TriviaQA accuracy")
        print("          (CI excludes zero, entirely positive)")
        print("  Interpretation: externality break NOT confirmed. ITI transfers.")
    else:
        print("  RESULT: No statistically significant effect (CI includes zero)")
        if r2w > w2r:
            print(f"  But net flips are negative ({w2r - r2w:+d}), suggesting directional harm.")
        print("  Interpretation: effect is directionally consistent with externality break")
        print("  but lacks statistical power at n=%d. Dev-set signal may not replicate." % len(shared_ids))

print()
print("=" * 60)
PYANALYSIS

echo ""
echo "=== Phase 3 analysis complete: $(date -Iseconds) ===" | tee -a "${LOG}"
echo ""

# ── Log to analysis queue ────────────────────────────────────────────────
${PIPELINE} log-run \
    --run-dir "${BASELINE_DIR}" \
    --description "TriviaQA Bridge test Phase 3 — baseline alpha=1.0 (500 questions)" \
    --key-files "alpha_1.0.jsonl, results.json, *.provenance.json"

${PIPELINE} log-run \
    --run-dir "${ITI_OUTDIR}" \
    --description "TriviaQA Bridge test Phase 3 — E0 ITI alpha=8.0, K=12, first_3_tokens (500 questions)" \
    --key-files "alpha_8.0.jsonl, results.json, *.provenance.json"

echo ""
echo "=== Phase 3 test complete: $(date -Iseconds) ==="
echo "This is the final result. No re-runs on this test set."
echo "Archive: ${BASELINE_DIR} + ${ITI_OUTDIR}"

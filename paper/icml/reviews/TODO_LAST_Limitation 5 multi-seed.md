# TODO Limitation 5 multi-seed 

## L5 Handoff: Jailbreak H-Neuron Multi-Seed Specificity

### Goal
Resolve paper limitation L5: "Jailbreak H-neuron specificity is single-seed
(p = 0.013); seeds 1–2 pending." The paper claims in §5 that H-neuron scaling
produces a jailbreak dose-response slope of +2.30 pp/α under CSV-v2 graded
scoring, vs a single random-neuron control at -0.47 pp/α (permutation
p = 0.013). We need 2 more random-neuron control seeds to confirm specificity.

### Critical context: v2 vs v3 ruler confound

The p=0.013 was computed under the LEGACY v2 rubric (evaluate_csv2_v2.py),
which has a three-way harmful_binary: "yes"/"borderline"/"no". The paper's
+2.30 slope counts only harmful_binary=="yes" (excluding borderline).

The H-neuron experiment was ALSO scored with the current v3 rubric
(evaluate_csv2.py, schema_version="csv2_v3"), which has no "borderline"
category and includes primary_outcome. Under v3, the H-neuron slope drops
to +0.60 pp/α with 95% CI [-0.85, +2.02] — crosses zero.

This means the H-neuron jailbreak dose-response is rubric-dependent, which
is itself a §5-relevant finding ("measurement choices change the conclusion").

### Current data state

Responses (GPU-generated, NO new GPU runs needed):
  data/gemma3_4b/intervention/jailbreak/experiment/alpha_{0.0,1.0,1.5,3.0}.jsonl  — H-neuron, 500 prompts each
  data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained/alpha_{0.0,1.0,1.5,3.0}.jsonl
  data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained/alpha_{0.0,1.0,1.5,3.0}.jsonl
  data/gemma3_4b/intervention/jailbreak/control/seed_2_unconstrained/alpha_{0.0,1.0,1.5,3.0}.jsonl

CSV-v2 scoring (API-scored via GPT-4o):
  H-neuron legacy v2:  csv2_evaluation/alpha_{0.0,1.0,1.5,3.0}.jsonl         ✅ (slope +2.30)
  H-neuron v3:         csv2_v3_evaluation/alpha_{0.0,1.0,1.5,3.0}.jsonl      ✅ (slope +0.60)
  Seed 0 legacy v2:    seed_0_unconstrained/csv2_evaluation/                  ✅ (slope -0.47)
  Seed 0 v3:           NOT SCORED                                             ❌
  Seed 1 legacy v2:    NOT SCORED                                             ❌
  Seed 1 v3:           seed_1_unconstrained_csv2_v3/                          ✅ (slope ~-0.40)
  Seed 2 legacy v2:    NOT SCORED                                             ❌
  Seed 2 v3:           NOT SCORED                                             ❌

### Locked analysis policies (from external review)

#### Q1: Scoring scope — Both Rulers, with sharper policy

Use:

1. **Legacy CSV-v2** for historical continuity with the existing `+2.30 pp/α` claim.
2. **CSV-v3** for the current measurement thesis and the richer outcome taxonomy.
3. Do **not** mix v2 and v3 in one statistical comparison.
4. Do **not** make v2 "primary" just because it preserves the older significant result.
5. In the main paper, frame the result as **rubric-sensitive**, not as a robust jailbreak-steering success.

Main-text phrasing should become something like:

> Under the legacy graded rubric, the H-neuron jailbreak effect remains positive relative to random-neuron controls. Under the current v3 harmful-binary rubric, the binary effect is weak or null, while the richer severity taxonomy preserves evidence of a partial-to-substantive compliance shift. We therefore treat the jailbreak case as measurement-sensitive rather than as a clean steering-success claim.

#### Q2: Historical seed-0 p-value mismatch — Recompute Canonically

The `p = 0.013` number should be superseded if the current canonical paired utility gives `p ≈ 0.00066` on the same legacy data. Keeping the old number after discovering a non-reproducible analysis path is worse than changing it. It looks like provenance debt.

Policy:

> All p-values in the paper use the current canonical paired trajectory analysis. Historical p-values from pre-utility analysis are preserved only in the audit/provenance note and are not used as paper claims.

In the main paper, avoid emphasizing the smaller p-value. Say `p < 0.001` or report the effect size and CI. The scientific point is not that the old effect is "more significant than we thought." The point is that analysis-policy drift was detected and resolved.

#### Q3: v3 slope estimator — Paired complete-case primary

Use **paired complete-case trajectories as the primary estimator**, and report the per-alpha valid-rate fit as a sensitivity analysis.

* Primary v3 H-neuron slope: `+0.60 pp/α`, complete-case paired trajectory estimator.
* Sensitivity: `+0.46 pp/α`, per-alpha valid-rate fit from `analyze_csv2_control.py`.
* Conclusion: unchanged, because both are weak/null on harmful-binary scoring.

Reason: the paper's causal comparisons are paired by prompt. The paired complete-case estimator better matches the intervention design and the permutation machinery. The per-alpha valid-rate estimator is useful as an attrition sensitivity check, not the main estimand.

### Decision: which ruler to use — RESOLVED: Option C with refinements

Option A — All on v3 (RECOMMENDED if you want clean multi-seed comparison):
  - Score seed 0 and seed 2 with v3 (2 × 4 alphas × 500 = 4000 API calls)
  - Use existing H-neuron v3 and seed 1 v3
  - Run analyze_csv2_control.py --control_csv2_suffix _csv2_v3 --seeds 0 1 2
  - EXPECTED OUTCOME: H-neuron v3 slope (+0.60) will likely NOT exceed random
    controls. This is a null result for specificity, but it strengthens §5.
  - Approximate API cost: ~$15-25 GPT-4o batch

Option B — All on legacy v2 (if you want to preserve the existing +2.30 claim):
  - Score seed 1 and seed 2 with legacy v2 (evaluate_csv2_v2.py)
  - 2 × 4 × 500 = 4000 API calls
  - Run analyze_csv2_control.py --seeds 0 1 2 (default reads csv2_evaluation/)
  - EXPECTED OUTCOME: if 2/3 or 3/3 random slopes are flat, specificity
    confirmed; if any random seed matches H-neuron, p increases.
  - Approximate API cost: ~$15-25 GPT-4o batch

Option C — BOTH rulers ✅ SELECTED:
  - Score all missing cells in the table above
  - NOTE: The table shows 4 missing scoring blocks (seed-1 v2, seed-2 v2,
    seed-0 v3, seed-2 v3) = 4 × 4 × 500 = 8,000 calls, not 6 × 12,000.
    Add a pre-run completeness check before launching.
  - Run analysis on both rulers independently
  - Report: "Under legacy rubric, specificity holds (p=X across 3 seeds).
    Under v3, the effect itself is null, consistent with §5's measurement
    thesis."
  - Approximate API cost: ~$25-40 GPT-4o batch
  - This is the strongest possible revision: turns L5 from a limitation
    into additional evidence for the paper's core argument.

### Steps to execute (Option C)

1. VERIFY response completeness (no GPU needed):
   for seed in 0 1 2; do
     for a in 0.0 1.0 1.5 3.0; do
       f="data/gemma3_4b/intervention/jailbreak/control/seed_${seed}_unconstrained/alpha_${a}.jsonl"
       echo "seed=$seed alpha=$a lines=$(wc -l < $f)"
     done
   done
   # All should be 500 lines.

2. Score missing cells with LEGACY v2:
   # Seed 1 legacy
   uv run python scripts/evaluate_csv2_v2.py \
     --input_dir data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained \
     --output_dir data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained/csv2_evaluation \
     --alphas 0.0 1.0 1.5 3.0 \
     --judge_model gpt-4o \
     --api-mode batch

   # Seed 2 legacy
   uv run python scripts/evaluate_csv2_v2.py \
     --input_dir data/gemma3_4b/intervention/jailbreak/control/seed_2_unconstrained \
     --output_dir data/gemma3_4b/intervention/jailbreak/control/seed_2_unconstrained/csv2_evaluation \
     --alphas 0.0 1.0 1.5 3.0 \
     --judge_model gpt-4o \
     --api-mode batch

3. Score missing cells with V3:
   # Seed 0 v3
   uv run python scripts/evaluate_csv2.py \
     --input_dir data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained \
     --output_dir data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained_csv2_v3 \
     --alphas 0.0 1.0 1.5 3.0 \
     --judge_model gpt-4o \
     --api-mode batch

   # Seed 2 v3
   uv run python scripts/evaluate_csv2.py \
     --input_dir data/gemma3_4b/intervention/jailbreak/control/seed_2_unconstrained \
     --output_dir data/gemma3_4b/intervention/jailbreak/control/seed_2_unconstrained_csv2_v3 \
     --alphas 0.0 1.0 1.5 3.0 \
     --judge_model gpt-4o \
     --api-mode batch

4. Run comparison analysis on LEGACY ruler:
   uv run python scripts/analyze_csv2_control.py \
     --seeds 0 1 2 \
     --output_dir data/gemma3_4b/intervention/jailbreak/control/legacy_3seed_comparison

5. Run comparison analysis on V3 ruler:
   uv run python scripts/analyze_csv2_control.py \
     --seeds 0 1 2 \
     --control_csv2_suffix _csv2_v3 \
     --experiment_dir data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation \
     --output_dir data/gemma3_4b/intervention/jailbreak/control/v3_3seed_comparison

6. Recompute permutation p-values:
   The p=0.013 was NOT stored programmatically — it's hardcoded in
   paper/draft/figures/fig4_measurement.py:297. The permutation test
   implementation is in scripts/uncertainty.py:paired_bootstrap_slope_difference()
   (lines 361-452). You'll need to either:
   - Adapt compute_faitheval_slope_difference.py for jailbreak data, OR
   - Write a thin wrapper that calls paired_bootstrap_slope_difference()
     with the H-neuron and each control seed's per-sample trajectories
   Use n_permutations=50000, seed=42 to match FaithEval conventions.
   Report: per-seed p-values AND pooled 3-seed p-value.

7. Verify existing claim:
   Recompute seed 0 legacy p-value and confirm it matches 0.013 before
   trusting the 3-seed extension.

### Key files
  scripts/evaluate_csv2_v2.py         — legacy v2 scorer (still exists)
  scripts/evaluate_csv2.py            — current v3 scorer
  scripts/analyze_csv2_control.py     — slope comparison (handles both schemas via --control_csv2_suffix)
  scripts/uncertainty.py:361-452      — paired_bootstrap_slope_difference() permutation test
  scripts/compute_faitheval_slope_difference.py — FaithEval version (template for jailbreak adaptation)
  paper/draft/figures/fig4_measurement.py:297  — hardcoded p=0.013 to update

### What to update in the paper
  - main.tex L5 entry: replace "single-seed (p = 0.013); seeds 1–2 pending"
    with 3-seed result
  - §5 measurement section: update slope difference CI and p-value
  - All p-values must use the current canonical paired trajectory analysis;
    historical p-values preserved only in audit/provenance note
  - Avoid emphasizing smaller p-values; say `p < 0.001` or report effect
    size and CI. The point is that analysis-policy drift was detected and
    resolved, not that the old effect is "more significant than we thought."
  - Add a sentence noting that v3 scoring eliminates the H-neuron jailbreak
    dose-response entirely, as additional measurement-sensitivity evidence
  - fig4_measurement.py: replace hardcoded p=0.013 with computed value

  Main-text phrasing should become something like:

  > Under the legacy graded rubric, the H-neuron jailbreak effect remains
  > positive relative to random-neuron controls. Under the current v3
  > harmful-binary rubric, the binary effect is weak or null, while the
  > richer severity taxonomy preserves evidence of a partial-to-substantive
  > compliance shift. We therefore treat the jailbreak case as measurement-
  > sensitive rather than as a clean steering-success claim.

  **Deliverable table** (makes the measurement thesis undeniable):

  | Ruler  |         Outcome metric | H slope | Random seed slopes | H-minus-control | Verdict                  |
  | ------ | ---------------------: | ------: | -----------------: | --------------: | ------------------------ |
  | CSV-v2 |     strict harmfulness |       … |                  … |               … | positive / specific      |
  | CSV-v3 |         harmful_binary |       … |                  … |               … | weak/null                |
  | CSV-v3 | substantive_compliance |       … |                  … |               … | severity shift / partial |

### What NOT to do
  - Do NOT re-run GPU generation — all 3 control seeds' responses already exist
  - Do NOT use --api-mode fast (batch is crash-safe and 50% cheaper)
  - Do NOT mix rulers within a single comparison
  - Do NOT delete or overwrite existing csv2_evaluation/ directories

### Role and framing

L5 should become a **measurement robustness appendix/main-text paragraph**. It should **not** become a fourth anchor. The main paper already has enough anchors: FaithEval localization→control, ITI bridge externality, and jailbreak measurement sensitivity.

### Reporting structure

Do **not** report a pooled 3-seed p-value that treats `seed × prompt` rows as independent. The H-neuron trajectory is reused across controls. Report:

* H slope.
* Each random seed slope.
* H-minus-random slope difference per seed.
* H-minus-mean-random slope difference with prompt bootstrap.
* Optional jackknife over the three random seeds, clearly labeled as rough because `n_seed = 3`.

### v3 slope estimator policy

Use **paired complete-case trajectories as the primary estimator**, and report the per-alpha valid-rate fit as a sensitivity analysis.

* Primary v3 H-neuron slope: `+0.60 pp/α`, complete-case paired trajectory estimator.
* Sensitivity: `+0.46 pp/α`, per-alpha valid-rate fit from `analyze_csv2_control.py`.
* Conclusion: unchanged, because both are weak/null on harmful-binary scoring.

Reason: the paper's causal comparisons are paired by prompt. The paired complete-case estimator better matches the intervention design and the permutation machinery. The per-alpha valid-rate estimator is useful as an attrition sensitivity check, not the main estimand.

## Operator Note — 2026-04-18 (append-only correction, does not replace mentor policy above)

These notes only correct concrete execution/planning mismatches found during repo-grounding.

### 1. API cost estimates above are materially low

The earlier rough estimates (`~$15–25`, `~$25–40`) are not token-grounded and understate likely spend for the current prompts.

Using the current official GPT-4o Batch price (`$2.50 / 1M` input tokens, `$10.00 / 1M` output tokens; Batch is the discounted async tier) and estimating token usage from the actual `evaluate_csv2_v2.py` / `evaluate_csv2.py` message builders plus existing scored payloads:

* **Legacy v2 missing cells** (`seed_1 v2` + `seed_2 v2`, 4,000 calls total):
  * input tokens: ~10.705M
  * output tokens: ~0.538M
  * estimated cost: **~$32.15**
* **v3 missing cells** (`seed_0 v3` + `seed_2 v3`, 4,000 calls total):
  * input tokens: ~16.230M
  * output tokens: ~0.970M
  * estimated cost: **~$50.28**
* **Option C total** (all 8,000 missing calls): **~$82.42**

Per-request averages from the current prompt templates:

* v2: ~2,676 input tokens + ~135 output tokens/request
* v3: ~4,058 input tokens + ~243 output tokens/request

Practical planning range:

* Option A (v3 only): **~$50**
* Option B (legacy v2 only): **~$32**
* Option C (both rulers): **~$82**

Add ~10% headroom if you expect retries, partial reruns, or materially longer-than-observed output JSONs.

### 2. One command in the legacy-v2 path is objectively wrong as written

`scripts/analyze_csv2_control.py` does **not** read legacy controls from `seed_X_unconstrained/` by default. It reads `alpha_*.jsonl` directly from the directory named by `seed_name + control_csv2_suffix`.

Because legacy v2 control outputs live under nested directories like:

* `seed_0_unconstrained/csv2_evaluation/`
* `seed_1_unconstrained/csv2_evaluation/`
* `seed_2_unconstrained/csv2_evaluation/`

the legacy analysis command in Step 4 needs the explicit suffix:

```bash
uv run python scripts/analyze_csv2_control.py \
  --seeds 0 1 2 \
  --control_csv2_suffix /csv2_evaluation \
  --output_dir data/gemma3_4b/intervention/jailbreak/control/legacy_3seed_comparison
```

Why this is a correction, not a preference:

* the existing committed `comparison_csv2_v2_summary.json` was generated with `control_csv2_suffix = /csv2_evaluation`
* without that suffix, the script would read raw unscored control files and fail the CSV2 validity checks

### 3. Seed-0 legacy p-value does not reproduce under the current canonical paired utility

This is already reflected in the locked mentor policy above, but it is worth making explicit at the bottom because Steps 6–7 still read like the old plan.

Using the current `scripts/uncertainty.py:paired_bootstrap_slope_difference()` on the existing seed-0 legacy data:

* H slope: `+2.2987 pp/α`
* seed-0 control slope: `-0.4747 pp/α`
* slope difference: `+2.7733 pp/α`
* 95% CI: `[+1.1199, +4.4107]`
* one-sided permutation p-value: **`~0.00066`** (`32 / 50,000` extreme permutations, with the utility's +1 correction)

So:

* the historical `p = 0.013` should be treated as **legacy provenance only**
* Step 7 should **not** require "confirm it matches 0.013 before trusting the 3-seed extension"
* the execution target should be: recompute seed-0 canonically, record the mismatch, then use the same canonical machinery for the 3-seed extension

### 4. Keep the v3 estimator distinction explicit in any final write-up

The repo currently contains two different v3 harmful-binary slope summaries for the H-neuron branch:

* **paired complete-case estimator**: `+0.5984 pp/α` with 95% CI `[-0.8482, +2.0160]` from `csv2_v3_evaluation/v3_slope_bootstrap.json`
* **per-alpha valid-rate fit**: `+0.4572 pp/α` from `analyze_csv2_control.py`

These are not interchangeable. For paper claims and permutation-based control comparisons, use the paired complete-case estimator as primary and keep the per-alpha fit as sensitivity only.

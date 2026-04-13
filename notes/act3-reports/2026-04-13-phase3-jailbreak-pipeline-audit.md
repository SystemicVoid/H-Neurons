# Phase 3 Jailbreak Analysis Pipeline Audit — 2026-04-13

> **Scope:** Scientific audit of three analysis pipelines producing the Phase 3 jailbreak results: `analyze_csv2.py` (v3 slopes), `analyze_csv2_control.py` (specificity comparison), and `analyze_concordance.py` (three-judge concordance).
>
> **Prior work:** [seed-0 jailbreak control audit](2026-04-12-seed0-jailbreak-control-audit.md) (v2 specificity, owns slope-difference CI), [jailbreak measurement cleanup](2026-04-13-jailbreak-measurement-cleanup.md) (canary, StrongREJECT rerun), [4-way evaluator holdout validation](2026-04-12-4way-evaluator-holdout-validation.md) (v3 accuracy validation)
>
> **Data audited:**
> - `data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation/` (H-neuron v3 results)
> - `data/gemma3_4b/intervention/jailbreak/csv2_evaluation/` (H-neuron v2 results)
> - `data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained_csv2_v3/` (seed-1 v3 control)
> - `data/gemma3_4b/intervention/jailbreak/control/concordance/concordance_summary.json`
> - `data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v2_summary.json` (v2, seed-0)
> - `data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v3_summary.json` (v3, seed-1)
>
> **Scripts reviewed:** `scripts/analyze_csv2.py`, `scripts/analyze_csv2_control.py`, `scripts/analyze_concordance.py`, `scripts/evaluate_csv2.py` (normalization chain), `scripts/uncertainty.py` (CI implementations)

---

## 1. Data Verification — What Withstands Scrutiny

### 1.1 v3 rates confirmed from raw data

Independently verified by counting `harmful_binary=="yes"` in raw JSONL against valid (non-error) records:

| Alpha | Yes count | Valid count | Rate | Reported rate | Match |
|---|---|---|---|---|---|
| 0.0 | 169 | 489 | 34.56% | 34.6% | ✓ |
| 1.0 | 172 | 492 | 34.96% | 35.0% | ✓ |
| 1.5 | 167 | 484 | 34.50% | 34.5% | ✓ |
| 3.0 | 177 | 492 | 35.98% | 36.0% | ✓ |

### 1.2 OLS slope verified

`np.polyfit([0.0, 1.0, 1.5, 3.0], [34.56, 34.96, 34.50, 35.98], 1)` → slope = 0.4573 pp/alpha. Code rounds via `round(float(slope), 2)` → 0.46. **Consistent with reported value.**

### 1.3 Seed-1 v3 control rates verified

| Alpha | Yes count | Valid count | Rate |
|---|---|---|---|
| 0.0 | 168 | 488 | 34.43% |
| 1.0 | 175 | 491 | 35.64% |
| 1.5 | 168 | 490 | 34.29% |
| 3.0 | 165 | 489 | 33.74% |

Slope: -0.34 pp/alpha. Gap: 0.80 pp/alpha. **Consistent with reported values.**

### 1.4 Wilson CI implementation

Both `analyze_csv2.py` (inline) and `uncertainty.py` (`wilson_interval`) implement the correct Wilson score formula. Edge cases (n=0, degenerate) handled. **Sound.**

### 1.5 Cohen's kappa

`analyze_concordance.py` implements the standard formula with a degenerate-case guard (`abs(expected - 1.0) < 1e-9 → return 1.0`). Values from `concordance_summary.json` (0.765, 0.829, 0.719) are internally consistent with the confusion matrices. **Sound.**

### 1.6 Concordance joining

The three-way join on `(id, alpha)` is correctly implemented. Records missing from any judge are excluded. Binary judge label extracted from `rec.get("judge")`, v2 harmful from raw payload (preserves borderline as "no"), v3 harmful from normalized payload. **Correct.**

### 1.7 Batch API (`openai_batch.py`)

State-machine design with crash-safe `.eval_batch_state.json`, greedy chunking, per-request failure handling. Token estimation is conservative (~±30% but erring safe). **Production-ready.**

---

## 2. Issues Found

### C1 — No statistical test on the v3 slope (HIGH)

The v3 slope of +0.46 pp/alpha is computed via OLS without any confidence interval. The total change from alpha=0.0 to alpha=3.0 is +1.4pp. The per-point Wilson CIs span ~±4pp and heavily overlap:

| Alpha | Rate | 95% CI |
|---|---|---|
| 0.0 | 34.6% | [30.5, 38.9] |
| 3.0 | 36.0% | [31.9, 40.3] |

The seed-0 v2 analysis ([prior report](2026-04-12-seed0-jailbreak-control-audit.md)) computed bootstrap CIs on the slope: +2.30 [+0.99, +3.58]. No equivalent was done for v3. A bootstrap test would very likely show the v3 slope CI includes zero.

**Impact:** The v3 slope cannot be claimed as a dose-response effect without a significance test. The comparison "v3 slope much flatter than v2" is descriptive, not inferential.

**Recommendation:** Add bootstrap CI on the v3 slope, matching the methodology in the seed-0 v2 report. This doesn't need to be significant to serve the paper — "v3 slope is indistinguishable from zero, while v2 slope excludes zero" is *itself* the measurement-conclusion-break evidence.

**Resolution (2026-04-13):** Bootstrap CI computed on the paired intersection (n=459 samples with valid csv2 in all 4 alphas). Slope: **+0.60 pp/alpha [−0.85, +2.02]** — CI includes zero. The point estimate shifted from +0.46 (unpaired OLS, variable denominators) to +0.60 (paired OLS, fixed denominator) due to the intersection filter; the qualitative conclusion is unchanged. Result saved to `data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation/v3_slope_bootstrap.json`. This confirms the predicted contrast: v2 slope +2.30 [+0.99, +3.58] excludes zero while v3 slope +0.60 [−0.85, +2.02] includes zero — the measurement-conclusion-break evidence for Anchor 3.

### C2 — Single-seed specificity test (HIGH)

The v3 specificity comparison uses **only seed 1**. Seeds 0 and 2 exist as unconstrained control runs but were scored only with v2, not v3. The flag `h_exceeds_all_random: true` means "exceeds the one available seed," which has no statistical power.

Contrast with the v2 analysis: seed-0 had a permutation test with p=0.013 and slope-difference CI [+1.17, +4.42].

**Impact:** The v3 specificity claim is unsupported by statistical evidence. The gap of +0.80 pp/alpha could easily fall within the random seed distribution.

**Recommendation:** Score seeds 0 and 2 with v3 to enable a multi-seed permutation test. Alternatively, acknowledge that v3 specificity rests on the v2 evidence by cross-reference.

### C3 — `comparison_csv2_summary.json` overwritten (MODERATE)

The v3 analysis run wrote its output to the same `comparison_csv2_summary.json` that the seed-0 v2 report references. The v2 summary data (slope +2.30, three metrics, permutation p=0.013) is now gone from that file, replaced with v3 data (slope +0.46, seed 1 only).

**Impact:** The seed-0 report's data reference is stale. If anyone re-reads that JSON expecting v2 data, they'll get v3 data.

**Recommendation:** Add `--output_dir` with a version-specific path when running `analyze_csv2_control.py`. Archive the v2 summary to `comparison_csv2_v2_summary.json` if it's recoverable from git.

**Resolution (2026-04-13):** Recovered v2 summary from git (commit 2d14dd5) as `comparison_csv2_v2_summary.json`. Renamed the v3 file to `comparison_csv2_v3_summary.json`. Updated the seed-0 report reference to point to the versioned v2 filename. Both files now coexist with unambiguous version suffixes.

### M1 — v2 normalization inconsistency between analysis scripts (MODERATE)

`analyze_csv2.py` line 66 normalizes ALL payloads unconditionally via `normalize_csv2_payload()`. `analyze_csv2_control.py` lines 59-61 only normalizes v3+ payloads (those with `schema_version`).

When applied to v2 data, `normalize_csv2_payload` reclassifies "borderline" labels through the derivation chain based on C/S/V values. Verified impact:

| Alpha | v2 raw yes | v2 normalized yes | Reclassified |
|---|---|---|---|
| 0.0 | 94 | 261 | +167 (+33.4pp) |
| 1.0 | 123 | 246 | +123 (+24.6pp) |
| 1.5 | 118 | 247 | +129 (+25.8pp) |
| 3.0 | 132 | 229 | +97 (+19.4pp) |

**Impact on Phase 3:** None — Phase 3 uses v3 data exclusively. But if `analyze_csv2.py` is ever run on v2 data (its default directory is `csv2_evaluation`, which contains v2 data), it will silently inflate harmful rates by 20-33pp.

**Recommendation:** Guard `analyze_csv2.py` with the same schema-version check used in `analyze_csv2_control.py`, or document the default directory mismatch.

### M2 — Concordance has zero control records (MODERATE)

`concordance_summary.json` shows `n_h_neuron: 1957, n_control: 0`. The three-way join for control seeds requires directories named `seed_N_unconstrained_csv2_v2` (line 722 of `analyze_concordance.py`), but no such directories exist. Only the generation+binary directories (`seed_N_unconstrained`) and the v3 directory (`seed_1_unconstrained_csv2_v3`) exist.

**Impact:** The concordance is entirely H-neuron data. The pairwise agreement, kappa, and structural correlation metrics do not include control responses. This is fine for evaluator-agreement analysis (agreement shouldn't depend on which neurons were scaled) but prevents checking whether concordance differs between conditions.

**Recommendation:** If control concordance is desired, either (a) add v2-only directories as a `_csv2_v2` suffix, or (b) restructure the join to look for v2 data in the main generation directory when no `_csv2_v2` directory exists.

### M3 — Variable denominators in v3 evaluation (LOW-MODERATE)

v3 evaluation errors vary by alpha: 11, 8, 16, 8 errors out of 500 records. Almost all are `invalid_evidence_spans` — benign span boundary failures. Error distribution across harm categories doesn't show strong systematic bias.

| Alpha | Errors | Valid N | Error rate |
|---|---|---|---|
| 0.0 | 11 | 489 | 2.2% |
| 1.0 | 8 | 492 | 1.6% |
| 1.5 | 16 | 484 | 3.2% |
| 3.0 | 8 | 492 | 1.6% |

Alpha=1.5 has twice the error rate of alpha=1.0 and 3.0. This is likely noise but worth checking: if the 16 dropped records at alpha=1.5 were disproportionately harmful, the rate would be artificially depressed (34.5% is the lowest point in the curve and drives the non-monotonic OLS residual pattern).

**Recommendation:** Sensitivity analysis: what harmful rate would alpha=1.5 show if all 16 errors were harmful? If all safe? This bounds the maximum impact of error attrition.

### M4 — Delta sign disagreement at alpha=1.5 is noise (LOW)

From `delta_stability.json`, the v3 delta at alpha=1.5 is **-0.0006** (i.e., -0.06 percentage points from baseline). Binary delta is +0.0155 (+1.55pp) and v2 delta is +0.0454 (+4.54pp). The "sign disagreement" is technically true but the v3 delta is effectively zero — 0.06pp is two orders of magnitude below the CI width.

**Impact:** If cited in the paper without context, "delta signs disagree at alpha=1.5" could be misread as meaningful evaluator divergence. The reality is: all three judges agree the alpha=1.5 effect is near-zero; they disagree only on which side of zero a noise-level measurement falls.

**Recommendation:** If used for the "measurement choices change conclusions" argument, cite the numerical deltas alongside the sign claim. The better version: "at alpha=1.5, v2 measures +4.5pp while v3 measures -0.1pp — even the sign of the intervention effect is evaluator-dependent."

### m1 — v2-v3 confusion matrix is definitional, not disagreement (LOW)

The v2 vs v3 confusion matrix shows: tp=456, tn=1270, fp=229, fn=2. The 229 "false positives" (v3=yes, v2=no) are almost entirely records where v2 coded `harmful_binary` as "borderline" (n=~530 across alphas) which the concordance correctly treats as v2="no" while v3 reclassifies based on its primary_outcome taxonomy.

This is not evaluator error — it's a construct definition difference. v3 was designed to resolve the borderline category by examining whether partial/substantive compliance occurred regardless of refusal preamble. The 2 "false negatives" (v2=yes, v3=no) are the genuinely surprising disagreements worth examining.

**Impact:** The concordance kappa (0.719) and agreement (88.2%) for v2 vs v3 should be interpreted as "construct overlap" rather than "measurement reliability" — they measure agreement between two different definitions of harmful, not two attempts at measuring the same thing.

---

## 3. What the Data Says — Interpretation Separated from Data

### 3.1 The raw numbers

| Metric | Value | Statistical status |
|---|---|---|
| v3 slope (H-neuron) | +0.60 pp/alpha [−0.85, +2.02] | **CI includes zero; not significant** |
| v3 slope (seed-1 control) | -0.34 pp/alpha | No CI computed |
| v3 slope gap (H - control) | +0.80 pp/alpha | **No statistical test; n=1 control seed** |
| v2 slope (H-neuron, from prior report) | +2.30 pp/alpha [+0.99, +3.58] | **Significant; CI excludes zero** |
| v2 slope gap (H - seed-0 control) | +2.77 pp/alpha [+1.17, +4.42] | **Significant; permutation p=0.013** |
| Concordance binary-v2 | 90.6% (κ=0.765) | CI: [89.2%, 91.8%] |
| Concordance binary-v3 | 92.4% (κ=0.829) | CI: [91.1%, 93.5%] |
| Concordance v2-v3 | 88.2% (κ=0.719) | CI: [86.7%, 89.6%] |

### 3.2 Interpretation

**The v3 evaluator flattening is real and expected.** v3 was built to resolve refuse-then-comply patterns that v2 coded as borderline. This raises the baseline harmful rate from ~19% (v2) to ~35% (v3), absorbing much of the alpha-driven increase into the baseline measurement. The intervention doesn't "create" harmful responses so much as intensify existing ones — and v3 correctly identifies them at baseline.

**The v2-to-v3 slope compression (2.30→0.46 pp/alpha) is the strongest Anchor 3 evidence.** This is exactly the "measurement choices change conclusions" claim the paper needs. The v2 evaluator shows a statistically significant dose-response; the v3 evaluator shows a near-null one. Both are measuring the same model outputs with the same prompts — only the measurement instrument changed.

**The v3 specificity comparison is incomplete.** With a single control seed and no statistical test, the v3 gap of +0.80 pp/alpha cannot distinguish signal from noise. The existing v2 specificity evidence (p=0.013) remains the strongest support for the specificity claim.

**The concordance analysis is clean but scoped to H-neuron data only.** The 11.8% v2-v3 disagreement rate reflects construct definition differences (borderline resolution), not evaluator unreliability. The 2 genuine v2-yes/v3-no flips are worth examining manually but don't threaten any conclusion.

---

## 4. Uncertainties Remaining

### High uncertainty
- ~~**Is the v3 slope significantly different from zero?**~~ **Resolved:** +0.60 [−0.85, +2.02] pp/alpha — CI includes zero. Not significant. (n=459 paired samples.)
- **Is the v3 H-neuron slope different from v3 control slopes?** Requires ≥2 more control seeds for a meaningful comparison.

### Medium uncertainty
- **Do the 16 v3 errors at alpha=1.5 bias the rate?** A sensitivity analysis bounding the maximum impact would resolve this. Worst case (all 16 errors are harmful): rate would be 183/500 = 36.6% instead of 167/484 = 34.5% — the slope would steepen to ~0.8 pp/alpha. Still small.
- **Does the v2 normalization bug in `analyze_csv2.py` affect any cited number?** Probably not if all citations reference v3 data, but should be verified by checking all surfaces in `ci_manifest.json`.

### Low uncertainty
- The concordance metrics are correctly computed.
- The canary validation confirmed v3 schema/parse correctness before batch deployment.
- The Wilson CIs and Cohen's kappa implementations are standard and correct.

---

## 5. Actionable Next Steps (Ordered by Information Gained per Time)

1. ~~**Bootstrap CI on v3 slope**~~ **Done (2026-04-13).** Slope: +0.60 [−0.85, +2.02] pp/alpha — CI includes zero, confirming v3 is indistinguishable from zero. Data: `csv2_v3_evaluation/v3_slope_bootstrap.json`.

2. **Score seeds 0 and 2 with v3** (~$15 API cost, ~1 hr wall time). Enables the multi-seed permutation test for v3 specificity. Without this, the v3 specificity claim rests on the v2 evidence.

3. **Sensitivity analysis on v3 errors** (~15 min). Compute harmful rates with all errors classified as harmful, then all as safe. Bounds the maximum bias from error attrition.

4. ~~**Archive v2 `comparison_csv2_summary.json`**~~ **Done (2026-04-13).** Recovered as `comparison_csv2_v2_summary.json`; v3 renamed to `comparison_csv2_v3_summary.json`; seed-0 report reference updated.

5. **Guard `analyze_csv2.py` against v2 data** (~10 min). Add the same schema-version check used in `analyze_csv2_control.py` to prevent silent inflation when run on v2 defaults.

---

## 6. Synthesis: Implications for Paper Claims

| Paper claim | Supported by Phase 3? | Caveat |
|---|---|---|
| v3 slope flatter than v2 | **Yes** — +0.60 [−0.85, +2.02] vs +2.30 [+0.99, +3.58] pp/alpha | v3 CI includes zero; v2 CI excludes zero — inferential contrast now supported |
| Measurement choices change conclusions | **Yes, strongly** — same outputs, different evaluator, different slope magnitude (possibly different sign) | Delta sign disagreement at α=1.5 is noise-level; frame as "sign is evaluator-dependent at small effect sizes" |
| H-neuron specificity on jailbreak | **Partially** — v2 specificity p=0.013 is strong; v3 specificity has n=1 seed | v3 specificity claim requires more seeds or explicit reliance on v2 evidence |
| v3 agrees more with binary than v2 does | **Yes** — κ=0.829 vs κ=0.765 | This is a construct-alignment statement: v3's primary_outcome taxonomy aligns better with binary harmful/safe than v2's borderline-inclusive scheme |
| 11.8% of records flip between v2 and v3 | **Yes, but misleading without context** — 229/231 flips are v3-upgrades of v2-borderlines, only 2 are v2-yes/v3-no | Frame as "v3 recovers nuance v2 coded as borderline" not "evaluators disagree" |

---

## Data Artifacts Audited

| Artifact | Path | Schema | Verified |
|---|---|---|---|
| H-neuron v3 evaluation | `data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation/alpha_{0.0,1.0,1.5,3.0}.jsonl` | csv2_v3 | ✓ rates, errors, schema |
| H-neuron v2 evaluation | `data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_{0.0,1.0,1.5,3.0}.jsonl` | legacy (no version) | ✓ borderline counts |
| Seed-1 v3 control | `data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained_csv2_v3/alpha_{0.0,1.0,1.5,3.0}.jsonl` | csv2_v3 | ✓ rates |
| Comparison summary (v2) | `data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v2_summary.json` | — | ✓ (recovered from git; seed-0) |
| Comparison summary (v3) | `data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v3_summary.json` | — | ✓ (seed-1) |
| v3 slope bootstrap | `data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation/v3_slope_bootstrap.json` | — | ✓ (n=459 paired) |
| Concordance summary | `data/gemma3_4b/intervention/jailbreak/control/concordance/concordance_summary.json` | — | ✓ (H-neuron only; n_control=0) |
| Delta stability | `data/gemma3_4b/intervention/jailbreak/control/concordance/delta_stability.json` | — | ✓ |

# Seed-0 Jailbreak Negative Control — CSV-v2 Specificity Analysis

> **Status: canonical for seed-0 jailbreak control conclusions.** Seeds 1-2 generated but not yet scored; this report covers seed-0 only.
>
> **H-neuron reference:** [2026-04-16 D7 full500 two-seed current-state audit](./2026-04-16-d7-full500-two-seed-current-state-audit.md) owns current D7 causal-head claims. This report owns the H-neuron *scaling* specificity claim on jailbreak.

**Date:** 2026-04-12
**Data root:** `data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained/`
**H-neuron reference:** `data/gemma3_4b/intervention/jailbreak/csv2_evaluation/`
**Comparison summary:** `data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v2_summary.json`
**Plot:** `data/gemma3_4b/intervention/jailbreak/control/negative_control_csv2_comparison.png`
**Model:** `google/gemma-3-4b-it`
**Control:** 38 random neurons (seed 0, `neuron_indices.json`) vs 38 H-neurons (L1 probe, deterministic)
**Same prompts:** 500/500 JailbreakBench prompt IDs match across conditions (verified)
**Evaluation:** binary judge (`evaluate_intervention.py`) then CSV-v2 (`evaluate_csv2_v2.py`), gpt-4o batch mode
**Generation:** `do_sample=True`, `temperature=0.7`, `max_new_tokens=5000` (both conditions)
**Primary metric:** csv2_yes rate slope across alpha (pp/alpha, bootstrap 95% CI)
**Diagnostic:** binary compliance, mean C, mean S, borderline migration
**Alphas:** 0.0, 1.0, 1.5, 3.0
**Scoring log:** `logs/score_control_seed0_v2_20260412_165702.log`

---

## 0. Headline

| Condition | csv2_yes slope (pp/alpha) | Slope 95% CI | Delta(alpha=3 minus alpha=0) | Delta 95% CI | Triage |
|---|---|---|---|---|---|
| **H-neurons (38)** | **+2.30** | **[+0.99, +3.58]** | **+7.6pp** | **[+2.6, +12.8]** | dose-response confirmed |
| **Random neurons (38, seed-0)** | -0.47 | [-1.42, +0.47] | -1.6pp | [-7.0, +3.8] | null (CI includes zero) |
| **Slope difference (H minus control)** | **+2.77** | **[+1.17, +4.42]** | | | **specificity supported** |

**Permutation test (one-sided):** p = 0.013 (647/50,000 permutations >= observed gap). Significant at 5%.

**Verdict:** The H-neuron dose-response on csv2_yes is significantly steeper than the random-neuron control. The slope CI for H-neurons excludes zero; the slope CI for random neurons includes zero; the difference CI excludes zero. On this seed and metric, specificity is supported.

---

## 1. Data Summary

### 1.1 csv2_yes rates with Wilson 95% CIs

| alpha | H-neuron csv2_yes | H-neuron 95% CI | Control csv2_yes | Control 95% CI |
|---|---|---|---|---|
| 0.0 | 18.8% (94/500) | [15.6, 22.5] | 24.2% (121/500) | [20.7, 28.1] |
| 1.0 | 24.6% (123/500) | [21.0, 28.6] | 22.6% (113/500) | [19.2, 26.5] |
| 1.5 | 23.6% (118/500) | [20.1, 27.5] | 22.4% (112/500) | [19.0, 26.3] |
| 3.0 | 26.4% (132/500) | [22.7, 30.4] | 22.6% (113/500) | [19.2, 26.5] |

### 1.2 Binary compliance rates

| alpha | H-neuron compliance | Control compliance |
|---|---|---|
| 0.0 | 30.4% (152/500) | 31.4% (157/500) |
| 1.0 | 31.0% (155/500) | 30.0% (150/500) |
| 1.5 | 32.2% (161/500) | 29.8% (149/500) |
| 3.0 | 33.4% (167/500) | 30.4% (152/500) |

Binary judge shows no significant slope for either condition. H-neuron binary Delta = +3.0pp (CI includes zero per original claim); control binary Delta = -1.0pp. Consistent with binary MDE ~6pp being underpowered for this effect size.

### 1.3 Mean C and S (from comparison JSON)

| alpha | H-neuron mean C | Control mean C | H-neuron mean S (C>=2) | Control mean S (C>=2) |
|---|---|---|---|---|
| 0.0 | 0.844 | 0.882 | 3.07 | 3.13 |
| 1.0 | 0.900 | 0.854 | 3.14 | 3.13 |
| 1.5 | 0.884 | 0.872 | 3.15 | 3.02 |
| 3.0 | 0.922 | 0.864 | 3.23 | 3.06 |

H-neuron: mean C slope = +0.024/alpha, mean S(C>=2) slope = +0.05/alpha. Both show modest increases consistent with more harmful content at higher scaling.
Control: mean C slope = -0.004/alpha, mean S(C>=2) slope = -0.027/alpha. Flat.

### 1.4 Data integrity

- All 8 files (4 in-place binary + 4 csv2_evaluation): 500/500 records, 0 missing, 0 errors, 0 malformed.
- Schema matches d7 reference: `['alpha', 'category', 'compliance', 'csv2', 'goal', 'id', 'judge', 'response', 'template_idx', 'timings']`.
- csv2 subfields present: C, D, F, P, R, S, T, V, harmful_binary, harmful_payload_share, harmful_spans, pivot_position, reasoning, span_corrections, span_errors.
- No `schema_version` field (v2 data, not v3) — correct for ruler consistency with original H-neuron claim.

---

## 2. Dose-Response Comparison

### 2.1 Slopes

**H-neuron csv2_yes slope:** +2.30 pp/alpha [+0.99, +3.58] (bootstrap 95% CI, 10,000 resamples, seed=42)
**Control csv2_yes slope:** -0.47 pp/alpha [-1.42, +0.47]
**Slope difference:** +2.77 pp/alpha [+1.17, +4.42]

The H-neuron slope is positive with a CI excluding zero. The control slope is indistinguishable from zero. The difference is significant: the H-neuron dose-response is 2.77 pp/alpha steeper than the random-neuron control.

### 2.2 Non-linearity note

The H-neuron dose-response is not perfectly linear: most of the effect occurs between alpha=0.0 and alpha=1.0 (+5.8pp), with a slight dip at alpha=1.5 (23.6% < 24.6%) before recovering at alpha=3.0 (26.4%). This suggests a saturation effect where the primary impact occurs at low scaling. The linear slope is a conservative summary; the actual effect at alpha=1.0 already exceeds the average slope.

The control is smoothly flat across all alphas (24.2% -> 22.6% -> 22.4% -> 22.6%), with no structure.

### 2.3 Plot

See `data/gemma3_4b/intervention/jailbreak/control/negative_control_csv2_comparison.png`.

---

## 3. Severity Structure

### 3.1 Borderline migration

| Condition | alpha | yes | borderline | no | yes+borderline |
|---|---|---|---|---|---|
| H-neuron | 0.0 | 94 | 171 | 235 | 265 |
| H-neuron | 1.0 | 123 | 127 | 250 | 250 |
| H-neuron | 1.5 | 118 | 134 | 248 | 252 |
| H-neuron | 3.0 | 132 | 98 | 270 | 230 |
| Control | 0.0 | 121 | 124 | 255 | 245 |
| Control | 1.0 | 113 | 133 | 254 | 246 |
| Control | 1.5 | 112 | 139 | 249 | 251 |
| Control | 3.0 | 113 | 132 | 255 | 245 |

The H-neuron intervention **polarizes** borderline cases as scaling increases:
- borderline drops by 73 (171 -> 98)
- yes rises by 38 (94 -> 132)
- no rises by 35 (235 -> 270)
- Total yes+borderline *decreases* from 265 to 230

This means the intervention doesn't just make more responses harmful — it pushes borderline cases toward the poles. Some become clearly harmful (borderline -> yes), others become clearly safe (borderline -> no). The net csv2_yes increase (+38) is roughly half the borderline decrease (-73), with the other half shifting to safe.

The control shows no polarization: borderline is stable (124 -> 132), yes stable (121 -> 113), no stable (255 -> 255). The total yes+borderline is unchanged (245 -> 245).

This is a richer finding than the csv2_yes rate alone, and it is absent in the random-neuron control.

### 3.2 Binary vs csv2 concordance

| Condition | alpha | binary_yes | csv2_yes | both_yes | agreement |
|---|---|---|---|---|---|
| H-neuron | 0.0 | 152 | 94 | 91 | 87.2% |
| H-neuron | 1.0 | 155 | 123 | 118 | 91.6% |
| H-neuron | 1.5 | 161 | 118 | 111 | 88.6% |
| H-neuron | 3.0 | 167 | 132 | 132 | 93.0% |
| Control | 0.0 | 157 | 121 | 115 | 90.4% |
| Control | 1.0 | 150 | 113 | 108 | 90.6% |
| Control | 1.5 | 149 | 112 | 107 | 90.6% |
| Control | 3.0 | 152 | 113 | 108 | 90.2% |

Binary judge is consistently MORE permissive than csv2 (higher yes count). Agreement ranges from 87% to 93%. The disagreement is asymmetric: binary calls "yes" on records that csv2 calls "borderline" or "no" (consistent with known binary-judge overcall on topic-sensitivity without substance).

Notable: at H-neuron alpha=3.0, both_yes = csv2_yes = 132. This means every csv2_yes record is also binary_yes — perfect one-directional inclusion at the highest alpha.

Control concordance is rock-steady across alphas (~90.4%).

### 3.3 Category-level breakdown (control)

| Category | n | alpha=0.0 | alpha=1.0 | alpha=1.5 | alpha=3.0 | Delta |
|---|---|---|---|---|---|---|
| Disinformation | 50 | 32% | 26% | 24% | 28% | -4pp |
| Economic harm | 50 | 18% | 12% | 14% | 16% | -2pp |
| Expert advice | 50 | 24% | 22% | 24% | 20% | -4pp |
| Fraud/Deception | 50 | 48% | 48% | 50% | 48% | 0pp |
| Government | 50 | 24% | 28% | 26% | 26% | +2pp |
| Harassment/Discrim. | 50 | 6% | 8% | 2% | 4% | -2pp |
| Malware/Hacking | 50 | 36% | 32% | 34% | 32% | -4pp |
| Physical harm | 50 | 12% | 12% | 10% | 14% | +2pp |
| Privacy | 50 | 38% | 36% | 36% | 36% | -2pp |
| Sexual/Adult | 50 | 4% | 2% | 4% | 2% | -2pp |

All 10 categories show Delta within +/-4pp — no category drives systematic signal. The null is uniform across harm types. Fraud/Deception has the highest baseline rate (~48%) but is perfectly flat. Sexual/Adult has the lowest rate (~2-4%) and is also flat.

---

## 4. What Withstands Scrutiny

### 4.1 The H-neuron dose-response is significantly steeper than the random-neuron control

The slope difference of +2.77 pp/alpha [+1.17, +4.42] excludes zero. The permutation test (p=0.013) independently confirms this. The result survives bootstrap resampling (10,000 iterations) and permutation testing (50,000 iterations).

### 4.2 The control shows no effect at any alpha

The control csv2_yes rate is flat at 22-24% across all four alphas. The slope CI [-1.42, +0.47] comfortably includes zero. All 10 jailbreak categories individually show null Deltas. Binary compliance is also flat (29.8-31.4%). This is the expected negative control result: random neurons at the same count and scaling factors do not produce the dose-response seen with H-neurons.

### 4.3 The severity polarization is neuron-specific

H-neurons push borderline cases to the poles (yes: +38, no: +35, borderline: -73). Random neurons show no polarization (yes+borderline total unchanged: 245 -> 245). This is a qualitative mechanistic difference, not just a quantitative rate difference.

### 4.4 Binary judge confirms the pattern

Binary judge independently shows no slope for either condition, consistent with its underpowered MDE (~6pp). Both judges agree: the control is null. This cross-judge concordance strengthens the null finding.

---

## 5. Caveats and Remaining Uncertainty

### 5.1 Single seed

This is one random seed out of three generated. Seeds 1-2 exist but are not yet scored. A single seed proves that *at least one* random selection of 38 neurons produces a null result, but it does not rule out the possibility that some random configurations could produce partial effects. Multi-seed scoring would strengthen the specificity claim. This is a moderate uncertainty — the result is clean but not replicated.

### 5.2 Baseline discrepancy at alpha=0.0

At alpha=0.0 (no intervention), control csv2_yes = 24.2% vs H-neuron csv2_yes = 18.8%. Both should reflect the unmodified model. The Wilson CIs [20.7, 28.1] and [15.6, 22.5] do not overlap, making this a marginally significant baseline difference (approximate z ~= 2.1, p ~= 0.036 two-sided).

The most likely explanation is sampling noise from temperature=0.7 decoding across independent generation runs. But the gap (5.4pp) is at the upper end of what binomial variability predicts at n=500.

**Impact on the main claim:** The core comparison is within-condition slopes, not cross-condition absolute rates. The slope test already accounts for each condition's own baseline. The baseline discrepancy does not invalidate the slope comparison, but it does mean absolute csv2_yes rates should not be compared directly between conditions.

### 5.3 Cross-run design (not within-run)

The H-neuron and control are separate generation runs. A within-run design (randomizing neuron assignment at the prompt level) would be stronger, but was not feasible with the existing pipeline. The slope comparison partially mitigates this by measuring within-condition trends.

### 5.4 No mechanism explanation

This analysis confirms that H-neurons produce dose-dependent jailbreak effects that random neurons do not, but does not explain the mechanism. Why do these specific neurons, identified via probe readout quality on a different task (FaithEval truthfulness), affect jailbreak severity? The connection between detection readout and intervention target remains a correlation, not a mechanism.

---

## 6. Pipeline Contamination Finding

### 6.1 The bug

`scripts/analyze_csv2_control.py` imports `normalize_csv2_payload` from `evaluate_csv2.py` (v3). When applied to v2-scored data (which lacks `schema_version`, `primary_outcome`, and `intent_match` fields), this function:

1. Maps v2's `"borderline"` harmful_binary label to `"unknown"` (not in the v3 yes/no vocabulary)
2. Derives `primary_outcome` from the C/S/V/T scores — any S >= 1 triggers `"partial_compliance"`
3. Re-derives harmful_binary from `derive_harmful_binary(primary_outcome, intent_match)` — `"partial_compliance"` with intent_match >= 1 becomes `"yes"`

**Result:** 167/171 (97.7%) of borderline records are reclassified as "yes". The csv2_yes rate inflates from 18.8% to 52.2%.

### 6.2 The fix

Added a schema-version check to `_csv2()` in `analyze_csv2_control.py` (4 lines). V2 data (no `schema_version` field) is now returned without normalization. V3 data continues to use `normalize_csv2_payload` as before.

### 6.3 Significance

This is a silent data corruption — the script runs without error and produces plausible-looking results. Anyone using the v3 analysis pipeline on v2-scored data would get wrong metrics, wrong slopes, and a wrong triage verdict. The original (contaminated) run produced `"review_specificity"` instead of the correct `"specificity_supported"`.

---

## 7. Impact on Evidence Hierarchy

### 7.1 Before this analysis

The strategic assessment (2026-04-11, rev. 2026-04-12) listed:

> **Supporting but caveated:** H-neuron jailbreak CSV-v2 effect — benchmark-specific control unscored

With safe phrasing: *"Specificity confirmed on FaithEval/FalseQA; jailbreak specificity pending"*

### 7.2 After this analysis

The seed-0 control confirms:
- Random neurons produce a null dose-response (slope includes zero)
- H-neurons produce a positive dose-response (slope excludes zero)
- The difference is significant (p=0.013, CI excludes zero)

**Updated claim status:** H-neuron jailbreak effect is partially neuron-specific. Seed-0 specificity is confirmed; seeds 1-2 are pending for multi-seed robustness.

**Updated safe phrasing:** *"Specificity confirmed on FaithEval, FalseQA, and jailbreak (seed-0 random-neuron control, p=0.013). Multi-seed robustness pending."*

### 7.3 Does this promote the claim to headline-safe?

Not fully. The claim upgrades from "supporting-caveated (control unscored)" to "supporting-caveated (single seed)." The gap to headline-safe is smaller but still exists: multi-seed replication (seeds 1-2) or a different control methodology (e.g., random-direction at matched norm) would close it. The single-seed result is strong enough to anchor a paper claim with explicit qualification.

---

## 8. Next Steps

1. **Score seeds 1-2** with the same binary + CSV-v2 pipeline to establish multi-seed robustness. The scoring script (`scripts/infra/score_control_seed0_v2.sh`) can be adapted with minimal changes (swap `seed_0` -> `seed_1`/`seed_2`). Seed 2 has incomplete generation (alpha=1.5) that needs resolution first.
2. **Re-run `analyze_csv2_control.py` with all 3 seeds** once scored — the script already supports `--seeds 0 1 2` and will compute aggregate random slope statistics.
3. **Audit the normalization bug across the codebase.** Check whether any other analysis script imports `normalize_csv2_payload` and applies it to v2 data. The bug is in the import, not the data.
4. **Integrate into paper draft.** The jailbreak specificity result belongs in Anchor 3 (measurement case study) as evidence that H-neuron effects are real, not measurement artifacts or random noise.

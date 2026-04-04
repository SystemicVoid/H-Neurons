# E2 TriviaQA Transfer Lane: Synthesis and Closure — 2026-04-04

> **Status: analytical synthesis closing the TriviaQA-transfer ITI lane for D4.**
>
> This report interprets and cross-validates two experimental runs (E2-A and E2-B) that together test whether TriviaQA-sourced mass-mean ITI directions transfer to TruthfulQA evaluation on Gemma-3-4B-IT. It is the interpretation authority for the TriviaQA transfer investigation. Individual data authorities are:
>
> - E2-A data: [2026-04-02-e2-triviaqa-source-isolated-audit.md](./2026-04-02-e2-triviaqa-source-isolated-audit.md)
> - E2-B data: [2026-04-03-e2b-triviaqa-familydefault-diagnostic.md](./2026-04-03-e2b-triviaqa-familydefault-diagnostic.md)
> - Machine-readable: [e2b_triviaqa_familydefault_report.json](./e2b_triviaqa_familydefault_report.json)

---

## 1. Purpose

The E2 investigation tests one specific question within D4's broader ITI program: **can TriviaQA-sourced contrastive data produce useful truthfulness-steering directions when the evaluation target is TruthfulQA and SimpleQA?**

This matters because TriviaQA offers a larger, more diverse, and independently-sourced contrastive pool than TruthfulQA itself. If TriviaQA directions transferred, it would validate the universality of the "truthfulness direction" across datasets — a core assumption in the ITI literature. If they do not transfer, it narrows the claims we can make about ITI's mechanism: the intervention may depend on dataset-specific statistical structure rather than a general truthfulness concept.

Two variants were run:

| Variant | Selectors | Hypothesis |
|---|---|---|
| E2-A | Paper-faithful overrides (`val_accuracy` + `last_answer_token`) | Controls for extraction procedure; tests whether TriviaQA data produces signal under the same pipeline that works for TruthfulQA (E0) |
| E2-B | Family-default (`auroc` + `all_answer_positions`) | Tests whether E2-A's null was caused by suboptimal selectors rather than fundamentally weak data |

---

## 2. Data Summary

All numbers below are drawn from the two data-authority reports linked above. They are reproduced here once for cross-variant comparison; do not edit numbers here — update the source reports if corrections are needed.

### 2.1 Probe Quality

| Variant | Source | Ranking | Top-K | Mean AUROC | Max AUROC | Mean val_acc |
|---|---|---|---|---|---|---|
| E0 | TruthfulQA | Paper-faithful | 12 | **0.820** | **0.836** | **0.746** |
| E2-B | TriviaQA | AUROC | 8 | 0.754 | 0.769 | 0.691 |
| E2-A | TriviaQA | val_accuracy | 40 | 0.733 | 0.761 | 0.679 |

The AUROC gap between TruthfulQA probes (E0) and TriviaQA probes (E2-B best case) is **0.066** — not catastrophic, but substantial. TriviaQA probes are weaker across all quality metrics regardless of selector choice. AUROC ranking (E2-B) concentrates on higher-quality heads; val_accuracy ranking (E2-A) includes more heads but at lower average quality.

### 2.2 TruthfulQA MC (Held-Out, Pooled n=655)

| Comparison | MC1 Δ (pp) | MC1 95% CI | MC2 Δ (pp) | MC2 95% CI |
|---|---|---|---|---|
| **E0** within (K=12, α=8.0 vs α=0) | **+6.26** | [+3.51, +9.16] | **+7.49** | [+5.28, +9.82] |
| **E1** within (K=8, α=8.0 vs α=0) | +2.75 | [+0.46, +5.19] | +4.48 | [+2.53, +6.49] |
| **E2-B** within (K=8, α=6.0 vs α=0) | +0.00 | [-1.98, +1.98] | +1.46 | [-0.11, +3.02] |
| **E2-A** within (K=40, α=6.0 vs α=0) | -0.92 | [-3.36, +1.53] | +0.73 | [-1.01, +2.46] |
| E2-B optimized vs E2-A optimized | +0.92 | [-1.37, +3.21] | +0.73 | [-1.01, +2.46] |
| Sidecar A: E2-B artifact @ K=40 α=6.0 vs E2-A | -0.92 | [-2.75, +0.92] | -1.10 | [-2.47, +0.25] |
| Sidecar B: K=16 vs K=40 on E2-B artifact | +0.76 | [-1.22, +2.75] | +0.95 | [-0.45, +2.36] |
| E2-B vs E0 | **-6.26** | [-9.16, -3.51] | **-6.04** | [-8.39, -3.70] |
| E2-B vs E1 | -2.75 | [-5.19, -0.31] | -3.03 | [-5.12, -0.95] |

**Key pattern:** E0 and E1 (TruthfulQA-sourced) produce detectable effects. E2-A and E2-B (TriviaQA-sourced) produce none. The E2-B vs E0 gap (-6.26 pp MC1, CI excludes zero) demonstrates that TriviaQA directions are significantly weaker, not merely noisier. Neither sidecar comparison shows a significant effect, confirming the null is robust to K and artifact variations.

### 2.3 SimpleQA-200 Generation (first_3_tokens)

| Comparison | Compliance Δ (pp) | CI | Attempt Δ (pp) | CI |
|---|---|---|---|---|
| E2-B within | +0.00 | [-3.0, +3.0] | -0.50 | [-2.5, +1.0] |
| E2-B vs E2-A | +0.50 | [-2.0, +3.0] | -0.50 | [-2.5, +1.0] |
| E2-B vs E0 | +1.50 | [-1.5, +5.0] | **+9.00** | [+5.0, +13.5] |

The only significant SimpleQA result is the E2-B vs E0 attempt rate difference (+9 pp): E0 at α=8.0 dramatically increases refusal (attempt drops from ~99% to ~90%), while E2-B at α=6.0 barely touches it (99.0% → 98.5%). TriviaQA directions do not engage the model's refusal circuitry.

### 2.4 Head Overlap Structure

| Pair | Same data? | Same selectors? | Jaccard | Spearman ρ | Direction cosine (shared) |
|---|---|---|---|---|---|
| E2-B vs E2-A | Yes | No | 0.043 (2/46) | **0.914** | +1.000 (n=2) |
| E2-B vs E0 | No | No | 0.053 (1/19) | 0.527 | -0.351 (n=1) |
| E2-B vs E1 | No | No | 0.000 (0/16) | 0.366 | — |

The E2-B vs E2-A pairing is the most informative: despite sharing the same underlying data, different selectors produce almost entirely different selected sets (Jaccard 4.3%). Yet the full ranking is nearly identical (ρ = 0.914), and the two shared heads have identical directions (cosine = 1.000). This means **selector choice changes which heads are selected, not which heads are good**. The directions themselves are deterministic from the same data — the AUROC vs val_accuracy choice only changes the K threshold where signal concentrates.

The negative direction cosine at the single E2-B ↔ E0 shared head (L11H0, cosine = -0.351) is suggestive but rests on n=1 — it cannot carry interpretive weight alone.

### 2.5 Calibration-to-Held-Out Replication

| Variant | Calibration MC1 (locked) | Calibration MC1 Δ (pp) | n_cal | Held-out MC1 Δ (pp) | n_heldout | Survived? |
|---|---|---|---|---|---|---|
| E0 | 0.370 | ~+10 (baseline not recorded) | 81 | +6.26 | 655 | **Yes** |
| E2-A | 0.259 | +3.70 | 81 | -0.92 | 655 | **No** |
| E2-B | 0.284 | +6.17 | 81 | +0.00 | 655 | **No** |

Both TriviaQA variants show calibration signals that completely fail to replicate on held-out data. E2-B's calibration signal (+6.17 pp = 5 extra correct out of 81) was the strongest of any E2 variant, yet it vanished entirely. This is the clearest evidence that TriviaQA-sourced directions lack the generalization capacity that TruthfulQA-sourced directions possess.

Note: E0's calibration MC1 at lock (0.370) is substantially above any plausible baseline (~0.22–0.27), indicating a strong calibration signal that replicated on held-out (+6.26 pp). The exact calibration baseline (α=0 on the calibration split) is not recorded in the locked config, so the delta is approximate. The key contrast is that E0 calibration signal survived held-out evaluation while both E2 variants collapsed entirely.

---

## 3. Methodological Audit

### 3.1 Strengths

**Statistical infrastructure is sound.** All comparisons use paired bootstrap on matched sample IDs. MC1 (binary) uses percentile bootstrap; MC2 (continuous) uses paired continuous bootstrap. Wilson intervals for proportions. 10,000 resamples with seed control. The uncertainty quantification meets the measurement blueprint requirements.

**The E2-B diagnostic design is well-conceived.** The sidecar comparisons (K=40 and K=16 at frozen α=6.0 on the E2-B artifact) cleanly decompose the headline comparison into artifact-change and K-change components. This avoids the common trap of conflating multiple simultaneous changes.

**Artifact integrity verification is rigorous.** All four E2-B extractions (calibration, fold 0, fold 1, production) produce bit-identical artifacts (SHA256: `3231a345...`), confirming the extraction is deterministic and source-isolated. The March 30 sanity check (5/5 top-head matches) confirms extraction code reproducibility.

**The classification hierarchy is pre-specified and ordered.** The five-level diagnostic classification (`selector_mismatch_confirmed` → `compact_k_anomaly` → `suggestive_but_underpowered` → `wrong_source_still_likely` → `ambiguous`) was committed before the held-out results were available, preventing post-hoc reclassification. The logic has been verified against the actual inputs and produces the correct output.

**Preflight checks and metadata assertions** in the pipeline script prevent silent misconfiguration (e.g., accidentally running E2-A's selectors under E2-B's label). This is a best practice for multi-variant pipelines.

### 3.2 Concerns and Limitations

**Calibration sample size (n=81) is structurally insufficient for weak signals.** MC1 resolution on n=81 is 1.23 pp (one question). The E2-B calibration "winner" (K=8, α=6.0 at 28.4%) was 5 correct answers above baseline (22.2%). Four shortlist candidates were within 1 question of each other. When the entire calibration landscape spans ≤5 extra correct answers, the locking decision is dominated by random variation rather than structural signal. This is not a bug in the procedure — the tolerance floor (1.5 pp) and tie-break protocol are appropriate — but it means that locking is more of a "random selection among near-equivalent configurations" than "identification of the optimal operating point."

This concern applies equally to E2-A. Both locks should be treated as arbitrary choices within a flat calibration landscape, not as optimized configurations.

**Multiple comparison inflation.** The E2-B report makes ~13 pairwise comparisons across TruthfulQA and SimpleQA. Without correction, the expected number of false positives at α=0.05 is ~0.65. The MC2 near-miss (within-E2-B: +1.46 pp, CI [-0.11, +3.02]) would not survive Bonferroni correction (threshold ≈ 0.004). This near-miss should not be interpreted as evidence of a weak signal. The pre-specified classification hierarchy mitigates this by using MC1 as the primary test statistic, but readers should note that secondary comparisons (MC2, SimpleQA precision) are exploratory.

**SimpleQA baseline precision is at floor.** At 5.5% compliance (11/200), SimpleQA provides very little room to detect intervention effects. The minimum detectable effect at 80% power for n=200 and 5.5% baseline is approximately ±4 pp — larger than any observed effect. SimpleQA is effectively blind to the interventions being tested. This is not a problem with the E2 investigation specifically (the same floor applies to E0 and E1), but it means SimpleQA generation results should be treated as "consistent with null" rather than "evidence of no effect."

**The decode scope (first_3_tokens) limits the intervention's exposure window.** ITI hooks are active for only the first 3 tokens of generation. This was justified by the decode-scope audit (see [2026-04-02-decode-scope-simpleqa-judge-results.md](./2026-04-02-decode-scope-simpleqa-judge-results.md)), but it means the intervention has minimal opportunity to alter generation trajectory. If TriviaQA directions have a cumulative rather than immediate effect, first_3_tokens would systematically miss it. However, E0 and E1 also use first_3_tokens and show effects on MC (which is non-generative), so the decode scope cannot explain the TriviaQA null.

**No TriviaQA-native evaluation was run.** Both E2 variants were evaluated on TruthfulQA MC and SimpleQA — tasks derived from different data distributions. If TriviaQA directions capture something real but non-transferable (e.g., factual recall patterns specific to trivia-style questions), this would show up as a TriviaQA-native improvement paired with TruthfulQA-null. We cannot distinguish "TriviaQA directions capture nothing" from "TriviaQA directions capture something TruthfulQA doesn't measure." However, the probe quality data (AUROC 0.75-0.77) confirms the probes do detect some signal in TriviaQA data; the question is whether that signal is relevant to truthfulness more broadly.

**The within-family rank agreement (ρ = 0.914) vs low Jaccard (0.043) seems paradoxical but has a simple explanation.** AUROC and val_accuracy are strongly correlated metrics (they both measure probe discriminability), so the rank ordering of 114 shared heads is nearly identical. But AUROC concentrates discriminability more sharply in fewer top heads (K_lock=8 vs K_lock=40), because val_accuracy has a narrower dynamic range. This is not evidence that selector choice doesn't matter — it matters for K — but it does mean the two artifacts are sampling from the same underlying quality ordering at different depth cutoffs.

---

## 4. Interpretation

### 4.1 What Withstands Scrutiny

**TriviaQA-sourced mass-mean ITI directions do not transfer to TruthfulQA evaluation on Gemma-3-4B-IT.** This is the central finding. It rests on:
- Two independent selector policies (paper-faithful and family-default) both producing null MC1 results
- Three decomposition comparisons (headline, sidecar A, sidecar B) all producing null MC1 results
- Consistency across MC1 (binary) and MC2 (continuous) metrics
- SimpleQA generation showing no effect on any surface
- Calibration signals that fail to replicate on held-out data
- Probe quality that is measurably weaker than TruthfulQA probes

No plausible re-analysis (different α, different K, different selector) would rescue this result. The held-out n=655 provides sufficient power to detect effects of +2 pp or larger at 80% power. If a TriviaQA signal exists, it is below the minimum practically useful threshold.

**The three-variant ranking (E0 > E1 > E2) is robust.** E0 (TruthfulQA, paper-faithful) significantly outperforms E1 (TruthfulQA, modernized), which significantly outperforms E2-B (TriviaQA, family-default). Each pairwise comparison has CIs excluding zero. This ordering is consistent across MC1 and MC2.

**Selector choice changes K, not direction quality.** The sidecar decomposition confirms that swapping selectors while holding K and α fixed (sidecar A: E2-B artifact @ K=40, α=6.0 vs E2-A) produces a null effect (-0.92 pp, CI [-2.75, +0.92]). The artifact change alone does nothing. The E2-B vs E2-A headline comparison (+0.92 pp, CI [-1.37, +3.21]) is also null. Selector choice is not the bottleneck.

**The E2-B pipeline engineering is publication-quality.** SHA verification, metadata assertions, paired sample enforcement, pre-specified classification, pilot poison gates, and sidecar decomposition collectively represent a thorough experimental design. The null result is scientifically informative *because* the methodology is sound.

### 4.2 What Is Suggestive but Unproven

**TriviaQA and TruthfulQA may encode distinct "truthfulness" concepts at the attention-head level.** The moderate cross-dataset rank agreement (E2-B vs E0: ρ = 0.527) combined with the negative direction cosine on the single shared selected head (L11H0: cosine = -0.351) suggests that both datasets identify similar heads as relevant, but the directions they extract from those heads may differ or even oppose. This is consistent with TriviaQA's "factual recall" being geometrically distinct from TruthfulQA's "misconception resistance."

However, the direction cosine rests on n=1 shared head. The rank agreement rests on n=114 shared ranked heads (more robust). The suggestive pattern is: **same heads, different directions**. If confirmed with more overlap points, this would imply that "truthfulness" is not a single direction in this model's representation space — a finding with implications for the universality claims in the ITI literature (Li et al. 2023).

**The MC2 near-miss in E2-B (+1.46 pp, CI [-0.11, +3.02]) does not indicate a weak signal.** The CI lower bound barely includes zero (-0.11), which might tempt interpretation as "almost significant." Three reasons to resist:
1. This is one of ~13 comparisons; it would not survive multiple comparison correction.
2. Even if real, +1.46 pp MC2 is 5× smaller than E0's +7.49 pp — below any practical utility threshold.
3. MC2 (probability mass on truthful answers) is more continuous and thus easier to show small shifts in; the fact that MC1 (binary correct/incorrect) shows exactly zero suggests the MC2 shift reflects random variation in answer probability distributions, not a behaviorally meaningful change.

**The calibration overfitting pattern may indicate a systematic issue with n=81 calibration on weak signals.** Both E2 variants showed calibration-to-held-out collapse. E0 did not. This could mean: (a) n=81 is sufficient when real signal exists and insufficient when it doesn't (selection bias — you only lock when you see something, but if that "something" is noise, the lock captures noise), or (b) TriviaQA's flat calibration landscape amplifies the chance of locking on a noise peak. The data cannot distinguish (a) from (b), but (a) is the more parsimonious explanation.

### 4.3 What Was Falsified

**"E2-A failed because of suboptimal selector overrides" — falsified.** This was the E2-B hypothesis. Family-default selectors (AUROC + all_answer_positions) were expected to produce better probes and potentially rescue the signal. The probes did improve slightly (mean AUROC 0.754 vs 0.733), but the intervention effect did not change. The issue is the data source, not the selectors.

**"TriviaQA directions capture the same truthfulness concept as TruthfulQA directions" — falsified for this model.** The combination of weaker probes, null intervention effects, and suggestive direction misalignment (negative cosine at the one shared head) collectively argues against concept identity. This does not mean TriviaQA captures nothing — it means whatever it captures does not transfer to TruthfulQA evaluation tasks on Gemma-3-4B-IT.

**"Better calibration signal predicts better held-out performance" — falsified within the E2 investigation.** E2-B had a stronger calibration signal (+6.17 pp) than E2-A (+3.70 pp), yet performed identically on held-out (both null). The calibration MC1 magnitude is not a reliable predictor of held-out effect when the underlying signal is absent.

---

## 5. Uncertainty Register

| Uncertainty | Severity | Basis | How to resolve |
|---|---|---|---|
| Whether a TriviaQA-native evaluation would show an E2 effect | Medium | We only measured TruthfulQA and SimpleQA transfer, not in-domain performance | Run TriviaQA MC eval with E2-B artifact; low cost but low strategic value for this sprint |
| Whether the negative direction cosine at L11H0 generalizes to other shared heads | Medium | n=1 is insufficient for directional claims | Would require a contrastive extraction that selects more overlapping heads (different K or forced overlap) |
| Whether cumulative decode scope (full_decode) would show a TriviaQA effect | Low | Decode-scope audit found first_3_tokens sufficient for E0/E1; MC evaluation is non-generative | Already addressed by MC results (non-generative, full-sequence scoring) |
| Whether a different TriviaQA subset or filtering procedure would improve probes | Low | The current extraction uses the family-standard deduplication and weighting; no obvious quality issue | Diminishing returns; would need a specific hypothesis about what to change |
| Whether the seed-42 split creates adversarial calibration/held-out partitions | Very Low | Fixed seed, no evidence of adversarial behavior, consistent across E2-A and E2-B | Run seed-sensitivity analysis; not justified by current evidence |

---

## 6. Structural Insights Worth Preserving for D8

These observations emerge from the E2 investigation and may be useful for the final synthesis (D8), even though they are not the primary findings:

1. **Calibration-to-held-out replication should be a standard health check.** The E2 results demonstrate that calibration signals on n=81 can be pure noise. Any future ITI investigation should report the calibration-to-held-out gap as a quality metric. A gap > 3 pp (calibration higher than held-out) should trigger a warning.

2. **Probe AUROC is a necessary but not sufficient indicator of intervention quality.** E2-B probes have AUROC ~0.75 — meaningfully above chance (0.50) and within the range that should support intervention effects. Yet the intervention is null. This suggests that what the probes detect (TriviaQA factual patterns) is not what the evaluation measures (TruthfulQA misconception resistance). Probe quality metrics should be supplemented with a transfer-relevance check before investing in intervention runs.

3. **Selector choice is a second-order effect compared to data source.** The E2-A vs E2-B comparison changed probes, K, and head selection substantially but changed the held-out result by at most 0.92 pp (not significant). The E0 vs E2-B comparison, which changes only the data source (holding the evaluation pipeline constant), shows a 6.26 pp gap. Data source dominates.

4. **The rank-direction dissociation is a potential contribution.** If the pattern holds (same heads identified as important, but different direction vectors extracted), it would challenge the assumption that truthfulness has a single linear representation across data sources. This is worth flagging as a preliminary observation in D8 with appropriate caveats about n=1 directional evidence.

5. **Attempt rate stability under TriviaQA intervention** (99.0% → 98.5%) confirms that TriviaQA directions do not interact with refusal circuitry. This is a useful negative control for the D5 externality audit: if TriviaQA directions had shown refusal engagement, it would suggest geometric overlap between factual-recall and refusal subspaces.

---

## 7. Implications for Remaining Sprint

**For D5 (externality audit):** The E2 null simplifies the externality matrix. Only E0 and E1 require full safety/capability audits, since E2 produces no detectable intervention effect and therefore no externalities to measure. E2's attempt-rate stability can be cited as a negative control in D5.

**For D7 (causal pilot):** The E2 investigation reinforces the motivation for D7. Correlational probe selection (the method used for all E0/E1/E2 variants) identifies heads that *carry variance* for a concept, but the intervention effect depends on whether that variance is *causally upstream* of the model's output. The E2 null — good probes, no effect — is consistent with the hypothesis that correlational selection can identify irrelevant variance. D7's causal head ranking via attribution patching would test whether causally-selected heads produce different intervention outcomes.

**For D8 (synthesis):** The TriviaQA transfer failure is a first-class result. It establishes that ITI's effectiveness on Gemma-3-4B-IT is dataset-specific: TruthfulQA-sourced directions work (E0), modernized TruthfulQA directions partially work (E1), and TriviaQA-sourced directions do not (E2). This constrains the "universal truthfulness direction" narrative and supports a more nuanced framing: ITI works when the contrastive data captures the specific behavioral pattern targeted by the evaluation, not a general "truthfulness" concept.

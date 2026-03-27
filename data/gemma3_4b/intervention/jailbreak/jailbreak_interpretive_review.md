# Jailbreak Intervention Interpretive Review

**Date:** 2026-03-20
**Model:** `google/gemma-3-4b-it`
**Related reports:** [intervention_findings.md](../../intervention_findings.md), [jailbreak_pipeline_audit.md](jailbreak_pipeline_audit.md), [falseqa_negative_control_audit.md](../falseqa/falseqa_negative_control_audit.md), [jailbreak_truncation_audit.md](../../../tests/gold_labels/jailbreak_truncation_audit.md)

> **2026-03-27 Status**: This review covers the original 256-token/7-alpha analysis, now superseded. The stochastic-generation confound critique (Section 3) remains valid. The endpoint effect was falsified by 5000-token reruns then recovered by CSV-v2 graded evaluation (+7.6pp [+3.6, +11.6]). Alpha=1.0 identity baseline added; see truncation audit Finding 29.

---

## Bottom Line

- **Endpoint effect is real.** The +6.2 pp [2.4, 10.0] CI excludes zero. H-neuron amplification increases jailbreak compliance.
- **Monotonic dose-response is NOT confirmed.** Spearman ρ=0.679, p=0.094 — non-significant at α=0.05. The curve shape is better described as threshold-then-saturation, not linear dose-response.
- **Flip analysis is invalid.** Stochastic generation (`do_sample=True, temp=0.7`) means per-item flips between adjacent alphas are dominated by sampling noise, not H-neuron effects. The cross-benchmark comparison to FalseQA's disjoint-subpopulation finding is not valid because FalseQA uses greedy decoding.
- **Negative control is the highest-priority gap.** Without a random-neuron baseline, the jailbreak effect cannot be confirmed as H-neuron-specific.
- **Template-conditioned analysis should be primary.** The aggregate number conflates 5 qualitatively different attack strategies with a 10× compliance range (T2 at 2-6% vs T1 at 34-50%).

---

## 1. Scope

This review examines **interpretive claims** made in the pipeline audit and intervention findings about the jailbreak experiment. It does not dispute the data, the CIs, or the pipeline integrity — the audit covers those thoroughly and correctly. Instead, it identifies cases where analytical patterns developed on deterministic-generation benchmarks (FaithEval, FalseQA) were transplanted to a stochastic-generation benchmark where they do not hold.

The pipeline audit (`jailbreak_pipeline_audit.md`) is the primary reference for:
- Data integrity verification (36/36 checks passed)
- Raw compliance tables and CIs
- Template and category breakdowns
- Heuristic vs GPT-4o comparison
- Safety considerations

This review is the primary reference for:
- Which interpretive claims survive scrutiny
- What cannot be concluded from the current data
- Where the audit's analytical framework breaks down

---

## 2. What the Audit Gets Right

The audit's core contributions are sound:

- **Data integrity** is thoroughly verified (line counts, field checks, ID consistency, recount validation).
- **Bootstrap CIs** are correctly computed and the endpoint CI [2.4, 10.0] properly excludes zero.
- **Template stratification** is the audit's most valuable analytical contribution. The 10× range across templates (T2 at 2-6% vs T1 at 34-50%) means the aggregate compliance number is misleading without conditioning on template. This finding should be promoted to the primary reporting frame.
- **Category breakdown** correctly identifies that strongly safety-trained categories (Harassment, Sexual content) are immune to H-neuron scaling, while moderately protected categories (Economic harm, Privacy) are most responsive.
- **The "no negative control" flag** is appropriately prominent and correctly identified as highest priority.

---

## 3. Flip Analysis Is Confounded by Stochastic Generation

### The problem

The audit's §3.4 reports per-item behavioral flips between alpha values and compares the overlap pattern to FalseQA's disjoint-subpopulation finding. This analysis assumes that when an item flips between SAFE and HARMFUL across alpha values, the flip reflects the H-neuron scaling intervention.

For FaithEval and FalseQA, this assumption holds: both use `do_sample=False` (greedy decoding), so the only source of variation between alpha runs is the neuron scaling itself. The same input deterministically produces the same output unless the intervention changes the model's behavior.

For jailbreak, this assumption **does not hold**: generation uses `do_sample=True, temperature=0.7`. Even at a fixed alpha, re-running the same item would produce a different response and potentially a different judge verdict. Per-item flips between adjacent alphas conflate:
1. Real H-neuron-driven behavioral changes
2. Sampling noise from stochastic generation
3. Judge boundary effects on stochastically varying responses

### Evidence from the data itself

The audit reports 76 swing items (15.2%) between α=1.0 and α=3.0, with a net effect of only +6 items (+1.2%). That means **92% of the churn is bidirectional** — items flipping in both directions at roughly equal rates. This is the signature of sampling noise, not a directional intervention effect. For comparison, FalseQA (greedy) shows a cleaner signal with net effects that are a larger fraction of total swings.

### The cross-benchmark comparison is invalid

The audit notes that the 5-item overlap between up-flips in ablation and amplification stages "replicates the disjoint-subpopulation finding from FalseQA." This comparison is invalid for two reasons:

1. **Different noise regimes.** FalseQA's near-zero overlap reflects genuine subpopulation structure because greedy decoding eliminates sampling noise. Jailbreak's near-zero overlap could equally reflect that both flip sets are dominated by random sampling noise, which would produce near-zero overlap by chance.

2. **Expected overlap under independence.** If up-flips are random with respect to items, the expected overlap between 41 up-flips (from 500) and 22 up-flips (from 500) is approximately `500 × (41/500) × (22/500) ≈ 1.8 items`. Observing 5 is not dramatically above this baseline — it is within the range expected from independence. The "near-zero" framing implies surprise, but there is nothing surprising about 5 items overlapping from two small random subsets.

### What can be salvaged

The **aggregate** endpoint effect (+6.2 pp) remains valid because it is computed from population-level rates that average over sampling noise. The per-item flip analysis is the part that breaks down — it requires deterministic generation to be interpretable.

---

## 4. Within-Item Correlation Is Real but Moderate

Despite the per-item noise, the audit's Jaccard analysis (if computed) and the template-level patterns show that H-neuron scaling does create real within-item structure — it is not purely noise.

### Evidence for real within-item effects

- **Template T1** shows consistent high compliance across alphas (34-50%), indicating that certain template × behavior combinations are reliably pushed toward harmful output by H-neuron amplification.
- **Template T2** shows consistent near-zero compliance (2-6%), indicating that this template's refusal mechanism is robust to H-neuron scaling.
- At high alpha values (α≥2.0), the compliance set stabilizes somewhat — fewer items are flipping in and out — suggesting the intervention has saturated for susceptible items.

### Evidence for substantial noise

- **Template T2** at n=100 per alpha: with compliance at 2-6%, the per-item signal is indistinguishable from sampling noise. Any item-level analysis within T2 is measuring temperature randomness, not H-neuron effects.
- The 92% bidirectional churn rate (§3 above) confirms that most per-item variation is not directional.

### Interpretation

H-neuron scaling creates a **real population-level shift** (some items become more likely to produce harmful output), but the **per-item reliability** is moderate at best. This is consistent with the intervention shifting a probability distribution rather than deterministically flipping a switch — which is exactly what you'd expect when generation is stochastic.

---

## 5. Dose-Response Monotonicity Is Not Established

### The statistical test

The audit reports Spearman ρ=0.679 with p=0.094 on 7 alpha points. At a conventional α=0.05 threshold, this is **non-significant**. The audit's framing — "not individually significant, but consistent with the dose-response pattern observed on other benchmarks" — transplants evidence from FaithEval (where monotonicity is perfect, ρ=1.0) to prop up a failed test on jailbreak. Each benchmark must be evaluated on its own data.

### What the curve actually shows

The compliance rates are: 20.2%, 23.8%, 25.2%, 28.6%, 27.8%, 28.2%, 26.4%.

This is better described as:
- **Phase 1 (α=0.0→1.5):** Compliance rises from 20.2% to 28.6% (+8.4 pp)
- **Phase 2 (α=1.5→3.0):** Compliance fluctuates in a 26-29% band with no trend

The curve shape is **threshold-then-saturation**, not linear dose-response. The linear slope of +2.14 pp/α overstates the marginal effect at high alpha — most of the effect is concentrated in the first half of the alpha range.

### What to claim

- **Valid:** "H-neuron amplification increases jailbreak compliance (endpoint CI excludes zero)."
- **Valid:** "The effect appears to saturate above α≈1.5."
- **Not valid:** "There is a dose-response relationship" (Spearman test fails).
- **Not valid:** "The slope of +2.14 pp/α characterizes the marginal effect" (most of the effect is below α=1.5).

---

## 6. Heuristic Phrase List Has a Design Flaw

The audit's §7 compares a 27-phrase refusal heuristic against GPT-4o verdicts and correctly concludes that heuristics are insufficient. However, the precision/recall numbers are partially artifactual due to a design flaw in the phrase list.

### The flaw

Several phrases in the heuristic list — "harmful", "dangerous", "illegal" — appear naturally in **both** refusals and harmful completions. A response that says "Here's how to do the harmful thing you asked about" would match "harmful" and be classified as a refusal, when it is actually a harmful compliance. Conversely, a refusal that avoids these specific words would be missed.

This means:
- **Precision is artificially depressed** because harmful responses containing these words are false-positived as refusals.
- **Recall is artificially depressed** because some refusals use different phrasing.

### Impact on conclusions

The audit's conclusion — that heuristics severely underestimate harmful compliance and should not substitute for GPT-4o judging — remains correct. The heuristic is inadequate for multiple reasons, not just the phrase overlap. But the specific precision (0.43–0.50) and recall (0.27–0.35) numbers should not be cited as calibrated estimates of heuristic performance, because the phrase list was not designed to avoid this confound.

---

## 7. Missing Measurements

### Judge test-retest reliability

FalseQA established a baseline for GPT-4o judge nondeterminism: re-judging the same responses at α=1.0 produced 0.4% verdict changes (3/687). No equivalent measurement exists for jailbreak. This matters because:

- Jailbreak responses are longer and more varied than FalseQA responses (mean ~1300 chars vs ~900 chars).
- The judge rubric is more complex (structured rubric with 6 few-shot examples vs simple ACCEPTED/REFUSED).
- Harm categories span a wider range of ambiguity.

Judge nondeterminism could be higher on jailbreak than FalseQA. Without a test-retest measurement, the judge's contribution to apparent alpha-to-alpha variation is unknown.

### Negative control

Confirmed as the highest-priority gap. Estimated cost from the audit: ~4h GPU + ~$19 API for quick mode (3 seeds × 3 alphas).

Without this control, the claim that the +6.2 pp effect is H-neuron-specific (rather than a generic perturbation artifact) is unsupported. The FaithEval and FalseQA negative controls cannot be transferred because the generation regime is different (stochastic vs greedy).

---

## 8. Claim Status Summary

| Claim | Status | Basis |
|-------|--------|-------|
| H-neuron amplification increases jailbreak compliance | **CONFIRMED** | Endpoint CI [2.4, 10.0] excludes zero |
| Effect is H-neuron-specific | **NOT CONFIRMED** | No negative control |
| Dose-response is monotonic | **NOT CONFIRMED** | Spearman p=0.094, non-significant |
| Effect plateaus above α≈1.5 | **DIRECTIONAL ONLY** | Descriptive pattern, not formally tested |
| Ablation and amplification affect disjoint subpopulations | **INVALID** | Stochastic generation confounds per-item flip analysis |
| Cross-benchmark flip pattern replicates FalseQA | **INVALID** | Greedy vs stochastic decoding makes comparison meaningless |
| Linear slope +2.14 pp/α characterizes the effect | **DIRECTIONAL ONLY** | Overstates marginal effect at high α due to saturation |
| Template T1 drives ~40% of harmful responses | **CONFIRMED** | Directly from data, template-conditioned rates are stable |
| Template T2 is immune to H-neuron scaling | **CONFIRMED** | 2-6% across all alphas |
| Categories with deep safety training are immune | **DIRECTIONAL ONLY** | n=50 per category per alpha; Wilson CIs are ±13-15 pp |
| Heuristic refusal detection is insufficient | **CONFIRMED** | Conclusion valid despite phrase-list design flaw |
| Three-benchmark convergence supports general compliance circuit | **DIRECTIONAL ONLY** | Jailbreak specificity unconfirmed; stochastic regime differs |

---

## 9. Recommended Downstream Corrections

### In `intervention_findings.md`

1. **§1.8 Spearman framing (line 208):** Reframe from implied support for plateau to explicit non-significance. The parenthetical currently reads as evidence; the p-value says otherwise.

2. **§1.8 after template table (line 220):** Add paragraph noting stochastic generation invalidates per-item flip analysis and cross-benchmark flip comparisons.

3. **Finding 5 caveats (line 264):** Add: Spearman non-significance, T2 Jaccard=0.000 (compliance is sampling noise), flip analysis invalidity, and judge reliability gap.

4. **§3 Missing controls (line 289):** Add bullet for missing judge test-retest reliability on jailbreak.

### In `jailbreak_pipeline_audit.md`

5. **§3.4 flip analysis (line 106):** Add interpretive caution that the disjoint-subpopulation comparison to FalseQA is invalid due to the decoding difference. The flip data is reported correctly; the interpretation that it "replicates" the FalseQA finding is not warranted.

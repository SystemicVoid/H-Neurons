# FalseQA Negative Control Audit: Gemma-3-4B H-Neuron Specificity

**Date:** 2026-03-19
**Model:** `google/gemma-3-4b-it`
**Related reports:** [intervention_findings.md](../../intervention_findings.md), [pipeline_report.md](../../pipeline/pipeline_report.md)

---

## Bottom Line

- **H-neuron specificity confirmed on FalseQA.** The H-neuron slope of **+1.55 pp/α** (3-point OLS, α ∈ {0.0, 1.0, 3.0}) falls well outside the empirical random-set slope interval of **[-0.40, +0.38] pp/α** from 3 unconstrained seeds. At α=3.0, H-neuron compliance (**74.4%**) exceeds the random interval **[71.8%, 72.9%]**.
- **This is the second benchmark confirming H-neuron specificity**, after FaithEval anti-compliance. The two use different evaluators (GPT-4o judge vs regex letter extraction) and task types (open-ended false-premise rejection vs MC counterfactual context).
- **The result is directionally conclusive but statistically lighter than the FaithEval control.** Quick mode used 3 seeds × 3 alphas vs FaithEval's 8 seeds × 7 alphas. The 3-seed "95% interval" is effectively [min, max]. A full sweep (8 seeds × 7 alphas, ~$16 API + ~8h GPU) is warranted for publication.

---

## 1. Design

**Scientific question:** Is the H-neuron compliance effect on FalseQA specific to H-neurons, or does scaling any 38 random neurons produce a similar trend?

**Pipeline (quick mode):**
1. **Generation** (`scripts/run_negative_control.py --benchmark falseqa --quick`): 3 unconstrained random seeds × 3 alphas (0.0, 1.0, 3.0) × 687 FalseQA items = 6,183 responses. Bare-question prompt, `do_sample=False`, `max_new_tokens=256`.
2. **GPT-4o judging** (`scripts/evaluate_intervention.py --benchmark falseqa`): Same `judge_falseqa()` prompt as the H-neuron experiment. Binary ACCEPTED/REFUSED.
3. **Analysis** (`scripts/run_negative_control.py --benchmark falseqa --quick --analysis_only`): `build_comparison_summary()` computes OLS slopes via `np.polyfit`, empirical percentile intervals via `np.quantile([2.5%, 97.5%])`.

**H-neuron baseline:** 3-point subset (α=0.0, 1.0, 3.0) extracted from the full 7-alpha experiment (`data/gemma3_4b/intervention/falseqa/experiment/results.json`). Both the 3-point OLS slope (1.55 pp/α) and the 7-point bootstrap slope (1.62 pp/α, CI [0.52, 2.74]) are reported below where applicable.

---

## 2. Data Integrity Verification

| Check | Result |
|-------|--------|
| JSONL line counts | 687 per file, all 9 files (3 seeds × 3 alphas) |
| Neurons per seed | 38 per seed |
| Seed–seed overlap | Zero (all 3 pairs) |
| Seed–H-neuron overlap | Zero (all 3 seeds) |
| Compliance recount from JSONL | Matches stored `results.json` for all 9 conditions |
| Parse failures | Zero across all conditions |
| JSONL schema | `{id, alpha, question, response, judge, compliance}` — identical to experiment |

---

## 3. Results

### 3.1 Compliance Rates

| α | H-neurons | Seed 0 | Seed 1 | Seed 2 | Random mean |
|---|-----------|--------|--------|--------|-------------|
| 0.0 | 69.6% (478) | 71.8% (493) | 72.1% (495) | 73.1% (502) | 72.3% |
| 1.0 | 71.9% (494) | 72.1% (495) | 72.3% (497) | 72.3% (497) | 72.2% |
| 3.0 | 74.4% (511) | 72.9% (501) | 72.2% (496) | 71.8% (493) | 72.3% |

### 3.2 Slopes

| Set | OLS slope (pp/α) | Spearman ρ |
|-----|-------------------|------------|
| H-neurons | +1.55 | +1.0 |
| Seed 0 | +0.40 | +1.0 |
| Seed 1 | +0.03 | +0.5 |
| Seed 2 | -0.42 | -1.0 |
| Random mean | +0.00 | — |

H-neuron slope of +1.55 pp/α is outside the random empirical interval [-0.40, +0.38] pp/α.

### 3.3 Endpoint Comparison at α=3.0

| Metric | H-neurons | Random 95% interval |
|--------|-----------|---------------------|
| Compliance rate | 74.4% | [71.8%, 72.9%] |
| Slope | +1.55 pp/α | [-0.40, +0.38] pp/α |

---

## 4. Critical Findings

### Finding 1: Two-sided specificity — ablation drops compliance, amplification raises it

At α=0.0 (ablation), H-neuron compliance is **69.6%** — 2.3pp below the α=1.0 unperturbed baseline of **71.9%**. Random ablation shows no such drop: random α=0.0 mean is **72.3%**, indistinguishable from random α=1.0 mean (**72.2%**).

This mirrors the FaithEval ablation finding (§1.5 of intervention_findings.md) where H-neuron ablation dropped compliance ~1.8pp while random ablation had no effect. FalseQA ablation is stronger (~2.3pp), possibly because the GPT-4o judge captures subtler behavioral shifts than regex letter matching.

### Finding 2: The swing population is H-neuron-specific

Per-question analysis between α=0.0 and α=3.0:

| Metric | H-neurons | Random (seed 0) |
|--------|-----------|-----------------|
| Total swing items (flip either direction) | 153 (22.3%) | 60 (8.7%) |
| Net swing (up − down) | +33 (+4.8pp) | +8 (+1.2pp) |
| Swing up (non-compliant → compliant) | 93 (13.5%) | 34 (4.9%) |
| Swing down (compliant → non-compliant) | 60 (8.7%) | 26 (3.8%) |

H-neuron scaling affects **2.6× more questions** than random scaling. The aggregate effect comes from a net surplus of upward flips, not from a uniform shift. Random noise produces some bidirectional churn (~8.7%), but it nearly cancels out.

### Finding 3: Ablation-recovery and amplification flip disjoint question sets

Decomposing the H-neuron swings into two stages:

| Stage | Up flips | Down flips | Net |
|-------|----------|------------|-----|
| α=0→1 (ablation recovery) | 65 | 49 | +16 |
| α=1→3 (amplification) | 67 | 50 | +17 |
| Overlap (same questions flip up in both stages) | **0** | — | — |

Zero overlap means the questions sensitive to H-neuron ablation are entirely different from those sensitive to amplification. This suggests H-neurons affect at least two distinct subpopulations of FalseQA items — one where removing the neurons suppresses the compliance decision, another where amplifying them promotes it.

### Finding 4: H-neuron amplification shortens responses

| α | H-neuron mean length (chars) | Random seed 0 mean length |
|---|------------------------------|---------------------------|
| 0.0 | 930 | 894 |
| 1.0 | 890 | 890 |
| 3.0 | 848 | 897 |

H-neuron amplification reduces mean response length by ~9% (930→848 chars), while random neurons leave it flat. Shorter responses on false-premise questions likely reflect more confident acceptance of the premise (less hedging, less elaboration). This is behavioral corroboration of the compliance shift: the model doesn't just flip its binary label more often — it changes *how* it responds.

### Finding 5: GPT-4o judge nondeterminism is measurable but small

At α=1.0, both H-neuron and random-seed-0 scaling should be no-ops (multiply by 1). Comparing judge labels:

- **Agreement: 684/687 (99.6%)**
- **Disagreements: 3 (0.4%)** — all on *identical* response text, confirming these are pure judge variance

This 0.4% judge nondeterminism contributes negligibly to the 4.8pp H-neuron signal. It does mean that any per-question analysis has a ~3-item noise floor.

---

## 5. Methodological Concerns

### 5.1 Three-seed empirical interval is inherently fragile

With n=3 seeds, `np.quantile([0.025, 0.975])` returns a linearly interpolated value near the min and max. The "95% interval" [-0.40, +0.38] is effectively the observed range of 3 slopes. Adding more seeds could widen this interval if the true null distribution has heavier tails.

For comparison, the FaithEval control used 8 seeds (5 unconstrained + 3 layer-matched), producing a tighter and more credible interval of [-0.106, +0.164] pp/α.

**Recommendation:** A full sweep with 5+ unconstrained seeds is needed before using this interval in a publication figure. The current result is strong enough for internal confidence but the null distribution is undersampled.

### 5.2 Three-point sampling conceals intermediate non-monotonicity

The 3-point H-neuron curve (69.6% → 71.9% → 74.4%) is perfectly monotonic (ρ=1.0). But the full 7-point curve from the experiment is not:

| α transition | Δ (pp) |
|-------------|--------|
| 0.0 → 0.5 | +1.74 |
| 0.5 → 1.0 | +0.59 |
| 1.0 → 1.5 | **-0.73** |
| 1.5 → 2.0 | +3.78 |
| 2.0 → 2.5 | **-1.02** |
| 2.5 → 3.0 | +0.44 |

The α=1.5 dip (-0.73pp) and α=2.5 dip (-1.02pp) are invisible in quick-mode sampling. These reversals are consistent with GPT-4o judge variance on borderline items (as noted in intervention_findings.md §Finding 2), not with a non-monotonic underlying dose-response.

The comparison_summary's ρ=1.0 for H-neurons is an artifact of the 3-point subset. Report the 7-point data alongside the 3-point comparison.

### 5.3 Slope rounding introduces minor precision loss

The `comparison_summary.json` rounds per-seed slopes to 2 decimal places (0.40, 0.03, -0.42) before computing the empirical interval. The exact OLS slopes are (0.3971, 0.0329, -0.4157). The rounding shifts the interval by ~0.02pp — negligible, but worth fixing in the pipeline for cleanliness.

### 5.4 No bootstrap CI on the 3-point H-neuron slope

The per-seed `results.json` files contain paired-bootstrap CIs (10,000 resamples), but `comparison_summary.json` only stores the OLS point estimate for the H-neuron slope (1.55). The full 7-point experiment has a bootstrap CI of [0.52, 2.74] pp/α. Ideally the comparison summary would also carry a bootstrap CI on the 3-point H-neuron slope for a formal separation test.

### 5.5 Judge-model error is outside the CIs

The Wilson and bootstrap intervals quantify sampling uncertainty over the 687 items, not systematic error in GPT-4o's labels. The 0.4% measured nondeterminism is a lower bound — systematic bias (e.g., the judge's threshold for "accepting" a false premise) could be higher but cancels in the paired comparison as long as the same judge version was used for both control and experiment.

---

## 6. Comparison with FaithEval Negative Control

| Aspect | FaithEval anti-compliance NC | FalseQA NC |
|--------|------------------------------|------------|
| Seeds | 5 unconstrained + 3 layer-matched | 3 unconstrained |
| Alphas | 7 (0.0 – 3.0 step 0.5) | 3 (0.0, 1.0, 3.0) |
| Total generations | 56,000 | 6,183 |
| H-neuron slope | 2.09 pp/α [1.38, 2.83] | 1.55 pp/α (OLS, no bootstrap CI) |
| Random mean slope | 0.02 pp/α | 0.00 pp/α |
| Random slope interval | [-0.106, +0.164] pp/α | [-0.40, +0.38] pp/α |
| H-neuron endpoint Δ | +6.3pp [4.2, 8.5] | +4.8pp [1.3, 8.3] (7-point) |
| Evaluator | Regex letter match | GPT-4o binary judge |
| Parse failures | Zero | Zero |
| Verdict | Clean dose-response proof | Specificity confirmed, weaker precision |

The FalseQA result is directionally identical to FaithEval but statistically lighter — fewer seeds, fewer alphas, wider intervals. The two results are complementary rather than redundant: different tasks, different evaluators, same conclusion.

---

## 7. Data Manifest

| Artifact | Path |
|----------|------|
| Control responses | `data/gemma3_4b/intervention/falseqa/control/seed_{0,1,2}_unconstrained/alpha_{0.0,1.0,3.0}.jsonl` |
| Neuron indices | `data/gemma3_4b/intervention/falseqa/control/seed_{0,1,2}_unconstrained/neuron_indices.json` |
| Per-seed results | `data/gemma3_4b/intervention/falseqa/control/seed_{0,1,2}_unconstrained/results.json` |
| Comparison summary | `data/gemma3_4b/intervention/falseqa/control/comparison_summary.json` |
| Plot | `data/gemma3_4b/intervention/falseqa/control/negative_control_comparison.png` |
| H-neuron experiment | `data/gemma3_4b/intervention/falseqa/experiment/results.json` (7 alphas, 687 items, GPT-4o judged) |
| Generation script | `scripts/run_negative_control.py` |
| Evaluation script | `scripts/evaluate_intervention.py` |

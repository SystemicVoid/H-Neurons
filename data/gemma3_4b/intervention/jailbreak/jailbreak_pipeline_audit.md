# Jailbreak Intervention Pipeline Audit: Gemma-3-4B H-Neuron Scaling

**Date:** 2026-03-20
**Model:** `google/gemma-3-4b-it`
**Related reports:** [intervention_findings.md](../../intervention_findings.md), [falseqa_negative_control_audit.md](../falseqa/falseqa_negative_control_audit.md)

---

## Bottom Line

- **H-neuron amplification modestly increases jailbreak compliance.** GPT-4o-judged harmful response rate rises from **20.2%** at α=0.0 to a peak of **28.6%** at α=1.5, yielding an endpoint effect of **+6.2 pp** [2.4, 10.0] (α=0→3) and a slope of **+2.14 pp/α** [0.91, 3.39]. The effect plateaus and slightly reverses above α=1.5.
- **Template heterogeneity dominates the signal.** Template 1 (prefix injection) has **+12pp** swing and ~46-50% compliance at high α, while Template 2 (role-play) stays near **2-6%** regardless of scaling. Aggregate rates obscure this structure.
- **Category sensitivity varies widely.** Economic harm (+18pp), Privacy (+14pp), and Disinformation (+12pp) are most responsive to H-neuron scaling. Harassment/Discrimination and Sexual/Adult content show near-zero effect.
- **No negative control exists for jailbreak.** Unlike FaithEval and FalseQA (both with random-neuron baselines), we cannot yet confirm that the jailbreak compliance increase is H-neuron-specific versus a generic perturbation artifact.

---

## 1. Design

**Scientific question:** Does amplifying 38 H-neurons increase the rate at which Gemma-3-4B complies with harmful requests?

**Data source:** JailbreakBench (Chao et al., NeurIPS 2024). 100 harmful behaviors across 10 categories, each with 5 jailbreak prompt templates = 500 items per alpha.

**Generation:** `scripts/run_intervention.py --benchmark jailbreak --jailbreak_source jailbreakbench`. `do_sample=True`, `temperature=0.7`, `max_new_tokens=512`. Unlike FaithEval and FalseQA (which use greedy decoding), jailbreak generation includes sampling noise.

**Evaluation:** `scripts/evaluate_intervention.py --benchmark jailbreak`. GPT-4o judge with a structured rubric (system prompt + 6 few-shot examples covering both HARMFUL and SAFE cases). Judges substance, not framing — an affirmative opener without harmful follow-through is scored SAFE.

**Scale:** 7 alphas × 500 items = 3,500 total responses judged.

---

## 2. Data Integrity Verification

| Check | Result |
|-------|--------|
| JSONL line counts | 500 per file, all 7 files (3,500 total) |
| Required fields | All 8 fields present (`id, alpha, goal, category, template_idx, response, judge, compliance`) in all records |
| Same IDs across alphas | 7/7 match |
| Compliance recount from JSONL | Matches `results.json` at all 7 alphas |
| Judge verdict distribution | Only HARMFUL and SAFE; zero UNKNOWN/ERROR/MISSING |
| Balanced design | 10 categories × 50 items each; 5 templates × 100 items each |
| Total checks | **36 passed, 0 failed** |

---

## 3. Results

### 3.1 Overall Compliance Rates

| α | Compliant | Total | Rate | 95% CI (Wilson) |
|---|-----------|-------|------|-----------------|
| 0.0 | 101 | 500 | 20.2% | [16.9, 23.9] |
| 0.5 | 119 | 500 | 23.8% | [20.3, 27.7] |
| 1.0 | 126 | 500 | 25.2% | [21.6, 29.2] |
| 1.5 | 143 | 500 | 28.6% | [24.8, 32.7] |
| 2.0 | 139 | 500 | 27.8% | [24.1, 31.9] |
| 2.5 | 141 | 500 | 28.2% | [24.4, 32.3] |
| 3.0 | 132 | 500 | 26.4% | [22.7, 30.4] |

**Endpoint effect** (α=0→3): **+6.2 pp**, paired-bootstrap 95% CI **[2.4, 10.0] pp**.
**Slope:** **+2.14 pp/α**, paired-bootstrap 95% CI **[0.91, 3.39] pp/α**.
**Spearman ρ:** 0.679 (p=0.094 on 7 points — not individually significant, but consistent with the dose-response pattern observed on other benchmarks).

The curve is not monotonic: compliance peaks at α=1.5 (28.6%) and slightly declines at α=2.0–3.0. This plateau-and-reversal contrasts with the steadily monotonic FaithEval anti-compliance curve (ρ=1.0).

### 3.2 Template-Level Breakdown

| Template | α=0.0 | α=0.5 | α=1.0 | α=1.5 | α=2.0 | α=2.5 | α=3.0 | Slope (pp/α) | Δ₀→₃ |
|----------|-------|-------|-------|-------|-------|-------|-------|--------------|-------|
| T0 | 30.0% (30/100) | 32.0% (32/100) | 35.0% (35/100) | 40.0% (40/100) | 37.0% (37/100) | 37.0% (37/100) | 38.0% (38/100) | +2.57 | +8.0pp |
| T1 | 34.0% (34/100) | 38.0% (38/100) | 46.0% (46/100) | 50.0% (50/100) | 45.0% (45/100) | 50.0% (50/100) | 46.0% (46/100) | +4.21 | +12.0pp |
| T2 | 6.0% (6/100) | 5.0% (5/100) | 6.0% (6/100) | 2.0% (2/100) | 4.0% (4/100) | 2.0% (2/100) | 2.0% (2/100) | -1.43 | -4.0pp |
| T3 | 9.0% (9/100) | 15.0% (15/100) | 10.0% (10/100) | 17.0% (17/100) | 19.0% (19/100) | 20.0% (20/100) | 20.0% (20/100) | +3.71 | +11.0pp |
| T4 | 22.0% (22/100) | 29.0% (29/100) | 29.0% (29/100) | 34.0% (34/100) | 34.0% (34/100) | 32.0% (32/100) | 26.0% (26/100) | +1.64 | +4.0pp |

Templates T1 and T3 drive the signal (+12pp, +11pp). Template T2 is nearly immune to H-neuron scaling (stays at 2-6%), suggesting this template's resistance is structural — probably a prompt formulation that the safety circuit handles independently of H-neuron activity.

### 3.3 Category-Level Breakdown

| Category | α=0.0 | α=1.0 | α=3.0 | Δ₀→₃ |
|----------|-------|-------|-------|-------|
| Disinformation | 20% (10/50) | 26% (13/50) | 32% (16/50) | +12pp |
| Economic harm | 16% (8/50) | 32% (16/50) | 34% (17/50) | +18pp |
| Expert advice | 24% (12/50) | 18% (9/50) | 32% (16/50) | +8pp |
| Fraud/Deception | 36% (18/50) | 40% (20/50) | 38% (19/50) | +2pp |
| Government decision-making | 20% (10/50) | 28% (14/50) | 28% (14/50) | +8pp |
| Harassment/Discrimination | 8% (4/50) | 6% (3/50) | 8% (4/50) | +0pp |
| Malware/Hacking | 42% (21/50) | 48% (24/50) | 40% (20/50) | -2pp |
| Physical harm | 6% (3/50) | 16% (8/50) | 8% (4/50) | +2pp |
| Privacy | 28% (14/50) | 34% (17/50) | 42% (21/50) | +14pp |
| Sexual/Adult content | 2% (1/50) | 4% (2/50) | 2% (1/50) | +0pp |

**Caveat:** n=50 per category per alpha. Wilson 95% CIs are ~±13-15pp at these sample sizes. Category-level trends are directional only; not suitable for standalone claims.

The categories with near-zero H-neuron effect (Harassment, Sexual content) are also those with the lowest baseline compliance, suggesting the model's safety training on these topics may operate through circuits that are independent of the 38 H-neurons.

### 3.4 Behavioral Flip Analysis

| Direction | α=1.0→3.0 (amplification) | α=1.0→0.0 (ablation) |
|-----------|---------------------------|----------------------|
| Up flips (safe→harmful) | 41 (8.2%) | 22 (4.4%) |
| Down flips (harmful→safe) | 35 (7.0%) | 47 (9.4%) |
| Net | +6 (+1.2%) | -25 (-5.0%) |
| Total swing items | 76 (15.2%) | 69 (13.8%) |

Overlap between items flipping up in both stages: **5** (near-zero). This replicates the disjoint-subpopulation finding from FalseQA (§1.7 of intervention_findings.md) — ablation and amplification affect different items.

The net amplification effect (+6 items) is small compared to total churn (76 swing items). The high churn likely reflects stochastic generation noise (do_sample=True, temp=0.7) rather than a strong directional signal at the per-item level.

### 3.5 Response Length Analysis

| α | Overall (chars) | Harmful (n) | Safe (n) |
|---|-----------------|-------------|----------|
| 0.0 | 1295±78 | 1264 (101) | 1303 (399) |
| 1.0 | 1311±89 | 1289 (126) | 1318 (374) |
| 3.0 | 1305±109 | 1280 (132) | 1314 (368) |

Response length is largely stable across alphas, unlike FalseQA where H-neuron amplification shortened responses by ~9%. Harmful responses are marginally shorter than safe ones (~30 chars difference), but the effect is small. This may reflect the max_new_tokens=512 ceiling constraining length variation.

---

## 4. Critical Findings

### Finding 1: H-neuron scaling increases jailbreak compliance with a plateau

The compliance curve rises +8.4pp from α=0.0 to α=1.5 (20.2% → 28.6%), then flattens. The endpoint Δ of +6.2pp is statistically significant (CI excludes zero), but the non-monotonic shape means the linear slope of +2.14pp/α overstates the marginal effect at high α. The effect is better described as: H-neuron amplification up to moderate scaling (α≤1.5) weakens jailbreak resistance, but further scaling does not increase harm.

### Finding 2: Template structure is a first-order confound

The aggregate compliance rate masks a 10× range across templates (T2 at 2-6% vs T1 at 34-50%). Template T1 (prefix injection) alone accounts for ~40% of all harmful responses at high α. Any claim about H-neuron effects on jailbreak must be conditioned on template — the aggregate number conflates template susceptibility with H-neuron scaling.

### Finding 3: Category sensitivity reveals a selectivity pattern

Categories with strong safety-training protection (Harassment, Sexual content: 2-8% baseline) show zero H-neuron effect. Categories with moderate baseline compliance (Economic harm, Privacy, Malware: 16-42%) show the largest scaling effects. This suggests H-neuron amplification weakens an intermediate layer of safety restraint rather than overriding deep safety conditioning.

---

## 5. Methodological Concerns

### 5.1 No negative control

Unlike FaithEval (8 seeds) and FalseQA (3 seeds), jailbreak has no random-neuron baseline. The +6.2pp effect could in principle result from scaling any 38 neurons, not just H-neurons. A negative control (~4h GPU + ~$19 API for quick mode) is the highest-priority missing piece.

### 5.2 Stochastic generation introduces per-item noise

FaithEval and FalseQA use `do_sample=False` (greedy decoding), making the only source of variance the neuron scaling itself. Jailbreak uses `do_sample=True, temperature=0.7`, so each response has sampling noise. This explains the high per-item churn (15.2% swing items, net +1.2%) and the non-monotonic curve — the aggregate trend is real but individual alpha-to-alpha transitions include substantial sampling variation.

### 5.3 GPT-4o judge calibration across harm categories

The judge uses a single rubric for all 10 harm categories. Its threshold for "actionable harmful content" may differ between domains (e.g., a vague phishing outline vs. a specific chemical synthesis procedure). Category-level compliance differences partially reflect judge sensitivity, not just model behavior.

### 5.4 The corrected results.json replaces a misleading stub

The original `results.json` (created during generation) showed 0/500 compliance at α=3.0 because `aggregate_results()` treats missing `compliance` fields as `False`. The current file contains the actual GPT-4o verdicts and should be the sole reference for jailbreak compliance data.

### 5.5 Template confound in the aggregate number

The 5 templates have qualitatively different attack strategies (direct instruction, prefix injection, role-play, etc.). Aggregating across templates treats them as exchangeable, which they are not. Future analysis should report template-conditioned effects as primary and the aggregate as a summary.

---

## 6. Safety-Specific Considerations

This experiment shows that amplifying 38 neurons increases harmful output from 20% to 29% on a curated jailbreak benchmark. While the effect is moderate, the content at high alpha includes specific harmful information: procedural guidance for illegal activities, social engineering templates, and manipulative content across multiple harm categories.

The harmful content was generated for research purposes (understanding model safety mechanisms via mechanistic interpretability). The raw responses are stored in JSONL files and should be handled with the same care as any safety-evaluation dataset — they contain content that could cause harm if used outside the research context.

No harmful content is reproduced in this audit. The JailbreakBench dataset itself is a published NeurIPS 2024 artifact with responsible disclosure practices.

---

## 7. Appendix: Heuristic vs GPT-4o Comparison

A refusal-phrase heuristic (27 phrases like "I cannot", "as an AI", "harmful", etc.) was compared against GPT-4o verdicts:

| α | GPT-4o compliant | Heuristic compliant | Agreement |
|---|-----------------|---------------------|-----------|
| 0.0 | 101 (20.2%) | 63 (12.6%) | 390/500 (78.0%) |
| 0.5 | 119 (23.8%) | 74 (14.8%) | 383/500 (76.6%) |
| 1.0 | 126 (25.2%) | 72 (14.4%) | 374/500 (74.8%) |
| 1.5 | 143 (28.6%) | 77 (15.4%) | 358/500 (71.6%) |
| 2.0 | 139 (27.8%) | 79 (15.8%) | 362/500 (72.4%) |
| 2.5 | 141 (28.2%) | 82 (16.4%) | 365/500 (73.0%) |
| 3.0 | 132 (26.4%) | 92 (18.4%) | 368/500 (73.6%) |

**Confusion matrix at α=0.0** (GPT-4o = ground truth):
- Precision: 0.429 (27 TP, 36 FP)
- Recall: 0.267 (27 TP, 74 FN)

**Confusion matrix at α=3.0:**
- Precision: 0.500 (46 TP, 46 FP)
- Recall: 0.348 (46 TP, 86 FN)

The heuristic severely underestimates harmful compliance (~12-18% vs ~20-29%). Most GPT-4o-HARMFUL responses contain refusal-like language somewhere but still provide substantive harmful content — exactly the scenario the judge rubric is designed to catch and the heuristic misses. The heuristic should not be used as a substitute for GPT-4o judging on jailbreak evaluations.

---

## 8. Data Manifest

| Artifact | Path |
|----------|------|
| Evaluated responses | `data/gemma3_4b/intervention/jailbreak/experiment/alpha_{0.0-3.0}.jsonl` |
| Aggregate results | `data/gemma3_4b/intervention/jailbreak/experiment/results.json` |
| Analysis script | `scripts/analyze_jailbreak.py` |
| Evaluation script | `scripts/evaluate_intervention.py` |
| Generation script | `scripts/run_intervention.py` |

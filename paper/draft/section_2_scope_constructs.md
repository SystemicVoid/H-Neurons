# 2. Scope, Constructs, and Reporting Standard

## 2.1 Paper Identity

This paper is a single-model comparative case study in Gemma-3-4B-IT (Google DeepMind, 2025). It tests whether strong predictive internal signals — features, neurons, or attention heads that discriminate well between behavioral categories on held-out data — reliably identify good targets for activation-level steering interventions.

Box A fixes the paper's claim boundary before we define the constructs that later sections measure.

We organize our evidence through four analytic stages — **measurement**, **localization**, **control**, and **externality** — each representing a distinct empirical gate in the path from "a feature predicts behavior $X$" to "intervening on that feature usefully changes behavior $X$." These stages are a methodological decomposition for auditing intervention claims, not a claim that each experiment belongs to exactly one stage.

> **Box A — What This Paper Is / Is Not**
>
> | This paper is | This paper is not |
> |---|---|
> | A comparative intervention case study in one model | A new steering method |
> | An empirical test of the readout→steering heuristic | A universal theorem about LLM steering |
> | A four-stage audit framework for intervention claims | An evaluator benchmark paper |
> | A documentation of when and how the heuristic fails | An argument that detection-based targets never work |

## 2.2 Construct Map

Each evaluation surface in this study measures a specific behavioral construct. We avoid the term "truthfulness benchmark" because the surfaces differ in what they test. Table 1 defines each construct precisely.

**Table 1 — Benchmark Construct Map**

| Benchmark | Construct Measured | Why Included | Evaluator | Primary Metric | Main Interpretive Caution |
|---|---|---|---|---|---|
| TruthfulQA MC1/MC2 | Answer selection under a constrained candidate set | Cleanest answer-selection surface; ITI achieves +6.3 pp MC1 | Deterministic MC scoring | MC1 accuracy | Does not measure open-ended generation; a model can select correct answers without being able to generate them |
| TriviaQA bridge | Short-form factual generation accuracy | Primary generation surface (test baseline 45.0% adjudicated, $n = 500$); reveals wrong-entity substitution failure mode | Adjudicated fact-match accuracy + deterministic floor | Adjudicated accuracy | Failure-mode coding is single-rater (no inter-rater reliability); E1 comparison is dev-only |
| FaithEval | Context-resistance under anti-compliance prompting | Compliance/anti-compliance diagnostic; H-neurons achieve +4.5 pp above no-op (slope +2.1 pp/$\alpha$) | Compliance scoring (counterfactual chosen = misleading answer chosen) | Compliance rate | Measures a credulity lever — acceptance of context even against explicit instruction — not standard truthfulness |
| FalseQA | Resistance to false presuppositions in questions | Validates H-neuron scaling on a second compliance surface ($n = 687$) | Compliance scoring | Compliance rate | Smaller sample; effects below ${\sim}4$ pp may not reach significance |
| JailbreakBench | Harmful compliance under adversarial prompting | Tests whether steering succeeds on a refusal-adjacent domain ($n = 500$) | Graded harmful severity (CSV-v2) | Strict harmfulness rate (graded) | Binary evaluation is underpowered (MDE ${\sim}6$ pp); truncation artifacts and evaluator construct mismatch are documented in §6 |
| BioASQ | Domain-specific factual QA (biomedical) | Scope test for H-neuron portability; endpoint accuracy is flat | Factual accuracy | Accuracy | Alias accuracy is flat despite substantial answer-style perturbation, so this is a portability limit on the endpoint metric rather than behavioral inactivity |
| SimpleQA | Hard OOD factual generation stress test | Extreme stress test with near-floor baseline (4.6%); ITI harms performance | Strict accuracy | Accuracy | Cannot distinguish "lacks knowledge" from "steering suppressed answer" at near-floor baselines |

## 2.3 Reporting Standard

Throughout the paper, we treat evaluation design as part of the scientific claim rather than as background bookkeeping. Headline results use full-generation scoring where relevant, pre-specified primary metrics, matched controls where available, and at least one non-target surface to reveal externalities. When a result falls short of one of those conditions, we still report it, but we describe the missing control or scope limit explicitly in the section where it appears.

**Table 2 — Minimum Detectable Effect by Benchmark**

| Benchmark | $n$ | Primary Metric | Observed H-neuron Effect (no-op to max) | Slope | MDE (paired, 80% power) | Status |
|---|---|---|---|---|---|---|
| FaithEval | 1,000 | Compliance rate | +4.5 pp [2.9, 6.1] | +2.09 pp/$\alpha$ [1.38, 2.83] | ${\sim}3$ pp | Well-powered |
| FalseQA | 687 | Compliance rate | +2.5 pp [$-0.6$, 5.5] | +1.62 pp/$\alpha$ [0.52, 2.74] | ${\sim}4$ pp | Slope significant; endpoint borderline |
| JailbreakBench | 500 | Strict harmfulness rate | +7.6 pp [2.6, 12.8] ($\alpha = 0 \rightarrow 3$ full sweep)[^fn-jailbreak-graded-baseline] | +2.30 pp/$\alpha$ [0.99, 3.58] | ${\sim}5$ pp | Graded well-powered; binary underpowered |
| BioASQ | 1,600 | Accuracy | $-0.06$ pp [$-1.5$, 1.4] | — | ${\sim}2$ pp | Well-powered flat endpoint |

Effect sizes in the "no-op to max" column report the change from the $\alpha = 1.0$ identity baseline (unperturbed model) to the maximum scaling factor. Slopes are from ordinary least squares fits across the full alpha grid with paired bootstrap 95% CIs (10,000 resamples).

[^fn-jailbreak-graded-baseline]: The JailbreakBench graded metric is reported as the full $\alpha = 0 \rightarrow 3$ sweep because the no-op-to-max value for this metric has not been recomputed from the CSV-v2 evaluation pipeline. Section 6 uses the slope ($+2.30$ pp/$\alpha$) as the primary effect size.

## 2.4 Interpretation Boundary

Three boundaries matter for reading the rest of the paper.

First, our strongest localization claim comes from the FaithEval neuron-versus-SAE comparison: matched detection quality did not translate into matched steering utility. The jailbreak selector evidence is supporting rather than co-equal. The matched pilot is useful because probe-ranked heads with AUROC 1.0 were inert while gradient-ranked heads were not, but the current full-500 panel supports only a benchmark-local selector-divergence claim because it combines branches with different ruler histories, uses a single random-control seed, and still carries explicit evaluator errors on the non-causal comparators.

Second, successful interventions remain surface-local. H-neuron scaling produced clear compliance effects on FaithEval and related surfaces, but on BioASQ it produced no robust net alias-accuracy effect despite substantial answer-style change and other behavioral perturbation. Likewise, ITI improved TruthfulQA answer selection while harming open-ended factual generation.

Third, the measurement story is about conclusion sensitivity, not evaluator superiority. In the jailbreak setting, truncation, scoring granularity, and evaluator choice all changed the apparent result. After the StrongREJECT GPT-4o rerun, holdout binary accuracy is tied with CSV-v3; the reason to use CSV-v3 in this paper is its richer outcome taxonomy and measurement granularity, not superior held-out binary accuracy.

We therefore do not claim that detectors fail as a class, that SAEs are poor steering mediators in general, that causal selectors are universally better than correlational ones, or that any of these findings automatically generalize beyond Gemma-3-4B-IT.

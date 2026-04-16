# 2. Scope, Constructs, and Reporting Standard

## 2.1 Paper Identity

This paper is a single-model comparative case study in Gemma-3-4B-IT (Google DeepMind, 2025). It tests whether strong predictive internal signals -- features, neurons, or attention heads that discriminate well between behavioral categories on held-out data -- reliably identify good targets for activation-level steering interventions.

Box A fixes the paper's claim boundary before we define the constructs that later sections measure.

We organize our evidence through four analytic stages -- **measurement**, **localization**, **control**, and **externality** -- each representing a distinct empirical gate in the path from "a feature predicts behavior $X$" to "intervening on that feature usefully changes behavior $X$." These stages are a methodological decomposition for auditing intervention claims, not a claim that each experiment belongs to exactly one stage.

> **Box A -- What This Paper Is / Is Not**
>
> | This paper is | This paper is not |
> |---|---|
> | A comparative intervention case study in one model | A new steering method |
> | An empirical test of the readout->steering heuristic | A universal theorem about LLM steering |
> | A four-stage audit framework for intervention claims | An evaluator benchmark paper |
> | A documentation of when and how the heuristic fails | An argument that detection-based targets never work |

## 2.2 Construct Map

Each evaluation surface in this study measures a specific behavioral construct. We avoid the term "truthfulness benchmark" because the surfaces differ in what they test. Table 1 defines each construct precisely.

**Table 1 -- Benchmark Construct Map**

| Benchmark | Construct Measured | Why Included | Evaluator | Primary Metric | Main Interpretive Caution |
|---|---|---|---|---|---|
| TruthfulQA MC1/MC2 | Answer selection under a constrained candidate set | Cleanest answer-selection surface; ITI achieves +6.3 pp MC1 | Deterministic MC scoring | MC1 accuracy | Does not measure open-ended generation; a model can select correct answers without being able to generate them |
| TriviaQA bridge | Short-form factual generation accuracy | Primary generation surface (test baseline 45.0% adjudicated, $n = 500$); reveals wrong-entity substitution failure mode | Adjudicated fact-match accuracy + deterministic floor | Adjudicated accuracy | Failure-mode coding is single-rater (no inter-rater reliability); E1 comparison is dev-only |
| FaithEval | Context-resistance under anti-compliance prompting | Compliance/anti-compliance diagnostic; H-neurons achieve +4.5 pp above no-op (slope +2.1 pp/$\alpha$) | Compliance scoring (counterfactual chosen = misleading answer chosen) | Compliance rate | Measures a credulity lever -- acceptance of context even against explicit instruction -- not standard truthfulness |
| FalseQA | Resistance to false presuppositions in questions | Validates H-neuron scaling on a second compliance surface ($n = 687$) | Compliance scoring | Compliance rate | Smaller sample; effects below ${\sim}4$ pp may not reach significance |
| JailbreakBench | Harmful compliance under adversarial prompting | Tests whether steering succeeds on a refusal-adjacent domain ($n = 500$) | Graded harmful severity (CSV-v2) | Strict harmfulness rate (graded) | Binary evaluation is underpowered (MDE ${\sim}6$ pp); truncation artifacts and evaluator construct mismatch are documented in §6 |
| BioASQ | Domain-specific factual QA (biomedical) | Scope test for H-neuron portability; endpoint accuracy is flat | Factual accuracy | Accuracy | Alias accuracy is flat despite substantial answer-style perturbation, so this is a portability limit on the endpoint metric rather than behavioral inactivity |
| SimpleQA | Hard OOD factual generation stress test | Extreme stress test with near-floor baseline (4.6%); ITI harms performance | Strict accuracy | Accuracy | Cannot distinguish "lacks knowledge" from "steering suppressed answer" at near-floor baselines |

## 2.3 Reporting Standard

Evaluation design is part of the claim, not background bookkeeping. Headline results use full-generation scoring where relevant, pre-specified primary metrics, matched controls where available, and at least one non-target surface to reveal externalities. When a result falls short of one of those conditions, we still report it, but we state the missing control or scope limit in the local section rather than letting it float as a headline claim. Benchmark-level power summaries and minimum detectable effects appear in Appendix Table B1.

## 2.4 Interpretation Boundary

Three boundaries matter for reading the rest of the paper. Our strongest localization claim comes from the FaithEval neuron-versus-SAE comparison in Section 4. The JailbreakBench selector material is included more narrowly: it supports the claim that selector choice can matter within that intervention family on that benchmark, but it does not settle the broader selector question. Later sections then ask how far successful interventions travel. H-neuron scaling is behaviorally active on BioASQ but shows no robust net alias-accuracy gain there, and the measurement section argues for CSV-v3 because it preserves richer outcome structure, not because it now exceeds StrongREJECT on holdout binary accuracy.

We therefore do not claim that detectors fail as a class, that SAEs are poor steering mediators in general, that causal selectors are universally better than correlational ones, or that any of these findings automatically generalize beyond Gemma-3-4B-IT.

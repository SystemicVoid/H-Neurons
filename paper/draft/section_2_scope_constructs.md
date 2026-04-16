# 2. Scope, Constructs, and Reporting Standard

## 2.1 Study Orientation

This paper is a single-model comparative case study in Gemma-3-4B-IT (Google DeepMind, 2025). It asks whether strong predictive internal signals -- features, neurons, or attention heads that discriminate well between behavioral categories on held-out data -- reliably identify useful targets for activation-level steering interventions. Three case studies anchor the answer: a matched FaithEval neuron-versus-SAE comparison, a TriviaQA bridge test of whether answer-selection gains transfer to nearby generation, and a jailbreak measurement audit asking whether the evaluation itself changes the verdict. We organize those results through four analytic stages -- **measurement**, **localization**, **control**, and **externality** -- as empirical gates in the path from "a feature predicts behavior $X$" to "intervening on that feature usefully changes behavior $X$." The paper does not claim a new steering method, a universal theorem about LLM steering, or a general failure of detector-based targets.

## 2.2 Construct Map

Each evaluation surface in this study measures a specific behavioral construct. We avoid the term "truthfulness benchmark" because the surfaces differ in what they test. Table 1 defines the surfaces in the order they matter to the paper's evidence spine.

**Table 1 -- Benchmark Construct Map**

| Benchmark | Construct Measured | Why Included | Evaluator | Primary Metric | Main Interpretive Caution |
|---|---|---|---|---|---|
| FaithEval | Context-resistance under anti-compliance prompting | Cleanest localization/control anchor; H-neurons achieve +4.5 pp above no-op (slope +2.1 pp/$\alpha$) | Compliance scoring (counterfactual chosen = misleading answer chosen) | Compliance rate | Measures a credulity lever -- acceptance of context even against explicit instruction -- not standard truthfulness |
| TruthfulQA MC1/MC2 | Answer selection under a constrained candidate set | Cleanest answer-selection surface; ITI achieves +6.3 pp MC1 | Deterministic MC scoring | MC1 accuracy | Does not measure open-ended generation; a model can select correct answers without being able to generate them |
| TriviaQA bridge | Short-form factual generation accuracy | Primary generation surface (test baseline 45.0% adjudicated, $n = 500$); reveals wrong-entity substitution failure mode | Adjudicated fact-match accuracy + deterministic floor | Adjudicated accuracy | Failure-mode coding is single-rater (no inter-rater reliability); E1 comparison is dev-only |
| JailbreakBench | Harmful compliance under adversarial prompting | Measurement-sensitive safety surface; truncation, scoring granularity, and evaluator choice change the apparent verdict ($n = 500$) | Graded harmful severity (CSV-v2) | Strict harmfulness rate (graded) | Binary evaluation is underpowered (MDE ${\sim}6$ pp); truncation artifacts and evaluator construct mismatch are documented in §6 |
| BioASQ | Domain-specific factual QA (biomedical) | Scope test for H-neuron portability; endpoint accuracy is flat | Factual accuracy | Accuracy | Alias accuracy is flat despite substantial answer-style perturbation, so this is a portability limit on the endpoint metric rather than behavioral inactivity |
| FalseQA | Resistance to false presuppositions in questions | Validates H-neuron scaling on a second compliance surface ($n = 687$) | Compliance scoring | Compliance rate | Smaller sample; effects below ${\sim}4$ pp may not reach significance |
| SimpleQA | Hard OOD factual generation stress test | Extreme stress test with near-floor baseline (4.6%); ITI harms performance | Strict accuracy | Accuracy | Cannot distinguish "lacks knowledge" from "steering suppressed answer" at near-floor baselines |

## 2.3 Reporting Standard

Evaluation design is part of the claim, not background bookkeeping. Headline results therefore use full-generation scoring where relevant, pre-specified primary metrics, matched controls where available, and at least one non-target surface to test transfer or externality. Results that fall short of those conditions are still reported, but their missing controls or scope limits are stated locally rather than elevated into headline claims. Within that hierarchy, the FaithEval neuron-versus-SAE comparison is the paper's cleanest localization/control anchor, the jailbreak selector material is supporting only, and benchmark-level power summaries and minimum detectable effects appear in Appendix Table B1.

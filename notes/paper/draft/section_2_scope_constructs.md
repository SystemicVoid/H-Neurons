# 2. Scope, Constructs, and Evaluation Contract

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
| TriviaQA bridge | Short-form factual generation accuracy | Primary generation surface (dev baseline 47.0% adjudicated); reveals confident wrong-entity substitution failure mode | Adjudicated fact-match accuracy + deterministic floor | Adjudicated accuracy | Dev-set results only ($n = 100$); test split not yet used for the main ITI configuration |
| FaithEval | Context-resistance under anti-compliance prompting | Compliance/anti-compliance diagnostic; H-neurons achieve +4.5 pp above no-op (slope +2.1 pp/$\alpha$) | Compliance scoring (counterfactual chosen = misleading answer chosen) | Compliance rate | Measures a credulity lever — acceptance of context even against explicit instruction — not standard truthfulness |
| FalseQA | Resistance to false presuppositions in questions | Validates H-neuron scaling on a second compliance surface ($n = 687$) | Compliance scoring | Compliance rate | Smaller sample; effects below ${\sim}4$ pp may not reach significance |
| JailbreakBench | Harmful compliance under adversarial prompting | Tests whether steering succeeds on a refusal-adjacent domain ($n = 500$) | Graded harmful severity (CSV-v2) | Strict harmfulness rate (graded) | Binary evaluation is underpowered (MDE ${\sim}6$ pp); truncation artifacts and evaluator construct mismatch are documented in §6 |
| BioASQ | Domain-specific factual QA (biomedical) | Scope test for H-neuron portability; result is null | Factual accuracy | Accuracy | H-neurons are task-local; null here establishes that working interventions do not generalize universally |
| SimpleQA | Hard OOD factual generation stress test | Extreme stress test with near-floor baseline (4.6%); ITI harms performance | Strict accuracy | Accuracy | Cannot distinguish "lacks knowledge" from "steering suppressed answer" at near-floor baselines |

## 2.3 Evaluation Contract

We define what counts as a headline-safe steering claim in this study. The six requirements below are the paper's full-pass standard for promoted claims and section-level conclusions. Some supporting results later in the paper intentionally fall short on one requirement; when that happens, we label them as qualified evidence and state the missing control or scope limit explicitly. These requirements were established in the project's measurement blueprint prior to the analysis reported here.

1. **Full-generation evaluation where relevant.** We do not use systematically truncated generations as headline evidence. For jailbreak evaluation, generation length was set to 5,000 tokens to avoid truncation artifacts that hide downstream content after refusal preambles. <!-- Source: notes/measurement-blueprint.md, Headline Rules -->

2. **Pre-specified primary metrics.** Each benchmark has a designated primary metric (Table 1) and diagnostic metrics. We do not select metrics post hoc to favor a particular conclusion. For jailbreak, graded severity is primary; binary compliance is diagnostic only due to insufficient power (MDE ${\sim}6$ pp at $n = 500$). <!-- Source: notes/measurement-blueprint.md, MDE table -->

3. **Matched negative control or explicit justification.** Neuron-level interventions require multi-seed random-neuron controls through the same pipeline. Direction-level interventions require at least one random-direction baseline. When a control is absent, we state why and mark the claim as caveated. <!-- Source: notes/measurement-blueprint.md, Negative Control Requirements -->

4. **Externality and retained-capability check.** Every steering method reports both the target-behavior effect and its impact on at least one surface where it was not tuned. We report harms as first-class outcomes, not footnotes. <!-- Source: notes/measurement-blueprint.md, Steering Externality Audit -->

5. **Per-example outputs for promoted claims.** Row-level predictions, not only summary statistics, are retained for every headline result. This enables post-hoc auditing and downstream analysis. <!-- Source: notes/measurement-blueprint.md, Run Manifest Requirements -->

6. **Explicit claim boundary.** Each result states precisely what was tested, what generality the claim carries, and what it does not carry.

**Table 2 — Minimum Detectable Effect by Benchmark**

| Benchmark | $n$ | Primary Metric | Observed H-neuron Effect (no-op to max) | Slope | MDE (paired, 80% power) | Status |
|---|---|---|---|---|---|---|
| FaithEval | 1,000 | Compliance rate | +4.5 pp [2.9, 6.1] | +2.09 pp/$\alpha$ [1.38, 2.83] | ${\sim}3$ pp | Well-powered |
| FalseQA | 687 | Compliance rate | +2.5 pp [$-0.6$, 5.5] | +1.62 pp/$\alpha$ [0.52, 2.74] | ${\sim}4$ pp | Slope significant; endpoint borderline |
| JailbreakBench | 500 | Strict harmfulness rate | +7.6 pp [2.6, 12.8] ($\alpha = 0 \rightarrow 3$ full sweep)[^fn-jailbreak-graded-baseline] | +2.30 pp/$\alpha$ [0.99, 3.58] | ${\sim}5$ pp | Graded well-powered; binary underpowered |
| BioASQ | 1,600 | Accuracy | $-0.06$ pp [$-1.5$, 1.4] | — | ${\sim}2$ pp | Well-powered null |

Effect sizes in the "no-op to max" column report the change from the $\alpha = 1.0$ identity baseline (unperturbed model) to the maximum scaling factor. Slopes are from ordinary least squares fits across the full alpha grid with paired bootstrap 95% CIs (10,000 resamples).

[^fn-jailbreak-graded-baseline]: The JailbreakBench graded metric is reported as the full $\alpha = 0 \rightarrow 3$ sweep because the no-op-to-max value for this metric has not been recomputed from the CSV-v2 evaluation pipeline. Section 6 uses the slope ($+2.30$ pp/$\alpha$) as the primary effect size.

## 2.4 Claim Ledger

We state upfront what this paper claims, what it suggests with caveats, and what it explicitly does not claim. This section is not a summary of results — it is a pre-commitment that later sections will honor.

### Primary claims

The following claims are supported by the strongest evidence in this study and can be stated without additional qualification:

- Strong predictive readouts were **not sufficient evidence** for useful steering targets. Matched or even perfect readout performance did not reliably predict intervention success. <!-- H-neuron vs SAE: AUROC 0.843 vs 0.848, divergent steering; Probe heads: AUROC 1.0, null intervention -->
- When interventions did work, their effects were often **surface-local** rather than generally transferable. <!-- ITI: +6.3 pp MC1 vs −7 pp to −9 pp on generation; H-neurons: +4.5 pp FaithEval above no-op, null on BioASQ -->
- Measurement choices — truncation depth and binary versus graded scoring — **materially altered** the inferred intervention conclusion on the same underlying data. <!-- Binary judge null vs graded significant on same data -->

### Qualified claims (valid with explicit caveats)

- Gradient-based causal selection identified different intervention targets than probe-based selection, with large behavioral divergence on jailbreak. *Caveat: no random-head negative control; benchmark-local evidence only.* <!-- Pilot + full-500 selector comparison; Jaccard 0.11 -->
- H-neuron specificity on jailbreak is supported at single-seed strength. *Caveat: slope difference +2.77 pp/$\alpha$ [1.17, 4.42], $p = 0.013$; seeds 1–2 pending.* <!-- Seed-0 control audit -->
- A structured evaluator calibrated for refuse-then-comply patterns achieved higher accuracy than alternatives on a dev set of jailbreak outputs from this intervention regime. *Caveat: holdout validation compressed the apparent advantage from 12.2 pp to 2.0 pp (not significant); 24 of 74 dev-set records overlap with calibration data; judge-model confound not yet removed.* <!-- 4-way evaluator comparison + holdout validation -->

### Claims we do not make

- We do not claim that detectors fail as a class. H-neurons are detector-selected targets that work on compliance surfaces.
- We do not claim that SAEs are poor steering mediators in general. We tested one configuration on one benchmark.
- We do not claim that causal selectors outperform correlational selectors in general. The missing random-head control prevents this.
- We do not claim that evaluator dependence is a novel discovery. We claim that evaluator choices changed the conclusion in this specific mechanistic intervention setting.
- We do not claim that these findings generalize beyond Gemma-3-4B-IT. The model name is in the title for this reason.
- We do not claim that CSV-v3 is a broadly validated evaluator, or that it clearly outperforms StrongREJECT outside this specific response regime.
- We do not claim that ITI improved truthful generation. ITI improved answer selection; it harmed generation on every tested surface.
- We do not claim that probe-selected components are non-causal. We show they do not steer behavior in our setting, not that they lack causal function.

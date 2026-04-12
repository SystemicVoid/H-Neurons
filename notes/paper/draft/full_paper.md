# Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT

*A staged audit of measurement, localization, control, and externality in mechanistic intervention research.*

Hugo Minh D. Nguyen

---

# Abstract

Predictive internal signals are often treated as natural targets for steering large language models, but the reliability of this heuristic has not been systematically tested. We study this question in Gemma-3-4B-IT by comparing multiple intervention families — neuron scaling, sparse autoencoder feature steering, inference-time intervention, and gradient-based causal head selection — under a shared evaluation contract across contextual faithfulness, answer selection, open-ended factual generation, and jailbreak settings. We find repeated dissociations between four stages that are routinely conflated: *measurement* (truncation and evaluator choices changed the apparent intervention conclusion), *localization* (matched or even perfect readout quality failed to predict steering utility — SAE features with AUROC 0.848 produced null steering alongside H-neurons with AUROC 0.843 that achieved +6.3 pp compliance gain; probe-ranked heads with AUROC 1.0 produced null intervention), *control* (successful interventions were surface-local — ITI improved answer selection by +6.3 pp MC1 but reduced open-ended factual accuracy by 7–9 pp on the TriviaQA bridge dev set), and *externality* (the dominant single generation failure mode was not refusal but confident substitution of semantically nearby wrong entities). We organize these results into a four-stage audit framework and argue that strong readouts are insufficient evidence for good steering targets: credible mechanistic intervention claims require stage-specific validation of measurement, localization, control, and externality.


---

# 1. Introduction

A predictive internal signal — a neuron, feature, or direction that reliably discriminates between behavioral categories on held-out data — is a tempting steering target. If a feature predicts whether a language model will produce a hallucination, a harmful response, or a factually incorrect answer, it is natural to expect that amplifying or suppressing that feature will steer the model toward the desired behavior. This heuristic underlies much recent work in activation steering: identify a strong readout, then intervene through it (Li et al., 2023; Gao et al., 2025; Arditi et al., 2024).

The heuristic sometimes works. Refusal-mediating directions identified by simple difference-in-means produce reliable refusal modulation (Arditi et al., 2024). Hallucination-associated neurons selected by classification performance can shift over-compliance behavior (Gao et al., 2025). But the heuristic also sometimes fails, and the failure modes have received less systematic attention than the successes. When a strong readout fails as a steering target, is the problem in the readout, the intervention operator, the evaluation, or the generalization?

This paper tests the readout-to-steering heuristic empirically in Gemma-3-4B-IT. We compare multiple intervention families — neuron scaling, sparse autoencoder feature steering, inference-time intervention via attention heads, and gradient-based causal head selection — under a shared evaluation contract across contextual faithfulness, answer selection, open-ended factual generation, and jailbreak settings. We find repeated dissociations between four stages that are routinely conflated:

1. **Measurement** — Can we trust the evaluation? Truncation artifacts, binary-versus-graded scoring, and evaluator construct mismatch each changed the scientific conclusion about whether an intervention worked (§6).

2. **Localization** — Does the readout identify causally relevant components? SAE features matched H-neurons on detection quality (AUROC 0.848 vs. 0.843) yet produced null steering on the same benchmark. Probe-ranked heads achieved perfect discrimination (AUROC 1.0) yet null intervention (§4).

3. **Control** — Does intervention produce the intended behavioral change? When it did, the effect was narrow: H-neurons improved compliance on FaithEval but not on BioASQ; ITI improved answer selection but not open-ended generation (§5).

4. **Externality** — Does the effect transfer without causing harm? The TriviaQA bridge benchmark revealed that ITI does not merely degrade generation — it redistributes probability mass toward semantically nearby but wrong entities, producing confident factual substitutions rather than refusals (§5.3).

These results suggest that strong readouts are insufficient evidence for good steering targets and that credible mechanistic intervention claims require stage-specific validation. We organize these observations into a four-stage audit framework — measurement, localization, control, externality — and distill five concrete recommendations for researchers evaluating intervention claims (§7).

Figure 1 previews this four-stage scaffold and situates the paper's three anchor case studies within it.

**Contributions.** (1) A cross-method empirical dissociation between readout quality and steering utility, with matched-detection comparisons that control for evaluation surface and intervention operator. (2) An externality analysis showing that steering gains on answer-selection benchmarks do not transfer to open-ended generation, with a specific failure-mode diagnosis (confident wrong-entity substitution). (3) A staged audit framework and checklist for evaluating the credibility of mechanistic intervention claims.


---

# 2. Scope, Constructs, and Evaluation Contract

## 2.1 Paper Identity

This paper is a single-model comparative case study in Gemma-3-4B-IT (Google DeepMind, 2025). It tests whether strong predictive internal signals — features, neurons, or attention heads that discriminate well between behavioral categories on held-out data — reliably identify good targets for activation-level steering interventions.

The paper is **not** a new steering method, a universal anti-steering argument, an evaluator benchmark, or an anti-SAE position paper. It is an empirical audit of a common heuristic in mechanistic intervention research: the assumption that readout quality predicts intervention utility.

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
| TriviaQA bridge | Short-form factual generation accuracy | Primary generation surface (dev baseline 47.0% adjudicated); reveals confident wrong-entity substitution failure mode | Adjudicated fact-match accuracy + deterministic floor | Adjudicated accuracy | Dev-set results only ($n = 100$); test split not yet used for promoted candidates |
| FaithEval | Context-resistance under anti-compliance prompting | Compliance/anti-compliance diagnostic; H-neurons achieve +6.3 pp | Compliance scoring (counterfactual chosen = misleading answer chosen) | Compliance rate | Measures a credulity lever — acceptance of context even against explicit instruction — not standard truthfulness |
| FalseQA | Resistance to false presuppositions in questions | Validates H-neuron scaling on a second compliance surface ($n = 687$) | Compliance scoring | Compliance rate | Smaller sample; effects below ${\sim}4$ pp may not reach significance |
| JailbreakBench | Harmful compliance under adversarial prompting | Tests whether steering succeeds on a refusal-adjacent domain ($n = 500$) | Graded harmful severity (CSV-v2) | Strict harmfulness rate (graded) | Binary evaluation is underpowered (MDE ${\sim}6$ pp); truncation artifacts and evaluator construct mismatch are documented in §6 |
| BioASQ | Domain-specific factual QA (biomedical) | Scope test for H-neuron portability; result is null | Factual accuracy | Accuracy | H-neurons are task-local; null here establishes that working interventions do not generalize universally |
| SimpleQA | Hard OOD factual generation stress test | Extreme stress test with near-floor baseline (4.6%); ITI harms performance | Strict accuracy | Accuracy | Cannot distinguish "lacks knowledge" from "steering suppressed answer" at near-floor baselines |

## 2.3 Evaluation Contract

We define what counts as a credible steering claim in this study. A steering claim passes the evaluation contract if it satisfies all six requirements below. These requirements were established in the project's measurement blueprint prior to the analysis reported here and apply uniformly across all intervention families.

1. **Full-generation evaluation where relevant.** We do not use systematically truncated generations as headline evidence. For jailbreak evaluation, generation length was set to 5,000 tokens to avoid truncation artifacts that hide downstream content after refusal preambles. <!-- Source: notes/measurement-blueprint.md, Headline Rules -->

2. **Pre-specified primary metrics.** Each benchmark has a designated primary metric (Table 1) and diagnostic metrics. We do not select metrics post hoc to favor a particular conclusion. For jailbreak, graded severity is primary; binary compliance is diagnostic only due to insufficient power (MDE ${\sim}6$ pp at $n = 500$). <!-- Source: notes/measurement-blueprint.md, MDE table -->

3. **Matched negative control or explicit justification.** Neuron-level interventions require multi-seed random-neuron controls through the same pipeline. Direction-level interventions require at least one random-direction baseline. When a control is absent, we state why and mark the claim as caveated. <!-- Source: notes/measurement-blueprint.md, Negative Control Requirements -->

4. **Externality and retained-capability check.** Every steering method reports both the target-behavior effect and its impact on at least one surface where it was not tuned. We report harms as first-class outcomes, not footnotes. <!-- Source: notes/measurement-blueprint.md, Steering Externality Audit -->

5. **Per-example outputs for promoted claims.** Row-level predictions, not only summary statistics, are retained for every headline result. This enables post-hoc auditing and downstream analysis. <!-- Source: notes/measurement-blueprint.md, Run Manifest Requirements -->

6. **Explicit claim boundary.** Each result states precisely what was tested, what generality the claim carries, and what it does not carry.

**Table 2 — Minimum Detectable Effect by Benchmark**

| Benchmark | $n$ | Primary Metric | Observed H-neuron Effect | MDE (paired, 80% power) | Status |
|---|---|---|---|---|---|
| FaithEval | 1,000 | Compliance rate | +6.3 pp [4.2, 8.5] | ${\sim}3$ pp | Well-powered |
| FalseQA | 687 | Compliance rate | +4.8 pp [1.3, 8.3] | ${\sim}4$ pp | Borderline |
| JailbreakBench | 500 | Strict harmfulness rate | +7.6 pp [3.6, 11.6] at $\alpha = 0 \rightarrow 3$ | ${\sim}5$ pp | Graded well-powered; binary underpowered |
| BioASQ | 1,600 | Accuracy | $-0.06$ pp [$-1.5$, 1.4] | ${\sim}2$ pp | Well-powered null |

For JailbreakBench, Table 2 reports the total change from $\alpha = 0$ to
$\alpha = 3$. Section 6 reports the same experiment as a per-$\alpha$ slope
(+$2.30$ pp/$\alpha$); the two statistics summarize the same dose-response at
different resolutions.

## 2.4 Claim Ledger

We state upfront what this paper claims, what it suggests with caveats, and what it explicitly does not claim. This section is not a summary of results — it is a pre-commitment that later sections will honor.

### Primary claims

The following claims are supported by the strongest evidence in this study and can be stated without additional qualification:

- Strong predictive readouts were **not sufficient evidence** for useful steering targets. Matched or even perfect readout performance did not reliably predict intervention success. <!-- H-neuron vs SAE: AUROC 0.843 vs 0.848, divergent steering; Probe heads: AUROC 1.0, null intervention -->
- When interventions did work, their effects were often **surface-local** rather than generally transferable. <!-- ITI: +6.3 pp MC1 vs −7 pp to −9 pp on generation; H-neurons: +6.3 pp FaithEval, null on BioASQ -->
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


---

# 3. Related Work and Novelty Boundary

This section identifies what the literature has already established, what remains open, and where this paper sits relative to prior work. We keep it lean: the paper earns its value through the comparative case study in Sections 4--6, not through literature breadth.

## 3.1 What Is Already Established

**Decodability does not imply causal use.** The concern that high probe accuracy may reflect information the model does not functionally rely on is foundational. Hewitt and Liang (2019) showed that simple control tasks can achieve high probing accuracy for uninformative reasons, undermining naive interpretation of probe performance. Elazar et al. (2021) moved toward causal analysis with amnesic probing, demonstrating that removing a decodable property often does not change task behavior. Kumar et al. (2022) showed that probing classifiers can be unreliable even for concept *detection*, let alone concept removal. Together, these results established that decodability is a necessary but insufficient condition for causal relevance — a point we take as background, not contribution.

**Detection and steering can diverge.** Several more recent studies move from generic probe skepticism into the practical question most relevant to this paper: does a strong internal signal tell you where to intervene? Within SAE feature selection, Arad et al. (2025) showed that features ranking high on input-side activation scores rarely coincide with features that steer well, and that output-score filtering yields roughly 2--3$\times$ better steering quality. AxBench (Wu et al., 2025) explicitly measured detection and steering as separate evaluation axes and found that methods that dominate on one axis do not reliably dominate on the other — with simple prompting remaining the most reliable steering baseline in that benchmark. Bhalla et al. (2024) articulated the predict/control discrepancy directly, reporting that interpretability-based interventions can be inconsistent and coherence-damaging, sometimes underperforming simple prompting. Li et al. (2023), in the original ITI paper, already showed a large gap between generative truthfulness gains and multiple-choice accuracy gains, and their ablations showed that the probe-weight direction was not the best steering direction.

**Answer-selection success can mislead about generative usefulness.** The literature supports a consistent finding that success on constrained evaluation surfaces does not guarantee success on open-ended generation. Li et al. (2023) demonstrated this within ITI itself. Pres et al. (2024) argued that many steering evaluations overestimate efficacy by relying on simplified, non-deployment-like settings and pushed evaluation toward open-ended generation. Opielka et al. (2026) showed that causally effective function vectors can be format-specific: representations that transfer well within multiple-choice settings may be geometrically distinct from those that generalize to open-ended generation.

**Evaluator and metric choices can change safety conclusions.** Judge dependence, scoring granularity, and artifact sensitivity are already well documented in safety and jailbreak evaluation. StrongREJECT (Souly et al., 2024) showed that binary attack-success scoring systematically overestimates jailbreak effectiveness compared to graded rubrics. Chen and Goldfarb-Tarrant (2025) demonstrated that LLM-based safety evaluators are sensitive to formatting artifacts unrelated to content harmfulness. Eiras et al. (2025) found that false-negative rates vary by up to 25$\times$ across judge models on the same evaluation corpus. We do not claim novelty for the observation that measurement choices matter in the abstract; we claim only that they materially alter the scientific conclusion in the specific representation-engineering setting we study.

**Better localization does not reliably predict better editing.** Hase et al. (2023) showed that methods achieving better causal localization of factual knowledge did not reliably translate into better knowledge editing, establishing a precedent for the localization-control dissociation in a different intervention class (weight editing rather than activation steering).

## 3.2 What Remains Open

Despite the convergent evidence above, several specific gaps remain:

**No matched cross-method comparison on a single behavioral surface.** The studies cited above each demonstrate detection-steering divergence within a single method family (SAE features in Arad et al.; probe directions in ITI; synthetic concepts in AxBench). No published work conducts a matched comparison across method families — e.g., magnitude-ranked neurons, SAE features, probe-ranked attention heads, and gradient-selected heads — on the same model and evaluation surface, controlling for detection quality. This paper provides that comparison.

**The exact failure mode under truthfulness steering is uncharacterized.** The literature supports the general concern that steering success does not transfer across evaluation surfaces, but the specific mechanisms of failure under truthfulness interventions have received little attention. In particular, whether an intervention merely degrades generation quality or actively redirects probability mass toward wrong entities is an empirical question the literature leaves open. This paper characterizes one such failure mode.

**Measurement reversals have not been documented inside a mechanistic intervention setting.** Evaluator fragility has been established for safety benchmarking, but whether the same fragility reverses the conclusion about whether a *specific representation-engineering intervention worked* has not been directly tested. This paper shows that it can.

**No integrated audit scaffold.** The individual concerns — measurement validity, localization fidelity, control efficacy, externality assessment — have each been raised in isolation. No prior work organizes them into an explicit staged framework for auditing intervention claims. This paper proposes such a framework and uses it to structure the case study.

## 3.3 How This Paper Differs

It is important to be precise about what this paper does and does not claim. We do not claim to be the first to observe that predictive quality and intervention utility can diverge — that concern is well motivated by the work reviewed above. The correct framing is: **prior work motivates the concern; this paper tests it in a matched cross-method case study and organizes the resulting failures into a four-stage audit framework.**

The paper sits at the intersection of four threads that have not previously been brought together in a single study: mediator choice (which internal component to intervene on), selector choice (how to pick that component), surface validity (whether gains survive transfer to different evaluation formats), and measurement discipline (whether the evaluation itself is trustworthy). Each thread has strong prior work; the combination is what allows us to diagnose where and why the readout-to-steering heuristic breaks.

**Positive counterexamples are part of our claim, not an exception to it.** Some detector-selected interventions work remarkably well. Arditi et al. (2024) showed that refusal in chat models is mediated by a single direction that reliably modulates refusal behavior. Gao et al. (2025) showed that an extremely sparse neuron subset ($<$0.1\% of total neurons) can both detect hallucination and causally modulate over-compliance. Dunefsky and Cohan (2025) showed that one-shot optimized steering vectors mediate safety-relevant behaviors across inputs. These successes do not undermine the concern — they sharpen it. The question is not whether readouts ever identify useful targets, but whether readout quality alone is sufficient evidence that a target will be useful. The case study in the following sections shows that it is not.


---

# 4. Case Study I: From Localization to Control

This section presents the paper's core empirical contribution: across two intervention families and two evaluation surfaces in Gemma-3-4B-IT, strong or even perfect predictive readouts did not reliably identify useful steering targets. The failure was not the absence of signal, but the unreliability of signal quality as a target-selection heuristic.

We organize the evidence from cleanest to most informative. Section 4.1 establishes that the readouts under study are genuine held-out signals, not strawmen. Section 4.2 presents the sharpest single experiment: matched detection quality between magnitude-ranked neurons and SAE features, with divergent steering outcomes. Section 4.3 introduces the most reviewer-resistant result: perfect probe-head discrimination that yields zero intervention effect, contrasted with a gradient-based selector that achieves significant harm reduction. Section 4.4 positions the gradient-based result as supporting evidence with explicit caveats.

Figure 2 summarizes the matched-readout dissociations in this section and places the FaithEval and jailbreak comparisons on a shared visual footing.

## 4.1 The Readouts Are Real

The intervention targets examined below were selected through held-out predictive readouts that meet or exceed conventional standards. This matters because the subsequent null steering results are only informative if the underlying detection signal is genuine.

**Magnitude-ranked neurons.** A CETT probe (Gao et al., 2025) trained on FaithEval context-grounding activations identified 38 neurons (out of 348,160 total feed-forward neurons across 34 layers) with positive logistic regression weight at regularization strength $C = 1.0$. On a disjoint held-out split, this probe achieved AUROC $0.843$ (accuracy $76.5\%$, $n_{\text{test}} = 780$).[^fn-classifier-structure] The 38 neurons span 23 of 34 layers, with concentration in early layers (18 neurons, 47.4%, in layers 0--10).[^fn-classifier-structure]

**SAE features.** An L1 logistic regression probe trained on Gemma Scope 2 sparse autoencoder activations (16k-width SAEs across 10 layers) selected 266 positive-weight features at $C = 0.005$, achieving AUROC $0.848$ (accuracy $77.2\%$, $n_{\text{test}} = 782$).[^fn-classifier-sae] This marginally exceeds the CETT probe but falls within its bootstrap confidence interval $[0.815, 0.870]$ and uses $7\times$ more features.[^fn-sae-audit]

**Probe-ranked attention heads.** For the jailbreak intervention setting, a per-head AUROC probe trained on harmful/benign activation contrasts from JailbreakBench produced a top-20 head set where the two highest-ranked heads each achieved AUROC $1.0$ (balanced accuracy $1.0$ and $0.95$, respectively), and all 20 selected heads scored between $0.92$ and $1.0$.[^fn-probe-metadata]

**Interpretation caveats.** While the aggregate detection signal is robust, its interpretation at the individual-component level is less clear. The highest-weight neuron in the CETT probe (L20:N4288, weight $12.169$, contributing $30.7\%$ of top-10 weight mass) failed all six causal importance tests and enters the model only at a higher regularization threshold ($C = 3.0$, where a 219-neuron detector achieves $80.5\%$ accuracy with N4288 at rank 5).[^fn-classifier-structure] A verbosity confound analysis found that response length dominates truthfulness signal by a factor of $3.7$--$16\times$ in full-response readouts.[^fn-strat-assessment] These observations do not undermine the held-out AUROC values — which measure genuine discrimination — but they caution against interpreting probe weights as a guide to mechanistic importance. Appendix A summarizes the detector-interpretation audits that motivate this caution.

[^fn-classifier-structure]: `data/gemma3_4b/pipeline/classifier_structure_summary.json`; classifier: `models/gemma3_4b_classifier_disjoint.pkl`; test AUROC $= 0.8429$.
[^fn-classifier-sae]: `data/gemma3_4b/pipeline/classifier_sae_summary.json`; classifier: `models/sae_detector.pkl`; test AUROC $= 0.8477$, $n_{\text{positive}} = 266$ features across 10 layers.
[^fn-sae-audit]: `data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md`, Finding 3.
[^fn-probe-metadata]: `data/contrastive/refusal/iti_refusal_probe_d7/extraction_metadata.json`; top-2 heads: L10:H6 (AUROC $1.0$, balanced accuracy $1.0$) and L2:H6 (AUROC $1.0$, balanced accuracy $0.95$).
[^fn-strat-assessment]: `data/gemma3_4b/intervention/verbosity_confound/verbosity_confound_audit.md`; summarized in Appendix A alongside the N4288 audit.

## 4.2 Magnitude-Ranked Neurons vs. SAE Features on FaithEval

This comparison is the paper's cleanest single experiment. Both methods achieve matched detection quality on the same benchmark, same model, and same behavioral construct (context-grounding compliance on FaithEval, $n = 1{,}000$). The only difference is the representational basis for intervention: neurons in the feed-forward network's down-projection input space versus features in a sparse autoencoder's latent space.

### Setup

**Magnitude-ranked neuron intervention.** The 38 CETT-selected neurons were scaled multiplicatively: at each forward pass, the activation of each selected neuron was multiplied by $\alpha \in \{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0\}$, where $\alpha = 1.0$ is the identity (no-op). This intervention operates directly in the model's feed-forward computation, requiring no auxiliary encoding or decoding step.

**SAE feature intervention.** The 266 classifier-selected SAE features were scaled through an encode-modify-decode cycle: at each token position, activations were encoded through the Gemma Scope 2 SAE, target features were multiplied by $\alpha$, and the modified representation was decoded back to activation space. At $\alpha = 1.0$, the hook returned the original activation unchanged (early return, no encode/decode applied). This design follows the methodology described in Arad et al. (2025) for SAE-based behavioral steering.

**Evaluation.** Compliance was scored deterministically via regex-based letter extraction on FaithEval's multiple-choice format ($n = 1{,}000$ items). The primary metric was compliance rate (proportion of items where the model selected the misleading answer consistent with the provided context, against explicit instructions).

### Results

**Table 3 — FaithEval Compliance by Intervention Method and Scaling Factor**

| $\alpha$ | Neurons (38) | SAE H-features (266) | SAE random (mean $\pm$ SD, 3 seeds) |
|---|---|---|---|
| 0.0 | 64.2% [61.2, 67.1] | 72.3% [69.4, 75.0] | 74.9% $\pm$ 0.4 |
| 0.5 | 65.4% [62.4, 68.3] | 74.7% [71.9, 77.3] | 74.8% $\pm$ 0.4 |
| 1.0 | 66.0% [63.0, 68.9] | **66.0%** [63.0, 68.9] | **66.0%** $\pm$ 0.0 |
| 1.5 | 67.0% [64.0, 69.8] | 75.0% [72.2, 77.6] | 75.0% $\pm$ 0.2 |
| 2.0 | 68.2% [65.2, 71.0] | 75.1% [72.3, 77.7] | 74.9% $\pm$ 0.1 |
| 2.5 | 69.5% [66.6, 72.3] | 74.9% [72.1, 77.5] | 74.9% $\pm$ 0.1 |
| 3.0 | 70.5% [67.6, 73.2] | 69.9% [67.0, 72.7] | 74.6% $\pm$ 0.5 |

Wilson 95% CIs shown for neurons and SAE H-features ($n = 1{,}000$). $\alpha = 1.0$ is the no-op baseline for both intervention modes.[^fn-faitheval-results][^fn-sae-comparison]

**Neuron steering showed a significant, monotonic dose-response.** The magnitude-ranked neuron compliance slope was $+2.09$ pp/$\alpha$ $[1.38, 2.83]$ (paired bootstrap 95% CI, 10,000 resamples). The Spearman rank correlation between $\alpha$ and compliance rate was $\rho = 1.0$ (perfectly monotonic). At $\alpha = 3.0$, compliance increased by $+6.3$ pp $[4.2, 8.5]$ relative to $\alpha = 0.0$.[^fn-faitheval-results]

**SAE feature steering was indistinguishable from zero.** The H-feature compliance slope was $+0.16$ pp/$\alpha$ $[-0.51, 0.84]$ — the confidence interval includes zero. The Spearman correlation was $\rho = 0.18$ (no monotonic trend). Random SAE features (266 features drawn from zero-weight classifier positions, 3 seeds) produced a mean slope of $+0.59$ pp/$\alpha$ $[0.54, 0.64]$.[^fn-sae-comparison]

The distinction between the two SAE null summaries matters for cross-document consistency. The main full-sweep result reported in this paper is the $+0.16$ pp/$\alpha$ null above; the $+0.12$ pp/$\alpha$ figure reported later refers to the delta-only control that removes reconstruction error as an explanation for the null.

**H-features performed worse than random features at $\alpha = 3.0$.** At the highest scaling factor, classifier-selected SAE features yielded $69.9\%$ compliance versus $74.6\%$ for random SAE features — a $-4.7$ pp difference in the wrong direction. If the 266 selected features encoded the compliance mechanism, amplifying them should have produced larger gains than amplifying random features. The reversal is consistent with over-amplification of compliance-correlated but causally irrelevant features disrupting the decode reconstruction.[^fn-sae-audit-finding2]

### The SAE Encode/Decode Cycle Is Not the Explanation

A natural objection is that the SAE's lossy reconstruction (relative L2 error $= 0.1557$) destroyed the steering signal. We tested this directly with a delta-only architecture that cancels reconstruction error exactly: $\mathbf{h}_t + \text{decode}(\mathbf{f}_{\text{modified}}) - \text{decode}(\mathbf{f}_{\text{original}})$, where only the targeted feature modifications propagate to the residual stream.[^fn-sae-delta]

The delta-only H-feature slope was $+0.12$ pp/$\alpha$, and the delta-only random slope was $-0.09$ pp/$\alpha$ — both indistinguishable from zero. The neuron baseline on the same three-alpha subset was $+2.12$ pp/$\alpha$. The delta-only architecture also eliminated the ${\sim}8$--$9$ pp compliance shift caused by lossy reconstruction (all non-identity alphas had produced elevated compliance regardless of feature selection under the full-replacement architecture) and reduced parse failures from $1.4$--$2.3\%$ to zero.[^fn-sae-delta]

This rules out reconstruction error as the primary confounder. The SAE steering null reflects genuine feature-space misalignment: features that correlate with hallucination in static activation readouts do not causally control compliance when manipulated through the SAE's encode-modify-decode pathway.

### Neuron Specificity Is Confirmed by Negative Controls

To establish that the neuron dose-response reflects the specific identity of the 38 selected neurons rather than a generic perturbation effect, we ran two families of negative controls: 5 unconstrained random neuron sets (38 neurons each, drawn uniformly from all 348,160 feed-forward neurons) and 3 layer-matched random neuron sets (38 neurons with the same layer distribution as the CETT selection, drawn from non-selected neurons within those layers). In total, 8 independent random seeds were evaluated across the same alpha sweep.[^fn-faitheval-control]

All 8 random seeds produced null compliance slopes. The mean unconstrained-random slope was $+0.02$ pp/$\alpha$ $[-0.11, 0.16]$ (95% empirical percentile interval across 5 seeds), and the mean layer-matched slope was $+0.17$ pp/$\alpha$ $[0.15, 0.21]$. No random seed produced a monotonic dose-response. At $\alpha = 3.0$, the H-neuron compliance rate ($70.5\%$) exceeded the 95th percentile of the random distribution ($65.8$--$66.5\%$).[^fn-faitheval-control]

The H-neuron slope of $+2.09$ pp/$\alpha$ exceeds the maximum observed random slope ($+0.21$ pp/$\alpha$, layer-matched seed 0) by an order of magnitude. The intervention effect is neuron-specific, not a property of generic 38-neuron perturbations at this scale.

### Summary

Detection quality was matched: AUROC $0.843$ (neurons) versus $0.848$ (SAE features). Steering diverged completely: $+2.09$ pp/$\alpha$ $[1.38, 2.83]$ versus $+0.16$ pp/$\alpha$ $[-0.51, 0.84]$. The failure was not attributable to reconstruction error (delta-only architecture confirmed the null) or to generic perturbation effects (8 random-neuron seeds confirmed specificity). Matched readout quality did not predict matched intervention utility.

[^fn-faitheval-results]: `data/gemma3_4b/intervention/faitheval/experiment/results.json`; slope and delta CIs from paired bootstrap (10,000 resamples, seed 42).
[^fn-sae-comparison]: `data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json`.
[^fn-sae-audit-finding2]: `data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md`, Finding 2.
[^fn-sae-delta]: `data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md`, Confound 1; data in `data/gemma3_4b/intervention/faitheval_sae_delta/`.
[^fn-faitheval-control]: `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`; 5 unconstrained seeds (slopes: $+0.17$, $-0.00$, $-0.07$, $-0.11$, $+0.11$ pp/$\alpha$) and 3 layer-matched seeds (slopes: $+0.21$, $+0.16$, $+0.15$ pp/$\alpha$).

## 4.3 Probe-Ranked Heads vs. Gradient-Ranked Heads on Jailbreak

The FaithEval comparison (Section 4.2) showed that matched detection quality did not guarantee matched steering. This section presents a stronger version of the same dissociation: *perfect* detection quality producing zero intervention effect.

### Setup

Both interventions operate at the attention-head level using Inference-Time Intervention (ITI; Li et al., 2023): a learned direction vector is added to the residual stream contribution of each selected head during decoding. The two methods differ only in how heads are ranked for selection.

**Probe-ranked selection.** Each of the model's 272 attention heads (34 layers $\times$ 8 heads) was scored by its held-out AUROC on a harmful/benign activation contrast derived from JailbreakBench prompts. The top-20 heads were selected for intervention. As reported in Section 4.1, the top two heads achieved AUROC $= 1.0$, and all 20 selected heads scored between $0.92$ and $1.0$.[^fn-probe-metadata]

**Gradient-ranked selection.** Each head was scored by the mean absolute gradient of the model's refusal probability with respect to a rank-1 approximation of the head's output, computed on the same harmful/benign contrast set. This is a causal criterion: it measures how much perturbing each head's output changes the model's tendency to refuse, rather than how well the head's activations predict the behavioral label.[^fn-causal-metadata]

**Evaluation.** Both selectors were tested at matched intervention strength ($k = 20$ heads). The probe family was evaluated on the pilot ($n = 100$) across $\alpha \in \{0.0, 1.0, 2.0, 4.0, 8.0\}$. The gradient-ranked family was evaluated on the full-500 confirmatory run ($n = 500$) at its locked alpha ($\alpha = 4.0$) against a shared no-op baseline. The primary metric was strict harmfulness rate: the proportion of responses judged as unambiguously harmful under CSV-v2 graded severity scoring (Bhalla et al., 2024).[^fn-d7-pilot][^fn-d7-full500]

### Results

**Probe-ranked heads: null at every alpha.** The best probe intervention produced a $-2$ pp change in strict harmfulness rate $[-10, +6]$ (paired bootstrap 95% CI, $n = 100$). At $\alpha = 8.0$, the probe intervention *increased* harmful compliance by $+6$ pp (from 30 to 36 strictly harmful responses, with $+12$ pp $[+3, +21]$ on binary judge compliance), accompanied by 82\% of responses hitting the 5,000-token generation cap — consistent with model degeneration rather than genuine behavioral change.[^fn-d7-pilot]

**Gradient-ranked heads: significant harm reduction.** On the full-500 confirmatory run against a shared no-op baseline (strict harmfulness rate: $23.4\%$, 117/500), the gradient-ranked intervention at $\alpha = 4.0$ reduced strict harmfulness to $14.4\%$ (72/500): $-9.0$ pp $[-12.2, -5.8]$. The binary judge moved in the same direction: $-10.6$ pp $[-14.0, -7.2]$. Severity-sensitive CSV-v2 component scores (commitment $C$: $-0.40$ $[-0.47, -0.33]$; specificity $S$: $-0.46$ $[-0.55, -0.37]$) confirmed that the reduction reflected genuine deescalation, not merely a shift in judge threshold.[^fn-d7-full500]

**The two selectors identified fundamentally different heads.** Jaccard similarity between the probe-ranked and gradient-ranked top-20 sets was $0.11$: only 4 heads overlapped out of 36 unique heads. The gradient-ranked selector concentrated in layer 5 (4 heads) and selected late-layer heads (layers 27--28), while the probe selector concentrated in layers 4 and 9 with high AUROC ($0.92$--$1.0$) but no gradient signal.[^fn-d7-pilot]

**Table 4 — Probe vs. Gradient Selector: Detection Quality and Steering Outcome**

| Property | Probe-ranked (top 20) | Gradient-ranked (top 20) |
|---|---|---|
| Ranking criterion | Per-head AUROC on harmful/benign contrast | Mean $|\nabla|$ of refusal probability w.r.t. head output |
| Detection quality (top heads) | AUROC $1.0$ / $1.0$ / $0.99$ / $0.98$ / $0.96$ | Not applicable (causal, not discriminative) |
| Best steering effect | $-2$ pp $[-10, +6]$ (null) | $-9.0$ pp $[-12.2, -5.8]$ |
| Jaccard overlap with other selector | 0.11 (4/36 heads) | 0.11 (4/36 heads) |
| Head concentration | Layers 4, 9 | Layers 5, 27--28 |

### Interpretation

The probe-ranked heads discriminated harmful from benign activations perfectly, yet perturbing them produced no behavioral change. The gradient-ranked heads were never assessed for discriminative quality, yet intervening on them reduced harmful compliance by 9.0 pp with a confidence interval excluding zero.

This dissociation is stronger than the FaithEval result in one respect: the probe readout was not merely *matched* but *perfect* (AUROC $= 1.0$ on the top heads). If readout quality were a reliable proxy for steering utility, the probe-ranked set should have been the stronger intervention target. Instead, it was inert.

The result is consistent with a distinction between components that *encode* a behavioral label and components that *control* the labeled behavior. Probe-ranked heads may sit downstream of the decision mechanism — well-positioned to read out the model's already-committed behavioral state, but unable to change it when perturbed. Gradient-ranked heads, by contrast, are selected precisely for their causal influence on the behavioral output. Wu et al. (2025) document a related predict/control gap in the AxBench evaluation of SAE features. Bhalla et al. (2024) observe that some features score well on predict metrics and poorly on control metrics within their framework.

[^fn-causal-metadata]: `data/contrastive/refusal/iti_refusal_causal_d7/extraction_metadata.json`; gradient computed as mean $|\partial p_{\text{refuse}} / \partial \mathbf{v}_{\text{head}}|$ over the harmful/benign contrast set.
[^fn-d7-pilot]: `notes/act3-reports/2026-04-07-d7-causal-pilot-audit.md`; probe results from pilot ($n = 100$), 5 alphas.
[^fn-d7-full500]: `notes/act3-reports/2026-04-08-d7-full500-audit.md`; full-500 confirmatory run, structured summary at `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_csv2_report.json`.

## 4.4 Gradient-Based Selection as Supporting Comparator

The gradient-ranked result in Section 4.3 demonstrates that an alternative selection criterion can produce significant behavioral change where probe-based selection failed. This is valuable as a contrast, but three caveats prevent us from making a strong standalone claim about the gradient-based method.

**Caveat 1: No random-head control.** The full-500 confirmatory run did not include a random-head negative control at matched $k = 20$ and $\alpha = 4.0$. Without this control, we cannot distinguish between two explanations: (a) the gradient-ranked heads are specifically the right components to perturb, or (b) any sufficiently strong head-level perturbation at this alpha produces apparent harm reduction. The pilot probe-null is informative (probe-ranked perturbation at the same $k$ was inert) but is not a matched random-head control.[^fn-d7-full500]

**Caveat 2: Model degeneration is visible.** At $\alpha = 4.0$, 112 of 500 gradient-ranked responses (22.4%) hit the 5,000-token generation cap, compared to 0% at baseline. Subset analysis showed that the safety gain persisted in both cap-hit and non-cap subsets (non-cap: $-9.8$ pp $[-13.7, -5.9]$; cap-hit: $-6.3$ pp $[-11.6, -0.9]$), and 97 of 112 cap-hit responses were scored safe rather than strictly harmful. The degeneration pattern was consistent with verbose refusal drift rather than hidden harmfulness, but it represents a quality cost that must be reported alongside the safety gain.[^fn-d7-full500]

**Caveat 3: Benchmark-local.** The gradient-ranked intervention was evaluated on a single benchmark (JailbreakBench, $n = 500$). Its behavior on other evaluation surfaces — factual accuracy, fluency, instruction following — remains untested.

**What the comparator does establish.** Despite these caveats, the gradient-based result serves two roles in the paper's argument. First, it confirms that the probe-null is not an artifact of the ITI intervention family being inherently incapable on this benchmark: the same intervention architecture, applied to different heads, produced a significant effect. Second, the low Jaccard overlap (0.11) between selectors demonstrates that the two ranking criteria surface genuinely different model components — the behavioral divergence follows from a component-selection divergence, not from a superficial implementation difference.

For the purpose of this paper's thesis, the gradient-ranked result functions as an existence proof: on this benchmark and generation surface, a causally motivated selection criterion identified targets that a discriminative criterion missed. We do not claim that gradient-based selection is universally superior; the missing random-head control prevents this.

## 4.5 Synthesis

**Table 5 — Summary of Detection-Steering Dissociations**

| Comparison | Detection | Steering | Control evidence | Lesson |
|---|---|---|---|---|
| Neurons vs. SAE features (FaithEval) | AUROC $0.843$ vs. $0.848$ | $+2.09$ pp/$\alpha$ $[1.38, 2.83]$ vs. $+0.16$ pp/$\alpha$ $[-0.51, 0.84]$ | 8-seed neuron null; delta-only SAE null | Matched detection, divergent steering |
| Probe vs. gradient heads (jailbreak) | AUROC ${\geq}0.92$ (top-20) vs. not assessed | Best $-2$ pp $[-10, +6]$ vs. $-9.0$ pp $[-12.2, -5.8]$ | Probe null is clean; gradient lacks random-head control | Perfect detection, zero intervention |

Two patterns emerge from Table 5, and both are necessary for the paper's claim.

First, detection quality did not predict steering success. The SAE probe matched the neuron probe on held-out AUROC and failed entirely on steering; the probe-ranked heads achieved perfect discrimination and produced no behavioral change. In neither case was the failure attributable to a confound: the delta-only architecture ruled out reconstruction error in the SAE comparison, and the inert probe-ranked heads shared the same intervention family (ITI) that succeeded under gradient-based selection.

Second, the failures were not caused by the absence of signal — they were caused by the wrong *kind* of signal. The distinction between components that *read out* a behavioral state and components that *control* it is not captured by held-out discriminative performance. Magnitude-ranked neurons happened to lie in the causal path for FaithEval compliance; SAE features of matched detection quality did not. Gradient-ranked attention heads lay in the causal path for jailbreak refusal; probe-ranked heads of superior detection quality did not.

The positive counterexample is important: magnitude-ranked neurons *did* steer FaithEval compliance ($+6.3$ pp $[4.2, 8.5]$ at $\alpha = 3.0$), with specificity confirmed against 8 random-neuron seeds. The thesis is not that detection-based targets never work. It is that detection quality alone is an unreliable heuristic for identifying when they will.


---

# 5. Case Study II — Control Is Surface-Local and Can Externalize

The previous section established that strong readouts do not reliably identify useful steering targets. This section asks a complementary question: when steering *does* work, how far does the effect transfer? We find that successful interventions are surface-local — gains on one evaluation construct do not port to nearby constructs — and that the relevant externality is not generic degradation but a specific behavioral failure mode.

Figure 3 visualizes this bridge failure pattern, including the flip taxonomy and representative substitutions.

## 5.1 Positive Results Exist, but Are Task-Local

H-neuron scaling (magnitude-ranked selection, 38 neurons) produces clear, dose-dependent effects on compliance-adjacent surfaces. On FaithEval anti-compliance prompts, MCQ context-acceptance rate increases by +6.3 pp [4.2, 8.5] at $\alpha = 3.0$, confirmed by 8-seed negative controls that show flat random-neuron baselines (slope +0.02 pp/$\alpha$).^[Source: `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`; H-neuron sweep in `data/gemma3_4b/intervention/faitheval/experiment/results.json`.] On FalseQA, the same neurons produce +4.8 pp [1.3, 8.3].^[Source: `notes/measurement-blueprint.md`, benchmark MDE table; no dedicated act3 report yet.] On jailbreak, single-seed controls provide initial specificity support (seed-0 only; seeds 1–2 pending): H-neuron slope +2.30 pp/$\alpha$ [0.99, 3.58] versus random-neuron slope $-0.47$ pp/$\alpha$ [$-1.42$, 0.47], difference +2.77 pp/$\alpha$ [1.17, 4.42], $p = 0.013$.^[Source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`.]

These results establish that detector-selected targets can produce real, specific steering effects. The issue is not that detection-based selection never works — it is that working effects are task-local. The same 38 neurons that shift compliance on FaithEval produce a null effect on BioASQ factoid QA endpoint accuracy: $-0.06$ pp [$-1.5$, 1.4] on a well-powered sample ($n = 1{,}600$, MDE ${\sim}2$ pp), despite substantially perturbing answer style in 1,339 of 1,600 responses.^[Source: `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md`.] The endpoint is flat, but the intervention is behaviorally active — this is a portability limit on the metric, not behavioral inactivity. The intervention modulates compliance-related behavior but does not improve domain-specific factual accuracy.

## 5.2 ITI Improves Answer Selection but Not Open-Ended Generation

Inference-Time Intervention using TruthfulQA-sourced truthfulness directions (Li et al., 2023) produces a clear improvement on answer selection: +6.3 pp MC1 [3.7, 8.9] and +7.49 pp MC2 [5.28, 9.82] on held-out TruthfulQA folds, with 61 incorrect→correct flips against 20 correct→incorrect flips at $\alpha = 8.0$.^[Source: `notes/act3-reports/2026-04-01-priority-reruns-audit.md`, §2.]

This gain does not transfer to open-ended factual generation. On SimpleQA ($n = 1{,}000$) with a prompt that removes the explicit escape hatch, ITI at $\alpha = 8.0$ reduces compliance from 4.6% [3.5, 6.1] to 2.8% [1.9, 4.0] (paired $\Delta = -1.8$ pp [$-3.1$, $-0.6$]).^[Source: `notes/act3-reports/2026-04-01-priority-reruns-audit.md`, §3b.] The attempt rate collapses from 99.7% to 67.0%, indicating that the truthfulness direction promotes epistemic caution — the model hedges or refuses rather than generating factual answers. Precision among attempted answers remains stable (${\sim}4$%), suggesting the intervention does not improve factual accuracy; it merely suppresses generation.

The contrast is stark: on a constrained answer-selection task (TruthfulQA MC), the same direction helps the model pick correct options. On an open-ended generation task (SimpleQA), it suppresses attempts without improving accuracy. Answer-selection success is not evidence for generation-level truthfulness improvement.

## 5.3 Bridge: The Sharpest Behavioral Diagnosis

The TriviaQA bridge benchmark (dev set, $n = 100$, baseline adjudicated accuracy 47.0%) provides the most informative externality result because it reveals a specific behavioral failure mode, not just a score drop.

**The intervention is active, not simply suppressive.** On this dev split, at $\alpha = 8.0$, the ITI E0 (paper-faithful, $K = 12$) direction reduces adjudicated accuracy by $-7.0$ pp [$-14.0$, 0.0], with 10 right-to-wrong flips against 3 wrong-to-right rescues (net $-7$, McNemar $p = 0.096$). The modernized E1 variant ($K = 8$) is worse: $-9.0$ pp [$-16.0$, $-3.0$], with 10 right-to-wrong flips against only 1 rescue (net $-9$, McNemar $p = 0.016$, CI excludes zero).^[Source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`, §2.2.]

**The failure is not mainly refusal.** NOT\_ATTEMPTED counts increase only from 1 to 3 across the alpha range. The dominant failure mode is not refusal or abstention.

**The dominant single failure mode is confident wrong-entity substitution.** Of the 10 right-to-wrong flips at E0 $\alpha = 8.0$, five (50%) are substitutions where the model replaces a correct factual answer with a different, confidently wrong entity:^[Source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`, §4.2.]

| Question | Baseline (correct) | ITI $\alpha = 8$ (wrong) |
|---|---|---|
| Lewis Carroll hunting poem? | "Hunting for a Snark" | "Hunting for a caucus-race" |
| Lead singer of The Specials? | "Terry Hall" | "Horace Panter" (bassist) |
| Microcephaly = abnormally small \_\_? | "Head size" | "Brain size" |
| WWII coin gambling game? | "Two-up" | "Nimble Nick" |
| Scottish paper with Broons/Wullie? | "The Sunday Post" | "The Scotsman" |

The substitutions are semantically close: a different band member, a related anatomical term, a different Scottish newspaper. This pattern is consistent with the truthfulness direction redistributing probability mass over nearby factual candidates rather than suppressing generation entirely.

> **Box B — Worked Bridge Rescue Example**
>
> On `qw_4300` ("First pop video by John Landis?"), the baseline answer was
> "Saturday Night's Alright for Fighting," which is wrong. ITI corrected this
> to "Thriller" at both $\alpha = 4$ and $\alpha = 8$ on the bridge dev set.
> This matters because the bridge story is not "ITI only breaks generation."
> The intervention is behaviorally active in both directions: it can rescue a
> few questions, but the damage rate is larger than the rescue rate on this
> surface.^[Source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`,
> §4.1.]

**The failure is reproducible across intervention variants.** Both E0 and E1 produce exactly 10 right-to-wrong flips at $\alpha = 8.0$, sharing the same set of damaged questions.^[Source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`, §2.3.] The flip asymmetry structure is stable: both variants damage the same subpopulation, differing only in rescue capacity (E0 rescues 3, E1 rescues 1). This suggests the failure is driven by the intervention family's interaction with the model's factual retrieval, not by idiosyncratic properties of a single direction variant.

**Three additional failure modes account for the remaining flips:** verbosity-induced answer loss (3/10; the model elaborates past the target answer), evasion (1/10; the model describes instead of answering), and paraphrase drift (1/10).

## 5.4 Externality as First-Class Evidence

The bridge result illustrates why externality should not be an appendix concern. The same ITI direction that improves TruthfulQA MC1 by +6.3 pp causes $-7$ pp to $-9$ pp damage on open-ended factual generation on the bridge dev set (Limitation L5) — and the dominant damage mechanism (5 of 10 flips) is not generic quality loss but confident substitution of nearby wrong entities, with verbosity-induced answer loss (3/10), evasion (1/10), and paraphrase drift (1/10) accounting for the remainder. This failure would be invisible if the evaluation contract stopped at answer-selection benchmarks.

The H-neuron scope boundary reinforces the same lesson from a different direction: an intervention that works on compliance-adjacent surfaces (FaithEval, FalseQA, jailbreak) produces zero effect on a domain-specific factual QA task (BioASQ). The mechanism appears to be over-compliance modulation, not general truthfulness improvement. Without cross-surface evaluation, this scope boundary would go undetected.

> Steering success on one evaluation surface does not establish usefulness on a nearby surface. In this program, the relevant externality was not only score degradation but a specific failure mode: confident substitution of semantically nearby false answers for correct ones.

These bridge estimates remain dev-set point estimates. The promoted candidate has
not yet been run on the held-out test split ($n = 400$), and the failure-mode
taxonomy is based on manual analysis of 10 flips, so we use it for qualitative
diagnosis rather than population-rate estimation (Limitation L5). We also use
"confident wrong-entity substitution" as a behavioral diagnosis, not as a claim
about the internal circuit mechanism.


---

# 6. Measurement Choices Changed the Scientific Conclusion

<!-- Anchor 3 of the four-stage scaffold: Measurement → Conclusion break.
     Source files (canonical for all numbers):
       notes/act3-reports/2026-04-08-d7-full500-audit.md
       notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md
       notes/act3-reports/2026-04-12-4way-evaluator-comparison.md
       notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md
       notes/measurement-blueprint.md
-->

The preceding sections established that detection quality does not predict
steerability (Section 4) and that successful steering is narrow in scope
(Section 5). Both conclusions rest on behavioral measurements---jailbreak
compliance rates, severity scores, generation-surface accuracy---that are
themselves products of evaluation choices. In this section we show that those
choices are not clerical details: generation length, scoring granularity,
evaluator identity, and pipeline hygiene each independently shifted what we
would have concluded about whether a given intervention worked.

We organize the case study around the H-neuron jailbreak scaling experiment
(38 probe-selected neurons, $\alpha \in \{0, 1, 1.5, 3\}$, $n{=}500$ per
condition), because its moderate effect size makes it sensitive to every
measurement decision we examine. Where findings generalize beyond this
experiment, we note it explicitly.

Figure 4 collects the main measurement reversals from this section: binary
versus graded scoring, evaluator disagreement, and specificity under matched
controls.

## 6.1 Truncation Hides Downstream Content

Early jailbreak runs in this project used short generation caps (256 tokens
for the legacy setup; 1024 tokens for a greedy-decode cross-validation).
Gemma-3-4B-IT typically opens jailbreak responses with a refusal preamble
("I cannot help with that...") before pivoting to substantive compliance.
At 256 tokens the generation frequently terminates inside the preamble,
before the harmful payload begins; at 1024 tokens, the greedy decode
similarly truncated responses that exceeded the cap, masking degeneration
artifacts at high intervention strengths.

When we moved to the canonical generation policy ($\texttt{max\_new\_tokens}{=}5000$,
sampled decoding), two findings changed. First, the non-monotonic
bounce-back of the strict harmfulness rate at $\alpha{=}8.0$ in the gradient-based causal
pilot---which had appeared as an intervention *reversal*---was revealed to be
an artifact of degeneration: 74\% of gradient-ranked responses and 82\% of probe-ranked
responses hit the 5000-token cap at that strength, and the greedy decode's
1024-token truncation had hidden the degenerate text
(source: `notes/act3-reports/2026-04-07-d7-causal-pilot-audit.md`, Section 5).
Second, the full-generation confirmatory run ($n{=}500$) found 112/500
(22.4\%) of causal responses hitting the token cap at $\alpha{=}4.0$, but
these cap-hit responses were overwhelmingly safe: 97/112 were scored safe
rather than strictly harmful, with mean harmful payload share of 0.0192
(source: `notes/act3-reports/2026-04-08-d7-full500-audit.md`, Section 4.2).
The causal safety effect survived restriction to the non-cap subset:
$-9.8$ pp $[-13.7, -5.9]$.

The truncation artifact is not exotic. Any intervention that alters
generation length---through refusal elaboration, repetitive hedging, or
outright degeneration---will interact with a token cap. Short caps
preferentially hide post-preamble content, biasing evaluations toward
whatever behavior the model produces first. Because Gemma-3-4B-IT leads
jailbreak responses with refusals and follows with compliance, truncation
inflates apparent safety. Our measurement blueprint
(`notes/measurement-blueprint.md`) now prohibits citing the legacy
256-token setup as current evidence.

## 6.2 Binary Versus Graded Evaluation

The H-neuron jailbreak scaling experiment provides the clearest illustration
of how scoring granularity determines the conclusion.

**Binary judge.** A GPT-4o binary harmful/safe classifier showed an
H-neuron compliance increase of $+3.0$ pp from $\alpha{=}0$ to $\alpha{=}3$
(152/500 $\to$ 167/500). The 95\% confidence interval includes zero,
consistent with the minimum detectable effect of ${\sim}6$ pp at this sample
size. Under binary evaluation alone, the H-neuron intervention would be
judged null on jailbreak compliance
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 1.2).

**Graded evaluation (CSV-v2).** The same responses, scored by a structured
rubric that distinguishes refusal, borderline, and substantive compliance,
yielded an H-neuron strict harmfulness slope of $+2.30$ pp/$\alpha$
$[+0.99, +3.58]$ (bootstrap 95\% CI, 10{,}000 resamples). The confidence
interval excludes zero
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 2.1).

**Negative control.** A matched set of 38 randomly selected neurons
(seed 0) produced a slope of $-0.47$ pp/$\alpha$ $[-1.42, +0.47]$---flat,
with the CI comfortably including zero. The slope difference (H-neuron minus
random) was $+2.77$ pp/$\alpha$ $[+1.17, +4.42]$, with a permutation test
$p = 0.013$ (647/50{,}000 permutations $\geq$ observed gap)
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 2.1).

The mechanism behind this divergence is the borderline category. As H-neuron
scaling increased, the graded evaluator registered a *polarization* of
borderline responses: borderline count dropped by 73 (171 $\to$ 98), with 38
migrating to strict compliance and 35 to clear refusal. The random control
showed no polarization (borderline stable at 124--139 across all $\alpha$
values; total compliant-or-borderline unchanged at 245)
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 3.1). Binary evaluation collapsed this three-way structure into a
two-way count, washing out the signal that graded scoring recovered.

This is a single-seed result. Seeds 1--2 have been generated but not yet
scored with the graded rubric; multi-seed replication would strengthen the
specificity claim. But the methodological lesson does not depend on the
H-neuron effect being real in the strong sense: *any* intervention that
shifts responses along a refusal--compliance gradient will appear different
under binary versus graded evaluation, and the direction of the discrepancy
is predictable.

## 6.3 Evaluator Dependence Is Part of the Result

If the binary-versus-graded comparison reveals that *scoring granularity*
matters, the natural follow-up is whether *evaluator identity* matters among
scorers that all produce binary verdicts. We tested four evaluators on a
74-record gold-labeled subset (45 harmful, 29 safe) drawn from H-neuron
jailbreak responses across three intervention strengths ($\alpha \in \{0, 1.5,
3\}$). Gold labels were assigned by deep reading of full model outputs.

### Development-set results

| Evaluator | Judge model | Accuracy [95\% CI] | FP | FN |
|-----------|-------------|---------------------|----|----|
| CSV2 v3   | GPT-4o      | 86.5\% [76.9, 92.5] | 0  | 10 |
| Binary judge | GPT-4o   | 77.0\%              | 4  | 13 |
| StrongREJECT | GPT-4o-mini | 74.3\% [63.3, 82.9] | 0 | 19 |
| CSV2 v2   | GPT-4o      | 73.0\%              | 0  | 19 |

(Source: `notes/act3-reports/2026-04-12-4way-evaluator-comparison.md`,
Section 0.)

However, 24 of these 74 records overlap with the rows used during CSV2 v3's
calibration phase---the prompt IDs were hand-selected precisely because they
represented the failure surface where earlier evaluators struggled. After
removing them, the holdout ($n{=}50$, 27 harmful, 23 safe from 17 prompt
clusters) told a different story:

| Evaluator | Holdout accuracy | Holdout 95\% CI (prompt-clustered) |
|-----------|------------------|------------------------------------|
| CSV2 v3   | 96.0\%           | [90.0, 100.0]                      |
| StrongREJECT | 94.0\%        | [88.0, 100.0]                      |
| CSV2 v2   | 92.0\%           | [84.3, 98.0]                       |
| Binary judge | 90.0\%        | [80.0, 98.0]                       |

(Source: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`,
Sections 0 and 2.)

The v3--StrongREJECT gap compressed from 12.2 pp to 2.0 pp, resting on a
single discordant record. McNemar's exact test yielded $p = 1.0$---no
pairwise comparison reached significance (all $p > 0.25$). On held-out data,
all four evaluators exceeded 90\% accuracy, and no evaluator was
statistically distinguishable from any other.

> **Box C — Worked Evaluator Disagreement Example**
>
> One holdout jailbreak response began with a refusal preamble and then went on
> to provide substantive harmful content. StrongREJECT marked
> `refused = 1`, which zeroed its final score by construction, while the CSV2
> evaluators and the binary judge all marked the output harmful. This is a good
> example of why evaluator disagreement is part of the scientific result here:
> the intervention changes refusal framing, and different evaluators make
> different choices about whether a refusal preamble should dominate the final
> label.^[Source: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`,
> §3; `notes/act3-reports/2026-04-12-4way-evaluator-comparison.md`, §3.]

**We do not conclude from this that evaluator choice is irrelevant.** We
conclude the opposite: evaluator choice matters for precisely the response
regime that interventions produce, and the *apparent* magnitude of that
dependence is itself sensitive to the evaluation sample. Three observations
support this.

*First*, the evaluators disagree on different failure modes. The binary judge
produced 3 false positives on holdout (overcalling safe responses as
harmful); all other evaluators produced zero. StrongREJECT produced 1 solo
false negative on holdout; CSV2 v3 produced zero solo errors---its 2 errors
were in the universal-failure bucket where every evaluator was wrong
(source: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`,
Section 3). For an intervention study that needs to track harm
*reduction* across scaling strengths, false positives create phantom
treatment effects, making the binary judge's error profile qualitatively
different from the others'.

*Second*, the construct measured by StrongREJECT is subtly different from the
construct measured by the CSV2 family. StrongREJECT computes
$\text{score} = (1 - \text{refused}) \times \text{mean(convincingness, specificity)} / 5$.
When $\texttt{refused}{=}1$, the score is zero regardless of the
sub-scores. Among the 19 StrongREJECT false negatives on the full 74-record
set, all 19 had $\texttt{refused}{=}1$; their mean convincingness was 4.58/5
and mean specificity was 4.89/5
(source: `notes/act3-reports/2026-04-12-4way-evaluator-comparison.md`,
Section 3). The model correctly identified the content as convincing and
specific, but the formula discarded this assessment because a refusal
preamble was detected. This is not an error in the usual sense---StrongREJECT
was designed for the attack-evaluation setting, where a refusal is decisive.
In our intervention setting, where the model routinely refuses *and then
complies*, the refusal flag underweights harmful substance that follows
protective framing.

StrongREJECT also differs from the other three evaluators in judge model:
it used GPT-4o-mini while the others used GPT-4o. Sub-score analysis suggests
that the main disagreement is driven by the formula's treatment of the refused
flag rather than model capability, but that confound has not been removed
experimentally and remains part of Limitation L8.

*Third*, holdout validation compressed the apparent evaluator advantage but did
not eliminate the structural differences. CSV2 v3 maintained zero solo errors
and zero false positives on holdout. StrongREJECT's single holdout error was
the construct-mismatch pattern in miniature: the model detected a refusal
preamble, set $\texttt{refused}{=}1$, and the formula zeroed the score
despite substantive harmful content that all three other evaluators correctly
flagged
(source: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`,
Section 3). The question the holdout cannot answer is whether CSV2 v3
genuinely outperforms on *new* hard cases it was not calibrated on;
answering this requires fresh gold labels on unseen refuse-then-comply
responses.

We frame evaluator disagreement as *construct pluralism*, not one judge being
wrong. Each evaluator operationalizes "harmful" differently; interventions
that restructure refusal behavior will occupy the gap between these
operationalizations. Reporting results under a single evaluator without
acknowledging this sensitivity is a form of measurement monoculture that
risks conclusions that do not replicate under alternative scoring.

## 6.4 Pipeline Contamination and Schema Drift

The specificity analysis in Section 6.2 was nearly derailed by a silent
pipeline bug. The analysis script (`scripts/analyze_csv2_control.py`)
imported a normalization function (`normalize_csv2_payload`) from the v3
evaluation module and applied it to v2-scored data. Because v2 records lack
the `schema_version`, `primary_outcome`, and `intent_match` fields that v3
expects, the function silently reclassified 167 of 171 borderline records
(97.7\%) as strict compliance, inflating the strict harmfulness rate from
18.8\% to 52.2\%
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 6.1).

The root cause was a four-line code path: v2's `"borderline"`
$\texttt{harmful\_binary}$ label mapped to `"unknown"` under the v3
vocabulary; the function then re-derived $\texttt{harmful\_binary}$ from
C/S/V/T scores, and any specificity $\geq 1$ triggered
$\texttt{partial\_compliance}$, which in turn resolved to `"yes"`. The fix
was a schema-version guard: v2 data (identified by the absence of the
$\texttt{schema\_version}$ field) is now returned without normalization.

The contaminated pipeline produced plausible-looking output---no errors, no
warnings---and would have yielded the triage verdict
$\texttt{review\_specificity}$ instead of the correct
$\texttt{specificity\_supported}$
(source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`,
Section 6.3). Applied to the control comparison, it would have erased the
slope difference between H-neurons and random neurons by inflating both
baselines.

This episode illustrates that measurement discipline extends below the
evaluator-design level to code-level schema handling. The bug was caught
because the analysis pipeline was re-run from raw data with explicit
integrity checks (500/500 record counts, schema field verification, cross-
condition prompt-ID parity). Had the contaminated rates been accepted at
face value, the H-neuron jailbreak effect would have remained listed as
"unscored" rather than upgraded to "single-seed supported" with $p = 0.013$.

## 6.5 What Is Established and What Remains Open

We summarize the measurement findings by their epistemic status.

**Established:**

- *Truncation artifact.* Short generation caps hide post-preamble harmful
  content in Gemma-3-4B-IT jailbreak responses. Full-generation scoring is
  required for valid jailbreak severity measurement.
- *Binary-versus-graded shift.* Binary evaluation washed out a dose-response
  ($+2.30$ pp/$\alpha$, CI excludes zero) that graded evaluation recovered.
  The mechanism is collapse of the borderline category.
- *Holdout compression.* A 12.2 pp evaluator-accuracy gap compressed to
  2.0 pp (McNemar $p = 1.0$) after removing calibration-contaminated rows.
  Apparent evaluator advantages can be substantially inflated by
  development-set overlap.
- *Seed-0 specificity.* H-neuron scaling produced a steeper jailbreak
  dose-response than a matched random-neuron control (slope difference
  $+2.77$ pp/$\alpha$ $[+1.17, +4.42]$, permutation $p = 0.013$),
  but this is a single-seed result.
- *Contamination fix.* A schema-version mismatch silently reclassified
  97.7\% of borderline records. The fix was four lines; the cost of missing
  it would have been a qualitatively wrong triage verdict.

**Still pending:**

- Multi-seed replication (seeds 1--2 generated, not yet scored).
- Full CSV2 v3 and StrongREJECT scoring on control-condition data.
- Fresh hard-case gold labels for an uncontaminated test of evaluator
  advantages on new refuse-then-comply responses.
- Field-level audit of CSV2 v3 ordinal components (C, S, V, T) against
  human ordinal judgments.

\medskip

In this setting, measurement choices were not clerical details; they changed
what the project would have concluded about whether an intervention worked.
A 256-token generation cap would have hidden the harmful payload.
Binary scoring would have returned a null result. A single evaluator---any of
the four we tested---would have left the construct-sensitivity of the
conclusion invisible. And a schema mismatch in four lines of pipeline code
would have reversed the triage verdict. Each of these measurement decisions
interacts with intervention-altered response structure: longer refusal
preambles, graded compliance, and evaluator-specific operationalizations of
"harmful" are not noise to be averaged away but signal about how the
intervention reshapes model behavior. For intervention research that aims to
make safety claims, the measurement stack is part of the result.


---

# 7. Synthesis — A Four-Stage Audit Framework for Intervention Claims

The preceding case studies demonstrated repeated dissociations between stages that are often treated as a single inferential step: matched readouts diverged from steering outcomes (§4), successful steering did not transfer across evaluation surfaces (§5), and measurement choices materially altered the apparent conclusion (§6). These failures share a common structure: each occurs at a transition between adjacent empirical stages. We synthesize these observations into five practical recommendations and a compact checklist for evaluating mechanistic intervention claims.

## 7.1 The Four Stages Are Separable Empirical Gates

The four stages — **measurement**, **localization**, **control**, and **externality** — are not a theoretical taxonomy. They are an empirical observation about where intervention claims break in practice.

- **Measurement → Localization.** Before interpreting what a feature represents, one must trust the evaluation that defines the behavioral surface. In our jailbreak setting, truncation artifacts, binary-versus-graded scoring, and evaluator construct mismatch each altered the conclusion about whether H-neuron intervention produced a significant effect (§6.1–6.3). Measurement failures propagate: if the evaluation is wrong, localization targets selected against that evaluation are also wrong.

- **Localization → Control.** A feature that predicts behavior on held-out data need not causally control that behavior when perturbed. SAE features matched H-neurons on detection quality (AUROC 0.848 vs. 0.843) yet produced null steering on the same benchmark (§4.2). Probe-ranked attention heads achieved perfect discrimination (AUROC 1.0) yet null intervention on jailbreak (§4.3). The localization-to-control transition broke even under conditions designed to give the readout every advantage.

- **Control → Externality.** An intervention that succeeds on one surface may fail or cause harm on a nearby surface. ITI improved TruthfulQA answer selection by +6.3 pp MC1 but reduced open-ended factual accuracy by $-7$ pp to $-9$ pp on the TriviaQA bridge dev set, with the dominant failure mode being confident substitution of wrong entities rather than refusal or abstention (§5.3; Limitation L5). H-neurons improved compliance on FaithEval (+6.3 pp) but produced a null effect on BioASQ factoid QA (§5.1).

Each stage transition is a distinct empirical claim. Passing one does not license claims about the next.

## 7.2 Five Recommendations

We distill the case study evidence into five concrete recommendations for researchers making mechanistic intervention claims.

**Recommendation 1: Do not treat held-out readout quality as sufficient target-selection evidence.**
Readout quality — whether measured by AUROC, accuracy, or probe coefficient magnitude — is insufficient on its own for identifying useful steering targets. Our matched-readout comparison (§4.2) and perfect-readout null (§4.3) demonstrate that readout metrics can be uninformative or even misleading about intervention utility. Target selection should be validated by downstream behavioral evaluation, not by readout quality alone.

**Recommendation 2: Validate on the behavioral surface you actually care about.**
Answer-selection benchmarks and open-ended generation benchmarks measure different constructs, and intervention effects do not transfer reliably between them (§5.2–5.3). If the goal is to improve factual generation, evaluate on a factual generation benchmark. If the goal is to reduce harmful compliance, evaluate on graded harmful-compliance metrics, not binary proxies (§6.2). Match the evaluation surface to the deployment-relevant behavior.

**Recommendation 3: Use matched negative controls when selecting from many comparable components.**
With hundreds of thousands of neurons, thousands of SAE features, or hundreds of attention heads to choose from, post-hoc selection creates a multiple-comparisons problem. Multi-seed random-neuron controls (as used for H-neuron FaithEval specificity) or random-direction baselines are necessary to establish that the observed effect is component-specific rather than a generic perturbation artifact. When controls are absent, the claim should be marked as provisional (as with our gradient-based causal intervention in §4.4).

**Recommendation 4: Treat evaluator disagreement as information when interventions alter style or refusal structure.**
When an intervention changes the surface form of model outputs — for example, shifting from clean refusal to refuse-then-comply patterns — different evaluators may reach different conclusions about the same outputs. This is not noise; it reflects genuine construct mismatch (§6.3). Report evaluator agreement rates, use holdout validation to separate calibration artifacts from genuine performance differences, and do not rely on a single evaluation rubric for claims about interventions that alter output style.

**Recommendation 5: Report externality and quality debt as first-class outcomes.**
Interventions that help on one metric can harm others. Our bridge results show that the harm is not always generic degradation — it can be a specific, interpretable failure mode (confident factual substitution; §5.3). Report cross-surface effects, capability impacts, and residual quality issues (such as token-cap limitations or response-format distortions) alongside the target-behavior result, not as footnotes.

## 7.3 Theory of Change

If adopted, these recommendations would change the practical workflow of mechanistic intervention research in three ways:

1. **Fewer inflated latent-control claims.** Researchers would no longer move directly from "this feature predicts behavior $X$" to "this feature is a good target for steering behavior $X$." The intermediate validation steps would catch the dissociations we document.

2. **Better separation of monitoring and control.** Features that predict behavior are valuable for monitoring and interpretability even when they are poor steering targets. The four-stage framework makes this distinction explicit, allowing researchers to use strong readouts for monitoring while seeking separate evidence for control utility.

3. **Better-informed choices among intervention strategies.** When an activation-level intervention fails the control or externality gate, the appropriate response is not necessarily to find a better feature — it may be to choose a different intervention modality entirely (e.g., abstention, reranking, or training-time correction rather than inference-time steering).

## 7.4 Checklist for a Credible Steering Claim

> **Box D — Minimum Audit for an Intervention Claim**
>
> | Gate | Question | Evidence required |
> |---|---|---|
> | **Measurement** | Can you trust your evaluation? | Pre-specified primary metric; full-generation scoring where relevant; evaluator agreement check if intervention alters output style |
> | **Localization** | Does the readout identify causally relevant components? | Held-out readout quality; but also: does steering through these components change behavior? Readout alone is not enough. |
> | **Control** | Does intervention produce the intended behavioral change? | Dose-response curve on target surface; matched negative control (random component, random direction); effect size with CI |
> | **Externality** | Does the effect transfer and not externalize? | Evaluation on at least one non-target surface; capability mini-battery; report harms and scope conditions explicitly |
>
> A claim that passes only one or two of these gates is a monitoring or localization result, not a steering result.

We present this framework as a methodological synthesis, not as an ontological
claim that every experiment belongs to exactly one stage. Some evidence lines
span multiple gates, and the framework is validated here only by a single
case study in Gemma-3-4B-IT. Its utility for other models and intervention
families remains to be tested.


---

# 8. Limitations and External Validity

We organize limitations into three categories: those that constrain the thesis itself, those that constrain the scope of evidence behind it, and those that affect estimated effect sizes. Table 6 provides a compact summary; the paragraphs below contextualize the most consequential entries.

**Table 6.** Limitation inventory.

| # | Limitation | Which claim it constrains | Thesis or scope? | Next fix |
|---|---|---|---|---|
| L1 | **Single model.** All experiments use Gemma-3-4B-IT. | All claims. The dissociations we document may not replicate in models with different architectures, scales, or training regimes. | Scope | Replication on a second model family; we note that Gao et al. (2025) report qualitatively similar H-neuron compliance effects across six models, providing indirect evidence that at least the detection-side pattern is not model-specific. |
| L2 | **Matched-readout control variable is imperfect.** SAE features and H-neurons are matched on AUROC but differ in representational granularity, selection pathway, operator form, and layer coverage. | The localization-to-control dissociation (§4.2). The matched-AUROC design controls for readout quality but cannot rule out that some other property of the feature family explains the steering divergence. | Thesis | Systematic feature-family ablation: vary one dimension (e.g., operator form) while holding others constant. The probe-head AUROC 1.0 null (§4.3) provides partially independent evidence because it uses a different feature family and still shows a detection-steering disconnect. |
| L3 | **H-neuron selection is a detector baseline, not a full reproduction.** We replicated the neuron-selection procedure of Gao et al. (2025) on Gemma-3-4B-IT, not their full experimental pipeline or model family. | H-neuron effect sizes and their comparability to the original work. | Scope | A full-pipeline replication is outside the scope of this paper; we treat H-neurons as a strong detector-selected baseline rather than claiming equivalence with the originating study. |
| L4 | **SAE layer coverage is partial.** SAE feature steering was tested with one SAE width at a subset of available layers. | The SAE steering null (§4.2). A different SAE width, training procedure, or layer selection could yield non-null results. | Scope | Broader SAE sweep; the delta-only control rules out reconstruction noise as the explanation for the null, but does not rule out insufficient layer coverage. |
| L5 | **Bridge test split not yet used for the promoted candidate.** The confident-wrong-substitution analysis (§5.3) is based on dev-set evaluation ($n = 100$). | Bridge effect-size estimates and the mechanistic diagnosis of wrong-entity substitution. | Effect size | Run the promoted ITI configuration on the held-out bridge test split. The qualitative finding (wrong-entity substitution as the dominant failure mode) is unlikely to reverse; the quantitative estimates may shift. |
| L6 | **Gradient-based causal intervention lacks a random-head control.** The gradient-based causal intervention result (§4.4) compares gradient-ranked heads against probe-ranked heads and baseline, but does not include a matched random-head negative control. | Selector specificity for the gradient-based result. The observed harm reduction could partly reflect generic perturbation rather than component-specific causality. | Thesis (for selector-specificity sub-claim only) | Run a random-head control at the same intervention strength. This limitation is why §4.4 presents the result as supporting evidence with explicit caveats rather than a headline finding. |
| L7 | **Jailbreak specificity is single-seed.** The random-neuron negative control for H-neuron jailbreak intervention has been confirmed on seed 0 ($p = 0.013$); seeds 1 and 2 are pending. | H-neuron jailbreak specificity. The effect may be less robust than the single-seed $p$-value suggests. | Effect size | Score seeds 1 and 2 against the same evaluator panel. FaithEval and FalseQA specificity are multi-seed and robust; jailbreak is the remaining single-seed surface. |
| L8 | **Evaluator uncertainty remains live.** Holdout validation compressed the CSV-v3 advantage over StrongREJECT from 12.2 pp to 2.0 pp. All four evaluators exceed 90% agreement on the holdout, but the hard-tail cases where evaluators disagree most are precisely the intervention-altered outputs where accuracy matters most. StrongREJECT also uses GPT-4o-mini while the other evaluators use GPT-4o, so a judge-model confound remains entangled with formula differences. | Jailbreak effect-size estimates (§6); any claim about evaluator superiority. | Effect size | Multi-evaluator reporting with explicit disagreement analysis; a fresh hard-tail holdout to test calibration transfer; matched-model evaluator rerun to separate formula effects from judge-model effects. We report evaluator dependence as part of the scientific result (§6) rather than treating it as a resolved methods problem. |
| L9 | **Answer-token confound audit is incomplete.** Not all benchmarks have been checked for position-based artifacts in answer-token extraction. | Any benchmark-specific effect size where unaudited answer extraction could inflate or deflate the intervention signal. | Effect size | Systematic position-artifact audit across all evaluation surfaces. |

## 8.1 Limits on the Thesis

The two limitations that bear most directly on the paper's central claim are L1 (single model) and L2 (imperfect matching of the control variable).

The single-model limitation is real and we do not minimize it. Our contribution is methodological, not universal: we demonstrate that the readout-to-steering heuristic fails repeatedly within Gemma-3-4B-IT and provide a framework for auditing whether it holds in other settings. The model name appears in the title for this reason. We note that the detection-intervention dissociation manifests across multiple feature families (neurons, SAE features, probe-ranked heads) and multiple evaluation surfaces (contextual faithfulness, jailbreak, factual generation), which suggests the pattern reflects something about the inference from readout to steering rather than an idiosyncrasy of a single feature family or benchmark. But this is suggestive, not conclusive, and replication on a second architecture is the obvious next step.

The matched-readout design (L2) is the strongest methodological concern for the localization-to-control dissociation. SAE features and H-neurons share a readout quality (AUROC $\approx$ 0.84) but differ along multiple dimensions: the SAE operates on residual-stream activations while H-neurons are individual MLP outputs; the SAE steering operator replaces or adds to the reconstructed activation while H-neuron scaling multiplies pre-activations; the SAE was trained on a subset of layers while H-neurons were selected across all layers. Any of these differences could explain the steering divergence independently of readout quality. We chose to match on the metric most commonly used to justify steering (held-out discrimination quality) precisely because this is the heuristic under test, but we acknowledge that the matching is one-dimensional. The probe-head result (§4.3) provides a partially independent check: a different feature family, a different operator, and perfect readout quality, yet the same null outcome. Together, the SAE and probe-head results make it unlikely that the dissociation is an artifact of one particular feature-operator mismatch, but they do not rule out all possible confounds.

## 8.2 Limits on Scope and Effect Sizes

The remaining limitations (L3--L9) constrain the precision or generality of individual results without threatening the paper's central argument. We highlight two that most affect interpretation.

The bridge dev-set limitation (L5) means that the confident-wrong-substitution diagnosis rests on $n = 100$ examples. The qualitative pattern --- 5 of 10 accuracy-decreasing flips involved substitution of a semantically nearby but wrong entity, and the same wrong entities appeared under independent intervention replications --- is striking and mechanistically specific. But the quantitative framing ($-7$ pp to $-9$ pp generation accuracy) should be treated as a point estimate pending test-split confirmation. We report this limitation alongside the bridge analysis in §5.3 and do not claim precise transfer-harm magnitudes.

The evaluator-uncertainty limitation (L8) is by design part of the scientific contribution rather than a weakness we hope to resolve before submission. Our central claim in §6 is that evaluator choice materially affects intervention conclusions. The holdout compression of the CSV-v3 advantage demonstrates exactly this: an evaluator that appeared substantially better on the development set showed only a marginal edge on held-out data. We report multi-evaluator results throughout the jailbreak analysis and treat scorer disagreement as information. The limitation is that we cannot declare any single evaluator authoritative for intervention-altered outputs, which means our jailbreak effect-size estimates carry irreducible evaluator-dependent uncertainty. This uncertainty is partly conceptual and partly experimental: StrongREJECT's refusal-zeroing formula reflects a different measurement construct, and its use of GPT-4o-mini rather than GPT-4o means the formula effect is not perfectly separated from judge-model choice. We state this explicitly rather than resolving it by fiat.


---

# 9. Conclusion

We tested a widespread heuristic in mechanistic intervention research — that features which predict a behavior well make good targets for steering that behavior — in a comparative case study across multiple intervention families in Gemma-3-4B-IT. The heuristic failed repeatedly: matched or even perfect readout quality did not reliably identify useful steering targets, successful interventions were narrow in scope, and measurement choices materially altered the apparent scientific conclusion.

These findings do not imply that detection-based target selection never works. H-neurons, detector-selected through classification performance, produced clear compliance effects on multiple surfaces. The refusal direction literature demonstrates reliable single-axis steering. The thesis is *heuristic unreliability*, not impossibility: readout quality alone is insufficient for identifying good intervention targets, and the gap between readout quality and steering utility is large enough to invalidate the implicit assumption that one predicts the other.

The four-stage audit framework — measurement, localization, control, externality — provides a practical decomposition for evaluating mechanistic intervention claims. Each stage is a distinct empirical gate with its own evidence requirements. Passing one gate does not license claims about the next. We hope this framework helps the field move from "we found a feature that predicts behavior $X$" toward "we have stage-specific evidence that intervening on this feature usefully changes behavior $X$, on the surfaces we care about, without unacceptable externalities."


---

# References

[To be compiled from `notes/paper/citations/registry.json`.]

# Appendix

## Appendix A. Detector Interpretation Audits

This appendix collects the two detector-interpretation cautions referenced in §4.1.

**N4288 / L1 instability.** The top-weight CETT neuron at layer 20, neuron 4288, is not a stable "most important neuron" finding across regularization settings. In the pipeline audit, it is absent at $C <= 0.3$, appears only at $C = 1.0$, and drops to rank 5 at $C = 3.0$ while a wider detector reaches 80.5% test accuracy with 219 positive neurons. Across six follow-up analyses, the verdict is that the extreme weight is better explained by L1 concentration among correlated features than by unique causal importance.^[Source: `data/gemma3_4b/pipeline/pipeline_report.md`, detector audit section; `scripts/investigate_neuron_4288.py` generated the underlying analyses.]

**Verbosity confound.** The verbosity audit shows that under full-response aggregation, response length dominates truthfulness signal by roughly $3.7\times$ (mean aggregation) to $16\times$ (max aggregation), and 36 of 38 H-neurons are more length-dominant than truth-dominant in that setting. The mitigation is scope: the classifier result is still a held-out answer-token discrimination result, but it should not be interpreted as a pure hallucination detector without this qualification.^[Source: `data/gemma3_4b/intervention/verbosity_confound/verbosity_confound_audit.md`.]

These appendix notes are not separate headline claims. Their role is to justify the narrower wording used in the main text: the readouts are real, but detector interpretation is fragile.

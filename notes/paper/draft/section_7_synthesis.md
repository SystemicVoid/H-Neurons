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

# 7. Synthesis — A Four-Stage Audit Framework for Intervention Claims

Sections 4--6 isolate three different failure points: matched readouts diverged from steering outcomes, successful steering did not transfer cleanly across surfaces, and measurement choices altered the apparent conclusion. This section turns those case studies into a practical audit framework for evaluating mechanistic intervention claims.

## 7.1 The Four Stages Are Separable Empirical Gates

The four stages -- **measurement**, **localization**, **control**, and **externality** -- are not a theoretical taxonomy. They are an empirical observation about where intervention claims break in practice.

- **Measurement -> Conclusion.** Before deciding whether an intervention worked, one must trust the evaluation that defines the behavioral surface. In our jailbreak setting, truncation artifacts, binary-versus-graded scoring, and evaluator choice each altered the conclusion about the same outputs (§6.1--6.3). After the StrongREJECT GPT-4o rerun, the holdout binary-accuracy gap disappeared; the remaining case for CSV-v3 is richer outcome structure, not superior held-out binary accuracy.
- **Localization -> Control.** A feature that predicts behavior on held-out data need not causally control that behavior when perturbed. SAE features matched H-neurons on detection quality (AUROC 0.848 vs. 0.843), yet in the committed FaithEval comparison they did not translate into useful control while H-neurons did (§4.2). A narrower JailbreakBench selector comparison is directionally consistent, but remains supporting and caveated (§4.3--4.4).
- **Control -> Externality.** An intervention that succeeds on one surface may fail or cause harm on a nearby surface. ITI improved TruthfulQA answer selection by +6.3 pp MC1 but reduced open-ended factual accuracy by $-5.8$ pp [$-8.8$, $-3.0$] on the TriviaQA bridge test set, where wrong-entity substitution was the most frequent diagnosed failure mode (§5.3). H-neurons improved compliance on FaithEval but showed no robust net alias-accuracy effect on BioASQ despite substantial behavioral perturbation (§5.1).

Each stage transition is a distinct empirical claim. Passing one does not license claims about the next.

## 7.2 Five Recommendations

We distill the case study evidence into five concrete recommendations for researchers making mechanistic intervention claims.

**Recommendation 1: Do not treat held-out readout quality as sufficient target-selection evidence.**
Readout quality -- whether measured by AUROC, accuracy, or probe coefficient magnitude -- is insufficient on its own for identifying useful steering targets. Our matched-readout comparison in FaithEval (§4.2) shows that similar held-out discrimination quality can still fail to predict intervention utility.

**Recommendation 2: Validate on the behavioral surface you actually care about.**
Answer-selection benchmarks and open-ended generation benchmarks measure different constructs, and intervention effects do not transfer reliably between them (§5.2--5.3). If the goal is to improve factual generation, evaluate on a factual generation benchmark. If the goal is to reduce harmful compliance, evaluate on graded harmful-compliance metrics, not binary proxies (§6.2).

**Recommendation 3: Use matched negative controls when selecting from many comparable components.**
When component search is large, post-hoc selection creates a multiple-comparisons problem. Multi-seed random controls or random-direction baselines are necessary to establish that an observed effect is component-specific rather than a generic perturbation artifact. Partial controls still matter, but they do not close the question on their own.

**Recommendation 4: Treat evaluator disagreement as information when interventions alter style or refusal structure.**
When an intervention changes the surface form of model outputs, different evaluators may reach different conclusions about the same outputs. This is not just noise; it reflects construct mismatch (§6.3). Report evaluator agreement, separate rubric effects from judge-model effects when possible, and avoid single-rubric conclusions for style-altering interventions.

**Recommendation 5: Report externality and output-quality costs as first-class outcomes.**
Interventions that help on one metric can harm others. Our bridge results show that the harm is not always generic degradation; it can be a specific, interpretable failure mode (§5.3). Cross-surface effects and output-quality costs belong next to the headline result, not in the footnotes.

## 7.3 Theory of Change

If adopted, these recommendations would make mechanistic intervention studies harder to over-interpret. They would separate monitoring from control, force target-selection claims to survive downstream evaluation, and make researchers report transfer costs and evaluator sensitivity before presenting a steering result as settled.

## 7.4 Checklist for a Credible Steering Claim

> **Box D -- Minimum Audit for an Intervention Claim**
>
> | Gate | Question | Evidence required |
> |---|---|---|
> | **Measurement** | Can you trust your evaluation? | Pre-specified primary metric; full-generation scoring where relevant; evaluator agreement check if intervention alters output style |
> | **Localization** | Does the readout identify causally relevant components? | Held-out readout quality; but also: does steering through these components change behavior? Readout alone is not enough. |
> | **Control** | Does intervention produce the intended behavioral change? | Dose-response curve on target surface; matched negative control (random component, random direction); effect size with CI |
> | **Externality** | Does the effect transfer and not externalize? | Evaluation on at least one non-target surface; capability mini-battery; report harms and scope conditions explicitly |
>
> A result that clears only one or two of these gates is a monitoring or localization result, not a steering result.

We present this framework as a methodological synthesis, not as an ontological claim that every experiment belongs to exactly one stage. Some evidence lines span multiple gates, and the framework is validated here only by a single case study in Gemma-3-4B-IT. Its utility for other models and intervention families remains to be tested.

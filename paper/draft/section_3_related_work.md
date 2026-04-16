# 3. Related Work and Open Gaps

Prior work already establishes much of the cautionary background for this paper. The point of this section is therefore narrow: identify the specific results the literature has already earned, then locate the smaller gap addressed by the case studies in Sections 4--6.

## 3.1 What Prior Work Already Shows

**Decodability does not imply causal use.** Foundational probe critiques already establish that high readout accuracy can arise for reasons the model does not functionally rely on. Hewitt and Liang (2019) showed that simple control tasks can support high probing accuracy without informative structure; Elazar et al. (2021) showed that removing a decodable property often leaves task behavior unchanged; and Kumar et al. (2022) showed that probing classifiers can be unreliable even for concept *detection*. We therefore treat "predictive" as weaker than "causally useful" from the outset.

**Detection and steering can diverge.** More recent work moves from generic probe skepticism to the intervention question directly. Arad et al. (2025) showed within SAE feature selection that high-scoring features need not steer well. AxBench (Wu et al., 2025) reported that detection and steering need not be jointly optimized even on synthetic concepts. Bhalla et al. (2024) described a predict/control discrepancy for interpretability-based interventions, and Li et al. (2023) showed in ITI that multiple-choice gains can diverge from generative gains and that the probe-weight direction was not the best steering direction.

**Answer-selection success does not guarantee generative usefulness.** Li et al. (2023) already demonstrated this within ITI. Pres et al. (2024) argued that many steering evaluations look too good because they rely on simplified settings rather than open-ended generation, and Opielka et al. (2026) showed that causally effective function vectors can be format-specific across multiple-choice and open-ended tasks.

**Evaluator and metric choices can change safety conclusions.** StrongREJECT (Souly et al., 2024) showed that binary attack-success scoring can overestimate jailbreak effectiveness relative to graded rubrics. GuidedBench (Huang et al., 2025), Chen and Goldfarb-Tarrant (2025), and Eiras et al. (2025) further showed that evaluator guidelines, formatting artifacts, and modest presentation changes can materially affect safety judgments. The general point that measurement matters is therefore established before this paper begins.

**Better localization does not reliably predict better editing.** Hase et al. (2023) showed this in factual knowledge editing, providing a close precedent for the localization/control dissociation studied here even though the intervention class differs.

## 3.2 Gap Addressed Here

Three narrower gaps motivate this paper.

**Matched cross-target-type comparison on one behavioral surface.** Prior work shows detection-steering divergence within single method families, but we are not aware of prior work that matches held-out readout quality across different target types on the same model and behavioral surface. The paper's cleanest contribution here is the FaithEval comparison between magnitude-ranked neurons and SAE features. The jailbreak selector study is supporting evidence about selector choice, not part of that matched claim.

**Behavioral diagnosis of transfer failure.** The literature already warns that constrained-surface gains need not transfer to open-ended generation, but the exact failure mode under truthfulness steering is less characterized. The TriviaQA bridge case study asks whether the intervention merely suppresses answers or instead corrupts them in a patterned way; in this setting, the dominant observed failure mode is wrong-entity substitution.

**An integrated audit of measurement, localization, control, and externality.** Prior work raises these concerns mostly in isolation. This paper brings them together in one case study and shows that, in a mechanistic intervention setting, truncation, scoring granularity, and evaluator choice can change what the paper can honestly conclude about whether an intervention worked.

Prior work therefore already motivates both skepticism and positive counterexamples. Arditi et al. (2024), Gao et al. (2025), and Nguyen et al. (2025) show that detector- or direction-selected interventions can work very well in some settings. The contribution here is narrower: a matched FaithEval case study, a bridge diagnosis of nearby generation harm, and a four-stage audit scaffold. It is not a universal anti-detector claim.

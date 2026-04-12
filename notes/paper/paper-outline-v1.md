# Paper Outline

**Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT**

But make the *actual framing sentence* in the abstract and introduction more precise:

**A comparative case study in separating measurement, localization, control, and externality in mechanistic intervention research.**

That is the real scientific object. It is stronger than a narrow evaluator paper, and stronger than a D7-centered paper, because it lets the write-up center the deepest evidence you already have: the SAE vs H-neuron dissociation, the ITI answer-selection vs generation split with bridge-mechanism analysis, and the measurement discipline story that materially changes the jailbreak conclusion. D7 is still important, but until the random-head control is done it should be treated as strong benchmark-local evidence, not the single pillar the whole paper rests on.

## 1. Paper identity

### Central question

When a component, direction, feature set, or head set **predicts** harmfulness, refusal, or truthfulness, does that make it a good **intervention target**?

### Core answer

In Gemma-3-4B-IT, **not reliably**. Predictive readouts often failed to identify useful steering targets; when steering worked, it was narrow and surface-dependent; and measurement choices sometimes changed the apparent conclusion enough that “does this intervention work?” depended on using the right evaluator and the right generation budget.

### Three main contributions the paper should claim

First, a **comparative methodological result**: matched or even perfect readouts often did not yield useful steering targets. The cleanest examples are H-neurons vs SAE on FaithEval and probe-ranked vs gradient-ranked heads on jailbreak.

Second, an **externality result**: even successful steering was narrow and often failed off-surface. H-neurons help on compliance-style diagnostics but not everywhere; ITI improves TruthfulQA multiple-choice while harming open-ended generation; the bridge benchmark shows the truthfulness intervention fails by **confident wrong substitution**, not mainly refusal.

Third, a **measurement result**: truncation, coarse judging, and evaluator construct mismatch can materially change the scientific conclusion of a steering experiment. The binary jailbreak story and the CSV-v2/v3 story are not interchangeable.

## 2. The paper outline I would use

I would structure the paper in ten sections plus appendices.

### 0. Title page, abstract, and claim box

**Purpose:** tell the reader exactly what kind of paper this is before they can misread it as a new steering method paper.

**Must contain:** a one-sentence “problem,” one-sentence “question,” one-sentence “comparative setup,” one-sentence “main answer,” and one-sentence “implication.” Put a small boxed claim ledger immediately after the abstract or at the end of the introduction:

* **Earned:** predictive readout quality was an unreliable heuristic for intervention-target selection in this model.
* **Partially earned:** gradient-ranked head selection outperformed probe-ranked selection on the jailbreak surface, but selector specificity is still open without the random-head control.
* **Not earned:** detection and intervention are fundamentally different; causal selection is always better; CSV v3 significantly outperforms StrongREJECT in a broadly validated sense.

**Visual:** a small “claim box,” not a full figure.

---

### 1. Introduction

**Target length:** 1.5–2 pages.

**Purpose:** set up the common assumption in the literature that predictive internal signals are natural steering targets, then state that you are testing this assumption directly rather than taking it for granted.

**Paragraph plan:**

1. Why safety steering is attractive in mechanistic interpretability and why people keep moving from detector quality to intervention claims.
2. Why this jump is tempting: detector papers and steering papers have both been genuinely successful in nearby settings.
3. Why that still leaves a gap: few papers compare multiple mediator families and selector types under one measurement contract on the same model.
4. Your question, answer, and contributions.

**The introduction should explicitly name the positive counterexamples** so the reader knows you are not attacking a strawman. Detector- or direction-based control *can* work: H-Neurons report sparse detector neurons with causal intervention effects; ITI shows truthfulness gains with linear head directions; refusal-direction work shows strong residual-stream control; SAE refusal steering shows feature steering can materially alter refusal; GCM and recent steering-mechanism work show that stronger selector and circuit analysis can improve control. ([arXiv][1])

**Your gap statement should be explicit:** this paper is not asking whether steering ever works. It is asking whether **held-out predictive readout quality is a reliable heuristic for choosing steering targets**, under realistic generation and evaluation. That is a different and sharper question.

**Figure placement:** end of intro.

**Figure 1 — Conceptual map of the paper**
A diagram with four boxes:

**Measurement → Localization → Control → Externality**

Under each box, place the corresponding repo evidence:

* Measurement: truncation audit; binary vs CSV-v2/v3; holdout evaluator validation.
* Localization: H-neuron classifier; 4288 artifact; verbosity confound; SAE probe; probe-head AUROC 1.0.
* Control: H-neuron scaling; SAE steering null; ITI E0/E1/E2; D7 probe vs causal.
* Externality: BioASQ null; SimpleQA damage; bridge confident wrong substitution; D7 token-cap debt.

This figure should visually announce the paper’s thesis: these stages are empirically separable.

---

### 2. Literature landscape and gap

**Target length:** ~1 page.

This section matters more than usual because the field has moved since early 2024. The paper needs to sound like it understands where the frontier is *now*, not where it was when ITI first landed.

I would organize related work into four clusters.

#### 2.1 Readout-centric truth/hallucination/harm detection

This cluster includes H-Neurons, the Universal Truthfulness Hyperplane, semantic entropy probes, “Do Androids Know They’re Only Dreaming of Electric Sheep?”, and newer entity-level hallucination detectors. These works argue that hidden states contain predictive information about truthfulness, hallucination, or harmfulness, sometimes with strong transfer or OOD behavior. ([arXiv][1])

**How you sit relative to them:** you are not disputing that readouts exist. In fact, your H-neuron replication, SAE probe, and probe-head results reinforce that readout quality can be high. Your contribution is to test the next step: whether that readout quality reliably identifies good steering targets.

#### 2.2 Steering and intervention papers

This cluster includes ITI, LITO, TruthFlow, WACK-style intervention studies, refusal-direction work, SAE refusal steering, and newer steering-mechanism papers. This literature shows that steering can work, that adaptive intensity or query-specific correction can help, and that success depends on intervention surface, selector, and externality handling. ([arXiv][2])

**How you sit relative to them:** you are not offering a better steering method. You are offering a comparative diagnosis of why seemingly promising readouts fail to turn into good control targets in one real model. That makes your paper complementary to this cluster rather than competitive with it.

#### 2.3 Mediator choice, causal selection, and evaluation standards

This cluster includes *The Quest for the Right Mediator*, MIB, RAVEL-style benchmark thinking, GCM, and work showing that causally grounded internal mechanisms can better predict OOD behavior than causal-agnostic features. The shared theme is that mediator type, selector, and evaluation standard matter, and that the field needs common tests rather than one-off anecdotes. ([arXiv][3])

**How you sit relative to them:** your paper is a naturalistic comparative case study that supports this shift. MIB even reports that SAE features are not better than standard hidden dimensions for causal-variable localization, while causal-method papers like GCM and Internal Causal Mechanisms push toward better mediator/selector choices. That makes your D7 result legible, but also raises the bar for how carefully you should frame it. ([arXiv][4])

#### 2.4 Monitoring, conditional control, and feature-based supervision

This cluster includes semantic entropy probes, long-form entity-level hallucination detectors, and feature-based reward methods. The important theme is that internal signals may be more valuable for **monitoring, reranking, abstention, or scalable supervision** than for static global activation addition. ([arXiv][5])

**How you sit relative to them:** your bridge result is one of the strongest arguments for this future direction. The intervention is active, but indiscriminate. That is exactly the sort of result that should push a research program from “global static steering” toward “conditional monitoring or reward-guided policy shaping.”

**Table 1 — Related work positioning table**
Columns:

* Paper / method
* Mediator type
* Selection method
* Target surface
* Reported success surface
* Externality evaluation
* What your paper adds relative to it

This table belongs at the end of the related-work section or in Appendix A, depending on space.

---

### 3. Experimental program and measurement contract

**Target length:** 1–1.5 pages.

**Purpose:** establish that this is a comparative program, not a collection of random runs.

**Subsections:**

* 3.1 Model, intervention families, and baseline logic
* 3.2 Benchmarks and what each benchmark is for
* 3.3 Measurement contract and “definition of done”
* 3.4 Claim taxonomy: earned / partial / not earned

**This section should clearly separate surfaces:**

* TruthfulQA MC1 and MC2 are answer-selection surfaces.
* TriviaQA bridge is the main generation-usefulness surface.
* SimpleQA is a low-headroom OOD stress test.
* FaithEval and FalseQA are compliance/anti-compliance diagnostics.
* Jailbreak is the harm benchmark with evaluator sensitivity.
  That role separation is one of the paper’s strengths.

**Table 2 — Benchmark roles table**
Columns:

* Benchmark
* Behavior measured
* Why included
* Evaluator
* Primary metric
* Main caution

**Table 3 — Measurement contract table**
Columns:

* Requirement
* Why it matters
* Which experiments satisfy it now
* Which remain incomplete
  Use the sprint’s “definition of done”: full-generation eval, primary metrics, retained-capability mini-battery, per-example outputs, manifest, negative control or reason not needed, cross-benchmark consistency statement.

**Open checks that belong here as explicit TODO markers:**

* D7 capability battery
* D7 random-head control
* Seed-0 jailbreak control scoring
* tiny C/S/V/T audit
* new hard-tail gold for v3 vs SR.

---

### 4. Detection is real, but detector interpretation is fragile

**Target length:** 1–1.5 pages.

This section should do one thing: show that you are not making an anti-detection argument. Detection is real. But interpreting detectors carelessly is dangerous.

**Subsections:**

* 4.1 H-neuron replication is real on disjoint held-out data
* 4.2 The 4288 story: L1 weight ≠ causal or unique importance
* 4.3 Full-response readout is length-confounded

The H-neuron replication is strong enough to establish a real signal: 38 neurons, disjoint held-out AUROC 0.843, accuracy 76.5%, close to the paper’s reported Gemma-3-4B result. But the same repo work also shows that neuron 4288’s apparent dominance is an L1 regularization artifact, and the verbosity audit shows that full-response CETT readout is dominated by length effects rather than truth effects.

**Figure 2 — Detection is real, but interpretation is fragile**
Panel A: H-neuron replication summary (paper vs your disjoint held-out result).
Panel B: simplified C-sweep showing 4288 absent at low C and no longer dominant at C=3.0.
Panel C: effect-size ratio chart for truth vs length under full-response readout.

This figure should stay compact. The full 4288 deep dive belongs in the appendix, not the main text.

**Risk of overclaim here:** do **not** imply the classifier is invalid. The right claim is that readout exists but its interpretation is fragile.

---

### 5. Main Result I — Predictive readouts are unreliable steering heuristics

**Target length:** 2–3 pages.
**This is the center of gravity of the paper.**

I would structure it as two primary case studies plus one synthesis table.

#### 5.1 Case study A: H-neurons vs SAE features on FaithEval

This is your cleanest apples-to-apples result. Detection is matched: H-neurons 0.843 AUROC vs SAE 0.848. Steering diverges: H-neurons show a clean positive FaithEval anti-compliance slope and random controls stay flat, while SAE steering is null under both full-replacement and delta-only architectures. Delta-only matters because it removes the strongest reconstruction-noise objection.

**Figure 3 — Matched detection, divergent steering**
Panel A: AUROC bars for H-neurons and SAE probe.
Panel B: H-neuron FaithEval dose-response with empirical random-control band.
Panel C: SAE full-replacement and delta-only dose-response, showing nulls and α=1 identity.

This figure should probably be the single most polished figure in the paper.

#### 5.2 Case study B: probe-ranked vs gradient-ranked attention heads on jailbreak

The pilot and trimmed full-500 run give a strong benchmark-local story: AUROC-ranked probe heads are null on jailbreak intervention, while the gradient-ranked causal selector improves csv2_yes and beats both baseline and the L1 comparator on the full-500 surface. But the trimmed run skipped the random-head control and did not fully complete the probe branch at full 500, so the paper must frame this as **evidence that selector choice matters on this surface**, not as a clean selector-specificity proof.

This sits very naturally in the current literature: GCM argues that causal mediation can outperform correlational baselines for long-form response control, and work on internal causal mechanisms and MIB also pushes toward better mediator/selector choices than raw predictive ranking. Your D7 result is not a proof of that literature’s worldview, but it is absolutely in the same direction. ([arXiv][6])

**Figure 4 — Selector choice matters, but caveated**
Panel A: pilot scatter or bars showing probe AUROC 1.0 vs null intervention; causal selector positive.
Panel B: full-500 three-way comparison: baseline, L1 comparator, causal.
Panel C: Jaccard overlap = 0.11.
Panel D: quality-debt inset: token-cap hits at causal α=4.0.

#### 5.3 Synthesis table

**Table 4 — Central exhibit**
Columns:

* Mediator / selector family
* Detection evidence
* Steering result
* Negative control / specificity status
* Externality / quality debt
* Main lesson

Rows:

* H-neurons
* SAE features
* probe heads
* causal heads
* ITI truthfulness direction
* evaluator panel as “measurement layer”

This table is the paper’s real backbone. It should let a reader reconstruct the whole thesis at a glance.

**Open checks for this section:**

* D7 random-head control
* D7 capability mini-battery
* seed-0 jailbreak control scoring if you want H-neuron jailbreak specificity in the same section.

---

### 6. Main Result II — Even successful steering is narrow and surface-dependent

**Target length:** 2–3 pages.

This section is what prevents the paper from reading like “everything failed.” It shows that some interventions do work — just not broadly, and not on the surfaces people casually assume.

#### 6.1 H-neurons are a real positive result, but not a universal one

FaithEval anti-compliance and FalseQA are genuine positives with specificity controls. But BioASQ is null, and even the jailbreak story decomposes differently depending on whether you measure count or severity. This is exactly the right place to defang the “your thesis is false because H-neurons work” objection: yes, they work on some surfaces, which is why the thesis is **unreliable heuristic**, not **never works**.

#### 6.2 Truthfulness directions help answer selection but not generation

TruthfulQA MC is a real ITI win. But generation is null-to-harmful on SimpleQA and on the bridge benchmark, and the bridge benchmark matters because it has real headroom. That makes this a true externality result, not a floor artifact. E1 and E2 sharpen the story: data source matters more than extraction tweaks, and shrinking the head set does not fix the failure mode.

#### 6.3 The bridge benchmark reveals the mechanism: confident wrong substitution

This should be a centerpiece, not a footnote. The bridge benchmark does not merely show “generation no.” It shows *how* the intervention fails: semantically adjacent but wrong entity substitution, with the same wrong entities often reproduced across E0 and E1. That is a much deeper scientific contribution than a flat metric delta.

**Figure 5 — Surface dependence of control**
Panel A: TruthfulQA MC gains for ITI vs H-neurons.
Panel B: generation deltas on SimpleQA and bridge.
Panel C: flip taxonomy for bridge, ideally a Sankey or stacked-bar showing right→wrong, wrong→right, stable-correct, stable-wrong.
Panel D: small case-study box with 4–5 confident wrong substitutions.

This figure should feel like a mechanistic diagnosis, not just a benchmark plot.

**Table 5 — Cross-surface outcome matrix**
Rows:

* H-neurons
* ITI E0
* ITI E1
* ITI E2
  Columns:
* TruthfulQA MC1
* MC2
* SimpleQA
* TriviaQA bridge
* FaithEval
* FalseQA
* Jailbreak
* key caveat

---

### 7. Measurement changes intervention conclusions

**Target length:** 1.5–2 pages.

This section should be presented as a **supporting methodological chapter**, not the flagship thesis. But it should still be treated as a serious contribution, not an appendix afterthought.

#### 7.1 Truncation changed the jailbreak story

The 256-token story was wrong. The 5000-token rerun flattened the binary count effect and revealed that the more stable effect lives in graded severity and content metrics. That alone is a strong lesson about how easily safety conclusions can be warped by generation budget.

#### 7.2 Binary judging washes out graded effects

CSV-v2 recovers a significant jailbreak count effect and monotonic severity escalation that the binary judge largely flattens. That is not just an evaluator preference; it is part of the science of what the intervention is actually doing. 

#### 7.3 CSV v3 should be framed carefully

The holdout story is now clear: v3 is the best primary evaluator you tested for this response regime, because it has zero false positives and zero solo errors on holdout, and its error set is a strict subset of every competitor’s. But the dramatic dev-set accuracy gap to StrongREJECT does **not** survive holdout, and the hard-tail advantage on new refuse-then-educate cases is still unvalidated. The paper should lead with the robust properties, not with the compressed accuracy gap.

This section also benefits from the newer safety literature. Work on refusal direction and on harmfulness-vs-refusal separation makes it much more plausible that a refusal-weighted judge can miss harmful substance that follows protective framing; recent work also complicates the “single refusal direction” story by showing multiple refusal-related directions with similar refusal/over-refusal trade-offs. That is useful context for discussing construct mismatch without oversimplifying the safety geometry. ([arXiv][7])

**Figure 6 — Measurement matters**
Panel A: legacy 256-token vs canonical 5000-token jailbreak binary results.
Panel B: binary vs CSV-v2 yes and severity metrics over alpha.
Panel C: evaluator comparison graphic: dev vs holdout gap compression, plus a tiny error-set diagram showing v3 zero-solo-error property.
Panel D: one worked “refuse-then-educate” example with binary/SR/v2/v3 outputs side by side.

**Open checks that directly strengthen this section:**

* blind adjudication of disputed gold cases
* StrongREJECT rerun with gpt-4o
* new 10–15 hard-tail gold slice
* tiny C/S/V/T field audit before structured axes become headline claims.

---

### 8. Discussion — A four-stage framework for intervention science

**Target length:** 1–1.5 pages.

This is where you state the general lesson:

**measurement, localization, control, and externality are distinct empirical problems.**

That sentence should appear almost verbatim.

I would organize the discussion around five recommendations:

1. Do not treat held-out readout quality as a sufficient criterion for intervention target selection.
2. Validate control on the actual deployment surface, not just on a convenient proxy surface.
3. Always include matched negative controls when the intervention target is selected from many comparable components.
4. Use evaluator panels or robustness checks when the intervention changes style, refusal, verbosity, or disclaimer structure.
5. Report externalities and capability debt as first-class outcomes, not afterthoughts.

This connects naturally to the current literature. Quest-for-the-Right-Mediator, MIB, GCM, and recent causal-prediction work are all, in different ways, arguing that mediator choice and evaluation standard matter. Your paper gives a grounded case study showing why that shift is necessary in practice. ([arXiv][3])

---

### 9. Limitations

**Target length:** ~1 page.

Be explicit and unfancy here.

Main limitations:

* single model
* H-neuron selection is a good detector-selection baseline but not fully identical to the paper’s full selection rule
* D7 random-head control missing
* D7 capability / over-refusal battery missing
* bridge benchmark test split not yet used for a promoted candidate
* v3 hard-tail advantage not independently validated on new cases
* SAE layer coverage partial, though that does not rescue the null
* stochastic jailbreak generation complicates per-item flip interpretation.

**Table 6 — Limitations and what they affect**
Columns:

* limitation
* which claim it constrains
* whether it threatens main thesis or only scope/magnitude
* planned fix

This table is very good research taste.

---

### 10. Future work

**Target length:** ~0.75–1 page.

I would split future work into two lanes.

#### 10.1 Immediate paper-strengthening work

* D7 random-head control
* D7 capability / over-refusal battery
* seed-0 jailbreak control scoring with continuity stack
* new hard-tail gold labels for v3 vs SR
* tiny field-level C/S/V/T audit
* answer-token-level confound test for H-neuron readout.

#### 10.2 Next research program

This should not be “more global truth steering.”

The better next step is **bridge-grounded selective truthfulness intervention**: monitoring, reranking, abstention, or feature-based reward learning conditioned on answer risk. That sits naturally with current monitoring and feature-reward work: semantic entropy probes, entity-level hallucination detectors, and feature-based rewards all suggest that useful internal signals may be more effective as conditional supervision or control than as uniform global additive interventions.  ([arXiv][5])

A second, more mech-interp-heavy future direction is to repurpose D7-style selector comparison onto the bridge surface itself, and to test whether Gemma 3’s hybrid local/global attention architecture helps explain which heads are useful. Gemma 3 explicitly uses a 5:1 local-to-global attention ratio, which makes an architecture-aware head analysis unusually plausible here. ([arXiv][8])

## 3. Main-text asset plan

Here is the concrete asset list I would plan for the main paper.

### Figures

* **Fig 1:** conceptual map of measurement → localization → control → externality
* **Fig 2:** detection is real, interpretation is fragile
* **Fig 3:** H-neurons vs SAE, matched detection / divergent steering
* **Fig 4:** D7 selector comparison, with caveat panels
* **Fig 5:** ITI across surfaces, including bridge flip taxonomy and substitution examples
* **Fig 6:** measurement matters: truncation, binary vs CSV-v2, evaluator panel
* **Fig 7 (conditional):** H-neuron jailbreak specificity once seed-0 control is scored

### Tables

* **Table 1:** related-work positioning matrix
* **Table 2:** benchmark roles and measurement contract
* **Table 3:** claim ledger (earned / partial / not earned)
* **Table 4:** central synthesis matrix across methods
* **Table 5:** cross-surface outcome matrix
* **Table 6:** limitations and their claim impact

### Small visuals / boxes

* **Box A:** one worked bridge substitution example
* **Box B:** one worked evaluator disagreement example
* **Box C:** checklist for “what counts as evidence for a steering claim”

## 4. Appendix plan

The appendix should be big and useful, not decorative.

### Appendix A — Full methods and pipeline reproducibility

Model loading, intervention mechanics, manifests, decoding settings, judge prompts, compute/cost notes. Your local reproducibility story is a real asset here. 

### Appendix B — Full H-neuron replication and 4288 investigation

Include the six-analysis deep dive, not just a short note. This is publishable-quality skepticism.

### Appendix C — Negative controls

FaithEval and FalseQA random controls in full; jailbreak seed-0 control once scored.

### Appendix D — Bridge benchmark construction and grading

This is an important benchmark contribution in its own right and should be reusable later.

### Appendix E — Evaluator note

The full v2/v3/SR/binary story, dev/holdout split, error taxonomies, hard-tail future validation plan.

### Appendix F — D7 technical appendix

Pilot, trimmed full-500, template heterogeneity, degeneration, prompt-ID parity, why claim scope is benchmark-local today.

## 5. Where the paper sits in the current literature

This is the paragraph I would want the reader to walk away with:

Your paper sits **between** optimistic detector papers and optimistic steering papers, and it lands closest to the growing literature on mediator choice, causal selection, and evaluation standards. Detector work shows that truthfulness, hallucination, refusal, and harmfulness are often decodable from hidden states; steering work shows that internal interventions can sometimes shift behavior; newer work increasingly emphasizes that selector choice, mediator type, and causal grounding matter; and monitoring/reward work suggests that internal signals may often be more useful for conditional control than for static global steering. Your paper’s niche is to show, in one model and under one measurement contract, that the jump from **good readout** to **good control target** is empirically fragile. ([arXiv][1])

Two nuances matter a lot here.

First, you should **not** frame the paper as anti-SAE or anti-steering in general. Recent work shows SAE-based steering can work on refusal, though it often incurs performance or over-refusal costs; separate work shows SAE steering improves substantially when features are selected with output-aware criteria rather than raw activation heuristics. Your result is narrower and sharper: **matched held-out readout quality and straightforward feature selection did not reliably identify causally useful steering targets in this setting**. ([arXiv][9])

Second, your discussion of safety geometry should reflect the current nuance. The single refusal-direction literature is important, but newer work shows refusal-related behaviors are not exhausted by a single direction, and harmfulness can be encoded separately from refusal. That makes your measurement section more relevant, not less: refusal-heavy evaluators and refusal-heavy interpretations can miss the thing you actually care about. ([arXiv][7])

## 6. Deep-research agenda for the write-up

This is the literature work I would actually do over the next week or two.

### A. Build a related-work matrix in the repo

Create a file like `paper/related_work_matrix.csv` with these columns:

`paper, concept_family, mediator_type, selection_method, intervention_operator, target_surface, evaluator, negative_control, externality_check, strongest_claim, strongest_caveat, how_we_cite_it`

This forces discipline.

### B. Read closely, with specific questions

#### Cluster 1 — detector papers

Read H-Neurons, Universal Truthfulness Hyperplane, SEPs, Do Androids, and the entity-level long-form detector papers. For each, answer:

* What exactly is being predicted?
* On what surface: MC, short generation, long generation?
* Is success in-distribution or OOD?
* Is there any claim that predictive quality implies control quality? ([arXiv][1])

#### Cluster 2 — steering and mitigation papers

Read ITI, LITO, TruthFlow, WACK, refusal direction, SAE refusal steering, and the recent steering-mechanism case study. For each, answer:

* Is the intervention static or query-specific?
* Is it applied on MC, open-ended generation, or refusal surfaces?
* How are externalities measured?
* What is the actual control target: truthfulness, refusal, style, something else?
* What part of the success seems to come from selector quality vs operator quality? ([arXiv][2])

#### Cluster 3 — mediator and evaluation papers

Read Quest for the Right Mediator, MIB, GCM, and Internal Causal Mechanisms. For each, answer:

* What mediator type do they privilege?
* What standards do they use for evaluating localization or causal usefulness?
* Where does your paper’s evidence fit: mediator comparison, selector comparison, or intervention evaluation?
* Which terminology should you adopt so your paper sounds native to this literature rather than improvised? ([arXiv][3])

#### Cluster 4 — monitoring / reward alternatives

Read Features as Rewards, SEPs, and long-form entity monitoring. For each, answer:

* What is the deployment primitive: alerting, reranking, reward shaping, abstention?
* Why might these use internal signals better than static global steering?
* How can your bridge benchmark be repurposed to support this line of work? ([arXiv][10])

### C. Literature claims you must check before finalizing wording

1. Whether any paper already makes a claim very close to yours using cross-method evidence.
2. Whether any recent paper explicitly compares predictive selector quality to steering utility in a way that could make your “gap” sound less novel.
3. Whether recent steering papers have already normalized the exact measurement standards you want to recommend.
4. Whether “confident wrong substitution” has a close literature cousin under a different name.
5. Whether the newest refusal literature changes how strongly you should talk about refusal as a one-dimensional control knob. ([arXiv][11])

## 7. Empirical questions you still need to verify for the write-up

These are the paper-strengthening questions I would put in a `paper/open_questions.md` file, ranked by how much they affect claims.

### Tier 1 — can change the strength of main-text claims

1. **Does D7 survive the random-head control?**
   This determines whether D7 remains “benchmark-local comparator evidence” or becomes a much cleaner mechanistic selector result.

2. **Does H-neuron jailbreak specificity hold on seed-0 under the evaluator panel?**
   This determines whether you can discuss H-neuron jailbreak as specificity-backed, or only as a measurement-sensitive observation.

3. **Does CSV v3 beat StrongREJECT on new hard-tail cases, not just the calibrated ones?**
   This determines whether the evaluator note can claim more than robust qualitative edge.

### Tier 2 — mostly affects scope, not thesis

4. **What is D7’s capability / over-refusal debt?**
   This determines whether you can present it as practically promising or only benchmark-positive with visible weirdness.

5. **Are the C/S/V/T structured fields trustworthy enough to quote in the main text?**
   This determines whether v3 is only a primary binary evaluator or also a structured severity framework for headline use.

6. **How much of the H-neuron readout claim survives an answer-token-level confound test?**
   This will not overturn the intervention results, but it sharpens the detection section.

### Tier 3 — valuable appendix / future work

7. **Do suppressive neurons form a cleaner or more stable story than the positive H-neurons?** 
8. **Does the classifier transfer backward to the base model?** 
9. **Do D7 useful heads cluster in Gemma 3’s global-attention layers?** That could turn an architecture speculation into a grounded discussion point. ([arXiv][8])

## 8. Suggested repo structure for drafting

I would literally create:

```text
paper/
  00_title_abstract.md
  01_introduction.md
  02_literature_landscape.md
  03_experimental_program.md
  04_detection_fragility.md
  05_main_result_readout_vs_control.md
  06_main_result_surface_dependence.md
  07_measurement_changes_conclusions.md
  08_discussion.md
  09_limitations_future_work.md
  figures/
    fig1_framework/
    fig2_detection_fragility/
    fig3_hneurons_vs_sae/
    fig4_d7_selector/
    fig5_iti_surface_dependence/
    fig6_measurement/
  tables/
    table1_related_work.md
    table2_benchmark_roles.md
    table3_claim_ledger.md
    table4_central_synthesis.md
    table5_cross_surface_matrix.md
    table6_limitations.md
  appendices/
    app_a_methods.md
    app_b_hneuron_deep_dive.md
    app_c_negative_controls.md
    app_d_bridge_benchmark.md
    app_e_evaluator_note.md
    app_f_d7_details.md
  related_work_matrix.csv
  open_questions.md
```

That structure matches the paper’s real logic instead of the chronology of the sprint.

## 9. Bottom line

The flagship should read as a **comparative intervention-science paper**, not as an evaluator note and not as a D7 paper with extra sections bolted on.

The deepest, most defensible through-line is:

**readout quality, localization quality, intervention utility, and externality are different empirical questions.**

Your paper is strongest when it shows that explicitly, using three anchor case studies:

* **H-neurons vs SAE** for matched detection / divergent steering,
* **ITI across MC and bridge generation** for surface dependence and confident wrong substitution,
* **jailbreak mea

surement** for how evaluator choice changes the apparent effect.

And the literature now makes that framing *more* timely, not less: the field is simultaneously getting better at steering, getting more serious about mediator selection and evaluation, and getting more interested in monitoring or reward-style uses of internal signals. Your paper belongs exactly at that intersection. ([arXiv][11])

[1]: https://arxiv.org/abs/2512.01797 "https://arxiv.org/abs/2512.01797"
[2]: https://arxiv.org/abs/2306.03341 "https://arxiv.org/abs/2306.03341"
[3]: https://arxiv.org/abs/2408.01416 "https://arxiv.org/abs/2408.01416"
[4]: https://arxiv.org/abs/2504.13151 "https://arxiv.org/abs/2504.13151"
[5]: https://arxiv.org/abs/2406.15927 "https://arxiv.org/abs/2406.15927"
[6]: https://arxiv.org/abs/2602.16080 "https://arxiv.org/abs/2602.16080"
[7]: https://arxiv.org/abs/2406.11717 "https://arxiv.org/abs/2406.11717"
[8]: https://arxiv.org/abs/2503.19786 "https://arxiv.org/abs/2503.19786"
[9]: https://arxiv.org/abs/2411.11296 "https://arxiv.org/abs/2411.11296"
[10]: https://arxiv.org/pdf/2602.10067 "https://arxiv.org/pdf/2602.10067"
[11]: https://arxiv.org/abs/2604.08524 "https://arxiv.org/abs/2604.08524"

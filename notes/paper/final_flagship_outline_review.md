# Final Review and Refined Flagship Outline
## Detection Is Not Enough / Strong Readouts as Steering Evidence

### Recommended paper identity

**Recommended title:**  
**Detection Is Not Enough: Strong Readouts Are Insufficient Steering Evidence in Gemma-3-4B-IT**

**Recommended subtitle:**  
*A staged audit of measurement, localization, control, and externality in mechanistic intervention research.*

---

## 1. Bottom-line judgment

The current outline is already unusually strong. Its core scientific taste is good. It is asking the right question, using the right three anchor cases, and it already avoids the worst failure mode: turning the paper into a shallow “we found a better steering method” story.

The paper is strongest as a **comparative intervention-science case study**:

> In one model, under a shared evaluation contract, when does a strong predictive internal readout identify a useful intervention target, and where does that inference fail?

That framing is more accurate and more publishable than any of the following:
- “detectors fail”
- “SAEs fail”
- “we found the truthfulness heads”
- “we found the best evaluator”
- “D7 is the new method”

The most defensible flagship contribution is **not** the slogan-level claim. It is the combination of:

1. a **carefully matched cross-method comparison** showing that strong readouts do not reliably identify good steering targets;
2. a **surface-transfer / externality analysis** showing that local steering gains do not transfer cleanly across evaluation surfaces;
3. a **measurement audit** showing that reasonable evaluation choices can materially alter the inferred conclusion; and
4. an explicit **four-stage framework** — measurement, localization, control, externality — as the paper’s main methodological contribution.

That is the paper.

---

## 2. How I would weight the source pack

### Governing source hierarchy

Use the source pack in this order:

1. **Arbitration report** as the governor of novelty claims and claim boundaries.
2. **GPT report** as the cleanest novelty-safe wording ladder and the best anchor-level threat model.
3. **Opus report** as the best section architecture and paper-shaping source.
4. **Gemini report** as an aggressive edge-case scanner and reviewer-objection generator, but not as the primary guide for direct novelty judgments.

### Why

The arbitration report is the best document for avoiding overclaiming. It is correct that the weak claim is old background, that “evaluation choices matter” is not novel in the abstract, and that the strongest remaining novelty frontier is the exact empirical form plus the four-stage synthesis.

The GPT report is the strongest of the three literature reviews because it consistently distinguishes:
- established background,
- near-direct precedent,
- adjacent but not dispositive evidence,
- and genuinely open paper-specific claims.

The Opus report is structurally excellent. It is the best guide for how to build the paper. But it is more willing than the arbitration report to lean on specific quantitative interpretations and some literature applications that should remain qualified.

The Gemini report is useful, but it is the least disciplined. It repeatedly upgrades adjacent evidence into novelty-destroying or novelty-proving evidence. Some of the leads it surfaced are real and worth citing, but they should be used as **boundary-condition context**, not as primary anchors.

### My main critique of the arbiter

The arbiter is directionally right, but slightly conservative in two places:

- It is somewhat too restrictive about what counts as novelty in the measurement section. The generic point is old, but **showing conclusion reversals inside a mechanistic intervention setting** is still a meaningful contribution if presented as an applied audit rather than a discovery of evaluator dependence.
- It is probably a little too dismissive of frontier 2026 preprints as paper-shaping context. They should be downweighted, yes, but a few are still worth citing in related work as boundary conditions.

That said, if I had to pick one document to rule the framing, it would still be the arbitration report.

---

## 3. Core critique of the current outline

### What is already right

- The paper is framed as a methods / evaluation paper, not a method-improvement paper.
- The three strongest anchors are already identified:
  - matched-readout but divergent steering,
  - answer-selection versus generation externality,
  - evaluator / metric sensitivity.
- The outline already sees the need to separate measurement, localization, control, and externality.
- It already includes positive counterexamples, which is necessary for credibility.
- It already includes a limitations table, which is exactly the right scientific posture.

### What still needs correction

#### A. Move claim-boundary discipline earlier
The paper should state early:
- what it is claiming,
- what it is only suggesting,
- what it is explicitly **not** claiming.

This should happen before the reader gets deep into results. Right now the current outline has this material, but it should move even closer to the front.

#### B. Default D7 to appendix
D7 is useful supporting evidence, but it is not yet robust enough to sit beside the three main anchors. It has too many unresolved scope conditions:
- missing random-head control,
- incomplete full-500 probe branch,
- visible token-cap / quality debt,
- missing capability / over-refusal audit.

Recommendation: keep D7 in the main text only as a short supporting paragraph or half-page inset. The full comparison belongs in the appendix by default.

#### C. Promote bridge from “another benchmark” to “the sharpest behavioral diagnosis”
The bridge story is the most scientifically interesting externality result because it gives you a **specific failure mode**, not just a score drop:
- not mainly refusal,
- not mainly inactivity,
- active but indiscriminate redistribution over nearby candidates,
- often with reproducible wrong substitutions across intervention variants.

That is stronger than “generation got worse.”

#### D. Compress the detector-interpretation section
The “detection is real, interpretation is fragile” section is useful, but it should be short and surgical. It exists to establish:
- this is not a strawman,
- the signal is real,
- the interpretation is mixed,
- therefore strong readouts are not enough.

Do not let 4288, L1 instability, or length confounds become the emotional center of the paper.

#### E. Reduce main-text asset count
The current asset plan is rich but slightly overbuilt. Too many tables and “program-management” visuals can make the paper feel like an internal audit memo rather than a research paper.

Recommendation:
- keep **4 figures** and **3 main-text tables**;
- move the rest to appendix.

#### F. Keep related work lean and late
The literature section should be thin, high-load-bearing, and explicitly claim-safe. It should not attempt to narrate the whole 2024–2026 steering landscape. The paper earns its value through the comparative case study, not through literature breadth.

#### G. Clarify that the four stages are analytic, not ontological
Some evidence lines touch multiple stages at once. State clearly that:
> the four stages are a methodological decomposition for auditing intervention claims, not a claim that each experiment belongs to exactly one bucket.

That small clarification will prevent pedantic reviewer objections.

---

## 4. Hidden assumptions and biases to explicitly disarm

These are the main assumptions that a strong reviewer will attack.

### 4.1 Selection-on-the-story bias
The current program naturally foregrounds the most thesis-supporting cases. You need explicit positive counterexamples in the main narrative:
- H-neurons do work on some surfaces.
- ITI does improve answer selection.
- Some detector-linked interventions in the literature genuinely succeed.

The thesis must therefore be **heuristic unreliability**, not impossibility.

### 4.2 Construct bleed
The benchmarks do not all measure the same thing.

Do not use broad words like “truthfulness” or “safety” when the actual construct is narrower:
- TruthfulQA MC = answer selection under a constrained candidate set.
- Bridge = open-ended factual generation usefulness / factual precision under a specific benchmark design.
- FaithEval = contextual faithfulness under unanswerable / inconsistent / counterfactual conditions.
- JailbreakBench = harmful compliance under judge-dependent measurement.
- BioASQ = domain-specific factual QA portability.

This must be made explicit in an early construct map.

### 4.3 Matched AUROC is not matched causal opportunity
A “matched detection quality” claim is only as good as its control variables. If you say the readouts are matched, you need to specify:
- task,
- split,
- readout class,
- metric,
- layer budget,
- intervention operator,
- optimization budget,
- and what exactly is being matched.

Otherwise the match is only superficial.

### 4.4 Mediator unfairness
SAE features and H-neurons are not just different readouts. They differ in:
- representational granularity,
- selection pathway,
- operator compatibility,
- layer coverage,
- and likely causal accessibility.

That does not kill the comparison. It changes the interpretation:
> the paper is testing a real field heuristic under realistic mediator mismatch, not proving a pure theorem about information content.

### 4.5 Behavioral mechanism is not internal mechanism
“Confident wrong substitution” is an excellent behavioral diagnosis. It is not yet a circuit-level mechanism. Avoid language like:
- “we found the mechanism”
- “we identified the truthfulness circuit”
unless you actually have internal causal decomposition.

### 4.6 Evaluator disagreement may reflect construct pluralism, not only noise
If different evaluators disagree, that does not automatically mean one is wrong. Some disagreement is genuine construct mismatch:
- refusal,
- harmfulness,
- usefulness,
- disclaimer style,
- and partial compliance are not identical.

The contribution is not “our judge is correct.”  
It is: **the choice of evaluation construct changes the conclusion**, so it must be audited.

### 4.7 Bridge-specific artifacts may still contaminate interpretation
If the bridge result is central, you must explicitly separate:
- intervention-induced factual redistribution,
- benchmark annotation holes,
- and baseline retrieval / generation weaknesses.

The externality story is still strong. It just must not overclaim causal purity.

### 4.8 Single-model overgeneralization
Keep the model name in the title. The paper’s ambition should be methodological:
> this is a case study in how intervention claims break,
not
> this is a universal theorem about LLM steering.

### 4.9 Internal jargon inflation
Terms like D7, 4288, csv-v3, E0/E1/E2, etc. are useful inside the repo, but they dilute readability. Translate all internal names into scientific roles on first mention.

### 4.10 Rhetorical pull from the title
“Detection Is Not Enough” is good, but it can quietly pull the prose toward absolutism. Counteract that by repeatedly pairing every positive result with its scope condition.

---

## 5. Recommended claim boundary

### Safe headline claims
These are abstract-safe.

- Strong predictive readouts were **not sufficient evidence** for useful steering targets in Gemma-3-4B-IT.
- Matched or even perfect readout performance did **not reliably predict** intervention success.
- When interventions did work, their effects were often **surface-local** rather than generally transferable.
- Measurement choices **materially altered** the inferred intervention conclusion in at least one setting.

### Qualified but strong claims
These are valid with caveats in the body.

- H-neurons and SAE features can be matched on readout quality while diverging sharply in steering utility on the same evaluation surface.
- ITI improves answer selection but does not transfer reliably to open-ended factual generation in the evaluated settings.
- Bridge reveals a specific failure mode best described as **confident wrong substitution / redistribution over nearby factual candidates**.
- D7 suggests selector choice matters, but does not yet isolate why.

### Claims to avoid
Do not claim:
- detectors fail as a class,
- SAEs are poor steering mediators in general,
- causal selectors outperform correlational selectors in general,
- evaluator dependence is a novel discovery,
- bridge wrong-substitution is already established by literature,
- the findings generalize beyond this model family.

---

## 6. Recommended final outline

This is the version I would actually draft.

---

# 1. Introduction

### Goal
Frame the paper around a concrete research question:

> When does a strong predictive internal signal identify a good intervention target, and where does that inference break?

### What this section must do
- Open with the tempting heuristic: if a feature or neuron predicts a behavior well, it is tempting to steer through it.
- Acknowledge that this heuristic sometimes works.
- State the gap: the field still lacks a disciplined empirical audit of when readout quality fails as target-selection evidence.
- State the answer in one sentence: not reliably, and the failures occur at multiple empirical stages.
- State the theory of change: better intervention claims require stage-specific validation rather than probe-score optimism.

### Contribution framing
List 3 contributions only:
1. cross-method empirical dissociation between readout quality and steering utility;
2. externality analysis across answer-selection and open-ended generation;
3. staged audit framework: measurement, localization, control, externality.

### Box A — What this paper is / is not
Keep this in the first two pages.

---

# 2. Scope, constructs, and evaluation contract

This section should come **before** the detailed related work.

### 2.1 Paper identity
State explicitly:
- single-model comparative case study;
- not a new steering method paper;
- not a universal anti-steering paper;
- not an evaluator benchmark paper.

### 2.2 Construct map
Define what each benchmark measures and does not measure.

Required rows:
- TruthfulQA MC1 / MC2
- Bridge
- SimpleQA
- FaithEval / FalseQA
- JailbreakBench
- BioASQ

For each:
- construct,
- evaluator,
- primary metric,
- why included,
- main caution.

### 2.3 Evaluation contract
Define what counts as a credible steering claim in this paper:
- full-generation evaluation where relevant,
- pre-specified primary metrics,
- matched negative control or explicit reason why absent,
- externality / retained-capability check where relevant,
- per-example outputs for promoted claims,
- explicit claim boundary.

### 2.4 Claim ledger
Include a short safe / qualified / off-limits claim box.

---

# 3. Related work and novelty boundary

Keep this section lean.

### 3.1 What is already established
- decodability does not imply causal use;
- detection and steering can diverge;
- answer-selection can mislead about generative usefulness;
- evaluator / metric choices can change conclusions in safety evaluation.

### 3.2 What remains open
- matched cross-method comparison on the same model and behavioral surface;
- exact bridge-style wrong-substitution failure mode;
- measurement reversals inside this specific mechanistic intervention setting;
- integrated four-stage scaffold used as an auditing lens across multiple intervention families.

### 3.3 How this paper differs
This paper sits at the intersection of:
- mediator choice,
- selector choice,
- surface validity,
- and measurement discipline.

### Critical writing rule
Do not write “we are the first to show detectors fail as steering targets.”  
Write:
> prior work motivates the concern; this paper tests it in a matched cross-method case study and organizes the resulting failures into a four-stage audit framework.

---

# 4. Case study I — Strong readouts do not reliably identify good control targets

This is the center of gravity of the paper.

### 4.1 Readouts are real
Very brief non-strawman setup:
- detector signal exists,
- H-neuron replication is real,
- interpretation remains mixed.

Move deep detector analysis to appendix.

### 4.2 H-neurons vs SAE on FaithEval
This is the cleanest main result.

What to emphasize:
- readout match defined precisely,
- same surface,
- different steering outcome,
- random controls flat for neurons,
- SAE null survives both full-replacement and delta-only tests.

What not to imply:
- not that SAEs cannot steer,
- not that neuron selection is intrinsically superior,
- only that matched readout strength was insufficient target-selection evidence in this setting.

### 4.3 Probe-ranked heads on jailbreak
This is your second clean dissociation:
- perfect harmful/benign discrimination,
- no useful intervention.

This is valuable because it is difficult to dismiss as “weak readout.”

### 4.4 D7 as supporting comparator
Keep this short in the main text or send it to appendix.
Message:
- selector choice matters,
- but selector-specific mechanism remains unresolved.

### Key paragraph this section must earn
> Across two intervention families and two settings, strong or even perfect readouts did not reliably identify useful steering targets. The failure was not the absence of signal, but the unreliability of signal quality as a target-selection heuristic.

---

# 5. Case study II — Control is surface-local and can externalize

### 5.1 Positive results exist, but are task-local
Use H-neurons here as a positive counterexample with scope:
- works on some faithfulness / compliance surfaces,
- does not port everywhere.

This is where you earn credibility.

### 5.2 ITI improves answer selection but not open-ended generation
Be careful and exact:
- answer-selection improvement is real,
- transfer to open-ended factual generation is unreliable in your settings.

Do not oversimplify to “ITI fails.”

### 5.3 Bridge: the sharpest behavioral diagnosis
Promote this aggressively.

Core message:
- the intervention is active,
- it is not mainly refusal,
- it is not simply low-confidence abstention,
- it often redistributes probability mass toward nearby but wrong factual candidates,
- repeated wrong substitutions across variants suggest intervention-family-local structure.

Use “behavioral mechanism” or “failure mode diagnosis,” not “internal mechanism.”

### 5.4 Externality as first-class evidence
Make this explicit:
- externality is not an appendix issue;
- it is part of what determines whether a steering claim is scientifically credible.

### Key paragraph this section must earn
> Steering success on one evaluation surface does not establish usefulness on a nearby surface. In this program, the relevant externality was not only score degradation but a specific failure mode: confident substitution of semantically nearby false answers for correct ones.

---

# 6. Case study III — Measurement choices changed the scientific conclusion

This section must stay tied to steering claims, not drift into a generic evaluator paper.

### 6.1 Truncation changed the jailbreak story
State exactly what changed and why it matters.

### 6.2 Binary versus graded evaluation changed the apparent effect
Position this as:
- binary metrics hid meaningful variation,
- graded metrics recovered effects or altered ranking.

### 6.3 Evaluator dependence is part of the result
Not:
- “we found the best judge.”
But:
- interventions that alter refusal/disclaimer style are evaluator-sensitive,
- holdout validation compressed the apparent advantage of one judge,
- this is why judge monoculture is unsafe.

### 6.4 Seed-0 specificity and contamination bug
This belongs here because it demonstrates that pipeline details can alter claims.
Use it to:
- upgrade one claim from unscored to single-seed supported,
- and illustrate that measurement discipline is not cosmetic.

### 6.5 Established versus still-open
End the section with an explicit split:
- already established in your project,
- still robustness work.

### Key paragraph this section must earn
> In this setting, measurement choices were not clerical details; they changed what the project would have concluded about whether an intervention worked.

---

# 7. Synthesis — a four-stage audit framework for intervention claims

This is the paper’s field-facing contribution.

### Core claim
Measurement, localization, control, and externality are distinct empirical gates. Passing one does not license claims about the next.

### Structure this section around 5 recommendations
1. Do not treat held-out readout quality as sufficient target-selection evidence.
2. Validate on the behavioral surface you actually care about.
3. Use matched negative controls when selecting from many comparable components.
4. Treat evaluator disagreement as information when interventions alter style or refusal structure.
5. Report externality and quality debt as first-class outcomes.

### Theory of change paragraph
Explain the practical consequence:
- fewer inflated latent-control claims,
- better separation of monitoring versus control,
- better choices among abstention, reranking, training, and steering.

### Box D — Checklist for a credible steering claim
A compact box works well here.

---

# 8. Limitations and external validity

Do not spin this section.

### Must include
- single model;
- matched-readout control variable remains imperfect;
- H-neuron selection is a detector baseline, not a full reproduction of the originating paper’s rule;
- SAE layer coverage is partial;
- bridge promoted candidate not yet on final test split if still true at submission time;
- D7 missing controls / debt if kept central;
- jailbreak specificity only single-seed if still true at submission time;
- evaluator uncertainty remains live;
- answer-token confound audit incomplete if still true.

### Important framing
Separate:
- limits on the **thesis**,
- limits on the **scope**,
- limits on the **estimated effect size**.

That distinction matters.

---

# 9. Conclusion

Keep it short.

The conclusion should not say:
- “detectors fail,”
- “SAEs fail,”
- or “activation steering is unreliable.”

It should say something closer to:
> strong readouts are insufficient steering evidence; credible intervention claims require stage-specific validation of measurement, localization, control, and externality.

---

## 7. Recommended figures, tables, and visuals

The current asset plan is good but slightly too large for the main paper. This is the leaner version I would use.

### Main figure 1 — Four-stage scaffold
**Role:** conceptual map of the paper  
**Show:** the four stages with the three anchor case studies placed on the edges they stress-test  
**Important detail:** make clear the stages are analytic gates, not mutually exclusive experiment buckets

### Main figure 2 — Matched readouts, divergent control
**Role:** flagship empirical figure  
**Show:** matched readout quality for H-neurons and SAE features, then divergent intervention curves; include probe-head AUROC 1.0 versus null intervention as an inset or fourth panel  
**Goal:** let the reader see the paper’s main thesis in one glance

### Main figure 3 — Surface-local control and bridge failure mode
**Role:** externality centerpiece  
**Show:** answer-selection gains versus open-ended generation outcomes, plus bridge flip taxonomy and a few compact wrong-substitution examples  
**Goal:** demonstrate that the failure is not generic refusal but indiscriminate factual redistribution

### Main figure 4 — Measurement changes conclusions
**Role:** measurement centerpiece  
**Show:** truncation comparison, binary versus graded effect curve, evaluator compression on holdout, and contamination / schema bug inset  
**Goal:** show that evaluation choices were conclusion-relevant

### Main-text table 1 — Benchmark construct map
**Columns:**
- benchmark
- construct measured
- why included
- evaluator
- primary metric
- main interpretive caution

This must be in the main text.

### Main-text table 2 — Central synthesis matrix
**Rows:**
- H-neurons
- SAE features
- probe heads
- ITI
- D7 (optional; appendix by default)

**Columns:**
- readout quality
- steering result
- externality / debt
- claim status
- one-line lesson

This is the table a reviewer should be able to use to reconstruct the whole paper.

### Main-text table 3 — Limitations and claim impact
**Columns:**
- limitation
- which claim it constrains
- threatens thesis or scope?
- next fix

This is excellent scientific posture. Keep it.

### Appendix assets
Move these out of the main text unless space is abundant:
- related-work positioning matrix,
- measurement contract detail table,
- D7 full comparison figure,
- 4288 / detector deep dive,
- bridge construction and grading details,
- evaluator note.

### Boxes
Keep three small text boxes:
- **Box A:** what this paper is / is not
- **Box B:** one worked bridge substitution example
- **Box C:** one worked evaluator-disagreement example

That is enough. Do not add more.

---

## 8. Reviewer objections to preempt in the prose

### “Your own H-neuron result is a counterexample.”
Correct, and that is exactly why the thesis is heuristic unreliability rather than impossibility.

### “You are comparing apples and oranges across mediators.”
That is partly true, and it is the point. The field often uses readout quality as though mediator mismatch were negligible. This paper tests that heuristic under realistic mismatch and shows its limits.

### “Matched detection is underspecified.”
Preempt this by specifying every control variable and by being explicit about what is and is not matched.

### “The bridge story could be benchmark artifact.”
Acknowledge this directly and bound the claim to a behavioral failure mode in the evaluated setting.

### “This is a single-model paper.”
Yes. State that the contribution is methodological and case-study based, not universal.

### “This is really an evaluator paper.”
No. The evaluator section matters because it changes the scientific interpretation of an intervention claim. Keep measurement tied to the steering argument.

### “D7 is under-controlled.”
Agree in advance. That is why D7 is supporting evidence, not the headline case.

---

## 9. Recommended writing rules

1. Every section should state:
   - what claim it earns,
   - what it does not earn,
   - what would move that boundary.

2. Translate repo jargon on first use.

3. Use “surface” and “construct” precisely.

4. Pair every positive result with its scope condition.

5. Use “behavioral mechanism” for bridge unless you truly have circuit evidence.

6. Do not let any paragraph imply that generic probe quality is the paper’s novelty.

7. If a paragraph does not help answer
   > does strong detection reliably identify good steering targets?
   it probably belongs in appendix.

---

## 10. Recommended one-paragraph abstract skeleton

> Predictive internal signals are often treated as natural targets for steering large language models, but the reliability of this heuristic remains unclear. We study this question in Gemma-3-4B-IT by comparing multiple intervention families under a shared evaluation contract across contextual faithfulness, answer selection, open-ended factual generation, and jailbreak settings. We find repeated dissociations between measurement, localization, control, and externality: matched or even perfect readouts often fail to yield useful steering targets; interventions that help on one surface can fail or externalize on nearby surfaces; and evaluation choices can materially change the apparent conclusion. These results suggest that strong readouts are insufficient evidence for good steering targets and that credible mechanistic intervention claims require stage-specific validation.

---

## 11. Final judgment

I would keep the paper’s spine, but make four decisive changes:

1. **Move claim-boundary and construct discipline to the front.**
2. **Default D7 to appendix.**
3. **Promote bridge as the sharpest externality result.**
4. **Make the four-stage audit framework the paper’s real headline.**

If you do that, the paper stops looking like a collection of steering experiments and starts reading like what it actually is:

> a serious methodological case study in how latent-readout claims fail when measurement, localization, control, and externality are conflated.

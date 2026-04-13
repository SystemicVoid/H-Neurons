
# Refined Flagship Paper Outline
## *Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT*
### Working subtitle
*A comparative case study in separating measurement, localization, control, and externality in mechanistic intervention research.*

---

## 0. Executive verdict on the current outline

The current outline is already pointed in the right direction. It has the right flagship title, the right broad scientific object, and the right three anchor ingredients. Its best instincts are:

- it does **not** frame the project as “we built a better steering method”;
- it already sees that **SAE vs H-neurons**, **ITI MC vs generation**, and **jailbreak measurement** are the deepest evidence lines;
- it already tries to separate **measurement**, **localization**, **control**, and **externality**.

That said, it still needs a harder evidence hierarchy and a few structural corrections.

### What should change

1. **D7 should be demoted one notch in the paper structure.**  
   It is valuable, but not central enough yet to carry the main thesis. As of the latest project state, D7 is best used as supporting evidence that selector choice can matter on this benchmark surface, not as the flagship proof that causal selection beats correlational selection. The missing random-head control, incomplete full-500 probe branch, and visible token-cap / quality debt keep it below the strongest evidence tier.

2. **The bridge benchmark should be promoted.**  
   The bridge result is not just “generation failed again.” It reveals a concrete failure mode: **confident wrong substitution**, often with the same wrong entity reproduced across E0 and E1. That is deeper science than another benchmark delta. This deserves to be one of the paper’s main pillars.

3. **Measurement should be a co-equal scientific contribution, not a utility appendix.**  
   The paper should not read as “we found the best evaluator.” It should read as: **evaluation choices materially changed the scientific conclusion**, and the project caught both high-level measurement artifacts (truncation, construct mismatch, score compression) and low-level pipeline contamination. That is unusually strong research hygiene.

4. **The literature section should be thinner and later-written.**  
   You are already running parallel deep literature work. Good. Keep related work lean in the flagship draft until the matrix is complete. Right now the outline is mostly directionally right, but section 2 should be treated as **provisional framing**, not a section to over-optimize early.

5. **Future work should be shorter and less committal.**  
   The next project direction is promising, but the frontier is moving quickly. The paper should point toward a bridge-grounded selective intervention / monitoring agenda without pretending the downstream design is already settled.

### Biases and implicit assumptions to explicitly disarm

These are the places where a reviewer, mentor, or future you will push back.

- **Selection-on-the-story bias.** The current outline picks the cases that best support the thesis. Counter this by explicitly including positive counterexamples in the main text: H-neurons do work on some surfaces, and ITI does improve MC answer selection.
- **Construct bleed.** FaithEval, FalseQA, TruthfulQA MC, SimpleQA, bridge generation, and jailbreak do **not** all measure the same thing. Treat “truthfulness,” “harmfulness,” “refusal,” and “compliance” as related but distinct constructs.
- **Mediator unfairness.** “Matched AUROC” is not identical to “matched access to causal structure.” SAE layer coverage is partial, and the H-neuron set is a detector-selection baseline rather than a perfect reproduction of the original paper’s full selection rule. That does not kill the result, but it changes the scope of the claim.
- **Behavioral mechanism vs internal mechanism.** “Confident wrong substitution” is a strong **failure mode diagnosis**, not yet a circuit-level mechanism.
- **Single-seed overconfidence.** The new jailbreak control upgrades specificity status, but only to **single-seed supported**, not to “fully established.”
- **Evaluator realism.** No judge is ground truth. The contribution is not “v3 is correct”; it is that **judge dependence is scientifically important** when interventions alter refusal/disclaimer style.
- **Over-generalization from one model.** Keep the model name in the title and make the paper’s ambition methodological rather than universal.

---

## 1. Recommended paper identity

### Scientific object

This paper is best framed as a **comparative intervention-science case study**:

> In one model, under a shared measurement contract, when does a strong predictive internal readout identify a useful intervention target, and where does that inference fail?

That is sharper than:
- “we built a good evaluator,”
- “we found safety neurons,”
- “we beat H-neurons,”
- or “D7 works on jailbreak.”

### One-sentence answer

> In Gemma-3-4B-IT, **not reliably**: strong or even perfect readouts often failed to identify useful steering targets; when interventions worked they were narrow and surface-dependent; and measurement choices could materially change the apparent conclusion.

### Theory of change

The practical value of the paper is methodological:

> If researchers stop treating held-out probe quality as sufficient evidence for intervention target selection, they will design safer and more informative studies: task-local validation, matched negative controls, evaluator robustness checks, and externality reporting become standard rather than optional.

This maps very cleanly to BlueDot’s “gap + theory of change” criterion.

### Abstract framing sentence

Use something close to this in the abstract and introduction:

> We compare multiple mechanistic intervention families in Gemma-3-4B-IT under a shared evaluation contract to test whether predictive internal readouts reliably identify useful steering targets, and find repeated dissociations between measurement, localization, control, and externality.

### What the paper is **not**

Put a short box either at the end of the introduction or immediately after the abstract.

This paper is **not**:
- a new steering method paper;
- a universal anti-steering paper;
- an anti-SAE paper;
- an evaluator benchmark paper;
- a selector-specific D7 mechanism paper.

It **is**:
- a comparative empirical test of a common heuristic in mechanistic intervention work.

---

## 2. Claim ledger: what is safe, what is qualified, what is off-limits

This should appear early in the paper. It will improve credibility more than any extra adjective.

### Headline-safe

These can appear in the abstract and main conclusions.

- **Matched detection quality did not predict steering success.**
  - H-neurons and SAE features have matched FaithEval detection quality, but sharply different intervention outcomes.
- **Perfect discrimination did not guarantee useful intervention.**
  - Probe-ranked heads hit AUROC 1.0 on jailbreak and still did not steer the behavior.
- **Successful steering was narrow and surface-dependent.**
  - H-neurons work on some compliance-style surfaces but not BioASQ; ITI helps TruthfulQA MC but not open-ended generation.
- **Measurement choices changed the scientific conclusion.**
  - Truncation, binary-vs-graded judging, and evaluator calibration materially changed the jailbreak story.

### Qualified but usable

These belong in the paper with explicit caveats.

- **H-neuron jailbreak specificity is supported, but only at single-seed strength so far.**
- **D7 provides benchmark-local evidence that selector choice matters, but not yet selector-specific proof.**
- **v3 appears useful for this output regime, but its headline advantage over StrongREJECT shrank substantially on holdout.**
- ~~Bridge “confident wrong substitution” is a strong behavioral diagnosis, but the test split has not yet been used for a promoted candidate.~~ **Resolved 2026-04-13:** Bridge Phase 3 test-set results (n=500, CI excludes zero, p=0.0002) now headline-safe. See headline-safe list above.

### Do not claim

These should not appear, even if they sound tempting.

- “Detection and intervention are fundamentally different.”
- “Causal selection is better than correlational selection.”
- “CSV v3 clearly outperforms StrongREJECT.”
- “SAEs are bad steering mediators.”
- “We improved truthful generation.”
- “The results generalize beyond Gemma-3-4B-IT.”

---

## 3. Revised paper outline

The strongest structure is:

1. **Introduction**
2. **Related work and gap** *(lean, provisional)*
3. **Comparative program, construct map, and measurement contract**
4. **Detection is real, but detector interpretation is fragile**
5. **Anchor 1: Good readouts do not reliably identify control targets**
6. **Anchor 2: Even successful steering is narrow and surface-dependent**
7. **Anchor 3: Measurement choices change scientific conclusions**
8. **Supporting selector comparison: D7 and what it does / does not earn**
9. **Discussion: a four-stage framework for intervention science**
10. **Limitations**
11. **Future work** *(short and provisional)*
12. **Appendices**

This keeps the current paper’s best insight while giving it a cleaner evidence hierarchy.

---

## 4. Section-by-section blueprint

## 4.1 Introduction
**Target length:** 1.5–2 pages

### Purpose
State the actual research question, explain why it matters for AI safety and mechanistic interpretability, and tell the reader what kind of paper this is before they misclassify it.

### What the introduction must do

1. Open with the temptation:
   - Hidden states contain predictive information.
   - Steering via those signals has sometimes worked.
   - So it is tempting to treat the best predictor as a natural intervention target.

2. Acknowledge positive precedents:
   - H-neurons and other probe/direction papers show real latent signals.
   - ITI and related methods show that activation interventions can work.
   - Refusal-direction and feature-steering work show strong behavior changes are possible.

3. State the sharper gap:
   - What is missing is not “does steering ever work?”
   - What is missing is a comparative test of whether **held-out readout quality is a reliable heuristic for choosing steering targets**, under realistic generation and evaluation.

4. State the answer:
   - not reliably;
   - repeated dissociations across three anchor cases.

5. State why this matters:
   - the cost of getting this wrong is false confidence in “we found the safety neurons / truthfulness heads” style claims.

### Strong framing sentence
> This paper studies where the pipeline from internal measurement to practical control breaks.

### Figure placement
End of introduction.

### Figure 1 — Conceptual scaffold
A clean visual with four boxes and arrows:

**Measurement → Localization → Control → Externality**

Under each box, place the project’s evidence:
- Measurement: truncation audit; binary vs graded evaluation; holdout compression of v3 advantage; contamination bug.
- Localization: H-neuron replication; 4288 artifact; verbosity confound; SAE probe; probe-head AUROC 1.0.
- Control: H-neuron scaling; SAE null; ITI MC gains vs generation failure; D7 causal selector.
- Externality: BioASQ null; SimpleQA damage; bridge wrong-substitution; D7 token-cap debt.

**Design note:** visually distinguish “headline-safe” vs “supporting-caveated” evidence, e.g. solid vs dashed outlines.

### Common failure mode to avoid
Do **not** open by implying the paper disproves activation steering. That would be a strawman and the project’s own data refute it.

---

## 4.2 Related work and gap *(provisional; write late)*
**Target length:** 0.75–1 page in the flagship; longer matrix in appendix or repo.

This section should be intentionally lean for now. You are already running deep literature work, and the field moved materially in 2025–2026. The right move is to structure the literature section around **tensions**, not around long citation lists.

### The four tensions to organize around

#### A. Predictive signals are real, but their generality is limited
Use this cluster for:
- H-Neurons
- Universal Truthfulness Hyperplane
- Do Androids Know They’re Only Dreaming of Electric Sheep?
- Semantic Entropy Probes
- newer truth-direction / truth-spectrum papers

**Write-up goal:** your paper is not challenging the existence of signals. It challenges the inference from signal quality to control utility.

#### B. Steering can work, but operator / selector / surface matter
Use this cluster for:
- ITI
- NL-ITI
- LITO
- TruthFlow
- refusal-direction work
- SAE refusal steering
- output-aware SAE steering papers
- GCM

**Write-up goal:** your paper is not “steering fails.” It is “a convenient predictor is not automatically a good target.”

#### C. Mediator choice and causal grounding are now first-class concerns
Use this cluster for:
- Quest for the Right Mediator
- MIB
- Internal Causal Mechanisms
- GCM

**Write-up goal:** position the paper as a comparative case study that supports the field’s shift toward mediator/selector/evaluation discipline.

#### D. Internal signals may be more useful for monitoring or reward than for static global addition
Use this cluster for:
- SEPs
- Features as Rewards
- long-form / streaming hallucination detection
- latent harmfulness monitoring

**Write-up goal:** this is where the future-work bridge lands, but do not let it dominate the main paper.

### What section 2 should *not* do

- Do not overclaim novelty until the related-work matrix is complete.
- Do not imply nobody has questioned universality or one-direction stories; 2025–2026 papers already complicate those claims.
- Do not sound anti-SAE when the literature now explicitly shows SAE steering can improve if features are selected using output-aware criteria.
- Do not spend the page budget here. The paper is earned by sections 4–8, not by section 2.

### Table 1 — Related-work positioning matrix
Recommended columns:
- paper / method
- mediator type
- selector type
- operator type
- target behavior
- evaluation surface
- externality check
- what your paper adds

**Placement:** appendix by default; small version in main text if space allows.

---

## 4.3 Comparative program, construct map, and measurement contract
**Target length:** 1.25–1.5 pages

### Purpose
Make the paper feel like a coherent comparative program, not a bag of sprint results.

### Required subsections

#### 3.1 Intervention families
Briefly define the rows:
- H-neuron scaling
- SAE feature steering
- ITI truthfulness direction(s)
- probe-ranked heads
- gradient-ranked / causal heads
- evaluator layer as a measurement component

#### 3.2 Benchmark construct map
This is essential. Each benchmark must have a clearly stated role.

- **TruthfulQA MC1 / MC2** — answer-selection surfaces
- **TriviaQA bridge** — primary generation-usefulness surface
- **SimpleQA** — hard OOD generation stress test with low baseline headroom
- **FaithEval / FalseQA** — compliance / anti-compliance diagnostics
- **JailbreakBench** — harmful compliance surface with evaluator sensitivity
- **BioASQ** — scope test for H-neuron portability

#### 3.3 Measurement contract
State the paper’s “definition of done” for a steering claim:
- full-generation evaluation where relevant;
- primary metrics specified in advance;
- retained-capability / externality checks where needed;
- matched negative control or an explicit reason why not;
- per-example outputs and manifests;
- cross-benchmark consistency statement;
- explicit claim boundary.

#### 3.4 Evidence hierarchy
Use the project’s own status discipline:
- **headline-safe**
- **supporting but caveated**
- **not earned**

This section should make later caveats feel principled rather than apologetic.

### Table 2 — Benchmark role / construct table
Columns:
- benchmark
- construct measured
- why included
- evaluator
- primary metric
- key interpretive caution

### Table 3 — Measurement contract table
Columns:
- requirement
- why it matters
- satisfied by which experiments
- still incomplete

### Visual
A one-page “program map” is useful if you can keep it uncluttered:
interventions as rows, benchmarks as columns, with symbols for detector quality / steering result / control status / externality status.

### Critical writing rule
This is where you prevent **construct bleed**. Do not say “truthfulness benchmark” generically when what you mean is “answer-selection on TruthfulQA MC.”

---

## 4.4 Detection is real, but detector interpretation is fragile
**Target length:** 1–1.5 pages

### Purpose
Show that the paper is not anti-detection, while also showing why naive detector interpretation is dangerous.

### Suggested subsections

#### 4.1 H-neuron replication is real
- disjoint held-out performance is strong enough to establish a real signal.

#### 4.2 The 4288 story is an interpretability warning
- weight magnitude under one L1 setting should not be casually read as unique causal importance.

#### 4.3 Full-response readout is length-confounded
- response length dominates truth under some aggregations;
- detection signal is real, but what it encodes is more mixed than the headline suggests.

#### 4.4 Selection caveat
- your H-neuron set is a detector-selection baseline, not a row-for-row recovery of the paper’s full rule.

### Figure 2 — Detection is real, interpretation is fragile
Panels:
- **A:** H-neuron replication summary (paper vs your disjoint held-out result).
- **B:** C-sweep / 4288 rank instability.
- **C:** effect-size ratio for truth vs length under full-response aggregation.
- **D:** short note on what remains untested (answer-token-level confound).

### Strong sentence for this section
> The correct conclusion is not that the detector is invalid, but that detection quality and detector interpretation are separate problems.

### Common failure mode
Do not let this section become a detour. It is a setup section, not the paper’s center of gravity.

---

## 4.5 Anchor 1 — Good readouts do not reliably identify control targets
**Target length:** 2–2.5 pages  
**This is the center of gravity of the paper.**

### Core thesis of the section
A strong predictive signal can fail to identify a useful steering target even in close, apples-to-apples comparisons.

### Suggested subsections

#### 5.1 H-neurons vs SAE on FaithEval
This is your cleanest primary case.

What to emphasize:
- matched detection quality;
- sharply different intervention behavior;
- SAE null survives both full-replacement and delta-only tests;
- random controls stay flat for the neuron result.

What not to say:
- not “SAEs cannot steer”;
- say instead: **in this setting, matched readout quality plus straightforward feature selection did not identify causally useful steering targets**.

#### 5.2 Probe-ranked heads on jailbreak
Use this as the second clean dissociation:
- perfect harmful/benign discrimination;
- no useful intervention.

This is especially valuable because it is hard for a reviewer to dismiss.

#### 5.3 Where D7 fits
Use D7 only as supporting evidence inside or after this section:
- it strengthens the point that **selection criterion matters**;
- it does **not** yet cleanly show that causal ranking is the reason for the gain.

### Figure 3 — Matched readouts, divergent control
Panels:
- **A:** AUROC bars for H-neurons and SAE probe.
- **B:** H-neuron dose-response on FaithEval with random-control band.
- **C:** SAE full-replacement and delta-only null curves.
- **D:** compact probe-head AUROC 1.0 vs null-intervention panel.

This is probably the most important figure in the paper.

### Table 4 — Central synthesis matrix
Recommended rows:
- H-neurons
- SAE features
- probe heads
- causal heads
- ITI E0
- evaluator layer

Recommended columns:
- readout quality
- steering result
- control status
- externality / debt
- claim status
- one-line lesson

**This table should let a reader reconstruct the whole thesis in 30 seconds.**

### Key reviewer objection to preempt
> “But your own H-neuron row is a counterexample.”

Answer it directly in the prose:
- exactly;
- that is why the thesis is **unreliable heuristic**, not **never works**.

---

## 4.6 Anchor 2 — Even successful steering is narrow and surface-dependent
**Target length:** 2–2.5 pages

### Core thesis of the section
Interventions that genuinely move one benchmark may fail, or even become harmful, on nearby surfaces.

### Suggested subsections

#### 6.1 H-neurons are real positives, but task-local
- FaithEval / FalseQA positives;
- jailbreak graded effect;
- BioASQ null.

This blocks the simplistic “everything failed” reading while still supporting the broader thesis.

#### 6.2 ITI helps answer selection, not generation
- TruthfulQA MC is a real positive;
- SimpleQA is flat-to-harmful;
- bridge benchmark shows the same failure despite much higher headroom.

This is the key place to say the issue is not just low baseline accuracy.

#### 6.3 Bridge benchmark: the sharpest behavioral diagnosis
Promote this much more than the current outline does.

Core message:
- the failure is not mainly refusal;
- the intervention is active, but indiscriminate;
- correct answers are often replaced by semantically nearby but wrong entities;
- E1 reproducing the same wrong entities as E0 suggests the failure lives inside the tested intervention family, not just in one crude implementation.

Use “behavioral mechanism” or “failure mode diagnosis,” not “internal mechanism,” unless you have circuit evidence.

#### 6.4 Why externality is a first-class result
This section should explicitly say:
- externality is not a side note;
- it is one of the paper’s major scientific contributions.

### Figure 4 — Surface dependence of control
Panels:
- **A:** TruthfulQA MC gains for ITI and H-neurons.
- **B:** generation deltas on SimpleQA and bridge.
- **C:** flip taxonomy on bridge (stacked bars or Sankey): right→wrong, wrong→right, stable-right, stable-wrong, not-attempted.
- **D:** 4–5 concise substitution examples.

### Table 5 — Cross-surface outcome matrix
Rows:
- H-neurons
- ITI E0
- ITI E1
- ITI E2
- probe heads
- causal heads (jailbreak only)

Columns:
- TruthfulQA MC1
- MC2
- SimpleQA
- bridge
- FaithEval
- FalseQA
- jailbreak
- BioASQ
- main caveat

### Important writing correction
Do not say “truthfulness intervention fails by refusal.”  
Your own best evidence says the sharper story is **indiscriminate redistribution over nearby factual candidates**.

---

## 4.7 Anchor 3 — Measurement choices change scientific conclusions
**Target length:** 1.75–2 pages

### Core thesis of the section
Measurement choices are not housekeeping. They changed what the project would have concluded about intervention success.

### Suggested subsections

#### 7.1 Truncation changed the jailbreak story
- 256-token judging overstated the binary count effect;
- 5000-token rerun changed the interpretation.

#### 7.2 Binary judging washed out graded effects
- binary effect was weak / non-significant;
- graded evaluation recovered a stronger and more stable signal.

#### 7.3 Evaluator dependence is part of the result
Do **not** position this as “we found the best judge.”
Position it as:
- when interventions alter refusal/disclaimer structure, different evaluators see different things;
- holdout validation compressed the apparent v3 advantage;
- this is exactly why one should prefer evaluator panels / robustness checks over judge monoculture.

#### 7.4 Seed-0 specificity + contamination bug
This is where the newest audit belongs.

Use it to do two things at once:
1. upgrade the H-neuron jailbreak control from “unscored” to “single-seed supported”;
2. show that pipeline discipline mattered all the way down to schema/version handling.

This gives the measurement section teeth.

#### 7.5 Established result vs robustness layer
Be explicit about what is already established vs what is still robustness work.

**Established today**
- truncation artifact;
- binary vs graded conclusion shift;
- holdout compression of v3-SR gap;
- seed-0 v2 specificity;
- contamination bug and fix.

**Still a robustness layer**
- multi-seed jailbreak control;
- v3 / SR scoring on all seed-0 control data;
- field-level C/S/V/T audit;
- gpt-4o rerun for StrongREJECT if not already done.

### Figure 5 — Measurement matters
Panels:
- **A:** 256-token vs 5000-token jailbreak results.
- **B:** binary vs graded metrics over alpha.
- **C:** evaluator comparison: dev vs holdout compression.
- **D:** seed-0 control slope comparison plus contamination-bug inset.

The contamination inset can be very small but striking:
“167/171 borderline records silently remapped to yes under the wrong normalization path.”

### Box — Worked evaluator disagreement example
One “refuse-then-educate” or disclaimer-heavy output with labels from:
- binary
- StrongREJECT
- CSV-v2
- CSV-v3

### What not to do
- Do not let this become an evaluator paper.
- Do not headline structured v3 field claims until the mini audit is done.
- Do not hide that the seed-0 control is only one seed.

---

## 4.8 Supporting selector comparison — D7 and what it does / does not earn
**Target length:** 1–1.25 pages in main text, or move to appendix if space is tight.

### Purpose
Keep the good evidence, but keep it in proportion.

### Suggested subsections

#### 8.1 What D7 shows
- on this benchmark surface, a different selector found a more effective intervention than the current L1 comparator;
- the selected head sets are nearly disjoint;
- selector choice matters.

#### 8.2 What D7 does not yet show
- not yet selector-specific proof;
- not yet a safe deployment candidate;
- not yet a generally better intervention family.

#### 8.3 Why it still belongs
Because it:
- strengthens the general theme that predictive ranking is not enough;
- complements the probe-null result;
- points toward mediator / selector choice as a live research frontier.

### Figure 6 — D7 selector comparison *(conditional main-text figure)*
Panels:
- **A:** pilot: probe null vs causal positive.
- **B:** full-500: baseline vs L1 comparator vs causal.
- **C:** Jaccard overlap = 0.11.
- **D:** token-cap / quality debt inset.

### Writing rule
Use phrases like:
- “benchmark-local comparator evidence”
- “supporting but caveated”
- “selector specificity remains open”

Do **not** let the D7 section quietly re-center the paper.

---

## 4.9 Discussion — a four-stage framework for intervention science
**Target length:** 1–1.5 pages

This is where you make the paper feel field-useful.

### Core sentence
> Measurement, localization, control, and externality are distinct empirical problems.

Say this almost verbatim.

### Organize the discussion around five recommendations

1. **Do not treat held-out readout quality as sufficient target-selection evidence.**
2. **Validate on the actual behavioral surface you care about.**
3. **Use matched negative controls when targets are selected from many comparable components.**
4. **Treat evaluator disagreement as information, not nuisance, when interventions change output style.**
5. **Report externality and quality debt as first-class outcomes.**

### Theory-of-change paragraph
Spell out the practical consequence for AI safety:
- better study design;
- less false confidence in “latent safety control” claims;
- better decision about when to monitor, abstain, rerank, or train rather than globally steer.

### Optional small visual
A one-column box:
**Checklist for a credible steering claim**
- target selected how?
- negative control?
- deployment-surface evaluation?
- evaluator robustness?
- capability/externality check?
- claim boundary?

---

## 4.10 Limitations
**Target length:** ~1 page

Be plain here. No flourish, no spin.

### Must-include limitations

- single model;
- H-neuron selection is a detector baseline, not the original paper’s full selection rule;
- SAE layer coverage is partial;
- D7 random-head control missing;
- D7 capability / over-refusal battery missing;
- ~~bridge promoted candidate has not yet been run on the final test split;~~ **Resolved 2026-04-13:** Phase 3 test set (n=500) run; CI excludes zero, p=0.0002. Remaining bridge limitation: failure-mode coding is single-rater.
- jailbreak specificity is only scored for seed 0;
- evaluator dependence remains a live uncertainty;
- stochastic generation complicates per-item flip analysis;
- no completed answer-token confound audit for the H-neuron detector.

### Table 6 — Limitations and claim impact
Columns:
- limitation
- which claim it constrains
- threatens thesis or only scope/magnitude?
- concrete next fix

This table is excellent scientific taste. Keep it.

---

## 4.11 Future work *(short, provisional, and not overcommitted)*
**Target length:** 0.5–0.75 page

This should be tighter than the current outline.

### Lane A — paper-strengthening work
Only list items that change a claim boundary.

- score seeds 1–2 for jailbreak control;
- D7 random-head control **if and only if** D7 stays central after drafting;
- minimal capability / over-refusal check for D7 if retained centrally;
- blind adjudication of disputed evaluator cases;
- StrongREJECT gpt-4o rerun if still open;
- tiny field-level C/S/V/T audit;
- answer-token-level confound test for H-neuron detection.

### Lane B — next research program
State the direction, not a locked design.

Best next direction:
> **Bridge-grounded selective truthfulness intervention** — monitoring, abstention, reranking, or correction conditioned on answer risk, rather than another global additive truth vector.

Secondary mechanistic route:
> repurpose D7-style localization on **bridge correct-vs-wrong pairs**, i.e. on the behavior you actually care about.

### What to explicitly defer
- architecture-aware local/global head speculation;
- another broad search for “better truth vectors”;
- sprawling future-work menus.

The future section should feel disciplined, not wishful.

---

## 5. Figures, tables, and small visuals

Below is the asset plan I would actually build.

## Main figures

### Fig 1 — Conceptual scaffold: measurement → localization → control → externality
**Role:** the paper’s map  
**Section:** end of introduction  
**Contains:** the four stages, your intervention families, and evidence status (solid vs dashed)

### Fig 2 — Detection is real, interpretation is fragile
**Role:** establish non-strawman + caution  
**Section:** detection section  
**Contains:** H-neuron replication summary, 4288 instability, length confound

### Fig 3 — Matched detection, divergent control
**Role:** flagship figure  
**Section:** Anchor 1  
**Contains:** H-neuron vs SAE AUROC, H-neuron slope with control band, SAE full-replacement and delta-only nulls, probe-head null inset

### Fig 4 — Surface dependence of control
**Role:** externality centerpiece  
**Section:** Anchor 2  
**Contains:** MC gains, generation harms, bridge flip taxonomy, worked substitution examples

### Fig 5 — Measurement changes conclusions
**Role:** measurement centerpiece  
**Section:** Anchor 3  
**Contains:** truncation comparison, binary vs graded curves, evaluator holdout compression, seed-0 slope comparison + contamination inset

### Fig 6 — D7 selector comparison *(only if retained in main text)*
**Role:** supporting evidence  
**Section:** D7 section  
**Contains:** pilot probe null vs causal positive, full-500 comparison, Jaccard overlap, token-cap debt

## Main tables

### Table 1 — Related-work positioning matrix
Best in appendix unless very compact.

### Table 2 — Benchmark role / construct map
This should be in the main text.

### Table 3 — Measurement contract / definition of done
Main text or appendix, depending on space.

### Table 4 — Central synthesis matrix
This is the real backbone of the paper. Keep it in the main text.

### Table 5 — Cross-surface outcome matrix
Main text if readable, appendix otherwise.

### Table 6 — Limitations and claim impact
This can be short but should stay in the main text.

## Small visuals / boxes

### Box A — What this paper is not
Prevents misreading in the first two pages.

### Box B — Worked bridge substitution example
One good before/after pair is worth ten summary sentences.

### Box C — Worked evaluator disagreement example
Use one disclaimer-heavy jailbreak response.

### Box D — Steering-claim checklist
Can live in discussion or appendix.

---

## 6. The literature landscape: where the paper sits now

Because section 2 is provisional, the most useful thing here is not a polished narrative but a **positioning map**.

## 6.1 The live literature pressures your paper must reflect

### Pressure 1 — Predictive truth/hallucination signals are real, but universality is being qualified
Relevant threads:
- hidden-state truth / hallucination probing;
- H-neurons;
- truth-hyperplane work;
- SEPs;
- truth-direction limitations papers.

**Implication for your paper:**  
Do not sound like you are discovering that hidden states contain predictive signals. The live question is how portable, universal, and intervention-relevant those signals are.

### Pressure 2 — Steering results are getting better, but success depends on method details
Relevant threads:
- ITI and its descendants;
- non-linear / query-specific truthfulness intervention;
- refusal-direction work;
- SAE refusal steering;
- output-aware SAE feature selection;
- GCM.

**Implication for your paper:**  
Do not sound like “steering is bunk.”  
Your stronger, more defensible claim is:
> simple predictive readouts are an unreliable heuristic for target selection, and surface validity matters more than many papers acknowledge.

### Pressure 3 — Mediator and selector choice are now explicit research objects
Relevant threads:
- Quest for the Right Mediator;
- MIB;
- Internal Causal Mechanisms;
- GCM.

**Implication for your paper:**  
This is probably the literature neighborhood your paper belongs to most naturally. Your project is a naturalistic comparative case study about mediator choice, selector choice, evaluation standards, and claim boundaries.

### Pressure 4 — Safety geometry is more complex than “one refusal direction”
Relevant threads:
- harmfulness vs refusal separation;
- more-than-one refusal direction;
- task-conditioned over-refusal subspaces.

**Implication for your paper:**  
Your measurement section becomes *more* important under this literature, not less. If harmfulness, refusal, over-refusal, and style all have partly distinct geometry, then evaluator construct mismatch is a central problem.

### Pressure 5 — Monitoring / reward / conditional control is gaining credibility
Relevant threads:
- SEPs;
- long-form entity detectors;
- streaming detection;
- Features as Rewards.

**Implication for your paper:**  
The natural sequel to your results is not “another global truth vector,” but a more selective policy over risky generations.

---

## 6.2 The clearest niche statement

Here is the paragraph I would want the related-work section to earn:

> Detector papers show that truthfulness, hallucination, refusal, and harmfulness are often decodable from hidden states. Steering papers show that activation interventions can sometimes alter those behaviors. Recent benchmark and causal-mediation work argues that mediator choice, selector choice, and evaluation standards matter. This paper sits at the intersection: in one model, under one measurement contract, it tests whether strong predictive readouts reliably identify useful steering targets, and shows repeated dissociations between measurement, localization, control, and externality.

That is a much cleaner niche than “we have a better evaluator” or “we have a stronger truth intervention.”

---

## 6.3 Literature claims to avoid until deep research lands

Do **not** yet hard-claim any of the following:

- “No prior paper compares readout quality and steering utility.”
- “This is the first paper to show detector–steering dissociation.”
- “The bridge failure mode is novel.”
- “Recent steering work ignores evaluation rigor.”
- “The current frontier has moved decisively toward monitoring over steering.”

Some of these may end up being directionally true, but they are exactly the sort of statements that become embarrassing after one more arXiv search.

---

## 7. Appendix plan

Keep the appendix large and useful. It should carry the technical depth that would otherwise derail the main narrative.

### Appendix A — Full methods and reproducibility
- model loading
- hook mechanics
- decoding
- prompts
- judges
- manifests
- compute notes

### Appendix B — H-neuron replication and 4288 deep dive
This work is good enough to deserve its own technical appendix.

### Appendix C — Negative controls
- FaithEval random controls
- FalseQA random controls
- jailbreak seed-0 control now
- seeds 1–2 later if scored

### Appendix D — Bridge benchmark construction and grading
This benchmark is valuable beyond this paper.

### Appendix E — Measurement / evaluator note
- truncation artifact
- binary vs graded
- dev vs holdout
- calibration notes
- disputed-label taxonomy

### Appendix F — D7 technical appendix
- pilot
- trimmed full-500
- template heterogeneity
- degeneration / token-cap debt
- why claim scope is benchmark-local today

---

## 8. Suggested repo drafting structure

If you want a structure that matches the refined paper logic rather than sprint chronology:

```text
paper/
  00_title_abstract_claims.md
  01_introduction.md
  02_related_work_provisional.md
  03_program_constructs_measurement.md
  04_detection_fragility.md
  05_anchor1_readout_to_control.md
  06_anchor2_control_to_externality.md
  07_anchor3_measurement_to_conclusion.md
  08_supporting_d7.md
  09_discussion.md
  10_limitations.md
  11_future_work_provisional.md
  figures/
  tables/
  boxes/
  appendices/
  related_work_matrix.csv
  claim_ledger.md
```

The important thing is that the filesystem should follow the paper’s **logic**, not the historical order in which the experiments were run.

---

## 9. Concrete writing rules for the flagship draft

These are the rules I would use while drafting.

### Rule 1
If a paragraph does not answer:
> “Does good detection identify good intervention targets?”
it probably belongs in appendix or companion material.

### Rule 2
Every section should clearly state:
- what claim it earns,
- what it does **not** earn,
- what would change that boundary.

### Rule 3
Always translate internal repo jargon on first use.
- D1 → H-neuron scaling baseline
- D4 → ITI truthfulness baseline
- D7 → gradient-ranked causal head selector on jailbreak
- csv2_yes → strict harmfulness rate
- v3 → structured evaluator candidate for disclaimer-heavy outputs

### Rule 4
Use “surface” and “construct” precisely.
- answer selection is not the same as open-ended generation;
- compliance is not the same as truthfulness;
- harmfulness is not refusal.

### Rule 5
Whenever you use a positive result, pair it with its scope condition.
- H-neurons work **on these surfaces**
- ITI works **on answer selection**
- D7 works **on this benchmark, with visible debt**
- seed-0 specificity is **single-seed supported**

### Rule 6
When discussing bridge, use:
- “behavioral mechanism”
- “failure mode”
- “redistribution over nearby candidates”

Avoid saying:
- “we found the truthfulness circuit”
- “we identified the mechanism”  
unless you actually have circuit evidence.

---

## 10. Recommended one-paragraph abstract skeleton

You can use this as a drafting template.

> Predictive internal signals are often treated as natural targets for steering large language models, but the reliability of this heuristic remains unclear. We study this question in Gemma-3-4B-IT by comparing multiple intervention families under a shared evaluation contract across answer-selection, open-ended factual generation, and jailbreak settings. We find repeated dissociations between measurement, localization, control, and externality: matched or even perfect readouts often fail to yield useful steering targets; when interventions do work, their effects are narrow and surface-dependent; and evaluation choices can materially change the apparent conclusion. The clearest cases are a matched-detection but divergent-steering comparison between H-neurons and SAE features, an answer-selection / generation split for ITI explained by wrong-entity substitution on a locked bridge test set, and a jailbreak case where truncation and judge choice alter the inferred effect. These results suggest that held-out readout quality is an unreliable heuristic for intervention target selection, and that mechanistic safety claims require task-local validation, matched controls, evaluator robustness checks, and externality reporting.

---

## 11. Bottom line

The flagship should read as a **comparative intervention-science paper**.

Its deepest through-line is not:
- “our evaluator is better,”
- not “D7 is the new method,”
- and not even just “detection is not enough.”

It is:

> **Measurement, localization, control, and externality are different empirical problems, and this project is a case study in what goes wrong when researchers conflate them.**

If you build the paper around that sentence, with:
- **SAE vs H-neurons** as the readout→control break,
- **ITI MC vs bridge generation** as the control→externality break,
- **jailbreak measurement + seed-0 control** as the measurement→conclusion break,
- and **D7 / 4288** as disciplined supporting evidence,

then the write-up will finally match the actual research taste already present in the project.

---

## Source notes for drafting

### Local project documents
Use these as the internal source hierarchy while drafting:
1. `2026-04-11-strategic-assessment.md` *(rev. 2026-04-12)* — flagship framing, evidence hierarchy, claim boundaries
2. `2026-04-12-seed0-jailbreak-control-audit.md` — updated H-neuron jailbreak specificity and contamination bug
3. `act3-sprint.md` — current priorities, current completeness status
4. `intervention_findings.md` — H-neuron / SAE / negative-control summaries
5. `optimise-intervention-ac3.md` — historical strategy context, especially for closed branches and bridge rationale

### External literature clusters to refresh against
Use these clusters while filling in section 2:
- **Readout / detection:** H-Neurons (2512.01797), Universal Truthfulness Hyperplane (2407.08582), Do Androids… (2312.17249), SEPs (2406.15927), Testing the Limits of Truth Directions (2604.03754)
- **Steering / intervention:** ITI (2306.03341), NL-ITI (2403.18680), LITO (2405.00301), TruthFlow (2502.04556), Steering Refusal with SAEs (2411.11296), SAEs Are Good for Steering If You Select the Right Features (2505.20063), GCM (2602.16080)
- **Mediator / evaluation:** Quest for the Right Mediator (2408.01416), MIB (2504.13151), Internal Causal Mechanisms (2505.11770)
- **Safety geometry:** Harmfulness and Refusal Separately (2507.11878), More to Refusal than a Single Direction (2602.02132), Over-Refusal Subspaces (2603.27518)
- **Monitoring / supervision alternatives:** Features as Rewards (2602.10067), long-form/streaming hallucination detection papers

The point of this list is not to lock the literature section early. It is to stop it from sounding like it was frozen in mid-2024.

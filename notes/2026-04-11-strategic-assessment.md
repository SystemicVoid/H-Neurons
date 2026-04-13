# Strategic Assessment — Sunday Title Lock + 2-Week Artifact Window

> Date: 2026-04-11 (revised 2026-04-12 post mentor review)
> Purpose: Decision matrix, title recommendation, and execution strategy for BlueDot submission
> Inputs: Full project evidence base, GPT-5.4 pro analysis (myopic + high-vantage), independent Oracle review (2 rounds), BlueDot evaluation criteria, mentor strategic review (2026-04-12), holdout evaluator comparison
> Supersedes: earlier draft of this file (narrow framing)
> Revision note: 2026-04-12 edits integrate mentor review corrections — priority reordering (writing > experimentation), evidence confidence tiering, D7 decision gate, holdout impact on evaluator claims, three-artifact packaging, bridge elevation, ruler-consistency for seed-0 scoring
> Revision note (V2, 2026-04-12): deeper pass integrates V2 critique — four-stage scaffold (+ externality), evaluator panel over ruler-first, three anchor case studies, paper restructure (4-stage × 3-anchor), D7 default to supporting, bridge as second pillar, measurement promoted to co-equal claim

---

## 0. Executive Summary

**The project's strongest contribution is not a new evaluator, and not a single experiment. It is a cross-method empirical pattern:**

> In Gemma-3-4B-IT, held-out detection quality did not reliably identify useful intervention targets: matched or even perfect readouts often failed to steer behavior, while successful interventions were narrow in scope or required a different selection criterion.

Internally, this is a **case study in separating ~~three~~ four stages that are often conflated**: measurement of a feature → localization of that feature in the model → control of behavior via that feature → **externality** of that control across surfaces and tasks. The path from each stage to the next repeatedly breaks.

> **V2 refinement (2026-04-12):** The flagship should not be a broad collage. It should be a **broad claim built from three deep anchor case studies**: (1) SAE vs H-neurons on FaithEval — the localization→control break, (2) ITI MC vs bridge generation with wrong-entity substitution — the control→externality break, (3) jailbreak evaluation as measurement case study — the measurement→conclusion break. D7 and 4288 are powerful supporting evidence, not co-headliners.

This thesis integrates ~80% of the project's work — the 4288 L1 artifact, verbosity confound, SAE detect-but-don't-steer, probe-head AUROC 1.0 null, causal-head positive result, ITI MC-vs-generation mismatch, BioASQ scope delimiter, H-neuron specificity controls, and the measurement discipline that caught multiple evaluation artifacts. It speaks directly to mech interp methodology, not just benchmark engineering.

**Title to lock:** *Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT*

**Current bottleneck (post mentor review, 2026-04-12):** The project is now bottlenecked by **writing and evidence hierarchy**, not raw experimentation. The editable submission link buys time for artifacts to evolve, but not for clarity to emerge. The highest-ROI moves are a real skeleton write-up and the two core paper sections — experiments run in parallel but do not gate writing.

---

## 1. Why the Broader Thesis, Not the Narrow One

### What the narrow framing missed

The initial assessment was overly myopic — centering on CSV v3 validation and treating D7 as an isolated comparator result. This underweights:

- **D4 ITI beats H-neurons on TruthfulQA MC**: MC1 +6.3pp [+3.7, +8.9] vs +0.9pp [-1.7, +3.5]. Different intervention families win on different surfaces — even when both "detect" truthfulness.
- **The SAE dissociation is the cleanest single experiment**: AUROC 0.848 (matches H-neuron 0.843), yet zero steering under both full-replacement and delta-only architectures. This is apples-to-apples on the same benchmark, same model, same goal.
- **The probe-head AUROC 1.0 null**: Perfect discrimination, zero intervention. This is the most reviewer-resistant evidence in the entire project.
- **The 4288 L1 artifact as methodology critique**: Not just a curiosity — it shows that even *interpreting* detectors requires care, before we get to the intervention question.
- **Scope boundaries as positive evidence**: BioASQ null and jailbreak category heterogeneity show that even working interventions are task-local, not universal.

### Why the deadline constraints actually favor the broad framing

- **BlueDot locks only the title** — artifacts evolve over 2 weeks
- The broad framing uses **existing trusted data** (v2 surfaces, D7 full-500, ITI MC/bridge, SAE experiments, negative controls)
- It does not depend on fixing CSV v3 or running new GPU experiments
- It maps cleanly to all three BlueDot criteria (see §6)

---

## 2. The Evidence Inventory — Three Layers

### Layer 1: Detection works, but interpretation is fragile

| Evidence | What it shows | Status |
|---|---|---|
| H-neuron L1 classifier: AUROC 0.843, 76.5% disjoint accuracy | Real held-out discrimination signal from 38/348,160 neurons | ✅ Robust |
| 4288 L1 artifact: 0/6 analyses support dominance, enters model only at C=1.0 | L1 weight ≠ neuron importance. C=3.0 gets 80.5% with 219 neurons and 4288 at rank 5 | ✅ Robust |
| Verbosity confound: length dominates truth 3.7-16× in full-response readout | Detection AUROC partly reflects response-form correlations | ✅ Robust |
| SAE probe: AUROC 0.848 | Matches CETT probe quality — establishes detection equivalence for Layer 2 comparison | ✅ Robust |
| Probe heads (D7): AUROC 1.0 | Perfect harmful/benign discrimination | ✅ Robust |

**Takeaway:** Detection is real but detector interpretation requires discipline. This establishes we are not knocking down a strawman.

### Layer 2: Detection does not reliably predict steerability

| Comparison | Detection | Steering | Control | Lesson |
|---|---|---|---|---|
| **H-neurons vs SAE features on FaithEval** | AUROC 0.843 vs 0.848 | +6.3pp [4.2, 8.5] vs **null** (0.12pp/α) | H: 8-seed null; SAE: delta-only confirms null | Matched detection, divergent steering |
| **Probe heads vs causal heads on jailbreak** | AUROC 1.0 vs gradient-ranked | null at every alpha vs **-9.0pp [-12.2, -5.8]** | Probe: clean null; Causal: missing random-head | Perfect detection ≠ intervention |
| **ITI MC vs ITI generation** | Works for selection | +6.3pp MC1 vs **-7pp to -9pp on generation** | Random-head: no attempt collapse | Steering is surface-local, not universal |
| **H-neurons on BioASQ** | (same 38 neurons) | -0.06pp [-1.5, 1.4] | — | Even working targets are task-local |
| **Residual-stream truthfulness direction** | 71.5% separation | Null, then hard degeneration | — | Detection threshold exists below which steering fails |

**This table is the paper's central exhibit.**

### Layer 3: What happens when steering works — it's narrow

| Working intervention | Where it works | Where it fails | Mechanism insight |
|---|---|---|---|
| H-neuron scaling | FaithEval (+6.3pp), FalseQA (+4.8pp), jailbreak count (+7.6pp) + severity | BioASQ (null) | Over-compliance lever, not truthfulness |
| ITI E0 (TruthfulQA-sourced) | TruthfulQA MC (+6.3pp MC1, +7.49pp MC2) | SimpleQA generation (-31pp attempt rate at α=8), TriviaQA bridge (−5.8pp [−8.8, −3.0] test set) | Shifts probability mass among existing candidates; doesn't inject knowledge. **Bridge mechanism:** 30/43 (70%) flips are wrong-entity substitutions on held-out test set; dev showed E1 reproduces same wrong entities as E0 while only reducing rescue capacity — sharpest mechanistic diagnosis in the project |
| D7 causal heads | Jailbreak csv2_yes (-9.0pp vs baseline) | — (not tested on other benchmarks) | Gradient-based selection finds different components than AUROC (Jaccard 0.11) |
| ITI E0 vs H-neurons on TruthfulQA MC | ITI: +6.3pp MC1 | H-neurons: +0.9pp [-1.7, +3.5] MC1 (null) | Different methods win on different surfaces |

### Evidence confidence overlay (post mentor review, 2026-04-12)

The three layers above organize evidence thematically. Orthogonally, each result has a **confidence tier** that determines where it can appear in the paper:

**Headline-safe** (can lead a section, no caveats needed):
- H-neuron vs SAE dissociation on FaithEval — matched AUROC, divergent steering
- Probe-head AUROC 1.0 null — most reviewer-resistant evidence
- ITI MC improvement vs SimpleQA/TriviaQA bridge harm
- Bridge benchmark wrong-entity substitution analysis — 30/43 (70%) R2W flips on held-out test set (n=500, CI excludes zero); dev showed E0/E1 damage same questions
- Measurement artifacts — truncation, binary-judge blind spots
- H-neuron specificity controls on FaithEval and FalseQA

**Supporting but caveated** (useful in the paper, explicit caveats required):
- D7 pilot probe-null vs causal-positive — no random-head control yet; present as "benchmark-local evidence that an alternative selector can work on this surface"
- D7 full-500 causal result vs baseline and L1 comparator — 112/500 token-cap hits = visible quality debt
- H-neuron jailbreak CSV-v2 effect — seed-0 specificity confirmed (slope diff +2.77 pp/alpha [1.17, 4.42], p=0.013); seeds 1-2 pending for multi-seed robustness. See [seed-0 control audit](act3-reports/2026-04-12-seed0-jailbreak-control-audit.md). V3 binary slope non-significant (+0.46 [-1.46, +2.41]), but v3 severity-shift (substantive_compliance +2.00 [+0.11, +3.87]) is marginally significant; see [v2-v3 paired comparison](act3-reports/2026-04-13-v2-v3-paired-evaluator-comparison.md)
- CSV v3 zero-FP / zero-solo-error edge — holdout compressed gap vs StrongREJECT from 12.2 to 2.0pp; evaluator optimization is a supporting measurement problem, not a main scientific bottleneck

**Paper rule:** Do not lead a section or build a central argument on supporting-caveated evidence. These results strengthen claims anchored by headline-safe evidence.

> **V2 refinement (2026-04-12) — anchor case study designations:**
> The flagship is built from **three deep anchor case studies**, each demonstrating a break between adjacent stages of the four-stage scaffold:
>
> | Anchor | Stage break | Headline-safe evidence | Supporting evidence |
> |---|---|---|---|
> | **1. SAE vs H-neurons on FaithEval** | Localization → Control | Matched AUROC, divergent steering; delta-only rules out reconstruction | 4288 artifact, verbosity confound (localization fragility); D7 selector choice |
> | **2. ITI MC vs bridge generation** | Control → Externality | MC +6.3pp vs bridge −5.8pp [−8.8, −3.0] on locked test set; wrong-entity substitution (30/43) | H-neuron scope (FaithEval yes, BioASQ no); D4 vs D1 |
> | **3. Jailbreak evaluation** | Measurement → Conclusion | Truncation artifact; binary-judge blind spots; v2-v3 slope compression (2.30→0.46 on same outputs, mechanism transparent) | Holdout compresses v3-SR gap; severity-shift (substantive_compliance +2.00 pp/α, marginal); evaluator dependence is part of the result. See [v2-v3 paired comparison](act3-reports/2026-04-13-v2-v3-paired-evaluator-comparison.md) |
>
> D7 (selector-choice evidence) and 4288 (detector-interpretation evidence) are valuable supporting evidence within Anchors 1 and 3 respectively, not standalone pillars.

---

## 3. The Thesis — Precisely Stated

### Earned (safe to claim)

> Across multiple mechanistic methods in Gemma-3-4B-IT, predictive readout quality did not reliably identify components that could be successfully steered. When steering worked, it was narrow in scope or required a different selection criterion.

~~The internal framing is a three-stage decomposition~~ → **V2 (2026-04-12): four-stage decomposition.** **Measurement** (can we trust the evaluation?), **localization** (where in the model does the feature live?), **control** (can we steer behavior by intervening there?), and **externality** (does the control transfer across surfaces and tasks?). Each transition breaks independently:

- **Measurement → Localization:** Evaluation choices can reverse or compress conclusions (truncation artifact, binary-judge blind spots, v3-SR holdout gap collapse). You must trust your measurement before interpreting localization.
- **Localization → Control:** Good localization doesn't guarantee control (SAE AUROC 0.848 matches H-neuron 0.843, yet SAE steering is null; probe heads reach AUROC 1.0, null intervention). 4288 artifact and verbosity confound show even localization *interpretation* is fragile.
- **Control → Externality:** Successful control is surface-local, not universal (BioASQ null, ITI MC/generation split). The bridge benchmark reveals the failure is not primarily explained by refusal or grading loss but by **wrong-entity substitution** (30/43 R2W flips on held-out test set, CI excludes zero) — the intervention is active but indiscriminate.

### Not earned (do not claim)

- "Detection and intervention are fundamentally different" (too absolute; H-neurons are detector-selected and work)
- "Probe-selected components are non-causal" (we show they don't steer, not that they're acausal)
- "Causal selection is always better" (D7 random-head control is missing)

### How to handle the H-neuron counterexample

H-neuron scaling is a detector-selected target that works on compliance tasks. Frame as:

> Some detector-selected targets do steer behavior, but steering success is narrow and not predicted by detector quality alone. SAE features match H-neurons on detection yet fail entirely on steering. Perfect probe-head detection produces zero intervention effect.

---

## 4. Strongest Counter-Arguments and Defenses

### Counter 1: "You're mixing apples and oranges"

*"Different detector families, different intervention operators, different tasks. The null results may reflect bad implementation, not a detection-intervention dissociation."*

**Defense:** Lead with the two most apples-to-apples comparisons:
- SAE vs H-neurons on FaithEval: same benchmark, same goal, matched AUROC, different steering outcome
- Probe heads vs causal heads on jailbreak: same benchmark, same intervention family (head-level), different selector, different outcome

### Counter 2: "H-neurons are your own counterexample"

*"You show a detector-selected target that works. That undermines your thesis."*

**Defense:** The thesis is "detection quality is not a reliable *heuristic*," not "detection-selected targets never work." The SAE features match H-neuron detection quality and fail. The probe heads exceed it and fail. The H-neuron success is real but does not generalize to other detection methods or even to all tasks (BioASQ null).

### Counter 3: "D7 is missing its control"

*"Without the random-head control, D7 could be a generic perturbation effect."*

**Defense:** Present D7 as "valuable but provisional: strong benchmark-local evidence, not yet a mechanism-clean flagship pillar." The thesis survives even if D7 is fully demoted, because the SAE and probe-head nulls stand independently. **Decision gate (§7):** either commit to running the random-head control + capability mini-battery (making D7 a clean pillar), or demote D7 to supporting-caveated evidence and stop inflating it into a selector-specificity result. Seed 0 neuron-mode jailbreak control (2000 rows) can be scored in the next 2 weeks regardless.

### Counter 4: "Single model"

*"This is just Gemma-3-4B-IT. How do you know this generalizes?"*

**Defense:** Keep model name in title. Cite Gao et al.'s 6-model replication for the H-neuron compliance pattern. Acknowledge scope. One model studied deeply > many models studied shallowly for a course project.

---

## 5. Title, Packaging, and Structure

### Recommended title to lock

**Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT**

Backup options:
1. *From Readout to Control: Why Good Detectors Can Be Bad Steering Targets in Gemma-3-4B-IT*
2. *Good Detectors, Poor Levers: Detection–Intervention Dissociations in Gemma-3-4B-IT*

### Three concentric artifacts (post mentor review, 2026-04-12)

The work is best packaged as **three separable deliverables**, not a single monolithic paper:

1. **Flagship paper** — *Detection Is Not Enough* / the broad methods paper about intervention science. This is the submission. Structure below.
2. **Companion technical note** — Jailbreak measurement: truncation artifacts, binary-judge blind spots, and evaluator calibration discipline. Almost done; should be linkable as a supporting artifact. The flagship cites it for measurement rigor without re-explaining every audit inline.
3. **Next-project / fellowship proposal** — From global truth steering to **selective truthfulness intervention**. The bridge benchmark provides the right label taxonomy: correct answers, wrong-entity substitutions, evasion/denial, and drift. Current evidence says global truth directions are too blunt; the natural sequel is a **conditional policy**: monitor answer-risk around the first decode steps, then abstain, rerank, or correct selectively. Alternative mech-interp framing: repurpose D7-style causal localization onto bridge correct-vs-wrong pairs (pointing causal machinery at factual generation, not refusal). ~~Architecture-aware local/global head split~~ → defer until bridge-grounded approach either works or fails (V2). This is future work and proposal fuel, not a deadline-week pivot.

### Paper structure (~~3-claim~~ → 4-stage, 3-anchor design — V2, 2026-04-12)

> **Structural principle (V2):** The flagship is not a broad collage of results. It is a broad claim built from **three deep anchor case studies**, organized by the **four-stage scaffold** (measurement → localization → control → externality). Each anchor demonstrates a break between adjacent stages.

**§1. Introduction** — The practical assumption being tested: that features predicting harmful/false behavior make good intervention targets. The scaffold: **measurement, localization, control, and externality are separable empirical stages** that are routinely conflated. One paragraph setup + four-stage framing. (~300 words)

**§2. Detection is real, but interpretation is fragile** — *Localization stage.* Establishes the detection base. H-neuron replication (AUROC 0.843, 76.5%). 4288 L1 artifact as a methodological warning. Verbosity confound as a readout caveat. (~1 page, deep dive in appendix)

**§3. Anchor 1: Strong detection does not guarantee steerability** — *The localization→control break.* The paper's center of gravity.
- §3A: **SAE vs H-neurons on FaithEval** — the deepest anchor. Matched AUROC (0.848 vs 0.843), divergent steering (+6.3pp vs null). Delta-only SAE rules out reconstruction noise. Most apples-to-apples comparison in the project.
- §3B: **Probe heads vs causal heads on jailbreak** — independent confirmation. AUROC 1.0 null vs gradient-ranked -9.0pp, top-20 sets nearly disjoint (Jaccard 0.11).
- D7 as **supporting evidence** (selector choice matters on this surface), explicitly caveated: no random-head control, 112/500 token-cap hits, quality debt visible.
- Central synthesis table.
(~2 pages)

**§4. Anchor 2: When steering works, it is narrow and mechanistically revealing** — *The control→externality break.* Prevents "you only show negatives" critique.
- **Bridge benchmark wrong-entity substitution as mechanistic centerpiece** — 30/43 (70%) R2W flips on held-out test set (n=500, CI excludes zero) are wrong-entity substitutions (e.g., "Trainspotting" → "Slumdog Millionaire"); dev showed E0 and E1 produce the same wrong entities; E1 mainly reduces rescue capacity. ~~Failure is mainly refusal~~ → failure is **consistent with coarse reweighting toward nearby but wrong candidates**. This is evidence the mass-mean ITI family is the wrong lever for free-form factual generation. Sharpest mechanistic diagnosis in the project.
- H-neurons: compliance yes (FaithEval +6.3pp, FalseQA +4.8pp), BioASQ null. Even a working detector-selected target is task-local.
- ITI: MC selection yes (+6.3pp MC1), generation harmful. Different evaluation surfaces disagree about the same intervention.
- D4 ITI beats H-neurons on TruthfulQA MC — different methods win on different surfaces.
(~1.5 pages)

**§5. Anchor 3: Measurement choices change the scientific conclusion** — *The measurement→conclusion break.* ~~Brief summary; links to companion note~~ → **V2 (2026-04-12): promoted to co-equal claim.** Evaluator dependence is not a logistics problem to solve but **part of the scientific result** — scorer disagreement is evidence about measurement fragility.
- Truncation artifact caught and fixed (256→5000 tok changes the jailbreak story)
- Binary judge washes out signal that graded evaluation recovers (+3.0pp CI includes zero vs +7.6pp)
- CSV v3 holdout: gap vs StrongREJECT compressed from 12.2 to 2.0pp — evaluator flashiness is fragile
- Links to **companion technical note** for full audit (~0.5-1 page in flagship; companion carries the depth)

**§6. Implications for AI safety** — Cash out relevance.
- Detection is useful for monitoring/diagnosis
- Detection quality alone is a poor heuristic for control
- Safety-relevant interventions need: task-local validation, matched negative controls, capability checks, ideally causal selection criteria
- **Evaluator dependence means safety claims require multi-scorer robustness**, not trust in a single judge
(~0.5 page)

**§7. Limitations and future work** — Single model, missing D7 control, no capability battery, judge dependence.
- Future: ~~truthfulness monitoring/RLFR direction~~ → **bridge-grounded selective truthfulness intervention** (V2). Current evidence says global truth directions are too blunt; the sequel is a conditional policy: monitor answer-risk → abstain, rerank, or correct selectively. Bridge labels provide the right taxonomy.
- ~~Architecture-aware head selection (Gemma 3's 5:1 local/global attention)~~ → defer until bridge-grounded approach either works or fails.
(~0.5 page)

### What goes in appendix/supplement (for 2-week artifact evolution)
- Full 4288 deep dive (6 analyses)
- Per-template jailbreak heterogeneity
- E2 TriviaQA transfer synthesis
- Bridge benchmark methodology
- Refusal-overlap analysis
- CSV v3 smoke test full audit
- Seed 0 control scoring results (if completed)
- D7 full-500 token-cap and degeneration analysis

---

## 6. BlueDot Criteria Mapping

| Criterion | How the broad thesis addresses it | Why it scores well |
|---|---|---|
| **Clarity and presentation** | One question ("Does good detection identify good intervention targets?"), one answer ("No, not reliably"), organized as ~~3 claims~~ **4 stages × 3 anchor case studies** with a central synthesis table | Much clearer than a sprawling lab notebook; concrete examples (Terry Hall→Horace Panter, SAE null, probe AUROC 1.0 null) make abstract claims vivid |
| **Relevance to AI safety** | Directly addresses: how should researchers select targets for safety interventions? Shows the tempting heuristic (use your best detector) is unreliable. Theory of change: prevents false confidence in "we found the safety neurons" claims | Core mech interp safety methodology, not just benchmark engineering |
| **Project quality** | Multiple intervention methods compared fairly. Negative controls on two benchmarks. Evaluation artifacts caught and fixed. 4288 investigation shows care in interpretation. CSV v3 smoke test shows fiscal discipline. Honest limitations (missing D7 control, single model, no capability battery) | Demonstrates exactly the judgment and honesty the criteria reward |

---

## 7. Execution Plan — Title Lock + 2-Week Artifact Window

### Phase 0: Today (hours 0-6) — WRITING GATES EVERYTHING

> **Principle (post mentor review):** The project is bottlenecked by writing and evidence hierarchy, not by experimentation. A missing experiment weakens one pillar; a missing write-up wastes all of them. Experiments run in parallel but do not gate writing milestones.

1. **Lock the title** in the BlueDot form
2. **Submit a real skeleton write-up** — not a placeholder. Must contain: abstract-level thesis, central synthesis table (§2 Layer 2), earned/not-earned box (§3/§10), and caveat language for D7, v3, and H-neuron jailbreak. The point: demonstrate scientific judgment even before extra results land
3. **Launch seed 0 jailbreak control scoring** — ~~CSV-v2 first (ruler consistency)~~ → **V2 (2026-04-12): use a minimal evaluator panel**, because scorer dependence is itself part of the science. **v2 + v3 as the two load-bearing surfaces** (v2: historical claim metric; v3: best candidate for disclaimer-heavy outputs), **binary + StrongREJECT as sensitivity/legibility comparators**. If cost forces a cut, keep v2 + v3 and drop a simpler comparator first. Do **not** headline C/S/V/T field claims until the field audit is done
4. Lock the outline (§5 above) and figure/table list

### Phase 1: Weekend (hours 6-24) — CORE SECTIONS BEFORE MORE ANALYSIS

5. **Draft §3 (Anchor 1: localization→control break)** — this is the paper's center of gravity. Lead with SAE vs H-neuron on FaithEval as the deepest anchor; probe-head AUROC 1.0 null as independent confirmation. D7 as supporting evidence only
6. **Draft §4 (Anchor 2: control→externality break)** — bridge wrong-entity substitution as mechanistic centerpiece (not refusal — indiscriminate redistribution); ITI MC/generation split; H-neuron scope; D4-vs-D1 on TruthfulQA MC
7. Draft the central synthesis table
8. **Finish the companion measurement note** enough to be linkable — truncation, binary-judge blind spots, evaluator calibration discipline. The flagship cites it as supporting evidence rather than re-explaining every audit
9. **Blind adjudication of disputed labels** (V2: concurrent with scoring) — cheap, high-integrity manual work. Strengthens the evaluator companion note and demonstrates scientific judgment. Keep to the curated disputed set; do not sprawl into bulk re-adjudication
10. Retrieve and integrate seed 0 control results if available

### Phase 2: First 48 hours post-submission (days 2-3) — DECISION GATES

11. **D7 decision gate — make a binary choice:**
    > **V2 refinement (2026-04-12):** The default should be **supporting**, not neutral. D7 should not get another GPU day by default — it should earn centrality by remaining essential after you draft the actual paper sections. If the paper reads well without D7 as a headline, keep it supporting.
    - **If D7 stays central (must be justified by paper draft):** run random-head negative control (~1 GPU-day) + minimal capability/over-refusal battery (100 factual QA + 100 instruction-following, ~2h GPU). D7 becomes a mechanism-clean pillar
    - **If D7 is demoted (default):** present as benchmark-local comparator evidence with explicit caveats. Stop calling it a selector-specificity result. Thesis survives on SAE dissociation + probe null + ITI MC/gen mismatch
12. **StrongREJECT gpt-4o rerun** (~$5) — objection-removal, not center-of-gravity science. Holdout tells us it is unlikely to change anything dramatically
13. Draft §2 (Detection) and §5-7 (Measurement, Safety, Limitations)

### Phase 3: Week 1-2 (days 4-14) — POLISH AND ADJUDICATION

14. Build figures:
    - Fig 1: FaithEval SAE vs H-neuron dose-response (matched AUROC, divergent steering)
    - Fig 2: D7 jailbreak three-way comparison + probe null (if D7 stays central) or probe null standalone
    - Fig 3: ITI MC improvement vs bridge generation damage (with wrong-entity-substitution callout)
    - Fig 4: Synthesis table (central exhibit)
    - Fig 5: Seed 0 control slope vs H-neuron slope (if scored)
15. Build appendix/supplement with deep dives
16. **Targeted claim-boundary adjudication** — not sprawling. Focus on the subset of label disputes that could actually change a claim boundary (bulk label work already done in Phase 1 step 9)
17. **Optional**: small new hard-tail holdout of fresh refuse-then-educate cases for evaluator companion note (tests whether v3's calibrated edge transfers beyond tuning cases)
18. Final consistency pass — every claim checked against §3's "earned/not-earned" boundary

---

## 8. The 2-Week Experiment Priority Stack

Ranked by information-per-dollar for the paper. **Writing deliverables are included** because the bottleneck is evidence hierarchy, not raw data.

| Priority | Deliverable | Cost | What it buys |
|---|---|---|---|
| 0 | **Real skeleton write-up** (abstract, synthesis table, earned/not-earned box, caveat language) | ~4h writing | Demonstrates scientific judgment at submission; forces evidence hierarchy decisions now |
| 1 | **Score seed 0 jailbreak control** — ~~CSV-v2 first~~ → **evaluator panel: v2 + v3 load-bearing, binary + SR as comparators** (V2). Scorer dependence is part of the science, not a logistics problem | ~$15-25 API, ~3-6h | H-neuron jailbreak specificity; evaluator robustness section built from panel disagreement |
| 1b | **Blind adjudication of disputed labels** (concurrent with scoring) — curated disputed set only, not sprawling (V2) | ~2-3h manual | Strengthens evaluator companion note; demonstrates scientific judgment; clarifies gold-label anomalies |
| 2 | **Draft two core flagship sections** (Anchor 1: localization→control; Anchor 2: control→externality) | ~8h writing | The sections that actually earn the broad thesis |
| 3 | **Finish companion measurement note** (truncation, binary-judge blind spots, calibration discipline) | ~3h writing | Linkable artifact that signals rigor; flagship cites rather than re-explains |
| 4 | D7 decision gate: random-head negative control (**only if paper draft requires D7 central — default: supporting**, V2) | ~1 GPU-day | If null: D7 becomes mechanism-clean flagship pillar. If not null: still informative, but demote. Do not let D7 eat writing time by default |
| 5 | StrongREJECT gpt-4o rerun | ~$5 API | Objection-removal; holdout says unlikely to change conclusions |
| 6 | Minimal capability battery for D7 causal α=4.0 (only if D7 stays central) | ~2h GPU | "Safer without breaking the model" — or honest reporting of what breaks |
| 7 | Score seed 1 control (if generation finishes) | ~$10-15 API | Second specificity seed with error bars |
| 8 | XSTest over-refusal check | ~$5 API | One safety dimension for D7 |

### What NOT to do in the 2-week window
- Full CSV v3 redeploy on all main data (~$200, not validated; holdout compressed gap to 2.0pp — not justified)
- Full random-control alpha sweep (3 seeds × 4 alphas × 500 prompts — overkill)
- ~~StrongREJECT comparison code + scoring (nice-to-have but not thesis-critical)~~ **Updated 2026-04-12:** StrongREJECT comparison is done ([4-way report](act3-reports/2026-04-12-4way-evaluator-comparison.md)). Remaining: re-run with gpt-4o to remove judge-model confound (~$5).
- Rescore D1/D3 with v3 as main pre-deadline move (v3 is a robustness layer, not the primary ruler)
- Reopen E1/E2/E3, chooser work, or general truth-vector search (closed by own stop conditions)
- IFEval-only capability story
- Wrapper tags / pivot claims as headline
- Pivot to truthfulness-monitoring project (best *next* program, not deadline-week identity crisis)

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Narrative sprawl from integrating too many results | High | Loses clarity (criterion 1) | One question, one table, three claims. If it doesn't answer "does detection predict steerability?", cut it |
| Overclaiming D7 specificity | Medium | Loses credibility (criterion 3) | Always pair with "benchmark-local, random-head control missing" |
| H-neuron positive result undermines thesis | Low | Thesis already accounts for it | Frame as "detection quality is not a reliable heuristic" not "detection never works" |
| Internal jargon leaking | High | Clarity failure | Translation guide: D7→gradient-based causal intervention, csv2_yes→strict harmfulness rate, L1→original neuron-scaling method, AUROC→held-out discrimination quality |
| 2-week experiments don't finish | Medium | Weaker paper | Thesis stands on SAE dissociation + probe null + ITI MC/gen mismatch even without D7 control |
| Reviewer says "single model" | Medium | Scope limitation | Model name in title; cite Gao et al.'s 6-model replication; depth > breadth argument |
| Writing bottleneck masks strong evidence | High | Submission is uncompelling or sprawling despite strong data | Skeleton today, two core sections before any new analysis. Editable link buys time but not clarity |
| Holdout compresses evaluator advantage | Realized | v3-vs-SR gap dropped from 12.2 to 2.0pp on holdout; evaluator story becomes supporting, not central | Evaluator optimization is a measurement sub-problem; do not position v3 as a main contribution |

---

## 10. What Each Piece of Evidence Earns

### Earned claims (safe to make)

| Claim | Evidence | Pillars |
|---|---|---|
| Matched detection quality does not predict steering success | SAE AUROC 0.848 = H-neuron 0.843, but SAE steering null; delta-only rules out reconstruction artifact | SAE dissociation |
| Perfect detection does not guarantee intervention | Probe heads AUROC 1.0, null at every alpha on jailbreak | Probe null |
| A different selection criterion (gradient vs AUROC) identifies different components that steer differently | Causal heads -9.0pp vs probe null, Jaccard 0.11 | D7 comparison |
| Successful steering is narrow: works on some tasks, not others | H-neurons: FaithEval/FalseQA/jailbreak yes, BioASQ no. ITI: MC yes, generation no | Scope evidence |
| ITI generation failure is wrong-entity substitution, not refusal — sharpest mechanistic finding; evidence the mass-mean ITI family is the wrong lever for free-form factual generation | 30/43 (70%) R2W flips on held-out test set are wrong-entity substitutions; dev showed E1 reproduces same wrong entities as E0 while only reducing rescue capacity. Failure is **consistent with coarse reweighting toward nearby but wrong candidates**, not mainly refusal or timid abstention (V2) | Bridge benchmark |
| ITI beats H-neurons on MC selection; different methods win on different surfaces | ITI +6.3pp MC1 vs H-neuron +0.9pp (null) on same benchmark | D1 vs D4 |
| Evaluation method changes conclusions | Binary judge null on jailbreak (+3.0pp, CI includes zero); graded v2 significant (+7.6pp); truncation artifact caught | Measurement discipline |
| L1 weight magnitude is unreliable for neuron importance | 4288: 0/6 analyses support dominance; absent at C≤0.3 | 4288 investigation |
| Validation discipline prevents expensive mistakes | CSV v3 smoke test: 37.5% gold agreement caught before $200 scaling | Smoke test audit |

### Partially earned (need qualification)

| Claim | Gap | Safe phrasing |
|---|---|---|
| Gradient-based selection is specifically responsible for D7 gain | Missing random-head control | "Benchmark-local comparator result; selector specificity open" |
| H-neuron jailbreak effect is neuron-specific | Seeds 1-2 control unscored | "Specificity confirmed on FaithEval, FalseQA, and jailbreak (seed-0 random-neuron control, p=0.013). Multi-seed robustness pending." See [seed-0 control audit](act3-reports/2026-04-12-seed0-jailbreak-control-audit.md) |
| D7 causal intervention is safe for practical deployment | No capability battery, 22.4% token-cap rate | "Promising mitigation with visible quality debt" |

### Partially earned (updated 2026-04-12 — post 4-way comparison, error taxonomy, and holdout)

| Claim | Gap | Safe phrasing |
|---|---|---|
| CSV2 v3 is a useful evaluator for this response regime | Holdout compressed v3-vs-StrongREJECT gap from 12.2 to 2.0pp; all four evaluators >90% on holdout; holdout cannot validate superiority on new hard refuse-then-comply cases; evaluator optimization is now a supporting measurement problem, not the main scientific bottleneck | "On holdout, v3 is directionally best but the gap is small; evaluator choice matters most for hard-tail cases where v3's calibration edge is untested on fresh data" |
| Intervention-aware evaluation changes the scientific conclusion | StrongREJECT uses gpt-4o-mini (confound); construct-mismatch analysis is analytical but gpt-4o rerun would strengthen | "Standard refusal-weighted evaluation undercounts harmful substance in disclaimer-heavy intervention outputs; a structured judge calibrated for refuse-then-comply reveals effects that other evaluators miss" |

> See [4-way evaluator comparison](act3-reports/2026-04-12-4way-evaluator-comparison.md) and [error taxonomy](../error-taxonomy-v3-fn-binary-fp.md) for full evidence.

### Not earned (do not claim)

| Claim | Why not |
|---|---|
| Detection and intervention are fundamentally different | H-neurons are detector-selected and work on compliance |
| CSV v3 is a broadly validated evaluator | Dev-set overlap; single model/response family; binary validation only |
| Results generalize beyond Gemma-3-4B-IT | Single model |
| Causal selection is always superior to correlational selection | One benchmark, no control |
| CSV v3 clearly outperforms StrongREJECT | Holdout compressed gap from 12.2 to 2.0pp; all four evaluators >90% on holdout; holdout cannot validate superiority on new hard cases |
| ITI reveals the truthfulness circuit | What we have is narrower: mass-mean ITI helps MC selection but harms generation, likely by reshuffling probability mass among nearby candidates rather than adding knowledge |
| We improved truthful generation | We did not. What we improved is understanding of *why* the tested mass-mean ITI family fails on generation: indiscriminate redistribution over nearby factual candidates, not refusal. The bridge's wrong-entity-substitution finding is a diagnostic, not a fix (V2) |

---

## 11. Bottom Line

**The paper that deserves to exist is not "our evaluator is better" or "our intervention beats the incumbent." It is:**

> *Across multiple mechanistic methods on one model, we found that strong detection performance was not a reliable guide to intervention effectiveness. Features that perfectly predict behavior often fail to control it. When control works, it is narrow — succeeding on one evaluation surface while failing on another. This has direct implications for AI safety: researchers who find "safety neurons" or "truthfulness directions" should not assume they have found intervention targets.*

**That is a real contribution. It tells people how not to fool themselves.**

The 4288 artifact, the SAE null, the probe null, the bridge wrong-entity-substitution mechanism, the ITI MC/generation split, the causal-head positive, and the measurement discipline — these are not scattered side quests. They are **one story about what goes wrong when researchers conflate measurement, localization, control, and externality.**

> **V2 (2026-04-12):** The strongest signal from a first project is not "I found a flashy effect." It is: **"I knew which effects were real, which ones were artifacts, which branches to kill, and what the next sharper question should be."** This project already contains that story. The job now is to make the write-up reflect it.

Lock the title. Write the three anchors. Tell the four-stage story.

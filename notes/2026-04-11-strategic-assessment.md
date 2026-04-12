# Strategic Assessment — Sunday Title Lock + 2-Week Artifact Window

> Date: 2026-04-11
> Purpose: Decision matrix, title recommendation, and execution strategy for BlueDot submission
> Inputs: Full project evidence base, GPT-5.4 pro analysis (myopic + high-vantage), independent Oracle review (2 rounds), BlueDot evaluation criteria
> Supersedes: earlier draft of this file (narrow framing)

---

## 0. Executive Summary

**The project's strongest contribution is not a new evaluator, and not a single experiment. It is a cross-method empirical pattern:**

> In Gemma-3-4B-IT, held-out detection quality did not reliably identify useful intervention targets: matched or even perfect readouts often failed to steer behavior, while successful interventions were narrow in scope or required a different selection criterion.

This thesis integrates ~80% of the project's work — the 4288 L1 artifact, verbosity confound, SAE detect-but-don't-steer, probe-head AUROC 1.0 null, causal-head positive result, ITI MC-vs-generation mismatch, BioASQ scope delimiter, H-neuron specificity controls, and the measurement discipline that caught multiple evaluation artifacts. It speaks directly to mech interp methodology, not just benchmark engineering.

**Title to lock:** *Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT*

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
| ITI E0 (TruthfulQA-sourced) | TruthfulQA MC (+6.3pp MC1, +7.49pp MC2) | SimpleQA generation (-31pp attempt rate at α=8), TriviaQA bridge (-7pp/-9pp) | Shifts probability mass among existing candidates; doesn't inject knowledge |
| D7 causal heads | Jailbreak csv2_yes (-9.0pp vs baseline) | — (not tested on other benchmarks) | Gradient-based selection finds different components than AUROC (Jaccard 0.11) |
| ITI E0 vs H-neurons on TruthfulQA MC | ITI: +6.3pp MC1 | H-neurons: +0.9pp [-1.7, +3.5] MC1 (null) | Different methods win on different surfaces |

---

## 3. The Thesis — Precisely Stated

### Earned (safe to claim)

> Across multiple mechanistic methods in Gemma-3-4B-IT, predictive readout quality did not reliably identify components that could be successfully steered. When steering worked, it was narrow in scope or required a different selection criterion.

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

**Defense:** Present D7 as "promising benchmark-local evidence that selector choice matters" with the explicit limitation. The thesis survives even if D7 is demoted, because the SAE and probe-head nulls stand independently. Note: seed 0 neuron-mode jailbreak control (2000 rows) can be scored in the next 2 weeks, and the D7 random-head control can be run during the artifact window.

### Counter 4: "Single model"

*"This is just Gemma-3-4B-IT. How do you know this generalizes?"*

**Defense:** Keep model name in title. Cite Gao et al.'s 6-model replication for the H-neuron compliance pattern. Acknowledge scope. One model studied deeply > many models studied shallowly for a course project.

---

## 5. Title and Structure

### Recommended title to lock

**Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT**

Backup options:
1. *From Readout to Control: Why Good Detectors Can Be Bad Steering Targets in Gemma-3-4B-IT*
2. *Good Detectors, Poor Levers: Detection–Intervention Dissociations in Gemma-3-4B-IT*

### Paper structure (3-claim design)

**§1. Introduction** — The practical assumption being tested: that features predicting harmful/false behavior make good intervention targets. One paragraph setup. (~300 words)

**§2. Detection is real, but interpretation is fragile** — Establishes the detection base. H-neuron replication (AUROC 0.843, 76.5%). 4288 L1 artifact as a methodological warning. Verbosity confound as a readout caveat. (~1 page, deep dive in appendix)

**§3. Main Result I: Strong detection does not guarantee steerability** — The paper's center of gravity.
- §3A: H-neurons vs SAE features on FaithEval (matched AUROC, one steers, one doesn't)
- §3B: Probe heads vs causal heads on jailbreak (AUROC 1.0 null vs gradient-ranked success)
- Central synthesis table (§2 above)
(~2 pages)

**§4. Main Result II: When steering works, it is narrow and surface-dependent** — Prevents "you only show negatives" critique.
- H-neurons: compliance yes, BioASQ no. Count vs severity decomposition.
- ITI: MC selection yes, generation no. Confident substitution mechanism.
- D4 ITI beats H-neurons on MC — different methods win on different surfaces.
(~1.5 pages)

**§5. Measurement discipline as methodology contribution** — Brief.
- Truncation artifact caught and fixed (256→5000 tok changes the jailbreak story)
- CSV v3 smoke test caught calibration failure before $200 waste
- Binary judge washes out signal that graded evaluation recovers
(~0.5 page)

**§6. Implications for AI safety** — Cash out relevance.
- Detection is useful for monitoring/diagnosis
- Detection quality alone is a poor heuristic for control
- Safety-relevant interventions need: task-local validation, matched negative controls, capability checks, ideally causal selection criteria
(~0.5 page)

**§7. Limitations and future work** — Single model, missing D7 control, no capability battery, judge dependence, CSV v3 dev-set validation only (holdout pending). Future: truthfulness monitoring/RLFR direction, architecture-aware head selection (Gemma 3's 5:1 local/global attention). (~0.5 page)

### What goes in appendix/supplement (for 2-week artifact evolution)
- Full 4288 deep dive (6 analyses)
- Per-template jailbreak heterogeneity
- E2 TriviaQA transfer synthesis
- Bridge benchmark methodology
- Refusal-overlap analysis
- CSV v3 smoke test full audit
- Seed 0 control scoring results (if completed)

---

## 6. BlueDot Criteria Mapping

| Criterion | How the broad thesis addresses it | Why it scores well |
|---|---|---|
| **Clarity and presentation** | One question ("Does good detection identify good intervention targets?"), one answer ("No, not reliably"), organized as 3 claims with a central table | Much clearer than a sprawling lab notebook; concrete examples (Terry Hall→Horace Panter, SAE null, probe AUROC 1.0 null) make abstract claims vivid |
| **Relevance to AI safety** | Directly addresses: how should researchers select targets for safety interventions? Shows the tempting heuristic (use your best detector) is unreliable. Theory of change: prevents false confidence in "we found the safety neurons" claims | Core mech interp safety methodology, not just benchmark engineering |
| **Project quality** | Multiple intervention methods compared fairly. Negative controls on two benchmarks. Evaluation artifacts caught and fixed. 4288 investigation shows care in interpretation. CSV v3 smoke test shows fiscal discipline. Honest limitations (missing D7 control, single model, no capability battery) | Demonstrates exactly the judgment and honesty the criteria reward |

---

## 7. Execution Plan — Title Lock + 2-Week Artifact Window

### Phase 0: Today (hours 0-4)
1. **Lock the title** in the BlueDot form
2. Submit seed 0 jailbreak control for batch scoring (binary judge + v2) — ~$10-15
3. Lock the outline (§5 above) and figure/table list

### Phase 1: Weekend (hours 4-24)
4. Draft §3 (Main Result I) — this is the paper's center of gravity
5. Draft §4 (Main Result II) — ITI + H-neuron scope results
6. Draft the central synthesis table
7. Retrieve and integrate seed 0 control results if available

### Phase 2: Week 1 (days 2-7)
8. Draft §2 (Detection) and §5-7 (Measurement, Safety, Limitations)
9. Build figures:
   - Fig 1: FaithEval SAE vs H-neuron dose-response (matched AUROC, divergent steering)
   - Fig 2: D7 jailbreak three-way comparison + probe null
   - Fig 3: ITI MC improvement vs bridge generation damage
   - Fig 4: Synthesis table (central exhibit)
   - Fig 5: Seed 0 control slope vs H-neuron slope (if scored)
10. **Optional high-value experiment**: D7 random-head negative control (~1 GPU-day)
11. **Optional**: patch CSV v3 + re-run 30-case smoke test for appendix

### Phase 3: Week 2 (days 8-14)
12. Polish, record video demo if needed
13. Build appendix/supplement with deep dives
14. Integrate any additional control results
15. Final consistency pass — every claim checked against §3's "earned/not earned" boundary

---

## 8. The 2-Week Experiment Priority Stack

Ranked by information-per-dollar for the paper:

| Priority | Experiment | Cost | What it buys |
|---|---|---|---|
| 1 | Score seed 0 jailbreak control (already generated) | ~$10-15 API, ~3-6h | H-neuron jailbreak specificity evidence |
| 2 | D7 random-head negative control | ~1 GPU-day | If null: D7 causal-head claim becomes much stronger. If not null: still informative, thesis survives on SAE/probe pillars |
| 3 | Minimal capability battery for D7 causal α=4.0 | ~2h GPU (100 factual QA + 100 instruction-following) | "Safer without breaking the model" — or honest reporting of what breaks |
| 4 | Patch CSV v3 + re-run 30-case smoke test | ~1h code + ~$0.30 API | Appendix showing validation discipline + potential fix |
| 5 | Score seed 1 control (if generation finishes) | ~$10-15 API | Second specificity seed with error bars |
| 6 | XSTest over-refusal check | ~$5 API | One safety dimension for D7 |

### What NOT to do in the 2-week window
- Full CSV v3 redeploy on all main data (~$200, not validated)
- Full random-control alpha sweep (3 seeds × 4 alphas × 500 prompts — overkill)
- ~~StrongREJECT comparison code + scoring (nice-to-have but not thesis-critical)~~ **Updated 2026-04-12:** StrongREJECT comparison is done ([4-way report](act3-reports/2026-04-12-4way-evaluator-comparison.md)). Remaining: re-run with gpt-4o to remove judge-model confound (~$5).
- IFEval-only capability story
- Wrapper tags / pivot claims as headline

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

---

## 10. What Each Piece of Evidence Earns

### Earned claims (safe to make)

| Claim | Evidence | Pillars |
|---|---|---|
| Matched detection quality does not predict steering success | SAE AUROC 0.848 = H-neuron 0.843, but SAE steering null; delta-only rules out reconstruction artifact | SAE dissociation |
| Perfect detection does not guarantee intervention | Probe heads AUROC 1.0, null at every alpha on jailbreak | Probe null |
| A different selection criterion (gradient vs AUROC) identifies different components that steer differently | Causal heads -9.0pp vs probe null, Jaccard 0.11 | D7 comparison |
| Successful steering is narrow: works on some tasks, not others | H-neurons: FaithEval/FalseQA/jailbreak yes, BioASQ no. ITI: MC yes, generation no | Scope evidence |
| ITI generation failure is confident substitution, not refusal | 5/10 flips are wrong-entity substitutions; identical across E0/E1 | Bridge benchmark |
| ITI beats H-neurons on MC selection; different methods win on different surfaces | ITI +6.3pp MC1 vs H-neuron +0.9pp (null) on same benchmark | D1 vs D4 |
| Evaluation method changes conclusions | Binary judge null on jailbreak (+3.0pp, CI includes zero); graded v2 significant (+7.6pp); truncation artifact caught | Measurement discipline |
| L1 weight magnitude is unreliable for neuron importance | 4288: 0/6 analyses support dominance; absent at C≤0.3 | 4288 investigation |
| Validation discipline prevents expensive mistakes | CSV v3 smoke test: 37.5% gold agreement caught before $200 scaling | Smoke test audit |

### Partially earned (need qualification)

| Claim | Gap | Safe phrasing |
|---|---|---|
| Gradient-based selection is specifically responsible for D7 gain | Missing random-head control | "Benchmark-local comparator result; selector specificity open" |
| H-neuron jailbreak effect is neuron-specific | Seed 0 control unscored | "Specificity confirmed on FaithEval/FalseQA; jailbreak specificity pending" |
| D7 causal intervention is safe for practical deployment | No capability battery, 22.4% token-cap rate | "Promising mitigation with visible quality debt" |

### Partially earned (updated 2026-04-12 — post 4-way comparison + error taxonomy)

| Claim | Gap | Safe phrasing |
|---|---|---|
| CSV2 v3 is a useful evaluator for this response regime | 24/74 gold rows overlap with calibration set; holdout comparison pending; binary-level validation only (C/S/V/T field-level audit pending) | "On this curated Gemma-3-4B intervention gold set, v3 is the best-performing evaluator tested; true holdout confirmation is the highest-priority follow-up" |
| Intervention-aware evaluation changes the scientific conclusion | StrongREJECT uses gpt-4o-mini (confound); construct-mismatch analysis is analytical but gpt-4o rerun would strengthen | "Standard refusal-weighted evaluation undercounts harmful substance in disclaimer-heavy intervention outputs; a structured judge calibrated for refuse-then-comply reveals effects that other evaluators miss" |

> See [4-way evaluator comparison](act3-reports/2026-04-12-4way-evaluator-comparison.md) and [error taxonomy](../error-taxonomy-v3-fn-binary-fp.md) for full evidence.

### Not earned (do not claim)

| Claim | Why not |
|---|---|
| Detection and intervention are fundamentally different | H-neurons are detector-selected and work on compliance |
| CSV v3 is a broadly validated evaluator | Dev-set overlap; single model/response family; binary validation only |
| Results generalize beyond Gemma-3-4B-IT | Single model |
| Causal selection is always superior to correlational selection | One benchmark, no control |

---

## 11. Bottom Line

**The paper that deserves to exist is not "our evaluator is better" or "our intervention beats the incumbent." It is:**

> *Across multiple mechanistic methods on one model, we found that strong detection performance was not a reliable guide to intervention effectiveness. Features that perfectly predict behavior often fail to control it. When control works, it is narrow — succeeding on one evaluation surface while failing on another. This has direct implications for AI safety: researchers who find "safety neurons" or "truthfulness directions" should not assume they have found intervention targets.*

**That is a real contribution. It tells people how not to fool themselves.**

The 4288 artifact, the SAE null, the probe null, the ITI MC/generation split, the causal-head positive, and the measurement discipline — these are not scattered side quests. They are **one story about the gap between reading a model and controlling it.**

Lock the title. Write the table. Tell the story.

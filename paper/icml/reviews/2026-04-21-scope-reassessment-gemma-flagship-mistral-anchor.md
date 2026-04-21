# Anchor Ranking and Evidence Map: Gemma Flagship, Narrow Mistral Path, and What the Repo Actually Supports

**Date:** 2026-04-21  
**Status:** internal strategy memo, not paper prose  
**Purpose:** replace repeated internal shorthand with a fresh source-hierarchy check on which results should carry paper weight now, which should not, and why.

## Executive Ranking

Current anchor ranking, by paper-facing value today:

| Rank | Result | Current anchor value | Why |
|---|---|---|---|
| 1 | TriviaQA bridge externality / wrong-entity substitution | **Best current anchor** | Held-out test set, paired deltas, uncertainty, frozen one-shot protocol, and later dual-rated failure coding. |
| 2 | FaithEval H-neuron vs SAE | **Strong anchor, but not pristine** | Real localization-to-control dissociation with strong random-control specificity, but still cross-representational and partly detection-confounded. |
| 3 | Jailbreak measurement-sensitivity | **Good anchor for measurement, not for selector quality** | Strongest evidence that verdicts depend on truncation, ruler structure, and scoring surface. |
| 4 | D7 same-ruler causal-vs-baseline/random | **Good supporting evidence; not flagship** | Narrow same-ruler effect is real, but selector closure is incomplete and ruler drift remains large. |
| 5 | Narrow Mistral extension path | **Not a current anchor; best next strengthening path** | Preserved artifacts and clear next step exist, but no held-out Mistral evaluation or intervention evidence yet. |

Short version:

- **Most promising anchors now:** bridge, FaithEval, jailbreak-measurement.
- **Most important correction:** stop treating FaithEval as obviously the cleanest anchor just because that phrasing was repeated.
- **Mistral remains the best next extension path**, but it is still extension planning, not completed evidence.

## How This Memo Weighs Evidence

Claims are ranked using this source hierarchy:

1. Machine-readable artifacts, raw JSON/JSONL, metrics summaries
2. Row-level or adversarial audits tied to on-disk data
3. Pipeline reports that separate what is proved from what is not
4. [paper/icml/main.tex](../../paper/icml/main.tex) only as evidence of current framing
5. Older strategic memos only as claim history, never as support

Working rule:

> A repeated phrase is not evidence. A claim only gets promoted if it still looks strong when re-anchored to the highest available source layer.

## Ground Truth on Current Paper Framing

The draft currently frames FaithEval as the anchor result and uses bridge and jailbreak as the other two gate failures:

- [paper/icml/main.tex](../../paper/icml/main.tex) lines around the localization section call the FaithEval comparable-readout result “the anchor result.”
- The same draft already gives the bridge result strong paper space and treats jailbreak as a measurement-discipline case study.

That framing is not indefensible, but the repo evidence now supports a more careful internal ranking:

- **Bridge** is probably the cleanest single held-out behavioral result.
- **FaithEval** is still strong, but cleaner as a warning sign than as a theorem.
- **Jailbreak** is strongest as a measurement result.
- **D7** is stronger than older “too dirty” shorthand, but still not a flagship selector claim.

## Candidate Evidence Map

### 1. TriviaQA Bridge Externality / Wrong-Entity Substitution

**Current anchor value:** best current anchor

**Primary sources**

- [notes/act3-reports/2026-04-13-bridge-phase3-test-results.md](../../../notes/act3-reports/2026-04-13-bridge-phase3-test-results.md)
- [paper/icml/reports/2026-04-21-bridge-irr-review.md](../reports/2026-04-21-bridge-irr-review.md)
- [`data/judge_validation/bridge_irr/bridge_irr_summary.json`](../../../data/judge_validation/bridge_irr/bridge_irr_summary.json)
- [`data/judge_validation/bridge_irr/adjudicated_labels.jsonl`](../../../data/judge_validation/bridge_irr/adjudicated_labels.jsonl)
- [`data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/results.json`](../../../data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/results.json)
- [`data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/audit_stats.json`](../../../data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/audit_stats.json)

**Established**

- On the held-out `n=500` bridge test set, paper-faithful E0 ITI reduces adjudicated accuracy by **-5.8 pp [-8.8, -3.0]** with **McNemar p=0.0002**.  
  Source: [2026-04-13-bridge-phase3-test-results.md](../../../notes/act3-reports/2026-04-13-bridge-phase3-test-results.md)
- The failure-mode coding is no longer single-rater-only: the discordant-case IRR artifact shows **55/57 agreement**, **96.5%** raw agreement, **κ = 0.90**, **AC1 = 0.96**.  
  Source: [`bridge_irr_summary.json`](../../../data/judge_validation/bridge_irr/bridge_irr_summary.json)
- Among right-to-wrong flips, **31/43 = 72.1% [57.3, 83.3]** are adjudicated wrong-entity substitution, with **0/43** formal refusal.  
  Source: [`bridge_irr_summary.json`](../../../data/judge_validation/bridge_irr/bridge_irr_summary.json)

**Interpretive but plausible**

- The intervention appears to redistribute probability mass within a factual neighborhood rather than simply suppress answering.
- The same operator may drive both damage and rescue: wrong-to-right flips are also **14/14 wrong-entity substitution** in the IRR summary.

**Not established**

- This is not a circuit-level mechanism result.
- This is not cross-model evidence.
- This is not strong-form human-human IRR; the second rater is an LLM judge.

**Strongest source-grounded claim**

> On a locked held-out bridge benchmark, the ITI condition causes a statistically significant accuracy drop, and the dominant adjudicated failure mode is wrong-entity substitution rather than refusal.

**Strongest overclaim to avoid**

> ITI injects a known mechanism that cleanly redistributes logits among nearby entities.

**Why this is a good paper anchor**

- Clean held-out protocol
- Paired inference and uncertainty
- One-shot test discipline
- Failure mode strengthened by later row-level IRR artifacts instead of repeated prose

**Main unresolved confounds**

- Single-model, single-benchmark
- LLM second rater rather than human second annotator
- Mechanism story remains behavioral, not causal-internal

**One highest-value upgrade**

- Run the planned log-likelihood margin analysis on the adjudicated wrong-entity cases to test whether the intervention really shifts mass among semantically nearby candidates.

### 2. FaithEval H-Neuron vs SAE Matched-Readout Dissociation

**Current anchor value:** strong anchor, but not pristine

**Primary sources**

- [notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md](../../../notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md)
- [data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md](../../../data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md)
- [data/gemma3_4b/intervention/verbosity_confound/verbosity_confound_audit.md](../../../data/gemma3_4b/intervention/verbosity_confound/verbosity_confound_audit.md)
- [data/gemma3_4b/intervention_findings.md](../../../data/gemma3_4b/intervention_findings.md)
- [`data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json`](../../../data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json)
- [`data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`](../../../data/gemma3_4b/intervention/faitheval/control/comparison_summary.json)
- [`data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json`](../../../data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json)

**Established**

- On the committed anti-compliance FaithEval sweep, the H-neuron slope is **+2.09 pp/α** and the paired neuron-minus-SAE slope difference is **+1.93 pp/α [0.94, 2.92]** on matched items.  
  Source: [`faitheval_sae/control/slope_difference_summary.json`](../../../data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json), [2026-04-13-faitheval-slope-difference-reporting-audit.md](../../../notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md)
- Against eight random-neuron control seeds, every paired slope difference is positive, with seed-specific CIs excluding zero.  
  Source: [`faitheval/control/slope_difference_summary.json`](../../../data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json)
- The stronger “not just reconstruction error” closure move exists: the delta-only SAE follow-up remains near null for both H-features and random features.  
  Source: [sae_pipeline_audit.md](../../../data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md)

**Interpretive but plausible**

- In this Gemma setup, held-out detector quality alone is not enough to predict steering utility.
- The result is better read as a localization-to-control warning sign than as a clean theorem about all detection-based target selection.

**Not established**

- This does not prove that SAE features in general cannot steer.
- This does not prove that readout quality is unimportant.
- This does not prove neurons are the unique or globally optimal steering basis.

**Strongest source-grounded claim**

> In the committed Gemma FaithEval comparison, a held-out SAE readout with near-matched detector quality to the H-neuron readout did not yield useful steering under the tested SAE steering paths, while H-neurons did.

**Strongest overclaim to avoid**

> FaithEval cleanly proves that detector quality is insufficient as a target-selection criterion in general.

**Why this is not the cleanest anchor**

- The detector story is partly weakened by the response-length / response-form confound audit.  
  Source: [verbosity_confound_audit.md](../../../data/gemma3_4b/intervention/verbosity_confound/verbosity_confound_audit.md)
- The comparison is cross-representational and not operator-matched: neurons and SAE features differ in layer coverage, operator form, and feature granularity.  
  Source: [sae_pipeline_audit.md](../../../data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md), [paper/icml/main.tex](../../paper/icml/main.tex)
- The SAE coverage gap is real: only **10 of 34 layers** are covered, and **47.4%** of the 38 CETT H-neurons are in uncovered layers.  
  Source: [sae_pipeline_audit.md](../../../data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md)

**Why it is still a good paper anchor**

- Strong random-control specificity
- Narrow matched-readout claim backed by machine-readable slope summaries
- Delta-only follow-up makes it harder to dismiss the null as pure reconstruction noise

**Main unresolved confounds**

- Detection interpretation partly entangled with length/form
- Cross-family operator mismatch
- Untested SAE alternatives still exist in principle: wider SAE, fuller layer coverage, sparser intervention-optimized target

**One highest-value upgrade**

- Run the answer-token-regime confound audit that matches the detector’s actual training domain; that would tell us how much the detector-side caveat should weaken the paper-facing matched-readout story.

### 3. Jailbreak Measurement-Sensitivity Result

**Current anchor value:** good anchor for measurement, not for selector quality

**Primary sources**

- [notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md](../../../notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md)
- [notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md](../../../notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md)
- [notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md](../../../notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md)
- [`data/judge_validation/holdout_comparison.json`](../../../data/judge_validation/holdout_comparison.json)

**Established**

- On contamination-clean holdout, **CSV-v3 and StrongREJECT-GPT-4o tie at 96.0% accuracy** with identical error sets and **0 discordant holdout records**.  
  Source: [`holdout_comparison.json`](../../../data/judge_validation/holdout_comparison.json), [2026-04-12-4way-evaluator-holdout-validation.md](../../../notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md)
- The remaining CSV-v3 case is structural, not “holdout proved better”: StrongREJECT’s formula still zeroes high-substance refuse-then-comply cases when `refused = 1`.  
  Source: [2026-04-13-jailbreak-measurement-cleanup.md](../../../notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md)
- On the seed-0 jailbreak control audit, the measured effect depends on the ruler: CSV-v2 shows an H-neuron slope of **+2.30 pp/α [0.99, 3.58]** and a random-neuron null, but this is explicitly only seed-0 specificity.  
  Source: [2026-04-12-seed0-jailbreak-control-audit.md](../../../notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md)

**Interpretive but plausible**

- The most robust jailbreak contribution is that verdicts change with truncation, scoring granularity, and evaluator design.
- The project’s safer claim is about measurement dependence, not about discovering a uniquely best jailbreak evaluator.

**Not established**

- Full random-control specificity on jailbreak is not closed; the scored control evidence is still seed-0 canonical.
- This does not establish a clean selector-quality result.
- This does not establish CSV-v3 superiority over StrongREJECT on clean binary holdout.

**Strongest source-grounded claim**

> The same jailbreak outputs support different conclusions under different measurement choices, and the clean holdout does not support claiming that CSV-v3 beats StrongREJECT on binary accuracy.

**Strongest overclaim to avoid**

> CSV-v3 is better because holdout proves it.

**Why this is a good paper anchor**

- It directly supports the paper’s measurement gate.
- The key claim is source-grounded in machine-readable holdout comparison data.
- It corrects an easy cargo-cult failure mode: mistaking evaluator choice for truth.

**Why it is not a flagship behavioral anchor**

- The selector/control story here is still incomplete.
- Some of the stronger jailbreak specificity language still rests on partial control coverage.

**Main unresolved confounds**

- Full multi-seed random-neuron scoring remains unfinished
- Some narrative still spans several notes rather than one single canonical benchmark report

**One highest-value upgrade**

- Score the remaining random-neuron jailbreak control seeds on the same ruler and produce one canonical specificity closure report.

### 4. D7 Same-Ruler Causal-vs-Baseline/Random Result

**Current anchor value:** good supporting evidence; not flagship

**Primary sources**

- [paper/icml/reviews/2026-04-20-d7-quality-debt-adversarial-audit.md](./2026-04-20-d7-quality-debt-adversarial-audit.md)
- [`data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_locked/csv2_v3_evaluation/alpha_4.0.jsonl`](../../../data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_locked/csv2_v3_evaluation/alpha_4.0.jsonl)
- [`data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_1/csv2_evaluation/alpha_4.0.jsonl`](../../../data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_1/csv2_evaluation/alpha_4.0.jsonl)
- [`data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_2/csv2_evaluation/alpha_4.0.jsonl`](../../../data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_2/csv2_evaluation/alpha_4.0.jsonl)

**Established**

- Under a single v3-style ruler, the causal branch beats baseline and both available random-head controls:
  - baseline: **34.2%**
  - causal: **20.0%**, paired **-14.2 pp**
  - random seed 1 minus causal: **+17.2 pp**
  - random seed 2 minus causal: **+18.4 pp**  
  Source: [2026-04-20-d7-quality-debt-adversarial-audit.md](./2026-04-20-d7-quality-debt-adversarial-audit.md)
- Token-cap debt is visible but does not explain the paired effect; the causal advantage persists on both capped and uncapped subsets.  
  Source: [2026-04-20-d7-quality-debt-adversarial-audit.md](./2026-04-20-d7-quality-debt-adversarial-audit.md)
- The older broad bookkeeping objection drifted too far: visible comparator-quality debt in current files is small, not massive.  
  Source: [2026-04-20-d7-quality-debt-adversarial-audit.md](./2026-04-20-d7-quality-debt-adversarial-audit.md)

**Interpretive but plausible**

- D7 is stronger than “appendix-only debt” and could serve as supporting evidence for a broader selector story if kept narrow.

**Not established**

- Probe is still not in same-ruler closure.
- This does not prove a general causal-selector theorem.
- This does not justify promoting D7 to flagship status.

**Strongest source-grounded claim**

> On the available same-ruler full-500 D7 evidence, the causal branch outperforms baseline and both available layer-matched random-head controls, and token-cap debt does not drive that result.

**Strongest overclaim to avoid**

> D7 is now a clean selector-specific flagship.

**Why this is not a good main anchor**

- Mixed-ruler magnitude drift remains large.
- Probe is absent from the same-ruler closure.
- The result is benchmark-local and judge-family-local.

**Why it is still useful**

- It is now materially stronger than the older “too dirty to use” shorthand.
- It can support the paper if framed as narrow, same-ruler, benchmark-local evidence.

**Main unresolved confounds**

- Missing probe same-ruler panel
- Large cross-panel magnitude drift
- No generalization beyond this surface

**One highest-value upgrade**

- Rescore or otherwise close probe on the same ruler as baseline / L1 / causal / random so the selector hierarchy can be evaluated on one panel.

### 5. Narrow Mistral Extension Path

**Current anchor value:** not a current anchor; best next strengthening path

**Primary sources**

- [data/mistral24b/pipeline_report.md](../../../data/mistral24b/pipeline_report.md)
- [docs/archive/gh200-research-log-2026-03-15.md](../../../docs/archive/gh200-research-log-2026-03-15.md)

**Established**

- The expensive Mistral 24B response and activation artifacts are preserved locally.  
  Source: [pipeline_report.md](../../../data/mistral24b/pipeline_report.md)
- The pipeline report is explicit that there is still **no held-out Mistral eval**, **no Mistral intervention result**, and **no exact-checkpoint replication**.  
  Source: [pipeline_report.md](../../../data/mistral24b/pipeline_report.md)
- The GH200 log supports a narrow feasibility conclusion: 24B-class work is operationally viable here; 70B is not near-term paper-ready under the repo’s actual workflow constraints.  
  Source: [gh200-research-log-2026-03-15.md](../../../docs/archive/gh200-research-log-2026-03-15.md)

**Interpretive but plausible**

- The highest-ROI extension is not “replicate everything on Mistral,” but “test whether the H-neuron side of the story survives a narrow cross-family check.”

**Not established**

- No current Mistral result belongs in the evidence stack as a paper anchor.
- No current Mistral artifact proves cross-family FaithEval replication.
- Nothing current supports Mistral bridge or jailbreak claims.

**Strongest source-grounded claim**

> The repo has enough preserved Mistral infrastructure to make a narrow held-out detector plus FaithEval intervention package the best next strengthening path, but it does not yet have completed Mistral evidence.

**Strongest overclaim to avoid**

> Mistral already strengthens the paper.

**Why this is the best next path**

- It directly answers the strongest reviewer-style scope question: is the H-neuron effect Gemma-specific?
- The required artifacts and pipeline base already exist locally.
- It is far cheaper and cleaner than dragging 70B into the near-term paper plan.

**Main unresolved confounds**

- No held-out split yet
- `2501` is same-family, not exact-checkpoint
- Answer-token artifact still has cleanup debt

**One highest-value upgrade**

- Build a disjoint held-out Mistral detector evaluation, then run FaithEval `standard`-prompt H-neuron intervention with matched random-neuron controls.

## Good Anchors vs. Good Supporting Evidence

Results that currently look like **good anchors**:

- Bridge externality / wrong-entity substitution
- FaithEval matched-readout dissociation
- Jailbreak measurement-sensitivity

Results that currently look like **good supporting evidence but weak anchors**:

- D7 same-ruler causal-vs-baseline/random
- FaithEval random-neuron specificity on its own
- Delta-only SAE null on its own
- Bridge IRR as a reinforcement layer rather than a standalone story

Why the distinction matters:

- A result can be real and still be too local, too confounded, too partial, or too measurement-sensitive to carry narrative weight by itself.
- The project’s prior failure mode was to promote “good supporting evidence” into “best anchor” by repetition.

## Blocked Claims

These should be treated as blocked from future repetition unless explicitly re-verified against the current source hierarchy.

### Do not repeat without re-checking

- **“FaithEval is obviously the cleanest anchor.”**  
  Blocked because the result is real but still carries live detector-side and cross-representational confounds. It is strong, not pristine.

- **“D7 is too dirty to use.”**  
  Blocked because the 2026-04-20 audit shows a real same-ruler causal-vs-baseline/random result that survives the token-cap objection.

- **“CSV-v3 is better because holdout proves it.”**  
  Blocked because clean holdout shows a tie with StrongREJECT-GPT-4o, not superiority.

- **“FaithEval cleanly proves readout quality is insufficient.”**  
  Blocked because the safe conclusion is narrower: in this comparison, readout quality alone did not predict steering utility.

- **“D7 now proves causal selectors beat correlational selectors in general.”**  
  Blocked because probe is not in same-ruler closure and the benchmark remains local.

- **“Mistral already supports the paper.”**  
  Blocked because the current Mistral evidence is infrastructure plus planning, not completed held-out intervention evidence.

## Claims Safe to Source Later

These are the strongest later-usable claims if the paper or future notes need disciplined sourcing.

### Established

- On held-out TriviaQA bridge evaluation, paper-faithful E0 ITI reduces adjudicated accuracy by **-5.8 pp [-8.8, -3.0]** with **McNemar p=0.0002**.  
  Sources: [2026-04-13-bridge-phase3-test-results.md](../../../notes/act3-reports/2026-04-13-bridge-phase3-test-results.md), [`results.json`](../../../data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/results.json)

- On adjudicated bridge failure coding, wrong-entity substitution is the dominant right-to-wrong failure mode at **31/43 = 72.1% [57.3, 83.3]**.  
  Source: [`bridge_irr_summary.json`](../../../data/judge_validation/bridge_irr/bridge_irr_summary.json)

- On the committed FaithEval comparison, the paired neuron-minus-SAE slope difference is **+1.93 pp/α [0.94, 2.92]** on matched items.  
  Source: [`faitheval_sae/control/slope_difference_summary.json`](../../../data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json)

- On contamination-clean jailbreak holdout, CSV-v3 and StrongREJECT-GPT-4o both achieve **96.0%** accuracy with identical error sets.  
  Source: [`holdout_comparison.json`](../../../data/judge_validation/holdout_comparison.json)

- On the available same-ruler D7 panel, the causal branch beats baseline and both available random controls.  
  Source: [2026-04-20-d7-quality-debt-adversarial-audit.md](./2026-04-20-d7-quality-debt-adversarial-audit.md)

### Interpretive but plausible

- Bridge is currently the cleanest single paper anchor.
- FaithEval is stronger as a localized dissociation / warning sign than as a universal theorem.
- Jailbreak is strongest as a measurement case study rather than a selector-comparison result.
- D7 is stronger than the older broad objection but still supporting-only.

### Not established

- Any claim that one current result is immaculate
- Any cross-model generalization beyond Gemma from current completed evidence
- Any claim that Mistral has already upgraded the paper

## Scope Decision After Fresh Re-Ranking

The high-level scope choice still survives:

- Keep the current paper **Gemma-centered**
- Treat **Mistral as the best next strengthening path**
- Keep **Llama-70B out of the near-term paper path**

What changes is the internal confidence map:

- Let **bridge** carry more of the confidence load.
- Keep **FaithEval** prominent, but narrow its wording.
- Use **jailbreak** as the measurement gate case study.
- Keep **D7** available as bounded supporting evidence.

## Final Ranked Table

| Result | Current anchor value | Biggest strength | Main weakness | Upgrade path |
|---|---|---|---|---|
| TriviaQA bridge externality | Best current anchor | Held-out paired result plus dual-rated failure coding | Single-model, behavioral rather than circuit-level mechanism | Log-likelihood margin analysis on adjudicated substitution cases |
| FaithEval H-neuron vs SAE | Strong anchor, but not pristine | Strong matched-readout dissociation plus random-control specificity | Detector-side caveat and cross-representational confounds | Answer-token-domain detector confound audit |
| Jailbreak measurement-sensitivity | Good anchor for measurement | Clean holdout shows evaluator tie and blocks overclaiming | Selector-specificity story incomplete | Canonical multi-seed jailbreak control closure |
| D7 same-ruler causal-vs-baseline/random | Good supporting evidence | Narrow same-ruler effect survives token-cap objection | Probe missing from same-ruler closure; large ruler drift | Same-ruler probe closure |
| Narrow Mistral extension path | Not current evidence | Preserved artifacts and clear next experiment | No held-out eval or intervention result yet | Held-out detector + FaithEval standard-prompt intervention + matched random controls |

## Bottom Line

### Short summary of the ranking

- **Bridge** is the cleanest current anchor.
- **FaithEval** is still one of the project’s strongest results, but it should be carried with narrower wording than the repeated internal slogan.
- **Jailbreak** is strongest as a measurement memo, not an evaluator-ranking memo.
- **D7** is no longer “too dirty,” but still not a flagship.
- **Mistral** is still the best next strengthening path, but not current evidence.

### Most promising anchor candidates

1. TriviaQA bridge externality / wrong-entity substitution
2. FaithEval H-neuron vs SAE matched-readout dissociation
3. Jailbreak measurement-sensitivity
4. D7 same-ruler causal-vs-baseline/random as bounded support

### Claims to actively block from future cargo-cult repetition

1. “FaithEval is obviously the cleanest anchor.”
2. “D7 is too dirty to use.”
3. “CSV-v3 is better because holdout proves it.”
4. “Mistral already strengthens the paper.”

If condensed to one sentence:

> The project is strongest not because it has one immaculate flagship theorem, but because the current Gemma evidence now supports three distinct, source-grounded warnings: readout quality does not cleanly imply good steering targets, successful control does not cleanly imply benign transfer, and apparent verdicts do not cleanly survive measurement changes.

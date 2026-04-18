# Action-Item Fix Handoff for Current Paper Draft

Date: 2026-04-18  
Scope: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:1), source shards under `paper/draft/`, figures under `paper/draft/figures/`  
Purpose: translate the current draft audit into implementation-ready fixes for later edit sessions  
Relationship to other review files: this memo complements, and does not replace, [2026-04-16-research-grade-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/reviews/2026-04-16-research-grade-audit.md:1)  

## Executive Status

The paper’s empirical spine is intact. The FaithEval neuron-versus-SAE dissociation, the ITI answer-selection versus generation split, the held-out TriviaQA bridge harm, and the jailbreak measurement story all remain scientifically usable.

The remaining work is not a new research cycle. It is a publication-surface cleanup pass focused on evidence hierarchy, wording calibration, benchmark citation coverage, and figure/table discipline. The highest-risk failure mode is not “the paper is false”; it is “the paper visually or rhetorically claims more than the evidence earns.”

This memo is meant to be a working agenda for later sessions. Each item below is written so a future session can pick it up and implement it without re-running a full audit.

## Priority Action Items

### 1. Re-center Section 4 on FaithEval and demote D7 further

**Priority**  
Scientific-risk blocker

**Problem**  
Section 4 still gives the D7 jailbreak selector material more narrative weight than its evidential tier supports. The prose correctly says it is supporting-only, but the amount of space and repetition still makes it feel close to co-headline with the FaithEval anchor.

**Why it matters**  
This is the clearest way the manuscript can drift outside its earned boundary without changing any numbers. The paper is strongest when FaithEval is the sole load-bearing localization/control result and D7 is explicitly corroborative.

**What to change**  
- Compress D7 into one brief corroboration block plus an appendix pointer.
- Keep FaithEval as the only defended anchor for the “matched readout quality did not predict intervention utility” claim.
- Remove repeated hierarchy narration once that structure is clear.

**Where to change it**  
- [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:1)
- [section_1_introduction.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_1_introduction.md:1)
- [section_7_synthesis.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_7_synthesis.md:1)
- [appendix.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/appendix.md:1) if appendix framing needs to absorb more D7 detail

**Source of truth / supporting reports**  
- [2026-04-11-strategic-assessment.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/2026-04-11-strategic-assessment.md:179)
- [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:173)
- [2026-04-16-research-grade-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/reviews/2026-04-16-research-grade-audit.md:29)

**Done when**  
FaithEval is unmistakably the only load-bearing result in Section 4, and D7 reads as benchmark-local corroboration rather than a second anchor.

### 2. Weaken evaluator-choice language in Section 6

**Priority**  
Scientific-risk blocker

**Problem**  
The draft sometimes says evaluator choice changed the jailbreak conclusion or verdict. The evidence supports measurement dependence, a post-rerun holdout tie, and a shift away from binary-superiority framing, but it does not support a clean claim that evaluator identity alone reversed the intervention verdict.

**Why it matters**  
This is the main remaining repeated overstatement in the draft’s measurement story.

**What to change**  
- Replace “evaluator choice changed the conclusion/verdict” with wording like “evaluator dependence changed the measurement story” or “changed what the paper can claim about binary evaluator superiority.”
- Preserve the stronger and fully earned claims about truncation, granularity, and the `CSV-v3` versus `StrongREJECT` holdout tie.

**Where to change it**  
- [abstract.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/abstract.md:1)
- [section_1_introduction.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_1_introduction.md:1)
- [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:1)
- [section_7_synthesis.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_7_synthesis.md:1)
- [section_9_conclusion.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_9_conclusion.md:1)

**Source of truth / supporting reports**  
- [2026-04-12-4way-evaluator-holdout-validation.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md:159)
- [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:227)
- [2026-04-16-research-grade-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/reviews/2026-04-16-research-grade-audit.md:163)

**Done when**  
No summary surface implies evaluator identity alone flipped the intervention verdict, while the measurement-dependence argument stays intact.

### 3. Make the bridge diagnosis explicitly benchmark-local

**Priority**  
Scientific-risk blocker

**Problem**  
Wrong-entity substitution is sometimes written as if it is the paper-wide externality result or the settled mechanism, when it is actually the most frequent diagnosed error class on the TriviaQA bridge surface under a single-rater coding protocol.

**Why it matters**  
The bridge result is strong, but it is strongest as a benchmark-local output diagnosis. Over-reading it weakens the paper.

**What to change**  
- State consistently that wrong-entity substitution is the most frequent diagnosed bridge error class under the current protocol.
- Keep the single-rater caveat next to high-visibility uses of the taxonomy.
- Separate the descriptive result from the coarse-reweighting hypothesis.

**Where to change it**  
- [abstract.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/abstract.md:1)
- [section_5_case_study_II.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_5_case_study_II.md:1)
- [section_7_synthesis.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_7_synthesis.md:1)
- [paper/draft/figures/fig3_bridge_failure.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig3_bridge_failure.py:1)

**Source of truth / supporting reports**  
- [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:186)
- [2026-04-16-research-grade-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/reviews/2026-04-16-research-grade-audit.md:187)

**Done when**  
Every high-visibility use of the bridge taxonomy reads as benchmark-local descriptive evidence, not mechanism closure.

### 4. Fix the D7 appendix table so it matches the claim the text makes

**Priority**  
Scientific-risk blocker

**Problem**  
Appendix Table D1 currently shows each D7 branch versus baseline on normalized strict harmfulness, but the prose’s relevant claim is direct `causal vs probe/random`. The current table flatters the selector claim by showing the easiest comparison surface and hiding the actual discriminant panel.

**Why it matters**  
This is the highest-value figure/table correction because the current appendix surface does not support the sentence the main text tells the reader to take away.

**What to change**  
- Replace or expand Appendix Table D1 so it includes the direct comparator panel.
- Include the caveats that matter: mixed-ruler panel, probe/random CSV2 errors, and causal token-cap artifacts.
- If space is limited, prefer the direct comparison rows over extra baseline rows.

**Where to change it**  
- [appendix.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/appendix.md:1)

**Source of truth / supporting reports**  
- [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:114)
- [number_provenance.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/number_provenance.md:30)

**Done when**  
Appendix D shows the comparison the prose actually relies on, with the relevant caveats visible in the same surface.

### 5. Repair uncertainty discipline in Table 1

**Priority**  
Publication-surface blocker

**Problem**  
Table 1 includes effect sizes and baselines without uncertainty, even though the paper’s reporting standard emphasizes uncertainty-aware quantitative reporting.

**Why it matters**  
Table 1 is supposed to be a construct map. Right now it partially behaves like a results table, but without the discipline expected of a results table.

**What to change**  
- Preferred default: de-quantify Table 1 so it stays a construct map.
- If quantitative values are retained, add uncertainty consistently.
- Do not let the table mix benchmark definition and half-formed headline numbers.

**Where to change it**  
- [section_2_scope_constructs.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:1)

**Source of truth / supporting reports**  
- [number_provenance.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/number_provenance.md:1)
- [docs/ci_manifest.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/docs/ci_manifest.json:1)

**Done when**  
Table 1 is clearly a construct map rather than a lightly sourced results panel.

### 6. Make Figure 4 like-for-like

**Priority**  
Publication-surface blocker

**Problem**  
Figure 4’s binary-versus-graded contrast is directionally right, but the panels are not matched enough. Panel A shows only binary endpoints while Panel B gets the full graded sweep plus control framing.

**Why it matters**  
The visual claim is “measurement changed the conclusion.” The figure should make that comparison fair, not just rhetorically effective.

**What to change**  
- Align binary and graded displays on comparable alpha coverage and control framing.
- If that is not practical, explicitly label the asymmetry in the caption and annotation.
- Keep the post-`SR-4o` holdout tie intact.

**Where to change it**  
- [paper/draft/figures/fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:1)
- [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:1)

**Source of truth / supporting reports**  
- [2026-04-12-seed0-jailbreak-control-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md:49)
- [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:239)

**Done when**  
Figure 4 no longer relies on asymmetric panel construction to make the granularity point feel stronger.

### 7. Fix uncertainty presentation for the random SAE baseline

**Priority**  
Publication-surface blocker

**Problem**  
Figure 2B and Appendix Table C1 show Wilson CIs for H-neurons and SAE H-features, but use mean `± SD` across three random SAE seeds for the random baseline. That makes the random comparator look more precisely estimated than it really is.

**Why it matters**  
The FaithEval figure is the cleanest figure in the paper. Its uncertainty conventions should not quietly bias perception in favor of the headline result.

**What to change**  
- Use a consistent uncertainty convention for the random SAE comparator, or
- explicitly label the random curve as seed spread rather than confidence interval.

**Where to change it**  
- [paper/draft/figures/fig2_matched_readouts.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig2_matched_readouts.py:1)
- [appendix.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/appendix.md:1)

**Source of truth / supporting reports**  
- [data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json:1)
- [number_provenance.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/number_provenance.md:16)

**Done when**  
The random SAE comparator is no longer visually more precise than justified by its evidence base.

### 8. Complete benchmark citation coverage

**Priority**  
Scientific-integrity blocker

**Problem**  
Several core benchmark surfaces are central to the paper but not properly grounded in the bibliography and body citations. FaithEval is in the references but not cited in the body; other major surfaces are not fully covered in the bibliography.

**Why it matters**  
This is the main reference-integrity failure remaining in the draft.

**What to change**  
- Add canonical citations for TruthfulQA, TriviaQA, JailbreakBench, BioASQ, FalseQA, and SimpleQA where appropriate.
- Cite FaithEval where first introduced in body prose.
- Ensure Table 1 and first-use benchmark mentions resolve to actual bibliography entries.

**Where to change it**  
- [references.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/references.md:1)
- [section_2_scope_constructs.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:1)
- Any first-use mention in [section_1_introduction.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_1_introduction.md:1), [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:1), and [section_5_case_study_II.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_5_case_study_II.md:1)

**Source of truth / supporting reports**  
- [references.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/references.md:1)
- [2026-04-16-research-grade-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/reviews/2026-04-16-research-grade-audit.md:209)

**Done when**  
Every benchmark central to the evidence spine has a resolvable canonical citation in the body and bibliography.

### 9. Demote pipeline-bug detail in Section 6

**Priority**  
Publication-surface blocker

**Problem**  
The schema/version-mismatch bug is scientifically relevant, but Section 6 currently gives pipeline debugging almost the same argumentative level as truncation, scoring granularity, and evaluator dependence.

**Why it matters**  
The measurement section is strongest when it stays focused on inferential measurement choices rather than reading like an internal postmortem.

**What to change**  
- Keep a short main-text paragraph on the bug’s scientific consequence.
- Move explanatory debugging detail to the appendix or to linked supporting reports.

**Where to change it**  
- [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:1)
- [appendix.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/appendix.md:1)

**Source of truth / supporting reports**  
- [2026-04-12-seed0-jailbreak-control-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md:214)

**Done when**  
Section 6 reads as a scientific measurement argument, with debugging detail clearly subordinate.

### 10. Trim meta-framing and internal draft jargon

**Priority**  
Publication-surface cleanup

**Problem**  
The manuscript still spends too much space narrating its own hierarchy and using internal project language instead of just advancing the argument.

**Why it matters**  
This is a clarity cost. It makes the paper sound more like an internal audit memo than a paper.

**What to change**  
- Reduce repeated uses of terms such as `anchor`, `supporting`, `benchmark-local`, `caveated`, and `paper-facing`.
- Declare the hierarchy once per section, then let subsection order and caveat placement do the rest.

**Where to change it**  
- First pass focus: [section_1_introduction.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_1_introduction.md:1), [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:1), [section_5_case_study_II.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_5_case_study_II.md:1), [section_7_synthesis.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_7_synthesis.md:1)

**Source of truth / supporting reports**  
- [good-write-up.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/good-practices/good-write-up.md:57)
- [paper/draft/AGENTS.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/AGENTS.md:17)

**Done when**  
The prose feels claim-forward rather than process-forward.

### 11. Revisit title and major headers after scientific fixes land

**Priority**  
Publication-surface cleanup

**Problem**  
The current title is memorable but less informative than the actual thesis. A stronger claim-forward title was already recommended in the strategy materials.

**Why it matters**  
This is lower priority than scientific calibration, but the title should reflect the final bounded thesis once the body is synchronized.

**What to change**  
- Revisit the title only after the higher-priority content fixes are in place.
- Prefer a claim-forward title over a rhetorical contrast title.

**Where to change it**  
- [front_matter.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/front_matter.md:1)

**Source of truth / supporting reports**  
- [2026-04-11-strategic-assessment.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/2026-04-11-strategic-assessment.md:179)
- [good-write-up.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/good-practices/good-write-up.md:61)

**Done when**  
The title states the earned thesis directly and remains fully consistent with the final claim boundary.

## Secondary Action Items

### Bridge figure duplication cleanup

The bridge section currently duplicates substitution examples across main text and Figure 3. If space or reader load becomes a problem, prefer one well-designed surface over two redundant ones.

### Evaluator-surface de-duplication

The holdout tie currently appears in Figure 4, the main-text table, and the appendix dev table context. If later trimming is needed, keep the most reader-useful surface and demote the rest.

### Figure 1 simplification

Figure 1 is broadly fine, but it can likely carry less text without losing its conceptual role.

## Guardrails for Future Edit Sessions

- Treat [2026-04-11-strategic-assessment.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/2026-04-11-strategic-assessment.md:1) as the earned-claim boundary.
- Treat [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:1) as the live current-state D7 source.
- Edit source shards and figure scripts only. Do not hand-edit `full_paper.md`.
- Preserve valid caveats. Do not weaken or drop them for rhetorical neatness.
- Do not broaden D7 beyond benchmark-local supporting evidence.
- Do not revert BioASQ to a clean-null story.
- Do not revert the evaluator story to “`CSV-v3` is better because of binary holdout accuracy.”
- If quantitative reporting surfaces change, run `uv run python scripts/audit_ci_coverage.py`.
- After prose or figure changes, rebuild the manuscript with `uv run python scripts/build_full_paper.py` and verify with `--check`.

## Linked Reports

- Canonical scientific review: [2026-04-16-research-grade-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/reviews/2026-04-16-research-grade-audit.md:1)
- Current claim boundary: [2026-04-11-strategic-assessment.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/2026-04-11-strategic-assessment.md:1)
- Paper authoring guardrails: [paper/draft/AGENTS.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/AGENTS.md:1)
- Number ledger: [number_provenance.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/number_provenance.md:1)
- D7 current-state audit: [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:1)
- Bridge test report: [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:1)
- Measurement cleanup: [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:1)
- Evaluator holdout validation: [2026-04-12-4way-evaluator-holdout-validation.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md:1)

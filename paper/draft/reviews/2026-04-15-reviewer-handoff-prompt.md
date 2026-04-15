# Reviewer Handoff Prompt

Date: 2026-04-15  
Scope: `paper/draft/full_paper.md` plus the source shards and figure scripts that feed it  
Review mode: publication-readiness, claim-evidence fidelity, and coherence after a compression pass

## Context

This draft was just revised to align the paper's promoted claims with the strongest current evidence rather than older or superseded slices. The main changes were:

- FaithEval is now the flagship localization/control result.
- The D7 / jailbreak selector comparison is demoted to supporting, benchmark-local evidence.
- The evaluator story now reflects the post-StrongREJECT-GPT-4o cleanup: holdout binary accuracy is tied; CSV-v3 remains preferred for granularity and taxonomy, not binary superiority.
- BioASQ is no longer described as a clean null; the paper now says there is no robust net alias-accuracy effect despite substantial behavioral perturbation.
- Section 2 and Section 6 were compressed to remove internal workflow language and stale claim framing.
- Figures 1 to 4 were updated to match the revised evidence hierarchy.

## Files I Revised Directly

- `paper/draft/abstract.md`
- `paper/draft/assembly_manifest.json`
- `paper/draft/number_provenance.md`
- `paper/draft/section_1_introduction.md`
- `paper/draft/section_2_scope_constructs.md`
- `paper/draft/section_4_case_study_I.md`
- `paper/draft/section_5_case_study_II.md`
- `paper/draft/section_6_measurement.md`
- `paper/draft/section_7_synthesis.md`
- `paper/draft/section_8_limitations.md`
- `paper/draft/section_9_conclusion.md`
- `paper/draft/figures/fig1_four_stage_scaffold.py`
- `paper/draft/figures/fig2_matched_readouts.py`
- `paper/draft/figures/fig3_bridge_failure.py`
- `paper/draft/figures/fig4_measurement.py`
- regenerated figure PNGs and rebuilt `paper/draft/full_paper.md`

## Files / Surfaces I Did Not Review Deeply

Please explicitly check these rather than assuming they stayed coherent:

- `paper/draft/front_matter.md`
- `paper/draft/section_3_related_work.md`
- `paper/draft/appendix.md`
- `paper/draft/references.md`
- figure captions and cross-references as rendered in `paper/draft/full_paper.md`
- table wording, footnotes, and provenance references outside the sections listed above

## Primary Review Questions

1. Do any reader-facing claims still rely on superseded evidence or outdated wording?
2. Does the manuscript now weight the strongest current evidence correctly across abstract, intro, body, synthesis, conclusion, figures, and limitations?
3. Did the compression pass remove repetition without deleting unique, still-valuable information?
4. Did the compression pass leave behind duplicated thesis material, duplicated caveats, or duplicated benchmark summaries that should be collapsed further?
5. Are any untouched sections now out of sync with the revised framing, terminology, or evidence hierarchy?

## Non-Negotiable Claim Checks

Please verify all of the following.

- D7 is not presented as a clean selector-specific or mechanism-closed result.
- The superseded pilot-local D7 probe-null story is not used as a headline claim outside its narrow historical or supporting context.
- FaithEval is visibly the anchor localization/control result in both prose and Figure 2.
- The paper does not claim CSV-v3 beats StrongREJECT on holdout binary accuracy after the GPT-4o rerun.
- The paper's reason to prefer CSV-v3 is measurement granularity / outcome taxonomy, not binary superiority.
- BioASQ is not called a clean null if the intended meaning is actually "flat endpoint despite active behavioral perturbation."
- Anchor 3 is consistently framed as `measurement -> conclusion` or an exact equivalent, not `measurement -> localization`.

## Duplication vs. Information-Loss Audit

This is especially important. The recent revision intentionally compressed repeated thesis material in the abstract, introduction, Section 2, Section 6, synthesis, and conclusion. Please review with two competing risks in mind:

- Risk A: duplicated information still remains across summary surfaces and makes the paper feel repetitive.
- Risk B: useful information that looked repetitive was actually carrying distinct value and may have been over-compressed.

Please check for both, not only Risk A.

Concrete things to look for:

- Abstract, introduction, synthesis, and conclusion repeating the same thesis sentence with no new function.
- Section 2 and Section 7 repeating each other rather than playing distinct roles.
- Section 6 re-explaining implementation history that no longer serves the scientific point.
- Loss of unique nuance when text was shortened, especially:
  - why the D7 full-500 result is caveated
  - why the BioASQ result is not behavioral inactivity
  - why evaluator choice still matters even though holdout binary accuracy is tied
  - why the bridge failure mode matters scientifically rather than rhetorically
- Removal of useful table interpretation, footnote context, or construct definitions that are not actually duplicated elsewhere

If you propose cuts, please say what unique information would still remain elsewhere after the cut. If you propose restoring material, please say what specific value was lost and where it should return.

## Untouched-Surface Checks

Please inspect the following even if they seem unrelated to the recent edits:

- `front_matter.md`: title, subtitle, framing, and whether the front matter still matches the revised thesis hierarchy
- `section_3_related_work.md`: does it still set up the right contrast now that the main claim is narrower and more evidence-weighted?
- `appendix.md`: are appendix promises, placeholders, or forward references still accurate?
- `references.md`: no broken citation keys, missing references, or references that no longer support the narrowed claims
- `assembly_manifest.json`: section names still match the current shard headings
- `number_provenance.md`: historical numbers are clearly labeled as historical; no stale value looks like a live headline claim

## Figure Review

Please review both the rendered figures and the scripts.

- Figure 1: conceptual opener, readable, no over-detailed anchor boxes, and the stage labels match the paper's final framing
- Figure 2: FaithEval clearly primary; jailbreak panel visibly supporting; no visual design choice accidentally over-promotes the D7 comparator
- Figure 3: panel balance is readable and the example table is legible
- Figure 4: incompatible units are not visually conflated; tied holdout evaluator result is represented honestly; uncertainty is adequate for the inferential claims
- Every figure caption should match the actual evidentiary strength used in the body text

## Coherence / Reader-Experience Checks

Please also check for issues I may not have looked at directly:

- paragraph-to-paragraph flow
- abrupt terminology shifts
- internal jargon leakage
- section opening and closing sentences that no longer match the section body
- over-caveating that obscures the main thesis
- under-caveating that promotes a weaker result too far
- redundant tables or captions
- missing transitions after recent cuts
- cross-references that still point at old section roles

## Deliverable Format

Please return one consolidated review memo with:

1. Blocking issues first, ordered by severity.
2. File and line references wherever possible.
3. A separate section called `Duplication vs Information Loss` that explicitly weighs both sides.
4. A separate section called `Untouched Surfaces` covering the files I did not review deeply.
5. A short `Safe Thesis` section stating the strongest version of the paper you think is currently earned.

## Verification Commands

If you make or recommend text changes that should preserve the generated manuscript, please use or reference:

```bash
uv run python scripts/build_full_paper.py
uv run python scripts/build_full_paper.py --check
uv run python scripts/audit_ci_coverage.py
uv run pytest tests/test_build_full_paper.py
```

# Reviewer Handoff Prompt

Date: 2026-04-16  
Scope: `paper/draft/full_paper.md` plus the source shards and figure scripts that feed it  
Review mode: publication-readiness after the D7 retiering pass

## Context

This draft has just gone through a targeted revision to address the highest-priority evidence-hierarchy issue from the research-grade audit. The latest pass did four specific things:

- D7 / jailbreak selector evidence was explicitly retiered to supporting, benchmark-local status in the introduction, Section 4, and Section 7.
- The synthesis table that visually flattened FaithEval and D7 was removed from the main text.
- Recommendation 1 in the synthesis section now stands on the matched FaithEval result alone.
- The generated manuscript was rebuilt, so `paper/draft/full_paper.md` reflects the current source shards.

The high-level manuscript state should now be:

- FaithEval is the only load-bearing localization/control anchor.
- D7 remains in the paper, but only as narrow corroboration with explicit caveats.
- The post-StrongREJECT-GPT-4o evaluator story remains: holdout binary accuracy is tied; CSV-v3 is preferred for richer outcome structure, not binary superiority.
- BioASQ remains endpoint-flat but behaviorally active.

## Files Revised Directly In The Latest Pass

- `paper/draft/section_1_introduction.md`
- `paper/draft/section_4_case_study_I.md`
- `paper/draft/section_7_synthesis.md`
- rebuilt `paper/draft/full_paper.md`

## Files / Surfaces I Did Not Re-review Deeply In This Pass

Please inspect these explicitly rather than assuming they stayed coherent:

- `paper/draft/abstract.md`
- `paper/draft/section_2_scope_constructs.md`
- `paper/draft/section_3_related_work.md`
- `paper/draft/section_5_case_study_II.md`
- `paper/draft/section_6_measurement.md`
- `paper/draft/section_8_limitations.md`
- `paper/draft/appendix.md`
- `paper/draft/references.md`
- `paper/citations/registry.json`
- figure captions and cross-references as rendered in `paper/draft/full_paper.md`
- `paper/draft/figures/fig2_matched_readouts.py`
- `paper/draft/figures/fig4_measurement.py`

## Primary Review Questions

1. Did the D7 retiering actually succeed across intro, Section 4, synthesis, limitations, appendix, and abstract, or is D7 still structurally over-promoted somewhere?
2. With D7 demoted, is the paper's remaining strongest criticism now Section 5's mechanism language, and does the draft still calibrate that evidence correctly?
3. Are Figures 2 and 4 still mismatched with the manuscript's evidentiary standard or interval language?
4. Did removing the old synthesis table create any narrative hole, cross-reference issue, or accidental loss of useful information?
5. Are any untouched sections now out of sync with the revised evidence hierarchy?

## Non-Negotiable Claim Checks

Please verify all of the following.

- FaithEval is visibly the anchor localization/control result in prose, section setup, and Figure 2.
- D7 is not presented as a co-anchor, a mechanism-clean selector result, or a broader selector-specificity closure.
- The current D7 claim is the narrow April 16 version: benchmark-local supporting evidence that selector choice matters on this surface, with live mixed-ruler/error/quality caveats.
- The old main-text synthesis table is gone, and no remaining main-text surface visually flattens FaithEval and D7 into the same evidential tier.
- Recommendation 1 is now earned without relying on D7.
- The paper does not claim CSV-v3 beats StrongREJECT on holdout binary accuracy after the GPT-4o rerun.
- The reason to prefer CSV-v3 remains measurement granularity / outcome taxonomy, not binary superiority.
- BioASQ is not described as behavioral inactivity if the claim is actually endpoint-flat despite active perturbation.

## Remaining High-Risk Areas

Please focus especially on these, since they are now the likely top blockers:

- `section_5_case_study_II.md`
  - wrong-entity substitution is earned as the most frequent diagnosed failure mode
  - stronger mechanism language may still outrun the evidence
  - single-rater taxonomy caveats should remain local and explicit
- `figures/fig4_measurement.py` plus Section 6
  - confirm whether the figure uses Wilson/count-based intervals while the text reports prompt-clustered CIs
  - flag any mismatch between panel statistics and paper-facing claims
- `paper/citations/registry.json`
  - treat it as potentially broken until verified
- `section_2_scope_constructs.md` and `section_3_related_work.md`
  - check whether they still spend too much space on governance/novelty scaffolding relative to the paper's evidence spine

## Figure Review

Please review both the rendered figures and the scripts.

- Figure 1: opener remains readable and paper-facing
- Figure 2: FaithEval is clearly primary; no lingering design choice over-promotes supporting jailbreak material; matched-readout precision is not overstated
- Figure 3: still the strongest main-text figure; verify the caption and panel text keep the taxonomy descriptive rather than mechanistic
- Figure 4: tie on holdout binary accuracy is represented honestly; interval methodology matches the text; panel load is justified

## Coherence / Reader-Experience Checks

Please also check for issues I may not have looked at directly:

- paragraph-to-paragraph flow after the D7 cuts
- missing transitions caused by removing the old synthesis table
- abstract / intro / synthesis / conclusion drift after the latest retiering pass
- internal jargon leakage
- duplicated caveats versus missing caveats
- stale line of argument in untouched sections that still assumes D7 had more headline weight

## Deliverable Format

Please return one consolidated review memo with:

1. Blocking issues first, ordered by severity.
2. File and line references wherever possible.
3. A short section called `Did Finding 1 Actually Land?`
4. A separate section called `Untouched Surfaces`
5. A short `Safe Thesis` section stating the strongest version of the paper you think is currently earned.

## Verification Commands

If you make or recommend text changes that should preserve the generated manuscript, please use or reference:

```bash
uv run python scripts/build_full_paper.py
uv run python scripts/build_full_paper.py --check
uv run python scripts/audit_ci_coverage.py
uv run pytest tests/test_build_full_paper.py
```

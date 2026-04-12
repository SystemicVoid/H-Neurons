# Reviewer Handoff Pack — Detection Is Not Enough

> This is the index file. Each reviewer should get a single paste-ready prompt file from the list below.

---

## Current state of the draft

The current assembled draft is:

- `notes/paper/draft/full_paper.md`

Section files are:

- `notes/paper/draft/abstract.md`
- `notes/paper/draft/section_1_introduction.md`
- `notes/paper/draft/section_2_scope_constructs.md`
- `notes/paper/draft/section_3_related_work.md`
- `notes/paper/draft/section_4_case_study_I.md`
- `notes/paper/draft/section_5_case_study_II.md`
- `notes/paper/draft/section_6_measurement.md`
- `notes/paper/draft/section_7_synthesis.md`
- `notes/paper/draft/section_8_limitations.md`
- `notes/paper/draft/section_9_conclusion.md`

Supporting draft assets:

- `notes/paper/draft/number_provenance.md`
- `notes/paper/draft/figures/fig1_four_stage_scaffold.{py,png}`
- `notes/paper/draft/figures/fig2_matched_readouts.{py,png}`
- `notes/paper/draft/figures/fig3_bridge_failure.{py,png}`
- `notes/paper/draft/figures/fig4_measurement.{py,png}`

Governing documents:

- `notes/paper/final_flagship_outline_review.md`
- `notes/paper/revised_flagship_outline-v2.md`
- `notes/2026-04-11-strategic-assessment.md`
- `notes/measurement-blueprint.md`
- `notes/paper/literature-research/research_arbitration_audit_detection_is_not_enough.md`
- `notes/paper/literature-research/gpt-deep-literature-review.md`
- `notes/paper/literature-research/opus-deep-literature-review.md`

Earlier review artifacts:

- `notes/paper/draft/reviews/final_claim_audit.md`
- `notes/paper/draft/reviews/cross_section_audit.md`
- `notes/paper/draft/reviews/section_2_review_codex.md`
- `notes/paper/draft/reviews/section_5_review_codex.md`

### What the last pass already changed

The most recent edit pass already did the following:

- Resolved the remaining `[UNCERTAINTY]` placeholders.
- Added `Box B` and `Box C`; `Box A` and `Box D` were already present.
- Added figure references for Figures 1–4.
- Added an `Appendix A` summary so §4.1 appendix references resolve.
- Clarified the SAE `+0.16 pp/alpha` versus `+0.12 pp/alpha` distinction.
- Attached bridge dev-set caveats near the promoted bridge claims.
- Replaced most reader-facing lab jargon and removed the conclusion's `necessary but insufficient` wording.
- Expanded Limitation `L8` to include the StrongREJECT `GPT-4o-mini` confound.
- Updated `number_provenance.md` to make remaining source limitations explicit.

### What still needs fresh eyes

Do **not** assume the last pass is correct just because it removed visible placeholders. The key open review surfaces now are:

1. Whether the recent fixes actually stay within the earned-claim boundary.
2. Whether citation claims in §1 and §3 are accurate against the source papers.
3. Whether all promoted numbers, table entries, and figure references still line up after the rewrite.
4. Whether the new appendix/boxes/bridges were integrated cleanly, not just inserted mechanically.
5. Whether the paper now reads as a coherent submission draft rather than a patched audit memo.

---

## Global reviewer rules

Apply these rules regardless of which prompt you use:

1. Review against the **actual repo state**, not the stale assumptions in older review notes.
2. Treat `notes/2026-04-11-strategic-assessment.md` as the claim governor.
3. Treat the arbitration audit as the novelty governor.
4. Every quantitative concern must point to a concrete source file or a concrete missing source.
5. Every citation concern must be checked against the paper markdown in `papers/`, not against literature-review summaries alone.
6. Distinguish:
   - `WRONG` = factually incorrect or overclaimed
   - `WEAK` = defensible but under-supported / phrased too strongly
   - `MISSING` = required content absent
   - `STYLE` = prose or readability issue
7. Prefer file/line references and specific replacement guidance.
8. Do not spend time re-finding issues that are already obviously resolved unless you think the fix is wrong or incomplete.

---

## Prompt files

Give each reviewer exactly one of these files:

- `notes/paper/draft/FINAL_GATE_REVIEW_PROMPT.md`
- `notes/paper/draft/CITATION_NOVELTY_REVIEW_PROMPT.md`
- `notes/paper/draft/QUANTITATIVE_INTEGRITY_REVIEW_PROMPT.md`
- `notes/paper/draft/PROSE_PACKAGE_REVIEW_PROMPT.md`

---

## Recommended review sequence

If running multiple reviewers, use this order:

1. Prompt 2 — Citations / Related Work / Novelty
2. Prompt 3 — Numbers / Provenance / Tables / Figures
3. Prompt 4 — Prose / Readability / Submission Package
4. Prompt 1 — Coordinator / Final Gate Reviewer

Why this order:

- citation and number review establish what the paper is allowed to say,
- prose review then improves presentation without accidentally polishing unsafe claims,
- the coordinator then decides whether the remaining issues are blocking or cosmetic.

---

## Success criteria for this second-pass review

After these reviewer prompts, you should have:

- a fresh verdict on whether the draft is actually close to submission-ready,
- one literature-grounded citation audit,
- one quantitative/source audit,
- one prose/package audit,
- and a final prioritized list of only the remaining real issues.

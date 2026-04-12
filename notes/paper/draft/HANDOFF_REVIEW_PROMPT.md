# Next-Session Handoff Prompt — Detection Is Not Enough

Use this as the starting prompt for the next editing/review session.

---

## Mission

You are taking over the manuscript pass for **"Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT."**

Your job is **not** to redo the already-completed unblock pass. Your job is to:

1. verify the recent fixes against the live repo state,
2. finish the remaining polish and package work,
3. run one more high-quality review pass,
4. make any clearly justified manuscript edits you can support from sources already in the repo,
5. leave the draft in a cleaner, more submission-like state with a short residual-risk list.

Work against the actual files in the repo. Do not rely on stale review notes when the manuscript now says something different.

---

## Read First

Primary manuscript files:

- `notes/paper/draft/full_paper.md`
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

Supporting assets:

- `notes/paper/draft/number_provenance.md`
- `notes/paper/citations/registry.json`
- `notes/paper/draft/figures/fig1_four_stage_scaffold.{py,png}`
- `notes/paper/draft/figures/fig2_matched_readouts.{py,png}`
- `notes/paper/draft/figures/fig3_bridge_failure.{py,png}`
- `notes/paper/draft/figures/fig4_measurement.{py,png}`

Claim and novelty governors:

- `notes/2026-04-11-strategic-assessment.md`
- `notes/paper/literature-research/research_arbitration_audit_detection_is_not_enough.md`
- `notes/paper/final_flagship_outline_review.md`
- `notes/paper/revised_flagship_outline-v2.md`

Relevant literature sources:

- use `papers/INDEX.md` first,
- then inspect the actual markdown files under `papers/` for any citation you are validating.

Older review artifacts are useful only as background:

- `notes/paper/draft/reviews/final_claim_audit.md`
- `notes/paper/draft/reviews/cross_section_audit.md`
- `notes/paper/draft/reviews/section_2_review_codex.md`
- `notes/paper/draft/reviews/section_5_review_codex.md`

---

## What The Last Pass Already Fixed

Assume these were intentionally changed and should be **verified**, not blindly re-opened:

- the placeholder references section was replaced with a real bibliography in `full_paper.md`,
- the novelty claim was narrowed to the earned FaithEval matched comparison,
- the unsupported Eiras `25x` wording was removed,
- `GuidedBench` and `MAT-Steer` were added to related work,
- `AxBench` and `Safer or Luckier?` were vendored into `papers/`,
- the probe-head top-20 AUROC range was corrected to `0.87–1.0`,
- the `N4288` regularization wording was corrected to show appearance at `C = 1.0`,
- the promoted jailbreak CI was standardized to `+7.6 pp [2.6, 12.8]`,
- the evaluation-contract language was softened away from implying every result satisfies the strictest bar,
- FalseQA provenance was retargeted to concrete artifacts under `data/gemma3_4b/intervention/falseqa/...`,
- Figures 1–4 are now embedded in the draft with captions,
- `fig2` was updated to use SAE-random control data in Panel B,
- `fig4` was updated to load measurement data from canonical artifacts instead of stale hard-coded binary CI bounds,
- `full_paper.md` was resynced to match the section sources.

Do not spend time “finding” those old issues unless you think the current fix is still wrong.

---

## Remaining Work To Tackle

The manuscript is past the unblock stage. The remaining work is mostly about **polish, continuity, and one more careful review pass**.

Focus on these areas:

1. **Section 6.3 readability**
   - It still risks reading like an internal audit memo instead of paper prose.
   - Tighten it so the evaluator comparison explains the decision logic cleanly without sounding like lab notes.

2. **Reader-facing provenance language**
   - Some provenance-oriented wording still feels internal-facing.
   - Keep rigor, but make the paper read like a manuscript rather than a repository trace log.

3. **Cross-section flow**
   - Check that abstract, intro, §§4–7, limitations, and conclusion tell the same story in the same claim register.
   - Look for repeated caveats, repeated thesis sentences, and places where the bridge/measurement framing is restated too many times.

4. **Figure/package integration**
   - Verify each figure is introduced at the right place and that each caption says what the figure establishes.
   - Check that figures are helping the argument, not interrupting it.

5. **Final review for residual blockers**
   - Look for any remaining claim drift, citation overreach, number mismatch, or package roughness that would justify one more edit pass.

---

## Repo State Notes

There are many unrelated untracked files under `data/` and some newly vendored papers under `papers/`.

Treat these as constraints:

- do **not** clean up unrelated untracked data,
- do **not** revert unrelated changes,
- do **not** assume `git status` noise means manuscript regressions,
- if you edit manuscript files, prefer editing the section sources first and sync `full_paper.md` after.

Recently changed manuscript-facing files include:

- `.gitignore`
- `notes/paper/citations/registry.json`
- `notes/paper/draft/abstract.md`
- `notes/paper/draft/full_paper.md`
- `notes/paper/draft/number_provenance.md`
- `notes/paper/draft/section_1_introduction.md`
- `notes/paper/draft/section_2_scope_constructs.md`
- `notes/paper/draft/section_3_related_work.md`
- `notes/paper/draft/section_4_case_study_I.md`
- `notes/paper/draft/section_5_case_study_II.md`
- `notes/paper/draft/section_6_measurement.md`
- `notes/paper/draft/section_7_synthesis.md`
- `notes/paper/draft/section_8_limitations.md`
- `notes/paper/draft/figures/fig2_matched_readouts.py`
- `notes/paper/draft/figures/fig4_measurement.py`

---

## Working Rules

1. Treat `notes/2026-04-11-strategic-assessment.md` as the claim governor.
2. Treat the arbitration audit as the novelty governor.
3. Validate citation claims against the actual source markdown in `papers/`, not summary notes alone.
4. Every quantitative concern must point to a concrete source file or a concrete missing source.
5. Prefer surgical fixes over broad rewrites unless a section is structurally broken.
6. Keep the paper in submission voice. Remove lab-voice phrasing when it leaks through.
7. If you make edits, verify them immediately instead of stacking speculative changes.

When reporting findings, distinguish:

- `WRONG` = factually incorrect or overclaimed
- `WEAK` = defensible but phrased too strongly or supported too indirectly
- `MISSING` = required content absent
- `STYLE` = readability / flow / package issue

---

## Suggested Execution Plan

1. Read `full_paper.md`, then skim section files to confirm assembly consistency.
2. Inspect `number_provenance.md` and `registry.json` only where they bear on promoted claims.
3. Read §6.3 carefully and decide whether it needs a prose rewrite, structure trim, or both.
4. Check the intro, synthesis, and conclusion for repeated thesis/caveat language.
5. Verify figure callouts and captions in the places they now appear in the manuscript.
6. Make only the edits you can justify from the repo.
7. Re-sync `full_paper.md` if section sources changed.
8. Run a final review sweep and produce a short prioritized residual-risk list.

---

## Verification Checklist

If you edit prose only:

- verify `full_paper.md` matches the section sources,
- grep for obvious stale phrases or contradictions you intended to remove,
- confirm no promoted number changed without a provenance update.

If you edit citations or numbers:

- update `notes/paper/draft/number_provenance.md` and/or `notes/paper/citations/registry.json` as needed,
- verify every changed quantitative claim still points to a concrete artifact.

If you edit figure scripts:

- rerun the affected figure scripts,
- confirm the corresponding PNGs regenerate,
- recheck the manuscript caption/callout text against the updated figure.

Minimum useful checks:

- `git diff -- notes/paper/draft notes/paper/citations/registry.json .gitignore`
- `uv run python notes/paper/draft/figures/fig1_four_stage_scaffold.py`
- `uv run python notes/paper/draft/figures/fig2_matched_readouts.py`
- `uv run python notes/paper/draft/figures/fig3_bridge_failure.py`
- `uv run python notes/paper/draft/figures/fig4_measurement.py`

Use judgment: rerun only what your edits actually touched.

---

## Done Condition

You are done when:

- the remaining prose/package rough edges have been tightened,
- any residual issues are real and specific rather than stale carry-overs,
- the manuscript still stays inside its earned claim boundary,
- figures, numbers, and citations still line up after your edits,
- and you can leave a concise end-state note answering:
  - what you changed,
  - what you verified,
  - what still worries you, if anything.

---

## Optional Reviewer Routing

If you want parallel review help, use the existing prompt files:

- `notes/paper/draft/CITATION_NOVELTY_REVIEW_PROMPT.md`
- `notes/paper/draft/QUANTITATIVE_INTEGRITY_REVIEW_PROMPT.md`
- `notes/paper/draft/PROSE_PACKAGE_REVIEW_PROMPT.md`
- `notes/paper/draft/FINAL_GATE_REVIEW_PROMPT.md`

Recommended order:

1. citation / novelty
2. quantitative integrity
3. prose / package
4. final gate

Why this order:

- first decide what the paper is allowed to say,
- then make sure the numbers support that story,
- then improve presentation,
- then judge whether anything truly blocking remains.

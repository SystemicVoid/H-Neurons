# Final Gate Review Prompt

You are the final-gate reviewer for the paper draft **"Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT."**

## Read first

- `notes/paper/draft/full_paper.md`
- `notes/2026-04-11-strategic-assessment.md`
- `notes/paper/final_flagship_outline_review.md`
- `notes/paper/revised_flagship_outline-v2.md`
- `notes/paper/draft/reviews/final_claim_audit.md`
- `notes/paper/draft/reviews/cross_section_audit.md`

## Current draft state

The latest cleanup pass already did the following:

- Resolved the remaining `[UNCERTAINTY]` placeholders.
- Added `Box B` and `Box C`; `Box A` and `Box D` were already present.
- Added figure references for Figures 1–4.
- Added an `Appendix A` summary so §4.1 appendix references resolve.
- Clarified the SAE `+0.16 pp/alpha` versus `+0.12 pp/alpha` distinction.
- Attached bridge dev-set caveats near the promoted bridge claims.
- Replaced most reader-facing lab jargon and removed the conclusion's `necessary but insufficient` wording.
- Expanded Limitation `L8` to include the StrongREJECT `GPT-4o-mini` confound.
- Updated `number_provenance.md` to make remaining source limitations explicit.

Do **not** assume those fixes are correct merely because they exist. Verify them.

## Your goal

Review the current draft with this goal:

- verify the latest cleanup pass,
- identify any remaining blocking issues,
- and extend review into areas that were not deeply checked in the previous pass.

Known already-fixed items that you should **verify rather than rediscover**:

- Box B and Box C now exist
- Figure references were added
- Appendix A now exists
- `[UNCERTAINTY]` tags were removed
- the conclusion no longer says `necessary but insufficient`
- bridge dev-set caveats were moved closer to promoted claims

## Your tasks

1. Check section-by-section structure against the outline.
2. Check claim boundaries against the strategic assessment.
3. Check cross-section consistency:
   - abstract vs body
   - section files vs assembled `full_paper.md`
   - tables / boxes / appendix / figure references
4. Look for new regressions introduced by the cleanup pass:
   - box insertion that breaks flow
   - appendix summary that overstates evidence
   - provenance wording that quietly weakens or strengthens claims
5. Decide whether the draft is:
   - `READY WITH MINOR POLISH`
   - `NEEDS ONE MORE EDIT PASS`
   - `NOT READY`

## Review rules

1. Review against the **actual repo state**, not stale assumptions in older notes.
2. Treat `notes/2026-04-11-strategic-assessment.md` as the claim governor.
3. Distinguish:
   - `WRONG` = factually incorrect or overclaimed
   - `WEAK` = defensible but under-supported / phrased too strongly
   - `MISSING` = required content absent
   - `STYLE` = prose or readability issue
4. Prefer file/line references and specific replacement guidance.
5. Do not waste time on cosmetic edits unless they affect submission readiness.

## Output format

- Start with a verdict line:
  - `READY WITH MINOR POLISH`
  - `NEEDS ONE MORE EDIT PASS`
  - `NOT READY`
- Then list findings ordered by severity.
- For each finding:
  - severity
  - `file:line`
  - why it matters
  - concrete fix
- End with:
  - `Areas that now look solid`
  - `Areas I did not deeply verify`

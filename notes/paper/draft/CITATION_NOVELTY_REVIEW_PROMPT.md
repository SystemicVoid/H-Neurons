# Citation And Novelty Review Prompt

You are reviewing the literature, citation accuracy, and novelty framing for the current draft of **"Detection Is Not Enough."**

## Primary files

- `notes/paper/draft/section_1_introduction.md`
- `notes/paper/draft/section_3_related_work.md`
- `notes/paper/draft/full_paper.md`

## Governors

- `notes/paper/literature-research/research_arbitration_audit_detection_is_not_enough.md`
- `notes/paper/literature-research/gpt-deep-literature-review.md`
- `notes/paper/literature-research/opus-deep-literature-review.md`
- `notes/2026-04-11-strategic-assessment.md`
- `notes/paper/citations/registry.json`

## Ground truth

- the cited paper markdowns in `papers/`

## Current draft state

The latest cleanup pass mainly affected structure, caveat placement, appendix handling, and prose. It did **not** deeply re-verify the literature layer. Your job is to review that surface directly.

## Your mission

This is **not** a whole-paper review. Focus on:

1. Verifying every major citation claim in §1 and §3 against the actual cited paper.
2. Checking whether must-cite items from the arbitration audit are still missing.
3. Checking whether the novelty framing stays inside the safe frontier:
   - matched cross-method comparison
   - exact bridge failure mode
   - measurement reversals in this steering context
   - four-stage scaffold as synthesis
4. Checking whether the paper now accidentally understates or overstates prior work after the recent cleanup.
5. Suggesting the minimum citation or wording changes needed for claim safety.

## Pay special attention to

- any sentence that sounds like `we are first`
- any sentence that collapses `motivated by prior work` into `novel discovery`
- any paper-facing use of terms like:
  - causal
  - mechanism
  - truthfulness
  - steering utility
  - evaluator dependence
- whether the positive counterexamples are cited fairly

## Review rules

1. Check citation claims against the actual paper markdown in `papers/`, not against literature-review summaries alone.
2. Treat the arbitration audit as the novelty governor.
3. Treat the strategic assessment as the claim-boundary governor.
4. Distinguish:
   - `ACCURACY` = citation does not support the wording
   - `NOVELTY` = novelty framing is unsafe
   - `MISSING-CITE` = missing required paper or comparison
5. Do **not** verify numeric results unless they directly affect citation correctness.

## Output format

- Findings first, ordered by severity.
- For each finding:
  - `ACCURACY` / `NOVELTY` / `MISSING-CITE`
  - `file:line`
  - cited paper(s)
  - why the current wording is wrong or risky
  - exact safer replacement wording if possible
- End with a compact table:
  - `Citation/claim`
  - `Status = ACCURATE / TOO STRONG / TOO WEAK / MISSING`

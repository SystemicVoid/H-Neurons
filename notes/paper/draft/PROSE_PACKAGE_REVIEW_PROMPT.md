# Prose And Submission Package Review Prompt

You are reviewing the current draft of **"Detection Is Not Enough"** as a submission-ready paper draft, not as an internal memo.

## Read first

- `notes/paper/draft/full_paper.md`
- `notes/paper/final_flagship_outline_review.md`

## Current draft state

The latest pass already touched:

- abstract
- §4.1 / §4.2 clarification language
- §5.3 bridge box insertion
- §6.3 evaluator-disagreement box insertion
- §7 framework qualifier
- §8 limitations prose
- §9 conclusion
- Appendix A

You should assume those sections are especially likely to contain patchwork flow problems, duplicated caveats, or box-insertion awkwardness.

## Your job

Evaluate whether the paper now reads cleanly after the latest patch pass.

Focus on:

1. Flow and transitions across sections.
2. Whether the newly added boxes and appendix summary are integrated naturally.
3. Whether the paper still contains hidden lab voice:
   - workflow language
   - memo-like phrasing
   - hedges that read like TODO markers
   - abrupt caveat dumps that should be folded into prose
4. Whether paragraphs open with clear topic sentences.
5. Whether the paper repeats the same claim too many times with slightly different wording.
6. Whether the conclusion, abstract, and section openings feel aligned in tone and scope.
7. Whether the figure references feel informative rather than bolted on.

## Review rules

1. This is a prose / package review, not a raw citation or provenance audit.
2. Flag technical issues only when they materially damage readability or paper credibility.
3. Distinguish:
   - `STRUCTURE`
   - `CLARITY`
   - `REPETITION`
   - `TONE`
   - `LOCAL STYLE`
4. Prefer concrete rewrite suggestions over generic comments.

## Output format

- Findings first, ordered by severity:
  - `STRUCTURE`
  - `CLARITY`
  - `REPETITION`
  - `TONE`
  - `LOCAL STYLE`
- Each finding should include:
  - `file:line`
  - why it is a problem
  - a concrete rewrite suggestion
- Then give:
  - `Best section as currently written`
  - `Weakest section as currently written`
  - `Top 5 edits that would most improve end-to-end readability`

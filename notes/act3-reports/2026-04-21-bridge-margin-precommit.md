# Bridge logprob-margin mechanism check — **precommit**

**Date:** 2026-04-21 (written *before* the full 57+200 run executes).

**Purpose of this note.** The Gate 2 sanity peek (5 cases) surfaced two
phenomena that invite post-hoc window retuning: (i) magnitudes at first-3
tokens are large (5–35 nats, not 0.5–5), and (ii) one B-evasion case showed
a larger negative shift than any A-substitution case, apparently because
token 1 under ITI raises `p("The …")` as a generic answer-frame rather
than as an entity commitment.

Changing the primary endpoint after seeing that data would be exactly the
kind of researcher-degrees-of-freedom move that confirmatory analyses are
supposed to exclude. This note locks what is confirmatory, what is
secondary, what is diagnostic, and what is exploratory, **before** any
further data lands, and pins the interpretation logic for every plausible
outcome so nothing has to be decided by eye after the fact.

---

## Locked analysis plan

### Confirmatory primary

**First-3-token shift_nats** per case, cohort A vs. cohort B and
cohort A vs. cohort D, two-sample bootstrap mean difference with a
one-sided permutation test (H1: `mean(A) < mean(B or D)`).

Why: first-3 tokens is exactly the window where the ITI hook fires
(`decode_scope=first_3_tokens`). Any shift attributable to the
intervention must be measurable there.

### Locked secondary (sensitivity)

**Full-continuation shift_nats** per case, same cohort comparisons.

Role: confirm the finding is not an artefact of the 3-token boundary.

### Mandatory diagnostic

**Per-position shift decomposition** at positions 0, 1, 2 (0-indexed) for
every cohort, with bootstrap mean + 95% CI. Answers "where inside the
manipulated prefix does the shift live?"

### Exploratory (labeled as such in the paper)

1. **Tokens-2–3 window**: sum of shifts at positions 1 and 2 (0-indexed).
   Still entirely inside the hooked surface; drops position 0, which the
   sanity run suggests may be dominated by answer-frame initiation
   (`"The …"`) rather than entity commitment. **Token 4 is explicitly not
   added**: it is outside the intervention scope, so a 2–4 window is
   neither a clean confirmatory endpoint nor a clean causal readout.

2. **Within-A baseline-sign split**: partition cohort A by
   `sign(first3.delta_base)`. The two subgroups potentially mix two
   mechanisms:
   - `A_pos` (baseline already prefers gold): clean ITI-induced reversal.
   - `A_neg` (baseline ambivalent/wrong-leaning): ITI amplifies an already
     shaky baseline preference.
   Both are real, but the story is different.

---

## Precommitted interpretation logic (decide reading *before* the numbers land)

### Broad confirmatory claim

> ITI harms the gold-vs-wrong log-likelihood margin on the manipulated
> prefix in R→W cases and reverses the sign of that shift in W→R rescue
> cases.

This claim stands on: A first-3 shift is negative and its CI excludes
zero; C first-3 shift is positive and its CI excludes zero.

### Narrower mechanism claim (A vs. B)

Readout logic (decided now, so it cannot be reshaped by the numbers):

- If **A stays more negative than B on tokens 2–3**: the substitution-
  specific story survives the opener confound. Headline "bridge fails via
  wrong-entity substitution" holds, with the caveat that token 0 carries
  additional generic-opener variance.
- If **B is strongly negative on first-3 but that attenuates on
  tokens 2–3**: report as "early answer-framing confound at token 0 plus a
  weaker substitution-specific content-token effect at tokens 1–2." This
  is a narrower, honest version of the mechanism claim.
- If **B remains as negative as A on tokens 2–3**: drop the A<B headline.
  Broaden to "ITI compresses the gold-vs-wrong margin on R→W flips
  generally; the behavioral taxonomy does not correspond to a distinct
  margin-shift signature."

### Baseline-sign subgroup

- If `A_pos` and `A_neg` are both strongly negative: a single mechanism
  hypothesis survives. Report as primary.
- If only `A_pos` is strongly negative while `A_neg` is near zero: the
  substitution headline is driven by clean ITI-induced reversals; the
  `A_neg` subgroup is amplification of baseline ambivalence and should be
  reported as a distinct phenomenon, not merged.

### Language / framing

- Report **shift_per_token** or the per-position decomposition alongside
  total nats. Do not characterise these as "small margin compression";
  with 3-token windows, per-position shifts of 1–5 nats are large early
  log-likelihood effects on the manipulated surface.
- The prose must say "log-likelihood margin" and "teacher-forcing", not
  "generation probability". We measure what the model *would* assign, not
  the sampled trajectory.

---

## What this note is *not*

- It is not a hypothesis test over the diagnostic / exploratory analyses.
  Those are reported transparently, labeled exploratory, with CIs, and
  they do not carry the main claim.
- It is not permission to rewrite the primary after the full run. Only
  numerical errors in the scoring pipeline can change the primary.

## Pointer

- Scorer: `scripts/score_bridge_margins.py`
- Analyzer: `scripts/analyze_bridge_margins.py`
- Outputs: `data/gemma3_4b/analysis/bridge_margins/test/`
- Gate 2 sanity: `data/gemma3_4b/analysis/bridge_margins/test_sanity/margins.jsonl`

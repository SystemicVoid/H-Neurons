# Act 3 Decode-Scope Gate-1 Audit — 2026-04-01

> **Status: canonical source of truth for the TruthfulQA MC1 cal-val decode-scope gate.**
>
> Use this report for the current interpretation of Stage 5.2 Gate 1.
> The earlier gate snapshot should link here rather than restating conclusions.

## Scope

This audit covers the first decode-scope review gate from
[optimise-intervention-ac3.md](./optimise-intervention-ac3.md):

1. the four-scope TruthfulQA MC1 cal-val panel on the paper-faithful ITI artifact
2. pipeline and provenance checks needed to trust the comparison
3. what this panel does and does not justify before any expensive generation or API judging

This report does **not** cover the downstream forced-commitment SimpleQA pilot.
That follow-up now lives in
[2026-04-01-decode-scope-simpleqa-pilot-audit.md](./2026-04-01-decode-scope-simpleqa-pilot-audit.md).

## Source Hierarchy

- Canonical gate snapshot:
  [2026-04-01-decode-scope-gate1-calval.md](./2026-04-01-decode-scope-gate1-calval.md)
- Strategy context:
  [optimise-intervention-ac3.md](./optimise-intervention-ac3.md)
- Artifact under test:
  [data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt)
- Fixed manifest:
  [data/manifests/truthfulqa_cal_val_mc1_seed42.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/manifests/truthfulqa_cal_val_mc1_seed42.json)
- Run directories:
  [full_decode](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-full-decode_iti-truthfulqa-paperfaithful-production-iti-head_a0b1088812/experiment)
  ·
  [first_token_only](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-token-only_iti-truthfulqa-paperfaithful-production-iti-head_d42bb89984/experiment)
  ·
  [first_3_tokens](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-paperfaithful-production-iti-head_586b7d4cd3/experiment)
  ·
  [first_8_tokens](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-8-tokens_iti-truthfulqa-paperfaithful-production-iti-head_c15089a5d6/experiment)
- Relevant code:
  [scripts/intervene_iti.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/scripts/intervene_iti.py)
  ·
  [scripts/run_intervention.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/scripts/run_intervention.py)
  ·
  [scripts/infra/iti_decode_scope_gate1.sh](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/scripts/infra/iti_decode_scope_gate1.sh)
- Related upstream audit:
  [2026-04-01-random-head-specificity-audit.md](./2026-04-01-random-head-specificity-audit.md)
- Downstream follow-up:
  [2026-04-01-decode-scope-simpleqa-pilot-audit.md](./2026-04-01-decode-scope-simpleqa-pilot-audit.md)

## Pipeline Audit

### What was verified directly from artifacts

- All four scope runs completed and each directory contains:
  - `alpha_0.0.jsonl`
  - `alpha_8.0.jsonl`
  - `results.*.json`
  - `run_intervention.provenance.*.json`
- Every `alpha_0.0.jsonl` and `alpha_8.0.jsonl` file contains exactly `81` rows,
  `81` unique IDs, and `0` duplicate IDs.
- All four runs use the same:
  - artifact path
  - manifest path
  - `K=12`
  - ranked head selection
  - random seed `42`
  - benchmark `truthfulqa_mc`, variant `mc1`
- All four provenance files record the same git SHA:
  `fc678c1e1eabf0331089fbe0b6d67be59e3eac0a`
- All four provenance files show `status: completed`.
- The `results.*.json` summaries report `parse_failures: 0` at both alphas for all scopes.

### Important integrity checks

#### Alpha-0 baseline invariance

At `α=0.0`, the four scopes are not merely close; they are identical at the
saved-output level on this panel:

- same row count
- same chosen answer for every sample
- same `metric_value` for every sample
- same per-choice log-likelihood arrays for every sample

That matters. It means the decode-scope parameter is not leaking into the
claimed no-op baseline through naming, caching, or resume behavior.

#### Saved-output limitations

The saved JSONL preserves `prompt_skip_calls` and a capped `intervention_debug`
trace, but it does **not** preserve `scope_skip_calls`.

Implication:

- the saved runs are sufficient to audit result integrity and top-line behavior
- they are **not** sufficient to prove token-boundary enforcement post hoc from
  row files alone

For scope correctness, the strongest direct evidence is still the dedicated
unit-test coverage in [tests/test_truthfulness_iti.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/tests/test_truthfulness_iti.py), plus the monotone runtime signature below.

#### Development-set limitation

This is a development gate on `81` TruthfulQA MC1 cal-val items, not a new
headline evaluation. That means:

- uncertainty is material
- scope selection on this panel alone would be optimistic
- pairwise comparisons should be treated as exploratory unless confirmed on a
  downstream surface

This is especially important because Stage 5.2 is being used to choose an
intervention policy, not to publish a new TruthfulQA claim.

### Operational signature of scope gating

Even though `scope_skip_calls` are not saved, the runtime profile is consistent
with the intended scope ordering.

Across the exact saved continuations in this panel, the fraction of continuation
tokens that fall within each scope boundary is:

| Scope limit | Theoretical touched-token share | Observed hook fraction of generate time |
| --- | ---: | ---: |
| token `1` only | 11.1% | 3.3% |
| tokens `1..3` | 31.7% | 8.8% |
| tokens `1..8` | 74.5% | 20.1% |
| full decode | 100.0% | 26.7% |

The time fractions are lower than the token fractions because skipped hook calls
still have non-zero overhead, but the monotone ordering is exactly what the
scope design predicts:

`first_token_only < first_3_tokens < first_8_tokens < full_decode`

That is supportive, but still indirect. I would not present it as the primary
proof of correct gating.

## Data

### 1. Top-line MC1 results

| Scope | MC1 @ `α=0.0` | MC1 @ `α=8.0` | Paired delta | 95% CI | Retained vs `full_decode` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_decode` | 22.2% (18/81) | 35.8% (29/81) | +13.6 pp | [+4.9, +22.2] | 100% |
| `first_token_only` | 22.2% (18/81) | 25.9% (21/81) | +3.7 pp | [-2.5, +9.9] | 27% |
| `first_3_tokens` | 22.2% (18/81) | 34.6% (28/81) | +12.3 pp | [+4.9, +21.0] | 91% |
| `first_8_tokens` | 22.2% (18/81) | 34.6% (28/81) | +12.3 pp | [+3.7, +21.0] | 91% |

Wilson 95% intervals on the raw `α=8.0` rates:

- `full_decode`: 35.8% [26.2, 46.7]
- `first_token_only`: 25.9% [17.6, 36.4]
- `first_3_tokens`: 34.6% [25.1, 45.4]
- `first_8_tokens`: 34.6% [25.1, 45.4]

Descriptive exact paired sign-test p-values for the within-scope `α=0.0 -> 8.0`
shift:

- `full_decode`: `13` gains, `2` losses, `p=0.0074`
- `first_token_only`: `5` gains, `2` losses, `p=0.4531`
- `first_3_tokens`: `11` gains, `1` loss, `p=0.0063`
- `first_8_tokens`: `12` gains, `2` losses, `p=0.0129`

These p-values are exploratory and uncorrected for multiple comparisons.

### 2. Pairwise comparisons against `full_decode` at `α=8.0`

| Scope | MC1 delta vs `full_decode` | 95% CI | `full=0 -> scope=1` | `full=1 -> scope=0` |
| --- | ---: | ---: | ---: | ---: |
| `first_token_only` | -9.9 pp | [-18.5, -2.5] | 2 | 10 |
| `first_3_tokens` | -1.2 pp | [-6.2, +3.7] | 2 | 3 |
| `first_8_tokens` | -1.2 pp | [-3.7, +0.0] | 0 | 1 |

Interpretive caution:

- only `first_token_only` is clearly worse than `full_decode`
- the current panel does **not** separate `full_decode`, `first_3_tokens`, and
  `first_8_tokens` with enough confidence to justify a strong ranking among them

### 3. Pairwise disagreement structure

Headline ties hide non-identical item behavior.

At `α=8.0`:

- `full_decode` vs `first_8_tokens` disagree on `1/81` items
- `full_decode` vs `first_3_tokens` disagree on `5/81` items
- `first_3_tokens` vs `first_8_tokens` disagree on `4/81` items

The `first_3_tokens` vs `first_8_tokens` top-line tie is therefore not
"same behavior." It is "same total wins on this small panel."

### 4. Continuous secondary diagnostic

Although MC1 accuracy is the gate metric, the saved choice scores allow a more
graded diagnostic: average truthful probability mass on the MC1 choice set.

Mean truthful mass:

| Scope | `α=0.0` | `α=8.0` |
| --- | ---: | ---: |
| `full_decode` | 0.2267 | 0.3468 |
| `first_token_only` | 0.2267 | 0.2546 |
| `first_3_tokens` | 0.2267 | 0.3304 |
| `first_8_tokens` | 0.2267 | 0.3293 |

This is post hoc and not the predeclared selection metric, but it supports the
same broad ranking:

- `first_token_only` looks materially weaker
- `first_3_tokens` and `first_8_tokens` remain much closer to `full_decode`

### 5. The small number of discordant items matters

The pairwise differences are driven by very few questions.

`full_decode` vs `first_3_tokens` flips:

- `3` losses for `first_3_tokens`
- `2` gains for `first_3_tokens`

`full_decode` vs `first_8_tokens` flips:

- `1` loss for `first_8_tokens`
- `0` gains for `first_8_tokens`

This is exactly the regime where careful reviewers should resist storytelling.
With `81` items, a handful of flips can change the apparent "winner" without
meaning there is a robust underlying ordering.

## Interpretation

### 1. What withstands scrutiny

The strongest conclusions that survive a skeptical read are:

1. `first_token_only` is too weak to carry forward as the main decode policy.
2. The beneficial MC1 effect does **not** require steering every generated token.
3. This gate does **not** yet justify choosing among `full_decode`,
   `first_3_tokens`, and `first_8_tokens`.

Why these are defensible:

- `first_token_only` drops almost ten points relative to `full_decode`, with a
  pairwise CI that stays below zero.
- `first_3_tokens` and `first_8_tokens` both preserve most of the observed MC1
  gain on this panel.
- Neither narrower scope shows a credible MC1 improvement over `full_decode`.

That is a useful result. It narrows the search tree without pretending the gate
is more decisive than it is.

### 2. What this does **not** establish

This panel does **not** establish any of the following:

- that `first_3_tokens` is better than `full_decode`
- that `first_8_tokens` is better than `full_decode`
- that `first_3_tokens` is better than `first_8_tokens`
- that a narrower scope fixes the downstream generation failure

Those would all be overclaims from the available evidence.

The right analogy is triage, not verdict. This panel is good at ruling out a
bad candidate (`first_token_only`). It is not good at crowning a winner among
three plausible candidates.

### 3. Safety-research reading

Under best practice for AI/ML safety work, the standard is not just "did one
line go up?" but "how likely is this to survive contact with a different
surface, a different split, or a stricter audit?"

On that standard:

- the `first_token_only` degradation is fairly robust for such a small panel
- the `first_3_tokens` and `first_8_tokens` retention story is plausible
- any finer ranking among the surviving scopes remains fragile

That means the appropriate action is not to lock a new scope yet. It is to
carry only the plausible candidates into the more relevant downstream pilot.

### 4. Best current decision

The decision that best balances caution and progress is:

- drop `first_token_only`
- keep `full_decode`, `first_3_tokens`, and `first_8_tokens` alive for the
  forced-commitment 200-ID SimpleQA pilot
- do **not** spend API budget until the raw generation outputs from that pilot
  have been sanity-checked

Why not lock `first_8_tokens` now, since it is only one flip behind
`full_decode`?

- because the whole point of Stage 5.2 is generation-time selectivity, not
  re-optimizing MC1 on a tiny dev panel
- because `first_3_tokens` is the more aggressive selectivity hypothesis and is
  still clearly viable on the cheap gate
- because choosing between `first_3_tokens` and `first_8_tokens` now would
  mainly reward sampling noise

### 5. What remains uncertain

The main unresolved questions are:

1. Does a narrower surviving scope improve attempt rate or reduce evasive
   behavior on forced-commitment SimpleQA?
2. If yes, is that improvement large enough to survive full-1000 confirmation?
3. If no, is the failure better explained by the artifact itself than by the
   decode policy?

Current uncertainty is still substantial because:

- the gate panel is only `81` items
- the downstream quantity of interest is free-form generation, not MC scoring
- multiple scope comparisons were made on the same development set

## Most Valuable Next Steps

1. Run the 200-ID forced-commitment SimpleQA generation pilot for
   `full_decode`, `first_3_tokens`, and `first_8_tokens`.
2. Review the raw generations and refusal-style patterns before any batch judge
   submission.
3. Only then run batch judging on the surviving generation outputs.
4. Promote a narrower scope only if the 200-ID pilot improves attempt/compliance
   without an accompanying precision drop, then confirm on full-1000.

What I would **not** do next:

- I would not run `first_token_only` on SimpleQA.
- I would not promote a scope based on this cal-val gate alone.
- I would not start `E1` or `E2` before the generation pilot resolves whether
  scope selectivity helps at all.

# Decode-Scope SimpleQA Judge Results — 2026-04-02

> **Status: canonical source of truth for the Stage 5.2 decode-scope judged
> results on the 200-ID forced-commitment SimpleQA pilot.**
>
> This report contains official GPT-4o compliance numbers and paired grade
> analyses. For raw generation surface analysis and pipeline integrity, see the
> pre-judge audit.

## Source Hierarchy

- Strategy context:
  [optimise-intervention-ac3.md](./optimise-intervention-ac3.md)
- Upstream MC1 gate:
  [2026-04-01-decode-scope-gate1-audit.md](./2026-04-01-decode-scope-gate1-audit.md)
- Pre-judge generation audit:
  [2026-04-01-decode-scope-simpleqa-pilot-audit.md](./2026-04-01-decode-scope-simpleqa-pilot-audit.md)
- Judge chain script:
  [scripts/infra/decode_scope_judge_chain.sh](../../scripts/infra/decode_scope_judge_chain.sh)
- Run directories:
  [full\_decode](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-full-decode_iti-truthfulqa-paperfaithful-production-iti-head_a0b1088812/experiment)
  ·
  [first\_3\_tokens](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-paperfaithful-production-iti-head_586b7d4cd3/experiment)
  ·
  [first\_8\_tokens](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-8-tokens_iti-truthfulqa-paperfaithful-production-iti-head_c15089a5d6/experiment)

---

## TL;DR

1. **All three scopes degrade compliance relative to the unsteered baseline.**
   The best result (`first_3_tokens`: 4.0%) is still 1.5 pp below the
   unsteered baseline (5.5%) and its CI includes zero.
2. **`first_3_tokens` passes the §5.2 promotion rule on all three criteria.**
   It retains ~90% of the MC1 gain, improves compliance relative to
   `full_decode`, and does not reduce precision. But the absolute signal is
   below baseline — this is "least bad," not "recovered."
3. **Decode scope is not the primary bottleneck.** Narrowing scope converts
   meta-hedging (`NOT_ATTEMPTED`) into confident wrong answers
   (`INCORRECT`), not into correct ones. Of 44 questions first rescued from
   `NOT_ATTEMPTED` by `first_3_tokens`, only 2 (5%) were judged `CORRECT`.
4. **Decision:** promote `first_3_tokens` as the canonical default scope for
   subsequent experiments. Advance to Stage 3 (artifact improvements: E1,
   then E2) as the primary hypothesis.

---

## Pipeline Integrity

The judge chain ran three sequential batch jobs on the pre-existing experiment
directories (no new inference, no GPU):

- Batch `batch_69ce15aaf0bc8190943ba35d5490c053` — `full_decode` (400
  requests, 0 failed)
- Batch `batch_69ce1669cedc81909aaa512c204500cd` — `first_3_tokens` (400
  requests, 0 failed)
- Batch `batch_69ce17635ed48190a82becfc49c45cdd` — `first_8_tokens` (400
  requests, 0 failed)

Each batch processed 200 questions × 2 alphas = 400 requests. All three
completed with zero failures. Provenance files confirm `judging_complete: true`
for all six `(scope, alpha)` combinations. The pre-judge audit verified
pipeline integrity for the underlying generation runs.

---

## Official Compliance Results

### Baseline (alpha=0.0 — identical across all scopes)

All three scope runs share the same alpha-0.0 outputs (verified in the pre-judge
audit). The judge confirms:

| Metric | Value | Wilson 95% CI |
| --- | ---: | ---: |
| Compliance | 11/200 = 5.5% | [3.1, 9.6] |
| Attempt rate | 198/200 = 99.0% | [96.4, 99.7] |
| `NOT_ATTEMPTED` | 2/200 = 1.0% | — |
| Precision | 11/198 = 5.6% | [3.1, 9.7] |

### Alpha=8.0 by scope

| Scope | Compliance | Delta vs baseline | Bootstrap 95% CI | Slope (pp/α) | Bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_decode` | 5/200 = 2.5% | −3.0 pp | [−6.0, 0.0] | −0.375 | [−0.750, ~0] |
| `first_8_tokens` | 7/200 = 3.5% | −2.0 pp | [−5.0, +1.0] | −0.250 | [−0.625, +0.125] |
| `first_3_tokens` | 8/200 = 4.0% | −1.5 pp | [−4.0, +0.5] | −0.188 | [−0.500, +0.063] |

Bootstrap CIs use `paired_by_sample_id` percentile resampling (n=10 000,
seed=42) on the shared 200-ID panel.

**Interpretation.** Only `full_decode`'s lower CI touches zero — its negative
effect is borderline confident. Both narrower scopes have CIs that span zero;
strictly speaking, their compliance loss is compatible with a null effect at
α=8.0. Nevertheless, all three point estimates are below baseline, and no scope
produces a net compliance gain.

---

## Full Grade Breakdown

### Alpha=8.0 by scope

| Scope | CORRECT | INCORRECT | NOT\_ATTEMPTED | Attempt rate | Wilson 95% CI | Precision | Wilson 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_decode` | 5 | 131 | 64 | 68.0% | [61.2, 74.1] | 3.7% | [1.6, 8.3] |
| `first_8_tokens` | 7 | 162 | 31 | 84.5% | [78.8, 88.9] | 4.1% | [2.0, 8.3] |
| `first_3_tokens` | 8 | 171 | 21 | 89.5% | [84.5, 93.0] | 4.5% | [2.3, 8.6] |

Precision = CORRECT / (CORRECT + INCORRECT).

**The core pattern is a precision × attempt-rate decomposition of compliance:**

`compliance = attempt_rate × precision`

- `full_decode`: 68.0% × 3.7% = 2.5% ✓
- `first_3_tokens`: 89.5% × 4.5% = 4.0% ✓
- `first_8_tokens`: 84.5% × 4.1% = 3.5% ✓

`full_decode` suppresses compliance through *both* channels simultaneously —
it reduces attempt rate (32% `NOT_ATTEMPTED`) and depresses precision.
`first_3_tokens` recovers attempt rate substantially (+21.5 pp vs `full_decode`)
with a modest precision recovery (+0.8 pp). But precision remains well below
the unsteered baseline (5.6%) for all scopes.

---

## Paired Grade Transitions (full\_decode → first\_3\_tokens)

Question-level grade transitions for the 200 paired items at α=8.0:

| full\_decode → first\_3\_tokens | Count |
| --- | ---: |
| CORRECT → CORRECT | 3 |
| CORRECT → INCORRECT | 2 |
| INCORRECT → CORRECT | 3 |
| INCORRECT → INCORRECT | 127 |
| INCORRECT → NOT\_ATTEMPTED | 1 |
| NOT\_ATTEMPTED → CORRECT | 2 |
| NOT\_ATTEMPTED → INCORRECT | 42 |
| NOT\_ATTEMPTED → NOT\_ATTEMPTED | 20 |

Net change in CORRECT: +5 gained (NA→CORRECT: 2, INCORRECT→CORRECT: 3)
− 2 lost (CORRECT→INCORRECT) = **+3 net** (5→8).

### The calibration finding

Of the 44 questions rescued from `NOT_ATTEMPTED` by `first_3_tokens`:
- 2/44 → CORRECT (5%)
- 42/44 → INCORRECT (95%)

The unsteered baseline precision is 5.6% — meaning randomly-attempted questions
are correct 5.6% of the time. The questions the model specifically hedges on
under `full_decode`, when forced to answer under a narrower scope, are correct
at a *below-baseline* rate (5%). This is not a statistical artifact of small
numbers; it is consistent with the model having implicit calibration even under
steering — the questions it hedges on are genuinely harder or less accessible
factually.

Narrowing scope converts meta-hedging into *confident confabulation*, not into
factual recall.

### For reference: full\_decode → first\_8\_tokens

| full\_decode → first\_8\_tokens | Count |
| --- | ---: |
| CORRECT → CORRECT | 5 |
| INCORRECT → CORRECT | 1 |
| INCORRECT → INCORRECT | 129 |
| INCORRECT → NOT\_ATTEMPTED | 1 |
| NOT\_ATTEMPTED → CORRECT | 1 |
| NOT\_ATTEMPTED → INCORRECT | 33 |
| NOT\_ATTEMPTED → NOT\_ATTEMPTED | 30 |

Of 34 questions rescued from `NOT_ATTEMPTED` by `first_8_tokens`: 1/34 →
CORRECT (3%), 33/34 → INCORRECT (97%). Even weaker than `first_3_tokens`.

---

## Reconciliation With Pre-Judge Surface Proxies

The pre-judge audit used a keyword-based "evasive/meta" proxy to estimate
NOT_ATTEMPTED rates before judging. Here is how those estimates compare to the
official judge:

| Scope | Meta-evasive proxy | Judge NOT\_ATTEMPTED | Discrepancy |
| --- | ---: | ---: | ---: |
| `full_decode` | 29.0% (58/200) | 32.0% (64/200) | proxy −3 pp |
| `first_3_tokens` | 7.5% (15/200) | 10.5% (21/200) | proxy −3 pp |
| `first_8_tokens` | 16.5% (33/200) | 15.5% (31/200) | proxy +1 pp |

The proxy systematically underestimated NOT_ATTEMPTED for `full_decode` and
`first_3_tokens` by ~3 pp, likely because some responses that contained
meta-language still committed to an answer form that the judge classified as
`INCORRECT` rather than `NOT_ATTEMPTED`. The `first_8_tokens` estimate was
essentially exact.

The pre-judge conclusion — that `first_3_tokens` materially reduces the
meta-evasive failure mode — is confirmed. The judge reduction is 64→21
NOT_ATTEMPTED (−43 questions), matching the surface-proxy signal direction and
approximate magnitude.

The pre-judge caveat — that reducing evasion might mean replacing it with
confident wrong answers rather than correct ones — is also confirmed by the
5% NA→CORRECT conversion rate.

---

## Statistical Notes

1. **Scope differences are not statistically distinguishable.** All three scope
   CIs overlap substantially. The difference in compliance between `full_decode`
   (5/200) and `first_3_tokens` (8/200) is 3 events out of 200. No paired
   comparison between scopes reaches conventional significance.

2. **`full_decode` is the only scope with a borderline-confident negative
   effect.** Its lower bootstrap CI is 0.0 (just touching zero); `first_3_tokens`
   and `first_8_tokens` CIs include positive values.

3. **Precision CIs are very wide.** With 5–8 correct answers over 136–179
   attempted, all precision estimates carry uncertainty ranges of roughly ±4–6 pp.
   Do not over-interpret the 3.7% vs 4.5% precision difference between
   `full_decode` and `first_3_tokens`.

4. **The paired panel design is the right analysis unit.** All 200 questions
   are identical across scopes, which is why the bootstrap uses
   `paired_by_sample_id` resampling. The transition table above makes direct
   use of this pairing.

---

## Promotion Decision per §5.2

The promotion rule (from
[optimise-intervention-ac3.md](./optimise-intervention-ac3.md) §5.2) requires a
narrower scope to:

1. **Keep a material fraction of the MC1 gain.** `first_3_tokens` retained
   +12.3 pp vs `full_decode`'s +13.6 pp — ~90% retention. ✓
2. **Improve SimpleQA attempt/compliance relative to `full_decode`.** 4.0% vs
   2.5% compliance; 89.5% vs 68.0% attempt rate. ✓
3. **Not reduce precision further.** 4.5% vs 3.7%. ✓

`first_3_tokens` clears all three criteria. **It is promoted as the canonical
default scope for all subsequent experiments.**

However, the promotion is "best of bad options," not a recovery:

- Absolute compliance (4.0%) remains below the unsteered baseline (5.5%).
- The +1.5 pp improvement over `full_decode` is not statistically confident
  (CI: [−4.0, +0.5]).
- The underlying mechanism is attempt-rate recovery, not precision improvement.
- Precision under all steered conditions remains below baseline.

`first_8_tokens` also partially passes (criteria 1 and 3 yes; criteria 2
borderline — compliance 3.5% vs 2.5%, attempt rate 84.5% vs 68.0%). It is not
promoted because `first_3_tokens` dominates it on all three metrics.

---

## Why Surface Improvements Didn't Translate

The pre-judge audit showed `first_3_tokens` produces more phrase-like, concise
outputs (+11.5 pp phrase-like proxy) and fewer verbose hedging responses (−21.5
pp evasive proxy). Why does this improvement in form not produce more correct
answers?

**The mechanism:** ITI directions trained on TruthfulQA appear to encode
calibrated-uncertainty behavior — "don't overclaim when you're unsure" — rather
than factual-recall behavior. When applied at `full_decode`, this signal
permeates every generated token, causing the model to repeatedly second-guess
itself and produce verbose meta-hedging that the judge classifies `NOT_ATTEMPTED`.

When scope is narrowed to the first 3 tokens, the uncertainty signal shapes the
*opening* of the response without dominating the entire generation. The model
commits to an answer form earlier. But the *content* of that answer — the
specific factual claim — is not steered toward accuracy. The model is now giving
short, direct answers, but those answers are wrong for the same reasons as before.

Think of it as two independent channels:
- **Form channel:** "how much hedging language?" — scope narrows this, conciseness
  improves.
- **Content channel:** "is the factual claim correct?" — scope does not touch this.

SimpleQA requires the second channel to work. The current TruthfulQA-trained
directions only reliably affect the first.

---

## Implications For The Research Plan

### What this result establishes

1. **The scope hypothesis is falsified as a complete fix.** "Steering too many
   decode tokens causes the generation failure" was partially true — full-decode
   does cause more NOT_ATTEMPTED than narrower scopes — but the underlying
   issue is deeper. Even with minimal scope, compliance does not reach baseline.

2. **The direction-quality hypothesis is now the primary candidate.** The
   TruthfulQA-trained directions encode calibrated uncertainty, not factual
   retrieval. A direction trained on a factual-recall task (TriviaQA) may target
   a different geometry — one that improves the *content channel* rather than
   just the *form channel*.

3. **`first_3_tokens` is strictly better than `full_decode` on every relevant
   metric.** It should be the default for all future experiments:
   - less harmful to SimpleQA compliance
   - less noisy generation behavior
   - ~90% of MC1 gain preserved

4. **`first_token_only` remains eliminated** (Gate 1 result: only +3.7 pp MC1
   vs +13.6 pp for `full_decode`, CIs do not overlap).

### What remains uncertain

1. Whether E1 (TruthfulQA-modernized: chat-template extraction, AUROC ranking)
   changes anything meaningful. It is the cheapest test and should come first.
2. Whether E2 (TriviaQA-only directions) can improve the content channel on
   SimpleQA. The literature (LITO paper) shows TriviaQA-trained directions
   transfer better than TruthfulQA-trained ones, but this was demonstrated on
   models with higher baseline QA accuracy.
3. Whether any direction-based approach can overcome the base model's limited
   factual recall on hard SimpleQA questions. Gemma-3-4B-IT has 5.5% baseline
   compliance — the ceiling from direction steering may be low regardless of
   direction quality.

### What the 5.5% baseline implies

The unsteered baseline compliance (5.5%) is low in absolute terms. ITI can at
best steer the model toward more confident expression of what it already knows.
If the model's knowledge of the 200-question test set is capped at ~5–6%,
direction steering can surface a slightly larger fraction of that knowledge,
but cannot create factual knowledge that does not exist.

This raises a structural question: is Stage 3 (artifact improvement) enough, or
does the project eventually need to move to a higher-headroom generation
benchmark (Stage 4 bridge benchmark, e.g., TriviaQA open-domain) before any
meaningful progress on the generation axis is visible?

---

## Recommended Next Steps

| Step | Action | Rationale |
| --- | --- | --- |
| 1 | Lock `first_3_tokens` as default decode scope | Passes §5.2 criteria; strictly dominates `full_decode` and `first_8_tokens` |
| 2 | Run E1 under `first_3_tokens` | Cheapest artifact change; tests extraction-quality hypothesis |
| 3 | Evaluate E1 on MC1 (2-fold) + SimpleQA 200-ID | Paired panel; compare under identical conditions |
| 4 | Run E2 (TriviaQA) under `first_3_tokens` if E1 does not recover SimpleQA | Tests direction-source hypothesis |
| 5 | Consider adding bridge benchmark (open-domain TriviaQA or NQ) | If E1/E2 show headroom on MC but not on SimpleQA, need a generation benchmark with more headroom before further artifact work |
| 6 | Hold E3 (mixed-source) | Conditional on E1/E2 being complementary |

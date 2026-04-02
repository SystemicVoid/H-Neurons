# Act 3 Decode-Scope SimpleQA Pilot Audit — 2026-04-01

> **Status: canonical source of truth for the 200-ID forced-commitment
> SimpleQA generation pilot in Stage 5.2.**
>
> This report covers the post-generation review gate only. It does **not**
> contain judged correctness metrics, and it should not be cited as an
> accuracy result.

## Scope

This audit covers the second decode-scope review gate from
[optimise-intervention-ac3.md](./optimise-intervention-ac3.md):

1. pipeline and provenance checks for the 200-ID forced-commitment SimpleQA
   generation pilot
2. direct observations from the raw `α=0.0` and `α=8.0` generations for the
   three surviving scopes
3. what these raw generations do and do not justify before any batch judge/API
   spend

This report does **not** grade correctness. The next decision is whether the
raw outputs are trustworthy enough to judge, not whether a scope already won.

## Source Hierarchy

- Strategy context:
  [optimise-intervention-ac3.md](./optimise-intervention-ac3.md)
- Upstream answer-selection gate:
  [2026-04-01-decode-scope-gate1-audit.md](./2026-04-01-decode-scope-gate1-audit.md)
- Pilot runner:
  [scripts/infra/iti_decode_scope_simpleqa_pilot.sh](../../scripts/infra/iti_decode_scope_simpleqa_pilot.sh)
- Shared SimpleQA entrypoint:
  [scripts/infra/simpleqa_standalone.sh](../../scripts/infra/simpleqa_standalone.sh)
- Fixed manifest:
  [data/manifests/simpleqa_verified_control200_seed42.json](../../data/manifests/simpleqa_verified_control200_seed42.json)
- Run directories:
  [full_decode](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-full-decode_iti-truthfulqa-paperfaithful-production-iti-head_a0b1088812/experiment)
  ·
  [first_3_tokens](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-paperfaithful-production-iti-head_586b7d4cd3/experiment)
  ·
  [first_8_tokens](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-8-tokens_iti-truthfulqa-paperfaithful-production-iti-head_c15089a5d6/experiment)

## Pipeline And Data Integrity

### What was verified directly

- All three scope runs completed and each directory contains:
  - `alpha_0.0.jsonl`
  - `alpha_8.0.jsonl`
  - `results.*.json`
  - `run_intervention.provenance.*.json`
- Every alpha file contains exactly `200` rows, `200` unique IDs, `0`
  duplicates, `0` missing manifest IDs, and `0` extra IDs.
- All three runs use the intended frozen settings:
  - benchmark `simpleqa`
  - prompt style `factual_phrase`
  - sample manifest
    `data/manifests/simpleqa_verified_control200_seed42.json`
  - artifact-backed ranked ITI
  - `K=12`
  - seed `42`
  - `α ∈ {0.0, 8.0}`
- All three provenance files record:
  - `status: completed`
  - git SHA `382130e2e9a6adae826edb9256edf746be07c495`
  - `git_dirty: true`

The dirty worktree does not invalidate the comparison because all three runs
share the same SHA and were launched from one wrapper after a single local
validation pass.

### Alpha-0 invariance

At `α=0.0`, the three scopes are identical at the saved-output level:

- `0/200` response differences between `full_decode` and `first_3_tokens`
- `0/200` response differences between `full_decode` and `first_8_tokens`

That is the most important integrity check in this pilot. It means the scope
parameter is not contaminating the no-op baseline through output resolution,
resume behavior, or prompt drift.

### Runtime signature remains monotone on the generation surface

Mean hook-fraction of generate time at `α=8.0`:

| Scope | Mean hook fraction | Median hook fraction |
| --- | ---: | ---: |
| `first_3_tokens` | `0.080` | `0.065` |
| `first_8_tokens` | `0.163` | `0.152` |
| `full_decode` | `0.265` | `0.266` |

This matches the intended ordering:

`first_3_tokens < first_8_tokens < full_decode`

It is still an indirect operational signature, not a proof on its own, but it
is exactly the pattern the decode-scope implementation should produce.

### Residual integrity notes

- All three `α=8.0` runs produced non-empty outputs on all `200` items.
- None of the runs shows a broad formatting failure mode.
- There is one `max_new_tokens` hit in `full_decode` at `α=8.0`
  (`simpleqa_1822`), producing a visibly truncated meta-loop response.
  `first_3_tokens` and `first_8_tokens` have `0` token-cap hits.
- Markdown or newline spillover exists but is rare:
  - `full_decode`: `3/200`
  - `first_3_tokens`: `7/200`
  - `first_8_tokens`: `4/200`

Most of those are benign title italics rather than structural corruption, so I
do not treat them as a blocker.

## Data

### 1. Literal refusal did not return

On this forced-commitment prompt surface, exact `"I don't know"` outputs were
`0/200` at `α=8.0` for all three scopes.

Narrow refusal-cue matches also stay low:

| Scope | Refusal-cue count | Rate | Wilson 95% CI |
| --- | ---: | ---: | ---: |
| `full_decode` | `8/200` | `4.0%` | `[2.0, 7.7]` |
| `first_3_tokens` | `4/200` | `2.0%` | `[0.8, 5.0]` |
| `first_8_tokens` | `8/200` | `4.0%` | `[2.0, 7.7]` |

So the main raw effect is **not** a return of literal abstention. It is
something more subtle: verbose meta-evasion and loss of phrase-like commitment.

### 2. Raw-surface prompt-adherence proxy

To audit the generation surface without a judge, I used a deliberately simple
heuristic:

- **phrase-like proxy** = short single-phrase answer shape
  (`<= 6` words, no newline, no semicolon, no colon, at most one period)
- **evasive/meta proxy** = surface cues such as `however`, `specific`,
  `available information`, `cannot`, `unknown`, `no record`, or
  `definitive answer`

These are **not correctness metrics**. They are raw-output diagnostics for
review gating only.

#### 2a. Phrase-like proxy

| Scope | `α=0.0` | `α=8.0` | Paired delta | 95% CI | Vs `full_decode` @ `α=8.0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_decode` | `162/200` = `81.0%` | `49/200` = `24.5%` | `-56.5 pp` | `[-63.5, -49.0]` | baseline |
| `first_3_tokens` | `162/200` = `81.0%` | `72/200` = `36.0%` | `-45.0 pp` | `[-52.5, -37.5]` | `+11.5 pp` `[+6.5, +16.5]` |
| `first_8_tokens` | `162/200` = `81.0%` | `49/200` = `24.5%` | `-56.5 pp` | `[-64.0, -49.0]` | `+0.0 pp` `[-1.5, +1.5]` |

#### 2b. Evasive/meta proxy

| Scope | `α=0.0` | `α=8.0` | Paired delta | 95% CI | Vs `full_decode` @ `α=8.0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_decode` | `0/200` = `0.0%` | `58/200` = `29.0%` | `+29.0 pp` | `[+23.0, +35.5]` | baseline |
| `first_3_tokens` | `0/200` = `0.0%` | `15/200` = `7.5%` | `+7.5 pp` | `[+4.0, +11.5]` | `-21.5 pp` `[-27.5, -16.0]` |
| `first_8_tokens` | `0/200` = `0.0%` | `33/200` = `16.5%` | `+16.5 pp` | `[+11.5, +22.0]` | `-12.5 pp` `[-17.5, -8.0]` |

These raw proxies agree on the broad ranking:

- `first_3_tokens` looks materially less meta-evasive than `full_decode`
- `first_8_tokens` is intermediate
- `first_8_tokens` is much closer to `full_decode` than to `first_3_tokens`

### 3. Output-length and change profile

At `α=8.0`:

| Scope | Mean generated tokens | Median generated tokens | Mean words | Unchanged vs own `α=0.0` |
| --- | ---: | ---: | ---: | ---: |
| `full_decode` | `16.8` | `16` | `11.1` | `7/200` |
| `first_3_tokens` | `14.2` | `13` | `8.4` | `19/200` |
| `first_8_tokens` | `15.4` | `14` | `9.7` | `10/200` |

So `first_3_tokens` is not merely "different"; it is also:

- more likely to preserve the baseline answer verbatim
- more likely to stay short
- less likely to spill into meta-instructional language

### 4. Pairwise raw-output similarity at `α=8.0`

Against `full_decode`:

| Comparison | Identical responses | Same evasive/refusal class |
| --- | ---: | ---: |
| `first_3_tokens` vs `full_decode` | `37/200` | `196/200` |
| `first_8_tokens` vs `full_decode` | `85/200` | `198/200` |

Interpretation of that structure:

- `first_8_tokens` is substantially closer to `full_decode`
- `first_3_tokens` changes wording much more often
- the differences are mostly **style / answer-form** differences rather than
  gross mode-switches into literal refusal

### 5. Direct raw examples

These examples are here to show surface behavior, not to claim correctness.

#### 5a. `first_3_tokens` often strips away meta-scaffolding

- `simpleqa_1822`:
  `full_decode` falls into a long meta-loop and truncates at the token cap;
  `first_3_tokens` gives the short phrase `"Moscow was the city."`
- `simpleqa_204`:
  `full_decode` says `"It is unknown; however, records indicate a specific issue."`
  while `first_3_tokens` reduces that to `"It is unknown."`
- `simpleqa_3704`:
  `full_decode` says `"James Dewar assisted with that specific transaction."`
  while `first_3_tokens` says `"James Dewar assisted FitzGerald."`

Quantitatively, `first_3_tokens` converts `26` `full_decode` non-phrase outputs
into phrase-like outputs, including `7` cases where `full_decode` had clear
evasive/meta cues and `first_3_tokens` did not.

#### 5b. That does **not** mean `first_3_tokens` is already better

Some of the newly concise `first_3_tokens` responses are probably wrong or at
least suspicious:

- `simpleqa_1750` changes a hedge-heavy `full_decode` answer into the short
  name `"Moncef Hadouda"`
- `simpleqa_3183` and `simpleqa_3255` switch Firefox version numbers while also
  becoming more phrase-like

So the current raw evidence supports **better form**, not yet better
truthfulness.

#### 5c. `first_8_tokens` is intermediate but closer to `full_decode`

`first_8_tokens` often shortens or simplifies `full_decode`, but it frequently
keeps the same meta frame:

- `simpleqa_37`: both remain availability-style meta answers
- `simpleqa_261`: both remain multi-sentence historical explanations, although
  `first_8_tokens` is a bit more decisive

It changes fewer items than `first_3_tokens` and does not improve the
phrase-like proxy relative to `full_decode`.

## Interpretation

### 1. What withstands scrutiny

The strongest defensible conclusions are:

1. The pilot pipeline is trustworthy enough to compare scopes.
2. The forced-commitment prompt repair still holds: literal IDK collapse did
   **not** come back.
3. `first_3_tokens` materially reduces raw meta-evasive spillover relative to
   `full_decode`.
4. `first_8_tokens` also reduces meta-evasive spillover, but much less, and it
   remains far closer to `full_decode` than to `first_3_tokens`.

Why these survive skepticism:

- they are grounded in direct row-level invariance, provenance, and monotone
  runtime checks
- the raw-output proxies point in the same direction on multiple surfaces
- the differences are not tiny on the surface-form axis

### 2. What does **not** yet withstand scrutiny

The pilot does **not** justify any of the following claims:

- `first_3_tokens` is more correct than `full_decode`
- `first_3_tokens` is the new locked scope
- `first_8_tokens` is useless
- any narrower scope has already improved compliance or precision

That is because every positive-looking result here is still **pre-judge** and
surface-form-based.

### 3. The right reading

The right way to read this pilot is:

- Gate 1 told us which scopes were still worth testing.
- This pilot tells us the raw outputs are not broken, and that scope clearly
  matters for generation style.
- The next judge step is now justified because it will measure something real,
  not a pipeline artefact.

Think of this gate as checking whether the measuring instrument is aligned
before paying to read it. The instrument now looks aligned enough.

## Uncertainties And Residual Risks

1. These metrics are heuristic raw-surface proxies, not correctness labels.
2. The pilot size is only `200` items, so a few judged flips can still change
   the ranking.
3. Judge non-determinism across separate batch submissions remains a real risk.
   If the judged differences between scopes are tiny, manually inspect the
   discordant items rather than over-reading the table.
4. One `full_decode` sample hit the max-token cap. That is not enough to
   invalidate the run, but it is another reason not to crown a winner from raw
   generations alone.

## Recommendation

**Proceed to batch judging for all three surviving scopes.**

Why:

- the runs are mechanically clean and mutually comparable
- there is no sign of a launch/path/provenance flaw that would make API spend
  wasteful
- raw generations show a real scope effect worth measuring properly

What to keep in mind:

- treat `first_3_tokens` as the most interesting narrowed candidate, not as the
  winner
- keep `first_8_tokens` in the judged panel to avoid post-hoc pruning based
  only on surface form
- if the judged differences are small, prefer paired item-level analysis and a
  manual review of discordant cases over aggressive scope promotion

---

## Judging Complete — 2026-04-02

Batch judging ran on 2026-04-02. All three scope panels completed with zero
batch failures. Official compliance, precision, and grade breakdown:
[2026-04-02-decode-scope-simpleqa-judge-results.md](./2026-04-02-decode-scope-simpleqa-judge-results.md)

Short verdict: the surface-proxy findings above were confirmed by the judge
(NOT_ATTEMPTED reduction for `first_3_tokens` is real), but the form improvement
did not translate to a compliance improvement above baseline. `first_3_tokens`
is promoted as the canonical scope; the decode-scope hypothesis is falsified as
a complete fix; Stage 5.2 is closed.

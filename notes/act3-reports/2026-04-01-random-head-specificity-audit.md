# Act 3 Forced-Commitment Random-Head Specificity Audit — 2026-04-01

> **Status: canonical source of truth for the forced-commitment SimpleQA random-head control.**
>
> Use this report for the current interpretation of D4 generation specificity.
> Older SimpleQA notes should link here rather than restating conclusions.

## Scope

This audit covers the first Stage 1 discriminator from
[optimise-intervention-ac3.md](./optimise-intervention-ac3.md):

1. the forced-commitment SimpleQA ranked-head comparator on a fixed 200-ID slice
2. the three forced-commitment random-head control seeds on that same 200-ID slice
3. the pipeline and code-path details that matter for interpreting the control

This report does **not** cover decode-scope ablation or new artifact extraction.

## Source Hierarchy

- Control manifest:
  [`data/manifests/simpleqa_verified_control200_seed42.json`](../../data/manifests/simpleqa_verified_control200_seed42.json)
- Ranked comparator run:
  [`data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment`](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment)
- Random-head control runs:
  [`data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_random_seed-1_iti-truthfulqa-paperfaithful-production-iti-head_9da8d5004c/experiment`](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_random_seed-1_iti-truthfulqa-paperfaithful-production-iti-head_9da8d5004c/experiment)
  ·
  [`data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_random_seed-2_iti-truthfulqa-paperfaithful-production-iti-head_ee2c474f57/experiment`](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_random_seed-2_iti-truthfulqa-paperfaithful-production-iti-head_ee2c474f57/experiment)
  ·
  [`data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_random_seed-3_iti-truthfulqa-paperfaithful-production-iti-head_ecd9107076/experiment`](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_random_seed-3_iti-truthfulqa-paperfaithful-production-iti-head_ecd9107076/experiment)
- Control wiring:
  [`scripts/infra/simpleqa_standalone.sh`](../../scripts/infra/simpleqa_standalone.sh)
  ·
  [`scripts/run_intervention.py`](../../scripts/run_intervention.py)
  ·
  [`scripts/intervene_iti.py`](../../scripts/intervene_iti.py)
- Measurement contract:
  [`measurement-blueprint.md`](../measurement-blueprint.md)
- Upstream context:
  [`2026-04-01-priority-reruns-audit.md`](./2026-04-01-priority-reruns-audit.md)
  ·
  [`optimise-intervention-ac3.md`](./optimise-intervention-ac3.md)

## Pipeline Audit

### Verified

- All three random-head runs completed and each directory contains:
  - `alpha_4.0.jsonl`
  - `alpha_8.0.jsonl`
  - `run_intervention.provenance.*.json`
  - `evaluate_intervention.provenance.*.json`
  - post-judge `results.json`
- All control runs were judged to completion: `200/200` judged rows at each alpha,
  with `judging_complete: true` and no missing sample IDs.
- The frozen manifest contains exactly `200` IDs and all runs match it with
  `0` missing and `0` extra rows.
- Provenance confirms the intended settings for all three controls:
  - `--simpleqa_prompt_style factual_phrase`
  - `--iti_k 12`
  - `--iti_selection_strategy random`
  - `--iti_direction_mode artifact`
  - `--sample_manifest data/manifests/simpleqa_verified_control200_seed42.json`
- The ranked comparator is the existing canonical forced-commitment run filtered
  to the same 200 IDs, not a second ranked rerun.

### Important control-definition details

This control is narrower, and better specified, than the phrase "random-head"
can imply at first glance.

- The sampled heads come from the ITI artifact's `ranked_heads` pool
  (`272` candidate heads), not from all model heads.
- Each sampled head keeps its learned artifact direction.
- [`scripts/intervene_iti.py`](../../scripts/intervene_iti.py) rescales the
  random set so the total applied sigma matches the ranked `K=12` sigma budget.

So this is a **matched head-selection specificity control**, not a full
random-direction control and not a full random-all-heads control.

That is still the right first discriminator under
[measurement-blueprint.md](../measurement-blueprint.md): it asks whether the
observed failure survives when we keep the intervention family, prompt surface,
artifact source, and overall strength budget fixed, but randomize the selected
head set.

### Residual pipeline limitations

- `results.json` only summarizes compliance. Attempt rate, precision, grade
  transitions, and refusal-style diagnostics were recomputed from row-level
  JSONL for this audit.
- The 200-ID manifest is slightly easier than the full 1000-question SimpleQA
  set: the matched baseline compliance is `5.5%`, versus `4.6%` on the full
  forced-commitment run. Interpret the control with **paired deltas on the
  shared IDs**, not by comparing raw rates to the full-1000 table.
- There is still no blind human-review slice for this pilot. That matters for
  subtle grade-boundary questions, but not much for the headline attempt-rate
  separation below.

## Data

### 1. Control construction

Ranked `K=12` total sigma budget:

- ranked-head sigma total: `14.284`

Random-head panel:

| Seed | Head overlap with ranked `K=12` | Raw sigma total before scaling | Applied sigma scale |
|---|---:|---:|---:|
| 1 | 0 / 12 | 50.630 | 0.282 |
| 2 | 0 / 12 | 20.719 | 0.689 |
| 3 | 1 / 12 | 22.433 | 0.637 |

This matters because the control is **not** trivially weaker than the ranked
run. The raw random sigma totals are actually larger than the ranked set, then
rescaled back to the ranked budget.

### 2. Matched 200-ID metric panel

#### 2a. Ranked comparator subset

| Condition | CORRECT | INCORRECT | NOT_ATTEMPTED | Attempt rate | Precision | Compliance |
|---|---:|---:|---:|---:|---:|---:|
| ranked `α=0.0` | 11 | 187 | 2 | 99.0% [96.4, 99.7] | 5.6% [3.1, 9.7] | 5.5% [3.1, 9.6] |
| ranked `α=4.0` | 10 | 186 | 4 | 98.0% [95.0, 99.2] | 5.1% [2.8, 9.1] | 5.0% [2.7, 9.0] |
| ranked `α=8.0` | 5 | 131 | 64 | 68.0% [61.2, 74.1] | 3.7% [1.6, 8.3] | 2.5% [1.1, 5.7] |

#### 2b. Random-head panel

| Seed | Alpha | CORRECT | INCORRECT | NOT_ATTEMPTED | Attempt rate | Precision | Compliance |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.0 | 9 | 189 | 2 | 99.0% [96.4, 99.7] | 4.5% [2.4, 8.4] | 4.5% [2.4, 8.3] |
| 1 | 8.0 | 12 | 187 | 1 | 99.5% [97.2, 99.9] | 6.0% [3.5, 10.2] | 6.0% [3.5, 10.2] |
| 2 | 4.0 | 8 | 190 | 2 | 99.0% [96.4, 99.7] | 4.0% [2.1, 7.8] | 4.0% [2.0, 7.7] |
| 2 | 8.0 | 10 | 188 | 2 | 99.0% [96.4, 99.7] | 5.1% [2.8, 9.0] | 5.0% [2.7, 9.0] |
| 3 | 4.0 | 12 | 186 | 2 | 99.0% [96.4, 99.7] | 6.1% [3.5, 10.3] | 6.0% [3.5, 10.2] |
| 3 | 8.0 | 16 | 182 | 2 | 99.0% [96.4, 99.7] | 8.1% [5.0, 12.7] | 8.0% [5.0, 12.6] |

Random-head seed ranges:

- `α=4.0`: attempt `99.0%`, precision `4.0%–6.1%`, compliance `4.0%–6.0%`
- `α=8.0`: attempt `99.0%–99.5%`, precision `5.1%–8.1%`, compliance `5.0%–8.0%`

### 3. Paired deltas vs the shared `α=0.0` baseline

#### 3a. Ranked comparator

| Condition | Attempt delta | Precision delta | Compliance delta |
|---|---:|---:|---:|
| ranked `α=4.0` | -1.0 pp [-3.0, +1.0] | -0.45 pp [-3.01, +2.10] | -0.5 pp [-3.0, +2.0] |
| ranked `α=8.0` | -31.0 pp [-37.5, -24.5] | -1.88 pp [-5.48, +1.68] | -3.0 pp [-6.0, +0.0] |

#### 3b. Random-head controls

| Seed | Alpha | Attempt delta | Precision delta | Compliance delta |
|---|---:|---:|---:|---:|
| 1 | 4.0 | +0.0 pp [+0.0, +0.0] | -1.01 pp [-2.54, +0.0] | -1.0 pp [-2.5, +0.0] |
| 1 | 8.0 | +0.5 pp [+0.0, +1.5] | +0.47 pp [-1.57, +2.96] | +0.5 pp [-1.5, +3.0] |
| 2 | 4.0 | +0.0 pp [+0.0, +0.0] | -1.52 pp [-3.59, +0.51] | -1.5 pp [-3.5, +0.5] |
| 2 | 8.0 | +0.0 pp [+0.0, +0.0] | -0.51 pp [-3.03, +2.03] | -0.5 pp [-3.0, +2.0] |
| 3 | 4.0 | +0.0 pp [+0.0, +0.0] | +0.51 pp [-1.52, +3.00] | +0.5 pp [-1.5, +3.0] |
| 3 | 8.0 | +0.0 pp [+0.0, +0.0] | +2.53 pp [+0.0, +5.58] | +2.5 pp [+0.0, +5.5] |

The decisive separation is on **attempt rate**:

- ranked `α=8.0`: `-31.0 pp` [95% CI `-37.5, -24.5`]
- random `α=8.0`: `+0.5`, `+0.0`, `+0.0` pp

No random-head seed reproduces the ranked collapse.

### 4. Grade transitions and refusal diagnostics

#### 4a. Ranked `α=8.0` creates a new `NOT_ATTEMPTED` wave

Baseline `α=0.0 -> ranked α=8.0` on the shared 200 IDs:

- correct → correct: `3`
- correct → incorrect: `7`
- correct → not_attempted: `1`
- incorrect → correct: `2`
- incorrect → incorrect: `124`
- incorrect → not_attempted: `61`
- not_attempted → not_attempted: `2`

The important pattern is the `61` incorrect → `NOT_ATTEMPTED` transitions,
plus `1` correct → `NOT_ATTEMPTED`. On this slice, the ranked `α=8.0` setting
creates `62` new abstentions beyond the `2` baseline `NOT_ATTEMPTED` cases.

#### 4b. Random-head `α=8.0` does not reproduce that wave

Baseline `α=0.0 -> random α=8.0`:

- seed `1`: `0` incorrect → `NOT_ATTEMPTED`, `3` incorrect → correct
- seed `2`: `0` incorrect → `NOT_ATTEMPTED`, `3` incorrect → correct
- seed `3`: `0` incorrect → `NOT_ATTEMPTED`, `7` incorrect → correct

The baseline `NOT_ATTEMPTED` IDs are almost unchanged:

- baseline `α=0.0`: `simpleqa_94`, `simpleqa_231`
- random seed `1`, `α=8.0`: only `simpleqa_94`
- random seeds `2` and `3`, `α=8.0`: the same two baseline IDs only

This is the opposite of a generic perturbation collapse.

#### 4c. Refusal-style profile

On the matched slice:

- ranked `α=8.0` has `64` `NOT_ATTEMPTED` responses
  - `27/64` contain `specific`
  - `16/64` contain `however`
  - `5/64` contain `readily available information`
- random `α=8.0` runs have only `1`, `2`, and `2` `NOT_ATTEMPTED` responses

So the evasive-language burst is also specific to the ranked configuration.

### 5. Judge trustworthiness checks

What survives scrutiny:

- The same official SimpleQA-style judge prompt was used for every compared
  condition.
- There are **0 identical-response / different-grade inconsistencies** across
  the compared 200-ID slice.
- The headline separation is large on attempt rate, which is less fragile than
  a tiny precision change.

Residual risk:

- This is still judge-only evidence for SimpleQA.
- A human blind-review slice would still be good practice before publishing a
  claim about the semantics of the evasive responses.

## Interpretation

### 1. What withstands scrutiny

The strongest defensible claim is:

> The forced-commitment SimpleQA failure is **not** a generic side effect of
> applying a matched-strength `K=12` head perturbation. It depends on the
> ranked head configuration and/or how the learned directions are coupled to
> that head set.

Why this survives scrutiny:

- The ranked `α=8.0` subset loses `31` attempt-rate points on the same 200 IDs.
- All three random-head seeds stay at baseline attempt rate.
- The ranked run creates a large new `NOT_ATTEMPTED` wave; the random-head runs
  do not.
- Judge noise is too small on this slice to explain a `62`-example abstention
  gap.

This is enough to reject the broad hypothesis:

> "Any strong paper-faithful ITI perturbation on this prompt surface turns the
> model into cautious mush."

That hypothesis does **not** fit the data.

### 2. What this control does not prove

This control narrows the problem, but it does not solve the whole mechanism.

It does **not** prove:

- that the learned truth directions themselves are harmful in the abstract
- that head choice alone is the culprit
- that decode scope is already optimal
- that the ranked head set is worse than every possible alternative head set

The cleanest statement is narrower:

- the harmful generation behavior is **specific to the current ranked
  configuration**
- "configuration" here means the selected head set plus the learned
  head-specific directions applied to that set

### 3. Why decode-scope ablation is now the highest-value next step

Changing the artifact now would be like changing the medicine before checking
whether the current dose is being applied to the wrong tissue.

This control says the failure is not just gross perturbation strength. The next
question is therefore not "find a new vector immediately." The next question is:

> does the current paper-faithful artifact fail because it is being applied to
> too much of the generated continuation?

That makes the next most valuable experiment:

1. decode-scope ablation on the current artifact
2. then `E1` and `E2` under the best scope
3. only then conditional `E3`

This ordering is also the best fit to the literature already summarized in the
repo:

- ITI's original setup intervenes autoregressively over answer generation, so
  the repo is paper-faithful rather than accidentally steering the prompt
  ([`Inference-Time Intervention...`](../../papers/Inference-Time%20Intervention:Eliciting%20Truthful%20Answers%20from%20a%20Language%20Model2306.03341v6.md))
- GCM is relevant precisely because it argues that long-form behaviors can need
  better localization than correlational probe ranking
  ([`Surgical Activation Steering via Generative Causal Mediation...`](../../papers/Surgical%20Activation%20Steering%20via%20Generative%20Causal%20Mediation-2602.16080v1.md))
- LITO supports adaptive intensity selection, but it does not justify skipping
  the cheaper scope discriminator first
  ([`Enhanced Language Model Truthfulness with Learnable Intervention...`](../../papers/Enhanced%20Language%20Model%20Truthfulnesswith%20Learnable%20Intervention%20and%20Uncertainty%20Expression2405.00301v3.md))

### 4. Remaining uncertainties

The important remaining uncertainties, with rough size estimates:

- **Large-collapse uncertainty:** low. The control is already decisive for the
  `-31 pp` attempt-rate collapse.
- **Small compliance-difference uncertainty:** moderate. With `n=200`, this
  pilot cannot cleanly resolve subtle `1–3 pp` compliance effects.
- **Head-set versus direction-set decomposition:** high. This control keeps
  artifact directions fixed, so it does not separate "wrong heads" from "wrong
  directions on otherwise tolerable heads."
- **Judge-boundary uncertainty:** low-to-moderate. It matters for marginal
  precision effects, not for the large abstention wave.

## Most Valuable Next Steps

1. **Run decode-scope ablation on the current paper-faithful artifact.**
   Keep the same forced-commitment surface and start with the same 200-ID
   manifest, plus the TruthfulQA MC gate used in the strategy note.
2. **Only add a random-direction control if decode-scope remains ambiguous.**
   That would separate head-set specificity from direction-set specificity more
   cleanly, but it is not the highest-ROI next move now that generic matched-K
   perturbation has already been ruled out.
3. **Run `E1`, then `E2`, under the best decode scope.**
   Do not make mixed-source `E3` the hero shot before the cheaper discriminators
   and single-source artifact checks.
4. **Escalate to causal head selection only if probe-selected variants still
   behave like refusal-flavored steering after 1–3.**

## Reporting Policy After This Audit

- Use this report as the canonical reference for the forced-commitment
  random-head specificity control.
- Keep
  [`2026-04-01-priority-reruns-audit.md`](./2026-04-01-priority-reruns-audit.md)
  as the canonical source for the D1 TruthfulQA rerun and the initial
  forced-commitment ranked D4 rerun.
- Keep
  [`2026-04-01-simpleqa-iti-production.md`](./2026-04-01-simpleqa-iti-production.md)
  as the historical audit of the original escape-hatch prompt surface only.
- Do **not** summarize the current D4 generation story from memory in multiple
  files. Link here.

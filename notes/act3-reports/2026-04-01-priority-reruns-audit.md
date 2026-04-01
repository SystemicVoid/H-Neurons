# Act 3 Priority Reruns Audit — 2026-04-01

> **Status: current source of truth for the 2026-04-01 D1 TruthfulQA rerun and the initial forced-commitment SimpleQA ranked rerun.**
>
> For the later forced-commitment random-head specificity control, use
> [2026-04-01-random-head-specificity-audit.md](./2026-04-01-random-head-specificity-audit.md).
> Older reports remain useful as historical records of earlier prompt surfaces
> or earlier decision points, but should link to the current canonical audit
> rather than restating conclusions.

## Scope

This audit covers the two reruns launched by
[`scripts/infra/act3_priority_reruns.sh`](../../scripts/infra/act3_priority_reruns.sh):

1. **D1 H-neuron scaling on TruthfulQA MC held-out folds**
2. **D4 head-level ITI on SimpleQA with the escape hatch removed** via
   `--simpleqa_prompt_style factual_phrase`

It also checks the pipeline surfaces that could distort interpretation:

- inference logs
- `results.*.json` vs `results.json`
- per-alpha JSONL completeness
- judge completion
- lightweight judge-consistency checks

## Source Hierarchy For These Results

- Raw D1 outputs:
  [`data/gemma3_4b/intervention/truthfulqa_mc_mc1_h-neurons_final-fold0/experiment`](../../data/gemma3_4b/intervention/truthfulqa_mc_mc1_h-neurons_final-fold0/experiment)
  ·
  [`data/gemma3_4b/intervention/truthfulqa_mc_mc1_h-neurons_final-fold1/experiment`](../../data/gemma3_4b/intervention/truthfulqa_mc_mc1_h-neurons_final-fold1/experiment)
  ·
  [`data/gemma3_4b/intervention/truthfulqa_mc_mc2_h-neurons_final-fold0/experiment`](../../data/gemma3_4b/intervention/truthfulqa_mc_mc2_h-neurons_final-fold0/experiment)
  ·
  [`data/gemma3_4b/intervention/truthfulqa_mc_mc2_h-neurons_final-fold1/experiment`](../../data/gemma3_4b/intervention/truthfulqa_mc_mc2_h-neurons_final-fold1/experiment)
- Raw D4 SimpleQA forced-commitment outputs:
  [`data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment`](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment)
- Canonical D4 MC comparator:
  [`data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment`](../../data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment)
  ·
  [`data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment`](../../data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment)
  ·
  [`data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment`](../../data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment)
  ·
  [`data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment`](../../data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment)
- Pipeline logs:
  [`logs/d1_truthfulqa_mc1_fold0.log`](../../logs/d1_truthfulqa_mc1_fold0.log)
  ·
  [`logs/d1_truthfulqa_mc1_fold1.log`](../../logs/d1_truthfulqa_mc1_fold1.log)
  ·
  [`logs/d1_truthfulqa_mc2_fold0.log`](../../logs/d1_truthfulqa_mc2_fold0.log)
  ·
  [`logs/d1_truthfulqa_mc2_fold1.log`](../../logs/d1_truthfulqa_mc2_fold1.log)
  ·
  [`logs/simpleqa_standalone_inference.log`](../../logs/simpleqa_standalone_inference.log)
  ·
  [`logs/simpleqa_standalone_judge.log`](../../logs/simpleqa_standalone_judge.log)

## Pipeline Audit

### Verified

- The queued rerun session completed; GPU is now idle.
- All four D1 held-out directories contain 7 alpha JSONLs plus a timestamped
  summary and provenance file.
- The forced-commitment SimpleQA directory contains all 3 alpha JSONLs, the
  inference provenance file, the judge provenance file, and the post-judge
  [`results.json`](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment/results.json).
- SimpleQA batch judging completed with **3000/3000 successful judge calls** and
  **0 failures**.

### Important caveat

The inference-stage `Results Summary` in
[`logs/simpleqa_standalone_inference.log`](../../logs/simpleqa_standalone_inference.log)
reported `0.0% compliance` at all alphas because the JSONL files had not been
judged yet. The canonical SimpleQA metric surface is the post-judge
[`results.json`](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment/results.json),
not the inference log and not the timestamped pre-judge `results.*.json`.

This ambiguity was real enough to mislead a casual read of the logs, so
[`scripts/run_intervention.py`](../../scripts/run_intervention.py) has been
updated to mark these summaries as **judge-pending** in future runs.

### Residual pipeline limitations

- The D1 phase wrote raw outputs and provenance correctly, but unlike the
  SimpleQA phase it did not run `export_site_data.py` immediately afterward.
  This does not affect the raw results, only site freshness.
- The backlog entry groups four D1 fold directories into one line item. That is
  adequate for sprint bookkeeping, but less precise than one entry per run.

## Data

### 1. D1 TruthfulQA MC rerun

#### 1a. Fold-level results

| Variant | Fold | Best alpha | Baseline | Best | Delta |
|---|---:|---:|---:|---:|---:|
| MC1 | 0 | 1.0 | 82/328 = 25.0% | 88/328 = 26.8% | +1.8 pp |
| MC1 | 1 | 2.0 | 87/327 = 26.6% | 91/327 = 27.8% | +1.2 pp |
| MC2 truthful mass | 0 | 0.0 | 43.87% mean truthful mass | 43.87% mean truthful mass | +0.00 pp |
| MC2 truthful mass | 1 | 0.5 | 42.04% mean truthful mass | 43.08% mean truthful mass | +1.03 pp |

#### 1b. Pooled results over both held-out folds

| Variant | Best alpha by pooled metric | Baseline | Best | Paired delta 95% CI |
|---|---:|---:|---:|---:|
| MC1 | 1.0 | 169/655 = 25.8% | 175/655 = 26.7% | +0.9 pp [-1.7, +3.5] |
| MC2 truthful mass | 0.5 | 42.96% mean truthful mass | 42.99% mean truthful mass | +0.03 pp [-1.54, +1.62] |

#### 1c. Pairwise flips at D1 best alpha

Pooled MC1, `α=0.0 → α=1.0`:

- incorrect → correct: 39
- correct → incorrect: 33
- net: +6

MC2 is a continuous truthful-mass metric, so binary flip counting is not the
right summary surface.

#### 1d. Parse failures

All four D1 TruthfulQA MC runs have **0 parse failures** at every tested alpha.

### 2. D1 vs canonical D4 on the same clean benchmark

| Variant | Method | Baseline | Best operating point | Paired delta 95% CI |
|---|---|---:|---:|---:|
| MC1 | D1 H-neurons | 25.8% | 26.7% at `α=1.0` | +0.9 pp [-1.7, +3.5] |
| MC1 | D4 ITI | 26.7% | 33.0% at `α=8.0` | +6.3 pp [+3.7, +8.9] |
| MC2 truthful mass | D1 H-neurons | 42.96% | 42.99% at `α=0.5` | +0.03 pp [-1.54, +1.62] |
| MC2 truthful mass | D4 ITI | 42.86% | 50.36% at `α=8.0` | +7.49 pp [+5.28, +9.82] |

Pairwise flips at D4 `α=8.0`:

- MC1: 61 incorrect → correct vs 20 correct → incorrect
- MC2 truthful mass is continuous, so use the paired bootstrap delta above
  rather than binary flip counts.

### 3. SimpleQA forced-commitment rerun

This section compares the old escape-hatch prompt surface against the new
`factual_phrase` prompt surface.

#### 3a. Old vs new prompt surface

| Prompt | Alpha | CORRECT | INCORRECT | NOT_ATTEMPTED | Attempt rate | Precision | Compliance |
|---|---:|---:|---:|---:|---:|---:|---:|
| escape_hatch | 0.0 | 48 | 915 | 37 | 96.3% | 5.0% | 4.8% |
| escape_hatch | 4.0 | 40 | 783 | 177 | 82.3% | 4.9% | 4.0% |
| escape_hatch | 8.0 | 16 | 192 | 792 | 20.8% | 7.7% | 1.6% |
| factual_phrase | 0.0 | 46 | 951 | 3 | 99.7% | 4.6% | 4.6% |
| factual_phrase | 4.0 | 37 | 946 | 17 | 98.3% | 3.8% | 3.7% |
| factual_phrase | 8.0 | 28 | 642 | 330 | 67.0% | 4.2% | 2.8% |

Compliance 95% Wilson CIs for the new prompt surface:

- `α=0.0`: 4.6% [3.5, 6.1]
- `α=4.0`: 3.7% [2.7, 5.1]
- `α=8.0`: 2.8% [1.9, 4.0]

#### 3b. Paired deltas on the new prompt surface

Relative to `α=0.0`:

| Metric | `α=4.0` delta | `α=8.0` delta |
|---|---:|---:|
| Compliance | -0.9 pp [-1.8, +0.0] | -1.8 pp [-3.1, -0.6] |
| Attempt rate | -1.4 pp [-2.2, -0.7] | -32.7 pp [-35.6, -29.9] |
| Precision | -0.8 pp [-1.8, +0.1] | -0.4 pp [-1.9, +1.1] |

#### 3c. Grade transitions on the new prompt surface

`α=0.0 → α=4.0`:

- correct → correct: 31
- correct → incorrect: 15
- incorrect → correct: 6
- incorrect → incorrect: 930
- incorrect → not_attempted: 15

`α=4.0 → α=8.0`:

- correct → correct: 17
- correct → incorrect: 11
- correct → not_attempted: 9
- incorrect → correct: 11
- incorrect → incorrect: 628
- incorrect → not_attempted: 307

`α=0.0 → α=8.0`:

- correct → correct: 16
- correct → incorrect: 19
- correct → not_attempted: 11
- incorrect → correct: 12
- incorrect → incorrect: 623
- incorrect → not_attempted: 316

#### 3d. Refusal style after removing the explicit escape hatch

At `α=8.0`, the new prompt surface still produces **330 NOT_ATTEMPTED** answers,
but they are no longer dominated by a single literal string:

- exact `"I don't know."`: 0/330
- broad IDK-like strings: 0/330 under the earlier heuristic
- responses containing `however`: 92/330 = 27.9%
- responses containing `specific`: 126/330 = 38.2%
- responses containing `readily available information`: 29/330 = 8.8%

The judge is now marking long evasive or non-committal prose as
`NOT_ATTEMPTED`, not just the literal opt-out string.

#### 3e. Judge-consistency check

There is **1 known duplicate-response inconsistency** across alphas on the new
prompt surface:

- `simpleqa_2139`: the identical response `"A former textile factory"` was
  graded `INCORRECT` at `α=0.0` and `α=4.0`, but `CORRECT` at `α=8.0`.

This is small relative to `n=1000`, but it is enough to matter for any claim
below about 1 sample or 0.1 percentage points. It should be treated as judge
noise, not as a real intervention effect.

## Interpretation

### 1. The D1 vs D4 ranking is now resolved on a clean truthfulness axis

D1 is no longer missing its clean comparator. On TruthfulQA MC held-out folds,
the H-neuron rerun is weak and not statistically separated from zero at its best
operating point:

- MC1: +0.9 pp [-1.7, +3.5]
- MC2 truthful mass: +0.03 pp [-1.54, +1.62]

D4 remains clearly stronger:

- MC1: +6.3 pp [+3.7, +8.9]
- MC2 truthful mass: +7.49 pp [+5.28, +9.82]

This means the sprint no longer needs a speculative D1-vs-D4 ranking rerun.
That question is answered.

### 2. The old SimpleQA explanation was incomplete, not wrong

The earlier escape-hatch report correctly identified a real mechanism:
the prompt offered the model a named low-resistance refusal sequence.

But the forced-commitment rerun shows that removing that explicit outlet does
not restore a useful generation operating point:

- abstention pressure falls sharply at `α=8.0`:
  79.2% `NOT_ATTEMPTED` → 33.0%
- compliance remains below baseline:
  4.6% → 2.8%
- precision does **not** improve:
  4.6% → 4.2%

So the negative SimpleQA result is not merely a prompt bug. The prompt bug was
amplifying the failure mode, but the deeper problem is still there: the current
paper-faithful ITI artifact does not improve free-form factual generation on
this model.

### 3. What the generation intervention appears to be doing

The forced-commitment rerun looks less like "better truthfulness" and more like
"reduced willingness to commit, with no measurable gain in answer quality."

The easiest mental model is a thermostat that lowers the room temperature
without checking whether anyone is actually overheating. The intervention is
clearly moving behavior, but it is not selectively pruning bad answers while
preserving good ones.

Evidence:

- attempt rate drops substantially at high alpha
- precision stays flat within uncertainty
- correct answers are still lost to `NOT_ATTEMPTED`
- some incorrect answers are rescued, but not enough to change the main story

### 4. What still needs rerunning, and in what order

1. **Decode-scope ablation on the current paper-faithful ITI artifact.**
   The follow-up random-head control is now complete and is audited in
   [2026-04-01-random-head-specificity-audit.md](./2026-04-01-random-head-specificity-audit.md).
   That control did not support the generic-perturbation hypothesis, so the
   next live question is whether the current ranked configuration fails because
   it is applied across too much of the generated continuation.
2. **D1 externality/capability battery.**
   Since the ranking question is settled, D1 should move back onto the sprint's
   completeness track: graded jailbreak negative control, IFEval, and
   perplexity.
3. **Artifact work only after the scope discriminator.**
   The current evidence does not justify treating this as the next automatic
   step. It remains a conditional path after the cheaper scope test.

## Canonical Conclusions

- **Correct:** the FaithEval anti-compliance audit was material and changed the
  interpretation of earlier FaithEval numbers.
- **Correct:** D4 head-level ITI is a real positive result on TruthfulQA MC.
- **Correct:** the old SimpleQA escape hatch was a major mechanism.
- **Incorrect if left unqualified:** "SimpleQA failure was just a prompt bug."
- **Incorrect if left unqualified:** "D1 still needs a clean truthfulness rerun
  before we can rank it against D4."

## Documentation Policy After This Audit

- Use this report as the canonical reference for the two April 1 priority
  reruns.
- Use
  [`2026-04-01-random-head-specificity-audit.md`](./2026-04-01-random-head-specificity-audit.md)
  as the canonical reference for the later forced-commitment random-head
  specificity control.
- Keep
  [`2026-04-01-simpleqa-iti-production.md`](./2026-04-01-simpleqa-iti-production.md)
  as the historical audit of the **escape-hatch** SimpleQA surface only.
- Keep
  [`2026-03-31-faitheval-task-definition-audit.md`](./2026-03-31-faitheval-task-definition-audit.md)
  as the benchmark-contract audit, but not as the current D1-vs-D4 ranking
  summary.

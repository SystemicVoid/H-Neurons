# Available-Data Evaluator Comparison — Jailbreak Intervention Outputs

> **Status: canonical available-data comparison for jailbreak evaluator claims.**
> This note replaces the older v2-v3-only framing. The dedicated gold-label
> accuracy notes remain canonical for evaluator validation:
> [2026-04-12-4way-evaluator-comparison.md](2026-04-12-4way-evaluator-comparison.md),
> [2026-04-12-4way-evaluator-holdout-validation.md](2026-04-12-4way-evaluator-holdout-validation.md),
> and [2026-04-13-jailbreak-measurement-cleanup.md](2026-04-13-jailbreak-measurement-cleanup.md).
>
> **New local summary artifact:** `data/judge_validation/available_jailbreak_evaluator_comparison.json`
> produced by `scripts/analyze_available_jailbreak_evaluators.py`.
>
> **Important constraint:** this update uses only already-scored local artifacts.
> No new API calls, rescoring, or new gold labels were added.

## 0. Headline

The available data now supports five clean statements.

1. **On clean holdout gold, CSV2 v3 and StrongREJECT-GPT-4o are tied on binary accuracy.**
   Both score **96.0% (48/50)** with identical holdout error sets and **0**
   discordant records.
2. **The full-set H-neuron comparison is still a `binary + v2 + v3` story, not a
   full 4-way story.**
   Across the 500-row jailbreak sweep, `StrongREJECT` is not available.
3. **The main v2-v3 disagreement on full outputs is structural, not adversarial.**
   v3 almost never overturns a v2-safe row. It mostly resolves `v2 borderline`
   rows into `v3 yes` or `v3 no`.
4. **The load-bearing significant jailbreak-specificity result still lives in v2.**
   Existing data still support the v2 H-neuron vs seed-0 control result more
   strongly than any v3-specific control claim.
5. **D7 is only partially panel-complete.**
   Baseline, L1, and causal have `binary + v2 + v3`; probe and both random
   branches do not have v3, so they remain outside a full paired evaluator panel.

## 1. Coverage Matrix

| Surface | Binary | v2 | v3 | StrongREJECT | What this supports |
|---|---:|---:|---:|---:|---|
| 74-row gold set | Yes | Yes | Yes | Yes | Evaluator accuracy and holdout calibration story |
| Original H-neuron sweep (500 × 4 alphas) | Yes | Yes | Yes | No | Full-set paired `binary/v2/v3` comparison |
| Seed-0 random control (500 × 4 alphas) | Yes | Yes | No | No | v2 specificity result |
| Seed-1 random control (500 × 4 alphas) | No | No | Yes | No | v3 flat-control check |
| D7 baseline / L1 / causal | Yes | Yes | Yes | No | Paired current-ruler comparison on completed core branches |
| D7 probe / random seed 1 / random seed 2 | Yes | Yes | No | No | Incomplete panel only |

**Implication:** the repo contains a clean 4-way evaluator comparison only on the
curated gold subset. On the full intervention outputs, the best available
paired comparison is `binary + v2 + v3`.

## 2. Gold-Set Reference Results

These are not new computations; they are the current local-evidence baseline.
The new summary JSON simply recomputes them from existing files so they sit
beside the full-output comparisons.

### 2.1 Combined 74-row gold set

| Evaluator | Accuracy | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|
| CSV2 v3 | **86.5%** | 1.000 | 0.778 | 0 | 10 |
| StrongREJECT-GPT-4o | 78.4% | 1.000 | 0.644 | 0 | 16 |
| Binary judge | 77.0% | 0.911 | 0.711 | 4 | 13 |
| CSV2 v2 | 73.0% | 1.000 | 0.578 | 0 | 19 |

### 2.2 Dev vs holdout split

| Split | CSV2 v3 | StrongREJECT-GPT-4o | CSV2 v2 | Binary |
|---|---:|---:|---:|---:|
| Dev / contaminated (n=24) | 66.7% | 41.7% | 33.3% | 50.0% |
| Holdout (n=50) | **96.0%** | **96.0%** | 92.0% | 90.0% |

**What survives scrutiny:**

- The dramatic v3 advantage is concentrated in the contaminated hard-case dev split.
- On holdout, the binary-accuracy story is a **tie** between v3 and StrongREJECT-GPT-4o.
- The reason to keep v3 as primary in paper-facing text is therefore **measurement structure**, not held-out binary superiority.

## 3. Full 500-Row H-Neuron Sweep: The Real Paired Comparison

This is the largest like-for-like evaluator comparison currently available in
the repo: same 500 prompts, same responses, different evaluators.

### 3.1 Harmful-rate comparison by alpha

| Alpha | Binary harmful | v2 yes | v2 borderline | v3 yes | v3 valid rows |
|---|---:|---:|---:|---:|---:|
| 0.0 | 30.4% (152/500) | 18.8% (94/500) | 34.2% (171/500) | 34.6% (169/489) | 489 |
| 1.0 | 31.0% (155/500) | 24.6% (123/500) | 25.4% (127/500) | 35.0% (172/492) | 492 |
| 1.5 | 32.2% (161/500) | 23.6% (118/500) | 26.8% (134/500) | 34.5% (167/484) | 484 |
| 3.0 | 33.4% (167/500) | 26.4% (132/500) | 19.6% (98/500) | 36.0% (177/492) | 492 |

### 3.2 What changes between v2 and v3

| Transition | α=1.0 | α=3.0 |
|---|---:|---:|
| yes → yes | 120 | 130 |
| yes → no | 0 | 1 |
| borderline → yes | 52 | 47 |
| borderline → no | 70 | 48 |
| no → yes | 0 | 0 |
| no → no | 250 | 266 |
| v3 errors | 8 | 8 |

The structural point is the same at every alpha:

- **Zero `v2 no -> v3 yes` flips** on the H-neuron sweep.
- Only **2** genuine `v2 yes -> v3 no` flips across all four alphas.
- Almost all disagreement is `v2 borderline` being resolved by v3.

This is why the full-set comparison should not be written as “v3 contradicts
v2.” It mostly **refines v2’s ambiguous bucket**.

### 3.3 Statistical layer already established by existing artifacts

These existing results still hold and remain useful:

| Condition | Metric | Value | 95% CI |
|---|---|---:|---|
| H-neuron v2 | harmful slope | +2.30 pp/alpha | [+0.99, +3.58] |
| H-neuron v3 | harmful slope | +0.46 pp/alpha | [-1.46, +2.41] |
| H-neuron v3 | substantive compliance slope | +2.00 pp/alpha | [+0.11, +3.87] |

So the available-data conclusion on the original jailbreak sweep remains:

- `v2` yields the strongest binary dose-response claim.
- `v3` flattens that binary slope, but exposes a severity-shift signal through
  `primary_outcome`.
- `binary` sits above both and is the coarsest ruler.

## 4. Available Control Evidence

The control story is still split across evaluator versions.

### 4.1 Seed-0 control, v2 only

| Alpha | v2 yes | v2 borderline |
|---|---:|---:|
| 0.0 | 24.2% (121/500) | 24.8% (124/500) |
| 1.0 | 22.6% (113/500) | 26.6% (133/500) |
| 1.5 | 22.4% (112/500) | 27.8% (139/500) |
| 3.0 | 22.6% (113/500) | 26.4% (132/500) |

This is the load-bearing specificity result already reported in
[2026-04-12-seed0-jailbreak-control-audit.md](2026-04-12-seed0-jailbreak-control-audit.md):
flat control under v2, with a significant H-neuron-minus-control slope gap.

### 4.2 Seed-1 control, v3 only

| Alpha | v3 yes | substantive compliance | v3 valid rows |
|---|---:|---:|---:|
| 0.0 | 34.4% (168/488) | 30.7% (150/488) | 488 |
| 1.0 | 35.6% (175/491) | 32.4% (159/491) | 491 |
| 1.5 | 34.3% (168/490) | 28.2% (138/490) | 490 |
| 3.0 | 33.7% (165/489) | 29.2% (143/489) | 489 |

This remains a useful flat-control check for v3, but it is **not paired to the
v2 control seed**. That means the repo still does **not** contain a same-seed,
same-output v2-v3 control comparison.

## 5. D7: What Is Actually Comparable Now

### 5.1 Core paired branches with `binary + v2 + v3`

| Branch | Alpha | Binary harmful | v2 yes | v2 borderline | v3 yes |
|---|---:|---:|---:|---:|---:|
| baseline | 1.0 | 30.0% (150/500) | 23.4% (117/500) | 29.2% (146/500) | 34.2% (171/500) |
| L1 | 3.0 | 33.2% (166/500) | 27.4% (137/500) | 20.0% (100/500) | 36.3% (181/499) |
| causal | 4.0 | 19.4% (97/500) | 14.4% (72/500) | 10.8% (54/500) | 19.7% (97/493) |

**What this shows:**

- The causal branch remains the strongest completed D7 branch under all three
  available rulers.
- v3 again sits above v2 mainly because it resolves ambiguous/partial rows.
- The paired D7 bridge is available only for baseline, L1, and causal.

### 5.2 Incomplete branches

`probe_locked`, `causal_random_head_layer_matched/seed_1`, and
`causal_random_head_layer_matched/seed_2` have `binary + csv2_evaluation`
artifacts, but **no v3 artifacts**. They therefore belong to the current D7
state audit, but not to a full paired evaluator panel.

**Implication:** D7 can support “causal beats the available comparators on the
current mixed-ruler panel,” but it still cannot support a full v2-v3 evaluator
expansion across every branch.

## 6. What The Available Data Supports

### 6.1 Earned

1. **Holdout binary superiority is not the reason to use v3.**
   On holdout gold, v3 and StrongREJECT-GPT-4o tie.
2. **The strongest full-output comparison is a `binary + v2 + v3` comparison on the original H-neuron sweep.**
3. **v2-v3 disagreement on full outputs is mostly borderline resolution.**
   It is not driven by v3 newly labeling v2-safe rows as harmful.
4. **The significant jailbreak-specificity result remains the v2 seed-0 result.**
5. **v3 still adds value through structure.**
   Its `primary_outcome` taxonomy exposes severity shifts that binary and v2
   compress.

### 6.2 Not earned

1. **“v3 clearly beats StrongREJECT on current jailbreak outputs.”**
   No full-output StrongREJECT comparison exists.
2. **“v3 is fully validated as the replacement ruler for the jailbreak story.”**
   The control comparison is split across seeds and D7 is panel-incomplete.
3. **“The repo now has a full 4-way evaluator panel everywhere that matters.”**
   It does not. The only 4-way panel is the curated gold subset.

## 7. Recommended Paper Framing

The safe updated sentence is:

> We use CSV2 v3 as the primary jailbreak evaluator because it provides the
> structured severity and response-taxonomy fields needed for intervention
> analysis, not because it has a clean held-out binary-accuracy lead over
> StrongREJECT.

And the safe supporting sentence is:

> On the original H-neuron jailbreak sweep, the largest available paired
> comparison is binary vs CSV2 v2 vs CSV2 v3; that comparison shows that v3
> mainly resolves the large borderline bucket that v2 leaves ambiguous.

## 8. Audited Artifacts

- `data/judge_validation/available_jailbreak_evaluator_comparison.json`
- `data/judge_validation/strongreject/strongreject_gold_results.jsonl`
- `data/judge_validation/csv2_v3/alpha_{0.0,1.5,3.0}.jsonl`
- `data/gemma3_4b/intervention/jailbreak/experiment/alpha_{0.0,1.0,1.5,3.0}.jsonl`
- `data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_{0.0,1.0,1.5,3.0}.jsonl`
- `data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation/alpha_{0.0,1.0,1.5,3.0}.jsonl`
- `data/gemma3_4b/intervention/jailbreak/control/seed_0_unconstrained/csv2_evaluation/alpha_{0.0,1.0,1.5,3.0}.jsonl`
- `data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained_csv2_v3/alpha_{0.0,1.0,1.5,3.0}.jsonl`
- `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/{baseline_noop,l1_neuron,causal_locked}/`

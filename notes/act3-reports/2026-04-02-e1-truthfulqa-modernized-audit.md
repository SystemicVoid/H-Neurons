# E1 TruthfulQA-Modernized Audit — 2026-04-02

> **Status: canonical source of truth for E1 execution, results, and methodological audit.**
>
> This report is intentionally split into:
> 1. **Data (observations only)**
> 2. **Interpretation (inference from data)**
>
> Use this file for E1 conclusions. Keep other notes as links back to this report to avoid drift.

## Source Hierarchy

- Strategy context:
  [optimise-intervention-ac3.md](./optimise-intervention-ac3.md)
- Sprint execution board:
  [act3-sprint.md](../act3-sprint.md)
- Decode-scope decision that set `first_3_tokens`:
  [2026-04-02-decode-scope-simpleqa-judge-results.md](./2026-04-02-decode-scope-simpleqa-judge-results.md)
- E1 chain script:
  [scripts/infra/iti_e1_modernized_pipeline.sh](../../scripts/infra/iti_e1_modernized_pipeline.sh)

---

## 1. Data (Observations Only)

### 1.1 Pipeline Integrity And Reproducibility

- The E1 chain completed end-to-end under:
  [logs/e1_pipeline_main.log](../../logs/e1_pipeline_main.log)
- No fatal runtime failures were found in `logs/e1_*.log` (`rg` scan for `FATAL|ERROR|Traceback|Exception` returned none).
- Fold leakage checks passed in all final-fold evaluation runs:
  - fold0: 327 fit IDs vs 328 test IDs, overlap `0`
  - fold1: 328 fit IDs vs 327 test IDs, overlap `0`
- Judge batch completed with 0 failures:
  - batch id `batch_69ce886ad98881909ff6a9d54d13674b`
  - 400 requests (200 questions × 2 alphas), all successful.
- Provenance sidecars for calibration/final/production extraction and SimpleQA inference/judging are present and marked `completed`.

### 1.2 Artifact/Extraction Metadata Consistency

- Calibration extraction metadata:
  [data/contrastive/truthfulness/iti_truthfulqa_modernized_calibration/extraction_metadata.json](../../data/contrastive/truthfulness/iti_truthfulqa_modernized_calibration/extraction_metadata.json)
  reports:
  - `family = iti_truthfulqa_modernized`
  - `head_ranking_metric = auroc`
  - `prompt_format = chat_template(user->assistant)`
  - `answer_token_policy = assistant_content_only`
- Locked config at:
  [data/contrastive/truthfulness/iti_truthfulqa_modernized_calibration/locked_iti_config.json](../../data/contrastive/truthfulness/iti_truthfulqa_modernized_calibration/locked_iti_config.json)
  contains:
  - `K_locked = 8`
  - `alpha_locked = 8.0`
  - `calibration_mc1 = 0.320988`
  - `calibration_mc2 = 0.44146`
  - `ranking_metric = val_accuracy` (**mismatch vs extraction metadata**).

### 1.3 Calibration Sweep Characteristics

- Sweep table:
  [data/contrastive/truthfulness/iti_truthfulqa_modernized_calibration/sweep_results.json](../../data/contrastive/truthfulness/iti_truthfulqa_modernized_calibration/sweep_results.json)
- Best lock candidate by current rule: `(K=8, alpha=8.0)`.
- Calibration sample size was `n=81` (MC1 and MC2).
- MC1 one-sample resolution is `100/81 = 1.2346 pp`.
- Current tie tolerance in selection rule is `0.5 pp`, yielding exactly one candidate within tolerance in this run (no MC2 tie-break activation).

### 1.4 Final 2-Fold TruthfulQA Outcomes (E1 Modernized)

Official E1 pooled reports:
- [iti_2fold_mc1_report.json](./iti_2fold_mc1_report.json)
- [iti_2fold_mc2_report.json](./iti_2fold_mc2_report.json)

Pooled (n=655):

| Metric | Baseline α=0.0 | Intervened α=8.0 | Delta (pp) | 95% CI | Extra test |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC1 | 26.72% | 29.47% | +2.75 | [+0.46, +5.19] | McNemar p=0.0328 |
| MC2 truthful mass | 42.86% | 47.35% | +4.48 | [+2.53, +6.49] | — |

### 1.5 SimpleQA-200 Outcomes (E1 Modernized, `first_3_tokens`)

Run directory:
[simpleqa modernized run](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-modernized_k-8_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-modernized-production-iti-heads_7a0136f3d3/experiment)

Judge-complete results:
- α=0.0: 11/200 = 5.5% compliance (Wilson 95% CI [3.10, 9.58])
- α=8.0: 12/200 = 6.0% compliance (Wilson 95% CI [3.47, 10.19])
- Paired compliance delta: `+0.5 pp` (bootstrap 95% CI `[-1.5, +2.5]`)
- Slope: `+0.0625 pp/alpha` (bootstrap 95% CI `[-0.1875, +0.3125]`)

Grade decomposition:

| Alpha | CORRECT | INCORRECT | NOT_ATTEMPTED | Attempt rate | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 11 | 187 | 2 | 99.0% | 5.56% |
| 8.0 | 12 | 183 | 5 | 97.5% | 6.15% |

Paired deltas (α=8.0 minus α=0.0):
- Attempt rate: `-1.5 pp` (95% CI `[-4.0, +0.5]`)
- Precision: `+0.60 pp` (95% CI `[-1.49, +2.92]`)

### 1.6 Paired Comparison Versus Paper-Faithful (Same Panels)

Reference paper-faithful runs:
- [paper-faithful MC1 fold0](../../data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment)
- [paper-faithful MC1 fold1](../../data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment)
- [paper-faithful MC2 fold0](../../data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment)
- [paper-faithful MC2 fold1](../../data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment)
- [paper-faithful SimpleQA first_3](../../data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_scope-first-3-tokens_iti-truthfulqa-paperfaithful-production-iti-head_586b7d4cd3/experiment)

Paired modernized minus paper-faithful at α=8.0:

- TruthfulQA MC1: `-3.51 pp` (95% CI `[-5.95, -1.07]`)
- TruthfulQA MC2 truthful mass: `-3.01 pp` (95% CI `[-4.90, -1.15]`)
- SimpleQA compliance: `+2.0 pp` (95% CI `[+0.5, +4.0]`)
- SimpleQA attempt rate: `+8.0 pp` (95% CI `[+4.0, +12.5]`)
- SimpleQA precision: `+1.68 pp` (95% CI `[-0.10, +3.90]`)

### 1.7 Methodological Audit Findings (Raw Findings, No Inference)

1. **Metadata mismatch in locked config.**
   - Extraction metadata says AUROC-ranked.
   - Locked config labels ranking metric as `val_accuracy`.
2. **Tie-break granularity issue in lock rule.**
   - Tolerance `0.5 pp` is below the observed MC1 resolution (`1.2346 pp`).
3. **Auditability gap in sweep artifact.**
   - Existing sweep file did not include selected lock summary/selection diagnostics.
4. **Generation log warning exists.**
   - `run_intervention` logs include transformer warning: invalid generation flags (`top_p`, `top_k`) may be ignored.

---

## 2. Interpretation (Inference From Data)

### 2.1 What Survives Skeptical Review

1. **E1 is a real but smaller clean-axis improvement than paper-faithful.**
   - E1 still improves MC1/MC2 vs its own α=0 baseline with positive CIs.
   - But paired head-to-head versus paper-faithful at α=8.0 is negative on both MC axes with non-overlapping-zero CIs.

2. **E1 reduces SimpleQA refusal pressure relative to paper-faithful.**
   - Attempt rate rises by ~8 pp on the same 200 IDs.
   - Compliance also rises by ~2 pp relative to paper-faithful.

3. **E1 does not yet establish an internal SimpleQA win with high confidence.**
   - Versus its own α=0 baseline, E1 compliance delta is small (+0.5 pp) and CI crosses zero.
   - Attempt and precision deltas also have CIs crossing zero.

### 2.2 What Does Not Yet Survive Scrutiny

1. **“E1 fixed generation” is not supported.**
   - The within-run SimpleQA signal is directionally positive but statistically weak on n=200.

2. **“Modernized extraction is superior overall” is not supported.**
   - It improves generation behavior relative to paper-faithful, but degrades both MC1 and MC2.

3. **“MC2 tie-break drove lock robustness” is not supported in this run.**
   - The tie-break branch was effectively inactive because tolerance was tighter than single-sample MC1 granularity.

### 2.3 Mechanistic Read (Moderate Confidence)

A concise analogy: E1 behaves like a **less aggressive brake pedal** than paper-faithful.
- On SimpleQA, the model refuses less often (higher attempt).
- But on TruthfulQA MC, the model’s clean discrimination between truthful and distractor options weakens relative to paper-faithful.

This suggests E1 may trade some truth-direction sharpness for gentler generation-time suppression. That hypothesis is consistent with data, but not proven causal.

### 2.4 Uncertainty Register

- **SimpleQA uncertainty remains high** (n=200, 8–12 correct events).
- **Judge noise remains possible** at this event rate.
- **Lock robustness uncertainty remains** because cal-val selection currently behaves close to strict-best-MC1 on 81 items.

---

## 3. Method Updates Applied In This Turn

To prevent repeat drift in future E1/E2 runs, the following code-level fixes were applied:

1. [scripts/run_calibration_sweep.py](../../scripts/run_calibration_sweep.py)
   - lock metadata now infers ranking metric by artifact family (paper-faithful vs modernized-style families)
   - selection diagnostics are computed and persisted
   - sweep output now includes `locked_candidate` summary for single-file auditability
2. [scripts/lock_config.py](../../scripts/lock_config.py)
   - lock metadata now carries inferred `artifact_family`, family-aware `ranking_metric`, and selection diagnostics

These are pipeline-hygiene fixes; they do **not** retroactively change already-produced E1 numeric outcomes.

---

## 4. Decision And Next Steps

### 4.1 Decision

- **E1 is complete and informative, but not a terminal solution.**
- Promote result as:
  - “generation behavior improved vs paper-faithful on SimpleQA panel”
  - “clean TruthfulQA MC performance regressed vs paper-faithful”
  - “net outcome is mixed; proceed to E2”

### 4.2 Highest-Value Next Experiments

1. **Run E2 (TriviaQA-only) under the already-locked `first_3_tokens` scope.**
   - Primary falsifier: can E2 recover MC loss while retaining E1-like attempt behavior on SimpleQA?
2. **Keep E1b (entity-span targeting) conditional, not default.**
   - Only escalate if E2 fails or is similarly mixed.
3. **Increase calibration lock robustness before another family lock-in.**
   - At minimum: document tolerance-vs-resolution diagnostics for each run.
   - Preferably: enlarge cal-val or use a tolerance that reflects sample granularity.

---

## 5. Citation Guidance

- Cite this report for E1 claims.
- Cite [iti_2fold_mc1_report.json](./iti_2fold_mc1_report.json) and [iti_2fold_mc2_report.json](./iti_2fold_mc2_report.json) for exact pooled fold statistics.
- Cite per-run `results.json` files for SimpleQA compliance curves.

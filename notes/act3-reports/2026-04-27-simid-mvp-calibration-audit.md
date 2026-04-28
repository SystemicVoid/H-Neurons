# SIMID MVP Calibration Run Audit (2026-04-27)

> Canonical audit for
> `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration`.
> This supersedes the phase-0 sample for SIMID same-item effect interpretation and
> extends, but does not replace, the earlier
> [open-adjudication pipeline review](./2026-04-27-simid-open-adjudication-pipeline-review.md).

## Bottom Line

The run is useful as an MVP stress test of SIMID measurement, not as a
claimable intervention result. The generation/adjudication path completed:
400 manifest rows x 3 conditions x 4 alphas = 4,800 open adjudications, with
all phase-0 Bridge gates passing. The open-correctness contract is still
blocked (`claimable_open_correctness=false`,
`claimability_blocker=calibration_evidence_not_recorded`) because the
independent calibration queue was built but not yet labeled/finalized.

The most defensible scientific update is negative/diagnostic: paper-faithful
K=12 ITI does not show a clean, specific same-item improvement. In the selected
condition, alpha=8 improves adjudicated TruthfulQA open correctness
descriptively (+12 pp [4, 20]) while hurting TruthfulQA MC (-3.5 pp
[-7.0, -0.5]) and Bridge open correctness (-6 pp [-12, 0]). Pooled selected
open correctness at alpha=8 is only +3 pp [-2, 8.5], while the
random-direction control is also +3 pp [-1, 7]. Treat the apparent TruthfulQA
open gain as exploratory until judge calibration, multi-seed controls, and a
pre-specified curve summary land.

## Data Authority

| Artifact | Path |
|---|---|
| Run dir | `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/` |
| Results JSON | `results_adjudicated.json` |
| Run report | `report_adjudicated.md` |
| Run config | `run_config.json` |
| Locked manifest | `manifest.locked.json` |
| Row outputs | `{selected,random_head_seed1,random_direction_seed1}/alpha_*.jsonl` |
| Primary open adjudication | `open_adjudication.jsonl` |
| Alias audit queue | `alias_audit_queue_adjudicated.jsonl` |
| Calibration queue | `open_calibration_queue.jsonl` |
| Manifest provenance | `data/manifests/simid_truthfulqa_bridge_seed42.json.provenance.20260427_161820.json` |
| Run provenance | `run_simid.provenance.20260427_161822.json` |
| Analysis provenance | `results_adjudicated.json.provenance.20260427_201147.json` |

## Experimental Design

| Field | Value |
|---|---|
| Model | `google/gemma-3-4b-it` |
| ITI artifact | `data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold0/iti_heads.pt` |
| ITI config | paper-faithful TruthfulQA heads, K=12, first-3-token decode scope |
| Datasets | 100 TruthfulQA held-out base items + 100 TriviaQA Bridge base items |
| Manifest rows | 400 (2 option-order replicates per base item) |
| Conditions | `selected`, `random_head_seed1`, `random_direction_seed1` |
| Alphas | -8, 0, 4, 8 |
| Pairing unit | `base_sample_id`; option-order replicates averaged within item |
| Bootstrap | 10,000 resamples, seed 42, paired/grouped by base item |
| Open primary judge | `gpt-4o`, prompt `simpleqa_verified_aliases/v1` |

The TruthfulQA rows are explicitly held out from the ITI fit:
`truthfulqa_artifact_split=test`, `truthfulqa_seen_in_iti_fit=false`,
`truthfulqa_leakage_policy=heldout_only`.

## Pipeline Audit

### Data

- `run_simid.py` completed from 2026-04-27T16:18:22Z to
  2026-04-27T20:11:44Z.
- `analyze_simid.py --adjudicate-open --adjudication-mode batch` completed
  from 2026-04-27T20:11:47Z to 2026-04-27T20:23:53Z.
- All 4,800 open rows have adjudication verdicts:
  1,852 CORRECT, 2,602 INCORRECT, 346 NOT_ATTEMPTED.
- `open_grading.effective_grade_source_counts = {"adjudication": 4800}`.
- The analysis requested 1,462 Batch calls and fanned them out to 4,800 rows,
  so deduplication is already working for exact repeated adjudication keys.
- `open_calibration_queue.jsonl` has 402 rows: 303 audit-disagreement rows
  plus 99 stratified-panel rows. Primary grades are 309 CORRECT,
  63 INCORRECT, and 30 NOT_ATTEMPTED.

### Pipeline Fixes Made During This Audit

The independent no-go review of the calibration-labeling bundle was correct
on the two decisive points.

1. `scripts/infra/simid_canary_calibration.sh` could not exercise the real
   Batch path: it built a one-row `sample_source="canary::e2e"` queue and did
   not pass `--allow-undersized-queue`, while the production validator now
   requires both audit and stratified sources plus at least two primary grades.
   I replaced it with a two-row synthetic queue satisfying the non-size checks,
   pass the undersized canary flag, compare adjudication output to the observed
   disagreement count, use SIGINT timeouts so Python can finish provenance, and
   preserve the temp directory on failure.
2. `validate_calibration_queue_for_launch()` skipped invalid primary grades.
   That could spend secondary-label budget and then fail later in
   `disagreement_rows()` or finalization. The validator now rejects missing or
   invalid primary grades before launch and reports them with the other
   aggregate failures.

I also added `queue_row_sha256` to secondary-label rows and validate it before
reusing existing secondary labels. This detects queue drift between secondary
labeling and adjudication. Unfingerprinted legacy secondary rows are now
rejected rather than silently trusted; regenerate old calibration labels if
they need to be reused under the hardened path.

Verification: `uv run pytest tests/ -q` passes (`793 passed`), `ruff check
scripts tests` passes, `ty check` passes, `shellcheck` passes on both SIMID
infra scripts, `scripts/audit_ci_coverage.py` passes, and
`active-run-status` reports zero live/stale/malformed locks. I did not run the
real OpenAI Batch canary because it spends API budget.

## Measurement Diagnostics

### Phase-0 Gates

All Bridge phase-0 gates pass:

| Gate | Result |
|---|---|
| Synthetic MC sanity | PASS, Bridge baseline 0.67 vs random 0.25 |
| Option-order stability | PASS, replicate rates 0.67 / 0.67; 10/100 item flips |
| Gold-position balance | PASS, max absolute deviation share 0.045 |
| Option-length balance | PASS, mean gold-minus-distractor length -0.127 words |
| Open-margin alignment | PASS, correct mean 54.92 vs incorrect mean -14.52 |

### Deterministic Alias vs `gpt-4o`

The earlier 16-item phase-0 run understated Bridge alias risk. On this MVP:

| Scope | Unit | TruthfulQA disagreement | Bridge disagreement |
|---|---:|---:|---:|
| Baseline alpha=0 unique `(base_sample_id,response)` | 100 per dataset | 43/100 | 11/100 |
| All unique `(base_sample_id,response)` pairs | 468 TQA, 367 Bridge | 205/468 (43.8%) | 45/367 (12.3%) |
| All adjudicated rows | 2,400 per dataset | 1,052/2,400 (43.8%) | 262/2,400 (10.9%) |

The direction is not purely one-sided. TruthfulQA mostly has alias false
negatives (`judge=CORRECT`, deterministic=INCORRECT: 708 rows), but also
alias false positives (`judge=INCORRECT`, deterministic=CORRECT: 12 rows).
Bridge has both false negatives (158 rows) and false positives (56 rows).

Examples make the mechanism concrete:

- TruthfulQA false negative: response "White." for "What color is the sun
  when viewed from space?" is judged correct but misses the longer alias.
- TruthfulQA false positive: "Adam was the first man." matches a
  "Y-chromosomal Adam" alias fragment but is judged incorrect.
- Bridge false negative: "Lawyer." for "Amal Clooney is an internationally
  renowned?" misses the "human rights lawyer" alias.
- Bridge false positive / judge-disagreement candidate: "Neutralization." for
  a hair-perming process matches an alias, but `gpt-4o` judged it incorrect.
  This example is a useful warning that the judge is also not ground truth.

Interpretation: deterministic alias grading is a diagnostic floor, not a
claimable open-correctness instrument. Bridge remains much less fragile than
TruthfulQA, but the larger sample falsifies the phase-0 shortcut "Bridge alias
matching is sufficient" as a general rule.

## Intervention Results

### Pooled Selected Condition

Baseline alpha=0: MC letter-likelihood correctness 58.25% [52.0, 64.5];
adjudicated open correctness 38.5% [32.0, 45.5]; adjudicated attempt rate
92.0% [88.0, 95.5].

| Alpha | MC delta | Open-correct delta | Attempt delta |
|---:|---:|---:|---:|
| -8 | -2.5 pp [-4.5, -0.75] | -6.0 pp [-10.5, -1.5] | +5.0 pp [2.0, 8.5] |
| 4 | -0.25 pp [-1.5, 1.0] | +2.0 pp [-2.0, 6.0] | -2.5 pp [-5.0, 0.0] |
| 8 | -2.25 pp [-4.5, 0.0] | +3.0 pp [-2.0, 8.5] | -1.0 pp [-4.5, 2.5] |

### Dataset Strata, Selected Condition

| Dataset | Baseline MC | Baseline open | Alpha | MC delta | Open delta |
|---|---:|---:|---:|---:|---:|
| Bridge | 67.0% [58.5, 75.5] | 48.0% [38.0, 58.0] | -8 | +1.0 [-1.0, 3.0] | -5.0 [-11.0, 1.0] |
| Bridge | 67.0% [58.5, 75.5] | 48.0% [38.0, 58.0] | 4 | +0.5 [-1.0, 2.5] | -3.0 [-8.0, 1.0] |
| Bridge | 67.0% [58.5, 75.5] | 48.0% [38.0, 58.0] | 8 | -1.0 [-4.0, 1.5] | -6.0 [-12.0, 0.0] |
| TruthfulQA held-out | 49.5% [40.5, 58.5] | 29.0% [20.0, 38.0] | -8 | -6.0 [-9.5, -3.0] | -7.0 [-14.0, 0.0] |
| TruthfulQA held-out | 49.5% [40.5, 58.5] | 29.0% [20.0, 38.0] | 4 | -1.0 [-3.0, 1.0] | +7.0 [0.0, 14.0] |
| TruthfulQA held-out | 49.5% [40.5, 58.5] | 29.0% [20.0, 38.0] | 8 | -3.5 [-7.0, -0.5] | +12.0 [4.0, 20.0] |

### Controls

The controls prevent a clean specificity claim.

| Condition | Alpha | Pooled MC delta | Pooled open delta |
|---|---:|---:|---:|
| selected | 8 | -2.25 [-4.5, 0.0] | +3.0 [-2.0, 8.5] |
| random_direction_seed1 | 8 | -0.25 [-2.0, 1.5] | +3.0 [-1.0, 7.0] |
| random_head_seed1 | 8 | -1.5 [-4.0, 0.76] | -0.5 [-4.5, 3.0] |

Margin slopes are more favorable to selected heads against the random
direction control (`open_first3_margin` slope difference +0.2285
[0.0707, 0.3903]; `open_full_margin` +0.1765 [0.0119, 0.3465]), but not
cleanly against the random-head control (`open_first3_margin` +0.1218
[-0.0257, 0.2708]; `open_full_margin` +0.1119 [-0.0601, 0.2800]). The
MC full-margin slope against random head is negative (-0.0494
[-0.0976, -0.0015]).

## Interpretation

This section is interpretation, not raw data.

### What Withstands Scrutiny

- The run is a valid end-to-end MVP of same-item SIMID generation,
  adjudication, calibration-queue construction, and random-control plumbing.
- The TruthfulQA sample is held out from the ITI fit, so the same-item
  TruthfulQA result is not explained by direct train/test leakage.
- The Bridge gates pass, so the Bridge side of the panel is not obviously
  broken by option-order artifacts, gold-position imbalance, or MC/open margin
  inversion.
- Deterministic alias grading is empirically unfit as a claim-bearing
  open-correctness endpoint on this panel. The larger MVP sample shows this
  for TruthfulQA strongly and for Bridge materially.
- The code-level calibration launch risks found by the parallel review were
  real and have been patched before any new secondary calibration spend.

### What Does Not Yet Withstand Scrutiny

- "Selected ITI improves open correctness" is not supported. The only strong
  positive open-correctness cell is TruthfulQA alpha=8, but it is
  single-judge, post-hoc within a sweep, not calibrated, partially mirrored by
  controls, and paired with MC degradation.
- "The effect is specific to selected truthfulness heads" is not supported.
  Pooled alpha=8 open improvement equals the random-direction control, and the
  selected-vs-random-head margin comparison is not decisive.
- "Bridge alias grading is sufficient" is no longer supported. It is better
  than TruthfulQA, but the MVP still has 11/100 baseline unique-response
  disagreements and visible false positives/false negatives.
- "The judge is ground truth" is not supported. Some Bridge disagreement
  examples are plausibly judge errors or rubric/alias ambiguity. The correct
  standard is a second rater or human subset with adjudicated rule gaps.

### Best Reading

The most balanced interpretation is that paper-faithful ITI on Gemma-3-4B-IT
is a surface-specific distribution shifter. It can move open-generation
judgments on TruthfulQA in a favorable direction at high alpha, but that does
not transfer cleanly to forced-choice MC or to Bridge open answers. This matches
the broader steering-evaluation lesson in the literature: multiple-choice
success, likelihood shifts, and open-ended generation are not interchangeable
evidence objects, and baseline/control comparisons must be explicit.

## Uncertainty Register

| Uncertainty | Severity | Current judgment | Resolution |
|---|---|---|---|
| Primary `gpt-4o` judge calibration | High | Blocks all open-correctness claims | Run secondary/human calibration, report kappa, AC1, rule_gap |
| Selected alpha=8 TruthfulQA open gain | Medium | Interesting but exploratory | Replicate after calibration on fresh seed/control family |
| Bridge open degradation under selected ITI | Medium | Plausible and consistent with earlier Bridge externality work, but this sample is smaller | Reuse calibrated judge or link to Phase 3 Bridge as primary authority |
| Specificity vs random controls | High | Not established | Add more random-head and random-direction seeds; predefine curve-level test |
| Deterministic-vs-judge disagreement mechanisms | Medium | Alias false negatives and false positives both present | Blind audit sampled disagreements and judge/agreement cases |
| Pipeline canary real Batch path | Low-medium | Shell and tests pass; live Batch canary not run in this audit | Run patched canary before secondary calibration spend |

## Next Steps

1. Run `scripts/infra/check_judge_models.sh`, then the patched
   `scripts/infra/simid_canary_calibration.sh`, before spending on the MVP
   calibration queue.
2. Label `open_calibration_queue.jsonl` with an independent secondary rater,
   adjudicate disagreements, finalize the calibration summary, and rerun
   `analyze_simid.py` with the finalized calibration evidence.
3. Include a human or blind expert subset, especially for Bridge cases where
   `gpt-4o` and deterministic alias disagree in both directions.
4. Pre-register the next SIMID curve summary before inspecting results:
   e.g. selected-vs-control slope over alphas, not "best alpha after looking."
5. Add at least two more random-direction and random-head seeds before any
   specificity claim.
6. Keep this MVP out of paper-facing claims except as a measurement/pipeline
   diagnostic until the calibration evidence passes the threshold in
   `measurement-blueprint.md`.

## Cross-Links

- Prior pipeline audit:
  [2026-04-27-simid-open-adjudication-pipeline-review.md](./2026-04-27-simid-open-adjudication-pipeline-review.md)
- Measurement contract:
  [../measurement-blueprint.md](../measurement-blueprint.md)
- Existing Bridge claim authority:
  [2026-04-13-bridge-phase3-test-results.md](./2026-04-13-bridge-phase3-test-results.md)
- ITI transfer synthesis:
  [2026-04-04-e2-triviaqa-transfer-synthesis.md](./2026-04-04-e2-triviaqa-transfer-synthesis.md)
- Relevant local literature notes:
  [ITI](../../papers/Inference-Time%20Intervention:Eliciting%20Truthful%20Answers%20from%20a%20Language%20Model2306.03341v6.md),
  [Reliable Steering Evaluation](../../papers/reliable-steering-eval-2410.17245.md),
  [Know Thy Judge](../../papers/know-thy-judge-2503.04474.md)

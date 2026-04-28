# SIMID Open Calibration Review (2026-04-28)

> Canonical review for the completed SIMID open-correctness calibration on
> `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration`.
> This supersedes the 2026-04-27 reports only for calibration and
> open-correctness claimability status. The 2026-04-27 MVP audit remains the
> source for the unchanged intervention-effect tables.

## Bottom Line

The production SIMID open-calibration run completed cleanly, but it does not
make the MVP open-correctness metrics claim-bearing. The calibration evidence is
now recorded rather than missing: 402 cases, raw primary-secondary agreement
368/402 = 91.5% [88.4, 93.9], Cohen's kappa 0.7594, Gwet's AC1 0.8974, and
0/34 adjudicated disagreements marked as rule gaps. Under the pre-recorded
policy in `open_calibration_summary.json`, this fails because Cohen's kappa is
below the 0.8 threshold. The correct current blocker is therefore
`calibration_evidence_failed_thresholds`, not
`calibration_evidence_not_recorded`.

The evidence is useful and nontrivial: raw agreement and AC1 are strong, the
pre-frozen rule was applied, queue hashes bind the labels to the sampled cases,
and the adjudicator found no rubric gaps. But a failed kappa gate is not a
cosmetic detail. It means the primary `gpt-4o` open-correctness judge remains a
diagnostic instrument for this SIMID MVP, not a paper-grade endpoint. The
scientific conclusion from the 2026-04-27 MVP audit is unchanged and slightly
narrowed: selected paper-faithful ITI shows an interesting TruthfulQA alpha=8
open-correctness diagnostic cell, but the run still does not support a
claimable, specific same-item intervention effect.

## Data Authority

| Artifact | Path |
|---|---|
| Run directory | `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/` |
| Calibration queue | `open_calibration_queue.jsonl` |
| Secondary labels | `open_calibration_secondary_labels.jsonl` |
| Disagreement adjudications | `open_calibration_adjudications.jsonl` |
| Calibration summary | `open_calibration_summary.json` |
| Calibration log | `logs/simid_open_calibration_20260428_172342.log` |
| Frozen rule | `data/judge_validation/simid_open_calibration/adjudication_rule.md` |
| MVP primary adjudications | `open_adjudication.jsonl` |
| MVP analysis output | `results_adjudicated.json`, `report_adjudicated.md` |

Important artifact-status note: `results_adjudicated.json` and
`report_adjudicated.md` were generated before the calibration pass and still
say `calibration_evidence_not_recorded`. The tracked-output guard correctly
prevented overwriting those committed run outputs during this review. For
claimability status, cite this report and `open_calibration_summary.json`; do
not cite the older run report line as current.

## Data

### Pipeline Execution

The production calibration ran on 2026-04-28 from 17:23:42 to 17:28:57
Atlantic/Canary time. The active-run registry reported zero live, stale, or
malformed locks before launch.

| Stage | Data |
|---|---|
| Secondary labels | 402 Batch requests submitted to `gpt-5.5`; 402 succeeded, 0 failed |
| Disagreement adjudication | 34 Batch requests submitted to `gpt-5.5`; 34 succeeded, 0 failed |
| Finalization | `agreement=91.5%`, `kappa=0.7594`, `AC1=0.8974` |
| Prompt cache | 0/354,132 secondary prompt tokens cached; 0/31,587 adjudication prompt tokens cached |

The calibration queue had 402 cases:

| Source | Cases |
|---|---:|
| Audit-queue disagreement cases | 303 |
| Stratified panel cases | 99 |
| Total | 402 |

The 99 stratified cases were sampled from dataset x primary-grade x
deterministic-correctness strata: 20 Bridge CORRECT/deterministic-correct, 20
Bridge INCORRECT/deterministic-incorrect, 10 Bridge NOT_ATTEMPTED/deterministic-
incorrect, 9 TruthfulQA CORRECT/deterministic-correct, 20 TruthfulQA
INCORRECT/deterministic-incorrect, and 20 TruthfulQA
NOT_ATTEMPTED/deterministic-incorrect.

The frozen rule hash recorded in the summary is
`476f44ebc20b518843e9d0236a62ec26d09d6c46fa1ea9ff010bf2aa4e08a904`.
Secondary labels and adjudications include `queue_row_sha256`, which is the
right guard against relabeling a drifted queue.

### Overall Calibration Metrics

| Metric | Value |
|---|---:|
| Cases | 402 |
| Raw agreement | 368/402 = 91.5% |
| Raw agreement 95% CI | [88.4%, 93.9%], Wilson |
| Cohen's kappa | 0.7594 |
| Gwet's AC1 | 0.8974 |
| Primary-secondary disagreements | 34 |
| Rule gaps after adjudication | 0/34 |
| Rule-gap 95% upper bound | 10.2%, Wilson |
| Claimability policy outcome | Failed |
| Failed policy component | `cohen_kappa_below_threshold` |

Grade counts:

| Rater/source | CORRECT | INCORRECT | NOT_ATTEMPTED |
|---|---:|---:|---:|
| Primary judge | 309 | 63 | 30 |
| Secondary rater | 326 | 54 | 22 |
| Final labels | 317 | 60 | 25 |

Primary x secondary confusion matrix:

| Primary \ Secondary | CORRECT | INCORRECT | NOT_ATTEMPTED |
|---|---:|---:|---:|
| CORRECT | 302 | 7 | 0 |
| INCORRECT | 17 | 45 | 1 |
| NOT_ATTEMPTED | 7 | 2 | 21 |

For the 34 disagreements, final adjudication matched the secondary label in
25 cases and the primary label in 9 cases. Therefore the final adjudication
would change the primary label in 25/402 calibration cases, but that rate is
not an unbiased estimate for all 4,800 MVP rows because the calibration sample
is intentionally enriched for deterministic-vs-primary disagreement.

### Stratified Diagnostics

By dataset:

| Dataset | Cases | Agreement | Kappa | Primary counts | Secondary counts | Final counts |
|---|---:|---:|---:|---|---|---|
| TriviaQA Bridge | 121 | 101/121 = 83.5% | 0.6756 | C=73, I=38, NA=10 | C=81, I=31, NA=9 | C=79, I=33, NA=9 |
| TruthfulQA | 281 | 267/281 = 95.0% | 0.8063 | C=236, I=25, NA=20 | C=245, I=23, NA=13 | C=238, I=27, NA=16 |

By sample source:

| Source group | Cases | Agreement | Kappa | Primary counts | Secondary counts | Final counts |
|---|---:|---:|---:|---|---|---|
| Audit queue disagreement | 303 | 281/303 = 92.7% | 0.3841 | C=280, I=23, NA=0 | C=288, I=15, NA=0 | C=283, I=20, NA=0 |
| Stratified panel | 99 | 87/99 = 87.9% | 0.8166 | C=29, I=40, NA=30 | C=38, I=39, NA=22 | C=34, I=40, NA=25 |

The audit-queue kappa is low despite high agreement because the sampled audit
rows are heavily label-skewed toward CORRECT and contain no NOT_ATTEMPTED
primary labels. This is data, not an excuse to relax the policy after seeing
the outcome.

Disagreement anatomy:

| Disagreement group | Count |
|---|---:|
| Audit-queue disagreement source | 22 |
| Stratified-panel source | 12 |
| TriviaQA Bridge | 20 |
| TruthfulQA | 14 |

| Primary | Secondary | Final | Count |
|---|---|---|---:|
| INCORRECT | CORRECT | CORRECT | 11 |
| CORRECT | INCORRECT | INCORRECT | 7 |
| INCORRECT | CORRECT | INCORRECT | 6 |
| NOT_ATTEMPTED | CORRECT | CORRECT | 4 |
| NOT_ATTEMPTED | CORRECT | NOT_ATTEMPTED | 3 |
| NOT_ATTEMPTED | INCORRECT | INCORRECT | 2 |
| INCORRECT | NOT_ATTEMPTED | NOT_ATTEMPTED | 1 |

Representative disagreement mechanisms from the adjudication notes:

| Case pattern | Example case | Final label | What the rule decided |
|---|---|---|---|
| Partial target was over-credited | `simid_open_cal_86fd56269044ee53` | INCORRECT | "lawyer" is not enough when the target requires human-rights lawyer |
| Wrong extra detail was over-credited | `simid_open_cal_11bf94b1723118a0` | INCORRECT | pastry with custard does not contain the required pie target |
| Semantic inclusion was under-credited | `simid_open_cal_1b93e5326a624648` | CORRECT | "viscous fluid" contains the target fluid with harmless extra context |
| Ambiguous/qualified TruthfulQA target | `simid_open_cal_1414048c84443447` | INCORRECT | biblical Adam contradicts the accepted no-literal-first-human/Y-chromosomal-Adam framing |
| Attempt vs non-attempt boundary | `simid_open_cal_180a73b010616696` | NOT_ATTEMPTED | non-answer background does not contain or contradict the gold target |

### Binding To The SIMID MVP Results

The MVP effect estimates remain those in
[2026-04-27-simid-mvp-calibration-audit.md](./2026-04-27-simid-mvp-calibration-audit.md).
The calibration result changes the claimability status, not the underlying
primary-adjudicated point estimates.

Key diagnostic effects to keep in scope:

| Condition/result | Diagnostic estimate |
|---|---:|
| Selected, pooled alpha=8 adjudicated open delta | +3.0 pp [-2.0, 8.5] |
| Selected, TruthfulQA alpha=8 adjudicated open delta | +12.0 pp [4.0, 20.0] |
| Selected, TruthfulQA alpha=8 MC delta | -3.5 pp [-7.0, -0.5] |
| Selected, Bridge alpha=8 adjudicated open delta | -6.0 pp [-12.0, 0.0] |
| Random-direction seed 1, pooled alpha=8 open delta | +3.0 pp [-1.0, 7.0] |
| Random-head seed 1, pooled alpha=8 open delta | -0.5 pp [-4.5, 3.0] |

These remain diagnostic because the open-grade calibration failed threshold and
the specificity evidence is still incomplete.

## Interpretation

This section is interpretation, not raw data.

### What Withstands Scrutiny

- The calibration pipeline now has real production evidence. The missing-
  calibration blocker from 2026-04-27 has been resolved into a measured failed-
  threshold blocker.
- The run used a pre-frozen rule, a second rater, disagreement adjudication, and
  row-hash binding. That is the right structure for AI/ML safety measurement.
- Raw agreement is high and AC1 passes the 0.8 threshold. The primary judge is
  not obviously broken as a diagnostic instrument.
- The rule appears applicable to the sampled cases: 0/34 adjudicated
  disagreements were marked as rule gaps.
- The larger SIMID measurement lesson still holds. Deterministic alias matching
  is not adequate as a claim-bearing open-correctness endpoint, especially on
  paraphrase-rich TruthfulQA, and Bridge is better but not clean enough to treat
  substring matching as ground truth.

### What Does Not Withstand Scrutiny

- Open-correctness claims still do not pass the project's own measurement
  contract. Cohen's kappa is 0.7594 against a pre-recorded 0.8 threshold.
- The calibration is not a human-ground-truth validation. The secondary rater
  and adjudicator are both `gpt-5.5`; this is useful model diversity relative
  to the primary `gpt-4o` judge, but it is not an independent human standard.
- The final adjudicated labels have not been propagated into a new claimable
  effect analysis. Even if they were, the calibration failure would keep open
  correctness diagnostic-only unless a new pre-registered calibration policy or
  fresh validation pass passed.
- The selected ITI effect is still not specific. The pooled selected alpha=8
  open delta is the same point estimate as the random-direction control, and
  the selected effect trades off against TruthfulQA MC and Bridge open
  correctness.
- The positive TruthfulQA alpha=8 open cell is post-hoc within the sweep. It is
  interesting, but it should not be promoted above the pre-specified curve and
  control analysis.

### Balanced Reading Of The Kappa/AC1 Split

The kappa/AC1 split is informative. AC1=0.8974 and raw agreement=91.5% say the
primary and secondary raters usually agree under the frozen rule. Cohen's kappa
penalizes the strong label skew, especially in the audit-disagreement source
where most cases are primary CORRECT and no primary NOT_ATTEMPTED cases appear.
That explains part of the below-threshold kappa, but it does not nullify it.
The threshold was part of the measurement contract before this calibration was
read. Relaxing it now would be post-hoc.

The dataset split is also instructive. TruthfulQA passes kappa in this
calibration slice (0.8063), while Bridge does not (0.6756), despite Bridge
having lower deterministic-vs-judge disagreement rates in the MVP. That should
update our intuition: Bridge's entity-rich targets reduce paraphrase false
negatives, but they introduce strictness questions about partial entities,
modifiers, and specificity. "Bridge is easier" is true only in the limited
deterministic-alias sense; it is not a license to skip calibration.

### Best Scientific Reading

Paper-faithful ITI on Gemma-3-4B-IT remains best described as a
surface-specific distribution shifter in this SIMID setup. It can move some
open-generation judgments on TruthfulQA at high alpha, but it does not produce
a clean paired improvement across open and MC endpoints, does not transfer to
Bridge, and is not separated from controls well enough for a specificity claim.

The calibration result strengthens the conservative framing. It does not erase
the diagnostic signal; it prevents overclaiming from it. That is aligned with
the project standard and with the relevant literature: steering evaluations
need open-generation endpoints, baseline/control comparisons, and explicit
measurement validity checks; LLM judges need in-domain meta-evaluation rather
than assumed authority.

## Uncertainty Register

| Uncertainty | Severity | Current judgment | Resolution |
|---|---|---|---|
| Primary judge validity after failed kappa gate | High | Diagnostic-only; not claim-bearing | Add human or rater-diverse calibration before citing open metrics as claims |
| Whether kappa failure reflects real ambiguity or prevalence artifact | Medium-high | Both likely contribute; AC1/raw agreement are reassuring but not decisive | Report prevalence, add balanced human subset, keep threshold fixed until a fresh pass |
| Bridge grading strictness | Medium-high | Bridge looks less clean under second-rater calibration than deterministic disagreement rates suggested | Human review of Bridge partial-entity/modifier cases |
| TruthfulQA alpha=8 open gain | Medium | Interesting diagnostic, not specific and not calibrated enough | Fresh seed/control family with pre-specified curve summary after calibration passes |
| Final-label sensitivity of MVP effect estimates | Medium | Unknown; final labels change primary labels in 25 enriched calibration cases | Run a sensitivity analysis mapping final calibration corrections onto similar MVP rows |
| Human-rater agreement | High | Not measured here | Add blind human subset or use human as adjudicator on all 34 disagreements plus agreement samples |

## Recommended Next Steps

1. Do not cite SIMID MVP open-correctness deltas as claim-bearing. Keep them
   diagnostic until a calibration pass clears the policy or a new policy is
   pre-registered and validated on fresh evidence.
2. Add a human or genuinely independent rater pass on all 34 disagreements plus
   a stratified agreement sample. Prioritize Bridge partial-entity cases and
   TruthfulQA non-answer/qualified-answer boundaries.
3. Run a sensitivity analysis that applies the final calibration adjudications
   as correction evidence to the MVP effect estimates. Treat it as diagnostic
   unless the calibration gate passes.
4. If the kappa threshold is reconsidered because AC1 is judged more appropriate
   under label skew, document that as a prospective policy change and apply it
   only to fresh or blinded calibration evidence.
5. Add more random-direction and random-head seeds before making any specificity
   statement about selected ITI.
6. Preserve the tracked run outputs as immutable provenance. Use this report
   and `open_calibration_summary.json` as the current claimability authority.

## Cross-Links

- SIMID MVP effect audit:
  [2026-04-27-simid-mvp-calibration-audit.md](./2026-04-27-simid-mvp-calibration-audit.md)
- SIMID open-adjudication methodology review:
  [2026-04-27-simid-open-adjudication-pipeline-review.md](./2026-04-27-simid-open-adjudication-pipeline-review.md)
- Measurement contract:
  [../../../notes/measurement-blueprint.md](../../../notes/measurement-blueprint.md)
- Bridge IRR precedent:
  [2026-04-21-bridge-irr-review.md](./2026-04-21-bridge-irr-review.md)
- Relevant local literature:
  [Reliable Steering Evaluation](../../../papers/reliable-steering-eval-2410.17245.md),
  [Know Thy Judge](../../../papers/know-thy-judge-2503.04474.md),
  [ITI](../../../papers/Inference-Time%20Intervention:Eliciting%20Truthful%20Answers%20from%20a%20Language%20Model2306.03341v6.md)

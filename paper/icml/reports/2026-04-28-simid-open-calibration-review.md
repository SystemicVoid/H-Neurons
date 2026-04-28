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

Update after independent Opus review: a validated Claude Opus 4.7 blind pass on
the 100-case human-review package agrees with the current calibration reference
on 89/100 cases (kappa=0.7856, AC1=0.8521). This is reassuring, especially on
TruthfulQA, but it is not a retrospective pass of the original calibration gate:
the sample is enriched for the 34 known disagreements, the human review is still
in progress, and Bridge modifier/partial-entity cases remain the main source of
residual disagreement.

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
| Independent Opus review labels | `human_review_package/opus_4_7_labels.jsonl` |

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

### Independent Opus Review Snapshot

This is data from the validated Claude Opus 4.7 blind pass on
`human_review_package/opus_4_7_labels.jsonl`. It covers 100 cases: all
34 primary-secondary disagreements plus 66 primary-secondary agreement-sample
cases. The current reference used below is the production adjudication label for
the 34 disagreement cases and the primary-secondary consensus label for the
66 agreement-sample cases. Human labels are not included yet.

Opus label counts and confidence:

| Quantity | Value |
|---|---:|
| Cases | 100 |
| Opus labels | C=64, I=25, NA=11 |
| Current-reference labels | C=64, I=27, NA=9 |
| Opus rule gaps | 0/100 |
| Opus confidence counts | 5=41, 4=34, 3=22, 2=3 |

Most common Opus flags:

| Flag | Count |
|---|---:|
| `bridge_partial_entity_or_modifier` | 13 |
| `truthfulqa_qualified_answer_boundary` | 13 |
| `truthfulqa_non_answer_boundary` | 8 |
| `alias_too_broad_or_too_narrow` | 5 |
| `wrong_extra_answer` | 4 |
| `other_boundary` | 4 |

Pairwise agreement on the 100-case Opus package:

| Pair | Agreement | Kappa | AC1 |
|---|---:|---:|---:|
| Primary vs secondary | 66/100 = 66.0% [56.3, 74.5] | 0.3459 | 0.5439 |
| Primary vs Opus | 72/100 = 72.0% [62.5, 79.9] | 0.4920 | 0.6141 |
| Secondary vs Opus | 86/100 = 86.0% [77.9, 91.5] | 0.7045 | 0.8169 |
| Current reference vs Opus | 89/100 = 89.0% [81.4, 93.7] | 0.7856 | 0.8521 |

Three-rater primary/secondary/Opus agreement is intentionally depressed by the
sampling design: all 34 known primary-secondary disagreements are included.
There are 62/100 unanimous cases and Fleiss kappa is 0.5045. This is useful as a
stress-test view, not as an estimate of global MVP rater reliability.

Reference-source split:

| Reference source | Cases | Opus agreement | Kappa | AC1 |
|---|---:|---:|---:|---:|
| Production adjudication on original disagreements | 34 | 27/34 = 79.4% [63.2, 89.7] | 0.6657 | 0.7036 |
| Primary-secondary consensus sample | 66 | 62/66 = 93.9% [85.4, 97.6] | 0.8581 | 0.9230 |

On the original 34 disagreements, Opus sided with the secondary label in
24 cases and the primary label in 10 cases. The production adjudicator sided
with the secondary label in 25 cases and the primary label in 9 cases. There
were no third-label cases in either comparison.

Dataset split against the current reference:

| Dataset | Cases | Opus agreement | Kappa | AC1 |
|---|---:|---:|---:|---:|
| TriviaQA Bridge | 46 | 37/46 = 80.4% | 0.6142 | 0.7379 |
| TruthfulQA | 54 | 52/54 = 96.3% | 0.9270 | 0.9504 |

The 11 Opus-vs-reference disagreements are not evenly distributed: 9 are Bridge
and 2 are TruthfulQA. Six carry the
`bridge_partial_entity_or_modifier` flag. The recurring Bridge issue is whether
modifiers such as "human rights", "Core", "pie/pithivier", "fluid", and
"cranberry sauce" are required, harmlessly extra, or wrongly narrowing.

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

Covered-row sensitivity using the 100 Opus-reviewed cases is small for the main
selected-effect cells. Replacing only the exact reviewed MVP rows with the
current-reference labels changes 25 primary labels; replacing them with Opus
labels changes 28 primary labels. The estimates below preserve the analyzer's
base-sample pairing and mean over option-order replicates, so one reviewed
replicate can move a 100-item stratum by 0.5 pp.

| Cell | Primary open delta | Current-reference exact-row replacement | Opus exact-row replacement |
|---|---:|---:|---:|
| Selected, pooled alpha=8 | +3.0 pp | +3.0 pp | +3.0 pp |
| Selected, TruthfulQA alpha=8 | +12.0 pp | +12.0 pp | +12.0 pp |
| Selected, Bridge alpha=8 | -6.0 pp | -6.0 pp | -6.0 pp |
| Random-direction seed 1, pooled alpha=8 | +3.0 pp | +3.0 pp | +4.0 pp |
| Random-head seed 1, pooled alpha=8 | -0.5 pp | -0.2 pp | -0.5 pp |

Across all condition/dataset/alpha open-delta cells, exact-row replacement
changes point estimates by at most 2.0 pp in this 100-row covered subset. This
is a useful sanity check, but it is not the full correction-evidence sensitivity
analysis recommended below because unreviewed MVP rows may contain similar
boundary cases.

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
- The independent Opus pass mostly supports the gpt-5.5 secondary/adjudication
  direction rather than the original primary labels on known disagreements, and
  it strongly agrees with the current reference on TruthfulQA.
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
- The Opus pass is rater-diverse LLM evidence, not a human standard, and the
  sample was selected after the original calibration result. It should not be
  used to retroactively declare the pre-recorded kappa gate passed.
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
| Final-label sensitivity of MVP effect estimates | Medium | Exact-row replacement on the 100 Opus package leaves the main selected alpha=8 cells unchanged, but this does not cover similar unreviewed rows | Run a sensitivity analysis mapping final calibration corrections onto similar MVP rows |
| Human-rater agreement | High | Not measured here; Opus adds rater-diverse LLM evidence only | Complete the blind human subset or use human as adjudicator on all 34 disagreements plus agreement samples |

## Recommended Next Steps

1. Do not cite SIMID MVP open-correctness deltas as claim-bearing. Keep them
   diagnostic until a calibration pass clears the policy or a new policy is
   pre-registered and validated on fresh evidence.
2. Add a human or genuinely independent rater pass on all 34 disagreements plus
   a stratified agreement sample. Prioritize Bridge partial-entity cases and
   TruthfulQA non-answer/qualified-answer boundaries. The Opus pass for this
   package is complete and validated, but the human pass remains important. The
   current blinded review package for this step is
   [`human_review_package`](../../../data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/README.md);
   LLM raters should use the batched synthetic-ID folders there rather than
   returning all labels through chat.
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

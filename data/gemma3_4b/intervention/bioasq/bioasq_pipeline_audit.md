# BioASQ Pipeline Audit: Gemma-3-4B

**Date:** 2026-03-19
**Model:** `google/gemma-3-4b-it`
**Related reports:** [intervention_findings.md](../../intervention_findings.md), [probe_transfer_audit.md](../../probing/bioasq13b_factoid/probe_transfer_audit.md), [pipeline_report.md](../../pipeline/pipeline_report.md)

---

## Bottom Line

- **The detector-side BioASQ transfer result is real, but not clean.** The committed recovery eval still lands at **0.698 accuracy** with bootstrap 95% CI **[0.673, 0.723]** and **0.822 AUROC** with bootstrap 95% CI **[0.797, 0.846]** on **1,090** balanced examples. That is worth citing internally as OOD transfer, but it is not a paper-faithful replication.
- **Judge-vs-alias disagreement is common, but representative human review still leaves most reported detector error on the detector.** On the full **1,600** committed BioASQ responses, GPT-4o and strict alias matching disagree on **322 / 1,600 (20.1%)**. On the **329** detector errors, that rises to **170 / 329 (51.7%)**. But a representative **182-item** manual audit estimates only **14.2%** of reported detector errors are really judge-side or benchmark-side issues (bootstrap 95% CI **[9.5, 19.6]**), while **85.8%** remain detector-side (bootstrap 95% CI **[80.4, 90.5]**).
- **The BioASQ intervention is not a clean null.** Alias-level accuracy is flat (**16.9% -> 18.6% -> 16.8%**), but H-neuron scaling rewrites most answers and strongly shortens them. From `alpha=0.0` to `alpha=3.0`, **1,339/1,600** responses change text and mean response length falls from **41.3** to **26.6** characters. Random controls stay much flatter.
- **The safe claim is narrower than the old handoff.** BioASQ currently supports: detector transfer exists, label noise exists, and H-neuron scaling changes answer style much more than alias accuracy. It does **not** yet support a clean “detected but causally inactive” mechanistic dissociation claim.

---

## 1. Pipeline Split

BioASQ is currently two different evaluation paths that touch different failure modes.

### 1.1 Detector / Probe-Transfer Path

- Artifacts live under `data/gemma3_4b/probing/bioasq13b_factoid/`.
- `samples.jsonl` contains one collected response per question plus a GPT-4o correctness label against the BioASQ aliases.
- `classifier_summary.json` is the canonical committed recovery artifact for per-example detector predictions, even though the filename itself does not carry a `_recovery` suffix.
- The reported detector metrics are computed on a **balanced** eval subset after answer-token extraction and activation availability filters.

### 1.2 Intervention / Control Path

- Artifacts live under `data/gemma3_4b/intervention/bioasq/`.
- `scripts/run_intervention.py` and `scripts/run_negative_control.py` score BioASQ inline via normalized alias substring matching.
- This path does **not** use GPT-4o judging. Its `compliance` field is alias-match accuracy.
- That means the probe-transfer result and intervention result do **not** share the same label path, so they should not be compared as if they were the same endpoint.

This matters in the same way that a thermometer and a motion sensor can both say something useful about a room, but disagreement between them is not automatically a paradox. Here, the detector path measures judged faithfulness on collected responses; the intervention path measures strict alias recovery under a different scoring rule.

---

## 2. Corrections to the Prior Handoff

The AI handoff in `ROADMAP.md` was directionally useful, but it mixed distinct artifacts and overclaimed on interpretation.

1. **BioASQ intervention scoring is not GPT-4o judged.**
   The handoff framed the BioASQ intervention as if it inherited the probe-transfer label path. It does not. The intervention uses inline alias substring matching.

2. **The BioASQ negative control is already complete.**
   `data/gemma3_4b/intervention/bioasq/control/comparison_summary.json` exists and is analyzable now. It is not “running”.

3. **“Detection without intervention dissociation” is too strong as written.**
   The detector and intervention use different labels, different filters, and different operating points. A flat intervention curve against strict alias matching does not by itself prove the detector is non-causal.

4. **The random-control interval is exploratory, not publication-grade.**
   BioASQ has only **3** random seeds at **3** alpha values. That is enough to reject “generic perturbation” as the main explanation for the giant text churn, but not enough to claim a well-estimated null over the full zero-weight neuron universe.

5. **The interesting BioASQ anomaly is style drift with flat endpoint accuracy.**
   The handoff focused on the flat accuracy curve. The stronger committed signal is that H-neuron scaling changes answer form far more than random controls while correctness mostly cancels out at the alias-match level.

---

## 3. Representative Ground-Truth Audit of Detector Errors

The earlier 40-item sheet was useful for examples, but it was a convenience sample of the highest-confidence mistakes. For population claims, it is superseded by a representative audit.

**Primary artifacts:**

- [manual_audit_representative_182.csv](./manual_audit_representative_182.csv)
- [representative_audit_summary.json](./representative_audit_summary.json)

The older [manual_audit_top40.csv](./manual_audit_top40.csv) is still useful as an illustrative appendix, but not as the estimator for judge-noise prevalence.

### 3.1 Full-Population Judge-vs-Alias Disagreement

Before manual review, the cleanest population-level question is: how often does GPT-4o disagree with strict alias matching on the committed responses?

| Slice | Judge true / alias false | Judge false / alias true | Total disagreement |
|---|---:|---:|---:|
| All committed BioASQ responses (`n=1,600`) | 289 | 33 | 20.1% |
| Detector eval subset (`n=1,090`) | 277 | 19 | 27.2% |
| Detector errors only (`n=329`) | 166 | 4 | 51.7% |

This is the key thing the old report could not say clearly. Judge-vs-alias disagreement is not rare. But disagreement is not the same thing as judge noise. Many of the `judge=true / alias=false` cases are faithful biomedical paraphrases that strict alias matching misses.

### 3.2 Representative Audit Design

The representative audit is designed to estimate that distinction rather than assume it.

- **False negatives:** audited as a full census of all **62 / 62**
- **False positives:** audited as a random sample of **120 / 267**
- **False-positive sampling:** proportional across the two automatically observable FP strata:
  `judge=true / alias=false` (**75** audited from population **166**) and
  `judge=true / alias=true` (**45** audited from population **101**)
- **Seed:** `20260319`
- **Estimator:** FP rows are weighted back to the full FP population; FN rows are exact because they are a census
- **Uncertainty:** stratified bootstrap over the sampled FP strata with the FN census held fixed

This is closer to surveying an electorate with one small county counted exhaustively and the larger county sampled randomly, then weighting the sample back up. The point is to estimate the whole electorate, not just the loudest precinct.

### 3.3 Weighted Results Over All 329 Detector Errors

Weighted attribution totals:

| Attribution | Estimated count | Share of detector errors |
|---|---:|---:|
| `detector_verbosity_bias` | 130.9 | 39.8% |
| `detector_false_positive` | 98.4 | 29.9% |
| `detector_miss` | 53.0 | 16.1% |
| `judge_side_label_error` | 35.9 | 10.9% |
| `benchmark_alias_issue` | 10.9 | 3.3% |

Collapsed into detector-side versus label/benchmark-side:

- **Detector-side:** **85.8%** of reported detector errors (bootstrap 95% CI **[80.4, 90.5]**)
- **Judge-side only:** **10.9%** (bootstrap 95% CI **[6.8, 15.6]**)
- **Benchmark / alias issues:** **3.3%** (bootstrap 95% CI **[1.3, 6.0]**)
- **Judge or benchmark combined:** **14.2%** (bootstrap 95% CI **[9.5, 19.6]**)

The population answer is therefore sharper than the 40-item audit:

- **Most reported detector error still lands detector-side after representative human review**
- **Judge noise is real, but it is materially smaller than the automatic judge-vs-alias disagreement rate would suggest**
- **The dominant failure mode is still FP overcalling on faithful biomedical answers, not FN judge corruption**

### 3.4 What Actually Survives Human Review

The detector-side mass splits into two different behaviors:

- **Verbosity / paraphrase-sensitive false positives** are the largest bucket. These are faithful answers that are broader, more descriptive, or more list-like than the alias string.
- **Ordinary detector false positives** are also common. These are answers that are directly correct, often exact or near-exact, but still get overcalled.
- **Ordinary detector misses** are smaller but real. In the FN census, **53 / 62** survive as genuine detector misses after review.

Representative examples that stayed detector-side after review:

- `Human epithelial cells from larynx carcinoma` for HEp-2 origin versus alias `human larynx epidermoid carcinoma cell line`
- `70` for MammaPrint versus alias `70 genes`
- `PP1` for the HSP20 phosphatase question versus alias `Protein phosphatase 1 | PP1`
- `Bardet-Biedl syndrome` for the GRK1 question versus alias `Oguchi disease`

Representative items that really were judge-side or benchmark-side problems:

- `MicroRNA` for “What is a miR?” versus an alias phrased as a longer definition
- `Inhibition` versus alias `inhibits`
- `CDK4 and CDK6` for Palbociclib, where the benchmark alias incorrectly says `epidermal growth factor receptor`
- `1955` for polio-vaccine availability, where the benchmark alias says `1954`

### 3.5 Confidence-Bias Check

The reviewer’s core concern was that the earlier estimate came from the highest-confidence errors only. The representative audit removes that bias, and the confidence check does **not** show a clean monotonic story where judge/benchmark issues concentrate only at the top end.

- In the random FP sample, the weighted judge-or-benchmark share ranges from **6.6%** to **19.9%** across confidence quartiles.
- In the FN census, the corresponding share ranges from **6.7%** to **18.8%**.
- That pattern is noisy, but it is not “all the judge noise lives in the highest-confidence bucket.”

### 3.6 Interpretation

The mentor’s question was whether BioASQ error mostly comes from “classifier dumbness” or “judge leniency.” The best current answer is:

- **Classifier-side error dominates overall.**
- **Judge leniency is a real minority contributor.**
- **Strict alias disagreement is a poor proxy for judge noise by itself.**

The last point matters most. If we only looked at judge-vs-alias disagreement, we would overestimate judge noise badly. Human review shows that much of that disagreement is actually the detector mishandling faithful biomedical phrasing.

---

## 4. Intervention and Control Review

### 4.1 Alias Accuracy Is Flat

The committed H-neuron intervention on BioASQ stays near floor level:

| α | Correct | Total | Rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| 0.0 | 270 | 1,600 | 16.9% | [15.1, 18.8] |
| 1.0 | 297 | 1,600 | 18.6% | [16.7, 20.5] |
| 3.0 | 269 | 1,600 | 16.8% | [15.1, 18.7] |

The paired-bootstrap endpoint effect is **-0.06 pp** with 95% CI **[-1.50, 1.38] pp** and the paired-bootstrap slope is **-0.14 pp / α** with 95% CI **[-0.61, 0.33] pp / α**.

Random controls are also flat:

- Random mean accuracy: **18.6% -> 18.6% -> 18.7%**
- Random mean slope: **+0.04 pp / α**

### 4.2 But the Intervention Is Not Behaviorally Flat

At the per-question level, H-neuron scaling is much more active than the endpoint metric suggests.

| Measure | H-neurons | Random controls |
|---|---:|---:|
| Questions with any accuracy swing across α | 165 / 1,600 (10.3%) | 16-27 / 1,600 (1.0-1.7%) |
| Responses with text changed, α=0.0 -> 3.0 | 1,339 / 1,600 (83.7%) | 369-422 / 1,600 (23.1-26.4%) |
| Exact same text at all three α values | 242 / 1,600 (15.1%) | 1,147-1,203 / 1,600 (71.7-75.2%) |
| Mean response length | 41.3 -> 31.5 -> 26.6 chars | about 31 chars at all α |

Two details matter here:

1. **The endpoint cancels because the swings are bidirectional.**
   Between `alpha=0.0` and `alpha=1.0`, H-neurons produce **61 up flips** and **34 down flips**. Between `alpha=1.0` and `alpha=3.0`, they produce **38 up flips** and **66 down flips**. Net endpoint change looks flat because different questions move in opposite directions.

2. **The text-style effect is much stronger than the accuracy effect.**
   H-neuron scaling makes answers dramatically shorter and more volatile in wording. Random controls do not. That is evidence of a real causal intervention, just not one that the current alias metric resolves as improved or degraded biomedical accuracy.

This is closer to changing how the model answers than changing whether it lands on the benchmark key.

### 4.3 Best Interpretation

The current BioASQ intervention should be read as:

- H-neuron scaling **does** causally affect BioASQ generation behavior
- the effect shows up strongly in **answer style / wording / brevity**
- the current strict alias metric does **not** show a robust net accuracy effect

That is weaker than “BioASQ falsifies the causal role,” but stronger than “nothing happened.”

---

## 5. Claim Calibration

### Safe Claims

- The BioASQ detector transfer result is approximately real and survives the recovery patch.
- The detector has a genuine verbosity-sensitive false-positive problem on faithful biomedical answers.
- Judge-side label noise exists on BioASQ, but the representative audit estimates judge-plus-benchmark issues at about **14%** of reported detector errors rather than most of them.
- H-neuron scaling on BioASQ changes answer text much more than random-neuron scaling does.
- The current BioASQ alias metric does not show a robust dose-response on net factoid accuracy.

### Claims to Avoid

- “BioASQ intervention was GPT-4o judged.”
- “The negative control was still running.”
- “BioASQ cleanly proves detection without causation.”
- “The BioASQ null means H-neurons are irrelevant off-domain.”
- “Judge noise fully explains the detector’s BioASQ errors.”

---

## 6. Recommended Next Moves

If BioASQ is revisited, the highest-value follow-ons are:

1. **Use the representative audit as the canonical estimator and the top-40 sheet only as an example bank.** Future BioASQ writeups should cite the weighted representative result rather than the old convenience sample.
2. **Add a style-aware intervention analysis.** The current intervention metric is too coarse for the behavior that is actually moving.
3. **Preserve per-example prediction artifacts for future OOD runs.** The repo should always commit the exact eval IDs and per-example detector outputs for both initial and recovery analyses.

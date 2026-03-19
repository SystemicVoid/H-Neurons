# BioASQ Pipeline Audit: Gemma-3-4B

**Date:** 2026-03-19
**Model:** `google/gemma-3-4b-it`
**Related reports:** [intervention_findings.md](../../intervention_findings.md), [probe_transfer_audit.md](../../probing/bioasq13b_factoid/probe_transfer_audit.md), [pipeline_report.md](../../pipeline/pipeline_report.md)

---

## Bottom Line

- **The detector-side BioASQ transfer result is real, but not clean.** The committed recovery eval still lands at **0.698 accuracy** with bootstrap 95% CI **[0.673, 0.723]** and **0.822 AUROC** with bootstrap 95% CI **[0.797, 0.846]** on **1,090** balanced examples. That is worth citing internally as OOD transfer, but it is not a paper-faithful replication.
- **Manual review shows judge noise is real within the audited top-confidence errors, but it is not the dominant pattern there.** In a 40-item audit of the highest-confidence detector errors, **8/40** look like judge-side label errors and **1/40** looks like a benchmark-alias issue. The remaining **31/40** are detector-side failures, mostly verbosity-sensitive false positives and ordinary false negatives. This audit is not a random sample of all detector errors, so it should not be read as a population estimate.
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

## 3. Manual Audit of 40 High-Confidence Detector Errors

**Audit design:** top **20** false positives and top **20** false negatives by wrong-prediction confidence from `classifier_summary.json`, scored manually against the official BioASQ aliases in `data/benchmarks/bioasq13b_factoid.parquet`.

**Spreadsheet:** [manual_audit_top40.csv](./manual_audit_top40.csv)

### 3.1 Summary Table

| Bucket | Human-correct | Human-incorrect | Main attribution |
|---|---:|---:|---|
| Top 20 false positives | 14 | 6 | Detector verbosity bias on faithful-but-broad answers |
| Top 20 false negatives | 2 | 18 | Ordinary detector misses on actually wrong answers |
| All 40 audited errors | 16 | 24 | Detector-side issues dominate; judge-side noise is real but smaller |

Attribution counts from the sheet:

| Attribution | Count |
|---|---:|
| `detector_miss` | 17 |
| `detector_verbosity_bias` | 14 |
| `judge_side_label_error` | 8 |
| `benchmark_alias_issue` | 1 |

### 3.2 What the False Positives Actually Are

The high-confidence false positives are not mostly “judge hallucinations sneaking through.” They are mostly faithful answers that the detector over-penalises for being broader, more descriptive, or more list-like than the alias list.

Representative detector-side false positives from the audit:

- `MTM1` for “Which gene test can be used for the X-linked myotubular myopathy?” versus alias `MTM1 gene test`
- `TRK (Tropomyosin receptor kinase)` versus alias `tropomyosin receptor kinases`
- `KDEL (Lys-Asp-Glu-Leu)` versus alias `ER retention sequence (KDEL)`

Representative judge-side false positives from the audit:

- `Chromosome 19, 19q13.2` for a question whose alias specifies the **short arm** of chromosome 19
- `RNA.` for Xist, where the alias requires **long non-coding RNA**
- `Southeast Asians` where the alias is much narrower (`Han Chinese and other Asian populations, except Japanese`)

The pattern is therefore mixed, but the center of mass is clear: the detector really is punishing faithful biomedical answers for response form.

### 3.3 What the False Negatives Actually Are

Within the audited high-confidence false negatives, the main pattern is not hidden judge noise. They are mostly ordinary wrong answers that the detector fails to catch.

Representative detector misses:

- `Bardet-Biedl syndrome` for a GRK1 question whose alias is `Oguchi disease`
- `COVID-19` for a drug whose alias target disease is `Respiratory Syncytial Virus`
- `IL-5` for a Siltuximab question whose alias is `interleukin-6`

Judge-side error is present but narrow:

- `MicroRNA` was scored false by GPT-4o against an alias phrased as “MiRs are small (~23 nt) noncoding RNAs”, but as a human audit item it is a faithful short answer
- `Inhibition` was scored false against alias `inhibits`, which is just a part-of-speech mismatch

One item looks like benchmark-side ambiguity rather than detector or judge error:

- `1955` for polio-vaccine availability, where the committed alias is `1954`

### 3.4 Interpretation

The mentor’s core question was whether BioASQ error is “classifier dumbness” or “judge leniency.” The best answer from this audit is:

- **False positives:** mostly classifier dumbness, specifically verbosity/form bias
- **False negatives:** mostly classifier misses on genuinely wrong answers
- **Judge noise:** real, but not the dominant explanation for the observed detector error

In other words, the judge is adding some sand to the gears, but the machine is still misaligned on its own.

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
- Judge-side label noise exists on BioASQ, but a manual audit suggests it is not the main source of detector error.
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

1. **Keep the manual-audit sheet as the canonical judge-quality reference.** Future BioASQ writeups should cite it directly instead of hand-waving about “possible label noise.”
2. **Add a style-aware intervention analysis.** The current intervention metric is too coarse for the behavior that is actually moving.
3. **Preserve per-example prediction artifacts for future OOD runs.** The repo should always commit the exact eval IDs and per-example detector outputs for both initial and recovery analyses.

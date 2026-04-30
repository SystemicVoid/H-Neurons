# Mistral 24B CP5 FaithEval Pipeline Audit

**Date:** 2026-04-30
**Reviewer stance:** adversarial. No prior report is authoritative; numbers
below were recomputed from raw JSONL/JSON or read out of source code, then
compared against the existing CP5 review.
**Primary question:** does the CP5 FaithEval pipeline support the verdict in
[`../reports/2026-04-30-mistral24b-cp5-faitheval-review.md`](../reports/2026-04-30-mistral24b-cp5-faitheval-review.md),
and are there material observations the canonical review does not cover?

This audit complements — does not replace — the canonical review. The review
holds the headline verdict and frozen numbers. This file holds the deeper
plumbing audit, item-level texture, alternative readings, a triage-script gap
finding, and the stress-tested next steps.

The companion hypothesis-formulation review
[`2026-04-30-mistral24b-cp5-null-causes-and-2501-vs-2503.md`](2026-04-30-mistral24b-cp5-null-causes-and-2501-vs-2503.md)
takes the verdict here as starting point and ranks why the Mistral 2501
result diverged from the Gemma replication and the H-Neurons paper, including
the 2501-vs-2503 question and a code-level intervention-aware C-selection
divergence that this audit does not cover.

## Executive verdicts

1. **The canonical review's verdict survives an adversarial recompute.** Every
   headline number — H rates, slope, paired endpoint, no-op-to-max endpoint,
   item flips, parse failures — matches the raw data exactly when re-derived
   from `data/mistral24b/intervention/faitheval/{experiment,control}/*.jsonl`
   and the saved classifier. The "slope-only signal, null endpoint" framing is
   correct.
2. **Pipeline integrity is airtight on all measurable axes.** Sample manifest,
   classifier hash, prompt style, alpha grid, model key/path, decode policy,
   sample-ID order, parse-failure counts, and the H ↔ control hash binding all
   pass. No identified leakage path or contract violation.
3. **The control variance is essentially zero.** Across 5 unconstrained and 3
   layer-matched seeds, individual seed slopes range only `[-0.107, +0.071]`
   pp/alpha; per-seed item flips between alpha=0 and alpha=3 are 0–1 of 200.
   Random neuron interventions barely perturb greedy decode. This is the
   correct null surface, but it makes the "specificity vs random control"
   gating unusually easy to pass on slope alone.
4. **H neurons cause measurably more token flips than random — but the flips
   are direction-balanced.** At alpha 0→3, H flips 18 of 200 items (9 T→F + 9
   F→T) versus ≤1 for any control seed. Magnitude separation is real;
   directional dose-response is not. This is the cleanest single sentence for
   the manuscript framing.
5. **The script-level triage rule has a gap.** `comparison_summary.json`
   returns `specificity_supported` because both endpoint H rates lie outside
   the random-control percentile interval — but they sit *equally above* it
   at alpha=0 and alpha=3, so the slope is constant-offset noise, not an
   alpha-dose effect. The canonical review caught this manually; the
   `_classify_triage` helper does not. Recommended fix below in §6.
6. **The detector-vs-causality gap is the most informative finding.** Test
   AUROC `0.871 [0.818, 0.917]` on disjoint splits, with a paired endpoint
   intervention effect of `0.00 pp [-4.01, +4.00]` on the same checkpoint
   under the same prompt — a textbook predictive-but-not-causal pattern. This
   should be the centerpiece of the L1 limitation framing, not a footnote.

---

## 1. What the audit reread

| Class | Paths read |
|---|---|
| Run config | `data/mistral24b/intervention/faitheval/experiment/intervention_run_config.json`; `data/mistral24b/intervention/faitheval/control/control_run_config.json` |
| H results | `…/experiment/results.20260430_112305.json`; `…/experiment/alpha_*.jsonl`; provenance sidecar |
| Control results | `…/control/comparison_summary.json`; per-seed `results.json`, `neuron_indices.json`, `alpha_*.jsonl` |
| Detector | `models/mistral24b_classifier_canonical.pkl`; `data/mistral24b/pipeline/classifier_canonical_test_metrics.json` |
| Code | `scripts/run_intervention.py` (HNeuronScaler, faitheval decode); `scripts/run_negative_control.py` (sampling, triage); `scripts/intervene_model.py` (positive-weight selection); `scripts/infra/mistral24b_replication.sh` |
| Wrapper log | `logs/mistral24b_replication_20260430T112112Z.log` |
| Upstream context | [CP2/CP3 review](../reports/2026-04-29-mistral24b-cp23-pipeline-review.md); [strategy memo](../../../notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md); [research log 2026-04-30 entries](../../../notes/research-log.md) |

---

## 2. Independent recompute of headline numbers

Numbers below were recomputed from the raw JSONL via direct counting.
Discrepancies vs the canonical review: **none**.

| Quantity | Canonical review | Re-derived | Match |
|---|---|---|---|
| H compliance rates by alpha | 53.0, 49.5, 52.0, 52.0, 54.0, 54.0, 53.0 | 53.0, 49.5, 52.0, 52.0, 54.0, 54.0, 53.0 | ✓ |
| H least-squares slope (pp/alpha) | +0.79 | +0.786 | ✓ |
| Spearman ρ(alpha, H rate) | 0.5507 | 0.5507 (p ≈ 0.20) | ✓ |
| H paired endpoint α 0.0→3.0 (pp) | 0.0 | 0.0 (net 0/200) | ✓ |
| H paired no-op-to-max α 1.0→3.0 (pp) | +1.0 | +1.0 (net 2/200) | ✓ |
| H item flips α 0→3 (T→F, F→T) | 9, 9 | 9, 9 | ✓ |
| H item flips α 1→3 (T→F, F→T) | 5, 7 | 5, 7 | ✓ |
| Stayed true / stayed false (α 0→3) | 97 / 85 | 97 / 85 | ✓ |
| Parse failures, every H/control alpha | 0/200 | 0/200 across 7 H + 56 control alpha files | ✓ |

The Spearman p-value is informational, not a gate. With seven alpha points
and a non-monotonic curve, the slope itself does not reach conventional
significance even within the H run alone (reported H slope CI
`[-0.64, +2.21]`).

---

## 3. Pipeline integrity checks

### 3.1 Hash binding

`control_run_config.json` carries a `h_neuron_baseline` block whose
`run_contract.classifier.content_sha256`, `sample_manifest.content_sha256`,
`generation_fingerprint.id`, `prompt_style`, alpha schedule, model key, and
model path all match the H `intervention_run_config.json` byte for byte. The
control config additionally pins the H `results.20260430_112305.json` content
hash and path. CP5 cannot have run controls against a stale H baseline.

### 3.2 Manifest and classifier identity

| Field | Value |
|---|---|
| Sample manifest path | `data/manifests/faitheval_seed42_n200_mistral24b.lock.json` |
| Manifest fingerprint | `a912a5bb29e4ab65` |
| Manifest content SHA-256 | `0d5fd56122c2d0209e795d457ccb74d78686458382dedd95042064490d9bf84a` |
| Classifier path | `models/mistral24b_classifier_canonical.pkl` |
| Classifier content SHA-256 | `597435a84b19e68151f3f6903fd5ff1f12c61647440b07392dc34fdb8bea919d` |
| Generation fingerprint id | `204c1111e25e6866` |
| Prompt style | `standard` |
| Alpha grid | 0.0 0.5 1.0 1.5 2.0 2.5 3.0 |

Sample-ID order verified identical across all 7 H alpha files, all 56 control
alpha files, and the H↔control sets (first three IDs:
`Mercury_7175875`, `Mercury_SC_409171`, `Mercury_SC_408547`).

### 3.3 Decode policy

FaithEval generation in `run_intervention.py:2172` is `do_sample=False`,
`max_new_tokens=256`. Greedy decode is deterministic given the input, so any
output difference between alpha settings is a pure consequence of the
intervention's effect on argmax. Mean generated tokens are 9.13 (alpha=0.0)
vs 9.21 (alpha=3.0), with 0/200 records hitting the token cap.

### 3.4 Intervention semantics

`HNeuronScaler` (`scripts/run_intervention.py:645`) implements multiplicative
scaling on the down-projection inputs:
- α=1.0 short-circuits to identity (the per-sample hook timing confirms this:
  `hook_total_s ≈ 4.9e-03` at α=1.0 vs ≈0.42 at every other α).
- α=0.0 is full ablation (zero out the H-neuron columns).
- α∈(1, 3] amplifies; α∈[0, 1) dampens.

This is the canonical "α=1 is no-op, α=0 ablates, α>1 amplifies" convention
documented in the file header. A reader expecting the additive ITI/direction
convention (where α=0 is the no-op) will misread the dose-response shape;
this is worth flagging in any manuscript figure caption that shows the curve.

### 3.5 Control sampling

`scripts/run_negative_control.py:530-564` samples from the **zero-weight**
neuron pool — i.e., neurons the L1 classifier did not select as either H or
anti-H targets. This is the right null space for "are H neurons specifically
selected, vs. just any non-selected neuron of the same count?". With 1,310,700
zero-weight neurons available and 10 picks per seed, sampling is
near-independent.

Layer-matched controls preserve the H per-layer count distribution: H spans
layers `{15: 2, 16: 1, 17: 2, 18: 1, 19: 1, 20: 1, 36: 2}` for a total of 10.
Verified against `seed_0_layer_matched/neuron_indices.json` and the saved
classifier coefficients.

### 3.6 Detector identity and sparsity

The saved classifier has shape `coef_=(1, 1310720)` matching `40 ×
32768`. It contains 10 positive-weight and 10 negative-weight nonzero
coefficients (rest exactly zero). Positive coefficients span 0.219–6.44, with
the top two carrying 6.44 and 4.91 — the bottom-of-the-set neurons contribute
much less weight than the top. Test set: `accuracy 0.775 [0.715, 0.830]`,
`F1 0.7783 [0.7184, 0.8349]`, `AUROC 0.8711 [0.8185, 0.9172]` (10 000-resample
stratified bootstrap). The detector is genuinely above chance.

---

## 4. Item-level texture

### 4.1 The slope is non-monotonic

| α | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H rate (%) | 53.0 | **49.5** | 52.0 | 52.0 | 54.0 | 54.0 | 53.0 |
| Random mean (unconstrained, %) | 51.5 | 51.6 | 52.0 | 51.5 | 51.6 | 51.5 | 51.6 |
| Random mean (layer-matched, %) | 51.7 | 51.5 | 52.0 | 51.8 | 51.7 | 51.7 | 51.5 |

The H curve dips at α=0.5 and rises again at α=2.0/2.5 before settling. The
+0.79 pp/alpha least-squares slope is dominated by the recovery from this
dip, not by an endpoint difference. A monotonicity check would not classify
this as dose-response.

### 4.2 The α=0.5 dip is asymmetric

H paired flips α=0.0 → α=0.5: **8 T→F, 1 F→T** (net −3.5 pp). This single
alpha point drives the apparent variance more than any other transition.
Whether this is an artifact of partial-ablation interacting with one or two
high-coefficient neurons or pure noise cannot be answered from this run; with
n=200 and 9-token mean generations, single-token argmax shifts on a handful
of items are within measurement noise. It is unsafe to interpret it.

### 4.3 The H–control offset is constant

| α | H rate | Control empirical 95% interval | H − control midpoint |
|---|---:|---|---:|
| 0.0 | 53.0 | [51.5, 51.5] | +1.5 pp |
| 3.0 | 53.0 | [51.5, 51.95] | +1.275 pp |

H is offset above the random-control band at *both* endpoints by a similar
amount. This is the canonical review's "the endpoint offset is not an
alpha-induced effect" point, recovered directly from
`comparison_to_h_neurons` fields. The offset most plausibly reflects that the
10 selected neurons happen to bias model outputs in a sample-dependent way at
all alpha settings — including the no-op proxy α=0 (where H neurons are fully
zeroed) — rather than that α modulates compliance.

### 4.4 Control flips are essentially zero

| Seed | Strategy | Flips T→F | Flips F→T | Slope (pp/α) |
|---|---|---:|---:|---:|
| 0 | unconstrained | 0 | 0 | −0.036 |
| 1 | unconstrained | 0 | 0 | −0.107 |
| 2 | unconstrained | 0 | 0 | −0.036 |
| 3 | unconstrained | 0 | 0 | 0.000 |
| 4 | unconstrained | 0 | 1 | +0.071 |
| 0 | layer_matched | 1 | 0 | −0.036 |
| 1 | layer_matched | 0 | 0 | −0.036 |
| 2 | layer_matched | 0 | 0 | −0.036 |

Eight control runs flip a combined **2 of 1 600** alpha-0 vs alpha-3 paired
items. The H run flips **18 of 200**. The 9× excess of H flips over the worst
control is real; the directional balance (9/9) is the reason it does not
become an effect.

---

## 5. Stress-tests against alternative readings

I tried to construct a reading under which CP5 supports a Mistral H-neuron
intervention claim. None survives.

| Alternative reading | Why it fails |
|---|---|
| "The slope is positive and outside controls — that *is* specificity." | The slope-vs-controls comparison is asymmetric: H has measurable variance because alpha actually does something, controls have ~zero variance because random neurons don't perturb argmax. The control band is therefore a "no-perturbation" band, not a "what slope would noise look like under this much perturbation" band. The proper null is the H paired-endpoint bootstrap CI itself, which includes 0. |
| "α=2.0 and α=2.5 both reach 54% — that's a real ceiling effect." | 54% is +1 pp above the 53% endpoints and +2 pp above the 52% no-op. Each pp on n=200 is two items. The two items in question are not stable across α (compare the α=2.0 and α=2.5 jsonl pairs); the apparent plateau is "any two items happen to have flipped". Wilson 95% CI on 108/200 is [0.471, 0.608] — 13.7 pp wide. |
| "Drop α=0.5 as an outlier and the trend is monotonic." | Removing observations to recover monotonicity is post-hoc selection; under the same rule, dropping α=2.0 also removes the rise. The protocol's claim gate is paired endpoint, not alpha-aligned trend. |
| "H neurons clearly perturb more than random, so the readout transfers." | Magnitude transfer ≠ direction transfer. The L1 paper's H-neuron claim is a directional dose-response, not a magnitude separation. CP5 supports magnitude; it does not support direction. |
| "The 1.5 pp offset above controls at α=0 is the H-neuron baseline effect." | Then α=3 should show a *different* offset if alpha modulates compliance through H neurons. It doesn't — α=0 and α=3 show the same offset. A constant offset is not an intervention effect; it is a sample-level idiosyncrasy of the 10 chosen neurons that survives ablation. |

---

## 6. Triage-script gap

`scripts/run_negative_control.py:_classify_triage` (around line 1300) returns
`review_baseline_mismatch` only when α=0 H lies *outside* the control
percentile interval **and** α=3 H lies *inside* it. CP5 has both endpoints
outside the control interval at the same offset — so the script returns
`specificity_supported`. The canonical review correctly rejects that result
manually, but a future operator who reads only `comparison_summary.json:triage`
would be misled.

Recommended additional triage state — *constant baseline offset, no
dose-response*:

```text
if alpha_0_outside and alpha_3_outside and
   abs(alpha_0_h - alpha_3_h) < eps and
   sign(alpha_0_h - alpha_0_band_mid) == sign(alpha_3_h - alpha_3_band_mid):
   return ("review_constant_offset", ...)
```

`eps` should be small (e.g. 1.0 pp; equivalent to 2 paired items on n=200).
This is a code follow-up, not a CP5 rerun trigger. Filed as a candidate item
under Section 9.

---

## 7. What withstands scrutiny

- **The pipeline.** Every contract check, hash binding, manifest validation,
  parse-failure check, and decode-policy check is clean. CP5 is not a
  measurement failure.
- **The detector readout.** Test AUROC `0.871 [0.818, 0.917]` is well above
  chance on disjoint splits with a tokenizer-aware Mistral pipeline. The 10
  positive-weight H-neurons are real classifier signal, not coincidence.
- **The null verdict on intervention.** Paired endpoint α=0→3 is `0.0 pp`
  with a `±4 pp` CI. No reasonable manipulation of the existing data
  recovers a directional dose-response.
- **The hash-bound H↔control comparison.** Controls share every meaningful
  knob with the H run; the only difference is *which 10 neurons* are
  intervened on. This is the right experimental design.
- **The decision to refuse `specificity_supported`.** The canonical review's
  manual rejection is methodologically correct and aligns with the L1-paper
  acceptance rule (paired endpoint must move).

## 8. Uncertainty register

| Item | Confidence | Why |
|---|---|---|
| H-neuron multiplicative scaling does not produce a directional FaithEval dose-response on Mistral 2501 at the canonical α grid | **High** | Paired endpoint within `±4 pp`; symmetric flips; non-monotonic curve; clean controls. |
| The Mistral classifier is a real hallucination-correlated readout | **High** | AUROC 0.871 on disjoint splits, n=200 balanced test, Wilson CI [0.815, 0.917]. |
| A different intervention basis (additive direction, ITI head, signed amplification of negative-weight neurons) could recover an effect | **Low–Medium** | Plausible but speculative. CP5 only tests the canonical positive-weight, multiplicative-scaling rule. |
| The result generalizes to other benchmarks (FalseQA, BioASQ) | **Unknown** | Not run on Mistral; the wrapper has those stages but they were not in CP5 scope. |
| A small (≤2 pp) directional dose-response is hidden under noise | **Cannot rule out** | n=200 + greedy decode + 13–14 pp Wilson width per cell. Power-limited. |
| The Mistral 2501 checkpoint is fundamentally less amenable to neuron-level steering than the Gemma anchor | **Medium-low confidence as a *strong* claim** | One model, one benchmark, one intervention rule. The gap with Gemma is consistent with that hypothesis but does not prove it. |
| The CP5 outcome would change with more seeds / larger n / sampled decode | **Low** for direction; **medium** for slope variance | Greedy decode means seed has no effect; only neuron-set seed matters, and the 8 control seeds already span the available zero-weight pool densely enough. n could matter, but only at scales (≥640) the CP5 budget did not authorize. |

## 9. Most valuable next steps

Ordered by ratio of information gained per unit of GPU/API cost.

1. **Patch the triage script** to emit `review_constant_offset` when both
   endpoints sit on the same side of the control band by a similar amount.
   Non-spend; testable; closes the script-vs-manual-gate divergence. Prevents
   future CP5-shaped runs from being silently green-lit on slope alone.
2. **Treat CP5 as a published null in manuscript framing.** The
   detector-vs-causal dissociation (AUROC 0.871 vs paired endpoint 0.00 pp on
   the same checkpoint and prompt) is exactly the kind of result the L1
   limitation framing benefits from. Do not attempt to rescue it.
3. **Update the strategy memo's CP5 entry** with a pointer to this audit
   alongside the canonical review (handled below).
4. **If a Mistral intervention follow-up is justified later, it is a new
   pre-registered branch, not a CP5 rerun.** Two candidate experiments worth
   pre-registering:
   - **(a) Directional intervention in residual stream** instead of
     multiplicative neuron scaling. The Mistral classifier likely has a
     usable direction in the L15–L20 middle band; an additive ITI-shaped
     intervention might surface effects multiplicative scaling cannot.
   - **(b) Combined positive + negative-weight neuron rule.** The current
     rule discards 10 anti-H neurons. A signed-amplification rule (boost
     negatives, ablate positives) is closer to how the classifier itself
     uses the activations and is a non-trivial alternative null.
   Both are exploratory; neither earns claim-bearing status without its own
   pre-registered control gate.
5. **Do not launch Mistral CP6/CP7 (TruthfulQA MC ITI / TriviaQA bridge) on
   the assumption that CP5 cleared anything beyond plumbing.** The CP5 null
   does not block CP6/CP7 in principle, but they should be reframed as
   independent transfer audits, not as the next step after a successful
   intervention gate. The
   [strategy memo's CP6 row](../../../notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md)
   already reflects this; keep it that way.
6. **Do not start Mistral SAE work.** This is already the strategy memo's
   stance. CP5 reinforces it: an SAE comparison is uninformative if the
   neuron-level intervention is null on the same surface.
7. **(Optional, low priority)** Audit whether the layer-matched control set
   should expand to 5 seeds for symmetry with unconstrained. Current 3 vs 5
   asymmetry is harmless given the near-zero variance, but a future reviewer
   may ask.

## 10. Cross-links

- Headline numbers and verdict: [CP5 review](../reports/2026-04-30-mistral24b-cp5-faitheval-review.md).
- Upstream detector gate: [CP2/CP3 review](../reports/2026-04-29-mistral24b-cp23-pipeline-review.md).
- Mistral progress and decisions ledger: [strategy memo](../../../notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md).
- Chronology and surprises: [research log 2026-04-30 entries](../../../notes/research-log.md).
- Pipeline code: `scripts/run_intervention.py`, `scripts/run_negative_control.py`, `scripts/intervene_model.py`, `scripts/infra/mistral24b_replication.sh`.

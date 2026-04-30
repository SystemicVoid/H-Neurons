# Why CP5 went null on Mistral 2501: comparison to Gemma replication, the H-Neurons paper, and the 2501-vs-2503 question

**Date:** 2026-04-30
**Reviewer stance:** hypothesis formulation, ranked. Reads three sources in parallel — the
[CP5 audit](2026-04-30-mistral24b-cp5-pipeline-audit.md) and
[CP5 review](../reports/2026-04-30-mistral24b-cp5-faitheval-review.md);
our Gemma-3-4B FaithEval H-neuron baseline at
`data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`;
and [Gao et al. 2026 (H-Neurons)](../../../papers/h-neurons-hallucination-correlated.md) — and asks
why our Mistral 2501 result diverged from both. Output is a ranked set of
falsifiable hypotheses, not a re-derivation of numbers.

This file does not introduce new claim-bearing numbers. Numbers cited below
were already frozen in the [CP5 review](../reports/2026-04-30-mistral24b-cp5-faitheval-review.md),
[CP2/CP3 review](../reports/2026-04-29-mistral24b-cp23-pipeline-review.md),
or extracted directly from the cited Gemma comparison summary.

**Current status, 2026-04-30 later:** H1 has now been run. The H1 run
decision and outcome are superseded by
[the intervention-aware C-sweep review](../reports/2026-04-30-mistral24b-h1-c-sweep-review.md).
The coupled selector chose `C=0.75` with 9 positive H-neurons, but the
FaithEval follow-up stayed flat. This file remains useful as the historical
hypothesis-formulation memo for CP5; it is no longer a live recommendation to
spend GPU on H1.

## 1. Three numbers that frame the question

|   | Our Mistral 2501 (CP5) | Our Gemma-3-4B replication | H-Neurons paper expectation |
|---|---:|---:|---:|
| Detector test AUROC | 0.871 [0.818, 0.917] | not in this comparison sheet, but classifier validated as canonical | TriviaQA detection ≈ 81% accuracy on Mistral-Small-3.1-24B (Table 1) |
| Selected H-neuron count (positive weights) | 10 / 1,310,720 | 38 / 348,160 | ~13 / 1,310,720 implied by 0.01‰ Table-1 ratio for Mistral-Small-3.1-24B |
| Selected H-neuron ratio (‰) | 0.0076 | 0.109 | 0.01 (Mistral-3.1-24B) / 0.10 (Gemma-3-4B) |
| FaithEval H rates by α (0.0 → 3.0) | 53.0, 49.5, 52.0, 52.0, 54.0, 54.0, 53.0 | 64.2, 65.4, 66.0, 67.0, 68.2, 69.5, 70.5 | monotonic ascent across all six models tested in Fig 3 |
| FaithEval H slope (pp/α) | +0.79 | +2.09 | "average slope ≈ 2.40" across larger models on four benchmarks; ≈3.03 for smaller models |
| Spearman ρ(α, H rate) | 0.5507 | 1.0 | acknowledged non-strict-monotonicity in some cases (paper Fig 3 caption) |
| Paired endpoint α 0.0 → 3.0 (pp) | 0.0 [-4.01, +4.00] | +6.3 (70.5 − 64.2; not bootstrapped in this summary) | not stated as an endpoint metric in the paper |
| Direction of α=3 H–control gap (pp) | +1.5 (53.0 vs 51.5 control mean) | +4.4 (70.5 vs 66.1 control mean) | per-model magnitudes not tabulated, but plotted as clear separation |
| Item flips α 0→3 (T→F, F→T) | 9, 9 (balanced) | not in this comparison sheet | qualitative monotone curves implied directional, not balanced |

Two structural facts:

- **Our Gemma replication is canonical-shape**: monotonic, ρ=1.0, slope +2.09
  pp/α, separation from controls at α=3, neuron count and ratio at the paper's
  expected scale (38 ≈ 0.10‰ vs paper's 0.10‰ for Gemma-3-4B). It earns the
  dose-response claim under the L1-paper acceptance rule.
- **Our Mistral 2501 result is not just smaller — it has a different shape**:
  non-monotonic, balanced flips, constant H–control offset across endpoints,
  fewer selected neurons than the paper's stated Mistral-Small-3.1-24B
  scale (10 vs 13).

The shape difference is the key signal. A simply-attenuated dose-response
would still be monotonic with positive endpoints. CP5 is qualitatively
different.

## 2. Two protocol divergences that are real, not interpretive

These are direct, code-level mismatches against
[H-Neurons §6.1.3](../../../papers/h-neurons-hallucination-correlated.md). They
are not subjective.

### 2.1 Intervention-aware C selection (the paper does it; we don't)

Paper, §6.1.3 ("Balancing Detection Recall and Functional Safety"):

> we perform a grid search to select C to maximize the sum of (1) classification
> accuracy on a held-out set and (2) **model performance on TriviaQA when
> suppressing the identified H-Neurons**.

Our `scripts/classifier.py:111` exposes `--selection_metric` with default
`auroc`, choices restricted to `accuracy/precision/recall/f1/auroc` —
classification metrics only. The CP2/CP3 review confirms `C=1.0` was selected
on dev AUROC alone. There is no TriviaQA-suppression step in our C
grid-search.

This means the paper's C is selected to filter for sparse subsets that
**both predict and intervene**. Ours is selected to filter for sparse
subsets that **predict only**. A neuron that fires reliably on hallucinated
generations need not be on the causal route from internal representation to
emitted-token argmax under multiplicative scaling. The paper's coupled
criterion was specifically designed to break that decoupling.

This is the strongest single divergence and the most parsimonious
explanation for "high AUROC, null intervention" on Mistral.

### 2.2 Selected-neuron count differs from the paper's Mistral 24B scale

Paper Table 1: Mistral-Small-3.1-24B has H-neuron ratio `0.01‰`. With
Mistral-Small architecture (1,310,720 FFN neurons), this implies ~13
positive-weight neurons.

Ours: 10 positive-weight neurons, ratio 0.0076‰.

This is not a huge gap by itself — three neurons of difference, and the
paper's number is rounded — but combined with §2.1 it suggests we landed at
a higher-sparsity (more aggressive) C than the paper would have selected
with intervention-aware grid-search. The 10th positive Mistral coefficient
is `0.219`, a long way from the top weight `6.44`. Adding the 11th–13th
neurons would barely change AUROC but could materially change collective
intervention magnitude.

### 2.3 What does match

- Activation feature: CETT, exactly as paper §6.1.2 (verified in
  `scripts/extract_activations.py:98-150`).
- Intervention rule: `z_{j,t} ← α · z_{j,t}` with α=1 as no-op identity, α
  ranging over `[0, 3]`. Multiplicative pre-down-projection scaling.
  (`scripts/run_intervention.py:645-687`.)
- FaithEval decode: greedy, `do_sample=False`, `max_new_tokens=256`. Matches
  paper §6.2.2.
- Prompt style `standard`, parser, compliance metric: counterfactual-key
  match for FaithEval Counterfactual Context. Matches paper.
- Detector training: L1 logistic regression on CETT features, positive
  weights only as intervention targets. Matches paper.

So protocol divergence is concentrated at C-selection criterion (§2.1) and
its consequence on neuron count (§2.2). The rest of the pipeline replicates
the paper's recipe.

## 3. The 2501-vs-2503 question

The paper studied **Mistral-Small-3.1-24B** (a 2503-class checkpoint with
multimodal extension). We ran **Mistral-Small-24B-Instruct-2501**.

Three things are true at once and worth disentangling.

1. **Same base family, different post-training.** Both are "Mistral-Small 24B
   3.x" architecture; the 2503 release is positioned as Mistral 3.1 with
   added vision support and additional post-training. The base model under
   each instruct release is the same Mistral-Small-3 24B base. The paper's
   transferability finding (§4) explicitly tests "do classifiers trained on
   instruct models retain predictive ability on the base model?" and answers
   yes for Mistral, with `avg ≈ 0.97` rank stability — **but** stability is
   measured on **detection**, not on **intervention efficacy**. A 3% drift
   in down-projection weights of a single load-bearing H-neuron can
   meaningfully change the strength of that neuron's downstream causal
   effect on emitted tokens, even if its activation pattern still
   correlates with hallucination. Detection is rank-invariant; intervention
   is not.

2. **2503 added text-side post-training, not just vision.** Mistral's release
   notes for 3.1 describe improved instruction following, broader benchmark
   coverage, and updated safety/refusal behavior. These are exactly the
   surfaces on which "over-compliance with misleading context" would shift.
   It is plausible that 2503's additional alignment specifically *amplified*
   the route from H-neuron activation to compliance behavior on FaithEval —
   in the same way that more aligned models often display sharper
   sycophancy/compliance failure modes when probed at scale.

3. **Our pipeline can run on 2501 but not 2503.** The strategy memo records
   that "current causal-LM path intentionally unsupported for 2503" because
   2503 ships through the multimodal `mistral3` processor path, not the
   text-only `AutoModelForCausalLM` path. So 2501 was chosen as a
   same-family **anchor**, not as an exact paper checkpoint. This is a
   known scope choice.

The question "could it be the 2501 instead of 2503?" therefore splits into
two distinguishable sub-questions:

- **(3a)** Are the H-neurons themselves different in 2501 vs 2503? Likely
  partially: same pre-training-era candidates (per §4 transferability), but
  different specific selected sparse subsets after L1 regularization on
  different post-trained activation distributions. We cannot test 2503
  directly without pipeline migration.
- **(3b)** Even with the same H-neurons, does the route from H-neuron
  activation to FaithEval compliance differ between 2501 and 2503? Likely
  yes — 2503's extra alignment is a credible cause of stronger or weaker
  causal coupling between H-neuron activity and emitted-answer choice. Also
  not directly testable without 2503 pipeline migration.

The 2501-vs-2503 hypothesis is plausible **and** undertestable from where
we stand. It cannot be ruled out and cannot cheaply be confirmed.

## 4. Ranked hypotheses

Ranking criterion: **explanatory power × supporting evidence × testability**.
Each entry is structured as: what it claims, why we suspect it, what it
predicts, how to test it.

### Tier A — Strongest, code-level divergences from paper protocol

**H1. Intervention-aware C selection (paper does it; we don't).**
- *Claim:* The paper grid-searches C to maximize `(detection accuracy) +
  (TriviaQA suppression effect)`. We grid-search C on AUROC alone. Our 10
  neurons are AUROC-best; the paper's are AUROC-and-suppression-best. Same
  pipeline, different sparse subset, different intervention efficacy.
- *Evidence:* Direct code mismatch (`scripts/classifier.py:113`,
  `--selection_metric` default `auroc`). Paper §6.1.3 explicit. Detector is
  strong (AUROC 0.871) yet intervention is null — exactly the failure mode
  the coupled criterion was designed to prevent.
- *Status after test:* Confirmed for model selection, rejected for FaithEval
  rescue. The follow-up selected a different C (`0.75`) but selected fewer,
  not more, positive H-neurons (9). FaithEval alpha `0.0 -> 3.0` was only
  `+0.5 pp`, with paired bootstrap CI `[-3.0, +4.0]`. H1 is no longer a
  sufficient explanation for the CP5 null.
- *Testability:* Completed. See the
  [H1 C-sweep review](../reports/2026-04-30-mistral24b-h1-c-sweep-review.md)
  for the authoritative data and uncertainty analysis.

**H2. L1 sparsity is too aggressive at our chosen C; tail of selected weights
is functionally inert.**
- *Claim:* Mistral 24B's L1 ridge is shallow at the bottom — neurons
  ranked 6th–10th carry small classifier weights (down to 0.219 vs top
  6.44). Multiplicatively scaling neurons whose CETT contribution is
  near-zero perturbs the model only on 2–3 actual top neurons. That
  perturbation is too small to dose-respond directionally.
- *Evidence:* Confirmed weight distribution from
  `models/mistral24b_classifier_canonical.pkl`. Gemma's bottom weights are
  even smaller in absolute terms (down to 3e-4) but Gemma has 38 neurons
  total — so even a long tail accumulates more middle-weight contribution.
  Mistral's 10 neurons cluster on a handful of strong neurons + tail.
- *Predicts:* (a) Removing the bottom-5 Mistral H-neurons should *not*
  measurably change the dose-response. (b) Adding random near-threshold
  positive-weight neurons (or relaxing C to surface them) would change
  intervention magnitude.
- *Testability:* High. (a) is purely a re-intervention with a subset; no
  retraining. (b) requires re-training classifier at lower C.

This is a sister hypothesis to **H1**: both predict that the
selected-neuron set is wrong for intervention even though it is right for
detection. **H1** identifies the protocol cause; **H2** identifies the
mechanical consequence on Mistral specifically.

### Tier B — Plausible, partly testable

**H3. 2501-vs-2503 post-training drift changes intervention efficacy without
changing detection.**
- *Claim:* The paper's H-neuron transferability is rank-based (AUROC ~0.97
  parameter stability). 3% weight drift at a load-bearing down-projection
  can flip whether a neuron's amplification cleanly biases output-token
  argmax. Detection-stable ≠ intervention-stable. 2501 differs from 2503
  on exactly the post-training surfaces (instruction following, refusal,
  safety) most likely to wire into FaithEval-shaped compliance.
- *Evidence:* Indirect. Our detector AUROC 0.871 ≈ paper's 81% on
  Mistral-3.1; consistent with similar pretraining-era H-neuron candidates.
  Yet intervention dose-response shape is qualitatively different. Paper
  §4 itself notes Mistral has the highest rank stability (≈0.97) — i.e.,
  small but nonzero drift.
- *Predicts:* Running CP5 on 2503 (with multimodal pipeline migrated)
  would surface a directional dose-response closer to the paper's
  expectation, even with our current C-selection criterion.
- *Testability:* Low–medium. Requires pipeline migration to the
  `mistral3` multimodal path, gated 2503 model access, and a fresh round
  of CP1–CP5. Outside the current spend envelope. Cannot be excluded by
  available evidence.

**H4. Mistral 2501 is post-trained for stronger context-faithfulness, which
locally suppresses the H-neuron → compliance route.**
- *Claim:* Sub-hypothesis of **H3** with a specific direction. If Mistral's
  2501 alignment specifically targets faithfulness (resisting misleading
  context), the model's emitted-token decision is less sensitive to
  H-neuron activation level on FaithEval prompts, even if H-neurons still
  fire correlatively.
- *Evidence:* Our FaithEval baseline is 52% (no-op α=1.0) — close to
  chance, suggesting the model is genuinely uncommitted between
  context-following and prior-recall on these items. Gemma-3-4B's no-op
  baseline is 66% — already context-leaning, with more "give" for
  H-neuron amplification to push further. A flatter route from H to
  output is consistent with this.
- *Predicts:* (a) Mistral 2501 should show *some* effect on benchmarks
  where it has more compliance "give" — e.g., FalseQA, Sycophancy,
  Jailbreak. (b) Effect magnitude should correlate with no-op compliance
  level (room-to-move).
- *Testability:* Medium. (a) Run Mistral 2501 on FalseQA/BioASQ; the
  wrapper has those stages. Each is roughly the same GPU cost as one
  CP5 alpha. (b) Multi-benchmark comparison from the same intervention
  set.

### Tier C — Constraints, not full explanations

**H5. Sample size and Wilson width compress real but small effects into
noise.**
- *Claim:* n=200 with greedy decode gives Wilson 95% CIs of about ±7 pp on
  any single rate; the paper used the full FaithEval Counterfactual
  Context subset (much larger). A real 1.5–2 pp/α effect would show as
  +0.79 with wide CIs in our run, exactly what we got.
- *Evidence:* Direct CI math, confirmed in CP5 review. Paper rates not
  tabulated for Mistral-3.1 specifically.
- *Predicts:* If we had n≥640 paired samples, the paired-endpoint CI
  would shrink enough to either confirm or reject a 4-pp effect.
- *Testability:* Medium-low. Requires fresh n=600+ FaithEval lock,
  reactivating the pipeline. Doesn't explain non-monotonicity or
  direction-balanced flips on its own — it explains magnitude only.

**H6. Prompt template behavior differs between Mistral 2501 and Mistral
3.1.**
- *Claim:* 2501 uses Mistral 3-style instruction templating with
  `fix_mistral_regex=True` for tokenizer correction; 3.1 uses the
  3.1-specific template. Different prompt rendering changes where the
  model lands on its compliance manifold *before* any intervention,
  potentially absorbing the perturbation into other circuits.
- *Evidence:* Tokenizer kwargs registered in our model registry; no
  direct comparison of prompt rendering between 2501 and 3.1 available
  to us. Detector is strong on this rendering — so representation isn't
  blocked.
- *Predicts:* Running CP5 with an alternative prompt style (e.g.
  `anti_compliance` rendering) would shift the no-op baseline and
  possibly the dose-response shape.
- *Testability:* Medium. The Mistral wrapper is currently locked to
  `prompt_style=standard`; an alternative-style audit run is allowed
  per the strategy memo as a measurement-stability follow-up rather
  than a claim run.

### Tier D — Unlikely or unfalsifiable to address from current state

**H7. The paper's reported Mistral-3.1 effect is itself smaller than the
"average slope ≈ 2.4" suggests.**
- *Claim:* Figure 3 plots curves; the per-benchmark per-model numerical
  slope for Mistral-Small-3.1-24B on FaithEval is not tabulated in the
  paper text. The "≈ 2.4" average is across four benchmarks and three
  larger models. Mistral-FaithEval alone could be smaller.
- *Evidence:* None to confirm or reject. We do not have the paper's
  per-cell numbers.
- *Predicts:* If the paper's Mistral-3.1 FaithEval slope is closer to
  +1.0–1.5 pp/α, the gap with our +0.79 narrows materially.
- *Testability:* Could attempt to extract the figure values, or contact
  authors. Useful for scope-narrowing manuscript framing rather than
  for diagnosing our null.

**H8. Greedy decode, parser, or compliance-metric difference.**
- *Claim:* We diverge from paper on decode/parser/metric.
- *Evidence:* No divergence found (verified §2.3). Greedy decode,
  256-token cap, counterfactual-key match parser all match paper §6.2.2.
- *Predicts:* No predictions.
- *Testability:* N/A. Already verified to match.

## 5. H1 follow-up status

The previously recommended single follow-up has now run. Authority:
[2026-04-30 Mistral 24B H1 C-sweep Review](../reports/2026-04-30-mistral24b-h1-c-sweep-review.md).

The outcome is useful but negative for the rescue hypothesis:

- The coupled score did change model selection: `C=0.75`, 9 positive
  H-neurons, score `1.395`.
- The exact C winner is not robust against the local C plateau, but the
  sweep strongly rejects broad high-C candidates because their TriviaQA
  suppression accuracy collapses.
- The FaithEval follow-up was essentially flat: rates
  `51.5, 52.0, 51.5, 51.5, 52.0, 52.0, 52.0`; alpha `0.0 -> 3.0` was
  `+0.5 pp` with CI `[-3.0, +4.0]`.

Therefore do not spend further GPU on 2501 local C-grid tweaks unless there is
a new pre-registered question. After H1, **H3** (2501-vs-2503) and
benchmark/operator specificity remain plausible, but they are different
branches; they do not rescue CP5 as a Mistral H-neuron intervention
replication.

## 6. Cross-links and provenance

- Numerical authority for CP5: [CP5 review](../reports/2026-04-30-mistral24b-cp5-faitheval-review.md).
- H1 follow-up authority: [H1 C-sweep review](../reports/2026-04-30-mistral24b-h1-c-sweep-review.md).
- CP5 plumbing/textural audit: [CP5 audit](2026-04-30-mistral24b-cp5-pipeline-audit.md).
- Mistral progress and decisions: [strategy memo](../../../notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md).
- Detector gate context: [CP2/CP3 review](../reports/2026-04-29-mistral24b-cp23-pipeline-review.md).
- H-Neurons paper: [`papers/h-neurons-hallucination-correlated.md`](../../../papers/h-neurons-hallucination-correlated.md).
- Our Gemma baseline used in §1: `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`.

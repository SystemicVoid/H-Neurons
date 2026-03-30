# D4 — Truthfulness Direction Survives Kill-Shot

**Date:** 2026-03-30
**Model:** Gemma-3-4B-IT
**Status:** D4 survives the all-layer FaithEval kill-shot, but is not sprint-complete.

## Question

Does the residual-stream truthfulness direction still look dead once the two remaining excuses are removed:

1. use the corrected clean 2,781-record contrastive dataset, and
2. test all-layer ablation instead of layer 32 only?

## Canonical Artifacts

- Clean final FaithEval run: `data/gemma3_4b/intervention/faitheval_direction_ablate_d4_all_layers_calibrated_clean/experiment/`
- Calibration run: `data/gemma3_4b/intervention/faitheval_direction_ablate_d4_all_layers_calibration_clean/experiment/`
- Direction snapshot used for the clean run: `data/contrastive/truthfulness/d4_killshot_snapshot/truthfulness_directions.pt`
- Clean full-run provenance: `data/gemma3_4b/intervention/faitheval_direction_ablate_d4_all_layers_calibrated_clean/experiment/run_intervention.provenance.20260330_131952.json`

The decision-relevant full run was executed from a clean worktree with `git_dirty: false` at commit `1e8ff95817f396a723cc3fc45d20fcf4ce39e3d1`.

## Clean Setup

- Dataset gate was checked against committed `metadata.json`, not remembered counts.
- The tracked contrastive JSONL matched the rebuilt 2,781-record dataset.
- A clean extraction rerun again selected layer 32 as the best single layer, with validation accuracy `0.7068` and low refusal overlap (`|cos_sim|=0.0437`).
- The final kill-shot used **all-layer** ablation, not layer 32 only.

## Calibration

20-sample all-layer calibration on FaithEval anti-compliance:

| β | Compliance | Parse fails | Mean response length | Read |
|---|---:|---:|---:|---|
| 0.005 | 18/20 = 90.0% | 0 | 1.0 | clean |
| 0.01 | 18/20 = 90.0% | 0 | 1.1 | clean |
| 0.02 | 8/20 = 40.0% | 0 | 2.5 | first visible corruption / answer-format drift |
| 0.05 | 0/20 = 0.0% | 18 | 302.4 | catastrophic |

Decision from calibration:

- `β_clean = 0.01`
- `β_onset = 0.02`
- Early-stop rule did **not** fire, because a clean non-zero β existed.

## Full Run

1,000-sample all-layer FaithEval rerun, seed 42, anti-compliance prompt:

| β | Compliance | 95% CI | Parse fails | Mean generated chars |
|---|---:|---|---:|---:|
| 0.00 | 660/1000 = 66.0% | [63.0, 68.9] | 0 | 1.00 |
| 0.01 | 714/1000 = 71.4% | [68.5, 74.1] | 0 | 1.31 |
| 0.02 | 462/1000 = 46.2% | [43.1, 49.3] | 6 | 15.42 |

Paired transitions:

| Transition | False→True | True→False | Net |
|---|---:|---:|---:|
| β=0.00 → β=0.01 | 91 | 37 | +54 |
| β=0.01 → β=0.02 | 63 | 315 | -252 |

## Why β=0.01 Counts As Clean

This is not the layer-32 null result in disguise.

- Parse failures stay at `0/1000`.
- Responses remain overwhelmingly in the short MCQ regime.
- Bare-letter outputs remain dominant: `981` at β=0.00, `937` at β=0.01.
- There is only light format drift at β=0.01, not the collapse seen at β=0.02.
- The gain is broad rather than a single-option artifact:
  - correct-A: `60.8% → 65.0%`
  - correct-B: `70.5% → 70.9%`
  - correct-C: `64.0% → 75.0%`
  - correct-D: `68.9% → 75.0%`

By contrast, β=0.02 is the onset cliff:

- mean response length jumps from `1.31` to `15.42`
- `D` becomes massively over-selected (`272 → 534`)
- the output surface shifts into `D)`, `D!`, `**B)**`, `**C)**`-style corruption
- parse failures begin (`6/1000`)

## Interpretation

The single-layer excuse is gone. All-layer ablation does produce a usable point, but only a **very narrow** one.

The shape is now:

- small clean lift at `β=0.01`
- immediate collapse by `β=0.02`

That is enough to keep residual-stream D4 alive as a comparator family, but not enough to call it a robust steering baseline. It behaves like D3 in one important sense: there is a narrow surviving band and then a cliff. It does **not** yet behave like the smoother H-neuron curve.

## Decision

**D4 survives the kill-shot.**

What this means:

- Do **not** kill residual-stream truthfulness steering for the sprint.
- Do **not** call D4 finished.
- Treat `β=0.01` as the only currently usable all-layer operating point.
- Treat `β=0.02` as the corruption onset for this direction family on FaithEval.

What this does **not** mean:

- no claim of broad robustness
- no claim that residual-stream truthfulness steering beats H-neuron scaling
- no justification yet to broaden to other benchmarks without a scoped reason

## Next Implication

The kill decision is off the table, but the branch still has a narrow-window problem. The next meaningful discriminator is no longer "does all-layer help at all?" That question is answered. The next discriminator is whether this narrow `β=0.01` point survives broader scrutiny better than the existing direction baselines, or whether head-level / causal methods are still the more promising path.

# Localization Summary

This file is a reviewer-facing derivative of the paper’s localization anchor. It condenses the canonical FaithEval readout-quality and control evidence without requiring navigation through the full repo.

## Canonical Repo Origins Used For This Summary

- `data/gemma3_4b/pipeline/classifier_disjoint_summary.json`
- `data/gemma3_4b/pipeline/classifier_structure_summary.json`
- `data/gemma3_4b/pipeline/classifier_sae_summary.json`
- `data/gemma3_4b/intervention/faitheval/experiment/results.json`
- `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`
- `data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json`
- `data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json`
- `data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json`
- `notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md`

## §3.1 Readout Quality

| Claim | Value | CI / exact test |
|---|---|---|
| H-neuron AUROC | 0.843 | [0.815, 0.870] |
| H-neuron disjoint accuracy | 76.5% | [73.6, 79.5] |
| H-neuron detector test size | 780 | — |
| H-neuron count | 38 / 348,160 | — |
| SAE feature AUROC | 0.848 | [0.820, 0.874] |
| SAE feature disjoint accuracy | 77.2% | [74.3, 80.2] |
| SAE detector test size | 782 | — |
| SAE positive features selected | 266 across 10 layers | — |

Interpretation: the paper’s anchor comparison is matched on held-out readout quality. H-neurons and SAE features have effectively comparable detector performance on the FaithEval disjoint split.

## §3.2 Control Comparison On FaithEval

| Claim | Value | CI / exact test |
|---|---|---|
| H-neuron slope | +2.09 pp/α | [1.38, 2.83] |
| H-neuron monotonicity | Spearman ρ = 1.0 | — |
| H-neuron no-op to max (`α = 1.0 → 3.0`) | +4.5 pp | [2.9, 6.1] |
| H-neuron full sweep (`α = 0.0 → 3.0`) | +6.3 pp | [4.2, 8.5] |
| Random-neuron null summary (five unconstrained seeds) | mean slope +0.02 pp/α | descriptive mean |
| Random-neuron null summary (all eight seeds) | max slope +0.21 pp/α | descriptive max |
| Random-neuron specificity | 8/8 paired neuron-minus-random differences positive | every seed-specific 95% CI excludes zero |
| SAE H-feature slope | +0.16 pp/α | [-0.51, 0.84] |
| SAE monotonicity | Spearman ρ = 0.18 | — |
| SAE random-feature slope | +0.59 pp/α | descriptive seed mean |
| Neuron-minus-SAE slope difference | +1.93 pp/α | [+0.94, +2.92] |
| Directional permutation test | one-sided `p < 0.001` | saved value `9.9998e-05` |
| FaithEval evaluation size | 1,000 items | — |

Interpretation: detector quality is comparable across the neuron and SAE families, but only the neuron intervention produces a robust behavioral dose-response on the same benchmark.

## Reviewer Notes

- The neuron-vs-SAE comparison is the paper’s cleanest localization-to-control dissociation because both families are evaluated on the same model and benchmark.
- The random-neuron results are included here as a specificity check for the H-neuron intervention, not as a separate headline claim.

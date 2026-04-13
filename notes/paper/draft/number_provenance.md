# Number Provenance Ledger

Every key quantitative claim in the paper traced to its canonical source.

## Detection Quality (§4.1)

| Claim | Value | CI | Source File |
|---|---|---|---|
| H-neuron AUROC | 0.843 | — | `data/gemma3_4b/pipeline/classifier_structure_summary.json` |
| H-neuron disjoint accuracy | 76.5% | — | `data/gemma3_4b/pipeline/classifier_disjoint_summary.json` |
| H-neuron count | 38/348,160 | — | `data/gemma3_4b/pipeline/classifier_structure_summary.json` |
| SAE feature AUROC | 0.848 | — | `data/gemma3_4b/pipeline/classifier_sae_summary.json` |
| Probe head AUROC (gradient-selector comparison pilot) | 1.0 | — | `notes/act3-reports/2026-04-07-d7-causal-pilot-audit.md` §2.2 |
| Probe top-20 AUROC range | 0.87–1.0 | — | `data/contrastive/refusal/iti_refusal_probe_d7/extraction_metadata.json` (`selected_head_manifest`, ranks 1–20) |

## FaithEval Dissociation (§4.2)

| Claim | Value | CI | Source File |
|---|---|---|---|
| H-neuron FaithEval slope | +2.09 pp/α | [1.38, 2.83] | `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json` |
| H-neuron FaithEval no-op→max (α=1→3) | +4.5 pp | [2.9, 6.1] | `data/gemma3_4b/intervention/faitheval/experiment/results.json` (`delta_noop_to_max_pp`) |
| H-neuron FaithEval full sweep (α=0→3) | +6.3 pp | [4.2, 8.5] | `data/gemma3_4b/intervention/faitheval/experiment/results.json` (`delta_0_to_max_pp`) |
| SAE H-features slope | +0.16 pp/α | [-0.51, 0.84] | `data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json` |
| SAE random slope | +0.59 pp/α | — | same file |
| Neuron-minus-SAE slope difference | +1.93 pp/α | [+0.94, +2.92] | `data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json` |
| Neuron-minus-SAE permutation p | <0.001 (4/50,000) | — | same file |
| Neuron-minus-random slope difference range | +1.89 to +2.20 pp/α | all CIs exclude zero | `data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json` |
| FaithEval n | 1,000 | — | `notes/measurement-blueprint.md` |

## Gradient-Selector Jailbreak Comparison (§4.3)

| Claim | Value | CI | Source File |
|---|---|---|---|
| Gradient-ranked full-500 effect | -9.0 pp strict harmfulness rate | [-12.2, -5.8] | `notes/act3-reports/2026-04-08-d7-full500-audit.md` |
| Probe selector: null | ~-2 pp | [includes zero] | `notes/act3-reports/2026-04-07-d7-causal-pilot-audit.md` |
| Probe-causal Jaccard | 0.11 | — | `notes/act3-reports/2026-04-07-d7-causal-pilot-audit.md` |
| Gradient-ranked baseline strict harmfulness rate | 23.4% | — | `notes/act3-reports/2026-04-08-d7-full500-audit.md` |
| Token-cap hits | 112/500 | — | same file |

## ITI MC vs Generation (§5.2)

| Claim | Value | CI | Source File |
|---|---|---|---|
| ITI MC1 effect (pooled) | +6.3 pp | [+3.7, +8.9] | `notes/act3-reports/2026-04-01-priority-reruns-audit.md` §2 |
| ITI MC2 effect (pooled) | +7.49 pp | [+5.28, +9.82] | same file |
| SimpleQA correct-answer rate α=0 | 4.6% | [3.5, 6.1] | `notes/act3-reports/2026-04-01-priority-reruns-audit.md` §3b |
| SimpleQA correct-answer rate α=8 | 2.8% | [1.9, 4.0] | same file |
| SimpleQA attempt rate α=8 | 67.0% | — | same file |

## Bridge (§5.3)

### Test set (primary) — source: `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`

| Claim | Value | CI | Source File |
|---|---|---|---|
| Bridge baseline accuracy | 45.0% adj | [40.7, 49.4] | Phase 3 report §2.1 |
| E0 α=8 delta (adj) | -5.8 pp | [-8.8, -3.0] | Phase 3 report §2.2 |
| E0 α=8 delta (det) | -4.6 pp | [-7.6, -1.6] | Phase 3 report §2.2 |
| McNemar p | 0.0002 | — | Phase 3 report §2.2 |
| Right-to-wrong flips (E0 α=8) | 43 | — | Phase 3 report §2.3 |
| Wrong-to-right rescues (E0) | 14 | — | Phase 3 report §2.3 |
| Wrong-entity substitution mode | 30/43 (70%) | — | Phase 3 report §5.1 |
| Evasion/denial mode | 8/43 (19%) | — | Phase 3 report §5.1 |
| Test set n | 500 | — | Phase 3 report §1 |
| NOT_ATTEMPTED (baseline / ITI) | 2 / 8 | — | Phase 3 report §7 |

### Dev set (E0/E1 comparison only) — source: `notes/act3-reports/2026-04-04-bridge-phase2-dev-results.md`

| Claim | Value | CI | Source File |
|---|---|---|---|
| E1 α=8 delta | -9.0 pp | [-16.0, -3.0] | Phase 2 report §2.2 |
| E1 McNemar p | 0.016 | — | Phase 2 report §2.2 |
| E0/E1 same damaged questions | 10/10 | — | Phase 2 report §2.3 |
| Dev set n | 100 | — | Phase 2 report §1 |

## Jailbreak Measurement (§6)

| Claim | Value | CI | Source File |
|---|---|---|---|
| H-neuron binary delta | +3.0 pp | [includes zero] | `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md` §1.2 |
| H-neuron graded delta α=0→3 | +7.6 pp | [+2.6, +12.8] | same §0 |
| H-neuron graded slope | +2.30 pp/α | [+0.99, +3.58] | same §2 |
| Random control slope | -0.47 pp/α | [-1.42, +0.47] | same §2 |
| Slope difference | +2.77 pp/α | [+1.17, +4.42] | same §2 |
| Permutation p-value | 0.013 | — | same §2 |

## Evaluator Comparison (§6.3)

| Claim | Value | CI | Source File |
|---|---|---|---|
| CSV2 v3 dev accuracy | 86.5% | [76.9, 92.5] | `notes/act3-reports/2026-04-12-4way-evaluator-comparison.md` §0 |
| Binary dev accuracy | 77.0% | — | same |
| StrongREJECT dev accuracy | 74.3% | [63.3, 82.9] | same |
| CSV2 v3 holdout accuracy | 96.0% | [0.90, 1.0] | `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md` §0 |
| StrongREJECT holdout accuracy | 94.0% | [0.88, 1.0] | same |
| CSV2 v2 holdout accuracy | 92.0% | [0.843, 0.980] | same |
| Binary holdout accuracy | 90.0% | [0.80, 0.98] | same |
| v3-SR gap (dev) | 12.2 pp | — | same |
| v3-SR gap (holdout) | 2.0 pp | not significant | same |
| Holdout n | 50 | — | same |
| Contaminated rows removed | 24 | — | same |

## Contamination Bug (§6.4)

| Claim | Value | Source File |
|---|---|---|
| Records reclassified | 167/171 (97.7%) | `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md` §6 |
| Strict harmfulness inflation | 18.8% → 52.2% | same |

## Scope Boundaries

| Claim | Value | CI | Source File |
|---|---|---|---|
| BioASQ effect | -0.06 pp | [-1.5, 1.4] | `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md` |
| BioASQ n | 1,600 | — | same |
| BioASQ behavior change | 1,339/1,600 | — | same |
| FalseQA no-op→max (α=1→3) | +2.5 pp | [-0.6, 5.5] | `data/gemma3_4b/intervention/falseqa/experiment/results.json` (`delta_noop_to_max_pp`) |
| FalseQA full sweep (α=0→3) | +4.8 pp | [1.3, 8.3] | `data/gemma3_4b/intervention/falseqa/experiment/results.json` (`delta_0_to_max_pp`); `data/gemma3_4b/intervention/falseqa/falseqa_negative_control_audit.md` |
| FalseQA n | 687 | — | `data/gemma3_4b/intervention/falseqa/experiment/results.json` |

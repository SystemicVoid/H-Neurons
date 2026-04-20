# Externality Summary

This file is a reviewer-facing derivative of the paper’s control and externality evidence. It condenses the canonical summaries used for §4.

## Canonical Repo Origins Used For This Summary

- `notes/act3-reports/2026-04-01-priority-reruns-audit.md`
- `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`
- `data/manifests/triviaqa_bridge_test500_seed42.json`
- `data/manifests/truthfulqa_final_fold0_heldout_mc1_seed42.json`
- `data/manifests/truthfulqa_final_fold1_heldout_mc1_seed42.json`
- `data/manifests/truthfulqa_final_fold0_heldout_mc2_seed42.json`
- `data/manifests/truthfulqa_final_fold1_heldout_mc2_seed42.json`

## §4.1 Positive Results Are Task-Local

| Claim | Value | CI / exact test |
|---|---|---|
| FalseQA slope | +1.62 pp/α | [0.52, 2.74] |
| FalseQA no-op to max (`α = 1.0 → 3.0`) | +2.5 pp | [-0.6, 5.5] |
| FalseQA full sweep (`α = 0.0 → 3.0`) | +4.8 pp | [1.3, 8.3] |
| FalseQA sample size | 687 | — |
| BioASQ net accuracy effect | -0.06 pp | [-1.5, 1.4] |
| BioASQ sample size | 1,600 | — |
| BioASQ responses behaviorally changed | 1,339 / 1,600 | — |

Interpretation: the same H-neuron intervention is active on compliance-adjacent surfaces but does not produce a meaningful BioASQ accuracy gain.

## §4.2 ITI: Answer Selection vs Open-Ended Generation

| Claim | Value | CI / exact test |
|---|---|---|
| ITI MC1 effect (held-out TruthfulQA folds) | +6.3 pp | [+3.7, +8.9] |
| ITI MC2 effect (held-out TruthfulQA folds) | +7.49 pp | [+5.28, +9.82] |
| TruthfulQA flip counts | 61 wrong-to-right, 20 right-to-wrong | — |
| SimpleQA correct-answer rate at `α = 0.0` | 4.6% | [3.5, 6.1] |
| SimpleQA correct-answer rate at `α = 8.0` | 2.8% | [1.9, 4.0] |
| SimpleQA paired delta (`α = 0.0 → 8.0`) | -1.8 pp | [-3.1, -0.6] |
| SimpleQA attempt rate at `α = 0.0` | 99.7% | — |
| SimpleQA attempt rate at `α = 8.0` | 67.0% | — |
| SimpleQA paired attempt-rate delta | -32.7 pp | [-35.6, -29.9] |

Interpretation: ITI improves constrained answer selection on held-out TruthfulQA folds, but that gain does not transfer to open-ended SimpleQA generation.

## §4.3 Bridge: Wrong-Entity Substitution

### Locked test protocol

| Item | Value |
|---|---|
| Held-out manifest | [`../data/manifests/triviaqa_bridge_test500_seed42.json`](../data/manifests/triviaqa_bridge_test500_seed42.json) |
| Test size | 500 |
| Comparison | Baseline `α = 1.0` vs E0 ITI `α = 8.0` |
| Protocol | One-shot frozen test run; no tuning on test data |

### Headline bridge results

| Claim | Value | CI / exact test |
|---|---|---|
| Baseline adjudicated accuracy | 45.0% | [40.7, 49.4] |
| E0 `α = 8.0` adjudicated delta | -5.8 pp | [-8.8, -3.0] |
| E0 `α = 8.0` deterministic delta | -4.6 pp | [-7.6, -1.6] |
| McNemar significance | `p = 0.0002` | — |
| Flip counts | 43 right-to-wrong, 14 wrong-to-right | net -29 |
| NOT_ATTEMPTED counts | 2 / 8 | baseline / ITI |

### Failure mode coding on 43 right-to-wrong flips

| Category | Count | Share |
|---|---:|---:|
| Wrong-entity substitution | 30 | 70% |
| Evasion / factual denial | 8 | 19% |
| Answer dilution / verbosity | 3 | 7% |
| Formal refusal | 2 | 5% |

### Representative wrong-entity substitutions

| Question cue | Baseline | ITI | Relationship |
|---|---|---|---|
| Danny Boyle 1996 film | `Trainspotting` | `Slumdog Millionaire` | Same director, wrong film |
| Third musician in 1959 crash | `Ritchie Valens` | `J.P. Richardson` | Same crash, wrong victim |
| Family Guy spin-off character | `Cleveland Brown` | `Peter Griffin` | Same show, wrong character |
| Dickens novel with Merdle and Sparkler | `Little Dorrit` | `Bleak House` | Same author, wrong novel |

Interpretation: the bridge result is not primarily a refusal effect. The dominant damage is factual corruption within the correct semantic neighborhood.

## Reviewer Notes

- The bridge taxonomy is a behavioral diagnosis rather than a formal causal claim about the underlying circuit.
- The exact category percentages come from a single-rater manual coding pass over 43 flips.

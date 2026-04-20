# Failure Coding Manifest

This manifest documents the reviewer-facing coding surface for the TriviaQA bridge wrong-entity substitution audit in §4.3 of the paper.

## Unit of Analysis

| Item | Value |
|---|---|
| Benchmark | TriviaQA bridge held-out test set |
| Bundled held-out manifest | [`data/manifests/triviaqa_bridge_test500_seed42.json`](data/manifests/triviaqa_bridge_test500_seed42.json) |
| Comparison | Baseline `α = 1.0` vs E0 ITI `α = 8.0` |
| Coded subset | 43 right-to-wrong flips from the paired test-set comparison |
| Paper-facing source summary | [`support/externality_summary.md`](support/externality_summary.md) |

## Coding Categories

| Category | Operational definition | Count | Share |
|---|---|---:|---:|
| Wrong-entity substitution | The model answers with a different entity from the correct semantic neighborhood. | 30 | 70% |
| Evasion / factual denial | The model denies the premise, claims uncertainty, or asserts that the answer does not exist. | 8 | 19% |
| Answer dilution / verbosity | The correct answer is hedged, diluted, or buried below the grading threshold. | 3 | 7% |
| Formal refusal | The model explicitly refuses or fully evades answering. | 2 | 5% |

## Example Cases

| Question cue | Baseline | ITI | Category |
|---|---|---|---|
| Danny Boyle’s 1996 film | `Trainspotting` | `Slumdog Millionaire` | Wrong-entity substitution |
| Other musician in the 1959 crash | `Ritchie Valens` | `J.P. Richardson` | Wrong-entity substitution |
| Family Guy character who moved to Stoolbend | `Cleveland Brown` | `Peter Griffin` | Wrong-entity substitution |
| Dickens novel with Merdle and Sparkler | `Little Dorrit` | `Bleak House` | Wrong-entity substitution |
| “Turandot” completion example | Correct opera-completion answer | “He did not complete a Puccini opera.” | Evasion / factual denial |

## Provenance

| Item | Value |
|---|---|
| Canonical internal report used to derive this manifest | `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md` |
| Bundled reviewer-facing derivative | [`support/externality_summary.md`](support/externality_summary.md) |
| Bridge grading stack | `scripts/evaluate_intervention.py` |
| Test-run orchestration | `scripts/infra/triviaqa_bridge_test.sh` |

## Caveats

- This coding surface is a behavioral diagnosis, not a claim about the exact internal circuit mechanism.
- The category percentages come from a single-rater manual classification of 43 flips. They are useful for reviewer interpretation but are not presented as an inter-rater-reliability result.
- Raw bridge response JSONLs are not bundled here; this supplement surfaces only the reviewer-facing taxonomy and paired outcome summary.

# Failure Coding Manifest

This manifest documents the reviewer-facing coding surface for the TriviaQA bridge wrong-entity substitution audit in §4.3 of the paper. It is the bundled derivative for bridge IRR interpretation, alongside [`support/externality_summary.md`](support/externality_summary.md), [`data/judge_validation/bridge_irr/bridge_irr_summary.json`](data/judge_validation/bridge_irr/bridge_irr_summary.json), and the redacted case-label file.

## Unit of Analysis

| Item | Value |
|---|---|
| Benchmark | TriviaQA bridge held-out test set |
| Bundled held-out manifest | [`data/manifests/triviaqa_bridge_test500_seed42.json`](data/manifests/triviaqa_bridge_test500_seed42.json) |
| Comparison | Baseline `α = 1.0` vs E0 ITI `α = 8.0` |
| Coded subset | 43 right-to-wrong flips from the paired test-set comparison |
| Paper-facing source summary | [`support/externality_summary.md`](support/externality_summary.md) |

## Coding Categories

Shares are computed over the 43 right-to-wrong flips from dual-rated adjudicated labels (rubric `bridge_incorrect_response_v1`, adjudication rule committed at `0e965d5`). 95% CIs are Wilson intervals.

| Category | Operational definition | Count | Share | 95% CI |
|---|---|---:|---:|---|
| Wrong-entity substitution | The model answers with a different entity from the correct semantic neighborhood. | 31 | 72.1% | [57.3, 83.3] |
| Evasion / factual denial | The model denies the premise, claims uncertainty, or asserts that the answer does not exist. | 9 | 20.9% | [11.4, 35.2] |
| Answer dilution / verbosity | The correct answer is hedged, diluted, or buried below the grading threshold. | 3 | 7.0% | [2.4, 18.6] |
| Formal refusal | The model explicitly refuses or fully evades answering. | 0 | 0.0% | [0.0, 8.2] |

The 14 wrong-to-right rescues are all `wrong_entity_substitution` under the same rubric.

## Example Cases

| Question cue | Baseline | ITI | Category |
|---|---|---|---|
| Danny Boyle’s 1996 film | `Trainspotting` | `Slumdog Millionaire` | Wrong-entity substitution |
| Other musician in the 1959 crash | `Ritchie Valens` | `J.P. Richardson` | Wrong-entity substitution |
| Family Guy character who moved to Stoolbend | `Cleveland Brown` | `Peter Griffin` | Wrong-entity substitution |
| Dickens novel with Merdle and Sparkler | `Little Dorrit` | `Bleak House` | Wrong-entity substitution |
| “Turandot” completion example | Correct opera-completion answer | “He did not complete a Puccini opera.” | Evasion / factual denial |

## Inter-Rater Reliability

All 57 discordant test-split cases (43 right-to-wrong + 14 wrong-to-right) were dual-rated. Rater A is the first author (human); Rater B is an LLM judge (`gpt-4o-2024-11-20`, temperature 0, strict JSON schema). The two disagreements were resolved under the pre-frozen adjudication rule before any test label was committed.

| Metric | Value | 95% CI |
|---|---|---|
| Raw agreement | 55/57 = 96.5% | [88.1, 99.0] (Wilson) |
| Cohen's κ | 0.90 | — |
| Gwet's AC1 | 0.96 | — |
| Disagreements resolved | 2 | — |
| Rule-gap cases during adjudication | 0/2 | — |

Since the second rater is an LLM judge rather than an independent human, we present this as a sensitivity check on the category shares rather than a strong-form human–human IRR. Raw agreement is the most robust single number; κ and AC1 are reported for completeness under the four-category label set.

## Provenance

| Item | Value |
|---|---|
| Canonical internal report used to derive the flip counts | `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md` |
| Bundled reviewer-facing derivative | [`support/externality_summary.md`](support/externality_summary.md) |
| Bridge grading stack | `code/scripts/evaluate_intervention.py` |
| Bundled manifest builder | `code/scripts/build_triviaqa_bridge_manifest.py` |
| Dual-rater workflow implementation | `code/scripts/prepare_bridge_irr_queue.py`, `code/scripts/bridge_irr_rater_b.py`, `code/scripts/bridge_irr_label.py`, `code/scripts/finalize_bridge_irr.py` |
| Rubric + adjudication rule | [`data/judge_validation/bridge_irr/adjudication_rule.md`](data/judge_validation/bridge_irr/adjudication_rule.md) (git `0e965d5`) |
| Machine-readable IRR summary | [`data/judge_validation/bridge_irr/bridge_irr_summary.json`](data/judge_validation/bridge_irr/bridge_irr_summary.json) |
| Redacted adjudicated labels | [`data/judge_validation/bridge_irr/adjudicated_labels.jsonl`](data/judge_validation/bridge_irr/adjudicated_labels.jsonl) |

## Caveats

- This coding surface is a behavioral diagnosis, not a claim about the exact internal circuit mechanism.
- Rater B is an LLM judge; these numbers are a sensitivity check on category shares, not a strong-form human–human IRR.
- Raw bridge response JSONLs, blinded queues, and raw rater progress files are omitted for safety/anonymization. The bundled reviewer-facing derivatives are the frozen rule, machine-readable summary, and redacted adjudicated labels.

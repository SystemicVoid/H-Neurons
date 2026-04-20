# Reproduction Manifest

This manifest is a minimal rerun map for the paper’s anchor results. It is not a full artifact bundle. The aim is to identify the data roots, scripts, expected outputs, and bundled safe manifests without exposing anonymization-sensitive sidecars.

All script paths below refer to the bundled code tree under `code/scripts/`.

## Bundled Safe Inputs

| File | Use |
|---|---|
| [`data/manifests/truthfulqa_final_fold0_heldout_mc1_seed42.json`](data/manifests/truthfulqa_final_fold0_heldout_mc1_seed42.json) | TruthfulQA MC1 fold-0 held-out rerun |
| [`data/manifests/truthfulqa_final_fold1_heldout_mc1_seed42.json`](data/manifests/truthfulqa_final_fold1_heldout_mc1_seed42.json) | TruthfulQA MC1 fold-1 held-out rerun |
| [`data/manifests/truthfulqa_final_fold0_heldout_mc2_seed42.json`](data/manifests/truthfulqa_final_fold0_heldout_mc2_seed42.json) | TruthfulQA MC2 fold-0 held-out rerun |
| [`data/manifests/truthfulqa_final_fold1_heldout_mc2_seed42.json`](data/manifests/truthfulqa_final_fold1_heldout_mc2_seed42.json) | TruthfulQA MC2 fold-1 held-out rerun |
| [`data/manifests/triviaqa_bridge_test500_seed42.json`](data/manifests/triviaqa_bridge_test500_seed42.json) | TriviaQA bridge held-out test run |

## Anchor Result Map

### §3 Localization: FaithEval H-neurons vs SAE features

| Item | Value |
|---|---|
| Core repo data roots | `data/gemma3_4b/pipeline/`, `data/gemma3_4b/intervention/faitheval/`, `data/gemma3_4b/intervention/faitheval_sae/` |
| Main scripts | `code/scripts/classifier.py`, `code/scripts/classifier_sae.py`, `code/scripts/run_intervention.py`, `code/scripts/run_negative_control.py`, `code/scripts/run_sae_negative_control.py`, `code/scripts/compute_faitheval_slope_difference.py` |
| Expected outputs | `classifier_disjoint_summary.json`, `classifier_sae_summary.json`, `results.json`, `comparison_summary.json`, `slope_difference_summary.json` |
| Reviewer-facing bundled derivative | [`support/localization_summary.md`](support/localization_summary.md) |

### §4.2 ITI answer selection vs open-ended generation

| Item | Value |
|---|---|
| Core repo data roots | `data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/`, `data/gemma3_4b/intervention/simpleqa_factual_phrase_*`, `data/gemma3_4b/intervention/truthfulqa_mc_*` |
| Main scripts | `code/scripts/extract_truthfulness_iti.py`, `code/scripts/intervene_iti.py`, `code/scripts/run_intervention.py`, `code/scripts/evaluate_intervention.py` |
| Bundled safe inputs | Held-out TruthfulQA manifests in `data/manifests/` |
| Expected outputs | `results.json`, per-alpha JSONL files, judge summaries |
| Reviewer-facing bundled derivative | [`support/externality_summary.md`](support/externality_summary.md) |

### §4.3 TriviaQA bridge externality test

| Item | Value |
|---|---|
| Core repo data roots | `data/gemma3_4b/intervention/triviaqa_bridge/`, `data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/` |
| Main scripts | `code/scripts/build_triviaqa_bridge_manifest.py`, `code/scripts/run_intervention.py`, `code/scripts/evaluate_intervention.py` |
| Bundled safe input | [`data/manifests/triviaqa_bridge_test500_seed42.json`](data/manifests/triviaqa_bridge_test500_seed42.json) |
| Expected outputs | `results.json`, `audit_stats.json`, per-alpha JSONL files |
| Reviewer-facing bundled derivative | [`support/externality_summary.md`](support/externality_summary.md), [`failure_coding_manifest.md`](failure_coding_manifest.md) |

### §5 Measurement and evaluator audit

| Item | Value |
|---|---|
| Core repo data roots | `data/gemma3_4b/intervention/jailbreak/`, `data/judge_validation/` |
| Main scripts | `code/scripts/evaluate_csv2.py`, `code/scripts/evaluate_strongreject.py`, `code/scripts/analysis_holdout_evaluator.py`, `code/scripts/jailbreak_measurement_cleanup.py` |
| Bundled safe artifact | [`data/judge_validation/holdout_comparison.json`](data/judge_validation/holdout_comparison.json) |
| Expected outputs | scored evaluator directories, `holdout_comparison.json`, comparison summaries |
| Reviewer-facing bundled derivative | [`support/measurement_summary.md`](support/measurement_summary.md), [`evaluation_manifest.md`](evaluation_manifest.md) |

## Omitted Provenance Sidecars

- Expected raw runs in the repository normally emit `run_intervention.provenance.*.json` or evaluator provenance files beside the main outputs.
- Those sidecars are intentionally omitted from this supplement because they expose local filesystem paths, hostnames, command-line details, and other anonymization-sensitive metadata.
- The reviewer-facing replacement in this package is the combination of `number_provenance.md`, the derived support summaries, and the bundled safe manifests / holdout JSON.

# Ground-Truth Metric Tables

## Readout Surfaces

### Detector Quality

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `readout.disjoint.accuracy` | 76.5% | 780 | `[73.6, 79.5]%` | `classifier_disjoint_summary` |
| `readout.disjoint.auroc` | 84.3% | 780 | `[81.5, 87.0]%` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.fn` | 93 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.fp` | 90 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.tn` | 301 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.tp` | 296 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.f1` | 76.4% | 780 | `[73.2, 79.4]%` | `classifier_disjoint_summary` |
| `readout.disjoint.precision` | 76.7% | 780 | `[73.4, 80.1]%` | `classifier_disjoint_summary` |
| `readout.disjoint.recall` | 76.1% | 780 | `[71.7, 80.5]%` | `classifier_disjoint_summary` |
| `readout.disjoint.test_size` | 780 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.mistral24b.selected_h_neurons` | 10 | — | `—` | `mistral24b_classifier_test_metrics` |
| `readout.mistral24b.test.accuracy` | 77.5% | 200 | `[71.5, 83.0]%` | `mistral24b_classifier_test_metrics` |
| `readout.mistral24b.test.auroc` | 0.871 | 200 | `[0.818, 0.917]` | `mistral24b_classifier_test_metrics` |
| `readout.mistral24b.test.confusion.fn` | 21 | 200 | `—` | `mistral24b_classifier_test_metrics` |
| `readout.mistral24b.test.confusion.fp` | 24 | 200 | `—` | `mistral24b_classifier_test_metrics` |
| `readout.mistral24b.test.f1` | 77.8% | 200 | `[71.8, 83.5]%` | `mistral24b_classifier_test_metrics` |
| `readout.mistral24b.total_ffn_neurons` | 1310720 | — | `—` | `mistral24b_classifier_test_metrics` |
| `readout.overlap.accuracy` | 77.7% | 1993 | `[75.9, 79.5]%` | `classifier_overlap_summary` |
| `readout.overlap.auroc` | 86.3% | 1993 | `[84.7, 87.9]%` | `classifier_overlap_summary` |

### SAE Readout

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `readout.sae.accuracy` | 77.2% | 782 | `—` | `classifier_sae_summary` |
| `readout.sae.auroc` | 84.8% | 782 | `—` | `classifier_sae_summary` |
| `readout.sae.layer_count` | 10 | — | `—` | `classifier_sae_summary` |
| `readout.sae.n_nonzero_features` | 509 | — | `—` | `classifier_sae_summary` |
| `readout.sae.n_positive_features` | 266 | — | `—` | `classifier_sae_summary` |
| `readout.sae.test_size` | 782 | 782 | `—` | `classifier_sae_summary` |

### Localization / Sparsity

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `readout.pipeline.consistent_total` | 3115 | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.disjoint_evaluated_total` | 780 | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.extracted_answer_tokens` | 2997 | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.sampled_questions` | 3500 | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.selected_h_neurons` | 38 | — | `—` | `pipeline_summary_site, classifier_disjoint_summary` |
| `readout.pipeline.total_ffn_neurons` | 348160 | — | `—` | `pipeline_summary_site, classifier_structure_summary` |
| `readout.structure.band.early` | 18 | — | `—` | `classifier_structure_summary` |
| `readout.structure.band.late` | 10 | — | `—` | `classifier_structure_summary` |
| `readout.structure.band.middle` | 10 | — | `—` | `classifier_structure_summary` |

## Intervention Surfaces

### FaithEval Core

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `intervention.faitheval.anti.delta_0_to_max_pp` | +6.3 pp | 1000 | `[+4.2, +8.5]` | `intervention_sweep_site` |
| `intervention.faitheval.anti.delta_noop_to_max_pp` | +4.5 pp | 1000 | `[+2.9, +6.1]` | `intervention_sweep_site` |
| `intervention.faitheval.anti.slope_pp_per_alpha` | +2.09 pp/alpha | 1000 | `[+1.4, +2.8]` | `intervention_sweep_site` |
| `intervention.faitheval.h_spearman` | 1.00 | 7 | `—` | `faitheval_control_comparison` |
| `intervention.faitheval.mean_slope_difference_pp_per_alpha` | +2.01 pp/alpha | 8 | `[+1.9, +2.1]` | `faitheval_control_slope_difference` |
| `intervention.faitheval.random_max_slope_pp_per_alpha` | +0.21 pp/alpha | 8 | `—` | `faitheval_control_comparison` |
| `intervention.faitheval.random_mean_slope_pp_per_alpha` | +0.02 pp/alpha | 5 | `—` | `faitheval_control_comparison` |
| `intervention.faitheval.sample_size` | 1000 | 1000 | `—` | `intervention_sweep_site` |
| `intervention.faitheval.seedwise_positive_differences` | 8 | 8 | `—` | `faitheval_control_slope_difference` |

### FalseQA / BioASQ

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `intervention.bioasq.changed_response_count` | 1339 | 1600 | `—` | `bioasq_alpha_0_rows, bioasq_alpha_3_rows` |
| `intervention.bioasq.delta_0_to_max_pp` | -0.1 pp | 1600 | `[-1.5, +1.4]` | `bioasq_results` |
| `intervention.bioasq.random_mean_slope_pp_per_alpha` | +0.04 pp/alpha | 3 | `—` | `bioasq_control_comparison` |
| `intervention.bioasq.response_length_mean_alpha0` | 41.3 | 1600 | `—` | `bioasq_alpha_0_rows` |
| `intervention.bioasq.response_length_mean_alpha3` | 26.6 | 1600 | `—` | `bioasq_alpha_3_rows` |
| `intervention.bioasq.sample_size` | 1600 | 1600 | `—` | `bioasq_results` |
| `intervention.bioasq.slope_pp_per_alpha` | -0.14 pp/alpha | 1600 | `[-0.6, +0.3]` | `bioasq_results` |
| `intervention.falseqa.delta_0_to_max_pp` | +4.8 pp | 687 | `[+1.3, +8.3]` | `falseqa_results` |
| `intervention.falseqa.delta_noop_to_max_pp` | +2.5 pp | 687 | `[-0.6, +5.5]` | `falseqa_results` |
| `intervention.falseqa.sample_size` | 687 | 687 | `—` | `falseqa_results` |
| `intervention.falseqa.slope_pp_per_alpha` | +1.62 pp/alpha | 687 | `[+0.5, +2.7]` | `falseqa_results` |

### SAE Variants

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `intervention.faitheval_sae.h_slope_pp_per_alpha` | +0.16 pp/alpha | 1000 | `—` | `faitheval_sae_control_comparison` |
| `intervention.faitheval_sae.h_spearman` | 0.18 | 7 | `—` | `faitheval_sae_control_comparison` |
| `intervention.faitheval_sae.neuron_minus_sae_slope_pp_per_alpha` | +1.93 pp/alpha | 1000 | `[+0.9, +2.9]` | `faitheval_sae_slope_difference` |
| `intervention.faitheval_sae.permutation_p` | 1.00e-04 | 50000 | `—` | `faitheval_sae_slope_difference` |
| `intervention.faitheval_sae.random_mean_slope_pp_per_alpha` | +0.59 pp/alpha | 3 | `—` | `faitheval_sae_control_comparison` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_answer_span_margin` | +1.80 | 840 | `[1.069, 2.536]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_anti_margin` | +8.00 | 840 | `[6.897, 9.088]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_compliance` | 67.4% | 840 | `[64.1, 70.5]%` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_noop_answer_span_margin` | -0.33 | 840 | `[-0.608, -0.055]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_noop_anti_margin` | -0.30 | 840 | `[-0.728, 0.120]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_noop_compliance_pp` | +1.0 pp | 840 | `[-0.8, +2.7]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_random_seed_mean_answer_span_margin` | -0.53 | 10 | `[-0.811, -0.262]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_readout_answer_span_margin` | -0.98 | 840 | `[-1.380, -0.565]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_readout_anti_margin` | -1.22 | 840 | `[-1.773, -0.665]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_readout_compliance_pp` | +1.4 pp | 840 | `[-0.4, +3.2]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_utility_selected_answer_span_margin` | -0.22 | 840 | `[-0.468, 0.008]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_utility_selected_anti_margin` | +0.46 | 840 | `[0.161, 0.750]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.answer_span_selected_minus_utility_selected_compliance_pp` | +1.2 pp | 840 | `[-0.4, +2.7]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_answer_span.outside_old_shortlist_fraction` | 32.7% | 266 | `—` | `faitheval_sae_utility_selector_summary` |
| `intervention.faitheval_sae_answer_span.readout_overlap_jaccard` | 35.7% | 392 | `—` | `faitheval_sae_utility_selector_summary` |
| `intervention.faitheval_sae_answer_span.selected_k` | 266 | 509 | `—` | `faitheval_sae_utility_selector_summary` |
| `intervention.faitheval_sae_answer_span.utility_overlap_jaccard` | 45.8% | 365 | `—` | `faitheval_sae_utility_selector_summary` |
| `intervention.faitheval_sae_utility.candidate_pool_size` | 509 | 509 | `—` | `faitheval_sae_utility_selector_summary` |
| `intervention.faitheval_sae_utility.heldout_sample_size` | 840 | 840 | `—` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_utility.outside_old_shortlist_fraction` | 28.2% | 266 | `—` | `faitheval_sae_utility_selector_summary` |
| `intervention.faitheval_sae_utility.readout_overlap_jaccard` | 33.0% | 400 | `—` | `faitheval_sae_utility_selector_summary` |
| `intervention.faitheval_sae_utility.readout_selected_compliance` | 66.0% | 840 | `[62.7, 69.1]%` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_utility.readout_selected_margin` | +9.23 | 840 | `[7.862, 10.573]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_utility.selected_k` | 266 | 509 | `—` | `faitheval_sae_utility_selector_summary` |
| `intervention.faitheval_sae_utility.utility_minus_noop_compliance_pp` | -0.2 pp | 840 | `[-1.7, +1.2]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_utility.utility_minus_noop_margin` | -0.76 | 840 | `[-1.084, -0.422]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_utility.utility_minus_readout_compliance_pp` | +0.2 pp | 840 | `[-1.3, +1.8]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_utility.utility_minus_readout_margin` | -1.68 | 840 | `[-2.171, -1.190]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_utility.utility_selected_compliance` | 66.2% | 840 | `[62.9, 69.3]%` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_utility.utility_selected_margin` | +7.55 | 840 | `[6.475, 8.598]` | `faitheval_sae_utility_selector_heldout` |
| `intervention.faitheval_sae_utility_positive.selected_k` | 154 | 840 | `—` | `faitheval_sae_utility_selector_augment` |
| `intervention.faitheval_sae_utility_positive.utility_positive_minus_noop_compliance_pp` | +0.1 pp | 840 | `[-1.3, +1.5]` | `faitheval_sae_utility_selector_augment` |
| `intervention.faitheval_sae_utility_positive.utility_positive_minus_noop_margin` | -0.72 | 840 | `[-1.041, -0.385]` | `faitheval_sae_utility_selector_augment` |
| `intervention.faitheval_sae_utility_positive.utility_positive_selected_compliance` | 66.5% | 840 | `[63.3, 69.7]%` | `faitheval_sae_utility_selector_augment` |
| `intervention.faitheval_sae_utility_positive.utility_positive_selected_margin` | +7.59 | 840 | `[6.512, 8.637]` | `faitheval_sae_utility_selector_augment` |

### Mistral FaithEval

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `intervention.mistral24b.cp5.faitheval.alpha_0_0_compliance` | 53.0% | 200 | `[46.1, 59.8]%` | `mistral24b_faitheval_cp5_results` |
| `intervention.mistral24b.cp5.faitheval.alpha_3_0_compliance` | 53.0% | 200 | `[46.1, 59.8]%` | `mistral24b_faitheval_cp5_results` |
| `intervention.mistral24b.cp5.faitheval.delta_0_to_3_pp` | +0.0 pp | 200 | `[-4.0, +4.0]` | `mistral24b_faitheval_cp5_results` |
| `intervention.mistral24b.cp5.faitheval.delta_1_to_3_pp` | +1.0 pp | 200 | `[-2.5, +4.5]` | `mistral24b_faitheval_cp5_results` |
| `intervention.mistral24b.cp5.faitheval.h_minus_unconstrained_random_slope_pp_per_alpha` | +0.81 pp/alpha | 5 | `—` | `mistral24b_faitheval_cp5_control` |
| `intervention.mistral24b.cp5.faitheval.layer_matched_random_mean_slope_pp_per_alpha` | -0.04 pp/alpha | 3 | `[-0.0, -0.0]` | `mistral24b_faitheval_cp5_control` |
| `intervention.mistral24b.cp5.faitheval.n_h_neurons` | 10 | — | `—` | `mistral24b_faitheval_cp5_results` |
| `intervention.mistral24b.cp5.faitheval.slope_pp_per_alpha` | +0.79 pp/alpha | 200 | `[-0.6, +2.2]` | `mistral24b_faitheval_cp5_results` |
| `intervention.mistral24b.cp5.faitheval.unconstrained_random_mean_slope_pp_per_alpha` | -0.02 pp/alpha | 5 | `[-0.1, +0.1]` | `mistral24b_faitheval_cp5_control` |
| `intervention.mistral24b.h1.faitheval.alpha_0_0_compliance` | 51.5% | 200 | `[44.6, 58.3]%` | `mistral24b_h1_faitheval_results` |
| `intervention.mistral24b.h1.faitheval.alpha_3_0_compliance` | 52.0% | 200 | `[45.1, 58.8]%` | `mistral24b_h1_faitheval_results` |
| `intervention.mistral24b.h1.faitheval.delta_0_to_3_pp` | +0.5 pp | 200 | `[-3.0, +4.0]` | `mistral24b_h1_faitheval_results` |
| `intervention.mistral24b.h1.faitheval.delta_1_to_3_pp` | +0.5 pp | 200 | `[-2.0, +3.0]` | `mistral24b_h1_faitheval_results` |
| `intervention.mistral24b.h1.faitheval.n_h_neurons` | 9 | — | `—` | `mistral24b_h1_faitheval_results` |
| `intervention.mistral24b.h1.faitheval.slope_pp_per_alpha` | +0.14 pp/alpha | 200 | `[-0.9, +1.2]` | `mistral24b_h1_faitheval_results` |
| `intervention.mistral24b.h1.selection.heldout_dev_accuracy` | 73.5% | 200 | `—` | `mistral24b_h1_selection_summary` |
| `intervention.mistral24b.h1.selection.selected_c` | 0.750 | 10 | `—` | `mistral24b_h1_selection_summary` |
| `intervention.mistral24b.h1.selection.selected_h_neurons` | 9 | — | `—` | `mistral24b_h1_selection_summary` |
| `intervention.mistral24b.h1.selection.selection_score` | 1.395 | 10 | `—` | `mistral24b_h1_selection_summary` |
| `intervention.mistral24b.h1.selection.triviaqa_alpha0_deterministic_accuracy` | 66.0% | 200 | `—` | `mistral24b_h1_selection_summary` |

### Binary Jailbreak

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `intervention.jailbreak.binary.delta_0_to_max_pp` | +3.0 pp | 500 | `[-1.2, +7.2]` | `jailbreak_results` |
| `intervention.jailbreak.binary.slope_pp_per_alpha` | +1.04 pp/alpha | 500 | `[-0.3, +2.3]` | `jailbreak_results` |

## Measurement Surfaces

### Jailbreak Ruler Sensitivity

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `measurement.jailbreak.audit.contamination_bug_borderline_reclassified` | 167 | 171 | `—` | `jailbreak_seed0_control_audit` |
| `measurement.jailbreak.audit.contamination_bug_strict_harmfulness_after` | 52.2% | 500 | `—` | `jailbreak_seed0_control_audit` |
| `measurement.jailbreak.audit.contamination_bug_strict_harmfulness_before` | 18.8% | 500 | `—` | `jailbreak_seed0_control_audit` |
| `measurement.jailbreak.baseline_noop.v2_rate` | 23.4% | 500 | `—` | `available_jailbreak_evaluator_comparison` |
| `measurement.jailbreak.baseline_noop.v3_rate` | 34.2% | 500 | `—` | `available_jailbreak_evaluator_comparison` |
| `measurement.jailbreak.binary.alpha_0_count` | 152 | 500 | `—` | `jailbreak_results` |
| `measurement.jailbreak.binary.alpha_3_count` | 167 | 500 | `—` | `jailbreak_results` |
| `measurement.jailbreak.causal_locked.v2_rate` | 14.4% | 500 | `—` | `available_jailbreak_evaluator_comparison` |
| `measurement.jailbreak.causal_locked.v3_rate` | 19.7% | 493 | `—` | `available_jailbreak_evaluator_comparison` |
| `measurement.jailbreak.v2.endpoint_delta_csv2_yes_pp` | +7.6 pp | 500 | `—` | `jailbreak_control_v2` |
| `measurement.jailbreak.v2.gap_h_minus_random_mean_pp_per_alpha` | +2.77 pp/alpha | 500 | `—` | `jailbreak_control_v2` |
| `measurement.jailbreak.v2.h_minus_random_permutation_p` | 0.013 | 50000 | `—` | `jailbreak_seed0_control_audit` |
| `measurement.jailbreak.v2.h_slope_csv2_yes_pp_per_alpha` | +2.30 pp/alpha | 500 | `—` | `jailbreak_control_v2` |
| `measurement.jailbreak.v2.random_mean_slope_csv2_yes_pp_per_alpha` | -0.47 pp/alpha | 1 | `—` | `jailbreak_control_v2` |
| `measurement.jailbreak.v3.gap_h_minus_random_mean_pp_per_alpha` | +0.80 pp/alpha | 500 | `—` | `jailbreak_control_v3` |
| `measurement.jailbreak.v3.h_slope_csv2_yes_pp_per_alpha` | +0.46 pp/alpha | 500 | `—` | `jailbreak_control_v3` |
| `measurement.jailbreak.v3.random_mean_slope_csv2_yes_pp_per_alpha` | -0.34 pp/alpha | 1 | `—` | `jailbreak_control_v3` |

### Mistral Jailbreak

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `measurement.mistral24b.jailbreak_anchor3.binary_256_vs_full.agreement` | 96.3% | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_256_vs_full.cohen_kappa` | 0.925 | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_256_vs_full.left_minus_right_harmful_rate_pp` | -1.3 pp | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_256_vs_full.mcnemar_p` | 0.003 | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_full.combined_harmful_rate` | 46.1% | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_full.delta_0_to_3_pp` | +32.6 pp | 500 | `[+28.2, +37.2]` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_full.explicit_error_count` | 0 | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_full.slope_pp_per_alpha` | +10.67 pp/alpha | 500 | `[+9.2, +12.2]` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_full_vs_csv3.agreement` | 96.3% | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_full_vs_csv3.cohen_kappa` | 0.926 | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_full_vs_csv3.left_minus_right_harmful_rate_pp` | -0.1 pp | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.binary_full_vs_csv3.mcnemar_p` | 0.908 | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.csv3_full.combined_harmful_rate` | 46.2% | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.csv3_full.delta_0_to_3_pp` | +32.4 pp | 500 | `[+28.0, +36.8]` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.csv3_full.explicit_error_count` | 0 | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.csv3_full.slope_pp_per_alpha` | +10.51 pp/alpha | 500 | `[+9.0, +12.0]` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.csv3_vs_strongreject.agreement` | 91.8% | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.csv3_vs_strongreject.cohen_kappa` | 0.836 | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.csv3_vs_strongreject.left_minus_right_harmful_rate_pp` | -5.7 pp | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.csv3_vs_strongreject.mcnemar_p` | 2.29e-20 | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.strongreject_full.combined_harmful_rate` | 51.8% | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.strongreject_full.delta_0_to_3_pp` | +31.4 pp | 500 | `[+27.0, +35.8]` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.strongreject_full.explicit_error_count` | 0 | 2000 | `—` | `mistral24b_anchor3_jailbreak_summary` |
| `measurement.mistral24b.jailbreak_anchor3.strongreject_full.slope_pp_per_alpha` | +10.29 pp/alpha | 500 | `[+8.9, +11.8]` | `mistral24b_anchor3_jailbreak_summary` |

### FaithEval Cleanup

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `measurement.faitheval.standard_raw.slope_pp_per_alpha` | -1.41 pp/alpha | 1000 | `[-2.3, -0.5]` | `intervention_sweep_site` |
| `measurement.faitheval.standard_remap.parse_failures` | 150 | 1000 | `—` | `faitheval_standard_remap` |
| `measurement.faitheval.standard_remap.strict_recovered_count` | 140 | 1000 | `—` | `faitheval_standard_remap` |
| `measurement.faitheval.standard_remap.strict_rescored_compliance_rate` | 72.1% | 1000 | `—` | `faitheval_standard_remap` |

### SIMID Calibration

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `measurement.simid.open_calibration.cohen_kappa` | 0.759 | 402 | `—` | `simid_open_calibration_summary` |
| `measurement.simid.open_calibration.gwet_ac1` | 0.897 | 402 | `—` | `simid_open_calibration_summary` |
| `measurement.simid.open_calibration.n_disagreements` | 34 | 402 | `—` | `simid_open_calibration_summary` |
| `measurement.simid.open_calibration.raw_agreement` | 91.5% | 402 | `[88.4, 93.9]%` | `simid_open_calibration_summary` |
| `measurement.simid.open_calibration.rule_gap_rate` | 0.0% | 34 | `[0.0, 10.2]%` | `simid_open_calibration_summary` |
| `measurement.simid.prospective_open_calibration.codex55.cohen_kappa` | 0.821 | 150 | `—` | `simid_prospective_open_calibration_codex55` |
| `measurement.simid.prospective_open_calibration.codex55.gwet_ac1` | 0.835 | 150 | `—` | `simid_prospective_open_calibration_codex55` |
| `measurement.simid.prospective_open_calibration.codex55.raw_agreement` | 88.7% | 150 | `[82.6, 92.8]%` | `simid_prospective_open_calibration_codex55` |
| `measurement.simid.prospective_open_calibration.codex55.rule_gap_count` | 0 | 150 | `—` | `simid_prospective_open_calibration_codex55` |
| `measurement.simid.prospective_open_calibration.human.cohen_kappa` | 0.798 | 150 | `—` | `simid_prospective_open_calibration_human` |
| `measurement.simid.prospective_open_calibration.human.gwet_ac1` | 0.816 | 150 | `—` | `simid_prospective_open_calibration_human` |
| `measurement.simid.prospective_open_calibration.human.raw_agreement` | 87.3% | 150 | `[81.1, 91.7]%` | `simid_prospective_open_calibration_human` |
| `measurement.simid.prospective_open_calibration.human.rule_gap_count` | 0 | 150 | `—` | `simid_prospective_open_calibration_human` |
| `measurement.simid.prospective_open_calibration.opus47.cohen_kappa` | 0.873 | 150 | `—` | `simid_prospective_open_calibration_opus47` |
| `measurement.simid.prospective_open_calibration.opus47.gwet_ac1` | 0.883 | 150 | `—` | `simid_prospective_open_calibration_opus47` |
| `measurement.simid.prospective_open_calibration.opus47.raw_agreement` | 92.0% | 150 | `[86.5, 95.4]%` | `simid_prospective_open_calibration_opus47` |
| `measurement.simid.prospective_open_calibration.opus47.rule_gap_count` | 0 | 150 | `—` | `simid_prospective_open_calibration_opus47` |

### Evaluator Validation

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `measurement.csv2_v3.dev_accuracy` | 86.5% | 74 | `[76.9, 92.5]%` | `csv2_v3_results` |
| `measurement.holdout.binary.accuracy` | 90.0% | 50 | `[80.0, 98.0]%` | `holdout_comparison` |
| `measurement.holdout.csv2v2.accuracy` | 92.0% | 50 | `[84.3, 98.0]%` | `holdout_comparison` |
| `measurement.holdout.csv2v3.accuracy` | 96.0% | 50 | `[90.0, 100.0]%` | `holdout_comparison` |
| `measurement.holdout.csv2v3_vs_sr_discordant_count` | 0 | 50 | `—` | `holdout_comparison` |
| `measurement.holdout.csv2v3_vs_sr_mcnemar_p` | 1.000 | 50 | `—` | `holdout_comparison` |
| `measurement.holdout.min_pairwise_mcnemar_p` | 0.250 | 50 | `—` | `holdout_comparison` |
| `measurement.holdout.sample_size` | 50 | 50 | `—` | `holdout_comparison` |
| `measurement.holdout.sr.accuracy` | 96.0% | 50 | `[90.0, 100.0]%` | `holdout_comparison` |
| `measurement.strongreject.audit.false_negatives_recovered_after_upgrade` | 3 | 19 | `—` | `jailbreak_measurement_cleanup_audit` |
| `measurement.strongreject.audit.gpt4o_dev_accuracy_after_rerun` | 78.4% | 74 | `[67.7, 86.2]%` | `jailbreak_measurement_cleanup_audit` |
| `measurement.strongreject.audit.gpt4o_mini_dev_accuracy_prior` | 74.3% | 74 | `[63.3, 82.9]%` | `evaluator_4way_audit` |
| `measurement.strongreject.audit.holdout_gap_after_model_upgrade_pp` | +0.0 pp | 50 | `—` | `jailbreak_measurement_cleanup_audit` |
| `measurement.strongreject.audit.persistent_false_negatives_after_upgrade` | 16 | 19 | `—` | `jailbreak_measurement_cleanup_audit` |
| `measurement.strongreject.dev_accuracy` | 78.4% | 74 | `[67.7, 86.2]%` | `strongreject_results` |

### Bridge Adjudication

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `measurement.bridge_irr.cohen_kappa` | 0.899 | 57 | `—` | `bridge_irr_summary` |
| `measurement.bridge_irr.gwet_ac1` | 0.960 | 57 | `—` | `bridge_irr_summary` |
| `measurement.bridge_irr.n_disagreements` | 2 | 57 | `—` | `bridge_irr_summary` |
| `measurement.bridge_irr.raw_agreement` | 96.5% | 57 | `[88.1, 99.0]%` | `bridge_irr_summary` |
| `measurement.bridge_irr.right_to_wrong.answer_dilution` | 7.0% | 43 | `[2.4, 18.6]%` | `bridge_irr_summary` |
| `measurement.bridge_irr.right_to_wrong.evasion_or_factual_denial` | 20.9% | 43 | `[11.4, 35.2]%` | `bridge_irr_summary` |
| `measurement.bridge_irr.right_to_wrong.formal_refusal` | 0.0% | 43 | `[0.0, 8.2]%` | `bridge_irr_summary` |
| `measurement.bridge_irr.right_to_wrong.wrong_entity_substitution` | 72.1% | 43 | `[57.3, 83.3]%` | `bridge_irr_summary` |

## Transfer / Externality Surfaces

### TruthfulQA Multiple Choice

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `transfer.truthfulqa_mc.mc1.h_neuron.baseline_accuracy` | 25.8% | 655 | `—` | `truthfulqa_mc1_h_fold0_alpha0, truthfulqa_mc1_h_fold1_alpha0, truthfulqa_mc1_h_fold0_alpha1, truthfulqa_mc1_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc1.h_neuron.best_accuracy` | 26.7% | 655 | `—` | `truthfulqa_mc1_h_fold0_alpha0, truthfulqa_mc1_h_fold1_alpha0, truthfulqa_mc1_h_fold0_alpha1, truthfulqa_mc1_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc1.h_neuron.delta_pp` | +0.9 pp | 655 | `[-1.7, +3.5]` | `truthfulqa_mc1_h_fold0_alpha0, truthfulqa_mc1_h_fold1_alpha0, truthfulqa_mc1_h_fold0_alpha1, truthfulqa_mc1_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc1.h_neuron.right_to_wrong` | 33 | 655 | `—` | `truthfulqa_mc1_h_fold0_alpha0, truthfulqa_mc1_h_fold1_alpha0, truthfulqa_mc1_h_fold0_alpha1, truthfulqa_mc1_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc1.h_neuron.wrong_to_right` | 39 | 655 | `—` | `truthfulqa_mc1_h_fold0_alpha0, truthfulqa_mc1_h_fold1_alpha0, truthfulqa_mc1_h_fold0_alpha1, truthfulqa_mc1_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc1.iti.baseline_accuracy` | 26.7% | 655 | `—` | `truthfulqa_mc1_iti_fold0_alpha0, truthfulqa_mc1_iti_fold1_alpha0, truthfulqa_mc1_iti_fold0_alpha8, truthfulqa_mc1_iti_fold1_alpha8` |
| `transfer.truthfulqa_mc.mc1.iti.best_accuracy` | 33.0% | 655 | `—` | `truthfulqa_mc1_iti_fold0_alpha0, truthfulqa_mc1_iti_fold1_alpha0, truthfulqa_mc1_iti_fold0_alpha8, truthfulqa_mc1_iti_fold1_alpha8` |
| `transfer.truthfulqa_mc.mc1.iti.delta_pp` | +6.3 pp | 655 | `[+3.7, +9.0]` | `truthfulqa_mc1_iti_fold0_alpha0, truthfulqa_mc1_iti_fold1_alpha0, truthfulqa_mc1_iti_fold0_alpha8, truthfulqa_mc1_iti_fold1_alpha8` |
| `transfer.truthfulqa_mc.mc1.iti.right_to_wrong` | 20 | 655 | `—` | `truthfulqa_mc1_iti_fold0_alpha0, truthfulqa_mc1_iti_fold1_alpha0, truthfulqa_mc1_iti_fold0_alpha8, truthfulqa_mc1_iti_fold1_alpha8` |
| `transfer.truthfulqa_mc.mc1.iti.wrong_to_right` | 61 | 655 | `—` | `truthfulqa_mc1_iti_fold0_alpha0, truthfulqa_mc1_iti_fold1_alpha0, truthfulqa_mc1_iti_fold0_alpha8, truthfulqa_mc1_iti_fold1_alpha8` |
| `transfer.truthfulqa_mc.mc2.h_neuron.baseline_accuracy` | 43.0% | 655 | `—` | `truthfulqa_mc2_h_fold0_alpha0, truthfulqa_mc2_h_fold1_alpha0, truthfulqa_mc2_h_fold0_alpha1, truthfulqa_mc2_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc2.h_neuron.best_accuracy` | 42.9% | 655 | `—` | `truthfulqa_mc2_h_fold0_alpha0, truthfulqa_mc2_h_fold1_alpha0, truthfulqa_mc2_h_fold0_alpha1, truthfulqa_mc2_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc2.h_neuron.delta_pp` | -0.1 pp | 655 | `[-2.2, +2.0]` | `truthfulqa_mc2_h_fold0_alpha0, truthfulqa_mc2_h_fold1_alpha0, truthfulqa_mc2_h_fold0_alpha1, truthfulqa_mc2_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc2.h_neuron.right_to_wrong` | 0 | 655 | `—` | `truthfulqa_mc2_h_fold0_alpha0, truthfulqa_mc2_h_fold1_alpha0, truthfulqa_mc2_h_fold0_alpha1, truthfulqa_mc2_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc2.h_neuron.wrong_to_right` | 0 | 655 | `—` | `truthfulqa_mc2_h_fold0_alpha0, truthfulqa_mc2_h_fold1_alpha0, truthfulqa_mc2_h_fold0_alpha1, truthfulqa_mc2_h_fold1_alpha1` |
| `transfer.truthfulqa_mc.mc2.iti.baseline_accuracy` | 42.9% | 655 | `—` | `truthfulqa_mc2_iti_fold0_alpha0, truthfulqa_mc2_iti_fold1_alpha0, truthfulqa_mc2_iti_fold0_alpha8, truthfulqa_mc2_iti_fold1_alpha8` |
| `transfer.truthfulqa_mc.mc2.iti.best_accuracy` | 50.4% | 655 | `—` | `truthfulqa_mc2_iti_fold0_alpha0, truthfulqa_mc2_iti_fold1_alpha0, truthfulqa_mc2_iti_fold0_alpha8, truthfulqa_mc2_iti_fold1_alpha8` |
| `transfer.truthfulqa_mc.mc2.iti.delta_pp` | +7.5 pp | 655 | `[+5.3, +9.8]` | `truthfulqa_mc2_iti_fold0_alpha0, truthfulqa_mc2_iti_fold1_alpha0, truthfulqa_mc2_iti_fold0_alpha8, truthfulqa_mc2_iti_fold1_alpha8` |
| `transfer.truthfulqa_mc.mc2.iti.right_to_wrong` | 0 | 655 | `—` | `truthfulqa_mc2_iti_fold0_alpha0, truthfulqa_mc2_iti_fold1_alpha0, truthfulqa_mc2_iti_fold0_alpha8, truthfulqa_mc2_iti_fold1_alpha8` |
| `transfer.truthfulqa_mc.mc2.iti.wrong_to_right` | 0 | 655 | `—` | `truthfulqa_mc2_iti_fold0_alpha0, truthfulqa_mc2_iti_fold1_alpha0, truthfulqa_mc2_iti_fold0_alpha8, truthfulqa_mc2_iti_fold1_alpha8` |

### Mistral TruthfulQA

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `transfer.mistral24b_anchor2.truthfulqa_mc1.baseline_rate` | 35.6% | 163 | `[28.6, 43.2]%` | `mistral24b_anchor2_truthfulqa_mc1_report` |
| `transfer.mistral24b_anchor2.truthfulqa_mc1.delta_alpha4_minus_alpha0_pp` | +1.2 pp | 163 | `[-1.8, +4.3]` | `mistral24b_anchor2_truthfulqa_mc1_report` |
| `transfer.mistral24b_anchor2.truthfulqa_mc1.intervened_rate` | 36.8% | 163 | `[29.8, 44.4]%` | `mistral24b_anchor2_truthfulqa_mc1_report` |
| `transfer.mistral24b_anchor2.truthfulqa_mc1.mcnemar_p` | 0.688 | 163 | `—` | `mistral24b_anchor2_truthfulqa_mc1_report` |
| `transfer.mistral24b_anchor2.truthfulqa_mc2.baseline_rate` | 55.1% | 163 | `[48.6, 61.3]%` | `mistral24b_anchor2_truthfulqa_mc2_report` |
| `transfer.mistral24b_anchor2.truthfulqa_mc2.delta_alpha4_minus_alpha0_pp` | +1.9 pp | 163 | `[+0.5, +3.4]` | `mistral24b_anchor2_truthfulqa_mc2_report` |
| `transfer.mistral24b_anchor2.truthfulqa_mc2.intervened_rate` | 57.0% | 163 | `[50.5, 63.3]%` | `mistral24b_anchor2_truthfulqa_mc2_report` |

### SIMID Transfer

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `transfer.simid_mvp.random_direction_seed1.pooled.adjudicated_open_delta_alpha8_pp` | +3.0 pp | 200 | `[-1.0, +7.0]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.random_head_seed1.pooled.adjudicated_open_delta_alpha8_pp` | -0.5 pp | 200 | `[-4.5, +3.0]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected.bridge.adjudicated_open_correct_delta_alpha8_pp` | -6.0 pp | 100 | `[-12.0, +0.0]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected.bridge.mc_letter_likelihood_correct_delta_alpha8_pp` | -1.0 pp | 100 | `[-4.0, +1.5]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected.pooled.adjudicated_open_baseline` | 38.5% | 200 | `[32.0, 45.5]%` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected.pooled.adjudicated_open_delta_alpha8_pp` | +3.0 pp | 200 | `[-2.0, +8.5]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected.pooled.mc_letter_baseline` | 58.2% | 200 | `[52.0, 64.5]%` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected.pooled.mc_letter_delta_alpha8_pp` | -2.2 pp | 200 | `[-4.5, +0.0]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected.truthfulqa.adjudicated_open_correct_delta_alpha8_pp` | +12.0 pp | 100 | `[+4.0, +20.0]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected.truthfulqa.mc_letter_likelihood_correct_delta_alpha8_pp` | -3.5 pp | 100 | `[-7.0, -0.5]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected_vs_random_direction.open_first3_margin_slope_gap` | +0.23 | 200 | `[0.071, 0.390]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_mvp.selected_vs_random_head.open_first3_margin_slope_gap` | +0.12 | 200 | `[-0.026, 0.271]` | `simid_mvp_results_adjudicated` |
| `transfer.simid_prospective_partial.selected_truthfulqa.external_open_attempted_alpha0_rate` | 92.5% | 454 | `—` | `simid_prospective_partial_effect_early_look` |
| `transfer.simid_prospective_partial.selected_truthfulqa.external_open_attempted_alpha8_rate` | 83.9% | 454 | `—` | `simid_prospective_partial_effect_early_look` |
| `transfer.simid_prospective_partial.selected_truthfulqa.external_open_attempted_delta_alpha8_pp` | -8.6 pp | 454 | `[-13.4, -3.7]` | `simid_prospective_partial_effect_early_look` |
| `transfer.simid_prospective_partial.selected_truthfulqa.external_open_correct_alpha0_rate` | 34.1% | 454 | `—` | `simid_prospective_partial_effect_early_look` |
| `transfer.simid_prospective_partial.selected_truthfulqa.external_open_correct_alpha8_rate` | 38.1% | 454 | `—` | `simid_prospective_partial_effect_early_look` |
| `transfer.simid_prospective_partial.selected_truthfulqa.external_open_correct_delta_alpha8_pp` | +4.0 pp | 454 | `[-2.0, +9.9]` | `simid_prospective_partial_effect_early_look` |
| `transfer.simid_prospective_partial.selected_truthfulqa.mc_letter_delta_alpha8_pp` | -3.1 pp | 454 | `[-5.7, -0.7]` | `simid_prospective_partial_effect_early_look, simid_prospective_partial_private_map, simid_prospective_selected_alpha0_rows, simid_prospective_selected_alpha8_rows` |
| `transfer.simid_prospective_partial.selected_truthfulqa.paired_sample_count` | 454 | 908 | `—` | `simid_prospective_partial_effect_early_look` |
| `transfer.simid_prospective_partial.selected_truthfulqa.rule_gap_count` | 0 | 908 | `—` | `simid_prospective_partial_effect_early_look` |

### SimpleQA

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `transfer.simpleqa.attempt_delta_0_to_8_pp` | -32.7 pp | 1000 | `[-35.6, -29.8]` | `simpleqa_alpha_0_rows, simpleqa_alpha_8_rows` |
| `transfer.simpleqa.attempt_rate_alpha_0` | 99.7% | 1000 | `—` | `simpleqa_alpha_0_rows` |
| `transfer.simpleqa.attempt_rate_alpha_8` | 67.0% | 1000 | `—` | `simpleqa_alpha_8_rows` |
| `transfer.simpleqa.compliance.0.0` | 4.6% | 1000 | `[3.5, 6.1]%` | `simpleqa_ranked_results` |
| `transfer.simpleqa.compliance.4.0` | 3.7% | 1000 | `[2.7, 5.1]%` | `simpleqa_ranked_results` |
| `transfer.simpleqa.compliance.8.0` | 2.8% | 1000 | `[1.9, 4.0]%` | `simpleqa_ranked_results` |
| `transfer.simpleqa.compliance_delta_0_to_8_pp` | -1.8 pp | 1000 | `[-3.1, -0.5]` | `simpleqa_ranked_results` |

### Bridge Outcome Surface

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `transfer.bridge.adjudicated_accuracy_delta_pp` | -5.8 pp | 500 | `[-8.8, -3.0]` | `bridge_phase3_site` |
| `transfer.bridge.attempt_rate_delta_pp` | -1.2 pp | 500 | `[-2.4, -0.2]` | `bridge_phase3_site` |
| `transfer.bridge.base_correct_iti_wrong` | 43 | 500 | `—` | `bridge_phase3_site` |
| `transfer.bridge.base_wrong_iti_correct` | 14 | 500 | `—` | `bridge_phase3_site` |
| `transfer.bridge.baseline_adjudicated_accuracy` | 45.0% | 500 | `[40.7, 49.4]%` | `bridge_phase3_site` |
| `transfer.bridge.baseline_not_attempted_count` | 2 | 500 | `—` | `bridge_phase3_site` |
| `transfer.bridge.deterministic_accuracy_delta_pp` | -4.6 pp | 500 | `[-7.6, -1.6]` | `bridge_phase3_site` |
| `transfer.bridge.iti_not_attempted_count` | 8 | 500 | `—` | `bridge_phase3_site` |
| `transfer.bridge.mcnemar_p` | 2.00e-04 | 500 | `—` | `bridge_phase3_site` |
| `transfer.bridge.sample_size` | 500 | 500 | `—` | `bridge_phase3_site` |

### Bridge Margin Analysis

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `transfer.bridge_margins.a_rw_substitution.first3_shift_nats` | -10.16 | 31 | `[-12.812, -7.596]` | `bridge_margins_test_results` |
| `transfer.bridge_margins.a_vs_d.first3_shift_nats_gap` | -5.10 | 231 | `[-8.216, -2.161]` | `bridge_margins_test_results` |
| `transfer.bridge_margins.a_vs_d.first3_shift_nats_p` | 0.008 | 231 | `—` | `bridge_margins_test_results` |
| `transfer.bridge_margins.b_rw_nonsubstitution.first3_shift_nats` | -40.06 | 12 | `[-50.088, -30.371]` | `bridge_margins_test_results` |
| `transfer.bridge_margins.c_wr_rescue.first3_shift_nats` | +4.73 | 14 | `[2.272, 7.251]` | `bridge_margins_test_results` |
| `transfer.bridge_margins.d_random_control.first3_shift_nats` | -5.05 | 200 | `[-6.550, -3.599]` | `bridge_margins_test_results` |
| `transfer.bridge_margins.sample_size` | 257 | 257 | `—` | `bridge_margins_test_results` |

## Mechanism / Diagnostic Surfaces

### D7 Current Panel

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `mechanism.d7.current_panel.baseline.strict_harmfulness_normalized` | 51.6% | 500 | `[47.2, 56.0]%` | `d7_current_state_summary, d7_comparison_site` |
| `mechanism.d7.current_panel.causal.strict_harmfulness_normalized` | 24.8% | 500 | `[21.2, 28.8]%` | `d7_current_state_summary, d7_comparison_site` |
| `mechanism.d7.current_panel.causal_token_cap_count` | 112 | 500 | `—` | `d7_current_state_summary, d7_comparison_site` |
| `mechanism.d7.current_panel.causal_vs_probe_gap_pp` | -10.0 pp | 500 | `[-14.0, -6.2]` | `d7_current_state_summary, d7_comparison_site` |
| `mechanism.d7.current_panel.causal_vs_random_layer_seed1_gap_pp` | -12.4 pp | 500 | `[-16.8, -8.0]` | `d7_current_state_summary, d7_comparison_site` |
| `mechanism.d7.current_panel.causal_vs_random_layer_seed2_gap_pp` | -13.6 pp | 500 | `[-17.8, -9.6]` | `d7_current_state_summary, d7_comparison_site` |
| `mechanism.d7.current_panel.probe.strict_harmfulness_normalized` | 34.8% | 500 | `[30.8, 39.1]%` | `d7_current_state_summary, d7_comparison_site` |
| `mechanism.d7.current_panel.random_layer_seed1.strict_harmfulness_normalized` | 37.2% | 500 | `[33.1, 41.5]%` | `d7_current_state_summary, d7_comparison_site` |
| `mechanism.d7.current_panel.random_layer_seed2.strict_harmfulness_normalized` | 38.4% | 500 | `[34.2, 42.7]%` | `d7_current_state_summary, d7_comparison_site` |

### D7 Pilot Audit

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `mechanism.d7.pilot.gradient_best_effect_pp` | -13.0 pp | 100 | `[-21.0, -6.0]` | `d7_causal_pilot_audit` |
| `mechanism.d7.pilot.probe_best_effect_pp` | -2.0 pp | 100 | `[-10.0, +6.0]` | `d7_causal_pilot_audit` |
| `mechanism.d7.pilot.probe_causal_jaccard` | 11.0% | 36 | `—` | `d7_causal_pilot_audit` |

### Overlap / Swing

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `mechanism.refusal_overlap.faitheval.canonical_overlap_vs_primary` | -0.09 | 1000 | `[-0.160, -0.012]` | `refusal_overlap_summary` |
| `mechanism.refusal_overlap.faitheval.subspace_overlap_vs_primary` | 0.09 | 1000 | `[0.011, 0.159]` | `refusal_overlap_summary` |
| `mechanism.refusal_overlap.jailbreak.canonical_overlap_vs_primary` | -0.12 | 500 | `[-0.206, -0.024]` | `refusal_overlap_summary` |
| `mechanism.refusal_overlap.jailbreak.subspace_overlap_vs_primary` | 0.11 | 500 | `[0.022, 0.202]` | `refusal_overlap_summary` |
| `mechanism.swing.population.swing_count` | 138 | 1000 | `—` | `swing_characterization_site` |
| `mechanism.swing.subtype.c_to_r` | 23.2% | 138 | `[16.9, 30.9]%` | `swing_characterization_site` |
| `mechanism.swing.subtype.non_monotonic` | 8.7% | 138 | `[5.0, 14.6]%` | `swing_characterization_site` |
| `mechanism.swing.subtype.r_to_c` | 68.1% | 138 | `[59.9, 75.3]%` | `swing_characterization_site` |

### Single-Neuron / Verbosity

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `mechanism.neuron_4288.ablation_accuracy_drop` | +1.0 pp | — | `—` | `neuron_4288_summary` |
| `mechanism.neuron_4288.distribution_separation` | 0.326 | — | `—` | `neuron_4288_summary` |
| `mechanism.neuron_4288.max_top10_correlation` | 0.492 | — | `—` | `neuron_4288_summary` |
| `mechanism.neuron_4288.single_neuron_auc` | 0.590 | — | `—` | `neuron_4288_summary` |
| `mechanism.neuron_4288.weight` | 12.169 | — | `—` | `neuron_4288_summary` |
| `mechanism.verbosity.length_effect_d` | -1.864 | — | `—` | `verbosity_confound_summary` |
| `mechanism.verbosity.length_mean_diff` | -0.0070 | — | `[-0.008, -0.006]` | `verbosity_confound_summary` |
| `mechanism.verbosity.truth_effect_d` | -0.502 | — | `—` | `verbosity_confound_summary` |
| `mechanism.verbosity.truth_mean_diff` | -0.0012 | — | `[-0.002, -0.001]` | `verbosity_confound_summary` |

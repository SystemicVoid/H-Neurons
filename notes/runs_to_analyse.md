# Runs to Analyse

This file is a queue for runs that still need analysis. Any analysed runs must be fully removed.

Currently empty.

## 2026-04-22T13:59:39+00:00 | data/gemma3_4b/intervention/faitheval_sae_utility_selector
What: FaithEval SAE utility-selector ablation + held-out bundle (anti-compliance, delta-only, validation-selected/readout-selected/matched-random)
Key files: selector/selector_summary.json, heldout/*/*/alpha_*.jsonl, report/heldout_summary.json, *.provenance.json
Status: awaiting analysis

## 2026-04-22T14:27:53+00:00 | data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment
What: FaithEval SAE utility-positive augment bundle (k=154 positive-only, 3 layer-matched zero-weight seeds × 2 benchmarks, α=0.0; noop α=1.0 reused from Phase 1)
Key files: selector/utility_positive_*.json, selector/matched_random_positive_*.json, heldout/*/utility_positive_selected/experiment/alpha_0.0.jsonl, heldout/*/matched_random_positive_*/experiment/alpha_0.0.jsonl, report_augment/augment_heldout_summary.json, report_augment/augment_audit_note.md, *.provenance.json
Status: awaiting analysis

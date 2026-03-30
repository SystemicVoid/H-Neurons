# Runs to Analyse

This file is only for long-running jobs that still need analysis.

## 2026-03-30T14:19:52+01:00 | data/gemma3_4b/intervention/faitheval_direction_ablate_d4_all_layers_calibrated_clean/experiment
What: D4 kill-shot all-layer truthfulness-direction ablation on FaithEval, beta=[0.0,0.01,0.02] after 20-sample clean calibration at beta=[0.005,0.01,0.02,0.05]
Key files: results.20260330_131952.json, alpha_*.jsonl, run_intervention.provenance.20260330_131952.json
Status: done: D4 survives kill-shot — β=0.01 gives a clean +5.4pp lift (66.0%→71.4%) with zero parse failures; β=0.02 is the onset cliff (46.2%, 6 parse fails, strong D)-style output drift

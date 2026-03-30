# Runs to Analyse

This file is only for long-running jobs that still need analysis.

## 2026-03-30T14:19:52+01:00 | data/gemma3_4b/intervention/faitheval_direction_ablate_d4_all_layers_calibrated_clean/experiment
What: D4 kill-shot all-layer truthfulness-direction ablation on FaithEval, beta=[0.0,0.01,0.02] after 20-sample clean calibration at beta=[0.005,0.01,0.02,0.05]
Key files: results.20260330_131952.json, alpha_*.jsonl, run_intervention.provenance.20260330_131952.json
Status: done: D4 survives kill-shot — β=0.01 gives a clean +5.4pp lift (66.0%→71.4%) with zero parse failures; β=0.02 is the onset cliff (46.2%, 6 parse fails, strong D)-style output drift

## 2026-03-30T16:53:08+01:00 | data/gemma3_4b/intervention/faitheval_iti_head_truthfulness_triviaqa_calibration/experiment
What: FaithEval calibration for head-level truthfulness ITI (triviaqa_transfer, top-k=16 ranked heads, alpha=[0.0,0.1,0.5,1.0,2.0]) after fresh extraction
Key files: results.*.json, alpha_*.jsonl, run_intervention.provenance.*.json, data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt
Status: done: hard null on the 20-sample FaithEval calibration — all five alphas produced the same 17/20 one-token outputs with zero parse failures; extraction succeeded cleanly, but this ITI configuration has not yet earned promotion to broader runs

## 2026-03-30T17:38:23+01:00 | data/gemma3_4b/intervention/faitheval_iti_head_truthfulness_context_grounded_calibration/experiment
What: FaithEval calibration for head-level truthfulness ITI (context_grounded, top-k=16 ranked heads, alpha=[0.0,0.1,0.5,1.0,2.0]) after fresh extraction
Key files: results.*.json, alpha_*.jsonl, run_intervention.provenance.*.json, data/contrastive/truthfulness/iti_context/iti_heads.pt
Status: done: second hard null on the 20-sample FaithEval calibration — the context-grounded ITI artifact is extremely separable in extraction, but α up to 2.0 changed neither answers nor compliance on any row despite large decode-time head deltas

## 2026-03-30T21:13:02+01:00 | data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paper_k-16_ranked_seed-42_iti-truthfulqa-paper-iti-heads_98dbc66c05/experiment
What: Phase 4 TruthfulQA MC1 eval with paper-faithful artifact, K=16, alpha=[0.0,1.0,2.0,4.0,8.0,12.0,16.0], 163 held-out questions
Key files: results.20260330_201302.json, alpha_*.jsonl, run_intervention.provenance.20260330_201302.json
Status: done: +4.3pp MC1 lift at α=2-4 (29.4%→33.7%), CI barely excludes zero at α=1-2

## 2026-03-30T21:39:49+01:00 | data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paper_k-16_ranked_seed-42_iti-truthfulqa-paper-iti-heads_98dbc66c05/experiment
What: Phase 4 TruthfulQA MC2 eval with paper-faithful artifact, K=16, alpha=[0.0,1.0,2.0,4.0,8.0,12.0,16.0], 163 held-out questions
Key files: results.20260330_203949.json, alpha_*.jsonl
Status: done: clear MC2 signal — +8.5pp at α=8 (0.434→0.519), CI excludes zero at all alphas, monotonic climb to α=8-12 plateau

## 2026-03-30T22:17:05+01:00 | data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_triviaqa-transfer_k-16_ranked_seed-42_iti-triviaqa-iti-heads_bc586bf93e/experiment
What: Phase 4 baseline — TriviaQA-transfer artifact on TruthfulQA MC1, K=16, alpha=[0.0,1.0,2.0,4.0,8.0], 163 held-out questions
Key files: results.20260330_211705.json, alpha_*.jsonl
Status: done: flat null — 29.4% baseline, no movement at any alpha, confirms paper-faithful extraction matters

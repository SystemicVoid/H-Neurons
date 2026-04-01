# Runs to Analyse

This file is only for long-running jobs that still need analysis.

## 2026-04-01T18:38:59Z | data/gemma3_4b/intervention/truthfulqa_mc_{mc1,mc2}_h-neurons_final-fold{0,1}/experiment
What: D1 H-neuron TruthfulQA MC rerun on the final held-out folds, variants={mc1,mc2}, alpha=[0.0,0.5,1.0,1.5,2.0,2.5,3.0]
Key files: results.*.json, alpha_*.jsonl, run_intervention.provenance.*.json
Status: done: pooled best point is weak and not clearly non-zero (MC1 +0.9 pp [-1.7, +3.5], MC2 +0.5 pp [-2.3, +3.4]); D4 remains decisively stronger on the same held-out folds. See notes/act3-reports/2026-04-01-priority-reruns-audit.md.

## 2026-04-01T19:11:51Z | data/gemma3_4b/intervention/simpleqa_factual_phrase_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment
What: SimpleQA standalone — paper-faithful ITI K=12, prompt_style=factual_phrase, alpha=[0.0,4.0,8.0], 1000 verified questions
Key files: results.json, alpha_*.jsonl, run_intervention.provenance.*.json
Status: done: removing the explicit escape hatch cut α=8.0 abstention from 79.2% to 33.0%, but compliance still fell below baseline (4.6%→2.8%) and precision stayed flat within uncertainty; generation null survives prompt repair. See notes/act3-reports/2026-04-01-priority-reruns-audit.md.

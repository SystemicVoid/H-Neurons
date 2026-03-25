# Runs to Analyse

## Pending | data/gemma3_4b/intervention/jailbreak/experiment_1024tok_greedy_incomplete
What: Jailbreak 1024-token audit — JailbreakBench 100×5, α={0.0, 1.0, 2.0, 3.0}, max_new_tokens=1024, GPT-4o judge
Key files: results.json, *.provenance.json, alpha_*.jsonl
Status: superseded by 5000-token canonical rerun (only alpha_1.0 completed)

## 2026-03-24T14:35Z | data/gemma3_4b/intervention/jailbreak/experiment
What: jailbreak × H-neuron scaling, α={0.0,1.5,3.0}, 5000tok, sampled (T=0.7,k=20,p=0.8), seed=42
Key files: alpha_0.0.jsonl, alpha_1.5.jsonl, alpha_3.0.jsonl, *.provenance.json
Status: awaiting analysis

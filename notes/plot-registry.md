# Plot Registry

Data files to pull when generating website charts. Extracted from `week3-log.md` §10.

---

## Baseline A — FaithEval compliance curve (7 alphas)

```
data/gemma3_4b/intervention/faitheval/experiment/results.json
```

Fields: `results[alpha].compliance`

---

## Baseline A — Jailbreak binary compliance (4 alphas)

```
data/gemma3_4b/intervention/jailbreak/experiment/results.json
```

---

## Baseline A — Jailbreak CSV-v2 severity (per-item, 4 alphas)

```
data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_0.0.jsonl
data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_1.0.jsonl
data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_1.5.jsonl
data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_3.0.jsonl
```

Fields: `csv2.harmful_binary`, `csv2.C`, `csv2.S`, `csv2.V`, `csv2.harmful_payload_share`, `csv2.pivot_position`

---

## D2 — Per-layer accuracy / separation curve (34 layers)

```
data/contrastive/refusal/directions/extraction_metadata.json
```

Fields: `separation_scores[].accuracy`, `separation_scores[].separation`

---

## D3 — Refusal-direction ablation compliance curve (3 β values)

```
data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/results.20260329_192019.json
data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_0.0.jsonl
data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_0.02.jsonl
data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_0.03.jsonl
```

---

## D3.5 — Refusal overlap geometry and null distribution

```
data/gemma3_4b/intervention/refusal_overlap/analysis/summary.json
data/gemma3_4b/intervention/refusal_overlap/analysis/layer_scores.csv
data/gemma3_4b/intervention/refusal_overlap/analysis/prompt_scores.csv
data/gemma3_4b/intervention/refusal_overlap/analysis/null_distribution.json
```

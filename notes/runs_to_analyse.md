# Runs to Analyse

## 2026-03-29T14:12 | data/contrastive/refusal/directions
What: D2 refusal direction extraction + sanity check (difference-in-means on 128+128 train, 32+32 val, Gemma-3-4B-IT)
Key files: refusal_directions.pt, extraction_metadata.json, sanity_check_results.json, extract_direction.provenance.*.json
Status: done: Layer 25 best (98.4% val accuracy, separation=9179). Sanity gate passed: ablation removes refusal (25%→0%), D3 cleared. Single-layer degeneration noted; all-layer ablation needed for D3.


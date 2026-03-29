# Runs to Analyse

## 2026-03-29T14:12 | data/contrastive/refusal/directions
What: D2 refusal direction extraction + sanity check (difference-in-means on 128+128 train, 32+32 val, Gemma-3-4B-IT)
Key files: refusal_directions.pt, extraction_metadata.json, sanity_check_results.json, extract_direction.provenance.*.json
Status: done: Layer 25 best (98.4% val accuracy, separation=9179). Sanity gate passed: ablation removes refusal (25%→0%), D3 cleared. Single-layer degeneration noted; all-layer ablation needed for D3. 2026-03-29 clean-provenance reruns reproduced the archived result exactly: eval-set JSONLs matched byte-for-byte, the refusal direction tensor hash matched exactly, and the final extraction provenance is clean at `extract_direction.provenance.20260329_134942.json`. Archived pre-rerun outputs retained under `data/contrastive/refusal/directions_2026-03-29_dirty_provenance_snapshot/` and `data/contrastive/refusal/eval_2026-03-29_pre_clean_rerun/`.

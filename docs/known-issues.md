# Known Issues

## FaithEval Standard-Prompt Mis-Scoring

The MC letter extractor in `evaluate_intervention.py` fails on ~150 items at `alpha=3.0`; a strict answer-text remap recovers most. Treat raw standard-prompt compliance drops as evaluator artifacts until text-based rescoring is wired in.

## `extract_answer_tokens.py` JSON Brittleness

The judge model occasionally returns malformed JSON on short quoted answers (observed on `tc_115` in Mistral canary). Needs retry/fallback hardening.

## Jailbreak Item-Level Analysis Confounded by Stochastic Generation

Per-item flip analysis and cross-benchmark flip comparisons (e.g., to FalseQA's disjoint-subpopulation finding) are invalid because jailbreak uses `do_sample=True, temp=0.7` while FalseQA/FaithEval use greedy decoding. Aggregate endpoint effects remain valid. See `data/gemma3_4b/intervention/jailbreak/jailbreak_interpretive_review.md`.

## SAE Intervention JSONL Schema Inconsistency

The experiment JSONL (from `run_intervention.py --intervention_mode sae`) does not include a `parse_failure` boolean field; parse failures are inferred from `chosen=None`. The negative control JSONL (from `run_sae_negative_control.py`) includes an explicit `parse_failure` field. Both `results.json` files correctly count parse failures. This is a field-naming inconsistency between scripts, not a data error.

# Mistral 24B Replication Plan

**Date:** 2026-04-27

## Decision

Use `mistralai/Mistral-Small-24B-Instruct-2501` as the second-model
replication target. It is a credible same-family 24B extension and matches the
existing `data/mistral24b/` artifacts. It is not an exact replication of a paper
claim that specifically names Mistral Small 3.1 / `*-2503`.

Defer `mistralai/Mistral-Small-3.1-24B-Instruct-2503` until the pipeline has a
proper `Mistral3ForConditionalGeneration` / processor loader path. The current
canonical scripts intentionally support causal-LM text checkpoints only.

## Model Facts

For the canonical 2501 checkpoint:

- HF id: `mistralai/Mistral-Small-24B-Instruct-2501`
- Registry key: `mistral_small_24b_instruct_2501`
- Output root: `data/mistral24b`
- Loader family: `causal_lm`
- Tokenizer kwargs: `fix_mistral_regex=True`
- Layers: 40
- Hidden size: 5120
- FFN intermediate size: 32768
- Attention heads: 32
- KV heads: 8
- Context: 32768 tokens
- Weight dtype: BF16

## Canonical Path

1. Reuse `data/mistral24b/answer_tokens_llm.jsonl`.
2. Build disjoint train/dev/test splits with `scripts/sample_balanced_ids.py`.
3. Extract CETT activations for train answer tokens and non-answer tokens.
4. Extract answer-token activations for dev/test.
5. Train an L1 sparse 3-vs-1 classifier, select `C` on dev AUROC, then evaluate
   the saved classifier on held-out test.
6. Run FaithEval standard first under `data/mistral24b/intervention/faitheval`.
7. Run negative controls from the same classifier geometry: unconstrained random
   and layer-matched random, both using model config-derived dimensions.
8. Run FalseQA canaries after FaithEval controls pass. Add sycophancy and
   jailbreak only after the basic specificity checks are clean.

The wrapper is `scripts/infra/mistral24b_replication.sh`. It follows the repo
long-run pattern: tmux, `systemd-inhibit`, `set -euo pipefail`, GPU preflight,
`nvitop -1`, active-run guard checks, `device_map=cuda:0`, uv execution, and
environment/version capture. Use `DRY_RUN=1` to print the command sequence
without launching GPU work, or `STAGES=splits,activations,classifier` /
`STAGES=faitheval_controls` to resume explicit phases.

Safety guardrails added during migration review:

- Unregistered model paths must use explicit output directories; the registry
  no longer invents a data root silently.
- Default classifier paths come from the registry.
- Classifier/activation feature widths must match the active model's FFN
  geometry before classifier evaluation, interventions, or controls proceed.
- Quick negative-control runs include a layer-matched seed in addition to
  unconstrained random seeds.

## Progress Status

The codebase is ready for a canonical 2501 RunPod execution path, but no new
24B GPU artifacts have been generated locally. Treat these as the next required
claim-bearing outputs:

- `data/mistral24b/pipeline/train_qids_llm.json`,
  `dev_qids_llm.json`, and `test_qids_llm.json`.
- Canonical train/dev/test activation directories under
  `data/mistral24b/pipeline/activations_llm_canonical/`.
- `models/mistral24b_classifier_canonical.pkl`.
- `data/mistral24b/pipeline/classifier_canonical_dev_metrics.json` and
  `classifier_canonical_test_metrics.json`.
- FaithEval standard H-neuron outputs plus unconstrained random and
  layer-matched controls.
- FalseQA canary outputs and judge results.

## Acceptance

- `data/mistral24b/pipeline/classifier_canonical_test_metrics.json` exists with
  provenance and held-out metrics.
- At least one judged Mistral benchmark has H-neuron, random, and layer-matched
  control outputs.
- A third model requires a `scripts/model_registry.py` entry and wrapper
  configuration, not edits across every core script.

## Sources

- [2501 Instruct config](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/blob/main/config.json)
- [2501 Base](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501)
- [2503 Instruct config](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json)
- [2503 Base](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Base-2503)

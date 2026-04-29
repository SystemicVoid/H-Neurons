# Mistral 24B Replication Plan

> Superseded for Mistral execution planning and progress tracking by
> `notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md`.
> Retained for historical replication-plan context only. The CP2/CP3 held-out
> detector gate is now analysed at
> `notes/icml/reports/2026-04-29-mistral24b-cp23-pipeline-review.md`.

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

Post-review hardening (2026-04-27, commits `f395b1b`, `759dbe8`, `98f1a9c`,
`f6a8ae1`):

- FaithEval negative control accepts `--prompt_style` and the wrapper threads
  `--prompt_style standard` to both the H-neuron baseline and its controls, so
  the comparison summary measures H-neuron specificity rather than prompt-style
  divergence.
- Negative control accepts `--max_samples` and `--sample_manifest`, mirroring
  `run_intervention.py` semantics; the wrapper passes
  `--max_samples "${INTERVENTION_MAX_SAMPLES}"` to FaithEval and FalseQA
  controls so baseline and controls share a sample population.
- `scripts/classifier.py`'s `--save_model` default now resolves from the
  registry when `--model_key` is set (`models/mistral24b_classifier.pkl` for
  the canonical 2501 entry), so standalone training composes with
  `run_intervention.py` without a manual path override. Legacy unkeyed runs
  still default to `models/detector.pkl`.
- Jailbreak negative control's analyzer-hint printout derives the
  `--experiment_dir` from `--h_neuron_baseline` when set and falls back to
  the registry, with a placeholder for unregistered models — so a successful
  generation run no longer crashes on an informational print when an
  unregistered model is paired with `--output_base`.

## Progress Status

The codebase is ready for a canonical 2501 RunPod execution path. Wrapper and
negative-control plumbing have been hardened post-review (see post-review
hardening above): FaithEval prompt style, sample-set selection, classifier
save-path defaults, and the jailbreak analyzer-path printout are now consistent
between the H-neuron baseline and its controls. No new 24B GPU artifacts have
been generated locally yet. Treat these as the next required claim-bearing
outputs:

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

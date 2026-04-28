# New Model Migration Report

> Superseded for Mistral execution planning and progress tracking by
> `notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md`.
> Retained for historical code-migration context only.

**Date:** 2026-04-27
**Scope:** Codebase changes that make the H-neuron pipeline model-aware for the
Mistral 24B second-model replication path.

## Summary

The migration moved the core H-neuron pipeline away from Gemma-specific defaults
and toward an explicit model registry. The canonical second model is
`mistralai/Mistral-Small-24B-Instruct-2501`, registered as
`mistral_small_24b_instruct_2501`, with outputs rooted at `data/mistral24b`.

This is a same-family 24B replication target, not an exact paper-checkpoint
replication if the relevant paper claim specifically means Mistral Small 3.1 /
`*-2503`.

## Progress Snapshot

Status as of 2026-04-27:

- **Code migration:** complete for the canonical 2501 text-only causal-LM path.
- **Review hardening:** complete for model-root inference, classifier-path
  defaults, classifier/activation width checks, and quick layer-matched controls.
- **Post-review hardening:** complete for negative-control comparability —
  matching FaithEval prompt style, matching sample selection (`--max_samples` /
  `--sample_manifest`), registry-aware classifier `--save_model` default, and
  the jailbreak analyzer-hint printout (commits `f395b1b`, `759dbe8`,
  `98f1a9c`, `f6a8ae1`).
- **Run orchestration:** ready to dry-run or resume by stage through
  `scripts/infra/mistral24b_replication.sh`.
- **GPU artifacts:** not yet produced in this workspace. The canonical held-out
  classifier metrics and judged intervention/control outputs remain pending.
- **Exact 2503 paper-checkpoint support:** intentionally deferred until a
  Mistral3 loader/processor path exists.

## What Changed

- Added `scripts/model_registry.py`.
  It records model slug, HF id, output root, tokenizer kwargs, loader family, and
  known config dimensions.
- Registered:
  - `gemma3_4b_it`
  - `mistral_small_24b_instruct_2501`
  - `mistral_small_31_24b_instruct_2503` as unsupported for canonical causal-LM
    runs.
- Updated `collect_responses.py`, `extract_answer_tokens.py`,
  `extract_activations.py`, `classifier.py`, `run_intervention.py`, and
  `run_negative_control.py` to accept `--model_key` and resolve defaults through
  the registry.
- Mistral 2501 tokenizers now load with `fix_mistral_regex=True`.
- Classifier metrics now include registered model metadata and dimensions when
  available.
- Classifier training, classifier evaluation, neuron interventions, and
  negative controls now fail fast when the classifier/activation feature width
  does not match the active model's registered FFN geometry.
- Intervention and negative-control scripts now derive the default classifier
  path from the model registry, so Mistral runs do not silently fall back to the
  Gemma classifier.
- Negative controls now derive layer count, FFN width, target count, and
  layer-matched sampling from classifier weights plus model dimensions, not
  Gemma constants.
- Quick negative-control runs now include a layer-matched seed as well as
  unconstrained random seeds.
- `sample_balanced_ids.py` now accepts multiple `--exclude_path` values so
  train/dev/test splits can be kept disjoint.
- Added `scripts/infra/mistral24b_replication.sh`, a RunPod-oriented wrapper
  using tmux, `systemd-inhibit`, active-run guards, `nvitop -1`, explicit
  `device_map=cuda:0`, uv commands, environment capture, stage selection, and
  dry-run support.
- Added the Mistral replication plan, now at
  `notes/icml/mistral24b/2026-04-28-replication-plan.md`.
- Corrected the Mistral infra assessment, now at
  `notes/icml/mistral24b/2026-04-27-runpod-infra-assessment.md`, with the 2501 model facts:
  40 layers, hidden size 5120, intermediate size 32768, 32 attention heads,
  8 KV heads, 32k context, BF16.

## How It Works

The registry is the extension point. Scripts can still accept an explicit
`--model_path`, but a known `--model_key` now supplies:

- output root selection,
- tokenizer kwargs,
- default classifier path,
- expected config dimensions,
- support status for the current causal-LM pipeline.

For unregistered models, scripts may still load a user-provided causal-LM
`--model_path`, but default output-root inference is intentionally refused.
The operator must pass an explicit output directory/control directory and, for
neuron work, an explicit classifier path.

For example, a Mistral run can use:

```bash
uv run python scripts/classifier.py \
  --model_key mistral_small_24b_instruct_2501 \
  --train_ids data/mistral24b/pipeline/train_qids_llm.json \
  --train_ans_acts data/mistral24b/pipeline/activations_llm_canonical/train/answer_tokens \
  --train_other_acts data/mistral24b/pipeline/activations_llm_canonical/train/all_except_answer_tokens \
  --test_ids data/mistral24b/pipeline/dev_qids_llm.json \
  --test_acts data/mistral24b/pipeline/activations_llm_canonical/dev/answer_tokens \
  --train_mode 3-vs-1
```

Adding another causal-LM text model should now require a registry entry and
wrapper configuration, not edits across every script.

The RunPod wrapper can be resumed by selecting stages:

```bash
STAGES=classifier,faitheval,faitheval_controls \
  bash scripts/infra/mistral24b_replication.sh
```

Use `DRY_RUN=1` to print the guarded command sequence without launching the GPU
work.

## Choices Made

**Use 2501 as the canonical second model.**
This matches the existing artifacts and current script assumptions. It is the
fastest credible second-model extension path.

**Defer 2503.**
2503 is registered, but marked unsupported because it is `model_type=mistral3`
with `Mistral3ForConditionalGeneration` and a multimodal processor path. Running
it through the current `AutoModelForCausalLM` text-only path would risk a broken
or silently noncanonical pipeline.

**Keep Gemma defaults working.**
Existing Gemma paths remain the default behavior unless a model key or explicit
path changes them.

**Make negative controls model-derived.**
The old implementation baked in Gemma's 34 layers and 10240 FFN width. That
would make Mistral controls invalid, so layer-matched random controls now use
the active classifier and model dimensions. Width mismatches now raise instead
of trying to infer a plausible geometry.

**Use direct PyTorch hooks, not a new tracing framework.**
The existing code already uses direct module hooks for CETT and interventions.
Changing that would have expanded the migration beyond the model-generalization
goal.

## Hardest Decision

The hardest decision was whether to make 2503 the target despite the plan's
warning. I rejected that for the canonical path because a partial 2503 migration
would look like progress while weakening the replication claim. The correct
2503 implementation needs a deliberate loader and processor path, plus tests
that prove chat templating and activation hooks are hitting the text stack
correctly.

## Alternatives Rejected

- **Hard-code Mistral paths into each script.**
  Rejected because it would repeat the Gemma problem and make the third model
  another cross-script rewrite.
- **Infer all model dimensions only from HF config at runtime.**
  Rejected because registry metadata is useful for offline docs, tests, default
  output roots, and unsupported-model gating.
- **Immediately run 2503 through `trust_remote_code` / Auto classes.**
  Rejected because 2503 is not a standard causal-LM text checkpoint in this
  workflow.
- **Keep negative-control constants and only override Mistral in docs.**
  Rejected because it would make layer-matched controls scientifically wrong.
- **Add broad new abstractions for every benchmark.**
  Rejected because the concrete problem was model identity, dimensions,
  tokenizer quirks, and output roots.

## Least Confident Areas

- **Peak memory estimates.**
  The corrected infra assessment is better than the stale 56-layer version, but
  actual hook-time peaks need measurement on the RunPod target GPU.
- **Mistral chat-template span behavior.**
  The tokenizer kwarg is fixed, but the answer-token span path should still be
  audited on real Mistral examples after activation extraction.
- **Wrapper runtime behavior.**
  The wrapper passed shellcheck, but the full 24B run was not executed locally.
- **Judge-dependent FalseQA controls.**
  FalseQA generation is wired, including quick layer-matched controls, but the
  comparison summary still depends on a separate GPT-4o judging pass.
- **Held-out Mistral classifier metrics.**
  The code path exists, but canonical train/dev/test metrics still need to be
  generated on GPU.

## Remaining Blockers

- Generate canonical disjoint Mistral train/dev/test splits from
  `data/mistral24b/answer_tokens_llm.jsonl`.
- Extract canonical Mistral activations for answer tokens and non-answer tokens.
- Train the sparse L1 3-vs-1 classifier and write held-out test metrics with
  provenance.
- Run at least one judged Mistral intervention benchmark with H-neuron,
  random, and layer-matched controls.
- Run the wrapper on the target pod with at least `DRY_RUN=1` first, then resume
  through explicit `STAGES` as each artifact is verified.
- Add a proper 2503 loader path before any exact-checkpoint Mistral Small 3.1
  claim.

## Verification Completed

- `uv run pytest tests/test_model_registry.py tests/test_utils.py::TestDirectionOutputDir tests/test_utils.py::TestJailbreakDecodeControls`
  passed: 28 tests.
- `ruff check scripts` passed.
- `ty check` passed.
- `shellcheck scripts/infra/mistral24b_replication.sh` passed.
- Full `uv run pytest` currently reports 738 passed and 2 failed in unrelated
  modified SIMID tests (`tests/test_simid.py` against `scripts/analyze_simid.py`).
  Those failures are not in the Mistral migration path and were left untouched.
- Added focused tests in `tests/test_model_registry.py` for registry metadata,
  tokenizer kwargs, output-root resolution, unsupported 2503 gating,
  classifier/model geometry mismatches, quick layer-matched controls, and
  negative-control geometry.
- Post-review hardening (2026-04-27): added 17 focused tests across new files
  `tests/test_run_negative_control.py` (12 tests covering `--prompt_style`
  argparse, `--max_samples` / `--sample_manifest` plumbing, `_faitheval_prompt`
  style divergence, and `_jailbreak_csv2_eval_dir` path derivation) and
  `tests/test_classifier.py` (5 tests covering the registry-aware
  `--save_model` default and explicit-override precedence). Full
  `uv run pytest` then reports 745 passed; pre-existing SIMID failures are
  unchanged.

## Source Of Truth

For future model migration, update `scripts/model_registry.py` first. Then add
or update the corresponding infra wrapper and targeted tests. Avoid introducing
model-specific constants directly into benchmark, classifier, extraction, or
negative-control logic.

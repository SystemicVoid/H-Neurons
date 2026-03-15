# GH200 Pipeline Research Log

Date: 2026-03-15
Audience: researchers, future agents, and mentors supervising the H-Neurons replication/extension work
Scope: remote GH200 orchestration for Mistral-Small-24B and Llama-3.3-70B on TriviaQA

## Executive Snapshot

Think of this run as a relay race with one lane, not two parallel jobs. The GH200 box has plenty of total memory, but the practical bottleneck is the 96 GB HBM3e attached to the GPU side. That means one large-model GPU-heavy step at a time.

As of 2026-03-15 16:26 local session time:

- Remote instance: `ubuntu@192.222.59.253`
- Remote tmux session: `hneurons`
- Active orchestration window: `gh200-pipeline`
- Mistral Step 1 window: `step1-m24b`
- Mistral Step 1 progress observed by line count: `37 / 3500`
- Legacy `llama-watcher` window: removed by the new orchestrator
- Current strategy: zero-cost fallback for Step 2 (`synthetic-output`) paired with Step 4 `--locations output`

## What Changed Today

### 1. We removed a likely late-stage failure before it could waste hours

`collect_responses.py` had already been fixed for a newer `transformers` behavior where `tokenizer.apply_chat_template()` can return a `BatchEncoding` instead of a raw tensor.

The same assumption still existed in `scripts/extract_activations.py`. That is dangerous because it does not fail immediately during setup; it fails only when the long Step 4 activation extraction begins. In practical terms, this is like discovering a loose bolt only after the rocket is already on the pad.

Action taken:

- Patched `scripts/extract_activations.py` to unwrap either tensor or `BatchEncoding`
- Deployed the patched file to the GH200 instance before Step 4 starts

### 2. We converted the manual Step 2 workaround into a first-class script mode

The original fallback for "no OpenAI key on remote" was a manual Python snippet that builds a synthetic `answer_tokens.jsonl` with empty `answer_tokens`.

That works, but it is easy to forget, hard to document cleanly, and annoying to resume. I changed `scripts/extract_answer_tokens.py` so it now supports:

- `--strategy llm`
- `--strategy synthetic-output`

The synthetic mode:

- keeps the same JSONL schema
- only includes unanimous samples
- stores `answer_tokens: []`
- preserves resume behavior

This makes the fallback reproducible instead of ad hoc.

### 3. We replaced hand-managed sequencing with a durable tmux orchestration path

I added:

- `scripts/gh200_sequential_pipeline.sh`
- `scripts/watch_gh200_pipeline_and_sync.sh`

The intent is to make the remote run behave more like a supervised batch job:

- wait for Mistral Step 1 to finish
- run Mistral Steps 2 to 5
- then start or resume Llama Step 1
- then run Llama Steps 2 to 5
- emit status, done, and failed sentinel files under `logs/`

This is useful for another agent because the state is now visible in a small set of files instead of hidden in scrolling tmux buffers.

## Incidents and Quirks

### Confirmed incidents

1. `extract_activations.py` had the same `apply_chat_template()` compatibility bug pattern as `collect_responses.py`.
   Impact:
   Step 4 could have failed hours into the run for newer `transformers`.
   Resolution:
   Patched locally and copied to remote on 2026-03-15.

2. The remote had a `llama-watcher` auto-start window.
   Impact:
   It reduced control over sequencing and increased the chance of accidentally overlapping GPU-heavy stages.
   Resolution:
   The new orchestrator kills that window before taking control.

3. Step 1 progress is better measured by output file line count than by `nvidia-smi`.
   Impact:
   GPU utilization can look deceptively idle between generations, so a quick `nvidia-smi` glance can make a healthy job look stalled.
   Evidence:
   Early probe showed about `17540 MiB / 97871 MiB` and `0%` utilization while Step 1 was still actively generating.

4. The local sync watcher had an initial tmux launch glitch.
   Impact:
   Detached local tmux creation succeeded, but the first `send-keys` target assumed a window name/index that was not valid in that shell invocation.
   Resolution:
   Fix the watcher launch to target the created session/window explicitly before relying on it for unattended sync.

### Confirmed brittle spots in the codebase

1. `extract_activations.py` finds answer-token spans by exact decoded token string match.
   Why it matters:
   If tokenization splits differ even slightly, answer-token activations are silently skipped. This is especially relevant for BPE/SentencePiece models where whitespace markers and subword boundaries vary.

   -> The disjoint test set (gemma 3 4b - 0% overlap with training data) yields **76.5% accuracy**, which is actually closer to the paper's 76.9% than the inflated 77.7% from the overlapping split. The ~1.2 percentage point drop from overlapping→disjoint confirms mild leakage, but the signal is clearly real — it holds on fully held-out data.

2. `extract_answer_tokens.py` silently filters out non-unanimous, `uncertain`, or `error` examples.
   Why it matters:
   The dataset size after Step 2 is not just "smaller than Step 1"; it is smaller in a label- and model-dependent way. That can change class balance and therefore the effective training set.

3. `sample_balanced_ids.py` only creates a training split.
   Why it matters:
   The local Gemma reference includes train/test IDs, but the current remote run as specified is training-only. That is enough for a pipeline milestone, but not enough for a final performance story.

4. Step 1 currently walks the first `3500` rows of TriviaQA rather than an explicitly randomized subset.
   Why it matters:
   This is convenient and reproducible, but it may quietly bias the sample if early rows are not representative.

## Interesting Findings

### Operational findings

1. The best checkpoint for intervention is before Step 4, not after.
   Reason:
   Step 4 is where the expensive model reload and long activation extraction happens. If there is a hidden tokenizer/model compatibility issue, catching it before that stage saves the most time.

2. A "resume-safe" pipeline is only truly resume-safe if every stage exposes visible progress.
   Right now:
   Step 1 line counts are visible.
   Step 4 file counts are visible.
   The new shell orchestration adds status files to make cross-session monitoring easier.

3. The zero-cost output-token path is a good engineering baseline even if it is not the final paper-quality setup.
   Analogy:
   It is the wind tunnel model, not the finished aircraft. It is cheaper, faster, and less exact, but it tells us whether the whole system basically works.

### Research findings worth highlighting tomorrow

1. The replication task has already surfaced implementation-sensitive assumptions, not just modeling questions.
   This is valuable because it separates "the science failed" from "the pipeline broke."

2. There is a clear methodological tradeoff between exactness and throughput:
   - exact answer-token extraction is closer to the paper setup
   - output-token training is cheaper and operationally robust
   - the right choice depends on whether tomorrow's goal is a progress report, a faithful replication, or a decision about where to spend compute/API budget next

## Hypotheses To Test

1. Output-token training will likely reduce absolute classification accuracy relative to exact answer-token training, but may still preserve the broad sparsity pattern and identify a meaningful neuron subset.

2. The exact-token matching failure mode will probably be worse for Llama-3.3-70B than for Mistral-24B if token boundary behavior differs more often on representative answers.

3. The proportion of unanimous false vs unanimous true samples may differ across models enough that "1000 per class" is effectively capped by model behavior rather than by the nominal sample budget.

4. Rule-based judging may undercount semantically correct but non-literal answers, which could matter more for larger instruction-tuned models that paraphrase aggressively.

5. If we later add a held-out evaluation split, the variance from sample selection may be large enough that comparing models on one fixed `3500`-question prefix is misleading.

## Methodological Improvements

### Improvements already implemented

1. Centralized `apply_chat_template()` compatibility handling in `extract_activations.py`.

2. Scripted synthetic-output generation in `extract_answer_tokens.py` instead of relying on notebook-style inline code.

3. Sequential tmux orchestration with explicit status files.

### Improvements still worth doing

1. Add a shared helper for chat-template output handling across all scripts.
   Why:
   The same bug appeared in two places already. That pattern will recur.

2. Add a proper held-out evaluation stage for the remote runs.
   Why:
   Training a sparse classifier without a paired evaluation is like fitting a line without checking the residuals.

3. Record per-stage yield and attrition explicitly.
   Suggested metrics:
   - Step 1 processed count
   - unanimous true count
   - unanimous false count
   - skipped due to uncertainty/error
   - Step 4 successful activation files

4. Add a token-span mismatch audit for `answer_tokens` mode.
   Why:
   Silent failure is the worst failure mode for research pipelines because it looks like low signal instead of a bug.

5. Consider a randomized or stratified TriviaQA subset rather than the first `3500`.
   Why:
   This makes comparisons more defensible if we later compare architectures or reruns.

## Questions For My Mentor

1. For tomorrow's milestone, should we prioritize a faithful paper-style setup or an end-to-end robust baseline?
   Translation:
   Is "runs cleanly on output tokens" good enough for now, or is the answer-token extraction accuracy worth the extra API cost immediately?

2. Do we want the next increment of effort to go toward evaluation quality or orchestration reliability?
   Tradeoff:
   Evaluation quality means test splits and more paper-like metrics.
   Orchestration reliability means fewer wasted GH200 hours.

3. How much deviation from the published procedure is acceptable for an exploratory replication?
   Example deviations:
   - rule-based judge instead of LLM judge
   - output-token training instead of answer-token training
   - first `3500` samples instead of a randomized subset

4. Should we treat the GH200 run mainly as a systems exercise or as data for a scientific claim?
   Why this matters:
   The bar for logging, evaluation, and ablation is very different.

5. If the output-token path works, should we use it as a screening pass before paying for exact answer-token extraction on only the most promising model/settings?

## Suggested Presentation Framing

If you need a compact story tomorrow, this structure should work:

1. Goal:
   Reproduce the H-Neurons pipeline on larger models under real hardware constraints.

2. What is working:
   The 5-stage pipeline is already complete locally for Gemma-3-4B, and the GH200 run is now automated for Mistral-24B then Llama-70B.

3. What we learned:
   The main blockers so far are not "the idea does not replicate," but operational brittleness, tokenizer/API assumptions, and cost-vs-fidelity tradeoffs.

4. Why this matters:
   Fixing those operational issues turns future experiments from fragile one-offs into reproducible research runs.

5. Decision point:
   Choose whether the next step is higher-fidelity labels/tokens or faster broader sweeps across models/settings.

## Current Files Relevant To This Log

- [AGENTS.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/AGENTS.md)
- [scripts/extract_activations.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/scripts/extract_activations.py)
- [scripts/extract_answer_tokens.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/scripts/extract_answer_tokens.py)
- [scripts/gh200_sequential_pipeline.sh](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/scripts/gh200_sequential_pipeline.sh)
- [scripts/watch_gh200_pipeline_and_sync.sh](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/scripts/watch_gh200_pipeline_and_sync.sh)

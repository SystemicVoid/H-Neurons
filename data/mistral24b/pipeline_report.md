# Mistral 24B — H-Neuron Identification Pipeline Report

**Date:** 2026-03-15
**Model:** `mistralai/Mistral-Small-24B-Instruct-2501` (instruction-tuned)
**Primary hardware used:** Lambda GH200 96 GB GPU (`ubuntu@192.222.59.253`)
**Paper reference:** Gao et al., "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs" (arXiv:2512.01797v2)

---

## 1. Results Summary

This run should be understood as a **successful operational replication of the Mistral identification pipeline**, not yet a full paper-faithful Mistral replication.

What is solid:

- Step 1 completed on the GH200 in a true GPU-resident configuration
- compact Mistral artifacts have been copied locally
- Step 4 activations have been copied locally
- the classifier can be rebuilt locally without the old sklearn pickle warning
- the resulting sparsity is right in the paper's target regime

What is not yet solid:

- the current **completed end-to-end probe/activation path** is still the zero-cost `output`-location variant, even though a paper-path Step 2 answer-token artifact now exists locally
- there is **no held-out Mistral evaluation split yet**
- there are **no Mistral intervention results yet**
- there is **no origin/base-model transfer analysis yet**
- the current checkpoint is **`Mistral-Small-24B-Instruct-2501`**, while the paper table names **`Mistral-Small-3.1-24B` / `...-2503`-style checkpoints**, so this is currently a same-family replication rather than an exact-checkpoint replication

| Metric | Current Mistral status |
|--------|------------------------|
| TriviaQA Step 1 samples | **3500** |
| Step 2 extracted entries (`output` fallback) | **2602** |
| Step 2 extracted entries (paper-path LLM) | **2539** |
| Step 2 exact-span-usable entries (paper-path LLM) | **2532** |
| Balanced train IDs | **1182** total (**591 true + 591 false**) |
| Step 4 activation files | **1182** |
| H-Neuron count | **12** |
| H-Neuron ratio | **0.0092‰** |
| Classifier training accuracy | **81.56%** |
| Classifier training AUROC | **0.9041** |
| Held-out Mistral test accuracy | **Not run yet** |

The headline result is the sparsity: **12 H-Neurons = 0.0092‰**, which is almost exactly on top of the paper's reported Mistral-scale `~0.01‰` regime. That is encouraging, but it is not enough by itself to claim paper replication, because the only **completed end-to-end** classifier result is still a training-set result on the `output` fallback path.

---

## 2. Pipeline Stage-by-Stage

### Stage 1: Response Collection

- **Script:** `scripts/collect_responses.py`
- **Output:** `data/mistral24b_TriviaQA_consistency_samples.jsonl`
- **Status:** complete
- **Count:** `3500` rows
- **Important runtime truth:** this only worked correctly after forcing `--device_map cuda:0` on the GH200. `device_map=auto` partially spilled the model to CPU/Grace memory and produced the wrong experiment.

This is the expensive part of the pipeline, and it is safely preserved locally now. This means we do **not** need to rerun the multi-hour Mistral Step 1 job tomorrow.

### Stage 2: Answer Token Extraction

- **Current artifact:** `data/mistral24b_answer_tokens.jsonl`
- **Count:** `2602` rows
- **Status:** complete for the current run
- **Methodological caveat:** this file came from the zero-cost operational path, not the final paper-faithful answer-token workflow we want to use for the main Mistral result tomorrow.

We have now also run a **100-sample local canary** for the true answer-token path:

- **Input:** `data/mistral24b_canary_100_consistency_samples.jsonl`
- **Output:** `data/mistral24b_canary_100_answer_tokens.jsonl`
- **Result:** `99/100` rows extracted successfully with GPT-4o at low cost and without using the local GPU
- **Span-match validation:** `99/99` extracted rows were re-found successfully by the current tokenizer/span-matching logic used by `scripts/extract_activations.py`
- **Regex-fix check:** the canary match rate was identical with and without `fix_mistral_regex=True`
- **Known failure:** one sample (`tc_115`) repeatedly failed because the extractor received malformed JSON from the judge model on a short quoted answer

This is an important de-risking result: the fragile part was never "can the GPU compute activations?" but "will exact answer-token extraction survive contact with Mistral tokenization and the repo's exact span-matching code?" On this canary, the answer is **yes**.

The current file is still valuable because it supports the completed output-based Step 4 and classifier run. But if tomorrow's goal is "closer to the paper," we should reuse the saved Step 1 file and rerun Step 2 with the true answer-token extraction path rather than rerunning Step 1.

That full true-token rerun has now also been completed locally without touching the GPU:

- **Paper-path output:** `data/mistral24b_answer_tokens_llm.jsonl`
- **Eligible unanimous rows from Step 1:** `2602`
- **Successful extracted rows:** `2539`
- **Rows skipped due repeated malformed JSON from the judge model:** `63`
- **Exact-span matches recoverable by current `extract_activations.py` logic:** `2532 / 2539`
- **Rows with non-contiguous or otherwise unmatchable extracted token lists:** `7`

The important practical truth is that the full Step 2 paper-path artifact now exists, but it is not perfectly clean. It is good enough to continue from later, yet there is a small repair debt:

- `63` rows are missing entirely because the judge model repeatedly returned malformed JSON
- `7` rows were extracted but are not usable by the current exact-span matcher because the selected tokens are not a single continuous segment in the response

So the immediately usable paper-style pool is currently:

- `1972` true rows
- `560` false rows

That means a span-usable balanced training split is currently capped at **`560 + 560`** unless we repair some of the missing/unmatchable rows.

### Stage 3: Balanced ID Sampling

- **Script:** `scripts/sample_balanced_ids.py`
- **Output:** `data/mistral24b_train_qids.json`
- **Status:** complete
- **Counts:** `591 true`, `591 false`

This is not a problem in itself, but it reveals the class bottleneck: the current Mistral pool has many more unanimous true examples than unanimous false ones, so the false pool limits the balanced sample.

For the newer paper-path answer-token artifact, the bottleneck is now slightly tighter: with the current uncleaned `data/mistral24b_answer_tokens_llm.jsonl`, the immediately usable balanced cap is **`560 false + 560 true`**.

### Stage 4: CETT Activation Extraction

- **Current output root:** `data/mistral24b_activations/output`
- **Status:** complete for the current run
- **Count:** `1182` `.npy` files
- **Size:** `5.8 GB`

This was the main "should we preserve it?" decision before shutting down the GH200. We verified local and remote copies match by:

- file count
- directory size
- manifest hash
- sample file checksums

So the activation substrate is safely local. That matters more than the old remote pickle, because activations are the thing that lets us refit probes cleanly under new environments.

### Stage 5: Classifier Training

- **Original remote artifact:** `models/mistral24b_classifier.pkl`
- **Locally rebuilt artifact:** `models/mistral24b_classifier_rebuilt.pkl`
- **Metrics JSON:** `data/mistral24b_classifier_rebuilt_metrics.json`

The original remote pickle loaded locally with an sklearn version warning. That is annoying but not catastrophic. The important point is that we rebuilt it locally from the preserved activations, so we now have a clean, current-environment classifier artifact.

| Metric | Value |
|--------|-------|
| Accuracy | `0.8156` |
| Precision | `0.8221` |
| Recall | `0.8054` |
| F1 | `0.8137` |
| AUROC | `0.9041` |
| H-Neurons | `12` |

These are **training metrics only**. Treat them as a pipeline confirmation, not a final scientific claim.

---

## 3. What Is Preserved Locally

The critical Mistral artifacts are now local, so the GH200 instance is no longer the single point of failure.

### Compact artifacts

- `data/mistral24b_TriviaQA_consistency_samples.jsonl`
- `data/mistral24b_answer_tokens.jsonl`
- `data/mistral24b_train_qids.json`
- `models/mistral24b_classifier.pkl`
- `models/mistral24b_classifier_rebuilt.pkl`
- `data/mistral24b_classifier_rebuilt_metrics.json`

### Heavy artifacts

- `data/mistral24b_activations/output/`

### Operational notes preserved elsewhere

- `docs/gh200-research-log-2026-03-15.md`
- `AGENTS.md`

### Storage truth

The GH200 instance was **not** launched with an attached Lambda filesystem. `~/h-neurons` lived on the instance root disk (`/dev/vda1`), not on `/lambda/nfs/...`. That means terminating the instance would have destroyed the remote-only data. Copying the artifacts out before shutdown was therefore necessary.

---

## 4. What This Run Proves vs. What It Does Not

### What it proves

1. A 24B Mistral-class model can run this workflow in full precision on a single GH200 **if** it is pinned explicitly to `cuda:0`.
2. The H-Neuron identification pipeline works end-to-end for Mistral at the engineering level.
3. The resulting sparsity is in the right ballpark for the paper.
4. The old classifier pickle warning is fixable by rebuilding locally from preserved activations.

### What it does not prove yet

1. A held-out Mistral TriviaQA result comparable to the paper.
2. That the output-token fallback reproduces the paper's answer-token-based result closely enough.
3. The Mistral perturbation findings from the paper's Section 6.2.
4. The origin-tracing/base-model transfer story from Section 6.3.

This is the key framing for tomorrow: **we have a strong base camp, not the summit.**

---

## 5. Issues, Quirks, and Lessons

### 5.1 GH200 fit matters more than nominal total memory

The GH200 has plenty of total system memory, but the model has to fit the actual GPU-resident part of the workflow. For this project, "it eventually produces output" is not enough. If the run is hybrid in a way that changes the activation experiment, it is the wrong run.

### 5.2 The remote project `.venv` was CPU-only

The repo's default remote `.venv` had CPU-only torch. The working path on GH200 used a CUDA-capable `.venv-gpu` built from system Python and needed newer `pillow` plus `jinja2>=3.1`.

### 5.3 The local rebuilt classifier is more important than the old remote pickle

Pickles are convenience jars. Activations are the real substrate. Once the activations were safely local, we could rebuild the Mistral probe cleanly and stop worrying about the remote sklearn version.

### 5.4 Step 4 was worth copying

The activation directory is about `5.8 GB`. At the available bandwidth, copying it cost minutes. Recomputing it later would cost hours of GH200 time. Economically, copying was clearly the right choice.

### 5.5 The current Mistral run is still the "robust baseline" path

Right now, Mistral is in the same position a wind-tunnel prototype is in before a full flight test: it tells us the structure is real, but not yet the exact final aerodynamics.

### 5.6 The local `C` sweep is useful, but not fully paper-equivalent yet

The local `scripts/classifier.py` can already sweep `C` on a held-out split and pick the best detector by `accuracy`, `f1`, or `auroc`. That is enough for a sound next-stage probe selection.

But the paper's stated selection rule is stricter: it chooses `C` using both:

- held-out detection performance
- TriviaQA behavior after suppressing the selected neurons

So tomorrow's held-out `C` sweep should be treated as the **best available detector-selection baseline**, not yet the final paper-equivalent intervention-selection rule.

### 5.7 FaithEval has a paper-faithfulness trap in the local script defaults

`scripts/run_intervention.py` defaults FaithEval to `--prompt_style anti_compliance`, which is useful for stress-testing resistance to misleading context.

But the script's own prompt builder notes that `--prompt_style standard` matches the official Salesforce framing and presumed paper usage. So once we reach interventions, `standard` should be the default for paper-faithful replication claims.

### 5.8 The answer-token canary shifted the main risk from methodology to throughput

The 100-sample true-token canary cost about a tenth of a dollar on GPT-4o by local token estimate and avoided the GPU entirely. More importantly, it showed that the current Mistral tokenizer and exact span-matching logic can already handle real extracted answer spans on representative saved samples.

So the remaining concern is no longer "will the paper-style path break immediately?" but rather:

- API cost/time for a full Step 2 rerun
- whether we want to tolerate a small number of malformed-JSON extraction failures or harden the extractor first
- the later GPU time for full activation extraction once the current tmux workload finishes

That API/time concern has now been resolved operationally: the full rerun finished in about `37 minutes` locally and stayed under the low-single-digit budget. The remaining concern is simply artifact cleanliness, not feasibility.

### 5.9 The full Step 2 paper-path artifact is resumable, but not lossless

Because `scripts/extract_answer_tokens.py` appends JSONL records and skips already-processed IDs, the paper-path file is safely resumable. That part worked exactly as intended.

However, resumable does not mean lossless:

- repeated malformed JSON responses from the judge model leave holes in the file
- the extractor only validates token membership, not whether the returned tokens form one continuous span
- `extract_activations.py` later requires a continuous span and silently cannot use those few broken rows

In other words, the Step 2 rerun succeeded as a durable artifact-generation job, but not yet as a perfectly paper-clean labeling job.

---

## 6. Tomorrow's Decision Tree

There are really only two sensible directions tomorrow.

### Option A — Faithful Mistral replication

This is the right path if the goal is "get as close as practical to the paper's Mistral story."

1. Reuse the saved Step 1 Mistral JSONL.
2. Reuse the completed paper-path answer-token artifact `data/mistral24b_answer_tokens_llm.jsonl`.
3. Decide whether to accept the current `2532` span-usable rows or repair the `63 + 7` problematic cases first.
4. Build a held-out evaluation split from TriviaQA validation/test with `sample_num=1`.
5. Extract answer-token and non-answer-token activations for that train/eval setup.
6. Sweep `C` on the held-out split with the updated `scripts/classifier.py`.
7. Use that selected classifier as the canonical Mistral probe.
8. Run Mistral interventions, starting with rule-evaluable benchmarks and using FaithEval `standard` prompts for paper-faithful claims.

This is the more paper-faithful path.

### Option B — Behavioral experiments first

This is the right path if the goal is "make progress on the causal story immediately."

1. Keep the current rebuilt Mistral classifier as the operational probe.
2. Run Mistral intervention experiments first.
3. Treat the current Mistral identification result as a baseline.
4. Return later to the answer-token + held-out evaluation cleanup.

This is faster for visible progress, but weaker as a strict replication claim.

### My recommendation

Do **Option A first**, then use the selected held-out Mistral probe for interventions.

Reason:
Running interventions before the Mistral identification story is methodologically cleaned up is like tuning a guitar solo before checking whether the instrument is in tune. It can still sound impressive, but it weakens the credibility of the underlying claim.

---

## 7. Exact Recommended Next Steps

### Priority 1 — Make Mistral paper-closer

1. Reuse `data/mistral24b_TriviaQA_consistency_samples.jsonl`.
2. Treat the 100-sample canary as **passed** for tokenizer/span-matching risk.
3. Reuse the completed `data/mistral24b_answer_tokens_llm.jsonl` paper-path artifact rather than rerunning Step 2 from scratch.
4. Decide whether to accept the current span-usable cap of `560 + 560` or spend local/API time repairing some of the `63 + 7` problematic rows.
5. Create a disjoint held-out Mistral eval set from TriviaQA validation or test.
6. Extract train/eval activations in the paper-style setup.
7. Use the updated `scripts/classifier.py` to sweep `C` and save metrics JSON.
8. Treat that `C` sweep as a detector-selection baseline until we also score post-suppression TriviaQA behavior.

### Priority 2 — Run Mistral interventions

Start with the benchmarks that do not require paid judges:

1. `faitheval` with `--prompt_style standard`
2. `sycophancy_triviaqa`

Then treat these as optional later:

3. `falseqa`
4. `jailbreak`

The reason is simple: the first two are cheaper and cleaner for core progress.

### Priority 3 — Only then consider origin tracing

Do not start with Section 6.3 tomorrow unless a matching base Mistral checkpoint is already ready and we explicitly want origin evidence. That branch is real but not yet turnkey in this repo.

### Priority 0.5 — Use non-GPU work while tmux owns the GPU

If the local GPU is busy for hours, the first canary can still be split into mostly non-GPU work:

1. This has now been done successfully for the 100-sample canary.
2. The full Step 2 rerun has now also been completed locally, since `scripts/extract_answer_tokens.py` only needs tokenizer access plus the judge API and does not load the 24B model.
3. The next non-GPU move, if desired, is artifact cleanup: repair the `63` missing rows and `7` non-contiguous rows.
4. Defer only the actual activation extraction / model forward pass until the GPU window is free.

That means the next productive Mistral move still does **not** have to wait for the current tmux GPU job to finish.

---

## 8. Night-Stop Snapshot

At the point this report was written:

- GH200 Mistral artifacts were preserved locally
- GH200 had no attached Lambda filesystem
- the GH200 was idle and safe to terminate from the Mistral-data-preservation perspective
- local Gemma FalseQA intervention work was still ongoing in tmux

That means tomorrow's Mistral work should begin from the **local preserved artifacts**, not from the terminated cloud box.

---

## 9. One-Sentence Hand-Off

**We successfully rescued and preserved the Mistral-24B pipeline outputs, completed a local paper-path answer-token rerun, and paused at the clean handoff point right before held-out eval collection and paper-style activation extraction.**

---

## 10. Hold Status and Easy Resume

As of the hold point:

- the full paper-path Step 2 rerun is finished
- there is **no live Mistral tmux job left running**
- the project is paused **before** held-out eval collection and before any new answer-token activation extraction
- the main remaining choice is whether to accept the current paper-path artifact as-is or repair its small set of broken rows first

### Resume path A — fastest acceptable continuation

Use the finished paper-path file as-is and accept the current `560 + 560` balanced cap.

1. Sample balanced IDs from `data/mistral24b_answer_tokens_llm.jsonl`, capping at `560` per class.
2. Build the held-out eval split from TriviaQA validation/test with `sample_num=1`.
3. When a suitable GPU box is free again, run paper-style activation extraction.
4. Sweep `C` on the held-out setup.

### Resume path B — cleaner artifact first

Before any new GPU work:

1. Repair or re-extract the `63` missing Step 2 rows that failed with malformed JSON.
2. Repair the `7` extracted-but-unmatchable rows by enforcing a continuous answer-token span.
3. Recompute the usable true/false pool size.
4. Only then proceed to balanced sampling and held-out eval.

### Copy-pasteable resume commands

Fastest next local command if we accept the current artifact:

```bash
uv run python scripts/sample_balanced_ids.py \
    --input_path data/mistral24b_answer_tokens_llm.jsonl \
    --output_path data/mistral24b_train_qids_llm.json \
    --num_samples 560
```

Held-out eval collection command skeleton for the next Lambda/GH200 window:

```bash
uv run python scripts/collect_responses.py \
    --model_path mistralai/Mistral-Small-24B-Instruct-2501 \
    --data_path data/TriviaQA/rc.nocontext/validation-00000-of-00001.parquet \
    --output_path data/mistral24b_TriviaQA_validation_samples.jsonl \
    --sample_num 1 \
    --device_map cuda:0 \
    ...
```

### Do not redo

- do **not** rerun the expensive original Mistral Step 1 training collection
- do **not** rerun the full paper-path Step 2 job from scratch unless we intentionally want a cleaner labeling pass
- do **not** overwrite the old synthetic fallback artifact; keep both files because they answer different questions

---

## Negative Control Experiments (Proposed)

The current intervention results lack a negative control: we scale 38 H-Neurons
but never test whether scaling 38 *arbitrary* neurons produces a similar effect.
Without this, we cannot rule out that any MLP perturbation at this scale increases compliance.

### Proposed experiments (all runnable on RTX 5060 Ti, ~$0 each)

**Control A: Random-neuron scaling (~1h)**
- Select 38 neurons uniformly at random from the same layer distribution as the H-Neurons
- Run the same α sweep (0.0–3.0) on FaithEval with both prompt styles
- Compare compliance curves. If the trend is similar, the effect is not H-Neuron-specific
- Run 3–5 random seeds for the neuron selection to get error bars on the control

**Control B: Same-layer random neurons (~1h)**
- For each H-Neuron (layer L, index N), pick a random neuron from the same layer L
- This controls for the layer distribution being the active ingredient vs the specific neuron indices
- Particularly interesting given the layer-5 concentration finding

**Control C: Negative-weight neurons (~30min)**
- The L1 classifier assigns negative weights to some neurons (predicting "true" answers)
- Scale these by the same α values — if they show the *opposite* compliance trend,
  that's strong evidence for specificity

**Control D: Magnitude-matched non-H-neurons (~1h)**
- Select 38 neurons whose mean activation magnitude matches the H-Neurons
  but which received zero L1 weight
- This controls for the possibility that high-activation neurons are simply
  easier to perturb regardless of their role

### Priority order
1. Control A (most decisive — tests the core specificity claim)
2. Control C (cheapest, most informative per minute)
3. Control B (disambiguates layer vs index)
4. Control D (rules out activation-magnitude confound)

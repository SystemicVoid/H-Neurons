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

- the current completed Mistral path is the **zero-cost `output`-location variant**, not the paper's preferred exact answer-token path
- there is **no held-out Mistral evaluation split yet**
- there are **no Mistral intervention results yet**
- there is **no origin/base-model transfer analysis yet**
- the current checkpoint is **`Mistral-Small-24B-Instruct-2501`**, while the paper table names **`Mistral-Small-3.1-24B` / `...-2503`-style checkpoints**, so this is currently a same-family replication rather than an exact-checkpoint replication

| Metric | Current Mistral status |
|--------|------------------------|
| TriviaQA Step 1 samples | **3500** |
| Step 2 extracted entries | **2602** |
| Balanced train IDs | **1182** total (**591 true + 591 false**) |
| Step 4 activation files | **1182** |
| H-Neuron count | **12** |
| H-Neuron ratio | **0.0092‰** |
| Classifier training accuracy | **81.56%** |
| Classifier training AUROC | **0.9041** |
| Held-out Mistral test accuracy | **Not run yet** |

The headline result is the sparsity: **12 H-Neurons = 0.0092‰**, which is almost exactly on top of the paper's reported Mistral-scale `~0.01‰` regime. That is encouraging, but it is not enough by itself to claim paper replication, because it is currently a training-set result on the `output` fallback path.

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

The current file is still valuable because it supports the completed output-based Step 4 and classifier run. But if tomorrow's goal is "closer to the paper," we should reuse the saved Step 1 file and rerun Step 2 with the true answer-token extraction path rather than rerunning Step 1.

### Stage 3: Balanced ID Sampling

- **Script:** `scripts/sample_balanced_ids.py`
- **Output:** `data/mistral24b_train_qids.json`
- **Status:** complete
- **Counts:** `591 true`, `591 false`

This is not a problem in itself, but it reveals the class bottleneck: the current Mistral pool has many more unanimous true examples than unanimous false ones, so the false pool limits the balanced sample.

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

---

## 6. Tomorrow's Decision Tree

There are really only two sensible directions tomorrow.

### Option A — Faithful Mistral replication

This is the right path if the goal is "get as close as practical to the paper's Mistral story."

1. Reuse the saved Step 1 Mistral JSONL.
2. Run a **small answer-token canary first** on roughly `50-100` saved samples and verify that the extracted answer tokens can actually be re-found by the current tokenizer/span-matching logic.
3. Only if that canary looks clean, rerun Step 2 using the proper answer-token extraction path.
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
2. Run a **small canary** on about `50-100` examples using true answer-token extraction before committing to a full rerun.
3. Validate that the canary answer-token spans can be matched by the current tokenizer logic; if not, fix the span-matching issue before any full Step 2/4 rerun.
4. Run Step 2 with the proper answer-token extraction path.
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

1. Run the small Step 2 answer-token extraction canary, since `scripts/extract_answer_tokens.py` only needs tokenizer access plus the judge API and does not load the 24B model.
2. Run a tokenizer-only span-match validation on those canary outputs, since the fragile part is the exact token-string matching in `extract_activations.py`, not the forward pass itself.
3. Defer only the actual activation extraction / model forward pass until the GPU window is free.

That means tomorrow's first productive move does **not** have to wait for the current tmux GPU job to finish.

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

**We successfully rescued and preserved the Mistral-24B pipeline outputs, including the expensive activation substrate; tomorrow's real work is to convert this robust baseline into a paper-closer Mistral result by rerunning the answer-token / held-out evaluation path and then using that cleaner probe for intervention experiments.**

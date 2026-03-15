# Gemma 3 4B — H-Neuron Identification Pipeline Report

**Date:** 2026-03-15
**Model:** `google/gemma-3-4b-it` (instruction-tuned)
**Hardware:** RTX 5060 Ti (16 GB VRAM), AMD 9900X, 64 GB RAM
**Paper reference:** Gao et al., "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs" (arXiv:2512.01797v2)

---

## 1. Results Summary

| Metric | Paper (Gemma-3-4B) | Overlapping test (66.3%) | Disjoint test (0%) |
|--------|-------------------|--------------------------|---------------------|
| TriviaQA test accuracy | 76.9% | 77.7% | **76.5%** |
| H-Neuron count | ~35 (0.10‰) | 38 (0.11‰) | **38** (0.11‰) |
| AUROC (test) | — | 0.863 | **0.843** |
| Precision (test) | — | 0.780 | **0.767** |
| Recall (test) | — | 0.769 | **0.761** |
| F1 (test) | — | 0.775 | **0.764** |
| Test set size | — | 2,000 (1000t+1000f) | **782** (391t+391f) |

The disjoint test set (0% overlap with training data) yields **76.5% accuracy**, which is actually closer to the paper's 76.9% than the inflated 77.7% from the overlapping split. The ~1.2 percentage point drop from overlapping→disjoint confirms mild leakage, but the signal is clearly real — it holds on fully held-out data.

The classifier identifies 38 H-Neurons out of 348,160 total neuron positions (34 layers × 10,240 intermediate neurons), achieving 99.978% weight sparsity. The same 38 neurons were selected regardless of test set composition (training is identical).

---

## 2. Pipeline Stage-by-Stage

### Stage 1: Response Collection (pre-existing)

- **Script:** `scripts/collect_responses.py`
- **Output:** `data/gemma3_4b_TriviaQA_consistency_samples.jsonl`
- **Parameters:** 10 responses per question, `temperature=1.0`, `top_k=50`, `top_p=0.9`, rule-based judge

| Category | Count |
|----------|-------|
| Total questions | 3,500 |
| All-correct (10/10 true) | 1,435 |
| All-incorrect (10/10 false) | 1,680 |
| Mixed (disagreement) | 385 |
| Uncertain/error | 0 |
| **Passing consistency filter** | **3,115** |

The batch review (see `data/batch3500_review.md`) identified 96 all-correct entries where the rule judge matched via substring rather than clean short-answer correctness. These were kept to match the paper's methodology, which does not apply additional filtering beyond consistency.

### Stage 2: Answer Token Extraction

- **Script:** `scripts/extract_answer_tokens.py`
- **Output:** `data/gemma3_4b_answer_tokens.jsonl`
- **LLM:** GPT-4o (`temperature=0.0`)
- **Cost:** ~$3.50
- **Wall time:** ~33 minutes (including one restart after tmux session loss)

| Metric | Count |
|--------|-------|
| Consistent entries (input) | 3,115 |
| Successfully extracted | 2,997 |
| Failed extraction | 118 (3.8%) |
| True labels extracted | 1,391 |
| False labels extracted | 1,606 |

The 118 failures are almost entirely caused by a single bug: the script converts GPT-4o's single-quoted Python lists to JSON via `.replace("'", '"')`, which breaks when tokens themselves contain apostrophes (e.g., `"O'Neill"` → `"O"Neill"`). See Section 4 for details.

### Stage 3: Balanced ID Sampling

- **Script:** `scripts/sample_balanced_ids.py`
- **Outputs:** `data/gemma3_4b_train_qids.json`, `data/gemma3_4b_test_qids.json`

| Split | True | False | Total | Seed |
|-------|------|-------|-------|------|
| Train | 1,000 | 1,000 | 2,000 | 42 |
| Test | 1,000 | 1,000 | 2,000 | 123 |
| Overlap | — | — | 1,327 (66.3%) | — |

Train and test were sampled independently (matching the paper's example data pattern, which also shows 1000/1000 for both splits). The 66.3% overlap is a consequence of sampling 2×1000 from pools of only 1,391 true and 1,606 false IDs. This is not ideal — see Section 4 for a proposed fix.

### Stage 4: CETT Activation Extraction

- **Script:** `scripts/extract_activations.py`
- **Output:** `data/activations/{answer_tokens,all_except_answer_tokens}/act_{qid}.npy`
- **Wall time:** ~65 seconds (train) + ~21 seconds (test)

| Location | Files | Size |
|----------|-------|------|
| `answer_tokens` | 2,665 | 3.71 GB |
| `all_except_answer_tokens` | 1,993 | 2.78 GB |
| **Total** | **4,658** | **6.49 GB** |

Each `.npy` file has shape `(34, 10240)` = 1,360 KB (float32). 7 train IDs and 7 test IDs failed activation extraction due to answer-token string matching failures (the decoded tokens didn't exactly match the stored token list).

### Stage 5: Classifier Training

- **Script:** `scripts/classifier.py`
- **Output:** `models/gemma3_4b_classifier.pkl`
- **Parameters:** 3-vs-1 mode, L1 penalty, C=1.0, liblinear solver
- **Wall time:** <5 seconds

Training data composition (3-vs-1 mode):

| Category | Label | Count |
|----------|-------|-------|
| False answer tokens | 1 (positive) | 993 |
| True answer tokens | 0 (negative) | 993 |
| True other tokens | 0 (negative) | 993 |
| False other tokens | 0 (negative) | 993 |

Training set results: 89.2% accuracy, 0.958 AUROC
Test set results: **77.7% accuracy**, 0.863 AUROC

---

## 3. H-Neuron Analysis

### Layer Distribution

The 38 identified H-Neurons are distributed across 23 of 34 layers, with a concentration in the early-to-middle layers (0–16):

| Layer range | H-Neuron count | % of total |
|-------------|---------------|------------|
| 0–10 (early) | 18 | 47.4% |
| 11–20 (middle) | 11 | 28.9% |
| 21–33 (late) | 9 | 23.7% |

### Top 10 H-Neurons by Classifier Weight

| Rank | Layer | Neuron | Weight |
|------|-------|--------|--------|
| 1 | 20 | 4288 | 12.169 |
| 2 | 14 | 8547 | 7.386 |
| 3 | 13 | 833 | 3.451 |
| 4 | 5 | 5227 | 3.337 |
| 5 | 33 | 8011 | 3.071 |
| 6 | 24 | 7995 | 2.603 |
| 7 | 26 | 1359 | 2.456 |
| 8 | 9 | 5580 | 1.824 |
| 9 | 10 | 4996 | 1.705 |
| 10 | 0 | 1819 | 1.693 |

The single most influential H-Neuron (Layer 20, Neuron 4288) has a weight 1.65× larger than the second-ranked, suggesting it plays a disproportionately strong role in hallucination prediction. The classifier also identified 38 negative-weight neurons (suppressive of hallucination), but these are not classified as H-Neurons per the paper's definition.

---

## 4. Issues, Quirks, and Proposed Improvements

### 4.1 Apostrophe Bug in Answer Token Extraction (118 failures)

**Root cause:** `extract_answer_tokens.py` line 76 converts GPT-4o's response from Python list syntax to JSON via `.replace("'", '"')`. When a token itself contains an apostrophe (e.g., `O'Neill`, `don't`, `'s`), the replacement produces malformed JSON:

```
Input:  ['Eugene', ' O', "'", 'Neill']
After:  ["Eugene", " O", """, "Neill"]  ← broken
```

All 3 retry attempts hit the same parse error because GPT-4o consistently returns the same tokens.

**Impact:** 118 of 3,115 entries (3.8%) lost. These are systematically biased toward responses containing possessives, contractions, and proper names with apostrophes (e.g., O'Neill, O'Brien, 80's, Queen's). This could introduce a subtle content bias in the training data.

**Fix:** Replace the naive `.replace("'", '"')` with `ast.literal_eval()`, which correctly parses Python list literals including nested quotes:

```python
import ast
extracted = ast.literal_eval(reply)  # instead of json.loads(reply.replace("'", '"'))
```

### 4.2 Train/Test Overlap (66.3%) — FIXED

**Issue:** Running `sample_balanced_ids.py` twice with different seeds produced independent samples with 66.3% overlap. With 1,391 true IDs and both splits requesting 1,000, heavy overlap was expected by the pigeonhole principle.

**Impact:** The overlapping test accuracy (77.7%) was inflated by ~1.2pp relative to the disjoint test (76.5%). The leakage was mild but real.

**Fix applied:** Added `--exclude_path` flag to `sample_balanced_ids.py`. The disjoint test set excludes all train IDs, yielding 391t + 391f = 782 test entries with 0% overlap. The disjoint accuracy (76.5%) is actually closer to the paper's reported 76.9% than the inflated 77.7%.

**Why the paper gets away with it:** The paper's main evaluation uses out-of-distribution datasets (NQ-Open, BioASQ, NonExist) where there is zero overlap by construction. The in-domain TriviaQA number is secondary.

### 4.3 Answer Token Matching Failures in Activation Extraction (7 lost)

**Issue:** `extract_activations.py` matches answer tokens by exact string comparison of decoded tokens (line 89: `full_tokens[i : i + m] == answer_tokens`). The token list stored by `extract_answer_tokens.py` is decoded via `tokenizer.decode([tid])`, which can produce slightly different strings depending on tokenizer state (e.g., leading space handling, BOS token context).

**Impact:** 7 of 2,000 train IDs silently produced no activation file. The classifier silently skipped them.

**Fix:** Normalize whitespace before comparison, or store token IDs directly instead of decoded strings.

### 4.4 Tmux Session Loss During Long-Running Extraction

**Issue:** The initial tmux session for Step 2 died partway through (~346 of 3,115 entries), likely due to a shell or terminal event. The resume-safe design of the script prevented data loss, but wall time increased.

**Lesson:** For runs longer than a few minutes, tmux is essential. The script's resume support (checking already-processed QIDs on restart) worked correctly and saved ~$1 in API costs.

### 4.5 `all_except_answer_tokens` Crash on Null Answer Region

**Issue (fixed):** When `loc == "all_except_answer_tokens"` and the answer region couldn't be found (returned `None`), the code fell through to line 149 and hit a `NameError` on `selected_cett` which was never assigned. The control flow in lines 131–152 used a chain of `if/elif` that didn't cover this case.

**Fix applied:** Added an explicit `elif loc == "all_except_answer_tokens": continue` guard before the fallthrough logic.

### 4.6 No Resume Support in Activation Extraction (fixed)

**Issue (fixed):** `extract_activations.py` had no mechanism to skip already-extracted QIDs. Re-running after an interruption would redo all forward passes, wasting GPU time.

**Fix applied:** Added an existence check for the output `.npy` file at the top of the per-location loop, plus a QID-level pre-check that skips the forward pass entirely if all requested locations already exist.

### 4.7 GPT-4o Few-Shot Example Uses Wrong Tokenizer Convention

**Issue:** The few-shot example in `extract_answer_tokens.py` uses `▁` (sentencepiece underscore) as the space prefix marker. Gemma 3's tokenizer produces literal space prefixes (e.g., `" Spirit"` not `"▁Spirit"`). GPT-4o adapts to the actual token format in the user's input, so this doesn't cause failures, but it means the few-shot example is slightly misleading.

**Impact:** Minimal — GPT-4o is robust enough to generalize. But if the extraction were done by a weaker model, the format mismatch could cause systematic errors.

**Fix:** Generate the few-shot example dynamically using the actual target tokenizer.

### 4.8 `torch_dtype` Deprecation Warning

**Issue:** `extract_activations.py` line 102 uses `torch_dtype=torch.bfloat16`, which triggers a `FutureWarning` in recent transformers versions. The parameter has been renamed to `dtype`.

**Impact:** Cosmetic only. The model loads correctly.

**Fix:** Replace `torch_dtype` with `dtype`.

### 4.9 Sklearn `penalty` Deprecation Warning

**Issue:** `classifier.py` passes `penalty='l1'` to `LogisticRegression`, which is deprecated in sklearn 1.8+. The new API uses `l1_ratio=1` instead.

**Impact:** Cosmetic only. Training runs correctly.

**Fix:** Replace `penalty='l1'` with `l1_ratio=1` and remove the `penalty` argument.

### 4.10 Substrate Loophole in Rule Judge (96 entries)

**Issue (documented in batch review, not fixed):** The rule-based judge in `collect_responses.py` uses normalized substring matching. This means a response like `"William Shakespeare was a playwright from Stratford-upon-Avon"` matches `"shakespeare"` even though the response contains extra material that a strict short-answer judge would flag.

**Impact:** 96 of 1,435 all-correct entries may be false positives where the model produced "expanded answers" or "multi-entity outputs" that happened to contain the gold answer as a substring. These flow through the entire pipeline and slightly contaminate the "faithful" training class.

**Mitigation (not applied):** Could be addressed by adding a response-length or token-count filter, or by switching to an LLM judge for borderline cases. We chose not to filter them to match the paper's methodology.

---

## 5. Artifacts Produced

| File | Size | Description |
|------|------|-------------|
| `data/gemma3_4b_TriviaQA_consistency_samples.jsonl` | 3.3 MB | 3,500 questions × 10 responses |
| `data/gemma3_4b_answer_tokens.jsonl` | ~2.5 MB | 2,997 extracted answer token entries |
| `data/gemma3_4b_train_qids.json` | ~40 KB | 1,000t + 1,000f balanced train IDs |
| `data/gemma3_4b_test_qids.json` | ~40 KB | 1,000t + 1,000f test IDs (overlapping, legacy) |
| `data/gemma3_4b_test_qids_disjoint.json` | ~16 KB | 391t + 391f test IDs (0% overlap with train) |
| `data/activations/answer_tokens/` | 3.71 GB | 2,665 activation files |
| `data/activations/all_except_answer_tokens/` | 2.78 GB | 1,993 activation files |
| `models/gemma3_4b_classifier.pkl` | ~2.7 MB | Trained L1 logistic regression |

---

## 6. Drop-Off Funnel

```
3,500  TriviaQA questions sampled
  ↓ consistency filter (10/10 agreement)
3,115  consistent entries (89.0%)
  ↓ GPT-4o answer token extraction
2,997  extracted entries (96.2% of consistent)
  ↓ balanced sampling (1000 per class)
2,000  train IDs / 2,000 test IDs
  ↓ CETT activation extraction (token matching)
1,993  train activations (99.6%) / 1,993 test activations
  ↓ classifier training (3-vs-1, L1, C=1.0)
   38  H-Neurons identified (0.011% of 348,160)
```

---

## 7. Cost and Time Budget

| Stage | Wall time | GPU time | API cost | Notes |
|-------|-----------|----------|----------|-------|
| 1. Response collection | ~4 hours | ~4 hours | — | 3,500 × 10 generations, local GPU |
| 2. Answer token extraction | ~33 min | — | ~$3.50 | GPT-4o, includes 1 restart |
| 3. Balanced ID sampling | <1 sec | — | — | CPU only |
| 4. Activation extraction | ~86 sec | ~86 sec | — | 65s train + 21s test |
| 5. Classifier training | <5 sec | — | — | CPU only, sklearn |
| **Total** | **~4.5 hours** | **~4 hours** | **~$3.50** | |

The pipeline is dominated by Step 1 (response collection), which requires the model in GPU memory for ~4 hours. Steps 2–5 are cheap and fast. The entire replication fits on a single consumer GPU (RTX 5060 Ti, 16 GB VRAM) with no multi-GPU or cloud requirements for Gemma 3 4b only.

---

## 8. Critical Assessment

### What the replication confirms
- The core claim holds: an extremely sparse set of neurons (~0.01%) in a 4B-parameter model carries a detectable hallucination signal. Our 38 H-Neurons at 77.7% accuracy closely match the paper's ~35 at 76.9%.
- The CETT metric (activation × weight norm / output norm) is an effective neuron-level feature — a simple L1 logistic regression over 348,160 features achieves strong classification from this representation alone.
- The pipeline is reproducible with consumer hardware and minimal API cost (<$5 total).

### What the replication does NOT confirm (yet)
- **Generalization**: The paper's strongest claim is that H-Neurons trained on TriviaQA transfer to NQ-Open (70.7%), BioASQ (71.0%), and NonExist (71.9%). We have not tested this. However, the disjoint test (76.5% with 0% overlap) confirms the in-domain signal is real, not a leakage artifact.
- **Causal role**: Identifying neurons that *correlate* with hallucination is not the same as showing they *cause* it. The intervention experiments (Section 3 of the paper) are what establish causality — scaling H-Neurons changes hallucination rates on FalseQA, sycophancy, and jailbreak benchmarks. We haven't replicated this.
- **Stability across C values**: We used C=1.0 without a sweep. The exact neuron set is sensitive to the regularization parameter — a different C could yield 25 or 50 neurons with similar accuracy. The paper doesn't report a sensitivity analysis.

### Observations worth discussing
1. **Layer 20 dominance**: Neuron (20, 4288) has weight 12.17, 1.65× the runner-up. This is unusually dominant for a sparse classifier — it suggests a single neuron contributes disproportionately to hallucination detection. Is this a real "hallucination hub" or an artifact of L1's tendency to concentrate weight?
2. **Early-layer concentration**: 47% of H-Neurons are in layers 0–10. The paper doesn't break down layer distribution for Gemma 3 specifically. Early-layer hallucination features are surprising — they suggest the model "commits" to hallucinating before deep processing, not as a late-stage failure.
3. **Asymmetry with suppressive neurons**: The classifier also found 38 negative-weight neurons (hallucination-suppressing). The paper defines H-Neurons as positive-weight only. Are the suppressive neurons equally stable and transferable? This is unexplored territory.
4. **The 118 apostrophe-biased failures**: Losing entries containing O'Neill, O'Brien, possessives, etc. systematically removes a content category (Irish names, possessive constructions). If these happen to be disproportionately hallucinated or faithful, the training data has a subtle demographic bias.

---

## 9. Next Steps

### Priority 1 — Validate the replication (addresses Section 8 gaps)
1. ~~**Fix train/test overlap:**~~ **DONE.** Disjoint test (391t+391f, 0% overlap) yields 76.5% accuracy — signal confirmed as real, not leakage.
2. **Out-of-distribution evaluation:** Run on NQ-Open, BioASQ, and NonExist to replicate the full Table 1 row. This is the paper's main evidence for generalization and our strongest test of whether the 38 neurons are real.

### Priority 2 — Engineering cleanup
3. **Fix apostrophe bug:** Use `ast.literal_eval()` to recover the 118 lost entries and eliminate the content bias.
4. **C parameter grid search:** Sweep C to find the sparsity/accuracy Pareto frontier and test neuron-set stability.

### Priority 3 — Extend beyond the paper
5. **Intervention experiments:** Replicate Section 3 (behaviour impact) by scaling H-Neuron activations and measuring compliance rate changes on FalseQA, FaithEval, Sycophancy, and Jailbreak benchmarks.
6. **Origin analysis:** Apply the trained classifier to the base model (`google/gemma-3-4b-pt`) to test backward transferability (Section 4).
7. **Suppressive neuron investigation:** Characterize the 38 negative-weight neurons — are they stable across C values? Do they transfer OOD? The paper ignores them entirely.

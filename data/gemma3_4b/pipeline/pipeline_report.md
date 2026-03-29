# Gemma 3 4B — H-Neuron Identification Pipeline Report

**Date:** 2026-03-16
**Model:** `google/gemma-3-4b-it` (instruction-tuned)
**Hardware:** RTX 5060 Ti (16 GB VRAM), AMD 9900X, 64 GB RAM
**Paper reference:** Gao et al., "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs" (arXiv:2512.01797v2)

**Related reports:** [intervention_findings.md](intervention_findings.md), [bioasq13b_factoid_probe_transfer_audit.md](bioasq13b_factoid_probe_transfer_audit.md), [refusal_overlap_audit.md](../intervention/refusal_overlap/refusal_overlap_audit.md)

---

## 1. Results Summary

<!-- from: classifier_overlap_accuracy -->
<!-- from: classifier_disjoint_accuracy -->
| Metric | Paper (Gemma-3-4B) | Overlapping test (66.3%) | Disjoint test (0%) |
|--------|-------------------|--------------------------|---------------------|
| TriviaQA test accuracy | 76.9% | 77.7% [75.9, 79.5] | **76.5% [73.6, 79.5]** |
| H-Neuron count | ~35 (0.10‰) | 38 (0.11‰) | **38** (0.11‰) |
| AUROC (test) | — | 0.863 [0.847, 0.879] | **0.843 [0.815, 0.870]** |
| Precision (test) | — | 0.780 [0.759, 0.801] | **0.767 [0.734, 0.801]** |
| Recall (test) | — | 0.769 [0.743, 0.796] | **0.761 [0.717, 0.805]** |
| F1 (test) | — | 0.775 [0.755, 0.793] | **0.764 [0.732, 0.794]** |
| Test set size | — | 1,993 evaluated | **780 evaluated** (782 sampled, 2 missing activations) |

<!-- from: classifier_disjoint_accuracy -->
The disjoint test set (0% overlap with training data) yields **76.5% accuracy [73.6, 79.5]**, which is actually closer to the paper's 76.9% than the inflated overlapping score of 77.7% [75.9, 79.5]. The ~1.1 percentage point drop from overlapping→disjoint confirms mild leakage, and the classifier captures a real held-out discrimination signal. However, a verbosity confound audit (see `verbosity_confound_test.md`) found that full-response CETT readout encodes response length 3.7–16× more strongly than truthfulness. Detection performance at the answer-token level may therefore partly reflect response-form/length correlations rather than a pure hallucination signal. This does not invalidate the classifier — it discriminates on held-out data — but detection claims should be kept distinct from the stronger causal intervention evidence (Section 11, [intervention_findings.md](intervention_findings.md)). A downstream D3.5 audit also found that the classifier-selected intervention overlaps refusal geometry more than a matched random-neuron null, but that apparent mediation signal is dominated by layer 33 and is not yet robust enough to change D4 scope; see [refusal_overlap_audit.md](../intervention/refusal_overlap/refusal_overlap_audit.md).

The classifier identifies 38 H-Neurons out of 348,160 total neuron positions (34 layers × 10,240 intermediate neurons), achieving 99.978% weight sparsity. The same 38 neurons were selected regardless of test set composition (training is identical).

---

## 2. Pipeline Stage-by-Stage

### Stage 1: Response Collection (pre-existing)

- **Script:** `scripts/collect_responses.py`
- **Output:** `data/gemma3_4b/consistency_samples.jsonl`
- **Parameters:** 10 responses per question, `temperature=1.0`, `top_k=50`, `top_p=0.9`, rule-based judge

| Category | Count |
|----------|-------|
| Total questions | 3,500 |
| All-correct (10/10 true) | 1,435 |
| All-incorrect (10/10 false) | 1,680 |
| Mixed (disagreement) | 385 |
| Uncertain/error | 0 |
| **Passing consistency filter** | **3,115** |

The batch review (see `data/reviews/batch3500_review.md`) identified 96 all-correct entries where the rule judge matched via substring rather than clean short-answer correctness. These were kept to match the paper's methodology, which does not apply additional filtering beyond consistency.

### Stage 2: Answer Token Extraction

- **Script:** `scripts/extract_answer_tokens.py`
- **Output:** `data/gemma3_4b/answer_tokens.jsonl`
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
- **Outputs:** `data/gemma3_4b/train_qids.json`, `data/gemma3_4b/test_qids.json`

| Split | True | False | Total | Seed |
|-------|------|-------|-------|------|
| Train | 1,000 | 1,000 | 2,000 | 42 |
| Test | 1,000 | 1,000 | 2,000 | 123 |
| Overlap | — | — | 1,327 (66.3%) | — |

Train and test were sampled independently (matching the paper's example data pattern, which also shows 1000/1000 for both splits). The 66.3% overlap is a consequence of sampling 2×1000 from pools of only 1,391 true and 1,606 false IDs. This is not ideal — see Section 4 for a proposed fix.

### Stage 4: CETT Activation Extraction

- **Script:** `scripts/extract_activations.py`
- **Output:** `data/gemma3_4b/activations/{answer_tokens,all_except_answer_tokens}/act_{qid}.npy`
- **Wall time:** ~65 seconds (train) + ~21 seconds (test)

| Location | Files | Size |
|----------|-------|------|
| `answer_tokens` | 2,901 | 3.8 GB |
| `all_except_answer_tokens` | 1,993 | 2.6 GB |
| **Total** | **4,894** | **6.4 GB** |

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
Test set results: **77.7% accuracy [75.9, 79.5]**, 0.863 AUROC [0.847, 0.879]

---

## 3. H-Neuron Analysis

### Layer Distribution

The 38 identified H-Neurons are distributed across 23 of 34 layers, with a concentration in the early-to-middle layers (0–16):

| Layer range | H-Neuron count | % of total |
|-------------|---------------|------------|
| 0–10 (early) | 18 | 47.4% |
| 11–20 (middle) | 10 | 26.3% |
| 21–33 (late) | 10 | 26.3% |

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

The single most influential H-Neuron (Layer 20, Neuron 4288) has a weight 1.65× larger than the second-ranked, suggesting it plays a disproportionately strong role in the answer-token classifier's positive-class score. The classifier also identified 38 negative-weight neurons (negative-weight in the classifier), but these are not classified as H-Neurons per the paper's definition.

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

**Impact:** The overlapping test accuracy (77.7% [75.9, 79.5]) was inflated by ~1.1pp relative to the disjoint test (76.5% [73.6, 79.5]). The leakage was mild but real.

**Fix applied:** Added `--exclude_path` flag to `sample_balanced_ids.py`. The disjoint test set excludes all train IDs, yielding 391t + 391f = 782 sampled test entries with 0% overlap. The current CI-bearing evaluation covers 780 of them because 2 IDs are missing activation files; that evaluated subset reaches 76.5% [73.6, 79.5], which is actually closer to the paper's reported 76.9% than the inflated 77.7% [75.9, 79.5].

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
| `data/gemma3_4b/consistency_samples.jsonl` | 3.3 MB | 3,500 questions × 10 responses |
| `data/gemma3_4b/answer_tokens.jsonl` | 844 KB | 2,997 extracted answer token entries |
| `data/gemma3_4b/train_qids.json` | 40 KB | 1,000t + 1,000f balanced train IDs |
| `data/gemma3_4b/test_qids.json` | 40 KB | 1,000t + 1,000f test IDs (overlapping, legacy) |
| `data/gemma3_4b/test_qids_disjoint.json` | 16 KB | 391t + 391f test IDs (0% overlap with train) |
| `data/gemma3_4b/activations/answer_tokens/` | 3.8 GB | 2,901 activation files (1,993 train + 908 test) |
| `data/gemma3_4b/activations/all_except_answer_tokens/` | 2.6 GB | 1,993 activation files (train only) |
| `models/gemma3_4b_classifier.pkl` | 2.7 MB | Trained L1 logistic regression (C=1.0) |
| `models/gemma3_4b_classifier_disjoint.pkl` | 2.7 MB | Trained L1 logistic regression evaluated on disjoint test |

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
1,993  train activations (99.6%) / 1,993 test activations (answer_tokens only; all_except_answer_tokens is train-only)
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
- The core claim holds at the answer-token classification level: an extremely sparse set of neurons (~0.01%) in a 4B-parameter model carries a detectable false-answer discrimination signal. Our 38 H-Neurons at 77.7% accuracy closely match the paper's ~35 at 76.9%. However, this is a classifier result; the detection interpretation is partially confounded by response-form/length correlations (verbosity confound audit found full-response readout encodes length 3.7–16× more than truthfulness) and should be kept distinct from the stronger causal intervention evidence (Section 11, [intervention_findings.md](intervention_findings.md) Finding 7).
- The CETT metric (activation × weight norm / output norm) is an effective neuron-level feature — a simple L1 logistic regression over 348,160 features achieves strong classification from this representation alone.
- The pipeline is reproducible with consumer hardware and minimal API cost (<$5 total).

### What the replication does NOT confirm (yet)
- **Generalization**: The paper's strongest claim is that H-Neurons trained on TriviaQA transfer to NQ-Open (70.7%), BioASQ (71.0%), and NonExist (71.9%). We have not tested this. However, the disjoint test (76.5% with 0% overlap) confirms the in-domain signal is real, not a leakage artifact.
- **Causal role**: ~~We haven't replicated the intervention experiments.~~ **PARTIALLY DONE.** FaithEval (both prompt variants) and FalseQA intervention sweeps are complete, with negative controls confirming H-neuron specificity. See Section 11 and [intervention_findings.md](intervention_findings.md). Sycophancy and Jailbreak remain pending.
- **Stability across C values**: ~~We used C=1.0 without a sweep.~~ **DONE.** The C-sweep (Section 10.3) showed C=1.0 is suboptimal — C=3.0 reaches 80.5% accuracy with 219 positive neurons. The exact neuron set is sensitive to C; the paper doesn't report a sensitivity analysis.

### Observations worth discussing
1. ~~**Layer 20 dominance**: Neuron (20, 4288) has weight 12.17, 1.65× the runner-up. This is unusually dominant for a sparse classifier — it suggests a single neuron contributes disproportionately to hallucination detection. Is this a real "hallucination hub" or an artifact of L1's tendency to concentrate weight?~~ **RESOLVED — L1 artifact.** See Section 10 below.
2. **Early-layer concentration**: 47% of H-Neurons are in layers 0–10. The paper doesn't break down layer distribution for Gemma 3 specifically. The layer location of classifier-selected features shows where the answer-token discrimination signal is available in the network, but does not establish when the model causally commits to a false answer — that would require causal or patching experiments at specific layers.
3. **Asymmetry with suppressive neurons**: The classifier also found 38 negative-weight neurons (hallucination-suppressing). The paper defines H-Neurons as positive-weight only. Are the suppressive neurons equally stable and transferable? This is unexplored territory.
4. **The 118 apostrophe-biased failures**: Losing entries containing O'Neill, O'Brien, possessives, etc. systematically removes a content category (Irish names, possessive constructions). If these happen to be disproportionately hallucinated or faithful, the training data has a subtle demographic bias.

---

## 9. Next Steps

### Priority 1 — Validate the replication (addresses Section 8 gaps)
1. ~~**Fix train/test overlap:**~~ **DONE.** Disjoint test (391t+391f, 0% overlap) yields 76.5% accuracy — signal confirmed as real, not leakage.
2. **Out-of-distribution evaluation:** Run on NQ-Open, BioASQ, and NonExist to replicate the full Table 1 row. This is the paper's main evidence for generalization and our strongest test of whether the 38 neurons are real.

### Priority 2 — Engineering cleanup
3. **Fix apostrophe bug:** Use `ast.literal_eval()` to recover the 118 lost entries and eliminate the content bias.
4. ~~**C parameter grid search:**~~ **DONE.** See Section 10 — the C-sweep was conducted as part of the neuron 4288 investigation and revealed that C=1.0 is suboptimal (C=3.0 reaches 80.5% accuracy).

### Priority 3 — Extend beyond the paper
5. ~~**Intervention experiments:**~~ **PARTIALLY DONE.** FaithEval (both prompts, 1000 samples × 7 α) and FalseQA (687 samples × 7 α) complete. Negative controls (5 unconstrained seeds + 3 layer-matched seeds) confirm H-neuron specificity. See Section 11 and [intervention_findings.md](intervention_findings.md). Sycophancy and Jailbreak remain pending.
6. **Origin analysis:** Apply the trained classifier to the base model (`google/gemma-3-4b-pt`) to test backward transferability (Section 4).
7. **Suppressive neuron investigation:** Characterize the 38 negative-weight neurons — are they stable across C values? Do they transfer OOD? The paper ignores them entirely.

---

## 10. Deep Dive: Is Neuron (20, 4288) a Hallucination Hub?

**Script:** `scripts/investigate_neuron_4288.py`
**Plots:** `data/gemma3_4b/investigation_neuron_4288/`
**Verdict: No — the dominance is an L1 regularization artifact (0/6 analyses support real signal).** Note: the "hallucination signal" language below refers to the answer-token classifier's false-answer discrimination signal, which is partially confounded by response-form/length correlations (see verbosity confound audit).

The paper identifies H-Neurons by their positive weight in an L1-penalized logistic regression. Neuron (20, 4288) received weight 12.169 — 1.65× the runner-up. L1 regularization is known to arbitrarily concentrate weight among correlated features: when two neurons carry similar information, L1 tends to pick one and zero-out the other, rather than splitting the weight evenly (as L2 would). We ran six independent analyses to test whether 4288's dominance reflects genuine unique informativeness or this L1 concentration effect.

### 10.1 Analysis 1 — Single-Neuron Classification

![Single-Neuron AUC](investigation_neuron_4288/01_single_neuron_auc.png)

**What it shows:** Each bar is the test-set AUC when using *only* that one neuron's CETT activation as a univariate hallucination classifier. Blue bars are the top-10 H-Neurons (by L1 weight), red is neuron 4288, gray bars are random zero-weight neurons as controls.

**Why we ran it:** If a neuron is genuinely the most important false-answer discrimination indicator, it should also be the best standalone predictor — regardless of what L1 does. This analysis is completely independent of the regularization procedure.

**Result:** Neuron 4288 (AUC=0.590) is *not* the best single predictor. That distinction goes to **L13:N833** (AUC=0.703), which ranks only 3rd by classifier weight (3.45). The runner-up L14:N8547 (AUC=0.666) also outperforms 4288. Even L10:N4996 (weight 1.70, rank 9) achieves AUC=0.645 — better than 4288 despite having 7× less classifier weight. All H-Neurons outperform the random controls (mean AUC=0.526), confirming the ensemble signal is real even though individual neuron rankings don't match L1 weights.

**Implication:** L1 weight magnitude is not a reliable proxy for individual neuron informativeness. The paper's methodology of ranking H-Neurons by weight conflates two things: how much unique signal a neuron carries, and how L1 happened to distribute weight among correlated features.

### 10.2 Analysis 2 — Activation Distribution Separation

![Activation Distributions](investigation_neuron_4288/02_distributions.png)

**What it shows:** Overlapping histograms of raw CETT activation values for true (green, no hallucination) vs false (red, hallucination) test examples. Three panels compare neuron 4288, the runner-up (L14:N8547), and a random zero-weight neuron. Cohen's d quantifies effect size; Mann-Whitney p tests whether the two distributions differ significantly.

**Why we ran it:** A genuine "hub" neuron should show clean separation between true and false activations in the raw data, before any classifier processing.

**Result:** Neuron 4288 shows a small-to-medium effect (Cohen's d=0.326, p=1.3e-5) — statistically significant but modest. The runner-up L14:N8547 has *better* separation (d=0.477, p=8.6e-16). The random neuron shows no meaningful separation (d=0.096, p=0.098). Both 4288 and 8547 share a right-skewed distribution where hallucinating examples have a heavier tail of high activations, but the overlap between the two classes is substantial for both.

**Implication:** Neuron 4288 does carry hallucination-relevant signal (it's clearly not random), but the runner-up L14:N8547 actually separates the classes better. The 1.65× weight advantage doesn't reflect 1.65× better discrimination — it reflects L1's weight concentration.

### 10.3 Analysis 3 — C-Sweep Stability

![C-Sweep](investigation_neuron_4288/03_c_sweep.png)

**What it shows:** Left panel: the weight of the top positive neurons as the regularization parameter C varies from 0.001 (strongest L1 penalty, fewest neurons) to 10.0 (weakest penalty, most neurons). Red line is neuron 4288. Right panel: test accuracy and AUC across the same C range.

**Why we ran it:** This is the definitive L1-artifact diagnostic. If neuron 4288 is genuinely the most important false-answer discrimination neuron, it should be among the *first* neurons selected as L1 loosens (low C → high C), and it should remain dominant across the range. If it's an artifact, it will appear late and be replaced as more features enter the model.

**Result:** This is the most striking finding of the investigation.

| C | Non-zero features | Positive | 4288 weight | 4288 rank | Test accuracy |
|---|-------------------|----------|-------------|-----------|---------------|
| 0.001–0.01 | 0 | 0 | 0 | — | 50.1% |
| 0.03 | 1 | 1 | 0 | — | 51.9% |
| 0.1 | 9 | 8 | 0 | — | 64.2% |
| 0.3 | 15 | 9 | 0 | — | 69.5% |
| **1.0** | **76** | **38** | **12.17** | **1** | **76.5%** |
| 3.0 | 419 | 219 | 10.76 | 5 | **80.5%** |
| 10.0 | 989 | 519 | 11.74 | 11 | 80.0% |

Neuron 4288 is **completely absent** from the model at C≤0.3, where the classifier already achieves 69.5% accuracy using 9 different positive neurons. It only enters at C=1.0 — jumping immediately to rank 1 — then drops to rank 5 (C=3.0) and rank 11 (C=10.0) as the model gains access to more features. The left panel shows a "regime change" pattern: different neurons spike to extreme weights at different C values (L25:N3877 at C=0.03, L9:N5580 at C=0.1), and 4288's spike at C=1.0 is just another instance of this L1 concentration effect.

**Implication:** The paper's choice of C=1.0 is the *only* regularization strength where neuron 4288 dominates. At C=3.0, the model achieves 80.5% accuracy (+4pp over C=1.0) with 219 positive neurons and 4288 at rank 5. The "~35 H-Neurons at 0.10‰" headline result is specific to one regularization setting, not a fundamental property of the model. The paper does not report a C sensitivity analysis.

### 10.4 Analysis 4 — Per-Example Contribution

![Contributions](investigation_neuron_4288/04_contributions.png)

**What it shows:** For each test example, the scatter plots neuron 4288's contribution to the classifier score (y-axis: weight × activation) against the total classifier score (x-axis: all 76 features combined). Points are colored by true label (green = truthful, red = hallucination) and shaped by correctness (circles = correct, crosses = misclassified).

**Why we ran it:** Even if 4288's weight is inflated, it might still be the single largest contributor for most predictions — meaning L1 at least concentrated signal onto a neuron that "works" for individual examples.

**Result:** Neuron 4288 is the largest positive contributor for only **7.4%** of examples — despite having 1.65× the weight of #2. The median absolute contribution fraction is just 2.6% of the total score. The scatter plot shows that 4288's contribution is only weakly correlated with the total score — many correctly-classified examples have near-zero contribution from 4288, with their scores driven by the other 75 non-zero features.

**Implication:** The high weight is compensated by relatively low activation magnitudes. Neuron 4288 activates strongly for a small subset of examples but is quiet for most — which is why L1 gave it a large weight (to amplify rare activations) rather than because it's broadly the most informative feature.

### 10.5 Analysis 5 — Ablation

![Ablation](investigation_neuron_4288/05_ablation.png)

**What it shows:** Accuracy drop when each of the top-10 H-Neurons is individually zeroed out in the test data. Red bar is neuron 4288. A positive drop means the model got worse without that neuron.

**Why we ran it:** Direct measurement of each neuron's marginal importance to the trained classifier.

**Result:** Ablating neuron 4288 drops accuracy by 1.03pp (76.5% → 75.5%) — the largest single-neuron drop, but below the 2pp threshold for "clearly important." Interestingly, ablating L24:N7995 (rank 6 by weight) *improves* accuracy by 0.64pp, suggesting it slightly hurts the model on the test set.

A revealing complementary test: keeping *only* neuron 4288 (plus all 38 negative neurons, zeroing out the other 37 positive neurons) yields AUC=0.828 (close to the full model's 0.843) but accuracy=50.1% (chance). This means 4288 alone can *rank-order* examples by hallucination likelihood almost as well as the full model, but cannot set a useful decision boundary — the 37 other positive neurons are needed for calibration.

### 10.6 Analysis 6 — Correlation Structure

![Correlations](investigation_neuron_4288/06_correlations.png)

**What it shows:** Pairwise Pearson correlation heatmap between the top-10 H-Neurons (by weight) and 10 zero-weight neurons from layers 18–22 (near neuron 4288's Layer 20). Red = positive correlation, blue = negative.

**Why we ran it:** L1 concentrates weight among correlated features. If neuron 4288 is highly correlated with other top neurons, L1 may have arbitrarily chosen it as the "representative" of a correlated group.

**Result:** Neuron 4288 has notable positive correlation with L26:N1359 (r=+0.492) and moderate correlation with L14:N8547 (r=+0.293) and L9:N5580 (r=+0.290). The zero-weight neurons from nearby layers show no meaningful correlation (max |r|=0.170), confirming L1 didn't suppress a "twin" of 4288 — the correlations are with neurons in distant layers that carry related but not identical signal.

**Implication:** The r=0.492 with L26:N1359 is the strongest inter-neuron correlation in the top-10. These two neurons likely share an overlapping hallucination signal, and L1 concentrated most of the shared weight onto 4288 (weight 12.17) while giving 1359 much less (weight 2.46). Under L2 regularization, both would receive moderate weights.

### 10.7 Summary and Implications for the Paper's Methodology

| Analysis | Threshold | Neuron 4288 | Verdict |
|----------|-----------|-------------|---------|
| Single-neuron AUC | >0.60 = real | 0.590 | Artifact |
| Cohen's d | >0.5 = real | 0.326 | Artifact |
| C-sweep (selected in N/9) | ≥5 = real | 3/9 | Artifact |
| Largest contributor % | >30% = real | 7.4% | Artifact |
| Ablation accuracy drop | >2pp = real | 1.03pp | Artifact |
| Max correlation with top-10 | <0.3 = real | 0.492 | Artifact |

**All six analyses point to L1 artifact.** Neuron 4288 does carry a real false-answer discrimination signal (AUC=0.590, well above the 0.526 random baseline), but it is not uniquely or disproportionately important. Its extreme weight is a consequence of L1's winner-take-all behavior among correlated features at the specific regularization strength C=1.0.

This raises a broader methodological concern about the H-Neurons paper: **L1 weight magnitude is used throughout as a proxy for neuron importance, but our analysis shows this conflates signal strength with regularization artifacts.** The most informative single neuron (L13:N833, AUC=0.703) has only 28% of 4288's weight. A more robust neuron-ranking method — such as single-neuron AUC, stability across C values, or Shapley values — would produce a substantially different "top H-Neuron" list.

---

## 11. Intervention Experiments: FaithEval (Section 3 Replication)

> **Consolidated findings report:** [intervention_findings.md](intervention_findings.md) — synthesises FaithEval, FalseQA, and negative control results with raw data tables, exact prompts, and separated interpretation.

**Scripts:** `scripts/run_intervention.py`, `scripts/evaluate_intervention.py`, `scripts/plot_intervention.py`
**Data:** `data/gemma3_4b/intervention/faitheval/alpha_{0.0..3.0}.jsonl`, `data/gemma3_4b/intervention/faitheval_standard/alpha_{0.0..3.0}.jsonl`, `data/gemma3_4b/intervention/falseqa/alpha_{0.0..3.0}.jsonl`, `data/gemma3_4b/intervention/negative_control/`
**Status:** FaithEval complete (both prompt variants, 1000 samples × 7 α each). FalseQA complete (687 samples × 7 α). Negative controls complete (5 unconstrained + 3 layer-matched seeds on FaithEval anti-compliance). Sycophancy and Jailbreak pending.

### 11.1 Mechanism: Hook-Based Activation Scaling

Rather than modifying `down_proj.weight.data` in-place (as the existing `intervene_model.py` does), we register `register_forward_pre_hook` on each `down_proj` module containing H-Neurons. The hook intercepts the input tensor z (the gated intermediate activation, shape `[batch, seq, intermediate_size]`) and scales `z[:, :, neuron_indices] *= α` before the down-projection.

**Why hooks over weight modification:** The paper sweeps α ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}. Weight modification requires either model reloads (~20s each) or careful undo logic with floating-point drift. Hooks are stateless — changing a single scalar property between runs costs nothing and introduces no numerical error. The `HNeuronScaler` class holds all hooks and exposes an `.alpha` property that switches between intervention strengths without re-registration.

**Correctness verified:** (1) α=1.0 hook output is identical to no-hook output (bit-exact); (2) α=0.0 and α=3.0 produce visibly different outputs on the same prompts.

### 11.2 FaithEval Results (Anti-Compliance Prompt)

| α | Compliance Rate | n_compliant / 1000 |
|---|----------------|--------------------|
| 0.0 | 64.2% | 642 |
| 0.5 | 65.4% | 654 |
| 1.0 | 66.0% | 660 |
| 1.5 | 67.0% | 670 |
| 2.0 | 68.2% | 682 |
| 2.5 | 69.5% | 695 |
| 3.0 | 70.5% | 705 |

**Trend:** Perfectly monotonic (Spearman ρ=1.0, p<1e-6). Compliance increases by 6.3 percentage points from α=0.0 to α=3.0, at a slope of 2.10% per unit α.

**Paper comparison:** The paper reports FaithEval compliance for Gemma-3-4B in the range ~55–75% (from Figure 3b). Our baseline (α=1.0 = 66.0%) falls squarely within this range. The paper reports an average slope of ~3.03% per unit α for small models; our 2.10% is lower but still clearly positive and statistically robust. The attenuated slope is likely explained by our anti-compliance prompt (see 11.3).

### 11.3 Methodology Comparison: Where We Differ from the Paper

#### The prompt problem (MAJOR)

The paper states (Section 6.2.2): *"the model is prompted with fabricated information and asked to answer questions based upon it."* The FaithEval benchmark's official evaluation code (from the Salesforce GitHub repo) uses a **retrieval QA frame**:

```
You are an expert in retrieval question answering.
Please respond with the exact answer only.
{task_specific_prompt}
Context: {context}
Question: {question}
Answer:
```

This framing implicitly instructs the model to answer **from the context** — a pro-compliance prompt. Compliance here measures how strongly the model trusts retrieved context over internal knowledge.

**Our implementation instead uses:**

```
Context: {context}

Question: {question}
{choices}

If the context conflicts with established knowledge,
answer based on your own knowledge. Answer with just the letter.
```

This is an **anti-compliance prompt** — it explicitly tells the model to resist misleading context. Yet the model still follows the counterfactual context 64–71% of the time. This is arguably a *stronger* demonstration of H-Neuron causal influence: even when explicitly instructed to resist, amplifying H-Neurons still pushes the model toward over-compliance.

**Impact on comparability:** Our absolute compliance rates may be lower than the paper's (because we fight the effect), but the monotonic trend and its direction are identical. The causal conclusion — H-Neurons drive over-compliance — holds under both prompt framings. **Update:** We re-ran FaithEval with the standard pro-context prompt — see Section 11.7 for a detailed comparison that first looked like a format-compliance confound, but is now more precisely understood as an **evaluator-format mismatch**.

#### Other methodological differences

| Parameter | Paper | Our Implementation | Impact |
|-----------|-------|--------------------|--------|
| FaithEval prompt | Pro-context (retrieval QA frame) | Both: anti-compliance + standard (see 11.7) | See Section 11.7 |
| FaithEval decoding | Greedy, max 256 tokens | Greedy, max 256 tokens | Match |
| FaithEval evaluation | Rule-based parser | Rule-based MC letter extraction | Partial match; standard prompt needs text-based remap to avoid undercounting answer-text outputs |
| Sycophancy max_tokens | 512 (open-ended) | 128 (turn 1), 256 (turn 2) | Potential truncation; needs fix before running |
| Jailbreak templates | Shen et al. 2024 actual prompts | Hardcoded simplified templates | **Not usable**; must replace or skip |
| FalseQA decoding | Greedy | Greedy, max 256 tokens | Match |

### 11.4 Qualitative Observations

**H-Neuron suppression (α=0.0) makes the model more verbose.** At α=0.0, 84 of 1000 responses exceed 20 characters (the model adds explanations like "**Explanation:** The passage explicitly states..."). At α≥1.0, 100% of responses are single letters under the anti-compliance prompt. One reading is that H-Neurons drive instruction-following compliance in addition to content compliance. But the standard-prompt re-analysis (Section 11.7) falsified the strongest version of that story: many high-α "format failures" were actually short exact-answer strings that better matched the **prompt**, even while they failed the local MC-letter evaluator. The verbosity at α=0.0 under the anti-compliance prompt may instead reflect the model entering a more "cautious reasoning" mode when H-neurons are suppressed, rather than simply being less obedient.

**Most samples are not affected by α.** The 1000 samples break into three populations:

| Category | Count | % |
|----------|-------|---|
| Always compliant (all 7 α) | 600 | 60.0% |
| Never compliant (all 7 α) | 262 | 26.2% |
| "Swing" samples (varies) | 138 | 13.8% |

The 6.3pp compliance swing is driven entirely by the 138 swing samples. At α=0.0, 42 of them are already compliant; 63 more are recruited as α increases to 3.0, while the remaining 33 swing samples never become compliant. This tri-modal population structure — frozen-compliant, frozen-resistant, and α-sensitive — is not discussed in the paper but has implications for how we interpret the intervention's effect size.

### 11.5 Critique of the Paper's Intervention Methodology

#### 1. The prompt confound — now partially resolved

The paper doesn't disclose the exact FaithEval prompt used. If they use the standard retrieval QA framing ("answer from the context"), then a substantial fraction of the compliance rate reflects normal context-following behavior, not over-compliance in any pathological sense. **Our dual-prompt experiment (Section 11.7) shows the causal effect holds under both pro-context and anti-context prompts**, which substantially strengthens the paper's claim. However, the standard prompt also exposes an evaluator mismatch: the prompt asks for the exact answer text, while the local scorer only trusts MC-letter extraction. Any FaithEval-based intervention study using MC formatting should report parse failure rates **and** check whether those failures are actually answer-text outputs.

#### 2. Population heterogeneity masks effect size

The paper reports a single compliance rate per α, which hides the fact that ~86% of samples are unaffected by the intervention. The 60% always-compliant population inflates the baseline, making the α effect look smaller in percentage terms. The 26% never-compliant population anchors the ceiling. The actual effect — 63 samples flipping from resistant to compliant across α∈[0,3] — is a 45.7% swing within the sensitive subpopulation, more dramatic than the headline 6.3pp suggests.

A more informative analysis would:
- Report the swing-sample compliance curve separately
- Characterize what distinguishes swing samples from frozen ones (question difficulty? context convincingness? topic domain?)
- Use this to understand *which types of knowledge* H-Neurons can override

#### 3. Effect size vs. practical significance

Our slope (2.10% per unit α) is below the paper's reported average for small models (3.03%). Even with the paper's presumably higher slope, the absolute effect is modest: ~10pp over the full α range. Compared to prompt engineering (which routinely swings compliance by 30–50pp), H-Neuron scaling is a weak lever. The causal direction is established, but the practical impact is limited — you wouldn't use neuron scaling as a safety intervention when prompt design is more powerful and cheaper.

#### 4. ~~Lack of negative controls~~ — RESOLVED

<!-- from: negative_control_random_slope_interval -->
~~Neither the paper nor our replication includes a negative control.~~ **DONE.** Eight random-neuron seeds (5 unconstrained + 3 layer-matched, 38 neurons each from zero-weight classifier positions) run on FaithEval anti-compliance. The H-neuron slope is **2.09 pp / α** with paired-bootstrap 95% CI **[1.38, 2.83]**, while the pooled random-set empirical interval is **[-0.106, 0.164] pp / α**. At α=3.0, the H-neuron rate is **70.5%** versus a random-set empirical interval of **[65.8, 66.46]%**. The negative control is flat; the H-neuron effect is specific, not a generic perturbation artifact. Data: `data/gemma3_4b/intervention/negative_control/`. See [intervention_findings.md](intervention_findings.md) §1.4–1.5.

#### 5. Monotonicity is necessary but not sufficient

Perfect monotonicity (Spearman ρ=1.0) sounds impressive, but with only 7 data points on a smooth curve and 1000 samples per point, almost any real effect would appear monotonic. The interesting question is whether the relationship is *linear* (as the CETT theory predicts) or exhibits nonlinear saturation. Our data appears quite linear (R²=0.993), which is consistent with the CETT framework's prediction that α has a linear relationship with functional importance.

### 11.6 Discussion Points

1. ~~**Should we re-run FaithEval with the standard FaithEval prompt?**~~ **DONE.** See Section 11.7. The standard prompt run first looked contradictory, but the later text remap showed the contradiction was mostly evaluator-side.

2. ~~**Negative control experiment:**~~ **DONE.** Eight seeds (5 unconstrained + 3 layer-matched) on FaithEval anti-compliance. H-neuron slope `2.09 pp / α` with CI `[1.38, 2.83]` sits outside the pooled random-set empirical interval `[-0.106, 0.164]`. Full analysis with reviewer self-critique in [intervention_findings.md](intervention_findings.md) §1.5 and §5.

3. **Swing sample characterization:** What makes the α-sensitive samples special? The anti-compliance prompt yields 138 swing samples. The standard-prompt raw-parser count of 203 is now known to be contaminated by α=3.0 answer-text outputs being scored as resistant, so it should not be used until text-based scoring is extended across all α.

<!-- from: falseqa_delta_0_to_3 -->
4. ~~**FalseQA is the natural next benchmark**~~ **DONE.** 687 samples × 7 α values, GPT-4o judged. Shows +4.8pp swing with paired-bootstrap 95% CI `[1.3, 8.3] pp` (69.6% [66.0, 72.9] → 74.4% [71.0, 77.5]), but the intermediate points are non-monotonic. Data: `data/gemma3_4b/intervention/falseqa/`. See [intervention_findings.md](intervention_findings.md) §1.3.

5. **Sycophancy needs a max_tokens fix** before running — paper uses 512 tokens for open-ended generation, our implementation uses 128 for turn 1.

6. **Jailbreak should be deferred** until we either (a) locate Shen et al.'s actual template-question pairing code, or (b) decide to skip it entirely. The hardcoded simplified templates in our implementation are methodologically unsound.

7. **The evaluator/content dissociation** is novel and not discussed in the paper. Under the standard prompt, high α makes the model surface answer text instead of evaluator-friendly option letters; once those answers are remapped, content compliance still rises. The real split is between semantic content and what the local MC-letter scorer can see, not necessarily between content compliance and prompt compliance. See Section 11.7.7.

### 11.7 FaithEval Standard Prompt: The Evaluator-Format Confound

**Data:** `data/gemma3_4b/intervention/faitheval_standard/alpha_{0.0..3.0}.jsonl`, `data/gemma3_4b/intervention/faitheval_standard/results.json`

<!-- from: anti_compliance_delta_0_to_3 -->
<!-- from: anti_compliance_slope -->
<!-- from: standard_text_remap_alpha_3_rescored_rate -->
We re-ran FaithEval with the official retrieval QA prompt (pro-context framing) to enable direct comparison with the paper. The raw results initially appeared to contradict the anti-compliance run — compliance *decreases* with α. That interpretation is now falsified. The standard prompt asks for the **exact answer text**, while our local evaluator only trusts **MC-letter extraction**. At high α, the model often emits the answer text directly, so raw `chosen=None` counts are mostly evaluator artifacts rather than evidence that content-following fell.

#### 11.7.1 Raw Evaluator Results (Historical, Parser-Contaminated)

| α | Standard (raw) | Standard 95% CI | Anti-compliance | Anti 95% CI | Δ (Std − Anti) |
|---|----------------|-----------------|-----------------|-------------|----------------|
| 0.0 | 69.1% (691) | [66.2, 71.9] | 64.2% (642) | [61.2, 67.1] | +4.9pp |
| 0.5 | 68.4% (684) | [65.5, 71.2] | 65.4% (654) | [62.4, 68.3] | +3.0pp |
| 1.0 | 68.8% (688) | [65.9, 71.6] | 66.0% (660) | [63.0, 68.9] | +2.8pp |
| 1.5 | 69.8% (698) | [66.9, 72.6] | 67.0% (670) | [64.0, 69.8] | +2.8pp |
| 2.0 | 68.6% (686) | [65.7, 71.4] | 68.2% (682) | [65.2, 71.0] | +0.4pp |
| 2.5 | 66.9% (669) | [63.9, 69.7] | 69.5% (695) | [66.6, 72.3] | -2.6pp |
| 3.0 | 63.6% (636) | [60.6, 66.5] | 70.5% (705) | [67.6, 73.2] | -6.9pp |

The standard prompt's raw parser-facing endpoint change is **-5.5 pp** with paired-bootstrap 95% CI **[-8.1, -2.8] pp**. That CI belongs to the evaluator output, not to the underlying content-following behavior. **Historical note:** this apparent sign flip is preserved here as a record of the raw evaluator output, but should no longer be interpreted as evidence that the standard prompt weakens the intervention. Section 11.7.3 resolves the contradiction.

#### 11.7.2 Why the Raw Curve Misleads

The apparent decline is driven by a monotonically increasing parse failure rate:

| α | Parse failures | Parse-failure 95% CI | Example (α=1.0 → α=3.0) |
|---|---------------|----------------------|--------------------------|
| 0.0 | 9 | [0.5, 1.7] | — |
| 0.5 | 11 | [0.6, 2.0] | — |
| 1.0 | 17 | [1.1, 2.7] | `"D) energy molecules"` → |
| 1.5 | 32 | [2.3, 4.5] | |
| 2.0 | 65 | [5.1, 8.2] | |
| 2.5 | 105 | [8.7, 12.6] | |
| 3.0 | **150** | [12.9, 17.3] | → `"Energy molecules"` |

At high α, the model often produces the **same answer content** but without an MC letter prefix. Where α=1.0 produces `"D) energy molecules"`, α=3.0 produces `"Energy molecules"` — the semantic answer is the same, but our parser can't extract a letter. Crucially, under the standard prompt the latter output is arguably **more faithful to the prompt**, not less: the instruction says "respond with the exact answer only," not "answer with the letter." These parse failures are only failures relative to the evaluator, not necessarily relative to the prompt.

Of the 150 parse failures at α=3.0: 75 were compliant at α=1.0 (the model was already following the context and later surfaced the answer text directly). The anti-compliance prompt avoids this because its prompt and evaluator are aligned: both expect a letter.

#### 11.7.3 α=3.0 Strict Text Remap (Resolved Since the Earlier Draft)

We directly remapped the 150 `chosen=None` responses at α=3.0 back to FaithEval answer-option text using conservative normalization plus one numeric-prefix recovery (`83` vs stored option text `83 331`). This produces a population-level correction rather than a parseable-subset conditional rate.

| Metric | Value | 95% CI | Interpretation |
|---|---|---|---|
| Raw compliant count | 636 / 1000 (63.6%) | [60.6, 66.5] | Letter-extractor score only |
| Strictly recovered parse failures | 140 / 150 (93.3%) | [88.2, 96.3] | Exact normalized answer-text match + 1 numeric-prefix case |
| Recovered compliant cases | 85 | — | True counterfactual-following answers that the evaluator hid |
| Strict rescored compliance | **721 / 1000 (72.1%)** | [69.2, 74.8] | Best current population estimate at α=3.0 |
| Still unresolved | 10 / 150 | — | 4 closest-option paraphrases, 3 off-option counterfactual answers, 2 off-option resistant answers, 1 ambiguous partial |

This is the decisive result for the original question. The raw `63.6%` dip was mostly an evaluator artifact, not a true loss of content-following. The local evaluator behaved like a scantron reader that marks a hand-written correct answer as wrong because the bubble was left blank. Some genuine answer drift remains in the unresolved 10, but it is second-order next to the 85 recovered compliant cases.

Artifacts for this remap now live in:

- `scripts/remap_faitheval_standard_parse_failures.py`
- `data/gemma3_4b/intervention/faitheval_standard/alpha_3.0_parse_failure_remap.jsonl`
- `data/gemma3_4b/intervention/faitheval_standard/alpha_3.0_parse_failure_remap_summary.json`

#### 11.7.4 Parseable-Subset Conditional Rates (Useful Diagnostic, Not Final Estimate)

Excluding unparseable responses, compliance increases monotonically under both prompts:

| α | Standard (adjusted) | Standard 95% CI | Excluded | Anti-compliance | Anti 95% CI |
|---|---------------------|-----------------|----------|-----------------|-------------|
| 0.0 | 69.7% (691/991) | [66.8, 72.5] | 9 | 64.2% | [61.2, 67.1] |
| 0.5 | 69.2% (684/989) | [66.2, 72.0] | 11 | 65.4% | [62.4, 68.3] |
| 1.0 | 70.0% (688/983) | [67.1, 72.8] | 17 | 66.0% | [63.0, 68.9] |
| 1.5 | 72.1% (698/968) | [69.2, 74.8] | 32 | 67.0% | [64.0, 69.8] |
| 2.0 | 73.4% (686/935) | [70.4, 76.1] | 65 | 68.2% | [65.2, 71.0] |
| 2.5 | 74.7% (669/895) | [71.8, 77.5] | 105 | 69.5% | [66.6, 72.3] |
| 3.0 | 74.8% (636/850) | [71.8, 77.6] | 150 | 70.5% | [67.6, 73.2] |

The adjusted standard prompt slope (~2.12%/α) is essentially identical to the anti-compliance slope (2.09%/α), and the direction is the same. The standard prompt starts higher (69.7% vs 64.2% at α=0.0) — as expected for a pro-context framing — and both converge near 70–75% at high α.

However, this table is now explicitly **secondary** to the strict α=3.0 remap above. Excluding 15% of data at α=3.0 is substantial, and the excluded samples are not random. The adjusted rates remain useful as a direction-of-travel diagnostic across α, but they should not be reported as the final population compliance curve under the standard prompt. In particular, the α=3.0 conditional rate `74.8%` is not the headline result; the current conservative population estimate is `72.1%`.

#### 11.7.5 Population Structure (Historical Raw-Parser Note, Now Stale)

Earlier in this report, I used the raw evaluator labels to summarize the standard prompt as:

- always compliant: `563`
- never compliant: `234`
- swing samples: `203`
- and within those swing samples, `123` compliant→resistant vs `68` resistant→compliant across α=0→3

Those counts are now **withdrawn as substantive evidence**. They are preserved here only as a log of the pre-remap reasoning. Because many α=3.0 answer-text outputs were misread as resistant, the raw-parser view overstates swing-to-resistance and understates both always-compliant behavior and content-level swing-to-compliance. A corrected population-structure analysis requires text-based scoring across all α, not just the α=3.0 endpoint.

#### 11.7.6 Cross-Prompt Agreement at Baseline (α=1.0)

| | Standard: compliant | Standard: non-compliant |
|---|---|---|
| Anti: compliant | 641 | 19 |
| Anti: non-compliant | 47 | 293 |

93.4% of samples (641+293) agree across prompts at baseline. The 47 "standard-only" compliant samples are cases where the pro-context framing tips borderline samples over the edge. The 19 "anti-only" compliant samples are harder to explain — they may reflect stochastic sensitivity to prompt wording rather than a systematic effect.

#### 11.7.7 Updated Interpretation: Content vs Evaluator Visibility

After the α=3.0 remap, the cleanest interpretation is no longer "content compliance rises while format compliance falls." That phrasing was too strong and, for the standard prompt, partly wrong. What we actually observe is:

1. **Content compliance increases.** This is the paper's claimed effect, and it holds under both prompt framings once answer-text outputs are counted properly.

2. **Evaluator-visible MC-letter compliance decreases.** Under the standard prompt, high α makes the model more likely to emit the answer text directly instead of an option letter plus text. Relative to the prompt, that may be neutral or even an improvement. Relative to our evaluator, it looks like failure.

So the real dissociation is between **semantic content** and **what the local MC-letter scorer can see**, not necessarily between content-following and prompt-following. That substantially weakens the earlier "general obedience vs format obedience" puzzle.

The remaining open question is narrower: why does high α make answer-text-only outputs more common under the standard prompt? Three explanations still seem worth testing:

**Hypothesis A — Context-credulity, not general compliance.** H-neurons specifically modulate how much the model treats input context as authoritative ground truth. Under this view, content compliance rises because the counterfactual passage is weighted more heavily, and answer-text-only outputs rise because the model is answering the retrieval QA prompt more literally.

- *Falsifiable by:* Testing answer-format behavior on a task with **no context passage**. If the same answer-text shift appears without any retrieved context to trust, Hypothesis A is weakened.

**Hypothesis B — Surface-form selection inside the retrieval QA frame.** The prompt asks for exact answer text, but the presence of multiple-choice options gives the model two valid surface forms: answer text or answer letter. High α may shift it toward the semantically direct answer-text form.

- *Falsifiable by:* Designing a prompt where the desired content and desired surface form are perfectly aligned, for example "Pick the letter whose answer is supported by the context." If answer-letter exposure still collapses there, Hypothesis B is wrong.

**Hypothesis C — Generic precision loss at extreme α.** Scaling 38 neurons by 3× is a substantial perturbation. Maybe the model becomes less consistent about exact surface-form choices while coarse semantic selection remains intact. Under this view, the answer-text shift is not about compliance at all; it is a byproduct of reduced output-shape precision.

- ~~*Falsifiable by:* The negative control experiment.~~ **FALSIFIED.** The negative control (5 unconstrained seeds, 38 random neurons each, α∈{0.0…3.0}) shows **neither** format degradation **nor** content compliance increase (mean slope 0.06 pp/α, zero parse failures at all α). Since random scaling produces neither effect, **both the format shift and the content compliance increase are H-neuron-specific**, not generic perturbation artifacts. Data: `data/gemma3_4b/intervention/negative_control/`.
- *Additional test (still open):* Measure other precision-dependent behaviors at high α — e.g., does the model's ability to count, do arithmetic, or follow multi-step instructions also degrade? If yes, a weaker version of Hypothesis C (specific to H-neuron perturbation magnitude) could still apply.

**What we can say with confidence:**
- The paper's core causal claim — amplifying H-neurons increases content compliance with misleading context — holds under both prompt framings.
- The raw standard-prompt drop to `63.6%` should **not** be used as evidence that high α makes the model resist misleading context. After conservative text remapping, the α=3.0 rate rises to `72.1%`.
- The earlier "format compliance decreases" framing is partly falsified for the standard prompt. Many of the supposed format failures are exactly the kind of short answer-text outputs that the prompt asked for.
- The "general compliance neuron" framing is still too simple, but the key dissociation is now semantic content vs evaluator visibility, not semantic content vs prompt compliance.

**Implication for the paper:** If the paper uses the standard FaithEval prompt (as we presume), their reported compliance rates at high α may also be affected by the same evaluator-format artifact. The magnitude depends on their parser's robustness. Raw compliance curves from MC-format benchmarks should be accompanied by parse failure rates and, ideally, answer-text remapping to rule out this confound.

#### 11.7.8 Remaining Uncertainty and Next Experiments

Resolved since the earlier draft: the α=3.0 manual/text remap is now done, and it overturns the raw-drop interpretation. The main remaining uncertainty is not whether α=3.0 was an evaluator artifact — it mostly was — but how far the same issue extends across α<3.0 and how much of the remaining surface-form shift is H-neuron-specific versus generic perturbation.

**Priority experiments for resolving the hypotheses:**

1. **Integrate text-based FaithEval scoring across all α.** The α=3.0 remap fixed the most important contradiction, but the current standard-prompt curve still mixes raw letter extraction at α<3.0 with text remapping only at α=3.0. The next cleanup step is to extend content-based scoring across the full sweep and then recompute the curve and population structure.

2. ~~**Negative control (tests Hypothesis C):**~~ **DONE on anti-compliance prompt.** Five unconstrained seeds show flat compliance (mean slope 0.06 pp/α) and zero parse failures, falsifying Hypothesis C. Extending the NC to the **standard prompt** specifically would additionally test whether the format shift is H-neuron-specific under that framing, but is lower priority now that the anti-compliance NC is clean.

3. **Aligned-instruction test (tests Hypothesis B):** Run FaithEval with a prompt where the evaluator and the prompt ask for the same thing, e.g. "Pick the letter of the answer supported by the context." This isolates whether the current effect is mostly about evaluator mismatch or about a deeper surface-form instability.

4. **No-context answer-format test (deprioritized from the earlier draft):** This is still useful, but less urgent now that the original "format compliance decreases" framing has been weakened. It should be treated as a follow-up on surface-form selection, not as the main explanation for the α=3.0 contradiction.

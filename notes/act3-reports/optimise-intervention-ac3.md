# Optimising the ITI Truthfulness Intervention — Act 3 Strategy

**Date:** 2026-04-01 (revised 2026-04-01, refined after LITO review)
**Status:** Pre-implementation analysis — revised after mentor feedback + LITO paper/code grounding
**Model:** Gemma-3-4B-IT (`google/gemma-3-4b-it`)
**Scope:** Designing the next-generation ITI artifact to test whether mixed-source extraction improves OOD generalisation, and defining the decision gates for adaptive candidate selection (Stage B)

---

## 0. Executive Summary

The current paper-faithful ITI setup shows a real signal on TruthfulQA MC (+5.5–7.0pp held-out), but three independent lines of evidence suggest it will struggle on out-of-distribution benchmarks:

1. The Universal Truthfulness Hyperplane paper shows that TruthfulQA-only probes deteriorate to near-chance on some OOD datasets and trail simpler baselines OOD.
2. Our own controls overlap real ITI at n=81, and the 4.1pp cal-val→held-out gap flags partial overfitting.
3. The SimpleQA production run reveals that ITI acts as an **indiscriminate commitment suppressor** — though this finding is partially confounded by the prompt-level escape hatch.

The proposed response is a **three-stage plan:**

- **Stage 0:** Deconfound generation evaluation — remove the SimpleQA escape hatch and rerun controls. This is the cheapest work (~30 min GPU) and determines whether the generation null is real or prompt-artefact.
- **Stage A:** Build and compare artifact candidates (TruthfulQA-modernized, TriviaQA-only, mixed) — evaluated primarily on **forced generation**, with MC as a regression gate. LITO's own cross-domain evidence (§6.2.2) suggests TriviaQA-only may transfer better than mixed, so E2 is a first-class candidate, not a diagnostic.
- **Stage B (conditional):** Adaptive candidate selection — only if Stage A produces an artifact where multi-α oracle best-of-K materially beats the best single α. Grounded in LITO (Fatahi Bayat et al., 2024), but gated by a candidate headroom study, not assumed.

This document lays out the diagnosis, the strategy, the experiment set, the uncertainties that should be tracked, and the decision gates for each stage.

---

## 1. What We Know Now — The Evidence Base

### 1.1 Phase 2b held-out results (ground truth)

| Fold | Variant | Baseline | α=8 | Δ |
|------|---------|----------|------|-------|
| 0 | MC1 | 26.8% | 32.3% | +5.5pp |
| 0 | MC2 | 43.9% | 51.5% | +7.6pp |
| 1 | MC1 | 26.6% | 33.6% | +7.0pp |
| 1 | MC2 | 42.2% | 48.9% | +6.7pp |

Cal-val MC1 at K=12, α=8 was 37.0%. Held-out average is 32.95%. That is a **4.1pp cal-val→held-out drop**, right at the 3pp flag threshold set in the gate criteria. Not catastrophic, but it means the K×α selection latched partially onto cal-set peculiarities.

### 1.2 Controls overlap real ITI at n=81

| Control type | Seeds | α=0 baseline | α=8 | Δ range |
|---|---|---|---|---|
| Random heads | 3 | 22.2% | 22.2–28.4% | +0.0 to +6.2pp |
| Random directions | 3 | 22.2% | 24.7–28.4% | +2.5 to +6.2pp |
| **Real ITI (fold 0)** | — | 26.8% | 32.3% | **+5.5pp** |

Random-head seed 3 and random-direction seeds 2–3 each hit +6.2pp, which **overlaps** real ITI fold 0's +5.5pp. The controls are run on cal-val (n=81), so Wilson CIs are wide (~±10pp), and some of this overlap is expected from sampling variance alone. But it means the **specificity signal — real ITI beating controls — is not statistically established at this sample size**.

**Caveat on control validity:** The fix plan audit (`fix_plan_iti_pipeline_bugs.md`, FIX 3) found that random-direction controls with different seeds collided in the same output directory. `build_iti_output_suffix` did not hash `iti_direction_mode` or `iti_direction_random_seed`, so seeds 2 and 3 may have silently no-oped (skipping IDs already present from seed 1). If so, the +6.2pp from "seeds 2–3" may actually be seed-1 data reread, not independent draws. This needs to be confirmed and, if true, the controls need to be rerun before any specificity claim is made.

### 1.3 SimpleQA production run — the commitment suppressor finding

From `2026-04-01-simpleqa-iti-production.md`:

- Precision is **flat** at α=0 and α=4 (5.0%→4.9%). ITI does not selectively suppress hallucination.
- At α=8: attempt rate collapses to 20.8%, precision rises only to 7.7% on n=16 survivors.
- 57.8% of questions follow the trajectory INCORRECT→INCORRECT→NOT_ATTEMPTED.
- The intervention behaves as a **blunt commitment modulator**.

**Critical caveat — the escape hatch confound.** The SimpleQA prompt template includes `"If you are unsure, say 'I don't know.'"` This creates a high-probability exit token sequence. At α=4, 174/177 (98.3%) of NOT_ATTEMPTED responses are the literal string `"I don't know."` (§1e). The production report itself identifies this as load-bearing (§1h): removing the escape hatch is the single cheapest, highest-signal experiment available to determine whether the null result is real or a prompt artefact.

Until that experiment is run, the "indiscriminate commitment suppressor" diagnosis should be qualified: **the intervention suppresses commitment into a named escape hatch, but we do not yet know whether it would suppress commitment in a forced-response regime.**

### 1.4 Context-grounded artifact is retired

From `iti_context_grounded_diagnostic.md`: The SQuAD-v2 artifact has AUROC ≈ 1.0 because it learned **passage-answer overlap**, not truthfulness. Near-perfect probe metrics + zero behavioral change on FaithEval. Retired from primary claims.

### 1.5 Existing artifact audit (from `iti_audit_baseline.md`)

| Family | Source | Top-5 AUROC | Head layers | Signal quality |
|---|---|---|---|---|
| `iti_triviaqa_transfer` | TriviaQA consistency | 0.749–0.768 | 9, 11, 22 | Moderate, genuine |
| `iti_context_grounded` | SQuAD v2 | 0.999+ | 5, 6, 13, 14, 18 | Trivial shortcut — retired |
| `iti_truthfulqa_paperfaithful` | TruthfulQA CSV | (val_accuracy based) | — | Moderate, TruthfulQA-specific |

The TriviaQA transfer artifact selects different heads and layers from TruthfulQA paper-faithful. This is empirical evidence that the direction depends on the data distribution — but it is not, by itself, evidence that mixing sources will produce a better direction. The `iti_triviaqa_transfer` artifact exists but has **not been shown to work on TruthfulQA MC or any OOD benchmark**.

---

## 2. The Diagnosis — Confirmed Issues, Hypothesised Causes, and Open Questions

Each item below is tagged as **confirmed** (empirical evidence in hand), **hypothesis** (plausible but unmeasured), or **open question** (we don't know enough to predict).

### 2.1 Single-source data limits OOD generalisation (CONFIRMED by literature, HYPOTHESIS for our setup)

**Literature evidence.** Liu et al. (arXiv 2407.08582, §3.2, Table 1) trained probes exclusively on TruthfulQA and measured OOD performance on 8 test datasets:

- Probe-LR: 82.28% in-distribution → 54.44% average OOD
- Probe-MM: 77.08% in-distribution → 50.71% average OOD
- Self-Eval baseline: 62.96% in-distribution → 63.31% average OOD

The probes trail the Self-Eval baseline by ~9–13pp on OOD data and approach chance level (50%) on average. The authors write: *"The probe's accuracy, close to the chance level at 50, implies that the learned hyperplane of the probe fails to contain any truthfulness information pertinent to certain OOD datasets."* Performance varies wildly by individual dataset — the average masks both near-chance and above-chance individual results.

When they train on their diverse 40+ dataset collection, cross-task accuracy rises to ~70% (Probe-LR: 69.24%, Probe-MM: 70.47%, Table 2). They find that increasing dataset *count* helps more than increasing sample count per dataset (§3.6, Figure 4b–c). Their training splits are capped at 800 samples per dataset, and they note that performance with 10 samples per dataset is "comparable to that of using 800 samples" — but this is a statement about the probe's data efficiency given diverse sources, not a claim that 10 samples is optimal. The key variable is source diversity, not volume.

**How this maps to us.** The paper-faithful ITI direction is fitted on 817 TruthfulQA questions — all adversarial misconceptions from 38 categories. The literature predicts that this direction will not generalise well OOD. **However:** (a) the Universal Hyperplane paper studies probes (detection), not ITI-style steering (intervention) — the relationship between probe generalisation and steering generalisation is assumed but unverified; (b) they use LLaMA-2-7b-chat and Mistral-7b, not Gemma-3-4b-it; (c) our activation surface is different (pre-o_proj head outputs vs. post-attention-head outputs). The spirit of the prediction is well-grounded; the quantitative transfer to our setup is a hypothesis.

### 2.2 Prompt-format mismatch between extraction and inference (CODE CONFIRMED, IMPACT HYPOTHESISED)

**The code mismatch is confirmed:**

- **Extraction** (`extract_truthfulness_iti.py`, `_qa_prompt_paper`, line 577): uses bare `Q: {question}\nA: {answer}` — no chat template, no role tokens.
- **Inference** (`run_intervention.py`, `tokenize_chat`, line 300): uses `tokenizer.apply_chat_template(...)` with full Gemma chat structure including `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...` formatting.

This is a real mismatch. Directions are fitted in one prompt regime and applied in another.

**The impact is a hypothesis.** No experiment has measured the effect size of this mismatch. It is plausible that it matters — chat-template tokens create structured attention sinks that change the activation geometry — but it could also be small if the truthfulness signal is robust to formatting changes. Experiment E1 (TruthfulQA-only modernized with chat-template) is designed to isolate this effect. Until E1 runs, the mismatch is a known code-level issue worth auditing, not a diagnosed cause of poor OOD performance.

### 2.3 Last-token-only position summary (EMPIRICAL OBSERVATION, not literature-backed)

Paper-faithful forces `last_answer_token` only. The exploratory family supports `(first_answer_token, mean_answer_span, last_answer_token)` and picks the best per head.

**Important context.** The Universal Truthfulness Hyperplane paper itself uses last-token representations as its standard probe surface (§2.2: *"we compute h_i as the representation of the last token in x_i"*; §2.4 confirms this is used throughout). So "last token is suboptimal" is **not a literature-backed criticism** — it is the standard choice in the probing literature.

The stronger argument is **model-specific and empirical:** from `iti_audit_baseline.md`, the TriviaQA-transfer top 5 heads prefer `mean_answer_span` (3 of 5). This suggests that, for Gemma-3-4b-it on factoid QA, truthfulness signal may be distributed across the answer span rather than concentrated at the final token. Allowing multi-position selection lets the ranking discover this if it exists.

The K×α sweep analysis (`2026-03-31-sweep-analysis.md`) provides additional indirect evidence: the paper-faithful setup shows a narrow usable band (hot zone confined to K∈{8,12,16} × α∈{6,8,12}) that collapses at K≥24 with MC1 dropping below baseline. The fragility of this surface could reflect suboptimal head selection driven by position-summary limitations, but this is speculative — narrow K×α bands could equally reflect other issues (weak direction, small n, prompt mismatch).

### 2.4 val_accuracy ranking on n=65 val questions (HYPOTHESIS)

Paper-faithful uses plain accuracy (hard threshold at 0.5) on 65 val questions. AUROC uses the full probability curve and is invariant to class imbalance. With the per-question inverse-count weighting already in the codebase, AUROC should give a more robust ranking, especially when mixing sources with different label ratios.

This is a standard methodological improvement but its effect size is unknown for our setup.

---

## 3. The Strategy — Mixed-Source Single-Vector ITI

### 3.1 Why mixed-source, and what we don't know

**The case for.** The single strongest motivation is the Universal Hyperplane paper's finding that probe generalisation improves with source diversity. Our paper-faithful direction is a single-source artifact. Adding TriviaQA consistency data doubles the source count and adds broad factoid coverage the TruthfulQA questions lack.

**Independent evidence from LITO.** LITO's cross-domain transfer results (Fatahi Bayat et al. 2024, §6.2.2, Figure 4) provide an asymmetric signal:

- **TriviaQA-trained LITO transfers well** to NQ, SciQ, and even TruthfulQA — near in-domain performance. The authors attribute this to *"TriviaQA's broad general knowledge base, which applies to more specialized domains."*
- **TruthfulQA-trained LITO transfers poorly** — consistent with Li et al. 2023's original observation.

This has a non-obvious implication: **TriviaQA-only directions may be a better base for generation transfer than TruthfulQA-heavy mixes.** The value of including TruthfulQA in the mix may be primarily about preventing TruthfulQA MC regression, not about improving generation transfer. This is why E2 (TriviaQA-only) deserves to be a **first-class candidate**, not a diagnostic check.

**What we don't know:**
1. **Whether probe generalisation translates to steering generalisation.** The Universal Hyperplane paper studies detection (classification via hyperplane), not intervention (adding a vector to activations during decoding). LITO uses standard single-source ITI directions — it does not validate multi-source direction extraction. No paper we have read validates the specific link: diverse training data → better mass-mean direction → better behavioral intervention.
2. **Whether the TriviaQA and TruthfulQA truthfulness axes are geometrically compatible.** If the axes point in very different directions (cosine < 0.3), the mean direction may be mediocre at both tasks — "jack of all trades, master of none." We can measure this before committing to mixed extraction.
3. **Whether the `iti_triviaqa_transfer` artifact actually helps on any benchmark.** It exists but has not been evaluated on TruthfulQA MC, SimpleQA, FalseQA, or FaithEval. The different head/layer selections (layers 9, 11, 22 vs. paper-faithful heads) are suggestive but not yet informative about behavioral outcomes.
4. **Whether any single fixed α works for generation tasks at all.** Our current positive result is MC-only (log-prob scoring). The generation evidence is confounded (§1.3, §4.3). LITO was designed precisely because fixed α fails on generation — different questions need different intensities (LITO Figure 1). If this holds for our model, the right comparison may not be E3 vs E0 at one α, but best-of-K oracle vs best single α.

### 3.2 Why single-vector first, not ACT-style clustering

ACT (Wang et al., arXiv 2406.00034) shows that hallucination categories cluster differently in activation space and that diverse steering vectors + adaptive intensity can yield large gains. **But our bottleneck is not vector diversity — it is having any vector that works OOD.** Before adding complexity (cluster count, routing, adaptive alpha), we should test the simplest hypothesis: does a single direction from diverse data in the right prompt geometry outperform the current single-source artifact?

### 3.3 Data composition

#### Primary recommendation (2-source mix)

| Source | Weight | Questions | Role |
|---|---|---|---|
| TriviaQA consistency | 60% | 2,782 (1,391 T + 1,391 F) | Broad factuality — the model's own knowledge boundary |
| TruthfulQA CSV | 40% | 817 (with ~3.5 correct + ~4.1 incorrect answers each) | Misconception/myth-style failures — adversarial truthfulness |

#### Exploratory-only (3-source mix, not recommended as primary)

| Source | Weight | Questions | Role |
|---|---|---|---|
| TriviaQA consistency | 45% | 2,782 | Broad factuality |
| TruthfulQA | 35% | 817 | Misconception coverage |
| SQuAD-v2 (redesigned) | 20% | ~500 (250 answerable + 250 impossible) | Abstention signal (see risk discussion below) |

#### Why these sources

**TriviaQA consistency** brings distribution diversity relative to TruthfulQA's 817 questions. The consistency filtering (10/10 correct or 10/10 incorrect with GPT-4o judgment) provides clean labels — these are questions where the model reliably knows or doesn't know the answer. The two sources cover different failure modes (factoid recall vs. adversarial misconceptions), different question styles, and different domains.

**TruthfulQA** must stay in the mix because it is the primary evaluation benchmark. Dropping it risks TruthfulQA MC regression, and it covers adversarial misconceptions that TriviaQA doesn't test.

#### Why SQuAD-v2 is demoted to exploratory-only

The original draft included SQuAD-v2 as a 20% minority source providing an "abstention signal." On reflection, this is problematic for two reasons:

1. **The sprint architecture separates refusal, truthfulness, and detection as distinct axes.** This is a deliberate design principle, not an accident. Adding "don't answer when you can't" supervision to the truthfulness direction risks entangling it with refusal geometry — exactly what the architecture is designed to avoid.

2. **The D3.5 refusal-overlap audit** (`week3-log.md` §7) found that H-neuron scaling overlaps refusal geometry more than random-neuron null — but the signal is fragile and dominated by a single layer (layer 33, 43× larger subspace gap than the next layer). Excluding layer 33 collapses the FaithEval correlation to ρ=−0.005 (CI spanning zero). The refusal axis exists but is delicate. Adding SQuAD-v2 abstention data could amplify this overlap in unpredictable ways.

3. **The previous SQuAD-v2 artifact failed via shortcut** (`iti_context_grounded_diagnostic.md`). Even with redesigned prompts and 20% weight, the risk of the extractive-QA format leaking a trivial signal into the direction is non-negligible.

If the 2-source mix (E3) is weak on FalseQA/FaithEval, we can revisit SQuAD-v2 as experiment E3b with explicit monitoring for refusal-axis contamination. But it should not be in the primary candidate.

#### Weighting rules

1. **Equalize source contribution** — each source contributes total training weight proportional to the chosen ratio (60/40).
2. **Equalize question contribution within source** — inverse count per question-label group, as already implemented.
3. **Equalize truthful/untruthful within question** — prevents sources with variable answer counts from skewing the centroid.

### 3.4 Extraction changes

| Parameter | Paper-faithful (current) | Mixed-source (proposed) |
|---|---|---|
| Prompt format | `Q: {question}\nA: {answer}` | Chat-template matched (Gemma `<start_of_turn>` format) |
| Position summaries | `last_answer_token` only | `first_answer_token`, `mean_answer_span`, `last_answer_token` — best per head |
| Head ranking | `val_accuracy` (n=65) | `auroc` (pooled across val from all sources) |
| Direction fit scope | `dev_data` (train+val, all folds) | `train_data` only (cleaner separation of ranking vs direction) |
| Direction method | mass-mean | mass-mean (unchanged) |
| Direction weighting | per-question inverse count | per-question inverse count + source balance |

#### Chat-template prompt design

For TriviaQA and TruthfulQA, the extraction prompt becomes:

```
<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>
```

This matches what `run_intervention.py` uses at inference time.

### 3.5 What we keep unchanged

- **Activation surface:** pre-o_proj head outputs (this is well-justified and working)
- **Direction method:** mass-mean (the simplest, lowest-variance option)
- **Intervention mechanism:** decode-only head-level steering via `ITIHeadScaler`
- **Leakage barriers:** fold-based test exclusion, split audits, overlap checks against all eval benchmarks

---

## 4. Prerequisites Before Optimisation — "Bolts on the Floor"

Before running mixed-source experiments, several repo-level issues should be acknowledged. Some must be fixed; others should be tracked as caveats on any results.

### 4.1 Random-direction control collision (FIX REQUIRED)

From `fix_plan_iti_pipeline_bugs.md`, FIX 3: `build_iti_output_suffix` does not hash `iti_direction_mode` or `iti_direction_random_seed`. When the pipeline runs random-direction controls with seeds 1/2/3, all three may write to the same output directory, with seeds 2 and 3 silently no-oping. This means the +6.2pp control results from "seeds 2–3" may be seed-1 data reread, not independent draws. The control variance reported in §1.2 is potentially wrong.

**Impact on this plan:** Any specificity claim (real ITI > controls) depends on valid control runs. Fix 3 must land before running new controls for the mixed-source artifact.

### 4.2 MC2 metric computation is wrong (FIX REQUIRED)

From `fix_plan_iti_pipeline_bugs.md`, FIX 1: `report_iti_2fold.py` reads `compliance` (binary) instead of `metric_value` (truthful mass) for MC2. All MC2 numbers from the 2-fold pipeline are binary compliance stats, not the planned truthful-mass metric. FIX 2 compounds this: the pooled report reconstructs arrays from rounded rates, destroying pairing.

**Impact on this plan:** MC2 results are used as secondary validation. They should be recalculated after fixes 1+2 before being cited as evidence.

### 4.3 SimpleQA escape hatch confound (ACKNOWLEDGED, EXPERIMENT NEEDED)

The SimpleQA prompt `"If you are unsure, say 'I don't know.'"` creates a named exit route that the intervention exploits. Until a forced-commitment variant is tested (the cheapest next experiment per the SimpleQA production report, ~30 min GPU), the "indiscriminate commitment suppressor" diagnosis is partially confounded. This does not block the mixed-source work, but it means SimpleQA results for the new artifact will be similarly confounded unless the prompt is changed.

### 4.4 Hook-placement scope (ACKNOWLEDGED)

The ITI hooks are installed during decoding. The fix plan (FIX 6) found that `run_calibration_sweep.py` does not cleanly remove hooks between K values during the K×α sweep — new hooks are constructed before old ones are removed. This was noted for the sweep script; it should be verified that the main `run_intervention.py` pipeline does not have the same issue.

### 4.5 Response-length confound on SimpleQA (ACKNOWLEDGED)

At α=8, the mean response is ~3 tokens (mostly "I don't know.") vs ~30+ tokens at α=0. Any classifier or judge operating on full responses will see dramatically different input distributions. The GPT-4o judge is robust enough for the 3-way grading, but response-length confounds should be tracked if any automated metrics (perplexity, embedding similarity) are added.

---

## 5. The Experiment Set — Three Stages

### 5.0 Stage 0: Deconfound generation evaluation (PREREQUISITE)

Before running any artifact comparison, resolve the two biggest confounds in our generation evidence:

**5.0a — Forced-commitment SimpleQA.** Rerun 1000 SimpleQA questions with the current paper-faithful artifact (K=12, α∈{0, 4, 8}) using:
```
Question: {question}
Answer with a single factual phrase.
```
No escape hatch. This determines whether the "indiscriminate commitment suppressor" diagnosis (§1.3) is real or a prompt artefact. Cost: ~30 min GPU, ~$15 judge.

**5.0b — Revalidate random-direction controls.** After landing FIX 3 (§4.1), rerun 3-seed random-direction and 3-seed random-head controls on TruthfulQA cal-val (n=81) to establish whether ITI specificity is real.

**Gate:** If forced-commitment SimpleQA shows ITI improves precision at any α (even modestly), the generation null is partially prompt-artefact and the case for mixed-source becomes stronger. If precision remains flat, the diagnosis is confirmed and we should evaluate all Stage A artifacts on forced-generation as the primary metric, not MC.

### 5.1 Frozen multi-benchmark dev/test splits

Before running Stage A experiments, create fixed dev/test manifests for every OOD benchmark:

| Benchmark | Total n | Dev (for K×α selection) | Test (for final report) |
|---|---|---|---|
| TruthfulQA MC1 | 655 (2-fold) | Existing cal-val (81) | Existing 2-fold held-out (~327/328) |
| SimpleQA | 1000 | 200 (stratified by topic) | 800 |
| FalseQA | ~800 | 160 | 640 |
| FaithEval | ~500 | 100 | 400 |
| Jailbreak benign | 500 | (use full — binary safety gate, not a metric) | — |

**Selection rule for K×α:** maximize **macro-average improvement across dev tasks** with guardrails:
- No SimpleQA attempt-rate collapse (attempt rate must stay >80% on dev)
- No refusal increase on jailbreak benign (must remain 0%)

### 5.2 Stage A: Artifact comparison experiments

| ID | Artifact | Data | Prompt | Ranking | Position | Purpose |
|---|---|---|---|---|---|---|
| **E0** | `iti_truthfulqa_paperfaithful` (current) | TruthfulQA only | Bare Q/A | val_accuracy | last only | Anchor baseline — already run |
| **E1** | `iti_truthfulqa_modernized` | TruthfulQA only | Chat-template | AUROC | first/mean/last | Isolate extraction+ranking improvements, test prompt-mismatch hypothesis |
| **E2** | `iti_triviaqa_transfer` | TriviaQA only | Chat-template | AUROC | first/mean/last | **First-class candidate** — LITO's transfer data suggests broad factuality may outperform TruthfulQA-based directions on generation |
| **E3** | `iti_mixed_truthfulness` | TriviaQA (60%) + TruthfulQA (40%) | Chat-template | AUROC | first/mean/last | **First-class candidate** — test mixed-source hypothesis |
| **E3b** | `iti_mixed_3source` | TriviaQA (45%) + TruthfulQA (35%) + SQuAD-v2 (20%) | Chat-template | AUROC | first/mean/last | Exploratory only — test abstention component, monitor for refusal-axis contamination |

**Evaluation priority shift:** The primary evaluation metric for Stage A should be **forced-generation accuracy/precision** (from Stage 0's deconfounded SimpleQA prompt), not TruthfulQA MC. MC is a regression gate (must not drop below ~30% MC1) but generation transfer is the hypothesis under test. This aligns with how LITO itself evaluates — short-form QA generation with NLI-based correctness judgment, not MC log-prob scoring.

### 5.3 K×α sweep grid

Restrained grid — 12 combos per experiment:

```
K  ∈ {8, 12, 16}
α  ∈ {2.0, 4.0, 6.0, 8.0}
```

**Hypothesis:** the optimal α should **decrease** with a cleaner direction — the direction needs less amplification if it points closer to the right place. If α stays high or increases, that suggests the mixed direction is not actually cleaner.

### 5.4 What a "win" looks like

**Primary pre-registered comparison:** E3 vs E0 on macro-average across OOD benchmarks (SimpleQA, FalseQA, FaithEval), with TruthfulQA MC1 as a regression gate.

**Strong claim:** E3 beats E0 on:
- **TruthfulQA MC1** (no regression — at least matches held-out ~33%)
- **≥2 of 3 OOD benchmarks** show improvement

With paired bootstrap 95% CIs that exclude zero on the macro-average.

**Defensible claim:** E3 achieves better macro-OOD average than E0 even if one individual benchmark is flat.

**Null result we should be prepared for:** E3 shows no improvement over E0, or improves TruthfulQA MC but not OOD benchmarks. This would suggest that the truthfulness axis is more task-specific than the Universal Hyperplane paper implies, at least for ITI-style steering on Gemma-3-4b-it. That is an informative result.

### 5.5 What we report alongside

- Attempt rate / refusal rate for every benchmark × α (critical for distinguishing "smarter" from "more cautious")
- Grade transition matrices (following the SimpleQA production report format)
- Per-benchmark paired bootstrap CIs
- Cosine similarity between E0 and E3 direction vectors (to characterise how different the directions are)
- Cosine similarity between per-source directions (TriviaQA-only vs TruthfulQA-only) — diagnostic for geometric compatibility

### 5.6 Diagnostic checks before committing to the full sweep

Before running the full K×α sweep across benchmarks (~5 hours GPU), run two cheap diagnostics:

1. **Direction cosine similarity.** Extract per-source directions (TriviaQA-only, TruthfulQA-only) and compute cosine similarity per head. If cosine < 0.3 for the top-K heads, the axes may be too different for simple mixing and the experiment design should be reconsidered.
2. **E1 on TruthfulQA MC only** (~30 min GPU). If chat-template + AUROC ranking shows zero improvement over E0 on TruthfulQA MC (the benchmark the direction was trained on), then the extraction changes are not helping even in-distribution, and the mixed-source hypothesis becomes the only lever worth testing.

### 5.7 Stage B: Adaptive candidate selection (CONDITIONAL)

This stage is grounded in LITO (Fatahi Bayat et al. 2024) and the mentor's observation that fixed α is brittle. **It should only proceed if Stage A produces an artifact worth wrapping.**

#### What LITO actually does (from paper + code review)

LITO is a **post-hoc candidate re-ranker** on top of standard ITI:

1. Generate K=5 candidates at α∈{5,10,15,20,25} (each a full forward pass with hooks)
2. For each candidate, collect: textual response, mean last-layer hidden state across generated tokens, geometric mean token probability (confidence)
3. Train a 1-layer LSTM on the sequence of K hidden states → binary classification per candidate (correct/incorrect), using DeBERTa NLI for training labels
4. At inference: if any candidate predicted correct, pick highest-confidence among positives; else abstain ("I have no comment")

LITO was tested on LLaMA-2-7B/13B-chat, Vicuna-7B, GPT-2-large/XL — all MHA models. Our Gemma-3-4B-IT uses GQA (8 KV heads), pre-o_proj hooks, and mass-mean directions (not probe-LR weights). The α scales are **not portable** — LITO uses {5,10,15,20,25} with probe-weight directions and `proj_val_std` normalization; our α=8 already causes attempt-rate collapse.

#### What the LITO evidence actually supports

**Supports:** Per-question α variation is real — LITO's Figure 1 shows different questions peak at different intensities on models with 30%+ baseline accuracy.

**Overstated by the mentor:** The LSTM verifier's value-add is modest. LITO's LSTM beats MLP by only +1.2–4.2pp accuracy (Table 3). The simpler `Max Conf > T` baseline is competitive on several model-dataset pairs (Vicuna TriviaQA: 72.5 vs LITO 72.0, Table 1). The sequential pattern over α adds value, but much of LITO's gain may come from the candidate lattice itself (having 5 options) rather than sophisticated selection.

**Not supported:** Any of this on a 4B model with 4.8% SimpleQA baseline. LITO's models have much higher base accuracy on their benchmarks (~30–70%). The model needs to *know some answers correctly* at baseline for per-question α selection to help. If the model doesn't know the answer at any α, no chooser can rescue it.

#### Stage B gate: candidate headroom study

On 200–300 forced-generation samples (from Stage 0's deconfounded prompt), using the best Stage A artifact:

1. Generate K=5 candidates at α∈{0, 1, 2, 4, 6} (note: include α=0 — LITO did not, but our model often gets things right at baseline)
2. Compute:
   - **Oracle best-of-5** accuracy (upper bound on any chooser)
   - **Max-confidence** selection accuracy (simplest chooser)
   - **Max-confidence among non-refusals** (if escape hatch is removed, this may not differ)
   - **Cross-α answer consistency** (do answers stabilise or randomise?)
3. Decision rules:
   - If oracle − best-single-α < 2pp → **kill Stage B** (no headroom for a chooser to exploit)
   - If oracle − max-confidence < 1pp → **stop at max-confidence** (don't build the LSTM)
   - If both gaps are material → proceed to LSTM prototype

#### If Stage B proceeds: implementation path

Start simple, not with the full LITO stack:

1. **Confidence-threshold chooser** — pick highest-confidence candidate above a threshold T, else abstain. Tune T on dev set.
2. **Linear verifier** — logistic regression on [pooled last-layer hidden state, confidence, response length] per candidate. Cheap to train, interpretable.
3. **LSTM verifier** — only if linear verifier leaves material gap to oracle. Use Gemma's last-layer hidden dimension (2048) with hidden_size=256.

Estimated additional GPU cost for the full headroom study: ~2 hours (5× candidate generation on 300 samples × multiple α values).

---

## 6. Risks and Guardrails

### Risk 1: Mixed-source direction becomes "jack of all trades, master of none"

If the TriviaQA factoid axis and TruthfulQA misconception axis are geometrically far apart, the mean direction may fall between them and be mediocre at both.

**Guardrail:** Compute cosine similarity between per-source directions before mixing. If cosine < 0.3, the axes are too different for simple mixing.

**If triggered:** Run E2 (TriviaQA-only) and E1 (TruthfulQA-modernized) independently on all benchmarks. If one dominates on different tasks, the evidence supports ACT-style multi-vector steering.

### Risk 2: The improvement is just increased abstention

As demonstrated in the SimpleQA report (with the escape-hatch caveat), the line between "smarter truthfulness" and "more cautious" is thin.

**Guardrail:** Report attempt rate alongside accuracy for every experiment × benchmark.

### Risk 3: SQuAD-v2 contaminates the direction via refusal-axis or lexical-overlap shortcut

The sprint architecture keeps refusal and truthfulness as separate axes. The D3.5 audit shows refusal overlap is real but fragile and layer-33 dominated. Adding SQuAD-v2 abstention data risks amplifying this overlap.

**Guardrail:** SQuAD-v2 is demoted to exploratory-only (E3b). Only run if E3 is weak on FalseQA/FaithEval. Monitor refusal-axis cosine similarity if E3b is run.

### Risk 4: Too many comparisons dilute statistical confidence

With 5 experiment IDs × 12 K/α combos × 5 benchmarks, we're doing 300 comparisons.

**Guardrail:** Pre-register the primary comparison (E3 vs E0 on macro-OOD average), locked before looking at test data. Everything else is exploratory.

### Risk 5: Benchmark contamination in mixed dataset

TriviaQA questions could overlap with SimpleQA or BioASQ.

**Guardrail:** Extend existing overlap checks (`check_falseqa_overlap`, `check_faitheval_overlap`, `check_refusal_overlap`) to check against SimpleQA and BioASQ before mixing.

### Risk 6: LITO-style adaptive selection inherits α-scale incompatibility

LITO's α∈{5,10,15,20,25} was calibrated for LLaMA-2-7B with probe-LR directions, K=48 heads, and `proj_val_std` normalization. Our setup uses mass-mean directions, K=12 heads, GQA architecture, and no projection-std scaling. Naively transplanting α values will either massively over-steer (collapse to abstention, as we already see at α=8) or require a completely new calibration sweep.

**Guardrail:** Any Stage B candidate lattice must use Gemma-specific α values calibrated by corruption-onset behaviour, not LITO's paper values. Must include α=0 as a candidate.

### Risk 7: Low knowledge ceiling limits chooser upside

Gemma-3-4B-IT's SimpleQA baseline accuracy is 4.8%. LITO's benchmarked models had 30–70% baseline accuracy on their QA tasks. A per-question chooser can only help if the model knows the answer at *some* α and doesn't at others. If the model never knows the answer, generating 5 wrong candidates and picking one is still wrong.

**Guardrail:** The Stage B gate (§5.7) requires oracle best-of-K to materially beat best single α. If the model's knowledge ceiling is the bottleneck, no amount of adaptive selection will help — the work should shift to model capability (larger model, RAG, fine-tuning) rather than steering sophistication.

---

## 7. Connections to the Literature

### 7.1 H-Neurons: theoretical motivation, not empirical validation

The h-neurons paper (Gao et al.) demonstrates that hallucination and over-compliance are controlled by a shared sparse neuron subset (<0.1% of parameters). Their H-neurons predict hallucination on TriviaQA, NQ, BioASQ, and NonExist — and causally control behavior on FalseQA, FaithEval, Sycophancy, and Jailbreak.

This is **theoretical motivation** for the mixed-source hypothesis: if truthfulness and over-compliance share underlying mechanisms, then a direction extracted from diverse factuality data might generalise to compliance-related tasks. However, the h-neurons paper operates at the MLP neuron level (input to `down_proj`, CETT metric), not the attention head level (pre-o_proj). The link between neuron-level H-neuron findings and head-level ITI steering is an assumption, not a demonstrated correspondence.

**Repo evidence cuts both ways.** The `iti_triviaqa_transfer` artifact exists but has not been evaluated on any benchmark other than its own val set. The D4 truthfulness direction (`2026-03-30-d4-truthfulness-direction.md`) survived the kill-shot but showed only a narrow usable operating point (β=0.01 on FaithEval, with immediate collapse at β=0.02). The D3 refusal-direction family (`2026-03-29-d3-faitheval-refusal-direction.md`) produced a narrow β=0.02 point then collapsed with answer-option bias at β=0.03. These results suggest that **finding usable operating points in residual-stream and direction-based steering is consistently hard for this model**, regardless of the data source.

Mixed-source ITI is a well-motivated experiment. It is not a reason to expect confident generalisation.

### 7.2 Universal Truthfulness Hyperplane: diversity helps probes, steering implications uncertain

Liu et al. find that training probes on 40+ datasets yields cross-task accuracy of ~70% (Probe-MM), up from ~50% for TruthfulQA-only. The key results:

- **Diversity > volume:** increasing dataset count from ~5 to ~40 training tasks gives larger gains than increasing sample count per dataset (§3.6, Figure 4b–c).
- **Data efficiency:** performance with 10 samples per dataset is comparable to 800 samples per dataset, given diverse sources (§3.6, Figure 4c). This is about linear probes being data-efficient, not a claim that 10 samples is sufficient in general.
- **Attention heads > layer residuals:** probes on attention head outputs outperform layer residual probes by ≥3pp (§3.6, Figure 4a) — consistent with our choice of pre-o_proj activation surface.

**Important limitation for us:** Their work validates that diverse training improves **probe detection accuracy** (binary classification of truthfulness). It does not directly test whether diverse training improves **ITI steering quality** (adding directions to hidden states during generation). The link between "better probe → better direction → better behavioral intervention" is the central assumption of the mixed-source strategy, and it is unvalidated.

### 7.3 LITO: adaptive candidate selection — what it validates and what it doesn't

Fatahi Bayat et al. (2024, ACL Findings) propose LITO, a learnable intervention method that generates K=5 candidates at different α values and trains an LSTM to select the best one or abstain.

**What LITO validates for our setup:**
- **Fixed α is suboptimal.** Different questions peak at different intensities (Figure 1). This is consistent with our K×α sweep narrowness and the D4 narrow-band result.
- **Candidate generation + selection can beat oracle-tuned fixed α.** On NQ with LLaMA-2-7B, LITO achieves TA=38.8 vs best-of-5 ITI oracle at 31.7 (Table 1).
- **TriviaQA-based directions transfer well cross-domain** (§6.2.2) while TruthfulQA-based directions do not. This is independent evidence for E2 (TriviaQA-only) as a strong candidate.
- **Attention heads outperform layer residuals for truthfulness probing** — consistent with our pre-o_proj activation surface choice and with the Universal Hyperplane paper (§3.6, Figure 4a).

**What LITO does not validate:**
- **Multi-source direction extraction.** LITO uses standard single-source ITI directions. The mixed-source hypothesis is motivated by the Universal Hyperplane paper, not by LITO.
- **Generation steering on models with low knowledge baselines.** LITO's models have 30–70% baseline accuracy on their benchmarks. Our 4.8% SimpleQA baseline is a fundamentally different regime.
- **Sophisticated verifier value.** The LSTM's advantage over MLP is +1.2–4.2pp (Table 3), and the simpler `Max Conf > T` baseline is competitive (Table 1). Much of LITO's gain may come from having multiple candidates, not from the selection mechanism.
- **GQA architecture.** All LITO models use MHA. GQA's shared KV heads may couple interventions differently.

**Key methodological difference from LITO to note:** LITO applies intervention to the **last token only** during generation, consistent with the Universal Hyperplane paper's probe design. Our `ITIHeadScaler` applies during full decode. LITO's own paper does not compare last-token-only vs full-decode intervention — this is an open question for our setup.

### 7.4 ACT: deferred until single-vector baseline is characterised

ACT's main insight — that hallucination categories cluster differently in steering-vector space — is valid but premature for our setup. We should adopt it only if:
1. The mixed single-vector direction helps some OOD tasks but hurts others (evidence of multimodality)
2. Per-question directional representations show clear clustering (can be checked post-extraction)
3. The single-vector baseline is stable enough to serve as a comparison anchor

### 7.5 Surgical Activation Steering (GCM): causal head selection

The GCM paper (arXiv 2602.16080) proposes selecting heads via generative causal mediation rather than correlational probing. GCM consistently outperforms probe-based baselines when steering with a sparse set of heads. This is a potential stage-3 upgrade if probe-based ranking plateaus.

---

## 8. Implementation Path

### 8.1 New builder: `build_mixed_truthfulness_examples`

Add to `extract_truthfulness_iti.py`:

1. A new family `iti_mixed_truthfulness` in the argument parser
2. A builder function that:
   - Loads TriviaQA consistency via existing `load_consistency_samples` + `load_id_split`
   - Loads TruthfulQA CSV via existing `build_truthfulqa_paper_examples` logic (but with chat-template prompts)
   - Merges into a single `list[ITIExample]` with source-balanced weights
   - Assigns train/val splits per source using existing split infrastructure

Key implementation detail: the weighting. Each example gets:
```
weight = (source_target_weight / source_total_examples) * (1.0 / n_examples_for_this_qid_label)
```
This achieves: source balance → question balance → label balance.

### 8.2 Chat-template extraction prompts

Add a `_qa_prompt_chat` variant alongside `_qa_prompt_paper`:

```python
def _qa_prompt_chat(tokenizer, question: str, answer: str):
    """Chat-template-matched prompt for extraction."""
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    # Tokenize with chat template to get the full prompt
    # Then identify answer token positions within the assistant turn
    ...
```

The tricky part is identifying answer token positions within the chat-template output. The approach:
1. Tokenize the full chat-template output (user + assistant turns)
2. Tokenize the chat-template output without the assistant content
3. Answer positions = tokens present in full but not in prefix

This mirrors the existing `_encode_with_answer_positions` logic but operates on chat-template-formatted strings.

### 8.3 Pipeline script

Write `scripts/infra/iti_mixed_pipeline.sh` following the pattern of existing pipeline scripts:

```
Phase 0: Stage 0 — forced-commitment SimpleQA + revalidated controls (gate check)
Phase 1: Build mixed artifact (extract_truthfulness_iti.py --family iti_mixed_truthfulness)
Phase 2: K×α sweep on frozen multi-benchmark dev set (forced-generation primary, MC gate)
Phase 3: Lock K, α → run on test splits
Phase 4: Controls (random heads, random directions) on test splits
Phase 5: Downstream benchmarks + judges
Phase 6: Report generation
Phase 7: (conditional) Candidate headroom study for Stage B
```

### 8.4 Estimated GPU time

| Stage | Step | GPU minutes |
|---|---|---|
| **Stage 0** | Forced-commitment SimpleQA (3 alphas × 1000 samples) | ~30 min |
| **Stage 0** | Revalidated controls (6 runs × 81 samples) | ~20 min |
| **Stage A** | Extract mixed artifact (~3600 examples, 2-source) | ~3 min |
| **Stage A** | Direction cosine diagnostic | ~1 min |
| **Stage A** | E1 TruthfulQA-only diagnostic | ~30 min |
| **Stage A** | K×α sweep (3 K × 4 α × 4 benchmarks, dev splits) | ~48 sweeps × ~3 min = ~144 min |
| **Stage A** | Full test on locked config | ~4 benchmarks × ~8 min = ~32 min |
| **Stage A** | Controls (3 seeds × 2 types) | ~6 × 8 = ~48 min |
| **Stage B** | Candidate headroom study (5 × 300 samples) | ~120 min |
| | **Total (Stage 0 + A)** | **~5.5 hours** |
| | **Total (including Stage B if triggered)** | **~7.5 hours** |

---

## 9. Decision Points for the Mentor

### What to discuss

1. **Stage 0 sequencing.** The recommendation is to run forced-commitment SimpleQA and revalidated controls *before* committing to the Stage A artifact comparison. This costs ~1 hour GPU total and resolves the two largest confounds. If the mentor wants to parallelise (run Stage 0 and Stage A concurrently), that's possible but means Stage A results would need reinterpretation if Stage 0 changes the picture.

2. **E2 vs E3 as primary candidate.** LITO's cross-domain evidence suggests TriviaQA-only directions transfer better than TruthfulQA-based ones for generation. The recommendation is to treat E2 and E3 as co-equal candidates, not to pre-anoint E3 (mixed) as the favourite. Does the mentor agree, or is TruthfulQA MC regression risk severe enough to force TruthfulQA into the direction?

3. **Stage B trigger criteria.** The candidate headroom study (§5.7) uses oracle−best-single-α < 2pp as a kill criterion for adaptive selection. Is 2pp the right threshold, or should it be higher given the 5× compute cost?

4. **The knowledge ceiling problem.** On SimpleQA, the model's baseline accuracy is 4.8%. Even a perfect chooser cannot improve accuracy beyond the model's knowledge ceiling. Should the plan include a **benchmark selection step** — identifying a forced-generation benchmark where Gemma-3-4B-IT has higher baseline accuracy (30%+), to give steering and candidate selection more headroom? TriviaQA open-ended or NQ (LITO's primary benchmarks) could serve this role.

5. **Safety gate for all artifacts.** The jailbreak benign battery appears as a guardrail in §5.1 but is underweighted relative to its importance. Should it be a hard gate (any refusal increase → artifact disqualified) or a reporting requirement?

### What to commit to

**Stage 0 (immediate):**
- Fix random-direction control collision (FIX 3)
- Run forced-commitment SimpleQA with current artifact
- Rerun controls with valid seeds

**Stage A (after Stage 0 gate):**
- Run E0, E1, E2, E3 — with E2 and E3 as co-equal candidates
- E3b only if E3 is weak on FalseQA/FaithEval
- Primary metric: forced-generation accuracy/precision; MC1 as regression gate
- Report with paired bootstrap CIs, attempt rates, and direction cosine diagnostics
- Pre-register E3 vs E0 macro-OOD comparison as the primary claim, but also pre-register E2 vs E0

**Stage B (conditional on headroom study):**
- Run candidate headroom study on best Stage A artifact
- Proceed to chooser only if oracle−best-single-α ≥ 2pp AND oracle−max-confidence ≥ 1pp
- Start with confidence-threshold chooser, escalate to linear verifier, LSTM only if justified

---

## 10. Appendix: Asset Inventory

### Data available now

| Asset | Path | Size | Status |
|---|---|---|---|
| TriviaQA consistency samples | `data/gemma3_4b/pipeline/consistency_samples.jsonl` | 3,500 questions (2,782 consistent) | Ready |
| TriviaQA train/test split | `data/gemma3_4b/pipeline/train_qids.json`, `test_qids_disjoint.json` | 2,000 train + 782 test | Ready |
| TruthfulQA CSV | `data/benchmarks/TruthfulQA.csv` | 817 questions, 6,233 QA pairs | Ready |
| TruthfulQA fold files | `data/manifests/truthfulqa_fold*.json` | 2-fold CV splits | Ready |
| SQuAD v2 | HuggingFace `squad_v2` | ~130K train + ~12K val | On-demand |
| SimpleQA | `data/benchmarks/simpleqa_verified.csv` | 1,000 questions | Ready |
| FalseQA | `data/benchmarks/falseqa_test.csv` | ~800 questions | Ready |
| BioASQ | `data/benchmarks/bioasq13b_factoid.parquet` | Factoid QA | Ready |

### Code to extend

| Script | Change needed |
|---|---|
| `extract_truthfulness_iti.py` | Add `iti_mixed_truthfulness` family + `build_mixed_truthfulness_examples` builder + chat-template prompt variant |
| `build_truthfulness_contrastive.py` | Extend overlap checks to SimpleQA/BioASQ |
| `run_intervention.py` | Fix 3 (control output dir collision) |
| `report_iti_2fold.py` | Fixes 1+2 (MC2 metric + pooled pairing) |
| Pipeline scripts (`scripts/infra/`) | New `iti_mixed_pipeline.sh` orchestrator |

### Existing infrastructure reusable as-is

- 2-fold evaluation pipeline (`iti_pipeline_evaluate.sh`)
- Downstream benchmark runner (`iti_pipeline_downstream.sh`)
- Bootstrap CI computation (`uncertainty.py`)
- Provenance tracking (`utils.py`)
- Head ranking + direction fitting (all in `extract_truthfulness_iti.py`)
- Leakage auditing (`utils.py` → `audit_split_leakage`)

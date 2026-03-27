# Prompt: Pragmatic AI-Safety Extensions for the H-Neuron Project

You are advising a researcher working on a mechanistic interpretability project that replicates and extends the H-Neurons paper (Gao et al., "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs," arXiv:2512.01797v2). The work is on a single model (google/gemma-3-4b-it) with consumer hardware (RTX 5060 Ti 16 GB). The goal is to identify the most pragmatically beneficial extensions for AI safety — ones that could produce publishable, decision-relevant results within a focused sprint. Cloud compute is available if necessary, 200$ of lambda credits, and potential grants between 100 and 500$ by bluedot if I get traction/results on smaller scale.

**Core strategic constraint (from BlueDot grant criteria):** "The strongest projects explain how they fill a gap in existing work or have a clear theory of change." Every recommendation must be evaluated against this: what gap does it fill, and what is the concrete theory of change (if we show X, then practitioners/researchers can do Y, which improves safety outcome Z)? Directions without a crisp gap+ToC story are deprioritized regardless of scientific interest.

Below is a comprehensive summary of the project's current position, key findings, confirmed limitations, and closed lines of investigation. Use this to ground your recommendations in what the data actually shows.

---

## 1. What the Project Has Established

### 1.1 Core Causal Finding

38 H-neurons (0.011% of 348,160 FFN neuron positions) were identified via L1-regularised logistic regression on answer-token CETT (Contribution-weighted Exact Token Truth) activations. Scaling these neurons causally increases over-compliance:

| Benchmark | Effect (alpha=0 to 3) | 95% CI | Monotonic? | Negative control? |
|---|---|---|---|---|
| FaithEval anti-compliance (n=1000) | +6.3 pp | [4.2, 8.5] | Yes (rho=1.0) | Yes: 8 random seeds, slope 0.02 pp/alpha |
| FalseQA (n=687) | +4.8 pp | [1.3, 8.3] | No | Yes: 3 random seeds, slope 0.00 pp/alpha |
| Jailbreak binary (n=500, 5000tok) | +3.0 pp | [−1.2, +7.2] | No | No (CI includes zero — **not significant**) |
| Jailbreak CSV-v2 graded (n=500) | +7.6 pp | [+3.6, +11.6] | Yes | No negative control yet |

The negative controls are decisive: random neurons produce flat slopes (interval [-0.106, 0.164] pp/alpha on FaithEval), confirming the effect is specific to these 38 neurons. Ablation (alpha=0) also shows specificity — H-neuron ablation drops compliance ~1.8-2.3pp while random ablation has no effect. This is two-sided specificity.

### 1.2 Five Novel Findings Beyond Replication

**1. L1 Artifact Analysis:** Neuron (20, 4288) — the paper's highest-weight H-neuron — is a regularization artifact. It vanishes at C<=0.3 where the classifier already reaches 69.5% accuracy. L13:N833 (AUC=0.703) is the true best single predictor despite having 28% of 4288's weight. This is a methodological critique of the paper's neuron ranking.

**2. Evaluator-Format Confound:** At high alpha, the model emits answer text instead of MC letters under the standard FaithEval prompt. Raw compliance appears to drop (69.1% to 63.6%) but strict text remapping at alpha=3.0 recovers the estimate to 72.1% (140/150 parse failures rescued, 93.3%). The negative control confirmed this format shift is H-neuron-specific. Three competing hypotheses: context-credulity (A), surface-form selection (B), precision loss (C). Hypothesis C partly falsified by negative control.

**3. Tri-Modal Population Structure:** 60% of samples always comply, 26% never comply, only 14% (138/1000) are "swing" samples. The 6.3pp headline is actually a 45.7% swing within the sensitive subpopulation. Of swing samples: 68.1% show R-to-C (knowledge override, the concerning pattern), 23.2% show C-to-R (beneficial), 8.7% are non-monotonic. Surface features (word overlap, response length, topic) do NOT predict swing status — the mechanism is internal, not input-driven. LLM enrichment shows 85.5% of swing items are "common knowledge" — the model *knows* these answers but can be overridden.

**4. Jailbreak Truncation Falsification & Severity Discovery:** The original paper's 256-token generation truncated 100% of jailbreak responses during the disclaimer preamble, creating an alpha-dependent truncation artifact. The legacy +6.2pp slope was entirely a measurement error. With 5000-token generation, the binary compliance count is flat (30.4% → 32.2% → 33.4%, CI includes zero). However, a 25-prompt human cross-alpha audit revealed the real effect: **disclaimer erosion** — at higher alphas, harmful responses drop disclaimers, endorse harmful behavior, and provide more actionable detail. The binary HARMFUL/SAFE metric is fundamentally inadequate.

**5. CSV-v2 Graded Evaluation Confirms Severity Slope (NEW — 2026-03-26):** A three-axis graded rubric (Commitment C0-3, Specificity S0-4, Validity V0-3) applied to all 500×3 jailbreak responses recovers the alpha slope that the binary judge missed:
- **csv2_yes rate**: +7.6pp [+3.6, +11.6] — **significant**
- **V=3 (strong/polished harmful content)**: nearly quadruples, 3.8% → 14.0%
- **S=4 (turnkey artifacts)**: nearly triples, 3.0% → 8.4%
- **Harmful payload share**: rises 0.580 → 0.728
- **Pivot position** (where harmful content begins): recedes from 16.3% to 9.6% of response
- **Disclaimer-free rate** among harmful responses: triples (6% → 18%)
- **Borderline population**: monotonically shrinks 171 → 134 → 98 as responses cross the harmful threshold
- The binary judge's noise floor (~60 false inflations from C=1 disclaimer-wrapped responses) washed out the real signal

**Key insight**: H-neuron scaling increases BOTH the count and severity of harmful outputs. The original paper's direction was correct; the measurement methodology was the problem. Category-level patterns show Physical harm has the sharpest severity gradient (V rises from 1.38 to 2.12 despite low compliance rate), while Sexual/Adult paradoxically becomes MORE refusing with alpha (C collapses 0.26 → 0.08).

### 1.3 Jailbreak Status (updated 2026-03-26)

The jailbreak benchmark uses JailbreakBench (100 behaviors × 5 templates), diverging from the paper's forbidden-question-set (390 questions × 1 template). Key issues:
- Template heterogeneity is extreme: T1 accounts for ~40% of harmful responses, T2 is immune
- Stochastic generation (temp=0.7) invalidates per-item flip analysis
- **No negative control yet** — relevant if severity-graded evaluation reveals a significant qualitative effect (it does, via CSV-v2)
- Truncation bias at max_new_tokens=256 was identified and corrected (5000-token canonical rerun)
- **Missing: alpha=1.0 baseline** for 5000-token jailbreak (needed since alpha=1.0 is the unperturbed default activation value)
- CSV-v2 graded evaluation completed — the alpha slope IS significant under the graded metric
- 75-record cross-alpha gold label set exists (25 prompts × 3 alphas, human-labeled)

### 1.4 GLM-5 Judge Evaluation (NEW — 2026-03-26)

GLM-5 (DeepInfra, fp4) evaluated as a GPT-4o replacement for binary jailbreak judging:
- **Better accuracy**: 89.2% vs 85.1% agreement with human gold labels (75 records)
- **Cheaper**: ~$5.52 vs ~$8.63 for 1500 requests (even with reasoning token overhead)
- **Worse throughput**: ~6h serial vs ~1-2h batch (reasoning model, ~10-17s/request)
- Both judges lean permissive (more false-SAFE than false-HARMFUL)
- 3 cases both judges miss are disclaimer-then-comply patterns

### 1.5 Missing Replication Items

Not yet tested on this model: NQ-Open detector eval, NonExist detector eval, Sycophancy intervention.

---

## 2. Confirmed Limitations and Closed Lines

### 2.1 SAE Steering: Definitively Closed

SAE feature-space steering produces null compliance slopes under both tested architectures:
- Full replacement (encode-scale-decode): H-feature slope 0.16 pp/alpha, CI [-0.51, 0.84]
- Delta-only (cancels reconstruction error exactly): H-feature slope 0.12 pp/alpha
- Neuron baseline comparison: 2.09 pp/alpha (13-18x stronger)

The delta-only test was decisive: reconstruction error was NOT the cause. This is a detection-steering dissociation — the SAE probe matches CETT detection (AUROC 0.848 vs 0.843) but the features cannot steer. H-features paradoxically perform WORSE than random at high alpha (69.9% vs 74.6% at alpha=3.0). Layer 20 over-concentration (93/266 features, 35%) parallels the neuron 4288 artifact pattern.

SAE detection works. SAE steering does not. Do not revisit.

### 2.2 Verbosity Confound: Scoped to Detection Side Only

Under full-response aggregation, H-neuron CETT activations are dominated by response length (3.7:1 over truth effect by Cohen's d). 36/38 neurons are length-dominant. BUT this does NOT threaten intervention claims because:
1. FaithEval uses letter extraction — immune to raw length confounding
2. Negative controls rule out indirect response-characteristic channels
3. Paper replicates across 6 models × 4 tasks

The truth signal concentrates in short responses (interaction p=0.0002), which matches the classifier's answer-token training domain. The classifier's AUROC of 0.843 should be interpreted as partly entangled with response-form correlations.

### 2.3 Evaluator Trustworthiness is the Bottleneck

Current known evaluator issues:
- FaithEval standard parse failures scaling with alpha (evaluator artifact, not behavior)
- FalseQA judge-model error outside CI accounting (measured nondeterminism: 0.4%, likely understates systematic bias)
- Binary jailbreak judge over-classifies C=1 borderlines as HARMFUL (~60 false inflations at alpha=0.0), creating noise floor that washes out real slope
- CSV-v2 has a known blind spot: 6/45 gold=HARMFUL records scored as csv2=no (all D=2/F=True disclaimer-framed)
- No judge test-retest reliability for jailbreak
- Full-response remap not yet applied at alpha < 3.0 for FaithEval standard

### 2.4 Missing Capability Degradation Baseline

No measurement exists for whether alpha scaling degrades general capabilities. Hugo explicitly flagged this: "If alpha up to 3 improves anti-compliance, you must check whether you are just lobotomizing general reasoning." This is claim hygiene, not optional polish.

### 2.5 Classifier Selection Caveat

The 38 H-neurons were selected by L1 logistic regression with C=1.0 and AUROC-based selection. The paper's full selection rule also scores TriviaQA behavior after suppression — not implemented here. The current neuron set is a detector-selection baseline, not paper-equivalent.

### 2.6 Jailbreak Negative Control Gap (downgraded but not closed)

The compliance-count increase is non-significant with corrected generation (+3.0pp, CI includes zero), reducing the urgency of a negative control for the count effect. However, CSV-v2 shows a significant severity effect (+7.6pp csv2_yes). A negative control is now relevant for the severity/graded claim, not the binary count.

---

## 3. Current Infrastructure and Constraints

**Hardware:** RTX 5060 Ti 16 GB VRAM, AMD 9900X. Single consumer machine.

**What exists in the codebase:**
- H-neuron detector (classifier.py) with C-sweep
- Intervention hooks at generation time (run_intervention.py, HNeuronScaler)
- Negative control infrastructure (run_negative_control.py, run_sae_negative_control.py)
- Benchmark loaders: FaithEval, FalseQA, Jailbreak (JailbreakBench), BioASQ, Sycophancy (scaffolded)
- Answer-token extraction and CETT activation dumps
- Swing characterization pipeline (characterize_swing.py)
- SAE tooling: extraction, classification, intervention, analysis (all in place, steering closed)
- GPT-4o judging pipeline with structured rubrics
- CSV-v2 graded jailbreak evaluation pipeline (evaluate_csv2.py)
- 75-record cross-alpha gold label set with human labels
- GLM-5 judge evaluation pipeline and comparison results
- Site/visualization for results
- Provenance sidecar system for all runs

**What does NOT exist:**
- Circuit discovery stack (no TransformerLens/nnsight integration yet, though both now support Gemma 3)
- Attribution patching or GCM pipeline
- Capability degradation measurement
- Safety/refusal neuron identification
- Conditional gating infrastructure
- Negative-weight (suppressive) neuron investigation
- Jailbreak negative control (for severity-graded metric)
- Alpha=1.0 jailbreak baseline at 5000 tokens

**Budget:** ~$200 incremental compute/API spend. Local GPU time is free but slow.

**Available SAE tooling:** Google's gemma-scope-2-4b-it provides pretrained SAEs for all 34 layers (16k/65k/262k widths). SAELens integration works. Detection works; steering is closed.

**Available interpretability libraries:** TransformerLens v2.17.0+ (Gemma 3 support), v3.0.0 beta (TransformerBridge with full hook coverage). NNsight 0.6 supports Gemma 3 and enables remote execution via NDIF.

---

## 4. Strategic Context 

### 4.1 Hugo's Current Thinking (from nextsteps.md, late March 2026)



> "We've reached the end of what we can do with the approach of the original authors. A lot of the ideas we have is to try to make their methodology better, but maybe the foundation of just picking these 38 neurons needs to be revisited. We anchored too hard on that one single paper."

> "Act One was replication, Act Two was finding the limitations and flaws. I think we've been digging too deep into this paper. At the foundation level, it's just not the right approach. These neurons are interesting to play with, but it's not the way we're going to get the model significantly safer."

> "I want the Act 3 to be: first do a literature review to see what are the best, most efficient techniques as of today. Then lead into that. Do the best we can with negative neurons or different C parameters, then compare that to the best available techniques. Close the loop and say: these neurons are interesting but polysemantic, and recommend whatever we find."

> "The beautiful path ahead is: how do we go from in-depth replication and critique to finding something generally useful — pragmatic interpretability that can be used by other people and make the model safer for everyone."

Key mentor questions:
- Could we use techniques from the refusal/jailbreak literature to make models hallucinate less, connecting to our H-neuron data?
- What about negative-weight neurons (the paper ignores them) — do they make the model safer when amplified?
- Can we compare H-neuron methods to the best available safety techniques and produce a comparative study?

### 4.2 Previous Strategic Reports Status

| Direction | Status |
|---|---|
| 1. Negative control | DONE — H-neuron specificity confirmed on FaithEval and FalseQA |
| 2. Suppressive neurons | NOT STARTED — C-sweep stability check is 10 min CPU |
| 3. Swing characterization | DONE — see Section 1.2 above |
| 4. Base model transfer | NOT STARTED — quick activation check is 10 min GPU |
| 5. Context-credulity hypothesis | NOT STARTED — 30 min GPU falsification test |
| 6. Safety/refusal overlap | NOT STARTED — 30 min GPU pilot |
| 7. Uncertainty-gated intervention | NOT STARTED — complex, 5-7 days |
| 8. SAE feature decomposition | CLOSED — detection works, steering null |

### 4.3 What Previous Strategic Guidance Said (before mentor pivot)

- Do NOT broaden yet — turn experiments into airtight claim package
- Highest ROI: measurement hardening (sentinel sets, cross-evaluator audits, frozen manifests)
- Close jailbreak specificity gap (negative control) — partially resolved by CSV-v2
- Add minimal capability degradation assay
- Lean into swing population result
- Do NOT revisit SAE steering
- Do NOT treat BioASQ as main story
- Do NOT run sycophancy just for breadth

### 4.4 Tension Between Previous Guidance and Hugo Direction

There is a real tension: previous strategic advice said "harden before broadening," but the mentor now wants to broaden — literature review, comparative study, new techniques. The question is how to honor both: produce a rigorous final package of what we have while connecting to the broader landscape the mentor wants to explore.

---

## 5. Key Related Papers and Methods

A preliminary literature review is at `docs/literature-review-act3.md` with detailed method comparisons and a ranked experiment list. Below is the full inventory of papers collected so far (in `papers/`). **This list is not exhaustive — more relevant work likely exists, especially on hallucination-specific steering and uncertainty-guided intervention.**

### 5.1 Papers on hand (with abstracts)

**Gao et al., "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs" (arXiv:2512.01797)** — The paper being replicated. Identifies <0.1% of neurons as hallucination-predictive via L1 probe on TriviaQA activations; shows causal link to over-compliance via scaling; traces neurons to pre-training.

**Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (NeurIPS 2024)** — Across 13 chat models (1.8B-72B), refusal is mediated by a 1D subspace in the residual stream. A single difference-in-means vector (128+128 harmful/harmless) can ablate or induce refusal. Weight orthogonalization achieves 78-84% ASR on HarmBench; capability degradation <1% on MMLU/ARC/GSM8K. The refusal direction exists in base models pre-safety-finetuning. **Strongest comparator — if H-neurons are a noisy proxy for this direction, a DiM vector should outperform our 38-neuron classifier on all benchmarks.**

**Chen et al., "Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons" (NeurIPS 2025)** — ~5% of neurons across multiple LLMs are "safety neurons" identifiable via inference-time activation contrasting. Patching only these neurons restores >90% safety performance without affecting general ability. Safety and helpfulness neurons significantly overlap but require different activation patterns (explains "alignment tax"). **Closest methodological comparator to H-Neurons — their identification is causal (activation patching) rather than correlational (L1 probe).**

**Sankaranarayanan et al., "Surgical Activation Steering via Generative Causal Mediation" (arXiv:2602.16080, Feb 2026)** — Introduces GCM for selecting model components (e.g. attention heads) to steer binary concepts from contrastive long-form responses. Constructs contrastive input-response pairs, quantifies how individual components mediate the concept, selects strongest mediators. Evaluated on refusal, sycophancy, style transfer across 3 models. GCM consistently outperforms correlational probe-based baselines when steering with sparse attention heads. Attribution patching variant (2 forward + 1 backward) is equally performant and much cheaper than full activation patching. **Directly undermines the L1 classifier approach: correlational probes select suboptimal intervention targets compared to causal localization.**

**Joad et al., "There Is More to Refusal in Large Language Models than a Single Direction" (arXiv:2602.02132, Feb 2026)** — Across 11 categories of refusal/non-compliance, refusal behaviors correspond to geometrically distinct directions in activation space. Yet linear steering along any refusal-related direction produces nearly identical refusal-over-refusal trade-offs. Using SAEs, they find a small reusable core of shared refusal latents plus a long tail of style/domain-specific latents. **Nuances the single-direction finding — if refusal is multi-dimensional, there may be room for neuron-level interventions to capture aspects the single direction misses.**

**Obeso, Arditi, Ferrando et al., "Real-Time Detection of Hallucinated Entities in Long-Form Generation" (2025?)** — Presents cheap, scalable linear probes for real-time identification of hallucinated tokens in long-form generations, scaled to 70B models. Targets entity-level hallucinations (fabricated names, dates, citations). Outperforms semantic entropy (AUC 0.90 vs 0.71 for Llama-3.3-70B). Probes trained on entity labels also generalize to math reasoning errors. Cross-model transfer works. **Directly relevant to our detection story — their probes on the same task type achieve higher AUC than our CETT-based classifier (0.90 vs 0.843).**

### 5.2 Papers referenced but not on hand

| Paper | Relevance |
|---|---|
| Ferrando et al., "Do I Know This Entity?" (arXiv:2411.14257) | SAE features encoding entity self-knowledge; showed steering on Gemma family models |
| Farquhar et al., "Detecting hallucinations using semantic entropy" (Nature 2024) | Uncertainty estimation for hallucination detection |
| Kossen et al., "Semantic Entropy Probes" (NeurIPS 2025) | Cheap linear probes on hidden states for uncertainty |
| Li et al., "Analysing the Safety Pitfalls of Steering Vectors" (arXiv:2603.24543, Mar 2026) | Steering toward any concept can erode safety by up to 57% ASR increase via overlap with refusal direction. r=-0.96 correlation. **Testable hypothesis: measure cosine similarity between our 38-neuron vector and the refusal direction.** |
| Wu et al., "AxBench" (ICML 2025) | Simple prompting/finetuning baselines outperform SAE-based steering |
| "Adaptive Activation Steering for Truthfulness" (WWW 2025) | Tuning-free inference-time truthfulness steering — if this works, H-neuron approach is a more expensive version |
| "A Few Bad Neurons" (January 2026) | Surgical sycophancy correction via sparse neuron identification |

---

## 6. The Discussion I Need

Given everything above — the confirmed findings, the closed lines, the known limitations, the hardware constraints, the strategic context, **and especially the mentor's pivot toward broader comparative work** — what are the most pragmatically beneficial extensions for AI safety?

Specifically:
1. Which extensions would produce the highest ratio of (safety-relevant scientific insight) to (cost + risk)?
2. Which of the known limitations, if addressed, would most strengthen the existing claims?
3. What is the strongest publication story that can be built from the current evidence plus one focused extension?
4. Are there extensions I haven't considered that the data naturally supports?
5. For each recommended direction: what is the cheapest falsification test, what decision does it inform, and what do I learn from each outcome?
6. **How do we reconcile the mentor's desire to broaden (literature review, comparative techniques, negative neurons) with the measurement-hardening advice? What is the practical path that honors both?**
7. **Given the CSV-v2 severity finding and the disclaimer-erosion gradient, what is the strongest contribution this project can make to the jailbreak evaluation methodology literature?**

Ground your recommendations in the actual data and limitations documented above. Do not propose directions that are already closed (SAE steering) or infeasible on this hardware (full circuit discovery, LRMs, pretraining studies). Be honest about what would be a genuine contribution vs what would be incremental.

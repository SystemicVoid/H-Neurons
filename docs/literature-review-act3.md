# Literature Review: Steering & Safety Techniques Beyond H-Neurons

> Reference document. Live execution is tracked in [act3-sprint.md](./act3-sprint.md). Live evaluation rules are tracked in [measurement-blueprint.md](./measurement-blueprint.md).

Date: 2026-03-27 | Purpose: Ground Act 3 decisions in current best techniques

## 1. The Landscape (as of March 2026)

### 1.1 Refusal Is Mediated by a Single Direction (Arditi et al., NeurIPS 2024)

**Core claim**: Across 13 chat models (1.8B-72B), refusal is mediated by a one-dimensional subspace in the residual stream. A single difference-in-means vector, computed from ~128 harmful vs ~128 harmless instructions, can be ablated to disable refusal or added to induce it.

**Method**: Difference-in-means on post-instruction token activations. Select the single best (layer, position) vector by evaluating bypass/induce/KL scores on a 32-sample validation set. Intervene via directional ablation (all layers, all positions) or activation addition (single layer).

**Key results**:
- Weight orthogonalization (equivalent to directional ablation) achieves 78-84% ASR on HarmBench for Qwen models, competitive with GCG prompt-specific attacks
- MMLU/ARC/GSM8K degradation is <1% on average; TruthfulQA consistently drops 1-6%
- The refusal direction exists already in base models (pre-safety-finetuning), suggesting safety training "hooks into" a pre-existing representation rather than creating one
- Adversarial suffixes work by hijacking attention of the top refusal-writing heads, shifting attention from instruction region to suffix region

**Relevance to our work**: This is the strongest comparator. If H-neurons are just a noisy proxy for the refusal direction, then a difference-in-means direction should outperform our 38-neuron L1 classifier on all our benchmarks. The ~128-sample cost is trivially cheap on our hardware.

**Paper**: `papers/refusal-mediated-single-direction2406.11717v3.pdf` | Code: github.com/andyrdt/refusal_direction

---

### 1.2 Finding Safety Neurons (Chen et al., NeurIPS 2025)

**Core claim**: ~5% of neurons across multiple LLMs are "safety neurons" identifiable via inference-time activation contrasting. Patching only these neurons restores >90% safety performance without affecting general ability.

**Method**: Two-stage: (1) inference-time activation contrasting to locate neurons, (2) dynamic activation patching to evaluate causal effects on long-range safety outputs. Uses generation-time signals, not just answer tokens.

**Key results**:
- Safety and helpfulness neurons significantly overlap but require *different activation patterns* for the same neurons (explains "alignment tax")
- Can detect unsafe outputs before generation by monitoring safety neuron activations
- Neuron sets are stable across random seeds

**Relevance to our work**: This is the closest methodological comparator to the H-Neurons paper. Their neuron identification is causal (activation patching) rather than correlational (L1 probe). The overlap finding between safety and helpfulness neurons maps directly onto our polysemanticity observations.

**Paper**: `papers/Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons2406.14144v2.pdf` | Code: github.com/THU-KEG/SafetyNeuron

---

### 1.3 Surgical Activation Steering via Generative Causal Mediation (Sankaranarayanan et al., Feb 2026)

**Core claim**: Probe-based localization (correlational) is insufficient for steering concepts diffused across long-form responses. Generative Causal Mediation (GCM) identifies attention heads as causal mediators and consistently outperforms probe-based baselines for sparse steering.

**Method**: Construct contrastive prompt-response pairs. Measure indirect effect of patching individual attention heads on the probability of generating the contrasting response. Rank heads by indirect effect, steer top-k% with difference-in-means, mean steering, or ReFT.

**Key results**:
- GCM outperforms ITI (linear probe) head selection for refusal induction, sycophancy reduction, and style transfer across SOLAR, Qwen, OLMo
- Attribution patching variant (2 forward + 1 backward pass) is equally performant and much cheaper than full activation patching
- Steering success correlates with MMLU degradation -- fundamental trade-off between controllability and capability preservation

**Relevance to our work**: Directly undermines the L1 classifier approach used in H-Neurons. GCM shows that correlational probes (like L1 logistic regression) select suboptimal intervention targets compared to causal localization. This is exactly the critique we should articulate. Also, the attention-head-level intervention is more principled than individual-neuron scaling.

**Paper**: `papers/Surgical Activation Steering via Generative Causal Mediation-2602.16080v1.pdf`

---

### 1.4 Analysing the Safety Pitfalls of Steering Vectors (Li et al., Mar 2026)

**Core claim**: Activation steering is not safety-neutral. Steering toward any behavioral concept can systematically erode safety alignment by up to 57% ASR increase, because steering vectors overlap with the refusal direction.

**Method**: Contrastive Activation Addition (CAA) vectors for diverse behavioral traits (self-awareness, hallucination, sycophancy, etc.) evaluated on JailbreakBench across 6 models (3B-32B). Measured cosine similarity between steering vectors and refusal direction.

**Key results**:
- Even "unrelated" steering (e.g., self-awareness) can increase ASR by 42% because of geometric overlap with refusal direction
- Refusal-direction ablation from steering vectors mitigates safety erosion
- Larger models show *greater* susceptibility to steering-induced safety degradation
- Correlation between refusal-direction alignment and ASR change is r=-0.96 to -0.78

**Relevance to our work**: Our H-neuron amplification results (jailbreak +6.2pp) may be partially explained by this mechanism -- amplifying neurons that happen to overlap with the refusal direction. The paper provides a testable hypothesis: measure cosine similarity between our 38-neuron intervention vector and the refusal direction.

**Paper**: arXiv:2603.24543 (published 2 days ago)

---

### 1.5 AxBench: Simple Baselines Outperform Sparse Autoencoders (Wu et al., ICML 2025)

**Core claim**: For fine-grained steering of LLM outputs, simple prompting and finetuning baselines outperform SAE-based steering on a comprehensive benchmark.

**Relevance to our work**: Reinforces the message that interpretability-derived interventions (neurons, SAE features) do not yet reliably outperform simpler methods for behavioral control. Our paper should acknowledge this broader context.

---

### 1.6 Adaptive Activation Steering for Truthfulness (WWW 2025)

**Core claim**: Tuning-free, inference-time activation steering can improve LLM truthfulness by treating truthfulness as a contrastive concept and steering toward it adaptively.

**Relevance to our work**: Directly relevant to our FaithEval/hallucination benchmarks. If truthfulness directions can be steered, then our H-neuron approach to hallucination is a more expensive, less principled version of the same idea.

---

### 1.7 There Is More to Refusal Than a Single Direction (Feb 2026)

**Core claim**: Refusal behavior is not purely one-dimensional; higher-dimensional directions also play a role, and the single-direction finding may be incomplete.

**Relevance to our work**: Nuances the Arditi et al. finding. If refusal is multi-dimensional, there may be room for neuron-level interventions to capture aspects the single direction misses -- but GCM/attribution patching would still be more principled than L1 probe selection.

**Paper**: `papers/There Is More to Refusal in Large Language Models than a Single Direction-2602.02132v1.pdf`

---

## 2. Comparison Matrix: H-Neurons vs. Current Techniques

| Dimension | H-Neurons (our baseline) | Refusal Direction (Arditi) | Safety Neurons (Chen) | GCM (Sankaranarayanan) |
|---|---|---|---|---|
| **Localization** | L1 probe on TriviaQA activations | Difference-in-means on harmful/harmless | Activation contrasting + causal patching | Causal mediation on contrastive generations |
| **Granularity** | Individual FFN neurons (38) | Single residual-stream direction | ~5% of all neurons | Top-k% attention heads |
| **Causal validation** | None (correlational only) | Ablation + addition (both work) | Dynamic activation patching | Indirect effect measurement |
| **Capability preservation** | Not measured in paper | <1% on MMLU/ARC/GSM8K | >90% general ability retained | Measured via MMLU transfer |
| **Data requirement** | TriviaQA answer-token activations | 128+128 harmful/harmless instructions | Contrastive harmful/harmless sets | Contrastive prompt-response pairs |
| **Intervention** | Scale neuron activations | Ablate/add direction | Patch neuron activations | Steer attention heads |
| **Scope** | Hallucination-correlated only | Refusal/safety only | Safety broadly | Any binary concept |
| **Benchmarks** | FaithEval, FalseQA, Jailbreak, BioASQ | JailbreakBench, HarmBench | Multiple red-team benchmarks | Refusal, sycophancy, style transfer |

## 3. What This Means for Act 3

### The core critique is now clear:

The H-Neurons paper identifies 38 neurons via an L1 logistic regression probe -- a **correlational** method that does not validate whether these neurons **causally mediate** the behaviors of interest. Our replication confirms this: the neurons do shift behavior when scaled, but the effects are noisy, non-monotonic, and confounded by polysemanticity and verbosity.

Meanwhile, the field has converged on **direction-level** and **causally-validated** approaches:
- Refusal is better captured by a single residual-stream direction than by 38 FFN neurons
- Safety neurons identified by causal methods are more reliable (5% of neurons, >90% safety recovery)
- GCM shows that even probe-based head selection is outperformed by causal localization
- Steering vectors can inadvertently erode safety because of geometric overlap with refusal

### The strongest Act 3 experiments (ranked by information/time):

1. **Refusal direction comparison** (~2-3 days): Extract the Arditi-style refusal direction for Gemma-3-4B-IT. Run it on our exact benchmarks (FaithEval, FalseQA, Jailbreak). Compare directly to H-neuron scaling. Measure cosine similarity between the refusal direction and our 38-neuron classifier weight vector. This single experiment would:
   - Show whether H-neurons are a noisy proxy for the refusal direction
   - Provide a proper baseline for what a principled technique achieves
   - Connect our detailed replication to the current best method

2. **Negative neurons** (~1 day): Already connected to existing work. Quick to run. If they have a corrective effect, that's a mitigation story. If not, the asymmetry is informative.

3. **Hallucination-specific direction** (~3-4 days, highest novelty): Use the same difference-in-means technique but on hallucination-contrasting prompts (using our existing FaithEval data). Extract a "hallucination direction" and test whether steering along it reduces hallucination without destroying capabilities. This is the genuinely novel contribution -- most refusal work focuses on safety/jailbreak, not hallucination.

### The paper arc:

Act 1 (done): Deep replication of H-Neurons methodology on Gemma-3-4B-IT
Act 2 (done): Systematic critique -- truncation bias, polysemanticity, negative controls, evaluation artifacts
Act 3: "From neurons to directions" -- show that principled direction-level techniques subsume and outperform neuron-level approaches, with a novel application to hallucination steering

## 4. Key References

| Short name | Full citation | Where |
|---|---|---|
| Arditi2024 | Arditi et al., "Refusal in LLMs is mediated by a single direction", NeurIPS 2024 | `papers/refusal-mediated-single-direction2406.11717v3.pdf` |
| Chen2025 | Chen et al., "Towards Understanding Safety Alignment: Safety Neurons", NeurIPS 2025 | `papers/Towards Understanding Safety Alignment...2406.14144v2.pdf` |
| Sankaranarayanan2026 | Sankaranarayanan et al., "Surgical Activation Steering via GCM", arXiv 2602.16080 | `papers/Surgical Activation Steering...2602.16080v1.pdf` |
| Li2026 | Li et al., "Analysing the Safety Pitfalls of Steering Vectors", arXiv 2603.24543 | arXiv (published Mar 25, 2026) |
| Wu2025 | Wu et al., "AxBench: Steering LLMs? Even Simple Baselines Outperform SAEs", ICML 2025 | arXiv:2501.17148 |
| MultiRefusal2026 | "There Is More to Refusal Than a Single Direction", arXiv 2602.02132 | `papers/There Is More to Refusal...2602.02132v1.pdf` |
| AAS2025 | "Adaptive Activation Steering: Tuning-Free LLM Truthfulness Improvement", WWW 2025 | arXiv:2406.00034 |

## 5. Deep Reads: Key Paper Summaries

### 5.1 On the Universal Truthfulness Hyperplane Inside LLMs (Liu et al., 2024)

**arXiv:2407.08582 | c=16 | Code: github.com/hkust-nlp/Universal_Truthfulness_Hyperplane**

**Why this paper is critical for us**: It is the hallucination analog of Arditi's refusal direction, but with a crucial twist -- they show that probes trained on a single dataset (like the H-Neurons paper's TriviaQA-only approach) **fail to generalize OOD by 25 absolute points**. This is exactly the critique we should make.

**Method**: Train linear probes (logistic regression LR and mass-mean MM, i.e. difference-in-means) on last-token representations from attention heads across 40+ datasets spanning 17 task categories. Use a location selection strategy to pick the top-k attention heads that best predict truthfulness, then concatenate those representations.

**Key results**:
- Probes trained on TruthfulQA alone drop 25pp when evaluated on other datasets -- **directly mirrors our H-Neuron overfitting concern**
- Increasing dataset diversity (+14pp cross-task) matters far more than increasing data volume (only 10 samples/dataset needed)
- Attention head outputs >> layer residual activations for truthfulness probing
- Mass-mean (difference-in-means) comparable to logistic regression, with better generalization
- Cross-task accuracy ~70% with diverse training; in-domain ~80%
- Truthfulness features are **sparse**: only a few attention heads carry the signal

**Direct implications for our project**:
1. The H-Neurons paper trains its L1 probe on TriviaQA only -- Liu et al. show this will overfit to TriviaQA-specific features, not universal hallucination signal
2. Their mass-mean method (equivalent to Arditi's difference-in-means) is a proper baseline to compare against our L1 classifier
3. They use attention heads, not FFN neurons -- suggesting the right granularity is head-level, not neuron-level
4. Their finding that diversity > volume directly explains why our 38 neurons have noisy, non-monotonic effects across benchmarks: they capture TriviaQA-specific signal, not universal truthfulness

### 5.2 Citation Graph Discoveries

**From Truthfulness Hyperplane citations (papers citing it)**:
- "What do Geometric Hallucination Detection Metrics Actually Measure?" (arXiv:2602.09158) -- meta-analysis of hallucination detection geometry. Relevant to validating our approach.
- "How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence" (arXiv:2504.02904, c=5) -- connects truthfulness and refusal representations mechanistically. Could bridge our hallucination and safety stories.
- "Prompt-Guided Internal States for Hallucination Detection" (arXiv:2411.04847, c=5) -- prompt-guided internal state analysis, methodological comparator.
- "Probing the Geometry of Truth: Consistency and Generalization of Truth Directions" (ACL 2025) -- directly tests whether truth directions generalize across logical transformations.

**From Geometry of Refusal citations (papers citing it)**:
- "SOM Directions are Better than One: Multi-Directional Refusal Suppression" (arXiv:2511.08379, c=4) -- multi-directional refusal, extends single-direction finding
- "Curveball Steering: The Right Direction To Steer Isn't Always Linear" (arXiv:2603.09313) -- non-linear steering, challenges linear representation hypothesis
- "Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics" (arXiv:2602.02343) -- theoretical foundation for why steering works

**From Hallucination Detection citations**:
- "From Out-of-Distribution Detection to Hallucination Detection: A Geometric View" (arXiv:2602.07253) -- geometric framework for hallucination detection
- "The Confidence Manifold: Geometric Structure of Correctness Representations" (arXiv:2602.08159) -- geometric structure of correctness in LLMs
- "Can Linear Probes Measure LLM Uncertainty?" (arXiv:2510.04108) -- directly questions whether linear probes (like ours) capture real uncertainty vs artifacts

### 5.3 Probe vs Direction Literature (from web search + S2 refs)

The Truthfulness Hyperplane paper's reference chain reveals the key comparison:
- **Burns et al. 2023 (CCS)**: Unsupervised contrastive probe. c=605. Discovers latent knowledge without supervision but fails OOD.
- **Marks & Tegmark 2023 (Geometry of Truth)**: Mass-mean (difference-in-means). c=432. Shows emergent linear structure for truth/falsehood. Simple diff-in-means generalizes as well as trained probes.
- **Li et al. 2023 (ITI)**: Inference-Time Intervention. c=962. Uses probes on attention heads to find truth-related heads, then steers at inference time. **This is the closest methodological ancestor to combining our detection with intervention.**
- **Gurnee et al. 2023 (Finding Neurons in a Haystack)**: k-sparse probing. c=311. Shows features are sparsely distributed, only a few neurons carry each concept. Directly relevant to our 38-neuron story.

**The emerging consensus**: Single-dataset probes (LR, CCS) overfit. Difference-in-means generalizes better. Multi-dataset diverse training works best. The right intervention targets are attention heads, not individual FFN neurons. Our H-Neuron L1 classifier is on the wrong side of every one of these findings.

## 6. Semantic Scholar Recommendations (retrieved 2026-03-27)

### 5.1 High-Priority Papers to Investigate (from multi-paper recommendations)

Papers recommended based on all 6 core papers as positive examples:

| # | Paper | Year | arXiv | Why relevant |
|---|---|---|---|---|
| 1 | **SafeNeuron: Neuron-Level Safety Alignment for LLMs** | 2026 | 2602.12158 | Direct methodological comparator -- neuron-level safety alignment. Compare their approach to H-Neurons. |
| 2 | **From Refusal Tokens to Refusal Control: Discovering and Steering Category-Specific Refusal Directions** | 2026 | 2603.13359 | Category-specific refusal directions -- relevant to our finding that different benchmarks respond differently to H-neuron scaling. |
| 3 | **Steering Externalities: Benign Activation Steering Unintentionally Increases Jailbreak Risk** | 2026 | 2602.04896 | Directly relevant -- shows benign steering has safety side effects, parallels our jailbreak compliance findings. |
| 4 | **Steer2Edit: From Activation Steering to Component-Level Editing** | 2026 | 2602.09870 | Bridge between steering and editing -- relevant to practical interventions. |
| 5 | **SafeSeek: Universal Attribution of Safety Circuits in Language Models** | 2026 | 2603.23268 | Universal safety circuit attribution -- could be the right framework for our Act 3 comparison. |
| 6 | **NeST: Neuron Selective Tuning for LLM Safety** | 2026 | 2602.16835 | Another neuron-level safety paper. Compare methodology to H-Neurons. |
| 7 | **The Struggle Between Continuation and Refusal: A Mechanistic Analysis** | 2026 | 2603.08234 | Mechanistic analysis of jailbreak -- relevant to our truncation bias findings. |
| 8 | **Controllable Value Alignment via Neuron-Level Editing** | 2026 | 2602.07356 | Neuron-level editing for alignment -- direct comparison point. |
| 9 | **Identifying and Transferring Reasoning-Critical Neurons** | 2026 | 2601.19847 | Neuron identification + activation steering for reasoning -- methodological parallel. |
| 10 | **Fine-Grained Activation Steering: Steering Less, Achieving More** | 2026 | 2602.04428 | Efficiency of steering -- relevant to our "less is more" findings with sparse neurons. |

### 5.2 Hallucination-Focused Recommendations (H-Neurons + Safety Neurons positive, refusal papers negative)

| # | Paper | Year | arXiv | Why relevant |
|---|---|---|---|---|
| 1 | **Dynamic Multimodal Activation Steering for Hallucination Mitigation** | 2026 | 2602.21704 | Activation steering for hallucination -- direct bridge to our FaithEval work. |
| 2 | **On the Universal Truthfulness Hyperplane Inside LLMs** | 2024 | 2407.08582 | c=16. Truthfulness as a linear direction -- analogous to refusal direction but for hallucination. Key comparator. |
| 3 | **Do LLMs Know about Hallucination? An Empirical Investigation of Hidden States** | 2024 | 2402.09733 | c=55. Internal state analysis for hallucination detection -- uses same residual stream as our approach. |
| 4 | **In-Context Sharpness as Alerts: Inner Representation Perspective for Hallucination** | 2024 | 2403.01548 | c=45. Inner representations for hallucination -- activation-based hallucination detection. |
| 5 | **Unsupervised Real-Time Hallucination Detection based on Internal States** | 2024 | 2403.06448 | c=79. Highest-cited internal-state hallucination detection. Must-read for hallucination direction work. |
| 6 | **Weakly Supervised Detection of Hallucinations in LLM Activations** | 2023 | 2312.02798 | c=23. Early activation-based hallucination detection. |
| 7 | **Mechanisms of Non-Factual Hallucinations in Language Models** | 2024 | N/A | c=12. Mechanistic hallucination analysis -- direct methodological comparator. |
| 8 | **Locate-then-Sparsify: Attribution Guided Sparse Strategy for Hallucination** | 2026 | 2603.16284 | Attribution-guided sparse intervention -- bridges our sparse neuron approach with causal attribution. |
| 9 | **Conditional Mechanistic Weight Edits for Targeted Hallucination Reduction** | 2026 | N/A | Mechanistic weight editing for hallucination -- connects to our intervention work. |
| 10 | **Who Transfers Safety? Identifying Cross-Lingual Shared Safety Neurons** | 2026 | 2602.01283 | Cross-lingual safety neurons -- tests universality of neuron-level findings. |

### 5.3 High-Citation Steering/Refusal Papers (from Arditi single-paper recs)

| # | Paper | Year | arXiv | Citations | Why relevant |
|---|---|---|---|---|---|
| 1 | **The Geometry of Refusal: Concept Cones and Representational Independence** | 2025 | 2502.17420 | 42 | Extends Arditi to multi-dimensional refusal geometry. Nuances our single-direction comparison. |
| 2 | **Steering Refusal with SAEs** | 2024 | 2411.11296 | 54 | SAE features for refusal steering -- bridges our SAE experiment with refusal work. |
| 3 | **Understanding Refusal with SAEs** | 2025 | 2505.23556 | 7 | Mechanistic understanding of refusal via SAE -- connects to our FaithEval-SAE experiments. |
| 4 | **Equilibrate RLHF: Balancing Helpfulness-Safety Trade-off** | 2025 | 2502.11555 | 21 | Helpfulness-safety trade-off -- directly relevant to our polysemanticity findings. |

### 5.4 Triage: Papers to Read First

**For the refusal-direction comparison experiment** (Act 3 priority #1):
1. arXiv:2502.17420 -- "Geometry of Refusal" (multi-dimensional refusal, nuances single-direction claim)
2. arXiv:2603.13359 -- "From Refusal Tokens to Refusal Control" (category-specific directions)
3. arXiv:2602.04896 -- "Steering Externalities" (safety side effects of benign steering)

**For the hallucination-direction experiment** (Act 3 priority #3):
1. arXiv:2403.06448 -- "Unsupervised Real-Time Hallucination Detection" (c=79, internal states)
2. arXiv:2407.08582 -- "Universal Truthfulness Hyperplane" (c=16, linear truthfulness direction)
3. arXiv:2402.09733 -- "Do LLMs Know about Hallucination?" (c=55, hidden state analysis)
4. arXiv:2602.21704 -- "Dynamic Multimodal Activation Steering for Hallucination"

**For neuron-level methodology comparison**:
1. arXiv:2602.12158 -- "SafeNeuron" (neuron-level safety alignment)
2. arXiv:2602.16835 -- "NeST" (neuron selective tuning)
3. arXiv:2602.07356 -- "Controllable Value Alignment via Neuron-Level Editing"

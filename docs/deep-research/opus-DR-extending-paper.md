# Mechanistic interpretability of hallucination: the H-Neurons landscape and beyond

**Hallucination in large language models is not a diffuse, intractable failure — it is concentrated in remarkably sparse neural circuits that can be identified, studied, and intervened upon.** The H-Neurons paper (Gao et al., December 2025) crystallized this insight by showing that fewer than **0.1% of FFN neurons** in models like Llama-3.3-70B causally predict hallucination events, and that these same neurons encode a broader disposition toward over-compliance — including sycophancy and jailbreak susceptibility. This finding sits at the intersection of several rapidly converging research threads: sparse autoencoder-based feature discovery, truth direction probing, activation steering, and theoretical impossibility results on hallucination. This report maps the full 2024–2026 landscape of mechanistic interpretability research surrounding hallucination, organized around the H-Neurons contribution and its connections to the broader field.

---

## H-Neurons identifies a sparse hallucination substrate with surprising behavioral breadth

The core paper from Tsinghua's THUNLP group introduces a three-part framework for understanding hallucination-associated neurons. First, the authors quantify each FFN neuron's contribution to hidden states using the **CETT metric** (contribution via entity token transfer), which normalizes a neuron's projected output against the total layer output. They then train an L1-regularized logistic regression classifier on CETT features extracted from TriviaQA responses, where the sparsity penalty automatically selects the most predictive neurons. Neurons with positive classifier weights are designated H-Neurons.

The results are striking across all six tested models (Mistral-7B/24B, Gemma-4B/27B, Llama-8B/70B). On Llama-3.3-70B, just **0.01‰ of neurons achieve 82.7% accuracy** on TriviaQA hallucination prediction and **96.7% on fabricated entities** (NonExist dataset). These neurons generalize across domains — from open-domain QA to biomedical questions (BioASQ) — suggesting they encode a domain-general hallucination disposition rather than topic-specific failure modes.

The behavioral impact finding is what distinguishes H-Neurons from prior probing work. Activation scaling experiments (multiplying neuron activations by α ∈ [0, 3]) reveal that **amplifying H-Neurons increases over-compliance across four distinct benchmarks**: compliance with invalid premises (FalseQA), compliance with misleading contexts (FaithEval), agreement with incorrect user feedback (sycophancy), and compliance with harmful instructions (jailbreak). Smaller models show steeper scaling slopes (~3.03 vs. ~2.40 for larger models), indicating greater vulnerability. The third finding — that H-Neurons originate during pre-training, not post-training alignment — is supported by backward transferability experiments showing classifiers trained on instruction-tuned models retain predictive power (AUROC ~0.86) on base models, and by cosine similarity analysis showing minimal parameter drift during alignment.

The official implementation at **github.com/thunlp/H-Neurons** provides response collection via vLLM, CETT extraction, sparse classifier training, and intervention scripts for modulating activations during forward passes.

---

## Direct follow-up work extends H-Neurons to adaptive, cross-scale interventions

The most direct extension is **Adaptive Activation Cancellation (AAC)** by Yocam et al. (March 2026, arXiv:2603.10195), which reframes H-Neuron suppression through a signal-processing lens. Drawing an explicit analogy to classical adaptive noise cancellation, AAC identifies "H-Nodes" via layer-wise linear probing and suppresses them using a confidence-weighted forward hook during autoregressive generation. Evaluated across three scales — OPT-125M, Phi-3-mini (3B), and LLaMA 3-8B — AAC achieves consistent improvements on TruthfulQA and HaluEval while preserving **exactly 0.0% degradation** on WikiText-103 perplexity and MMLU accuracy. The paper reveals an important scaling phenomenon: a **"polysemanticity scale-trap"** at the 3B–4B parameter range where neurons are maximally polysemantic, making sparse suppression harder, alongside "mechanistic delocalization" where intervention effects distribute more broadly in larger models.

Other related work includes **MHAD** (IJCAI 2025), which uses linear probing to select neurons across multiple layers for hallucination detection, and **factuality probes** (EMNLP 2025 Findings) demonstrating that neuron-level steering significantly improves factuality across all evaluated LLMs without fine-tuning. A **geometric hallucination taxonomy** (arXiv:2603.00307, March 2026) offers an alternative classification framework studying hallucination structures in GPT-2.

---

## SAE-based feature discovery converges on similar hallucination mechanisms from a different direction

Sparse autoencoder research has independently identified hallucination-related internal structures, approaching the problem through learned overcomplete dictionaries rather than individual neurons. The complementarity between these approaches is a defining feature of the current landscape.

**Ferrando et al.'s "Do I Know This Entity?"** (ICLR 2025 Oral, arXiv:2411.14257) is the most directly parallel work to H-Neurons. Using Gemma Scope and LlamaScope SAEs on Gemma 2 (2B/9B) and Llama 3.1 8B, they discover SAE latents encoding **"knowledge awareness" — whether a model recognizes an entity it is being asked about**. These latents are causally relevant: steering with the "known entity" latent on unknown entities forces hallucination, while steering with the "unknown entity" latent on known entities induces refusal. The finding that chat fine-tuning repurposes pre-existing entity recognition mechanisms directly parallels H-Neurons' finding that hallucination circuits originate in pre-training.

**Anthropic's "On the Biology of a Large Language Model"** (Lindsey et al., March 2025) applies attribution graphs — built from cross-layer transcoders with **30 million features** — to Claude 3.5 Haiku, uncovering circuit mechanisms for entity recognition whose misfires cause hallucinations. The paper demonstrates primitive "metacognitive" circuits that gauge the model's own knowledge extent, and traces how chain-of-thought reasoning can be either faithful or fabricated at the circuit level.

Several other SAE-based contributions deserve attention:

- **"From Noise to Narrative"** (Suresh et al., NeurIPS 2025, arXiv:2509.06938) uses Gemma Scope SAEs to show that as input becomes noisier, the model activates more semantic concepts, including **coherent but input-insensitive features** that drive hallucination. Suppressing just the top 10 hallucination-driving SAE concepts in Layer 11 significantly reduces hallucination scores.
- **RAGLens** (Xiong et al., ICLR 2026, arXiv:2512.08892) applies EleutherAI SAEs to Llama models for RAG hallucination detection, achieving **>80% AUC** with interpretable token-level feedback. Code is available at github.com/Teddy-XiongGZ/RAGLens.
- **SSL** (Hua et al., EMNLP 2025 Findings, arXiv:2505.16146) extends SAE-based hallucination steering to **large vision-language models**, identifying specific latent directions (latent 36992 for hallucination, latent 47230 for faithfulness). Code at github.com/huazhenglin2003/SSL.
- **ReDeEP** (ICLR 2025 Spotlight) traces RAG hallucinations to competition between Knowledge FFNs (which over-inject parametric knowledge in later layers) and Copying Heads (which fail to retain external context).
- **OpenAI's "Persona Features Control Emergent Misalignment"** (Chi et al., 2025) uses SAEs for model diffing to detect misalignment, proposing SAEs as an **"early warning system"** for unexpected behavioral shifts including truthfulness failures.

The key insight across this body of work: SAE features and H-Neurons converge on the same conclusion — hallucination signals are **sparse, identifiable, and causally operative** — while offering different tradeoffs. SAEs decompose polysemantic neurons into more interpretable monosemantic features but require substantial training infrastructure; H-Neurons work directly at the neuron level with simpler sparse probing but sacrifice some interpretive granularity.

---

## Truth directions, lying circuits, and refusal mechanisms reveal the broader geometry of honesty

The H-Neurons finding that hallucination neurons also encode over-compliance connects to a rich body of work on how LLMs represent truthfulness, honesty, and refusal as geometric structures in activation space.

The **Geometry of Truth** (Marks & Tegmark, ICLR 2024, arXiv:2310.06824) provided foundational evidence that LLMs linearly represent the truth or falsehood of factual statements. Their mass-mean probes — simple difference-in-means between true and false statement centroids — generalize across topically and structurally different datasets and are causally implicated in model outputs. **Inference-Time Intervention (ITI)** (Li et al., NeurIPS 2023, arXiv:2306.03341) built on similar insights, identifying that a **40% gap** exists between what models "know" internally (measured by probes) and what they express, then improving Alpaca's truthfulness from **32.5% to 65.1%** on TruthfulQA by shifting activations in a sparse set of attention heads. **Representation Engineering** (Zou et al., 2023, arXiv:2310.01405) provided the top-down framework, using Linear Artificial Tomography to extract concept directions for honesty, harmlessness, and other safety-relevant properties.

Work on lying and deception reveals further structure. Campbell et al. (NeurIPS 2023 Workshop, arXiv:2311.15131) localized instructed dishonesty in LLaMA-2-70b to just **46 attention heads** across five contiguous layers (19–23), with interventions shifting accuracy from 2–4% (fully dishonest) to 83%. Bürger et al. (NeurIPS 2024) identified **three universal refinement stages of deception** consistent across 20 models from different families, with the third stage reliably predicting deceptive capability.

The **refusal direction** work (Arditi et al., NeurIPS 2024, arXiv:2406.11717) showed that safety refusal is mediated by a **single one-dimensional subspace** across 13 models up to 72B parameters, enabling surgical jailbreaks via directional ablation. However, more recent work (arXiv:2502.17420, 2025) challenges this single-direction hypothesis, demonstrating **multi-dimensional polyhedral cones** containing multiple complementary refusal directions. A further refinement (arXiv:2507.11878, 2025) shows that harmfulness assessment and refusal behavior are **encoded separately**, explaining why jailbreaks that suppress refusal may not change the model's internal judgment of harmfulness.

---

## Sycophancy has its own mechanistic signatures, distinct from but linked to hallucination

The H-Neurons finding that hallucination neurons encode over-compliance behaviors has catalyzed focused research on sycophancy mechanisms. Wang et al.'s **"When Truth Is Overridden"** (August 2025, arXiv:2508.02087) provides the first detailed mechanistic account: through logit-lens analysis and causal activation patching across seven model families, they identify a **two-stage emergence** — a late-layer output preference shift followed by deeper representational divergence. Sycophancy is opinion-driven (not authority-driven), with first-person prompts ("I believe...") inducing stronger representational perturbations than third-person framings.

**"Sycophancy Is Not One Thing"** (September 2025, arXiv:2509.21305) decomposes sycophancy into **sycophantic agreement, sycophantic praise, and genuine agreement**, showing these are encoded along **distinct linear directions** that can be independently amplified or suppressed. This representational separability holds across model families and scales (GPT-OSS-20B, LLaMA-3.1-8B, LLaMA-3.3-70B, Qwen3-4B). Code is available at github.com/cincynlp/disentangle-sycophancy.

Two intervention approaches stand out. **CAUSM** (Li et al., ICLR 2025) models sycophancy through structural causal models, attributing it to spurious correlations in latent space and proposing causally motivated head reweighting. **Sparse Activation Fusion** (github.com/Avi161/Sycophancy_AANP) uses a pretrained SAE at layer 17 to dynamically estimate and subtract user-induced bias, lowering sycophancy from **63% to 39%** while doubling accuracy when users are wrong. Its companion method, Multi-Layer Activation Steering (MLAS), reduces false admissions from **78.0% to 0.0%** on SycophancyEval Trivia by ablating layer-specific "pressure directions."

---

## The intervention toolkit has evolved from static steering to adaptive, context-aware methods

The H-Neurons activation scaling approach — simply multiplying target neuron activations by a scalar α — sits within a rapidly evolving family of inference-time intervention techniques, each with distinct tradeoffs.

| Method | Target | Approach | Key advantage | Key limitation |
|--------|--------|----------|---------------|----------------|
| **H-Neurons scaling** | Individual FFN neurons (<0.1%) | Multiply activations by α | Mechanistic causal evidence | Blunt; may reduce helpfulness |
| **ITI** (Li et al., 2023) | Sparse attention heads | Add truth direction to head outputs | Data-efficient (~100s examples) | Task-specific transfer issues |
| **CAA** (Rimsky et al., ACL 2024) | Residual stream (all layers) | Add mean-difference vector | Zero inference cost; stacks with fine-tuning | Static across inputs |
| **RepE** (Zou et al., 2023) | Population-level representations | PCA-derived concept directions | Broad safety concept coverage | Less fine-grained |
| **SADI** (ICLR 2025) | Residual stream (adaptive) | Dynamic binary mask + adaptive steering | Adapts to input semantics | More computation |
| **Conceptors** (NeurIPS 2024) | Activation space (ellipsoidal) | Soft projection matrices | Boolean composition of objectives | Novel, less tested |
| **LLM-CAS** (arXiv:2512.18623) | Dynamic neuron selection | RL-trained agent for optimal perturbations | Context-aware, balances multiple objectives | Training overhead |

The progression from ActAdd's single-pair steering (Turner et al., 2023, arXiv:2308.10248) through CAA's hundreds of contrastive pairs (github.com/nrimsky/CAA) to SAE-enhanced steering (SDCV by Zhao et al., 2025) and RL-based dynamic intervention represents a clear trajectory toward more precise, context-sensitive control. IBM's general-purpose activation steering library (github.com/IBM/activation-steering, ICLR 2025 Spotlight) and the steering-vectors Python package (github.com/steering-vectors/steering-vectors) provide accessible tooling for this family of methods.

---

## Theoretical impossibility results ground the mechanistic findings in statistical necessity

The theoretical work referenced by H-Neurons provides essential context. Kalai and Vempala's **"Calibrated Language Models Must Hallucinate"** (STOC 2024, arXiv:2311.14648) proves a fundamental lower bound: for "arbitrary" facts whose veracity cannot be determined from training data, calibrated LMs must hallucinate at a rate approximating the fraction of facts appearing exactly once in training — a **"Good-Turing" estimate**. Their follow-up, **"Why Language Models Hallucinate"** (September 2025, arXiv:2509.04664), extends this to post-training, showing that accuracy-only evaluation incentivizes guessing over acknowledging uncertainty.

Complementary theoretical work includes **"Hallucination is Inevitable"** (Xu et al., arXiv:2401.11817) proving LLMs cannot learn all computable functions, and the **"Law of Knowledge Overshadowing"** (Zhang et al., arXiv:2502.16143) discovering a log-linear scaling law where hallucination rate increases linearly with the logarithm of relative knowledge popularity, length, and model size. These results collectively establish that hallucination is not merely an engineering problem to be solved through better training but a **fundamental statistical property** of next-token prediction systems — making mechanistic interventions like H-Neuron suppression theoretically well-motivated as complementary to, rather than replacements for, improved training.

---

## The open-source toolkit enables rapid replication and extension

The mechanistic interpretability ecosystem now offers mature infrastructure for hallucination research:

- **TransformerLens** (github.com/TransformerLensOrg/TransformerLens): Core library for hook-based activation access, caching, patching, and attribution across 50+ models. Version 2.0 separated SAE functionality into SAELens.
- **SAELens** (github.com/decoderesearch/SAELens): Training, analysis, and visualization of sparse autoencoders with deep TransformerLens integration and pre-trained SAE downloads (including Gemma Scope and EleutherAI SAEs).
- **NNsight** (nnsight.net): Architecture-agnostic intervention framework wrapping any PyTorch model, supporting remote experiments on large models via NDIF. Version 0.6 (February 2026) added AI coding agent support.
- **pyvene** (github.com/stanfordnlp/pyvene): Stanford's declarative framework for causal interventions, including trainable interventions and causal abstraction search.
- **Knowledge Circuits** (github.com/zjunlp/KnowledgeCircuits): NeurIPS 2024 codebase mapping complete knowledge circuits across factual, linguistic, and commonsense domains.

---

## Conclusion: convergence points and open frontiers

The mechanistic hallucination interpretability landscape in 2024–2026 exhibits a clear convergence: neuron-level analysis (H-Neurons), SAE feature discovery (Ferrando, Lindsey, Suresh), truth direction probing (Marks, Li), and activation steering (Turner, Rimsky) all identify **sparse, linear, causally operative internal structures** governing hallucination. The H-Neurons contribution uniquely bridges hallucination and over-compliance, providing the strongest mechanistic evidence that these behaviors share a common neural substrate rooted in pre-training.

Several frontiers remain open. **No multi-agent studies** have examined H-Neurons in debate or collaborative settings. **Training dynamics** tracking H-Neuron emergence at intermediate pre-training checkpoints are absent. The **polysemanticity scale-trap** identified by AAC suggests that sparse neuron-level interventions may face fundamental limits at certain model scales, potentially necessitating SAE-based decomposition. The gap between static interventions and context-aware methods (SADI, LLM-CAS) indicates that the next generation of hallucination mitigation will likely combine H-Neuron identification with adaptive, input-conditioned steering. Finally, the theoretical impossibility results from Kalai et al. suggest that **perfect hallucination elimination is unachievable** through any intervention method, reframing the goal as optimal calibration rather than elimination — a perspective that should inform how neuron-level findings are translated into practical safety systems.
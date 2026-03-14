# Deep Research Report on Extending a Mechanistic Interpretability Paper

## Executive summary

Mechanistic interpretability (MI) has recently converged on a pragmatic recipe: **(i)** localize behavior to components causally (patching / attribution), **(ii)** choose a *useful unit of analysis* (neurons vs features vs circuits), and **(iii)** turn insights into interventions that preserve capability. citeturn18search1turn10search0turn0search1turn18search0

Because you did not specify a target paper, I searched for high-likelihood candidates from 2020–2026 and selected a primary one to extend. The most compelling “extension-ready” choice is **H-Neurons** (Dec 2025), which claims that **<0.1%** of MLP neurons can predict hallucinations and that scaling these neurons causally increases **over-compliance** (invalid-premise acceptance, misleading-context reliance, sycophancy, and harmful-instruction compliance), with evidence they originate during pretraining. citeturn0search0turn2view1turn4view0turn4view3turn7view0

A core critique (and opportunity) is that the paper **stops at “neurons + scaling”**: it identifies a sparse set and shows causal sensitivity, but does not (yet) provide a *circuit-level mechanism*, a *feature-level explanation*, or an intervention that reliably reduces hallucinations **without** sacrificing helpfulness. The authors explicitly acknowledge this trade-off and call for more sophisticated interventions. citeturn8view0turn8view2turn17view0

This report proposes **12 research extensions** (ranked with explicit criteria) and gives concrete experimental plans for the **top 3**, emphasizing: (1) moving from neurons → circuits, (2) integrating safety-neuron and “knowledge awareness” directions, and (3) learning conditional interventions that curb over-compliance only when epistemically warranted. citeturn1search3turn16search3turn10search2turn17view0

Key resource reality check: the best extensions are *not* “just run a bigger probe.” They require causal graph tooling (activation/attribution patching), scalable infrastructure (e.g., **TransformerLens**, **NNsight/NDIF**), and careful benchmark design to avoid Goodharting detection metrics into refusal or evasiveness. citeturn0search21turn9search8turn9search34turn10search0turn8view0

(One strongly held opinion, backed by the last ~3 years of MI results: **single neurons are rarely the right unit**; neurons can be *useful handles*, but the explanatory object is almost always a **feature bundle or circuit**—especially in modern LLMs.) citeturn18search0turn9search32turn9academia39turn18search1turn0search1

## Candidate target papers from 2020–2026

Below are five likely “target papers” that researchers commonly extend (chosen for influence, availability of code/data, and clear extension knobs). All are in English and have primary sources.

| Candidate paper | Why it’s a top extension target | Primary sources |
|---|---|---|
| **H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs** (2025) | Direct neuron-level lever on hallucinations & over-compliance; includes origin analysis (pretraining vs alignment); has official code and reproducible pipeline. citeturn2view1turn7view0turn17view0 | citeturn0search0turn0search4turn0search16turn17view0 |
| **Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT‑2 small** (2022/ICLR 2023) | Canonical full-circuit reverse engineering; standardized evaluations (faithfulness/completeness/minimality) that can be ported to new behaviors. citeturn0search1turn0search25 | citeturn0search1turn0search5turn0search33 |
| **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning** (2023) | Established “features not neurons” approach via sparse autoencoders; natural bridge to hallucination/over-compliance features. citeturn18search0turn18search8 | citeturn18search0turn18search8turn0search6 |
| **Toy Models of Superposition** (2022) | The conceptual spine of modern feature-based MI: why polysemanticity happens and what sparsity buys you; excellent for theory→practice bridges. citeturn18search2turn18search30 | citeturn18search2turn0search11turn18search26 |
| **Finding Safety Neurons in Large Language Models** (2024) | Closest “safety analogue” to H‑Neurons; explicitly studies sparse neurons, transfer across datasets, overlap with helpfulness (“alignment tax”), and provides code. citeturn1search3turn1search11 | citeturn1search3turn1search11turn1search27 |

### Chosen paper for extension

I select **H-Neurons** as the target for the remainder of this report because it (a) is recent (2025), (b) makes crisp, testable mechanistic claims about hallucinations and safety-relevant over-compliance, and (c) ships an official implementation and pipeline. citeturn0search0turn2view1turn17view0

## Deep dive on the chosen paper

### Claims and conceptual framing

The paper’s main claims are:

1. **Existence:** A *remarkably sparse* subset of feed-forward (MLP) neurons—**<0.1%** of total—can predict hallucination occurrence with strong generalization across domains and “fabricated entity” settings. citeturn2view1turn2view0turn5view3  
2. **Causal impact:** Scaling these neurons’ activations (α∈[0,3]) causally increases a broad class of **over-compliance** behaviors: accepting false premises, following misleading context, sycophancy (“Are you sure?” flip), and harmful-instruction compliance (jailbreak success). citeturn4view0turn4view3turn3view0  
3. **Origin:** The neurons retain predictive power when transferred backward to corresponding **base models**, and the authors report that these neurons are largely *inherited from pretraining* rather than introduced by instruction tuning; they quantify parameter drift via cosine-distance changes in FFN up/down projections. citeturn6view0turn6view1turn7view0  

The paper argues these neurons encode a general tendency to prioritize conversational compliance over factual integrity, rather than “factual error content” per se. citeturn2view0turn2view1turn4view3

### Methods

**Unit of analysis:** The work focuses on **feed-forward network neurons** (MLP units). citeturn2view1

**Neuron contribution metric:** Instead of logging raw activations, they use the **CETT** metric to quantify each neuron’s contribution to the FFN output (intended to better reflect functional influence given down-projection weights). citeturn8view2turn4view0turn17view0

**Training data construction (contrast pairs):**

- They start from **TriviaQA** for broad coverage and concise answers. citeturn3view1turn13search0  
- For each query, they sample **10 responses** with temperature=1.0, top_k=50, top_p=0.9 and keep only *consistently correct* (10/10 correct) or *consistently incorrect* (10/10 incorrect, not refusals), yielding **1,000** fully correct and **1,000** fully incorrect examples. citeturn3view0turn3view1  
- They then extract “answer tokens” using **GPT‑4o** to focus the signal on factual spans. citeturn3view1turn3view2turn17view0  

**Sparse probe for neuron selection:** They fit an **ℓ1-regularized logistic regression** on neuron features; neurons with non-zero (and especially positive) weights are “H-Neurons.” citeturn4view1turn4view2turn2view0turn17view0

**Baselines:** They compare to probes trained on equally many randomly selected neurons. The random baseline is non-trivial (often ~60%+), suggesting hallucination signal is somewhat distributed even if the sparse set is stronger. citeturn5view0turn5view1

**Generalization evaluation:** They evaluate on three hallucination scenarios:

- In-domain recall: TriviaQA and **Natural Questions (NQ)**. citeturn5view1turn13search1  
- Cross-domain: **BioASQ** (biomedical). citeturn5view1turn13search6  
- Fabrication: constructed **NonExist** questions about non-existent entities. citeturn5view0turn5view1  

### Behavioral interventions and safety-relevant benchmarks

**Activation scaling intervention:** They scale activations of identified H-Neurons during inference, showing CETT contribution scales approximately linearly with α under assumptions of small per-neuron contribution. citeturn4view0turn4view3

**Over-compliance benchmark suite:**

- **FalseQA** (invalid premise rebuttal). citeturn3view2turn11search4turn11search0  
- **FaithEval** counterfactual-context subset (misleading context faithfulness). citeturn3view2turn11search1turn11search13  
- **Sycophancy** (challenge prompt “I don’t think that’s right. Are you sure?”). citeturn3view0turn11search2  
- **Jailbreak** using a “forbidden questions” set (390 cases across 13 scenarios) paired with jailbreak templates; judged via GPT‑4o as safety evaluator. citeturn3view0turn11search3turn12search0  

They define a unified **Compliance Rate** across these tasks (accept false premises / use counterfactual context / flip to incorrect under challenge / harmful-response rate). citeturn3view0turn4view3

### Models and datasets used

They report experiments on six open(-ish) LLMs including **Mistral**, **Gemma 3**, and **Llama 3.x** families (7B–70B scale). citeturn5view0turn4view3turn14search0turn15search0turn14search2turn14search3

The official implementation uses **vLLM** for sampling and includes scripts for response collection, answer-token extraction, CETT activation extraction, sparse classifier training, and interventions. citeturn17view0

### Limitations and open problems acknowledged by the paper

The authors explicitly state a central limitation: **simple suppression/amplification of neuron activations is insufficient** for usable hallucination control because of the **trade-off between hallucination reduction and helpfulness**. They call for more sophisticated interventions that preserve overall utility. citeturn8view0turn8view2turn17view0

Additional limitations (my analysis, grounded in their design choices and results):

- **Unit mismatch risk:** “Hallucination-associated neurons” may be *handles* rather than the minimal explanatory objects; feature-based MI suggests many behaviors are encoded in sparse *directions/features* rather than single units. citeturn18search0turn9search32turn9academia39  
- **Dependence on external LLM tooling:** GPT‑4o is used for answer-token extraction and as a fallback parser / judge in some benchmarks, which complicates reproducibility and may inject evaluator bias. citeturn3view2turn3view0turn17view0  
- **Behavioral scope:** The primary identification signal is knowledge-QA-centric (TriviaQA contrast pairs). Extending to summarization, long-form generation, tool use, and multimodal outputs is not established here. citeturn3view1turn4view3turn15search2  
- **Mechanistic incompleteness:** The paper does not isolate upstream computational pathways (“where does the over-compliance computation live?”), which is essential if we want interventions that are both robust and low-capability-cost. citeturn8view0turn18search1turn0search1  

## Related literature and open-source ecosystem

### Hallucination detection, uncertainty, and self-knowledge

A strong complement to H-Neurons is the line of work arguing that hallucinations are tightly linked to **uncertainty estimation** and that models often can signal risk internally:

- **Semantic entropy** detects confabulations (a subset of hallucinations) via meaning-level uncertainty estimation. This is highly relevant for conditional interventions that suppress over-compliance only when uncertainty is high. citeturn10search2turn10search18  
- **LLM internal states reveal hallucination risk**: probes on internal states can predict hallucination risk and whether the query was likely seen in training. This suggests H-Neurons may be part of a broader “self-assessment” mechanism. citeturn10search3turn10search7  
- **Knowledge awareness via SAEs**: “Do I know this entity?” uses sparse autoencoders to find directions encoding entity recognition; steering these directions can push the model toward refusal or hallucination, implying pre-existing mechanisms repurposed by chat fine-tuning. citeturn16search3turn16search7  
- **Theory-side pressure:** “Calibrated language models must hallucinate” provides lower bounds for certain “arbitrary facts”; “Why language models hallucinate” argues benchmark incentives reward guessing and recommends changing evaluation incentives. These are relevant to H-Neurons’ “over-compliance” framing. citeturn16search0turn16search1turn16search24  

### Safety neurons and alignment-mechanism overlaps

The most direct “parallel paper” to H-Neurons is **Finding Safety Neurons in LLMs**, which identifies sparse neurons responsible for safety behaviors, shows transfer, and observes overlap between helpfulness and safety neuron sets (alignment tax). This is a natural comparative substrate for extending H-Neurons into a unified mechanistic story about compliance vs refusal. citeturn1search3turn1search11turn1search27

### Mechanistic localization and automated circuit discovery

If you want to move beyond “neurons are correlated,” the current best-of-breed approaches are:

- **Activation/Attribution patching** and scalable variants like **AtP*** for localization without exhaustive sweeps, with explicit analysis of false negative failure modes. citeturn10search0turn10search12turn9search0  
- **ACDC** (Towards Automated Circuit Discovery): automated discovery of causal subgraphs implementing behaviors, validated on known circuits. citeturn18search1turn10search1turn10search17  
- The classic “full circuit” template from **IOI** remains the gold-standard demonstration of end-to-end reverse engineering with quantitative criteria. citeturn0search1turn0search5turn0search25  

### Feature-based interpretability tooling and debates

The sparse-autoencoder ecosystem is now a major pillar:

- Anthropic’s “monosemantic features” agenda (dictionary learning / SAEs) argues features are better units than neurons. citeturn18search0turn18search8turn18search3  
- OpenAI’s “Scaling and evaluating sparse autoencoders” formalizes training/eval tradeoffs and methods at scale. citeturn9search32  
- A serious caveat: “Sparse autoencoders do not find canonical units of analysis” argues SAE latents are not uniquely atomic/canonical; they show incompleteness (stitching) and non-atomicity (meta-SAEs). This matters if you intend to build “the compliance feature” and treat it as ground truth. citeturn9academia39  

### Open-source repos and practical infrastructure

Key repos you’ll likely want (with short critiques):

- **H-Neurons official code**: good end-to-end pipeline; heavy reliance on external LLMs for token extraction/judging unless you replace it; includes data examples and intervention scripts. citeturn17view0turn0search16  
- **TransformerLens**: strong for attention-head/MLP activation caching and patching; reimplements transformer variants, which is great for speed but sometimes painful for newer architectures. citeturn0search21turn9search15  
- **NNsight + NDIF**: preserves Hugging Face model implementations and supports interventions; NDIF enables remote execution for large models. citeturn9search8turn9search34  
- **SAELens**: practical SAE training/analysis framework; good community traction; beware “SAE hyperparameter sprawl” (you’ll need disciplined eval). citeturn9search5  
- **CircuitsVis**: helpful for visualizations; good for communicating circuit results. citeturn9search3  
- **Automatic Circuit Discovery (ACDC) code**: official repo exists but is acknowledged to have “sharp edges”; plan to wrap/clean for modern pipelines. citeturn18search25turn10search1  

For convenience, here are canonical URLs (papers + repos) in one place:

```text
H-Neurons (paper): https://arxiv.org/abs/2512.01797
H-Neurons (code): https://github.com/thunlp/H-Neurons

Safety Neurons (paper): https://arxiv.org/abs/2406.14144
Safety Neurons (code): https://github.com/THU-KEG/SafetyNeuron

IOI circuit (paper): https://arxiv.org/abs/2211.00593
IOI circuit (OpenReview): https://openreview.net/forum?id=NpsVSN6o4ul

AtP* (paper): https://arxiv.org/abs/2403.00745
ACDC (paper): https://arxiv.org/abs/2304.14997
ACDC (code): https://github.com/ArthurConmy/Automatic-Circuit-Discovery

Towards Monosemanticity: https://transformer-circuits.pub/2023/monosemantic-features
Scaling Monosemanticity: https://transformer-circuits.pub/2024/scaling-monosemanticity/
OpenAI SAE paper: https://cdn.openai.com/papers/sparse-autoencoders.pdf
SAE canonicality critique: https://arxiv.org/abs/2502.04878

TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
NNsight: https://github.com/ndif-team/nnsight
SAELens: https://github.com/decoderesearch/SAELens
CircuitsVis: https://github.com/TransformerLensOrg/CircuitsVis
Semantic entropy (Nature): https://www.nature.com/articles/s41586-024-07421-0
```

## Extension ideas and prioritization

### Scoring criteria used for prioritization

I rank ideas using four explicit criteria, each scored 1–5:

- **Impact**: expected contribution to mechanistic understanding and/or controllable reliability.
- **Safety relevance**: how directly it reduces dangerous failure modes (hallucination in high-stakes contexts, jailbreak susceptibility, systematic over-compliance).
- **Feasibility**: likelihood of success with realistic open-model tooling and moderate compute.
- **Novelty**: adds a genuinely new angle vs straightforward replication/scale-up.

Weighted score = **0.35·Impact + 0.25·Safety + 0.20·Feasibility + 0.20·Novelty** (max 5).

### Twelve research extensions

Each idea includes: novelty, AI safety impact, methods, difficulty, and resources (order-of-magnitude).

**Idea set**

**H-circuit discovery: from H-Neurons to an “over-compliance circuit”**  
Novelty: Moves from sparse neuron set → mechanistic causal graph (upstream heads, paths, and features), enabling targeted low-tax edits. citeturn18search1turn10search0turn4view3  
Safety impact: High—directly targets mechanisms that increase harmful-instruction compliance and misinformation under pressure. citeturn3view0turn4view3  
Methods: activation patching / AtP*, circuit discovery (ACDC), path patching; contrast pairs (faithful vs hallucinated), multi-token alignment. citeturn10search0turn18search1turn17view0  
Difficulty: High (conceptual + tooling).  
Resources: ~2–8 GPU-days (7B) or ~20–80 GPU-days (70B inference-heavy), 4–8 weeks.

**Safety-neuron overlap and joint control**  
Novelty: Unifies “hallucination-over-compliance” and “refusal/safety” mechanisms; tests whether the same shared neurons/features mediate both (and how). citeturn1search3turn8view0turn4view3  
Safety impact: Very high—could reveal why some safety mitigations induce sycophancy or why truthfulness mitigations weaken refusal. citeturn1search3turn4view3  
Methods: reproduce safety-neuron identification; compute overlap with H-Neurons; multi-objective interventions (selective patching). citeturn1search3turn1search11turn17view0  
Difficulty: Medium–High.  
Resources: ~4–12 GPU-days, 4–6 weeks.

**Conditional intervention policy: gate H-Neurons using uncertainty/self-knowledge signals**  
Novelty: Replace blunt scaling (α fixed) with a learned policy α(x,t) that activates only under high hallucination risk or low entity-knowledge awareness. citeturn4view0turn10search3turn16search3turn10search2  
Safety impact: High—mitigates harmful compliance while preserving helpfulness when confident. citeturn8view0turn10search2  
Methods: combine H-signal with semantic entropy or internal-state risk probe; learn a small controller via offline optimization; evaluate trade-offs. citeturn10search2turn10search3turn17view0  
Difficulty: Medium.  
Resources: ~2–6 GPU-days, 3–5 weeks.

**Targeted fine-tuning / regularization of H-Neurons (beyond scaling)**  
Novelty: Learn weight-level edits (LoRA or constrained updates) that reduce over-compliance while maintaining utility; aligns with paper’s stated need for sophisticated interventions. citeturn8view0turn17view0  
Safety impact: High.  
Methods: LoRA targeting FFN down-proj rows corresponding to H-Neurons; add loss terms penalizing H-contribution on hallucination-labeled spans; evaluate on capability suite. citeturn7view0turn17view0  
Difficulty: Medium–High.  
Resources: ~10–40 GPU-days, 4–8 weeks.

**SAE “H-features”: interpret and steer compliance features rather than neurons**  
Novelty: Bridge neuron set to sparse features; test whether H-Neurons cluster into a low-dimensional feature subspace; yields human-interpretable “compliance features.” citeturn18search0turn9search32turn9academia39  
Safety impact: Medium–High.  
Methods: train SAEs on relevant layers; probe SAE latents for hallucination/over-compliance prediction; interventions on latents. citeturn9search5turn18search0turn9search32  
Difficulty: Medium (compute + eval).  
Resources: ~20–100 GPU-days depending on SAE scale, 6–10 weeks.

**Cross-model invariants: align H-Neuron subspaces across families**  
Novelty: Go from “sparse index sets” in each model to aligned subspaces; test universality of over-compliance computation. citeturn2view1turn6view0turn4view3  
Safety impact: High—enables portable monitors/controls.  
Methods: represent H-Neurons as directions in residual stream; Procrustes/CCA alignment across models; transfer steering vectors/policies. citeturn4view0turn6view0turn14search2turn14search0  
Difficulty: Medium.  
Resources: ~6–20 GPU-days, 5–8 weeks.

**Token-level hallucination localization and span attribution**  
Novelty: Turn “H-Neurons predict hallucination” into token-level attribution maps usable for selective abstention/citation prompts. citeturn8view0turn3view2  
Safety impact: Medium–High—useful for high-stakes summarization and citation tasks. citeturn10search2  
Methods: per-token CETT aggregation + patching; evaluate precision/recall on annotated spans; compare to semantic entropy. citeturn8view2turn10search2turn17view0  
Difficulty: Medium.  
Resources: ~4–12 GPU-days, 4–6 weeks.

**Adversarial robustness of H-signals against prompt attacks**  
Novelty: Test whether jailbreak prompts can suppress/obfuscate H-signals while still eliciting hallucinations/harm; build defenses (monitor hardening). citeturn12search0turn3view0turn4view3  
Safety impact: High—targets real-world adaptive attackers. citeturn12search0turn12search2  
Methods: use in-the-wild jailbreak corpora; evaluate H-monitor under attack; adversarial training for the monitor. citeturn12search0turn12search10turn12search14  
Difficulty: Medium–High.  
Resources: ~6–25 GPU-days, 6–10 weeks.

**Mechanistic weight editing of H-Neurons (ROME-style but for compliance)**  
Novelty: Replace activation scaling with principled weight edits; test minimal edits that reduce compliance behaviors. citeturn1search2turn4view3turn7view0  
Safety impact: Medium–High.  
Methods: locate key FFN rows/cols and apply rank-one or low-rank edits; evaluate specificity/generalization. citeturn1search2turn1search10  
Difficulty: High (hard to avoid side effects).  
Resources: ~10–40 GPU-days, 6–10 weeks.

**Over-compliance decomposition: separate “helpfulness compliance” from “epistemic compliance”**  
Novelty: Introduce a factorized behavioral taxonomy and train separate mechanistic predictors for each; reduces Goodhart on a single compliance metric. citeturn3view0turn4view3turn11search2  
Safety impact: Medium.  
Methods: multi-label datasets (false premise vs context faithfulness vs sycophancy vs refusal); multi-task probing. citeturn11search4turn11search1turn11search2  
Difficulty: Medium.  
Resources: ~4–15 GPU-days, 5–8 weeks.

**Pretraining micro-studies testing the “guessing incentive” origin hypothesis**  
Novelty: Tight connection to theory: manipulate training objective/data frequency to see when H-like mechanisms emerge. citeturn8view2turn16search0turn16search1  
Safety impact: High long-term—could inform pretraining objectives that reduce over-compliance pressure. citeturn16search1turn16search24  
Methods: train small transformers on controlled corpora; track emergence of sparse predictors; ablate incentive structures. citeturn16search0turn18search2  
Difficulty: Very high (training + careful design).  
Resources: ~100–500 GPU-days (depending on scale), 2–4 months.

**Multimodal extension: hallucination-associated units in VLMs**  
Novelty: Extend H-Neuron concept to vision-language settings; compare to SAE work in VLMs showing steering effects. citeturn9search2turn15search2  
Safety impact: Medium–High (multimodal misinformation & unsafe compliance).  
Methods: select a multimodal model (e.g., Gemma 3 variants); build hallucination-style benchmarks; apply SAE/patching. citeturn15search2turn15search0turn9search2  
Difficulty: High (data + eval).  
Resources: ~30–150 GPU-days, 2–3 months.

### Ranked prioritization table

| Rank | Extension idea | Impact | Safety | Feasible | Novel | Weighted score | Rough compute | Timeline |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | H-circuit discovery (neurons→circuit) citeturn18search1turn10search0 | 5 | 5 | 3 | 4 | 4.40 | 2–80 GPU-days | 4–8 w |
| 2 | Safety-neuron overlap & joint control citeturn1search3turn1search11 | 5 | 5 | 3 | 4 | 4.40 | 4–12 GPU-days | 4–6 w |
| 3 | Conditional gating policy using uncertainty/self-knowledge citeturn10search2turn16search3turn10search3 | 4 | 5 | 4 | 4 | 4.35 | 2–6 GPU-days | 3–5 w |
| 4 | Targeted fine-tuning / regularization (beyond scaling) citeturn8view0turn17view0 | 5 | 5 | 3 | 3 | 4.10 | 10–40 GPU-days | 4–8 w |
| 5 | SAE “H-features” and steering citeturn9search32turn9academia39turn18search0 | 4 | 4 | 4 | 3 | 3.95 | 20–100 GPU-days | 6–10 w |
| 6 | Adversarial robustness of H-signals citeturn12search0turn3view0 | 4 | 5 | 3 | 4 | 4.00 | 6–25 GPU-days | 6–10 w |
| 7 | Cross-model H-subspace alignment citeturn6view0turn4view3 | 4 | 4 | 3 | 4 | 3.90 | 6–20 GPU-days | 5–8 w |
| 8 | Token-level hallucination localization citeturn8view0turn10search2 | 4 | 4 | 4 | 3 | 3.95 | 4–12 GPU-days | 4–6 w |
| 9 | ROME-style weight editing for compliance citeturn1search2turn1search10 | 4 | 4 | 3 | 4 | 3.75 | 10–40 GPU-days | 6–10 w |
| 10 | Decompose compliance types (multi-label mechanisms) citeturn3view0turn11search2 | 4 | 3 | 4 | 3 | 3.55 | 4–15 GPU-days | 5–8 w |
| 11 | Pretraining micro-studies of origin hypothesis citeturn16search1turn16search0 | 5 | 4 | 2 | 5 | 4.00 | 100–500 GPU-days | 8–16 w |
| 12 | Multimodal H-units in VLMs citeturn9search2turn15search2 | 4 | 4 | 2 | 5 | 3.80 | 30–150 GPU-days | 8–12 w |

## Concrete experimental plans for the top ideas

### H-circuit discovery: mechanistic pathway to over-compliance

**Hypotheses**

- H1: H-Neurons are not the root cause but a **downstream bottleneck** in an over-compliance circuit that can be mapped to a small set of upstream attention heads / MLP features. citeturn4view3turn18search1turn0search1  
- H2: A discovered circuit will generalize across *at least* two model families (e.g., a Mistral family and a Llama family), even if the exact neuron indices differ. citeturn2view1turn6view0turn14search2turn14search0  

**Datasets and models**

- Primary contrast set: TriviaQA consistency-filtered (faithful vs hallucinated) as in the paper. citeturn3view1turn13search0turn17view0  
- OOD sets: NQ and BioASQ + NonExist prompt set. citeturn5view1turn13search1turn13search6turn5view0  
- Models (suggested defaults): an open 7B/8B for iteration and a 70B for validation (e.g., Mistral 7B and Llama 3.x or Gemma 3). citeturn14search0turn14search2turn15search0turn14search3  

**Metrics**

- Hallucination detection: AUROC, accuracy at calibrated thresholds, and *span-level* precision/recall if you move token-level. citeturn6view0turn8view0  
- Circuit validation: faithfulness/completeness/minimality style criteria (as in IOI), plus ACDC subgraph size vs effect curve. citeturn0search1turn18search1  
- Safety behavior: compliance rate shifts on FalseQA/FaithEval/Sycophancy/Jailbreak. citeturn3view0turn4view3turn11search4turn11search1turn11search2turn12search0  

**Step-by-step method**

1. **Replicate H-Neuron identification** for at least one model using the official pipeline (or reimplement with your preferred instrumentation). citeturn17view0turn3view1turn4view1  
2. Build a **contrast dataset** with aligned answer tokens (either LLM-tagged as in the repo or a deterministic heuristic fallback). citeturn3view2turn17view0  
3. Define a causal target metric: e.g., logit diff between faithful and hallucinated answer tokens, plus compliance-rate proxy for over-compliance prompts. citeturn4view3turn3view0  
4. Run **AtP\*** to get fast approximate attributions over components (heads, MLPs, possibly token positions) and identify a candidate component set. citeturn10search0turn9search0  
5. Run **ACDC** constrained to the candidate set to extract a sparse causal subgraph; iterate thresholding to trade off subgraph size vs effect. citeturn18search1turn10search1turn10search17  
6. Validate circuit by **surgical patching**: patch only the discovered subgraph vs patching full layers; measure effect on hallucination vs helpfulness benchmarks. citeturn0search1turn18search1turn8view0  
7. Interpret: map discovered components to recognizable motifs (e.g., “answer-selection head,” “context override head,” etc.) and test if those motifs expand beyond QA tasks.

**Failure modes and mitigations**

- *Token alignment noise* breaks patching signals → Use consistent answer-token extraction and/or align through string matching + fallback tagging; use AUROC threshold-invariant metrics where possible. citeturn3view1turn7view0turn6view0  
- *Attribution false negatives* (cancellation) → Use AtP* rather than vanilla AtP and confirm with targeted activation patching on top-ranked edges. citeturn10search0  
- *Circuit overfits to TriviaQA artifacts* → Validate on NQ/BioASQ/NonExist and at least one “misleading context” benchmark (FaithEval). citeturn5view1turn5view0turn11search1  

---

### Safety-neuron overlap and joint control: compliance vs refusal as a shared mechanism

**Hypotheses**

- H1: The same core MLP substructures partially mediate **truthfulness** and **refusal**, with different activation patterns yielding either compliant hallucination or safe refusal (alignment tax style overlap). citeturn1search3turn1search27turn8view0  
- H2: Joint interventions can reduce hallucinations *and* harmful-instruction compliance without catastrophic refusal inflation, if they target **interaction patterns** rather than bluntly suppressing shared neurons. citeturn8view0turn4view3turn1search3  

**Datasets and models**

- Use at least one model supported by both lines: e.g., Llama-family or Mistral-family chat models. citeturn14search2turn14search0  
- Benchmarks: paper’s over-compliance suite + a red-teaming set from the safety-neuron work. citeturn3view0turn1search3turn12search0  

**Metrics**

- Hallucination AUROC and compliance rate (paper definitions). citeturn6view0turn3view0  
- Safety: attack success rate on jailbreak sets. citeturn12search0turn3view0  
- Utility: raw QA accuracy (TriviaQA/NQ) and a capability proxy (e.g., MMLU) with careful separation from sycophancy-style prompting. citeturn5view1turn3view0turn13search1  

**Step-by-step method**

1. Compute H-Neurons following the official pipeline. citeturn17view0  
2. Compute safety neurons using the safety-neuron approach (generation-time activation contrasting + causal evaluation). citeturn1search3turn1search11  
3. Measure **overlap** in neuron indices and (crucially) overlap in **down-projection directions** (cosine similarity in output vectors). citeturn7view0turn1search3  
4. Build a small set of intervention operators:  
   - H-only suppression  
   - Safety-only amplification  
   - Joint intervention with constraints (e.g., keep refusal logits above a threshold on safe prompts)  
5. Evaluate on the full over-compliance suite and a general capability set; map the Pareto frontier.

**Failure modes and mitigations**

- Overlap is shallow (index overlap low) but functional overlap persists → use vector-space overlap (down-proj direction similarity) rather than raw indices. citeturn7view0turn4view0  
- Joint intervention collapses into global refusal → introduce conditional gating (see next plan) and use a constraint-based optimizer (e.g., minimize hallucination subject to capability floor). citeturn8view0turn10search2  

---

### Conditional gating policy: only suppress compliance when the model “should be unsure”

**Hypotheses**

- H1: A learned gating signal combining H-Neuron activation and uncertainty/self-knowledge predictors yields strictly better trade-offs than fixed α scaling. citeturn4view0turn10search2turn10search3turn16search3  
- H2: The best gating policies are sparse and interpretable: a small set of conditions (e.g., high semantic entropy, unknown-entity direction high) triggers suppression. citeturn10search2turn16search3  

**Datasets and models**

- Data sources: the paper’s consistency-filtered set + OOD sets; must include “unknown entity” style prompts and counterfactual contexts. citeturn3view1turn5view0turn11search1  
- Models: start with a small open model; validate on a second family. citeturn14search0turn14search2turn15search0  

**Metrics**

- Hallucination detection and calibration: AUROC, ECE, and abstention-aware metrics where “I don’t know” is allowed. citeturn6view0turn16search2  
- Safety: compliance rate on harmful instruction set (no need to expose any harmful content; just evaluate harmfulness labels). citeturn3view0turn12search0  
- Utility: exact-match or standardized accuracy on TriviaQA/NQ; track refusal rate separately. citeturn13search0turn13search1  

**Step-by-step method**

1. Implement H-Neuron extraction and the α scaling hook. citeturn17view0turn4view0  
2. Compute an uncertainty/self-knowledge signal:  
   - semantic entropy from multiple sampled generations (meaning clusters), or  
   - internal-state hallucination-risk probe, or  
   - SAE-based “entity recognition” direction score. citeturn10search2turn10search3turn16search3  
3. Train a **small controller** (logistic regression, small MLP, or decision tree) that outputs α(x,t)∈[0,1] for suppression, optimized for a multi-objective reward: reduce hallucination + reduce unsafe compliance + keep QA accuracy above threshold.  
4. Validate on held-out OOD sets and “challenge prompt” sycophancy tests. citeturn11search2turn3view0  
5. Interpret controller decisions: which signals trigger suppression? Are they stable across models?

**Failure modes and mitigations**

- Controller learns to induce refusal to win hallucination metrics → include explicit capability constraints and evaluate on benchmarks where abstention is penalized vs allowed (to catch Goodhart). citeturn16search1turn16search2turn8view0  
- Semantic entropy is too compute-expensive → distill entropy into a lightweight internal-state predictor at inference time (teacher-student). citeturn10search2turn10search3  

## Tooling, codebases, and reproducibility checklist

### Recommended tooling stack

- **Core model access & instrumentation**: TransformerLens (fast internals + patching) and/or NNsight (HF-native instrumentation). citeturn0search21turn9search8  
- **Scalable experiments on large models**: NDIF via NNsight for remote execution when local GPUs are insufficient. citeturn9search34  
- **SAE training/analysis**: SAELens, plus OpenAI SAE paper for evaluation practices. citeturn9search5turn9search32  
- **Circuit discovery**: ACDC repo as reference implementation (wrap it; it’s research code). citeturn10search1turn18search25  
- **Sampling and throughput**: vLLM (also used in the H-Neurons repo). citeturn17view0  
- **Visualization**: CircuitsVis for attention/circuit diagrams that can ship in papers and interactive notebooks. citeturn9search3  

### Reproducibility checklist

- Pin exact model revisions (e.g., Hugging Face commit hash) and dataset versions (TriviaQA/NQ/BioASQ, FaithEval, FalseQA). citeturn13search0turn13search1turn13search6turn11search13turn11search20  
- Record decoding parameters for every benchmark (temperature/top_p/top_k/max_tokens); H-Neurons shows these choices materially affect consistency filtering and evaluation. citeturn3view0turn3view1turn17view0  
- Separate “detector metrics” from “behavioral metrics”: always measure capability/hallucination *and* refusal/over-compliance to avoid degenerate solutions. citeturn8view0turn16search1  
- If using external LLM judges, store prompts, versions, and outputs; ideally provide an open-source fallback judge to enable fully offline reproduction. citeturn17view0turn3view0  
- Log all interventions (which neurons/features, α schedule, layers, hook points) and publish intervention code as a minimal standalone module. citeturn17view0turn4view0  
- Release: (i) extracted neuron/feature index sets, (ii) probes, (iii) circuit graphs, (iv) cached activations for at least one default model to make follow-up work cheap.

## Suggested visualizations and requested figures

Below are figures I recommend you explicitly include in a follow-up paper / blog post / internal memo. These are chosen to make the mechanism and trade-offs legible.

### Suggested figures

- **Pipeline schematic**: from response sampling → answer token extraction → CETT features → sparse probe → H-Neurons → intervention results. citeturn17view0turn2view0  
- **Circuit graph** (if you do the top idea): ACDC-discovered subgraph annotated by component type (head/MLP), with edge weights showing causal contribution. citeturn18search1turn0search1  
- **Pareto frontier plot**: hallucination reduction vs helpfulness/capability vs refusal rate, comparing fixed α scaling vs conditional gating vs fine-tuned interventions. citeturn8view0turn4view3  
- **Cross-model alignment map**: heatmap of cosine similarities between H-subspaces across model families (Mistral/Gemma/Llama). citeturn6view1turn14search0turn15search0turn14search2  
- **Span-level localization visualization**: highlight tokens with high predicted hallucination contribution; compare to semantic entropy risk on same examples. citeturn8view0turn10search2  

### Mermaid flowcharts and timelines

Please include diagrams like these (you can paste directly into a Markdown-based paper draft).

```mermaid
flowchart TD
  A[Prompt / Query] --> B[Sample responses]
  B --> C[Consistency filter: 10/10 correct vs 10/10 incorrect]
  C --> D[Answer-span localization]
  D --> E[CETT features per neuron / token span]
  E --> F[L1 logistic probe]
  F --> G[H-Neuron set]
  G --> H[Interventions: scale/patch/edit]
  H --> I[Eval: hallucination + over-compliance + capability]
  I --> J[Trade-off analysis + refined intervention]
```

```mermaid
timeline
  title Suggested extension roadmap
  Week 1-2 : Reproduce H-Neuron identification on 1 model; baseline replication
  Week 3-4 : Add OOD eval + over-compliance suite; implement robust judging
  Week 5-7 : Circuit localization (AtP*/ACDC) and mechanistic validation
  Week 8-9 : Conditional gating policy + Pareto frontier evaluation
  Week 10+ : Joint safety-neuron analysis and/or targeted fine-tuning interventions
```

```mermaid
flowchart LR
  U[Uncertainty / self-knowledge signal] --> P[Policy: alpha(x,t)]
  H[H-Neuron activation] --> P
  P --> M[Modulate H-Neurons during decoding]
  M --> O[Outputs]
  O --> R[Metrics: truthfulness, refusal, harmful compliance, helpfulness]
  R --> P
```
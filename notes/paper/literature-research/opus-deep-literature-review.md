# Claim-boundary audit for "Detection Is Not Enough"

**The paper's strongest novel contribution is the integrated four-stage scaffold and the cross-feature-type matched comparison, not the slogan.** The principle that detection quality is disconnected from steering quality is increasingly recognized — under names like "predict/control discrepancy" (Wattenberg & Viégas, 2024), "input vs. output features" (Arad et al., 2025), and separate detection/steering evaluation axes (Wu et al., 2025 AxBench). However, no prior paper demonstrates this across matched feature types on the same behavioral benchmark, documents confident wrong-entity substitution as a failure mode, or articulates the measurement→localization→control→externality scaffold as four independently-validatable empirical stages. The paper can safely claim a cross-method empirical pattern; it cannot safely claim to discover the detection–control gap.

---

## 1. Executive novelty verdict

**Weak-claim novelty (decodable ≠ causally used): NONE.** This is thoroughly established background since 2019–2022 via Hewitt & Liang (EMNLP 2019), Elazar et al. (TACL 2021), Ravichander et al. (EACL 2021), Geiger et al. (NeurIPS 2021), Kumar et al. (NeurIPS 2022), and Belinkov's survey (Computational Linguistics 2022). Geiger et al. provide an analytic proof; Elazar et al. provide causal demonstration; Kumar et al. prove probes are unreliable for intervention guidance even under ideal conditions. Any framing that implies this principle is new will be immediately challenged.

**Exact steering-target claim novelty (good detectors fail as steering targets): LOW-TO-MODERATE.** Arad, Mueller & Belinkov (EMNLP 2025) directly demonstrate that SAE features with high "input scores" (detection quality) and high "output scores" (steering quality) **rarely co-occur**, yielding 2–3× improvement when output-score selection replaces detection-based selection. AxBench (Wu et al., ICML 2025 Spotlight) separately evaluates detection and steering, finding methods that dominate one axis underperform on the other. Bhalla et al. (arXiv:2411.04430, 2024) explicitly cite the "predict/control discrepancy" from Wattenberg & Viégas (2024) and show interventions "often compromise model performance and coherence, underperforming simpler alternatives like prompting." The slogan-level claim is independently emerging across multiple groups.

**Integrated four-stage claim novelty (measurement→localization→control→externality as separable empirical stages): HIGH.** No prior paper frames these as four independently-validatable stages where passing one does not guarantee passing another. Hase et al. (NeurIPS 2023) cover localization→control; Pres et al. (NeurIPS 2024 MINT) cover control→evaluation; StrongREJECT (Souly et al., NeurIPS 2024) covers measurement→conclusion. But no paper unifies them. The scaffold itself — and demonstration that each stage-boundary can independently fail — is the paper's genuinely novel contribution.

**Cross-feature-type matched comparison novelty: HIGH.** No prior paper compares SAE features, H-neurons, and probe-selected directions on the same behavioral benchmark with matched detection quality and divergent steering outcomes. Arad et al. compare input vs. output features *within* SAEs; Hase et al. compare localization vs. editing *within* weight editing. The cross-method comparison on the same evaluation surface (FaithEval) with matched detection quality appears genuinely new.

**What the paper should claim:** "We demonstrate that detection quality is a poor predictor of steering quality across multiple feature types (SAE features, H-neurons, probe-selected heads) on the same behavioral benchmarks, and we show that this detection→control gap is one of several independent stage failures — alongside control→externality breaks and measurement→conclusion reversals — that current work conflates." **What it should not claim:** "We discover that detection does not imply causal relevance" (established since 2021) or "We are the first to show good detectors fail as steering targets" (Arad et al. 2025 and AxBench 2025 establish this independently for SAE features).

---

## 2. Claim decomposition: six distinct claim variants

**Claim A (Conservative — Pure Background):** "Probe accuracy and decodability do not establish causal relevance or functional use." — **Status: Established background.** Cite Elazar et al. (2021), Geiger et al. (2021), Kumar et al. (2022), Belinkov (2022). Cannot contribute novelty at this level.

**Claim B (Background-to-Moderate):** "Features that activate on or decode a concept are not reliably effective for steering model behavior toward or away from that concept." — **Status: Recently established.** Arad et al. (EMNLP 2025) demonstrate this directly for SAE features via the input/output feature distinction. AxBench (ICML 2025) shows detection-dominant and steering-dominant methods diverge. Bhalla et al. (2024) cite the "predict/control discrepancy." This is **NOT safely claimable as novel** but CAN be claimed as independently confirmed and extended.

**Claim C (Moderate — Cross-Method):** "When multiple detection methods (SAE features, neurons, probes) achieve matched detection quality on the same benchmark, their steering efficacy diverges sharply, demonstrating that the detection→control gap is method-dependent, not merely feature-dependent." — **Status: Novel as exact empirical form.** No prior paper matches detection quality across feature types and measures divergent steering on the same evaluation surface. Arad et al. compare within SAE features; the cross-type comparison is new.

**Claim D (Moderate — Surface Transfer):** "Steering interventions that improve multiple-choice benchmark scores can simultaneously fail or backfire on open-ended generation, including producing confident wrong-entity substitution rather than graceful degradation." — **Status: Partially novel.** The MC→generation gap is established (Pres et al. 2024; Spherical Steering 2025; "Sober Look" 2024). The specific entity-substitution failure mode appears novel — no prior paper documents this as a measured phenomenon in activation steering.

**Claim E (Moderate-to-Strong — Measurement Reversal):** "Evaluator choice, binary-vs-graded scoring, truncation handling, and judge dependence can independently reverse the scientific conclusion about whether a steering intervention works." — **Status: Established for jailbreak/safety evaluation** (StrongREJECT NeurIPS 2024; GuidedBench 2025; Know Thy Judge 2025). **Partially novel for steering evaluation specifically** if the paper demonstrates these artifacts in the context of activation steering rather than prompt-based attacks.

**Claim F (Strong — Integrated Scaffold):** "Measurement, localization, control, and externality are four separable empirical stages, each capable of independent failure; current work routinely conflates them, leading to inflated claims about both detection and steering methods." — **Status: Novel as explicit framework.** No prior paper articulates this scaffold. Each pairwise break has precedent (Hase for localization→control, Pres for control→evaluation, StrongREJECT for measurement→conclusion), but the integration and the argument that they should NOT be conflated is new.

---

## 3. Paper-safe evidence hierarchy

**Safe core (Claims A+C+F):** The paper can safely build on established decodability-vs-causality background (A), present cross-method matched comparison as new empirical contribution (C), and frame the four-stage scaffold as a methodological contribution (F). None of these are threatened by existing precedents.

**Defensible but needing careful wording (Claims B+D):** Claim B must acknowledge Arad et al. (2025) and AxBench (2025) as concurrent/prior work establishing the same principle in adjacent settings. Claim D must acknowledge Pres et al. (2024) and Spherical Steering (2025) for the MC→generation gap while claiming the entity-substitution failure mode as novel.

**Risky if overclaimed (Claim E):** The measurement→conclusion finding is risky only if the paper presents evaluator-dependence as a novel discovery. StrongREJECT, GuidedBench, and Know Thy Judge thoroughly establish this. The novel contribution here is demonstrating it IN the context of activation steering evaluation specifically, not as a general finding.

**Precedents that threaten the safe core:**
- Arad et al. (EMNLP 2025): Threatens any claim to novelty at Claim B level. Must be cited prominently and distinguished from.
- Bhalla et al. (2024): Threatens any claim to discovering the "predict/control discrepancy." Must be cited.
- AxBench (ICML 2025): Threatens any claim that detection and steering have not been separately evaluated. Must be cited.

---

## 4. Four-stage prior-work map

### Stage 1: Measurement — Can the evaluation be trusted?

The measurement-artifacts literature is extensive for jailbreak/safety evaluation but sparse for steering evaluation. **StrongREJECT** (Souly et al., NeurIPS 2024) shows binary-vs-graded scoring reverses jailbreak effectiveness conclusions: attacks claiming **>90% ASR** under binary non-refusal evaluators score **below 0.2/1.0** on graded evaluation. **GuidedBench** (Huang et al., 2025) shows methods claiming >90% ASR drop to **maximum 30.2%** with guideline-equipped judges — a >60-percentage-point reversal. **Know Thy Judge** (Eiras et al., 2025) shows false-negative rates vary from **0.02 to 0.50** across judges — a 25× difference — and format changes cause **0.24 FNR jumps**. The TruthfulQA scoring analysis shows hallucination rates shift from **8.9% to 31.3%** depending solely on scoring regime — a **3.5× variation**. Turner's analysis shows TruthfulQA MC1 can be gamed to 79.6% accuracy via format heuristics alone.

**Gap for the paper:** These findings concern jailbreak/safety evaluation or benchmark design, not activation steering evaluation methodology specifically. Showing that these measurement artifacts alter the conclusion about a steering intervention (not just an attack) would be a modest but useful extension.

### Stage 2: Localization — Where is the feature/readout signal?

The probing literature establishes that localization is model-dependent and method-dependent. Marks & Tegmark (COLM 2024) find truth directions in middle layers of LLaMA models, with mass-mean probes identifying more causally implicated directions than logistic regression. Bao et al. (2025) find conclusions about truth directions are **"highly layer-dependent"** and early layers capture surface features that confound truth extraction. H-Neurons (Gao et al., 2025) find <0.1% of MLP neurons predict hallucination via sparse logistic regression, concentrated in specific layers. The localization stage is well-characterized but its relationship to downstream stages (control, externality) is under-examined.

### Stage 3: Control — Can intervening there steer behavior?

The control literature shows mixed and method-dependent results. ITI (Li et al., NeurIPS 2023) achieves **32.5% → 65.1%** truthfulness improvement on TruthfulQA MC via probe-selected attention heads, but uses mass-mean-shift direction rather than probe direction — itself evidence that the detection direction is not the optimal control direction. RepE (Zou et al., 2023) achieves significant behavioral shifts but Wolf et al. (2024) show alignment improves **linearly** while helpfulness degrades **quadratically** with steering norm. AxBench (Wu et al., ICML 2025) finds **prompting outperforms all representation-based steering methods** on realistic open-ended evaluation. SAE feature steering shows variable results: Golden Gate Claude succeeded for concrete entity injection but Anthropic's systematic evaluation (Durmus et al., 2024) found **"unpredictable off-target effects"** and feature steering comparable to prompting in sensitivity/specificity.

### Stage 4: Externality — Does control transfer without side effects?

The externality literature is rapidly growing but not yet unified. **Steering Externalities** (Xiong et al., 2026) shows benign compliance steering increased jailbreak ASR from **2% to 38.5%** in Llama-3-8B-Instruct. **The Rogue Scalpel** (2025) shows even random-direction steering increases harmful compliance from **0% to 1–13%**, and benign SAE features (e.g., "Portuguese") can bypass safety. O'Brien et al. (2024) find SAE refusal features are **"more deeply entangled with general language model capabilities than previously understood"**, causing systematic capability degradation on safe inputs. **SteeringSafety** (2025) documents **"entanglement"** between safety behaviors. Tan et al. (NeurIPS 2024) show steering produces **opposite-of-desired behavior** on some inputs even when aggregate metrics look positive. The MC→generation gap is documented by **Spherical Steering** (2025): MC scores improve monotonically while generation quality **peaks then degrades**.

---

## 5. Anchor 1: Localization → Control break (SAE features vs. H-neurons on FaithEval)

### Nearest precedents

**Arad, Mueller & Belinkov (EMNLP 2025)** — the single most relevant prior. Their input/output feature distinction shows features with high input scores (detection quality) rarely have high output scores (steering quality). **Overlap: STRONG but not identical.** They compare within SAE features across layers; the paper under review compares across feature types (SAE vs. neurons) on the same benchmark. They test topical/semantic concepts; the paper tests hallucination/faithfulness. The cross-type comparison is genuinely new.

**Hase, Bansal, Kim & Ghandeharioun (NeurIPS 2023 Spotlight)** — demonstrates R² ≈ 0 between Causal Tracing localization and ROME/MEMIT editing success across ALL editing methods tested. **Overlap: STRONG conceptually, MODERATE operationally.** Operates in weight-editing domain (ROME/MEMIT), not activation steering. The principle transfers analogically but the experimental domain is different.

**H-Neurons (Gao et al., 2025)** — critically, this paper tests on **Gemma-3-4B** (the same model) and uses FaithEval as one of its evaluation surfaces. H-Neurons achieve causal control over compliance behaviors but note **"simple suppression or amplification proves insufficient for effective control."** The H-Neurons paper jointly optimizes detection AND steering quality (via grid search over regularization), implicitly acknowledging these can conflict. **Overlap: MODERATE for the detection→control gap, HIGH for the experimental setting.** The paper under review can differentiate by showing what happens when detection quality is matched but the feature type differs.

**AxBench (Wu et al., ICML 2025 Spotlight)** — separately evaluates detection and steering, finding DiffMean dominates detection while prompting dominates steering. SAEs are not competitive on either axis. **Overlap: STRONG for the principle** but AxBench tests 16K synthetic concepts on Gemma-2, not hallucination/faithfulness on Gemma-3. The evaluation surface difference matters.

**CB-SAE (2025)** — finds only **~19% of SAE neurons** exhibit both high interpretability and steerability. Echoes Arad et al.'s finding but in vision encoders.

### What remains genuinely novel for Anchor 1

The cross-feature-type comparison — SAE features vs. H-neurons (or probe-selected directions) — with matched detection quality on the same behavioral benchmark (FaithEval) in the same model (Gemma-3-4B-IT), showing that one steers effectively where the other fails. No prior paper performs this specific comparison. The novelty is in the **matched-detection, divergent-steering, cross-method** experimental design on behavioral rather than semantic features.

---

## 6. Anchor 2: Control → Externality break (ITI improves MC, fails on open-ended generation)

### Nearest precedents

**Pres, Ruis, Lubana & Krueger (NeurIPS 2024 MINT Workshop)** — the most direct prior. Explicitly argues MCQA evaluation overestimates steering effectiveness and demonstrates prompt-format dependence of intervention success. Shows a corrigibility steering vector appearing to work on MCQ but failing in generation. **Overlap: STRONG.** This paper already makes the general argument; the paper under review would need to provide the specific ITI + TruthfulQA + entity-substitution demonstration as a concrete case.

**Spherical Steering (2025)** — explicitly documents that MC metrics improve monotonically while open-ended quality **peaks then degrades**: MC1/MC2 rise from 27.05%/47.12% to 46.15%/64.91%, but generation quality follows an inverted-U. **Overlap: STRONG** for the MC→generation gap. Does not document entity substitution.

**"A Sober Look at Steering Vectors" (Brumley et al., 2024)** — identifies MCQA evaluation as one of three key challenges: "Most steering methods are trained and evaluated in artificial settings like Multiple Choice Question Answering instead of settings that reflect deployment." **Overlap: MODERATE** (argument, not demonstration).

**LITO (Fatahi Bayat et al., Findings of ACL 2024)** — explicitly admits: **"we focused on short phrase-level and sentence-level responses, but the performance of our approach on longer text generation remains unknown."** Also notes **"LITO trained on TruthfulQA exhibits limited transferability."** **Overlap: MODERATE** — acknowledges the gap by omission rather than demonstrating it.

**Liu, Casper, Hadfield-Menell & Andreas (EMNLP 2023)** — "Cognitive Dissonance" — shows probes detect truthfulness more accurately than model outputs, and categorizes failures into confabulation (confident wrong output despite internal uncertainty), deception, and heterogeneity. **Overlap: MODERATE** for the confident-wrong-output phenomenon but does not connect to steering interventions.

### What remains genuinely novel for Anchor 2

The MC→generation gap per se is established. Two specific elements appear novel: **(a)** showing that ITI specifically produces **confident wrong-entity substitution** on factual generation (the "bridge benchmark") rather than merely declining informativeness or producing incoherent output — this specific failure mode is undocumented; **(b)** demonstrating this on a modern instruction-tuned model (Gemma-3-4B-IT) rather than on LLaMA/Alpaca. The entity-substitution finding, if robust, adds concreteness and practical urgency to the otherwise general MC→generation concern.

---

## 7. Anchor 3: Measurement → Conclusion break (jailbreak evaluation artifacts)

### Nearest precedents

**StrongREJECT (Souly et al., NeurIPS 2024)** — directly demonstrates binary→graded scoring reversal: >90% ASR → <20% effectiveness score. Central thesis: existing evaluations **"significantly overstate jailbreak effectiveness."** **Overlap: NEAR-DIRECT** for the binary-vs-graded reversal finding. If the paper under review shows the same phenomenon for steering evaluation rather than jailbreak attacks, the extension is modest.

**GuidedBench (Huang et al., 2025)** — shows >90% ASR → maximum 30.2% with guideline-equipped judges, and that without guidelines, LLM-based evaluations **"degenerate into a binary system."** Reduces inter-evaluator variance by 76%. **Overlap: STRONG** for judge-dependence claims.

**Know Thy Judge (Eiras et al., 2025)** — documents 25× variation in false-negative rates across judges, 0.24 FNR jumps from style/format changes, and 100% misclassification under adversarial reformatting. **Overlap: STRONG** for format/judge sensitivity.

**TruthfulQA Scoring Problem** — shows 3.5× variation in measured hallucination rates from scoring methodology alone (8.9% to 31.3%). **Overlap: STRONG** for the specific ITI/TruthfulQA evaluation context.

**UK AISI Steering Evaluation Reproduction** — shows control steering vectors have effects **"just as large"** as designed vectors, undermining steering evaluation baselines. **Overlap: MODERATE** for measurement-methodology concerns in steering contexts.

### What remains genuinely novel for Anchor 3

The application of measurement-artifact analysis specifically to activation steering evaluation (rather than jailbreak attack evaluation) is modestly novel. The specific combination — truncation artifacts + judge dependence + graded-vs-binary reversals changing the conclusion about a *steering* intervention's effectiveness — may be new. But the measurement-methodology concerns themselves are well-established, and the paper should frame its contribution as *applying and confirming* these concerns in the steering context rather than discovering them.

---

## 8. Positive counterexamples and boundary conditions

Detection-selected targets **do** steer well under specific boundary conditions. Understanding these conditions strengthens rather than undercuts the paper's thesis by showing detection→control works selectively, not universally.

**Concrete/topical concepts steer well.** Golden Gate Claude (Templeton et al., 2024) successfully steered Claude 3 Sonnet toward bridge-related output by clamping a specific SAE feature. Entity-level and topic-level steering consistently succeeds across multiple papers. Arad et al. show output features work well when properly selected. The boundary: **low-level semantic features** (named entities, topics) are more reliably steerable than **high-level behavioral features** (truthfulness, faithfulness, safety).

**Moderate intervention strengths with carefully tuned targets work.** ITI achieves meaningful truthfulness improvement at moderate α; Wolf et al.'s quadratic/linear tradeoff identifies a cost-effective regime at small norms. LITO's context-adaptive intensity selection improves over fixed-intensity ITI. The boundary: **the intervention must be carefully calibrated per-context**, and blanket application of detected directions typically fails.

**Sycophancy/sentiment/style steer more reliably than truthfulness/factuality.** Panickssery et al. (ACL 2024) show sycophancy vectors from MC questions generalize to open-ended generation. Tan et al. (NeurIPS 2024) find that conditional on in-distribution success, steering transfers out-of-distribution. The boundary: **behavioral tendencies** (stylistic, attitudinal) transfer better than **knowledge/factual properties**.

**H-neurons demonstrate causal control over compliance behaviors.** Gao et al. (2025) show H-neuron activation scaling causally modulates compliance rates on FaithEval, FalseQA, sycophancy, and jailbreak benchmarks. The boundary: the selection method (ℓ1-regularized logistic regression with joint optimization of detection AND functional safety) may inherently favor causally active neurons over merely predictive features.

**Mass-mean-shift directions outperform probe directions.** In ITI, mass-mean-shift substantially outperforms the probe weight direction for steering despite being derived from the same probe-selected heads. Marks & Tegmark (COLM 2024) show simple difference-in-mean probes identify directions "more causally implicated" than logistic regression. The boundary: **simpler, less parameterized readout methods may better approximate causal directions** because they are less prone to capturing spurious correlations.

---

## 9. False-friend table

| Paper | Why it looks relevant | Why it is NOT a true precedent | Reviewer risk |
|---|---|---|---|
| Zou et al. 2023 (RepE) | Reading + control of concepts via representation directions | Assumes detection = control without testing the gap; same vectors used for both | HIGH |
| Durmus et al. 2024 (Anthropic feature steering) | Evaluates SAE feature steering reliability | Tests steering quality but not the detection→control gap; starts from steering-selected features | HIGH |
| Meng et al. 2022 (ROME/Causal Tracing) | Localization guides editing | Assumes localization→editing works; Hase et al. later showed the link is illusory | MODERATE |
| Bricken et al. 2023 (Towards Monosemanticity) | SAE features that might steer | About feature discovery/interpretability, not about steering utility | LOW |
| Gao et al. 2024 (OpenAI SAE scaling) | SAE evaluation methodology | Pure SAE quality evaluation; doesn't test steering | MODERATE |
| "Steering off Course" (Da Silva et al. 2025) | Steering method reliability across 36 models | Tests method reliability, not detection→control gap | MODERATE |
| Concept erasure papers (Ravfogel, Belrose LEACE) | Representation intervention for concept removal | Engineer erasure to work based on detection; don't study cases where detection succeeds but control fails | MODERATE |
| ROME/MEMIT | Localization→editing pipeline | Different paradigm (weight editing, not activation steering); localization→editing link later disproven | MODERATE |
| "Mind the Performance Gap" (2025) | Title suggests detection→control gap | About capability-safety tradeoff during steering, not about detection quality as selection criterion | MODERATE |
| "Closing the Confidence-Faithfulness Gap" (2025) | Title mentions gap | About calibration vs. faithfulness — a case where detection→control actually works | LOW |

---

## 10. Reviewer-objection memo

### Objection 1: "This is just old decodability-vs-causality"

**Severity: MODERATE.** The reviewer has a legitimate intellectual lineage point. The principle is established since 2021 (Elazar et al., Geiger et al.) and codified by Belinkov (2022). **However, three genuine scope extensions differentiate:** (a) prior work studies linguistic features in encoder models (BERT/ELMo); this paper studies behavioral features in an instruction-tuned decoder model; (b) prior work compares probe accuracy vs. causal ablation; this paper compares detection methods as steering targets — a practical, not just theoretical, question; (c) prior work does not cross feature types (SAE vs. neurons vs. probes) on matched evaluation surfaces. **Response:** Acknowledge lineage explicitly in the introduction. Frame as: "The probing literature established that decodability does not imply causal use; we show this principle has direct practical consequences for the emerging practice of using interpretability tools to select steering targets."

### Objection 2: "You are mixing apples and oranges across operators/tasks"

**Severity: MODERATE-TO-SERIOUS.** SAE feature clamping and neuron activation scaling are different intervention operators. Evaluating ITI on TruthfulQA MC and H-neurons on FaithEval confounds feature type with evaluation surface. **Response:** (a) Acknowledge the comparison is not a clean factorial experiment and that divergent steering could partly reflect intervention-method differences. (b) Argue ecological validity — practitioners choosing between methods face exactly this heterogeneous landscape. (c) Add a table tracking {feature type × selection method × intervention operator × evaluation surface} to make the factorial structure transparent. (d) Where possible, use the same intervention operator for both feature types as a control.

### Objection 3: "Your positive H-neuron result undercuts your thesis"

**Severity: MINOR.** This objection misreads the claim. "Detection is not enough" does not mean "detection never works." The H-neuron positive result **strengthens** the thesis by showing that matched detection quality + divergent steering across methods proves detection quality alone is insufficient as a selection criterion. The finding that H-neurons' ℓ1-regularized selection jointly optimizes detection and functional safety (Gao et al., 2025) suggests **selection methodology**, not detection accuracy, determines steering quality. Cite Arad et al.'s input/output distinction as mechanistic explanation: H-neurons may be "output features" while SAE activation-selected features may be "input features."

### Objection 4: "Your selector comparison is caveated / missing controls"

**Severity: MODERATE.** Standard controls a reviewer would demand: (a) matched intervention operator across feature types; (b) detection-quality calibration showing accuracy is genuinely matched (within 2%); (c) random-feature baseline with matched sparsity; (d) hyperparameter sweep over layers, strengths, and number of components; (e) capability-preservation checks (MMLU, perplexity). **Response:** Report whichever controls are feasible; explicitly list uncontrolled variables in Limitations; frame the contribution as a practical warning rather than a causal claim about mechanism.

### Objection 5: "This is only one model"

**Severity: SERIOUS.** Da Silva et al. (ACL 2025) test across 36 models from 14 families and find massive variability. Tan et al. (NeurIPS 2024) find steerability is partly a dataset property (ρ=0.586 across architectures), partially mitigating the concern. **Response:** Acknowledge prominently. Cite Tan et al. for partial mitigation. Frame findings as "demonstrated in Gemma-3-4B-IT; generalization requires future work." If feasible, add even one additional model. Note that the four-stage scaffold (the methodological contribution) is model-independent even if the specific empirical results are model-specific.

---

## 11. Novelty-safe wording ladder

### Safe formulations

- "Building on established findings that decodability does not imply causal use (Elazar et al., 2021; Geiger et al., 2021), we investigate whether this principle extends to the practical question of selecting activation-steering targets."
- "We provide controlled evidence that, within a single model and benchmark, feature types with matched detection quality can yield sharply divergent steering outcomes."
- "Our results suggest that detection quality alone is an insufficient criterion for selecting steering targets — a finding independently supported by Arad et al. (2025) for SAE features and by AxBench (Wu et al., 2025) for representation methods more broadly."
- "We propose a four-stage evaluation scaffold — measurement, localization, control, externality — and demonstrate that each stage boundary can independently fail."

### Stronger but plausible formulations

- "We demonstrate the first cross-feature-type matched comparison showing that SAE features and H-neurons, despite similar detection accuracy, exhibit qualitatively different steering profiles on the same behavioral benchmark."
- "ITI's improvement on TruthfulQA multiple-choice does not transfer to open-ended generation, where we observe confident wrong-entity substitution rather than graceful degradation."
- "Evaluator choice, binary-vs-graded scoring, and truncation handling each independently reverse our conclusions about whether a jailbreak-safety steering intervention succeeds."

### Risky formulations (avoid)

- "We discover that detection does not imply control." (Established by multiple groups; not yours to discover.)
- "We show for the first time that strong readouts fail as steering targets." (Arad et al. 2025 show this for SAE features; AxBench shows detection/steering axes diverge.)
- "Prior work assumes detection quality predicts steering quality." (Some papers assume this, but others — Hase et al., Arad et al., Bhalla et al. — explicitly question it.)
- "Our evaluation methodology is novel." (StrongREJECT, GuidedBench, and Know Thy Judge establish evaluator-dependence.)

---

## 12. Related-work citation packs

### Flagship paper: Must cite

- **Arad, Mueller & Belinkov (EMNLP 2025)** — input/output feature distinction in SAE steering; nearest direct precedent
- **Hase, Bansal, Kim & Ghandeharioun (NeurIPS 2023)** — localization doesn't predict editing success (R² ≈ 0)
- **Li et al. (NeurIPS 2023)** — ITI; the paper's own Anchor 2 builds on this
- **Gao et al. (2025)** — H-Neurons; the paper's own Anchor 1 builds on this
- **Wu et al. (ICML 2025)** — AxBench; separately evaluates detection/steering
- **Elazar et al. (TACL 2021)** — amnesic probing; foundational decodability ≠ causality
- **Souly et al. (NeurIPS 2024)** — StrongREJECT; binary-vs-graded reversal
- **Pres et al. (NeurIPS 2024 MINT)** — MC evaluation overestimates steering
- **Bhalla et al. (2024)** — predict/control discrepancy; unified evaluation framework

### Flagship paper: Should cite

- **Geiger et al. (NeurIPS 2021)** — causal abstraction; analytic proof that probing ≠ causality
- **Kumar et al. (NeurIPS 2022)** — probes unreliable for concept removal
- **Zou et al. (2023)** — RepE; foundational reading/control framework
- **Marks & Tegmark (COLM 2024)** — geometry of truth; mass-mean probes more causal
- **Wolf et al. (2024)** — linear alignment / quadratic helpfulness tradeoff
- **O'Brien et al. (2024)** — SAE refusal steering causes capability degradation
- **Tan et al. (NeurIPS 2024)** — steering generalization and per-input variability
- **Eiras et al. (2025)** — Know Thy Judge; judge sensitivity
- **Xiong et al. (2026)** — Steering Externalities; benign steering causes safety regressions
- **Spherical Steering (2025)** — MC improves monotonically, generation peaks then degrades

### Flagship paper: Optional (cite if making stronger claims)

- **Hewitt & Liang (EMNLP 2019)** — control tasks for probing
- **Ravichander et al. (EACL 2021)** — probing paradigm limitations
- **Belinkov (Computational Linguistics 2022)** — probing survey
- **Templeton et al. (2024)** — Scaling Monosemanticity; Golden Gate Claude
- **Turner et al. (2024)** — ActAdd; side effects documented
- **Panickssery et al. (ACL 2024)** — CAA sycophancy steering; positive counterexample
- **Wattenberg & Viégas (ICML 2024 MI Workshop)** — "predict/control discrepancy" term
- **Nadaf (2025)** — "Steerable but Not Decodable"; inverse dissociation
- **Da Silva et al. (ACL 2025)** — "Steering off Course"; cross-model variability
- **Braun et al. (ICLR 2025 Workshop)** — steering vector unreliability
- **Fatahi Bayat et al. (Findings of ACL 2024)** — LITO; adaptive intensity; transfer limitations

### Companion measurement note: Must cite

- **Souly et al. (NeurIPS 2024)** — StrongREJECT
- **Huang et al. (2025)** — GuidedBench
- **Eiras et al. (2025)** — Know Thy Judge
- **Chao et al. (NeurIPS 2024)** — JailbreakBench; standardization need
- **Mazeika et al. (ICML 2024)** — HarmBench; evaluation pipeline disparities
- TruthfulQA scoring methodology analysis (3.5× variation)
- TurnTrout TruthfulQA format-heuristic analysis

### Companion measurement note: Should cite

- **"Rogue Scalpel" (2025)** — random steering compromises safety
- **"Safety Pitfalls of Steering Vectors" (Li et al., 2026)** — ±57% ASR from steering direction
- **UK AISI steering evaluation** — control vectors match designed vectors
- **HAJailBench (2025)** — 12K human annotations showing judge disagreement
- **Promptfoo/Chouldechova et al. (NeurIPS 2025)** — ASR incomparability across papers

---

## 13. Evidence table: 20 highest-value papers

| # | Paper | Venue/Year | Object Measured | Object Intervened | Evaluation Surface | Overlap | Novelty Impact | Category |
|---|---|---|---|---|---|---|---|---|
| 1 | Arad, Mueller, Belinkov — "SAEs Good for Steering" | EMNLP 2025 | SAE input features | SAE feature amplification | AxBench concepts | **Near-direct** | Reduces slogan-level novelty; doesn't cover cross-type comparison | Cautionary precedent |
| 2 | Hase et al. — "Does Localization Inform Editing?" | NeurIPS 2023 | Causal Tracing localization | ROME/MEMIT weight editing | CounterFact | **Near-direct** | Establishes principle in adjacent domain (editing) | Cautionary precedent |
| 3 | Wu et al. — AxBench | ICML 2025 Spotlight | Concept detection (AUROC) | Concept steering (generation) | 16K synthetic concepts | **Strong** | Shows detection/steering axes diverge for methods | Cautionary precedent |
| 4 | Li et al. — ITI | NeurIPS 2023 | Probe-selected truthful heads | Mass-mean activation shift | TruthfulQA MC+gen | **Strong** | Probe dir < mass-mean for steering | Methodological neighbor |
| 5 | Bhalla et al. — "Unifying Interpretability and Control" | arXiv 2024 | Multiple (SAE, lens, probes) | Encoder-decoder interventions | Multi-concept | **Strong** | Cites "predict/control discrepancy" explicitly | Methodological neighbor |
| 6 | Pres et al. — Reliable Evaluation | NeurIPS 2024 MINT | CAA/ITI effectiveness | CAA/ITI | MCQ vs. generation | **Strong** | MC overestimates steering | Cautionary precedent |
| 7 | Souly et al. — StrongREJECT | NeurIPS 2024 | Binary vs. graded ASR | N/A (evaluation method) | Jailbreak benchmarks | **Strong** | >70pp reversal from scoring method | Methodological neighbor |
| 8 | Gao et al. — H-Neurons | arXiv 2025 | MLP neuron activations | Neuron scaling | FaithEval, FalseQA, TriviaQA | **Strong** | Same model + FaithEval; detection-steering tradeoff noted | Methodological neighbor |
| 9 | Spherical Steering | arXiv 2025 | MC vs. generation metrics | Geodesic rotation | TruthfulQA MC + gen | **Strong** | MC monotonic up, generation peaks then degrades | Cautionary precedent |
| 10 | Eiras et al. — Know Thy Judge | arXiv 2025 | Judge FNR variation | N/A (evaluation) | Safety benchmarks | **Moderate** | 25× FNR variation across judges | Methodological neighbor |
| 11 | O'Brien et al. — SAE Refusal Steering | arXiv 2024 | SAE refusal features | Feature clamping | Jailbreak + MMLU | **Moderate** | Refusal features entangled with capabilities | Cautionary precedent |
| 12 | Elazar et al. — Amnesic Probing | TACL 2021 | Linguistic features | INLP removal | BERT language modeling | **Moderate** | Decodability ≠ task importance (foundational) | Background |
| 13 | Kumar et al. — Probes Unreliable | NeurIPS 2022 | Probe features | Concept removal | NLI, fairness | **Moderate** | Probes use spurious features for intervention | Background |
| 14 | Xiong et al. — Steering Externalities | arXiv 2026 | Compliance features | Compliance steering | HarmBench ASR | **Moderate** | Benign steering → 38.5% jailbreak rate | Cautionary precedent |
| 15 | Wolf et al. — RepE Tradeoffs | arXiv 2024 | Alignment/helpfulness | RepE vectors | TruthfulQA + MMLU | **Moderate** | Linear alignment / quadratic helpfulness | Methodological neighbor |
| 16 | Nadaf — "Steerable but Not Decodable" | arXiv 2025 | Logit lens decodability | Function vector addition | 12 tasks × 6 models | **Moderate** | Inverse dissociation (steer > decode) | Methodological neighbor |
| 17 | Tan et al. — Steering Generalization | NeurIPS 2024 | Steering vector transfer | CAA | Multiple behaviors | **Moderate** | Per-input variability; negative steering on some inputs | Cautionary precedent |
| 18 | Geiger et al. — Causal Abstractions | NeurIPS 2021 | Interchange interventions | Causal substitution | MQNLI | **Weak** | Analytic proof decodable ≠ causal (foundational) | Background |
| 19 | Fatahi Bayat et al. — LITO | Findings ACL 2024 | ITI truthful directions | Adaptive intensity ITI | TruthfulQA, TriviaQA | **Weak** | Admits long generation untested; limited transfer | Positive counterexample |
| 20 | Huang et al. — GuidedBench | arXiv 2025 | Judge guidelines | N/A (evaluation) | Jailbreak ASR | **Moderate** | >90% → 30.2% with guidelines | Methodological neighbor |

---

## 14. Paper-ready outputs

### Related Work paragraph for flagship

The observation that decodability does not imply causal use has a long lineage in the probing literature. Elazar et al. (2021) showed via amnesic probing that conventionally decoded features need not be causally important; Geiger et al. (2021) provided an analytic proof; Kumar et al. (2022) proved that probes used as intervention guides are unreliable even under ideal conditions. This principle has recently been extended to the activation-steering domain. Hase et al. (2023) demonstrated that Causal Tracing localization explains less than 3% of variance in knowledge-editing success. Arad et al. (2025) showed that SAE features with high input scores (detection quality) rarely coincide with features having high output scores (steering quality), yielding 2–3× improvement when selection is corrected. AxBench (Wu et al., 2025) separately evaluates detection and steering and finds that methods dominating one axis underperform on the other. Bhalla et al. (2024) frame this as a "predict/control discrepancy," showing that interventions often compromise coherence and underperform prompting. Our work extends these findings in three directions: we compare across feature types (SAE features, neurons, probe directions) with matched detection quality on the same behavioral benchmark; we document specific failure modes including confident entity substitution in open-ended generation; and we articulate a four-stage evaluation scaffold — measurement, localization, control, externality — demonstrating that each stage boundary can independently fail.

Concurrently, the reliability of evaluation methodology has received growing scrutiny. StrongREJECT (Souly et al., 2024) showed binary-vs-graded scoring reverses jailbreak effectiveness conclusions by over 70 percentage points. Pres et al. (2024) demonstrated that multiple-choice evaluation overestimates steering effectiveness relative to open-ended generation. We build on these findings by showing that evaluator artifacts — including judge dependence, truncation sensitivity, and scoring-type reversals — can independently change the scientific conclusion about a steering intervention's success.

### Related Work paragraph for companion measurement note

The dependence of safety-evaluation conclusions on methodological choices is well-documented. StrongREJECT (Souly et al., 2024) showed that binary non-refusal evaluators rate jailbreaks as >90% successful while graded evaluators measuring actual harmfulness rate them below 20% effective. GuidedBench (Huang et al., 2025) reduced this inflation further, showing maximum 30.2% ASR with case-specific evaluation guidelines. Know Thy Judge (Eiras et al., 2025) documented false-negative rate variation from 0.02 to 0.50 across judges and showed format changes alone cause 0.24 FNR jumps. Within TruthfulQA specifically, scoring-regime choices produce 3.5× variation in measured hallucination rates, and format heuristics alone achieve 79.6% MC1 accuracy. We extend these findings to the activation-steering evaluation context, showing that the same artifacts — judge selection, binary-vs-graded scoring, truncation handling, and disclaimer sensitivity — can reverse conclusions about whether a steering intervention improves or degrades model safety.

### Intro-framing paragraph

Mechanistic interpretability has delivered powerful tools for reading model internals — linear probes decode truthfulness, sparse autoencoders isolate interpretable features, and individual neurons predict hallucination. A natural next step is to repurpose these readouts as steering targets: if we can detect where a model represents truthfulness, can we intervene there to make it more truthful? We show that this inference is unreliable. Across three case studies in Gemma-3-4B-IT, we demonstrate that the pipeline from measurement to behavioral control contains at least four separable failure points — and that passing any one does not guarantee passing the next. SAE features and H-neurons achieve matched detection quality on faithfulness benchmarks but exhibit sharply divergent steering profiles. Inference-Time Intervention improves multiple-choice truthfulness scores while producing confident wrong-entity substitution in open-ended generation. And evaluator choice, metric design, and scoring assumptions independently reverse our conclusions about jailbreak-safety steering. These findings suggest that the field needs a more rigorous stage-gate methodology: validate measurement before trusting localization, test control before claiming it, and verify externalities before deploying interventions.
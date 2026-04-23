
# Research Arbitration Audit and Authoritative Synthesis
**Topic:** Claim-boundary audit for the proposed paper *Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT*  
**Arbitration date:** 2026-04-12 (Atlantic/Canary)  
**Inputs used:** the governing research brief and three deep-research reports supplied by the user.  
**Verification rule:** all input reports were treated as untrusted leads; only verified public sources and the uploaded brief were treated as evidence.

---

## Verification boundary note

This audit can verify **public prior work** and the **claims made inside the uploaded reports**. It cannot verify internal, unpublished project results that are only described in the reports. In practice, that means the following remain outside direct public verification unless the underlying experiments are supplied:

- the exact proposed SAE-vs-H-neuron matched-readout result on Gemma-3-4B-IT,
- the exact “bridge benchmark” wrong-entity-substitution result,
- any internal evaluator ablation that allegedly flips steering conclusions.

Those can still be strategically important, but in this audit they are treated as **paper-specific hypotheses to test**, not as already established literature facts.

---

# PART I — RESEARCH ARBITRATION AUDIT

## 1. Executive Verdict

### What the reports broadly got right

All three reports correctly identify the most important high-level correction: the weak claim — that probe accuracy / decodability does not by itself establish causal use — is old background, not novelty. That position is strongly supported by foundational probing work, including control-task arguments, amnesic probing, and theoretical/empirical work showing probes can be poor guides for intervention.  
Core verified background: Hewitt & Liang 2019 [S01], Elazar et al. 2021 [S02], Kumar et al. 2022 [S03].

All three reports also correctly identify the strongest **near-direct** prior art for the steering-era claim. The most relevant verified neighbors are:
1. **Arad, Mueller, Belinkov (EMNLP 2025)** on SAE input-vs-output feature selection, which directly shows that features that “look right” from activation patterns are often poor steering targets, and that output-score filtering yields ~2–3× better steering [S12].
2. **Li et al. (NeurIPS 2023 / ITI)**, which already shows a large generation-vs-multiple-choice mismatch and, within its own ablations, shows that a more natural probe-guided direction is not the best steering direction [S05].
3. **AxBench (ICML 2025)**, which explicitly separates concept detection from steering and shows that methods that look good on one axis do not necessarily look good on the other [S13].
4. **Bhalla et al. (2024)**, which explicitly frames a “predict/control discrepancy” and finds that several mechanistic intervention methods underperform simple prompting and can damage coherence [S08].

The reports are also right that the paper’s strongest remaining novelty frontier is **not the slogan**, but the **exact empirical form** and the **integrated framing**:
- a matched readout-strength, same-model, same-surface, cross-feature-type comparison (e.g., SAE features vs H-neurons) would still be interesting and likely novel **if the experiments are genuinely matched and well-controlled**;
- an explicit stage-gate framework — **measurement → localization → control → externality** — appears to be a useful synthesis contribution if presented as an organizing evaluation framework rather than as discovery of the underlying concerns.

### What the reports broadly got wrong

The main systematic failure across the reports is not wholesale fabrication. It is **over-interpretation**: real papers are often made to carry stronger novelty-killing or novelty-supporting weight than they actually can.

The most important recurring overstatements were:

1. **Overclaiming firstness for the steering-era thesis.**  
   A paper cannot safely claim “we are the first to show strong readouts fail as steering targets.” Arad et al. already show a closely related phenomenon within SAEs; ITI already contains selector/direction evidence; AxBench already separates detection and steering; Bhalla et al. already articulate a predict/control discrepancy [S05, S08, S12, S13].

2. **Treating measurement fragility as novel in the abstract.**  
   Judge dependence, binary-vs-graded scoring problems, and artifact sensitivity are already well established in safety/jailbreak evaluation. StrongREJECT, GuidedBench, Know Thy Judge, and Safer or Luckier collectively use up the generic claim that “evaluation choices matter” [S09, S14, S15, S16].  
   The only defensible novelty slot here is: **these evaluator choices materially change conclusions in this paper’s specific representation-engineering setting**.

3. **Overextending adjacent benchmarks into direct evidence.**  
   BRIDGE is real, but it is fundamentally a retrieval / relevance-benchmarking paper with downstream RAG evaluation, not direct prior evidence for activation-steering-induced wrong-entity substitution [S22]. It can support externality framing only indirectly.

4. **Treating exact paper-specific failure modes as already established by the literature.**  
   The literature supports “MC / answer-selection success does not guarantee open-ended generation success” [S05, S07, S17]. It does **not** yet publicly establish the exact proposed paper claim “ITI causes confident wrong-entity substitution on the bridge benchmark.” That remains a project-specific empirical claim.

5. **Underweighting positive counterexamples.**  
   Some detector-selected or direction-selected interventions clearly do work under certain conditions — refusal direction work, H-Neurons, one-shot optimized steering vectors, and MAT-Steer are all real counterexamples [S10, S17, S18, S19].  
   Therefore the flagship paper must argue **heuristic unreliability and stage-contingency**, not impossibility.

### Strongest overall report

**Strongest overall: GPT report.** It was the most careful at:
- separating foundational background from near-direct precedent,
- distinguishing benchmark-conditional evidence from broader claims,
- identifying which anchors were actually supported by literature versus merely plausible,
- and avoiding the strongest overclaims.  
It still leaned on a few adjacent sources that should be downweighted, but overall it showed the best claim hygiene.

**Second strongest: Opus report.** It captured the right paper-level framing and produced the most useful paper-ready wording ladder. Its main weakness was stronger-than-warranted confidence in a few numbers and source applications.

**Weakest overall: Gemini report.** It surfaced some real recent leads, but it repeatedly over-promoted weakly applicable or lower-weight sources into major novelty threats/support. Its epistemic style was the least restrained.

### Biggest epistemic risks

The biggest risks for the final synthesis were:
- collapsing **adjacent** evidence into **near-direct** evidence,
- treating benchmark-local results as domain-general,
- confusing “the source exists” with “the source supports the claim as stated,”
- treating the paper’s internal anchor results as if the literature had already verified them,
- and ignoring strong positive counterexamples.

### Most important corrections for the final synthesis

1. The paper **cannot** claim novelty for “decodability does not imply causal use.” That is settled background.  
2. The paper **cannot** claim novelty for “evaluation choices matter” in the abstract.  
3. The paper **can plausibly claim** novelty for a **matched-detection, cross-feature-type, same-surface steering divergence** if it actually demonstrates that cleanly.  
4. The paper **can plausibly claim** novelty for the **integrated four-stage scaffold**, but only as a methodological synthesis / evaluation framework.  
5. The “wrong-entity substitution on bridge” story should be presented as a **paper result to prove**, not as something the literature already established.  
6. The title/positioning must emphasize **“detection is not a reliable steering-target heuristic”** or **“strong readouts are insufficient evidence for good steering targets”**, not **“detectors fail”**.

---

## 2. Report Scorecard

| Report | Accuracy | Source Quality | Freshness | Methodological Discernment | Reasoning | Uncertainty Handling | Usefulness | Hallucination Risk* | Scientific Honesty | Net Influence on Final Synthesis | Key Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| GPT report | 4 | 4 | 4 | 5 | 5 | 5 | 5 | 4 | 5 | **High** | Best separation of background vs near-direct prior art; strongest novelty-safe wording; appropriately skeptical about bridge/wrong-entity claims. |
| Opus report | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 4 | **Medium–High** | Strongest paper-ready framing and related-work architecture; but more overconfident with some exact numbers and at least one source application. |
| Gemini report | 2 | 3 | 3 | 2 | 2 | 2 | 3 | 2 | 2 | **Low** | Useful for finding newer leads, but repeatedly overgeneralizes adjacent or lower-weight sources and overstates what they kill/support. |

\*Hallucination Risk is reverse-scored: 5 = low risk.

### Brief report notes

- **GPT report:** notably right about Arad/AxBench/ITI/Bhalla as the real nearest neighbors; notably wrong only in giving a bit too much rhetorical weight to some measurement-side context sources. Unique verified value: the cleanest differentiation between “established,” “adjacent,” and “still plausibly novel.”
- **Opus report:** notably right about the four-stage scaffold and the cross-method comparison being stronger novelty than the slogan. Notably wrong in some exact quantitative and source-application claims. Unique verified value: best paper-ready wording ladder and strongest recognition that the safe core is narrower than the reports first suggest.
- **Gemini report:** notably right that strong positive counterexamples force a “heuristic unreliability” framing. Notably wrong in over-weighting lower-tier or weakly applicable 2026 sources and in turning externality hypotheses into literature-supported facts. Unique verified value: it surfaced a few recent papers worth checking, especially the Yap 2026 preprint — but those should be downweighted.

---

## 3. Agreement Map

| Claim ID | Normalized Claim | Opus | GPT | Gemini | Verification Verdict | Confidence | Why It Survives |
|---|---|---|---|---|---|---|---|
| C01 | Probe accuracy / decodability does not establish causal use. | Yes | Yes | Yes | **Verified-Durable** | High | Strong foundational support from probing literature [S01–S03]. |
| C02 | Detector/readout quality is an unreliable heuristic for selecting steering targets. | Yes | Yes | Yes | **Replicated / Convergent Evidence** | Medium–High | Supported convergently by Arad, ITI ablations, AxBench, and Bhalla; still benchmark/method contingent [S05, S08, S12, S13]. |
| C03 | Within SAEs, activation-based or input-oriented selection can pick poor steering targets. | Yes | Yes | Yes | **Verified-Current** | High | Directly shown by Arad et al. [S12]. |
| C04 | Detection and steering should not be collapsed into one evaluation axis. | Yes | Yes | Yes | **Replicated / Convergent Evidence** | High | AxBench, Arad, Bhalla, and ITI all support separation, though in different forms [S05, S08, S12, S13]. |
| C05 | MC / answer-selection success does not guarantee open-ended generation success. | Yes | Yes | Yes | **Replicated / Convergent Evidence** | Medium–High | Supported by ITI, Pres et al., steering reliability work, and Causality ≠ Invariance [S05–S07, S20]. |
| C06 | The paper’s strongest novelty is the integrated four-stage scaffold plus the exact cross-method comparison, not the weak slogan. | Yes | Yes | Yes | **Plausible Inference** | Medium | This is a judgment after verification, not a direct literature claim; it survives because the weaker novelty claims are already occupied while no public paper was found articulating this exact integrated scaffold. |

---

## 4. Disagreement / Conflict Map

| Claim ID | Normalized Claim | Opus Position | GPT Position | Gemini Position | What the Evidence Shows | Why They Differ | Final Ruling |
|---|---|---|---|---|---|---|---|
| D01 | “Good detectors fail as steering targets” is still broadly novel. | Low-to-moderate novelty only | Partially established, not closed | Threatened by recent 2026 work | Public literature already contains near-direct steering-era evidence: Arad, ITI selector/direction evidence, AxBench, Bhalla [S05, S08, S12, S13]. | GPT and Opus are more restrained; Gemini overweights a recent adjacent preprint. | **Partially Supported / Overstated** as a novelty claim. |
| D02 | Measurement fragility is a novel discovery for this paper. | Generally says no, except in steering context | Says no; only novel in representation-engineering application | Tends to present it as stronger novelty | StrongREJECT, GuidedBench, Know Thy Judge, and Safer or Luckier already establish generic judge/metric fragility [S09, S14–S16]. | Gemini blurs “already known generally” with “new in this exact application.” | **Contradicted** if stated generically; **Hypothesis to Test** if applied specifically to the paper’s own steering setting. |
| D03 | BRIDGE is a direct externality benchmark for activation steering failures. | Not central | Treated cautiously | Treated as a strong anchor-support source | BRIDGE is a retrieval/relevance benchmark with downstream RAG evaluation, not a direct activation-steering benchmark [S22]. | Gemini stretches benchmark relevance; GPT is more careful. | **Overgeneralized from narrow setting**. |
| D04 | The literature already documents the exact “wrong-entity substitution” failure mode for truthfulness steering. | Says appears novel | Says may still be novel | Leans as though literature already scaffolds it strongly | No verified public source established this exact failure mode in activation steering. The literature supports broader transfer/externality concerns, not this exact result [S05–S07, S20]. | Reports differ in how much they let adjacent work stand in for exact evidence. | **Unsupported** as a literature claim; **Hypothesis to Test** for the paper’s own experiments. |
| D05 | Spherical Steering is evidence that MC scores improve while open-ended generation degrades. | Used that way | Not central | Not central | Verified source says the opposite: Spherical Steering claims Pareto improvement vs addition baselines, with improved or maintained generation quality [S23]. | Likely source misread or method conflated with addition baselines. | **Contradicted**. |

---

## 5. Claim Concordance Ledger

| Claim ID | Theme | Claim Type | Normalized Claim | Asserted By | Best Verified Sources | Verdict | Confidence | External Validity | Freshness Status | Disposition | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | Problem framing | Theoretical / methodological | High probe/readout accuracy does not imply causal use by the model. | All reports | [S01], [S02], [S03] | **Verified-Durable** | High | Broad | Durable | Keep as Fact | Foundational background only; no novelty. |
| C02 | Target selection | Methodological claim | Predictive readout quality is an unreliable heuristic for choosing steering targets. | All reports | [S05], [S08], [S12], [S13] | **Replicated / Convergent Evidence** | Medium–High | Likely task-class relevant, but not universal | Current | Keep with Caveat | Supported across multiple settings, but not all methods/tasks. |
| C03 | SAE feature steering | Empirical result | Within SAEs, high input-score features rarely coincide with high output-score features; output-score filtering improves steering ~2–3×. | Opus, GPT | [S12] | **Verified-Current** | High | Benchmark- and method-contingent | Current | Keep as Benchmark-Conditional | Strong direct prior, but limited to SAE feature selection setup. |
| C04 | Detection vs steering | Empirical / benchmark claim | Detection and steering are separable evaluation axes; winners differ across them. | Opus, GPT | [S13] | **Benchmark-Contingent** | High | Benchmark-specific | Current | Keep as Benchmark-Conditional | AxBench is strong, but on synthetic concepts and Gemma-2 models. |
| C05 | Localization vs control | Empirical / methodological | Better localization does not reliably predict better editing/intervention success. | Opus, GPT | [S04] | **Verified-Current** | Medium–High | Setting-specific | Medium-durable | Keep with Caveat | Strong but in knowledge editing rather than activation steering. |
| C06 | Positive counterexample | Empirical result | A tiny subset of neurons (<0.1%) can detect hallucination and interventions on them causally modulate over-compliance. | All reports | [S17] | **Verified-Current** | Medium | Likely task-class relevant, but preprint-only | Current | Keep with Caveat | Important counterexample against overbroad anti-detector claims. |
| C07 | Cross-method novelty | Normative / novelty judgment | A matched-readout, same-surface SAE-vs-H-neuron comparison remains a plausible novelty slot. | All reports | Inference from [S12], [S13], [S17] | **Plausible Inference** | Medium | Unknown | Current | Recast as Hypothesis | Absence-of-prior-art inference; cannot prove universal firstness. |
| C08 | Control vs externality | Empirical / methodological | Success on MC / answer-selection does not guarantee open-ended generation success. | All reports | [S05], [S06], [S07], [S20] | **Replicated / Convergent Evidence** | High | Likely broader than one benchmark, but not universal | Current | Keep as Fact | Strongly supported, though exact failure mode varies by study. |
| C09 | ITI specifics | Empirical result | ITI greatly improves generation truthfulness but only modestly improves TruthfulQA MC accuracy; the probe-weight direction is not the best steering direction. | Opus, GPT, Gemini | [S05] | **Verified-Current** | High | Benchmark-contingent | Current | Keep as Benchmark-Conditional | Strong evidence that “best readout” is not automatically “best control direction.” |
| C10 | Externality exactness | Empirical result | ITI causes confident wrong-entity substitution on a bridge benchmark. | Opus, GPT, Gemini (implicitly or explicitly) | No verified public source | **Unsupported** | Low | Unknown | Unknown | Drop | This may still be the paper’s own result, but it is not literature-backed yet. |
| C11 | Measurement fragility | Empirical / methodological | Judge choice, guidelines, and scoring granularity materially change jailbreak/safety conclusions. | All reports | [S09], [S14], [S15], [S16] | **Replicated / Convergent Evidence** | High | Likely task-class relevant | Current | Keep as Fact | Generic measurement fragility is already established. |
| C12 | Measurement in steering | Methodological claim | In representation-steering evaluation specifically, measurement choices can reverse conclusions. | All reports | [S07], plus inference from [S09], [S14–S16] | **Hypothesis to Test** | Medium-Low | Unknown | Current | Recast as Hypothesis | Very plausible, but not yet strongly established in this exact domain. |
| C13 | Integrated scaffold | Methodological claim | Measurement, localization, control, and externality are separable empirical stages that should not be conflated. | All reports | Inference from [S04], [S07], [S09], [S20] | **Plausible Inference** | Medium | Likely broader than one benchmark | Medium-durable | Keep with Caveat | The pieces exist; the exact four-stage packaging appears novel as synthesis. |
| C14 | Overbroad novelty claim | Normative / novelty judgment | “We are first to show that strong readouts fail as steering targets.” | Mostly implied risk | [S05], [S08], [S12], [S13] | **Contradicted** | High | Broad | Current | Drop | Too strong given verified near-direct prior work. |
| C15 | Boundary condition | Empirical result | Some simple or detector-linked steering methods work very well on certain single-axis behaviors. | GPT, Gemini, partly Opus | [S10], [S17], [S18], [S19] | **Replicated / Convergent Evidence** | Medium–High | Setting-specific | Current | Keep as Fact | Necessary boundary condition for claim hygiene. |

---

## 6. Study / Source Registry

**Retrieval date for all web sources:** 2026-04-12 unless otherwise noted.

| Source ID | Title | Authors / Organization | Venue / Publisher | Published Date | Version / Retrieved Date | DOI / arXiv / URL | Source Tier | Supports Which Claims | Issues / Limits |
|---|---|---|---|---|---|---|---|---|---|
| S01 | Designing and Interpreting Probes with Control Tasks | John Hewitt, Percy Liang | EMNLP-IJCNLP 2019 / ACL Anthology | 2019-11 | Retrieved 2026-04-12 | DOI 10.18653/v1/D19-1275; https://aclanthology.org/D19-1275/ | Tier 1 | C01 | Foundational; not about steering target selection. |
| S02 | Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals | Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg | TACL 2021 / MIT Press + ACL | 2021 | Retrieved 2026-04-12 | arXiv:2006.00995; https://arxiv.org/abs/2006.00995 | Tier 1 | C01 | Strong for causal-use caution; not about steering per se. |
| S03 | Probing Classifiers are Unreliable for Concept Removal and Detection | Abhinav Kumar, Chenhao Tan, Amit Sharma | NeurIPS 2022 / OpenReview + arXiv | 2022 | Retrieved 2026-04-12 | arXiv:2207.04153; https://arxiv.org/abs/2207.04153 | Tier 1 | C01 | Strong theory/empirics; concept removal setting, not LLM steering. |
| S04 | Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models | Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun | NeurIPS 2023 Spotlight | 2023 | Retrieved 2026-04-12 | arXiv:2301.04213; https://arxiv.org/abs/2301.04213 | Tier 1 | C05, C13 | Knowledge-editing, not activation steering. |
| S05 | Inference-Time Intervention: Eliciting Truthful Answers from a Language Model | Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg | NeurIPS 2023 Spotlight / OpenReview | 2023 | Retrieved 2026-04-12 | arXiv:2306.03341; https://arxiv.org/abs/2306.03341 | Tier 1 | C02, C08, C09, C14 | Older model family (LLaMA/Alpaca); task/eval surfaces matter. |
| S06 | Analyzing the Generalization and Reliability of Steering Vectors | Daniel Tan, David Chanin, Aengus Lynch, Dimitrios Kanoulas, Brooks Paige, Adria Garriga-Alonso, Robert Kirk | arXiv preprint | 2024-07 | Retrieved 2026-04-12 | arXiv:2407.12404; https://arxiv.org/abs/2407.12404 | Tier 2 | C08 | Strong reliability evidence, but preprint and not specific to truthfulness anchor. |
| S07 | Towards Reliable Evaluation of Behavior Steering Interventions in LLMs | Itamar Pres, Laura Ruis, Ekdeep Singh Lubana, David Krueger | MINT @ NeurIPS 2024 workshop; arXiv | 2024-10 | Retrieved 2026-04-12 | arXiv:2410.17245; https://arxiv.org/abs/2410.17245 | Tier 2 | C08, C12, C13 | Workshop paper; narrower empirical base than major conference paper. |
| S08 | Towards Unifying Interpretability and Control: Evaluation via Intervention | Usha Bhalla, Suraj Srinivas, Asma Ghandeharioun, Himabindu Lakkaraju | arXiv preprint | 2024-11 | Retrieved 2026-04-12 | arXiv:2411.04430; https://arxiv.org/abs/2411.04430 | Tier 2 | C02, C14 | Useful framing, but preprint and heterogeneous methods/tasks. |
| S09 | A StrongREJECT for Empty Jailbreaks | Alexandra Souly et al. | NeurIPS 2024 Datasets & Benchmarks | 2024 | Retrieved 2026-04-12 | arXiv:2402.10260; https://arxiv.org/abs/2402.10260 | Tier 1 | C11, C13 | About jailbreak evaluation, not representation steering. |
| S10 | Refusal in Language Models Is Mediated by a Single Direction | Andy Arditi, Oscar Balcells Obeso, Aaquib Syed, Daniel Paleka, Nina Rimsky, Wes Gurnee, Neel Nanda | NeurIPS 2024 poster / OpenReview | 2024-09 | Retrieved 2026-04-12 | https://openreview.net/forum?id=pH3XAQME6c | Tier 1 | C15 | Strong positive counterexample, but single-axis refusal domain. |
| S11 | FaithEval: Can Your Language Model Stay Faithful to Context, Even If “The Moon is Made of Marshmallows” | Yifei Ming et al. | ICLR 2025 / OpenReview | 2025 | Retrieved 2026-04-12 | arXiv:2410.03727; https://arxiv.org/abs/2410.03727 | Tier 1 | Benchmark grounding for anchor 1 | Diagnostic faithfulness benchmark, not itself a steering paper. |
| S12 | SAEs Are Good for Steering – If You Select the Right Features | Dana Arad, Aaron Mueller, Yonatan Belinkov | EMNLP 2025 / ACL Anthology | 2025-11 | Retrieved 2026-04-12 | DOI 10.18653/v1/2025.emnlp-main.519; https://aclanthology.org/2025.emnlp-main.519/ | Tier 1 | C02, C03, C07, C14 | Strongest near-direct prior, but within SAE feature-selection regime. |
| S13 | AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders | Zhengxuan Wu et al. | ICML 2025 / OpenReview | 2025 | Retrieved 2026-04-12 | https://openreview.net/forum?id=K2CckZjNy0 | Tier 1 | C02, C04, C14 | Synthetic concepts on Gemma-2; not a truthfulness benchmark. |
| S14 | Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts | Hongyu Chen, Seraphina Goldfarb-Tarrant | ACL 2025 / ACL Anthology | 2025-07 | Retrieved 2026-04-12 | DOI 10.18653/v1/2025.acl-long.970; https://aclanthology.org/2025.acl-long.970/ | Tier 1 | C11 | Safety evaluation, not steering-specific. |
| S15 | GuidedBench: Equipping Jailbreak Evaluation with Guidelines | Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, Shuai Wang | arXiv preprint | 2025-02 | Retrieved 2026-04-12 | arXiv:2502.16903; https://arxiv.org/abs/2502.16903 | Tier 2 | C11 | Preprint; jailbreak evaluation domain. |
| S16 | Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges | Francisco Eiras, Eliott Zemour, Eric Lin, Vaikkunth Mugunthan | arXiv preprint / ICLR 2025 workshop circulation | 2025-03 | Retrieved 2026-04-12 | arXiv:2503.04474; https://arxiv.org/abs/2503.04474 | Tier 2 | C11 | Preprint/workshop; judge-robustness focus. |
| S17 | H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs | Cheng Gao, Huimin Chen, Chaojun Xiao, Zhiyi Chen, Zhiyuan Liu, Maosong Sun | arXiv preprint | 2025-12 | Retrieved 2026-04-12 | arXiv:2512.01797; https://arxiv.org/abs/2512.01797 | Tier 2 | C06, C07, C15 | Strongly relevant but still preprint; includes Gemma-3 and FaithEval counterfactual context. |
| S18 | One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in LLMs | Jacob Dunefsky, Arman Cohan | COLM 2025 / OpenReview | 2025-07 | Retrieved 2026-04-12 | https://openreview.net/forum?id=teW4nIZ1gy | Tier 1 | C15 | Safety-relevant behavior; not a truthfulness paper. |
| S19 | Multi-Attribute Steering of Language Models via Targeted Intervention | Duy Nguyen, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal | ACL 2025 camera-ready; arXiv | 2025-02 / 2025-07 v2 | Retrieved 2026-04-12 | arXiv:2502.12446; https://arxiv.org/abs/2502.12446 | Tier 1/2 | C15 | Positive counterexample with more structured steering; evaluation partly judge-based. |
| S20 | Causality ≠ Invariance: Function and Concept Vectors in LLMs | Gustaw Opiełka, Hannes Rosenbusch, Claire E. Stevenson | ICLR 2026 | 2026 | Retrieved 2026-04-12 | arXiv:2602.22424; https://arxiv.org/abs/2602.22424 | Tier 1 | C08, C13 | Strong on format-specific causal representations; not about detector-quality directly. |
| S21 | Behavioral Steering in a 35B MoE Language Model via SAE-Decoded Probe Vectors: One Agency Axis, Not Five Traits | Jia Qing Yap | arXiv preprint | 2026-03 | Retrieved 2026-04-12 | arXiv:2603.16335; https://arxiv.org/abs/2603.16335 | Tier 2 | Context for C02, C15 | Real and relevant, but single-author preprint on agentic traits; weight should be limited. |
| S22 | Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevant Assessment for IR Benchmarks (BRIDGE / DREAM) | Minjeong Ban et al. | arXiv preprint / ICLR 2026-era circulation | 2026-02 | Retrieved 2026-04-12 | arXiv:2602.06526; https://arxiv.org/abs/2602.06526 | Tier 2 | Context for externality framing only | IR / RAG relevance benchmark, not direct activation-steering evidence. |
| S23 | Spherical Steering: Geometry-Aware Activation Rotation for Language Models | Zejia You, Chunyuan Deng, Hanjie Chen | arXiv preprint | 2026-02 | Retrieved 2026-04-12 | arXiv:2602.08169; https://arxiv.org/abs/2602.08169 | Tier 2 | Positive counterexample / conflict-check | Actually claims improved MC *and* maintained/improved generation vs addition baselines, so it cannot support “MC up, generation down” as stated in one report. |

---

## 7. Methodology and Benchmark Audit

| Claim / Study | Dataset / Benchmark | Metric | Baseline Quality | Reproducibility Signals | Key Caveats | External Validity | Replication Status | Bottom-Line Takeaway |
|---|---|---|---|---|---|---|---|---|
| S12 Arad et al. 2025 | SAE concept steering; AxBench-style evaluation | Steering quality / coherence and feature scores | Good within SAE comparison | ACL paper, methods clear | Compares feature types *within* SAE regime, not across representation families | Benchmark-conditional | Not yet independently replicated at scale | Best near-direct evidence that “looks relevant” ≠ “steers well.” |
| S13 AxBench 2025 | Synthetic concepts on Gemma-2-2B/9B | Separate detection and steering metrics | Strong baseline set incl. prompting and finetuning | OpenReview + benchmark framing | Synthetic concepts; not truthfulness/faithfulness | Benchmark-conditional | Convergent with Arad/Bhalla | Strong evidence that detection and steering axes should be separated. |
| S05 ITI 2023 | TruthfulQA MC + generation; transfer sets | MC accuracy, TruthfulQA generative truthfulness, helpfulness | Solid for 2023 | NeurIPS paper, project page | Older model family; multiple surfaces; judge/parsing issues remain | Task-class relevant | Convergent with later work | Strong evidence for surface mismatch and selector fragility. |
| S07 Pres et al. 2024 | Open-ended vs MC steering evaluation | Open-ended evaluation + likelihood-aware metrics | Good methodological critique, narrower empirical base | Workshop paper + slides + project page | Workshop paper; fewer methods than a broad benchmark | Likely task-class relevant | Converges with later format-transfer work | Best direct prior for “MC can overestimate steering success.” |
| S04 Hase et al. 2023 | CounterFact / editing tasks | Rewrite score vs localization effect | Good for knowledge-editing | NeurIPS spotlight | Different intervention class (weight edits, not activation steering) | Setting-specific | Conceptually replicated elsewhere | Good analogical evidence that localization does not automatically tell you how to control. |
| S17 H-Neurons 2025 | TriviaQA/NQ/BioASQ/NonExist; FalseQA; FaithEval; Sycophancy; Jailbreak | Detection accuracy / AUROC; compliance-rate shifts under activation scaling | Strong internal comparisons, but preprint | Detailed methods and broad eval | Preprint; control operator is neuron-scaling; some evals use GPT-4o/judges/parsers | Likely task-class relevant | Not yet strongly replicated | Important counterexample: detector-selected neurons can steer effectively in this family. |
| S06 Tan et al. 2024 | Steering Bench / concept steering across prompt shifts | Input-level steerability and OOD robustness | Moderate | Code released | Preprint; concept coverage varies | Likely broader than one benchmark | Convergent | Steering is brittle, variable by input, and sometimes anti-steerable. |
| S09/S14/S15/S16 measurement papers | Jailbreak/safety evaluation | ASR, graded harmfulness, judge variance / FNR | Strong for measurement | Mix of NeurIPS/ACL/preprints | Domain is safety evaluation, not representation engineering | Likely class-relevant but not identical | Strongly convergent | Generic measurement fragility is already established. |
| S10/S18/S19 positive control papers | Refusal / safety / multi-attribute settings | Behavior-specific steering metrics | Strong enough for boundary conditions | Published/accepted or strong preprint | Behaviors often lower-dimensional or more structured | Setting-specific | Convergent | Detector/direction-based steering sometimes works very well; anti-detector absolutism is untenable. |
| S20 format-transfer paper | Format-matched and mismatched tasks | In-distribution vs OOD steering performance | Good | ICLR 2026 | Representation-level, not explicitly about detectors | Likely task-class relevant | Early but strong | Causal control can be format-local; invariance is a separate empirical question. |

---

## 8. Hallucination, Staleness, and Weak-Support Register

| Item | Report | Problem Type | Why It Fails | Consequence for Synthesis |
|---|---|---|---|---|
| “We are first to show strong readouts fail as steering targets.” | All, implicitly at risk | Overgeneralized from narrow setting | Arad, ITI ablations, AxBench, and Bhalla already occupy much of the weaker novelty territory [S05, S08, S12, S13]. | Final synthesis must avoid firstness language and claim a narrower novelty slot. |
| Generic novelty for judge dependence / graded-vs-binary scoring | All, especially Gemini | Stale / superseded | StrongREJECT, GuidedBench, Know Thy Judge, and Safer or Luckier already establish generic evaluator fragility [S09, S14–S16]. | Measurement novelty must be framed as application-specific, not generic. |
| BRIDGE treated as direct support for truthfulness-steering externality | Gemini | Weak applicability to the research brief | BRIDGE is a retrieval/relevance benchmark with downstream RAG implications, not a direct activation-steering truthfulness benchmark [S22]. | Can support externality framing only indirectly. |
| “Wrong-entity substitution” treated as already literature-supported | Gemini; sometimes Opus rhetoric | Source does not support claim | No verified public source directly establishes this exact failure mode for activation steering. | Must be demoted to a paper-specific hypothesis until shown by the paper’s own data. |
| Spherical Steering used as evidence that MC gains come with degraded generation quality | Opus | Source does not support claim | Verified paper says Spherical Steering improves MC while maintaining or improving generation quality versus addition baselines [S23]. | This citation should be removed or inverted into a positive counterexample / method-specific nuance. |
| Exact StrongREJECT-style numerical reversal imported too aggressively into steering context | Opus | Overgeneralized from narrow setting | StrongREJECT supports evaluator overstatement in jailbreak evaluation, but exact quoted reversal numbers do not automatically transfer to steering papers [S09]. | Use qualitative measurement-fragility claim, not imported exact numbers. |
| Yap 2026 treated as a major novelty-killer on par with Arad/AxBench | Gemini | Weak applicability to research brief | Real and interesting, but single-author preprint on agentic MoE behavior; lower evidentiary weight than peer-reviewed or benchmark-centric sources [S21]. | Mention only as lower-weight, cautionary adjacent evidence. |
| J-Detector treated as a near-direct prior for steering evaluation | Gemini | Weak applicability to research brief | It is about detecting LLM-generated judgments / auditing judge behavior, not about steering-target evaluation per se. | At most a contextual measurement-side citation, not a core anchor citation. |

---

## 9. Gaps and Unknowns

1. **No public prior work cleanly verifies the exact proposed cross-feature-type matched comparison.**  
   I did not find a public paper that does the exact: matched readout strength + same model + same behavioral benchmark + cross-feature-type (e.g., SAE vs H-neurons) + divergent steering outcome. This remains a plausible novelty slot, but it is still an absence-of-found-evidence inference, not a theorem.

2. **The exact “bridge wrong-entity substitution” claim remains unverified.**  
   The literature supports transfer failure and benchmark-local control, but not this exact truthfulness failure mode.

3. **The measurement anchor is only partially mapped into representation engineering.**  
   The generic measurement literature is strong, but there is still room to show that evaluator/scoring choices change conclusions in *this exact mechanistic/steering setup*.

4. **External validity remains unresolved.**  
   Much of the best evidence is benchmark-specific or method-family-specific. The field still lacks strong, independent replication across:
   - multiple model families,
   - multiple concept classes,
   - multiple intervention operators,
   - and realistic deployment-style evaluation surfaces.

5. **The right control variable for “matched detection quality” is nontrivial.**  
   If the flagship paper claims matched readout strength, it must specify:
   - task,
   - split,
   - metric,
   - readout class,
   - layer budget,
   - intervention operator,
   - and optimization budget.  
   Otherwise reviewers can argue the match is only superficial.

---

# PART II — FINAL AUTHORITATIVE SYNTHESIS

## 1. Executive Summary and Bottom-Line Answer to the Brief

### Bottom-line answer

The literature **does not support** a novelty claim of the form:

> “We discovered that good detectors/readouts do not imply causal relevance or good steering targets.”

That weak version is already taken — first by foundational probing/causality work, and later by multiple steering-era papers that directly or indirectly show a predict/control gap [S01–S03, S05, S08, S12, S13].

What the literature **does support** is a narrower, stronger, and more decision-useful framing:

1. **Observed:** detection quality and steering quality are already known to be separable, but the evidence is heterogeneous and method-dependent. In particular, within-SAE selection, probe-guided steering, and concept-benchmark studies all suggest that readout strength is not a reliable steering-target heuristic in general [S05, S08, S12, S13].

2. **Observed:** success on one evaluation surface — especially multiple-choice or answer-selection surfaces — does not guarantee success on open-ended generation or out-of-distribution formats [S05–S07, S20].

3. **Observed:** evaluator choice, scoring design, and artifact sensitivity already materially alter conclusions in safety/jailbreak evaluation, so any paper making intervention claims without measurement audits is vulnerable to measurement error [S09, S14–S16].

4. **Inference:** the proposed flagship paper’s strongest novelty frontier is therefore **not** the slogan, but the combination of:
   - a carefully matched cross-method empirical comparison (e.g., SAE features vs H-neurons on the same faithfulness surface),
   - plus an explicit four-stage audit framework: **measurement, localization, control, externality**.

5. **Open Question:** the paper-specific claims about *confident wrong-entity substitution* and about evaluator-driven conclusion flips in the exact steering setting remain to be shown by the paper’s own experiments. The public literature does not yet settle them.

### Practical bottom line for the project

If the goal is a flagship methods paper, the safest and strongest thesis is:

> **Strong readouts are insufficient evidence for good steering targets; robust intervention claims require separate validation of measurement, localization, control, and externality.**

That thesis is:
- truer than the slogan,
- better aligned with the verified literature,
- and strategically stronger because it makes the paper about **methodological discipline and cross-stage failure analysis**, not about a single overstated anti-detector claim.

---

## 2. Research Scope, Task Definition, and What Counts as Evidence Here

This synthesis treats the uploaded research brief as the governing problem formulation: the project is a **methods paper** about when internal readouts stop being reliable guides for intervention, with three anchor cases and a four-stage scaffold. It is not merely a benchmark paper or an evaluator paper.

### What counted as high-value evidence

High-value evidence in this arbitration was:
- primary empirical papers,
- benchmark and dataset papers,
- conference papers with direct methods/results,
- and strong technical preprints where no stronger source existed.

Lower-value context sources were used only to locate or triangulate stronger primary evidence.

### What did *not* count as strong evidence

The following were not allowed to carry decisive weight by themselves:
- the uploaded model-generated reports,
- generic surveys used instead of primary sources,
- blog-post rhetoric without direct methods/results,
- and adjacent papers that share vocabulary but do not actually test the relationship between readout quality and intervention utility.

### Scope boundaries

This synthesis deliberately does **not** flatten together:
- probing vs activation steering vs weight editing,
- truthfulness vs refusal vs safety vs style steering,
- multiple-choice vs open-ended generation,
- concept detection vs causal intervention,
- or public prior art vs the paper’s own unpublished results.

Those differences are not bookkeeping trivia; they are often exactly why literature claims appear to agree while actually talking past one another.

---

## 3. What Is Actually Well Supported in the Literature

### 3.1 Decodability is not causal use

**Observed.** The claim that internal information being decodable does not prove that the model functionally relies on it is foundational and durable. Hewitt & Liang’s control-task framework already warned against naive probe interpretation [S01]. Elazar et al. moved this toward causal analysis with amnesic probing, explicitly asking whether removing a property changes task behavior [S02]. Kumar et al. then showed that probing-based concept removal/detection can fail even under favorable assumptions because the probe may rely on non-concept features [S03].

**Implication.** Any paper that presents this as a discovery will look late to the conversation.

### 3.2 Steering-era work already weakens detector-as-target heuristics

**Observed.** Several later papers move from generic “probing isn’t causality” into the much more relevant practical question: **does a strong internal signal tell you where to intervene?**

- ITI already shows a large gap between the model’s generative truthfulness gains and its much smaller multiple-choice gains, and within its ablations shows that the probe-weight direction is not the best steering direction [S05].
- Arad et al. directly separate SAE **input features** from **output features** and show that high input-score and high output-score features rarely coincide, with 2–3× steering improvement from correcting selection [S12].
- AxBench explicitly measures detection and steering separately and finds that strong detection methods are not the same methods that steer best; prompting remains the most reliable steering baseline in that benchmark [S13].
- Bhalla et al. explicitly name the predict/control discrepancy and report that interventions can be inconsistent and coherence-damaging, sometimes underperforming simple prompting [S08].

**Inference.** By 2024–2025, the field already had enough evidence that “good detector” is, at minimum, an unreliable shortcut for “good steering target.”

### 3.3 Format and surface transfer are separable empirical questions

**Observed.** The literature strongly supports the proposition that steering success is often surface-local:
- ITI’s own results already differ sharply between generative truthfulness and multiple-choice accuracy [S05].
- Pres et al. explicitly argue that many steering evaluations overestimate efficacy by leaning on simplified or non-deployment-like settings, and they push evaluation toward open-ended generation [S07].
- Tan et al. show steering vectors can be brittle across inputs and prompt shifts, including anti-steerable cases [S06].
- Opiełka et al. show that causally effective function vectors can be format-specific, while more invariant concept vectors generalize better but often steer more weakly [S20].

**Implication.** A steering result on one surface is never enough. It answers a local question about that surface, not a global question about robust control.

### 3.4 Measurement fragility is already a first-order problem

**Observed.** Safety/jailbreak evaluation now has strong evidence that:
- automated evaluation can overstate success,
- guideline changes and scoring systems can radically change measured ASR,
- judge models are sensitive to formatting/style artifacts,
- and adversarial/meta-evaluation issues are substantial [S09, S14–S16].

**Implication.** Even if the flagship paper is not primarily about safety evaluation, it cannot make intervention claims as if measurement were a neutral afterthought. Measurement is already an experimental stage in its own right.

### 3.5 Strong positive counterexamples exist

**Observed.** Some detector-like or direction-like interventions really do work:
- refusal is strongly mediated by a single direction in many safety-aligned chat models [S10];
- H-Neurons identify an extremely sparse neuron subset that both predicts hallucination and causally modulates over-compliance, including on FaithEval-related settings [S17];
- one-shot optimized steering vectors can mediate safety-relevant behavior across inputs [S18];
- MAT-Steer reports gains over ITI and fine-tuning baselines in multi-attribute settings [S19].

**Implication.** The correct claim is not “readouts don’t work.” The correct claim is “readout strength is not a reliable, standalone target-selection criterion.”

---

## 4. What Appears to Work, Under What Conditions

### 4.1 Simple, high-coherence directions can work for lower-dimensional behaviors

**Observed.** Refusal-like or safety-like dispositions appear especially amenable to low-dimensional steering [S10, S18]. This suggests that when a behavior is dominated by a relatively coherent geometric direction, a simple internal intervention can work quite well.

**Inference.** This likely explains why some steering papers appear much more positive than others: they are not necessarily methodologically better; they may simply be operating in easier behavior classes.

### 4.2 Feature-selection quality matters at least as much as feature detectability

**Observed.** Arad et al. show that selecting SAE features by what activates on inputs can be misleading, whereas output-score filtering yields much stronger steering [S12]. ITI similarly shows that not all natural probe-derived directions are equally useful for intervention [S05].

**Inference.** “Where the signal is” and “what intervention direction/operator is behaviorally effective” are distinct selection problems.

### 4.3 Control that is matched to the target format is often stronger

**Observed.** Opiełka et al. show that function vectors are powerful when extracted and applied on matching formats, but weaker out of format; more invariant concept vectors generalize better but with smaller in-distribution gains [S20].

**Practical implication.** A benchmark-local intervention may look strong because it is well matched to that surface, not because it discovered a robust causal control handle.

### 4.4 Structured or gated approaches can outperform cruder vector addition

**Observed.** MAT-Steer and Spherical Steering are both positive reminders that intervention design matters, not just target selection [S19, S23].

**Implication.** A paper arguing against naive target-selection heuristics should avoid sounding anti-steering in general; stronger operators can rescue some setups.

---

## 5. What Is Mixed, Benchmark-Contingent, or Easy to Overstate

### 5.1 “Detection and steering are separate” is true, but not in one single way

This claim survives, but only after being narrowed. The literature does **not** say one universal thing:
- Arad is about input-vs-output features within SAEs [S12].
- AxBench is about benchmark-level separation of detection vs steering on synthetic concepts [S13].
- Hase is about localization vs editing in weight-editing [S04].
- ITI is about direction choice and surface mismatch in truthfulness steering [S05].
- Bhalla is about interpretability/control evaluation across multiple method families [S08].

**Bottom line:** the field converges on a family resemblance — predictive signal quality alone is insufficient — but the exact mechanisms differ.

### 5.2 “MC doesn’t transfer to generation” is strong but still not universal

The literature strongly supports this as a risk, not as a law. Some methods may generalize better than others; some tasks are more format-sensitive than others; some format mismatches are more severe than others [S05–S07, S20].

**Bottom line:** safe to say “not guaranteed” and “often fragile,” unsafe to say “MC success never transfers.”

### 5.3 H-Neurons are highly relevant but still preprint evidence

H-Neurons are directly relevant and unusually close to the target model/benchmark family [S17]. But the paper is still a preprint. It deserves real weight, but not the same weight as an established benchmark paper plus independent replication.

**Bottom line:** use as a serious counterexample and near-neighbor, with an explicit preprint caveat.

### 5.4 Yap 2026 is relevant but should not dominate the arbitration

The Yap preprint is real and conceptually interesting because it shows similar probe quality with different steering utility in a large MoE setting [S21]. But it is single-author, very recent, uses a different behavioral domain, and has not yet accumulated the credibility of multi-author, benchmark-heavy, or peer-reviewed work.

**Bottom line:** cite only if making a broader “even more recent adjacent evidence exists” point; do not let it carry the paper’s core novelty boundary by itself.

---

## 6. Failure Modes, Contradictions, and Boundary Conditions

### 6.1 The slogan overstates what the literature proves

“Detection Is Not Enough” is directionally fine, but only if the paper makes clear:
- enough for **what**,
- on **which** evaluation surface,
- with **which** intervention operator,
- under **which** measurement regime,
- and relative to **which** counterexamples.

Otherwise the title invites a stronger reading than the literature supports.

### 6.2 A matched detector comparison must control more than readout AUROC/F1

If the flagship paper’s core experiment is “matched detection quality, divergent steering,” then reviewers will ask whether the match was real or cosmetic. At minimum the paper must control or transparently report:
- readout class and capacity,
- dataset and split,
- layer budget,
- operator family,
- intervention strength,
- token positions / timing,
- and evaluation parser/judge settings.

Without this, the result can be dismissed as a mismatch in intervention budget rather than a mismatch between detection and control.

### 6.3 Positive counterexamples are part of the claim, not an appendix inconvenience

The title and framing will be much stronger if the paper itself says:
- some detector-selected targets do work,
- especially for lower-dimensional or more coherent behaviors,
- but that success does not validate detector-quality as a general steering heuristic.

That is both truer and rhetorically stronger. A paper that ignores counterexamples looks ideological; a paper that incorporates them looks scientific.

### 6.4 Measurement errors can masquerade as control success or control failure

The generic measurement literature shows:
- style artifacts can sway judge verdicts [S14],
- guidelines can slash nominal ASR [S15],
- false negative rates can move dramatically under superficial variations [S16],
- and automated graders can systematically overstate success [S09].

**Inference:** in the proposed paper’s setting, an intervention might appear to “help” merely by shortening, stylizing, or sanitizing outputs in ways a judge/parser prefers. Conversely, a useful intervention might be underrated by a brittle parser or binary metric. This is exactly why measurement deserves stage-1 status.

---

## 7. Reproducibility and Implementation Reality Check

### 7.1 The literature is ahead on slogans and behind on controlled cross-method comparisons

There is now a lot of language about “predict/control discrepancy,” “feature steering,” and “benchmark-local effects,” but much less truly commensurable evidence across:
- the same model,
- the same benchmark,
- the same intervention budget,
- and different localization objects.

That gap is exactly where a good flagship paper can still contribute.

### 7.2 Preprints matter here, but they should be transparently downweighted

Several of the most relevant recent sources are preprints (H-Neurons, GuidedBench, Know Thy Judge, Yap, Spherical Steering, BRIDGE) [S15–S17, S21–S23]. In fast-moving AI, ignoring them would be foolish. But treating them as equivalent to an ACL/EMNLP/NeurIPS paper would also be foolish.

### 7.3 Evaluation plumbing is part of the method

For this topic, parser choice, fallback judging, truncation, temperature, and decoding regime are not incidental implementation details. They are often part of the causal story. Any serious paper here should treat them as part of the experiment, not as supplemental appendix debris.

---

## 8. Practical Implications for This Project or Research Direction

### 8.1 The safest flagship claim

The safest high-value claim is:

> **We show that strong readouts are not reliable steering-target heuristics across representation families, and that intervention claims must be validated separately at the stages of measurement, localization, control, and externality.**

This is substantially safer than:
- “detectors fail,”
- “we discovered detection doesn’t imply control,”
- or “our evaluation method is novel.”

### 8.2 What the intro should and should not say

### Safe
- Building on prior work showing that decodability does not imply causal use, we ask whether comparable readout strength predicts steering success across distinct representation families.
- We provide controlled evidence that feature types with matched detection quality can exhibit sharply different steering behavior on the same benchmark surface.
- We propose a four-stage evaluation scaffold — measurement, localization, control, and externality — and show why passing one stage does not guarantee passing the next.

### Stronger but still plausible
- We present a cross-feature-type matched comparison suggesting that readout quality alone is an insufficient steering-target criterion, even within a single model and evaluation surface.
- We show that benchmark-local control can fail to transfer to open-ended generation and may induce task-specific externalities.

### Avoid
- We discover that detection does not imply causal relevance.
- We are the first to show that strong readouts fail as steering targets.
- Prior work assumed that good detectors are good controllers.
- Judge dependence / graded-vs-binary scoring is a new finding.

### 8.3 Suggested related-work architecture for the flagship paper

The flagship paper’s related work should probably be organized into four short subsections:
1. **Probing and causal-use caution** — S01–S03.
2. **Target selection for steering / intervention** — S05, S08, S12, S13, with S04 as adjacent analogical support.
3. **Surface transfer and steering reliability** — S06, S07, S20.
4. **Measurement fragility** — S09, S14–S16.

Then add a short **boundary conditions / positive counterexamples** paragraph using S10, S17–S19.

---

## 9. Highest-Value Next Experiments, Evaluations, or Validation Steps

1. **Matched-readout cross-feature ablation**
   - Same model, same benchmark slice, same readout class, same amount of supervision.
   - Compare SAE features, H-neurons, and possibly probe-selected directions.
   - Match by detection metric *and* report operator budget.
   - This is the flagship’s cleanest novelty slot.

2. **Operator-matching ablation**
   - Hold localization object fixed, vary operator.
   - Hold operator fixed, vary localization object.
   - This helps separate “bad target” from “bad operator.”

3. **Surface-transfer grid**
   - Evaluate each intervention on:
     - the training/control surface (e.g., TruthfulQA MC),
     - open-ended generation,
     - harder OOD generation,
     - and at least one faithfulness/context benchmark like FaithEval.
   - This prevents one-surface success from standing in for robust control.

4. **Measurement-stability audit**
   - Binary vs graded scoring.
   - Rule-based parser vs LLM judge vs human sample.
   - Judge/model swap.
   - Truncation/output-length sensitivity.
   - This is essential if the paper wants to make any strong safety or harmfulness claims.

5. **Boundary-condition experiment**
   - Include at least one behavior where detector-linked control is known to work relatively well (e.g., refusal-like behavior).
   - This will sharpen the paper’s claim from “detectors fail” to “heuristic reliability depends on stage and task class.”

6. **Replication across at least one more model family**
   - Even a lightweight second model greatly improves external validity.
   - Without it, the paper remains a strong case study, but reviewers will reasonably call it model-specific.

---

## 10. Open Questions and Evidence Gaps

1. Which behaviors are intrinsically low-dimensional enough that detector-linked steering works reliably?
2. When target selection fails, is the bottleneck usually:
   - wrong feature,
   - wrong operator,
   - wrong timing/token location,
   - or wrong evaluation surface?
3. How much of the apparent detector→control gap is actually a detector→operator mismatch?
4. Are H-Neurons special because neurons are a better object, because the behavior family is more coherent, or because the intervention was better matched?
5. What is the best invariant evaluation surface for truthfulness steering, if any?
6. How often do measurement artifacts change the *ranking* of steering methods, not just their absolute scores?

---

## 11. Appendix — Claim-to-Source Notes (condensed)

- **Weak background claim occupied:** S01–S03.
- **Closest direct steering-era precedents:** S05, S08, S12, S13.
- **Best localization→control analogical support:** S04.
- **Best direct control→externality support:** S05, S07, S20.
- **Best measurement→conclusion support:** S09, S14–S16.
- **Strongest counterexamples:** S10, S17–S19.
- **Most important recent-but-lower-weight adjacent sources:** S21–S23.

---

## Short paper-ready positioning paragraph

Prior work has already established that decodability does not imply causal use, and recent steering work further shows that internal features selected for prediction are often poor guides for control [S01–S03, S05, S12, S13]. Our contribution is not to rediscover that gap in the abstract, but to characterize it under a stricter evaluation regime. We compare distinct representation families under matched detection quality on the same behavioral surface, then audit whether apparent control survives two further tests: transfer to more realistic generation settings and robustness to measurement choices. This motivates a stage-gate methodology for representation engineering — measurement, localization, control, and externality — in which success at one stage does not license conclusions about the next.


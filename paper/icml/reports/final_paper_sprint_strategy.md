# Final Paper Sprint Strategy Memo — Revised Grounded Version

**Strategy date:** May 7, 2026  
**Role:** external scientific strategy arbiter  
**Recommended center of gravity:** **Gemma-first gate-based audit of the readout-to-steering inference**, with Mistral and SIMID used only as stress-test / failed-gate evidence.  
**Package check:** requested files `01`–`10` were present. Machine-readable extras `11`–`13` were used for exact cross-checks where useful.

## 1. Verdict

The strongest honest framing is **not** “we found the right steering levers,” “SAEs fail,” or “Mistral replication.” It is: **a Gemma-3-4B-IT audit showing that readout, control, externality, and measurement validity are separable gates, and that skipping any gate licenses false claims.** The paper should be sold as a rigorous case study of the readout-to-steering inference, not as a broad mechanistic law.

The most important manuscript change is to make every claim explicitly conditional on **model, surface, operator, endpoint, and gate status**. The current draft is already close; the final sprint should harden it by adding a Mistral stress-test appendix/limitations row and by removing language that implies cross-model closure or truthfulness improvement.

The strongest claim is Gemma-local: H-neuron readout and SAE readout are comparable on FaithEval, but only the H-neuron intervention passes the FaithEval behavioral-control gate. The SAE utility and answer-span selectors deepen this result by showing metric-specific margin movement without behavioral compliance movement. The bridge result remains the cleanest externality finding: TruthfulQA MC gains do not imply open-generation gains.

Demote SIMID to a diagnostic-only omission/appendix note. Present Mistral as **surface-divergent and gate-incomplete**: FaithEval null, TruthfulQA MC1 gate failed, JailbreakBench strongly curved but uncontrolled. No new GPU/API experiment is worth doing before submission; manuscript hardening dominates.

## 2. Evidence State

| Evidence area | Status | What it supports | What it does not support | Paper placement |
|---|---|---|---|---|
| Gemma H-neuron FaithEval | **Promote** | Claim-bearing Gemma FaithEval control: 38 positive FFN neurons from 348,160, held-out AUROC 84.3% `[81.5,87.0]`, compliance slope +2.09 pp/alpha `[+1.4,+2.8]`, no-op to max +4.5 pp `[+2.9,+6.1]`, random controls flat. | Truthfulness improvement, architecture-general H-neuron control, or broad hallucination mitigation. FaithEval here is a context-faithfulness / anti-compliance surface. | Main result and Figure 2; keep as the primary positive control gate. |
| Gemma SAE comparable-readout and SAE selection | **Promote with caveat** | Comparable SAE readout AUROC 84.8%; 509 nonzero and 266 positive SAE features. H-neuron-vs-SAE slope gap +1.93 pp/alpha `[+0.9,+2.9]`. Utility selector moves prompt-end margin by -0.76 nats vs no-op and -1.68 nats vs readout; answer-span selector moves its own margin by -0.33 nats vs no-op and -0.53 nats vs random seed-mean; all compliance endpoints remain null. | “SAEs do not steer,” “SAE features are useless,” or “feature selection is solved.” The evidence is limited to the existing Gemma SAE layers, candidate pool, and delta/zeroing/scaling operators. | Main localization section plus appendix selector table. This is the strongest sharpening after the last update. |
| TruthfulQA / TriviaQA bridge externality | **Promote** | Gemma ITI improves constrained TruthfulQA MC1 by +6.3 pp `[+3.7,+9.0]` and MC2 by +7.5 pp `[+5.3,+9.8]`, but reduces TriviaQA bridge adjudicated accuracy by -5.8 pp `[-8.8,-3.0]` with 43 right-to-wrong vs 14 wrong-to-right flips; wrong-entity substitution is 72.1% `[57.3,83.3]` of R→W coded cases. | General truthfulness improvement; mechanistic claim that a substitution circuit was identified; direct replication of ITI’s original generation result. | Main externality result. Replace “replicates Li et al.” with “extends / echoes the TruthfulQA tradeoff concern.” |
| SimpleQA / BioASQ stress surfaces | **Preserve** | Surface-locality: SimpleQA attempt rate drops -32.7 pp and correct/compliance rate drops -1.8 pp under ITI; BioASQ H-neuron endpoint is flat (-0.1 pp). | A unified theory of factuality or capability preservation. | Supporting main/appendix; do not make them headline. |
| Jailbreak measurement sensitivity | **Preserve** | Gemma measurement lesson: binary endpoint is underpowered/ambiguous (+3.0 pp `[-1.2,+7.2]`), while grader choice and truncation/scoring granularity change the apparent verdict; v2 H slope +2.30 pp/alpha, random mean -0.47 pp/alpha, permutation p=0.013. | Settled jailbreak ground truth; universal claim that 256-token scoring is always invalid; high-confidence multi-seed specificity. | Main measurement section. Add nuance that Mistral anchor 3 behaves differently under truncation, so the measurement lesson is construct- and model-dependent. |
| SIMID calibration/prospective labels | **Drop from main; Appendix only if needed** | Measurement-governance evidence: historical open-label calibration failed the pre-specified kappa gate (402 cases, raw 91.5%, kappa 0.7594 < 0.8). Prospective Opus calibration passed for future grading, but the partial selected/TruthfulQA alpha 8 vs 0 effect missed the primary gate: +3.96 pp `[-1.98,+9.91]`, below +5 pp and lower-CI-positive requirements; same-scope MC accuracy degraded -3.08 pp `[-5.95,-0.66]`; attempt rate dropped -8.59 pp. | SIMID/ITI truthfulness improvement, retrospective upgrade of historical MVP labels, or selected-ITI specificity. | Prefer omission. If included, one appendix row explaining why it is not claim-bearing. |
| Mistral anchor 1 FaithEval CP5/H1 | **Limitations** | Clean stress test that readout quality does not guarantee FaithEval steering on another checkpoint/operator path: Mistral CP3 held-out AUROC 0.8711 `[0.8185,0.9172]`, accuracy 0.775 `[0.715,0.830]`; CP5 alpha 0→3 endpoint 0.0 pp `[-4.01,+4.00]` with 9/9 paired flips; H1 C=0.75 endpoint +0.5 pp `[-3.0,+4.0]`. | Cross-model replication, Mistral L1 closure, explanation of why Gemma differs. | Limitations plus appendix stress-test table; not abstract by default. |
| Mistral anchor 2 TruthfulQA MC | **Limitations** | Weak, failed-gate Mistral ITI result: MC1 58/163→60/163, +1.23 pp `[-1.84,+4.29]`, McNemar p=0.6875; MC2 truthful mass +1.91 pp `[+0.50,+3.40]`; wrapper correctly stopped before bridge. | Successful Mistral TruthfulQA transfer, confirmatory bridge launch, or Mistral MC-positive/generation-negative externality. | Appendix/limitations only. Use to justify not reporting a Mistral bridge result. |
| Mistral anchor 3 JailbreakBench measurement | **Appendix** | Strong Mistral safety-surface curve across four judges, all 2000/2000 valid: binary-256 +32.8 pp, binary-full +32.6 pp, CSV-v3 +32.4 pp, StrongREJECT +31.4 pp from alpha 0 to 3. CSV-v3 and StrongREJECT agree on 91.8% of same-output rows; kappa 0.836; StrongREJECT is 5.7 pp more permissive. | H-neuron specificity, mechanistic control, or model-general jailbreak result; no matched random/layer controls. | Appendix measurement stress test. It is too large to hide, but too uncontrolled to headline. |

## 3. Best Framing

### Thesis

**Predictive internal readouts are useful audit signals, but a steering claim requires separate evidence for measurement validity, target-surface control, and externality; in the provided evidence, those gates repeatedly separate.**

### Why this framing is scientifically mature

The literature already contains strong reasons not to infer causality from probes. Control-task and amnesic-probing work warned that high probe accuracy does not imply behavioral use; localization-vs-editing work found that where a fact is localized may not predict where editing works. The SAE literature is also no longer aligned with a crude “SAEs steer” or “SAEs fail” binary: recent SAE steering work argues feature selection can make SAEs effective, while other work reports only weak interpretability–utility association. The paper’s novelty should therefore be **narrower and stronger**: a real behavioral-surface audit where H-neurons and SAE features are compared under similar readout quality, followed by target-selection and externality checks that prevent the tempting overclaim.

The best center of gravity remains **Gemma comparative audit with Mistral stress-test limitations**. The plural-anchor framing would overstate symmetry: Mistral FaithEval failed, Mistral TruthfulQA MC1 failed, Mistral JBB lacks controls, and SIMID failed claim gates. A measurement-first reframing would bury the cleanest mechanistic contrast. The paper should instead say: “Here is what it looks like to audit the inference from readout to steering without flattening gates.”

### Safe abstract-level replacement

> Predictive internal signals are often used as steering targets, but readout quality alone does not establish behavioral control. We audit this inference in Gemma-3-4B-IT across contextual faithfulness, multiple-choice truthfulness, open-ended QA, and jailbreak evaluation. On FaithEval, H-neurons and SAE features achieve similar held-out readout quality, yet only H-neuron scaling produces a reliable behavioral dose-response under our intervention operators. Within the same SAE candidate pool, prompt-end utility and answer-span selectors produce metric-specific margin shifts but do not recover the compliance endpoint. ITI improves TruthfulQA multiple-choice answer selection while reducing open-ended TriviaQA bridge accuracy, most often through wrong-entity substitution. Jailbreak conclusions depend on scoring granularity and evaluator construct. We synthesize these results as a gate-based audit separating measurement, localization, control, and externality.

Do not put Mistral in the abstract unless Section 1 is also rewritten to say the paper includes appendix-only stress tests. If added, use one restrained sentence: “Appendix stress tests on Mistral-Small-24B-Instruct-2501 show that these gates can fail or remain incomplete on a second checkpoint.”

### Contribution list replacement

1. **Readout-to-control audit:** a Gemma FaithEval comparison showing that similar held-out readout quality for H-neurons and SAE features does not imply similar behavioral control.
2. **Within-SAE selector audit:** a target-selection ablation showing that readout, prompt-end utility, and answer-span utility produce different margin effects inside the same candidate pool while all fail the behavioral endpoint.
3. **Control-to-externality audit:** an ITI study showing that TruthfulQA MC gains do not license open-generation truthfulness claims, with a bridge failure taxonomy dominated by wrong-entity substitution.
4. **Measurement audit:** a jailbreak evaluation showing that truncation, scoring granularity, and evaluator construct choices materially affect verdicts.
5. **Claim-discipline framework:** a four-gate checklist—measurement, localization, control, externality—for reporting activation-steering claims.

### Title and scope

The current title, **“Strong Readouts, Local Levers,”** is acceptable. A slightly safer subtitle would be: **“A Gate-Based Audit of Activation-Steering Claims.”** Avoid any title implying “neurons beat SAEs,” “SAEs fail,” or “truthfulness steering.”

### What belongs only in limitations or appendix

Mistral belongs in a compact appendix/limitations table, not as a second-model replication. SIMID belongs nowhere in the main paper unless the authors need a one-row “excluded diagnostic” appendix for internal completeness. Anchor 3 JBB should be included only if paired with the no-specificity caveat, because the effect size is large enough that omission could look selective if reviewers see the repo lineage.

## 4. Claim Discipline

| Current or tempting claim | Verdict | Safer claim | Evidence anchor |
|---|---|---|---|
| “Strong readouts imply steering targets.” | **Remove** | Strong readouts identify candidate audit targets; steering requires a direct intervention/control gate on the same behavioral surface. | Gemma H vs SAE FaithEval; Mistral CP3 vs CP5/H1. |
| “H-neurons are better steering targets than SAE features.” | **Demote** | Under these Gemma FaithEval operators and layers, H-neuron scaling passes the behavioral gate while tested SAE feature interventions do not. | FaithEval core + SAE variants. |
| “SAE features do not steer.” | **Remove** | No tested selector inside this Gemma SAE candidate pool recovered FaithEval compliance; metric-level margin effects exist and other SAE selection/operator choices remain possible. | SAE utility and answer-span reports; external SAE literature. |
| “Utility selector solves SAE target selection.” | **Remove** | Utility selection beats readout on the prompt-end margin but not on compliance. It is a diagnostic target-selection improvement, not behavioral steering. | Utility - readout margin -1.68 nats; compliance CIs include 0. |
| “Answer-span selector supports generation steering.” | **Remove** | Answer-span selection generalizes on its own first-3-token answer margin but remains null on compliance and is worse than prompt-end utility on the original prompt-end margin. | Answer-span - no-op -0.33 nats; answer-span - utility +0.46 nats on anti-compliance margin. |
| “SIMID improves truthfulness.” | **Remove** | SIMID is diagnostic-only: historical calibration failed the pre-specified kappa gate; partial prospective effect missed the primary gate and showed MC/attempt degradation. | 07/08 reports and JSON 12. |
| “Mistral closes L1.” | **Remove** | Mistral is surface-divergent and gate-incomplete: FaithEval null, TruthfulQA MC1 failed, JBB curve uncontrolled. | 03/09/10 reports. |
| “Mistral replicates Gemma steering.” | **Remove** | Mistral 2501 does not replicate the FaithEval H-neuron intervention gate. It constrains generalization and supports the framework, not the Gemma effect. | CP5 0.0 pp; H1 +0.5 pp. |
| “Anchor 3 is H-neuron-specific.” | **Remove** | Anchor 3 shows a large same-output Mistral JBB alpha curve across evaluators, without random/layer-matched controls. | 09/11 reports. |
| “TruthfulQA gains transfer to bridge generation.” | **Reverse** | Gemma ITI TruthfulQA MC gains are accompanied by negative open-generation externalities on bridge/SimpleQA; Mistral did not pass the MC1 source gate, so no Mistral bridge confirmation exists. | TruthfulQA MC + Bridge metrics; 10 report. |
| “Jailbreak measurement is settled.” | **Demote** | The paper can argue that measurement choices matter and that full-output/graded scoring is better matched to the Gemma construct; it cannot declare a universal jailbreak ground truth. | Gemma jailbreak audit; Mistral anchor3 judge comparison. |
| “Wrong-entity substitution is a mechanism.” | **Remove** | Wrong-entity substitution is a behavioral failure taxonomy. The margin check is consistent with directional log-likelihood shifts but does not localize a substitution circuit. | Bridge IRR and margin table. |

## 5. Final Sprint Plan

| Priority | Work item | Files/sections affected | Acceptance check | Stop rule |
|---:|---|---|---|---|
| 1 | **Manuscript writer:** replace abstract/contributions with gate-local language. | `01_main.tex` abstract, Introduction contributions. | Every effect names model/surface/operator; no sentence says or implies SAE impossibility, truthfulness improvement, Mistral replication, or readout sufficiency. | Stop once abstract is Gemma-first and surface-local. Do not add Mistral abstract sentence unless Section 1 is also revised. |
| 2 | **Scientific editor:** add Mistral appendix/limitations mini-table. | Limitations; appendix claim-defense table. | Includes exact checkpoint `Mistral-Small-24B-Instruct-2501`; FaithEval CP3/CP5/H1 numbers; anchor2 MC1 failed gate; anchor3 large uncontrolled JBB curve. | One paragraph plus one table maximum. No new Mistral narrative thread. |
| 3 | **Results editor:** harden SAE wording. | §Localization, Appendix SAE selector, limitations L2/L3. | Says “inside existing 509-feature Gemma SAE pool and tested operators”; separates margin movement from behavioral compliance; mentions answer-span cross-metric tradeoff. | No additional SAE seeds, no Mistral SAE, no layer-expansion claim. |
| 4 | **Results editor:** revise TruthfulQA/bridge language. | §Externality and bridge figure captions. | Replace “replicates ITI generation divergence” with “extends/echoes tradeoff concern”; distinguish MC likelihood, open generation, behavioral taxonomy, and teacher-forced margins. | Do not claim mechanism for wrong-entity substitution. |
| 5 | **Measurement editor:** refine jailbreak claims. | §Measurement, limitations, appendix. | Main claim is “measurement choices can change verdict,” not “our evaluator is ground truth.” If Mistral anchor3 appears, state it shows cross-judge curve robustness but unresolved specificity. | Do not run JBB controls pre-submission. Do not universalize Gemma truncation finding. |
| 6 | **Evidence guardian:** decide SIMID inclusion. | Appendix/limitations only, or nowhere. | If included, exactly one diagnostic row: calibration kappa failed; prospective partial effect missed gate and degraded MC/attempt. | Drop SIMID entirely if it costs more than 150 words or creates a new story. |
| 7 | **Stats verifier:** number audit. | Whole manuscript; captions; appendix tables. | All headline numbers match `02_metric_tables_only.md` or later direct reports. Later direct reports override stale strategy prose. | No new analysis unless a reported number conflicts internally. |
| 8 | **Appendix curator:** build claim-defense ledger. | Appendix. | Columns: claim, model, surface, metric ID/artifact, estimator/CI, gate status, caveat. | Lookup table only; no literature survey. |
| 9 | **Repro/hygiene:** verify artifact paths and current limitation labels. | Appendix provenance, limitation inventory. | Stale rows like “seeds pending” or “single model only” are updated after Mistral: primary claims still Gemma-only, but Mistral stress tests exist and failed/incomplete. | Do not block submission on code patches unless they affect a reported number. |
| 10 | **Final red-team pass:** forbidden-claim sweep. | Whole manuscript. | Search and remove: “Mistral replicated,” “Mistral closes L1,” “SIMID improves,” “SAEs fail,” “TruthfulQA transfer,” “jailbreak settled,” unqualified “truthfulness.” | If a claim needs more caveats than content, move it to appendix or delete it. |

Not worth doing before submission: Mistral bridge continuation, Mistral SAE/transcoder, another FaithEval C-grid, SIMID full labels, JBB matched controls, extra SimpleQA/BioASQ Mistral, broad new literature review. These may be useful post-submission, but they do not beat manuscript hardening under the deadline.

## 6. Reviewer Objections

1. **“The main causal story is one model.”**  
Fair. The paper should not deny this. Best response: the primary claims are Gemma-local; Mistral appears only as a failed/incomplete stress test showing that the framework detects non-transfer. Land in Introduction study orientation, limitations, and Mistral appendix table.

2. **“Comparable AUROC is not comparable causal localization.”**  
Fair. AUROC parity does not equal identical representation, intervention operator, layer coverage, or causal role. Best response: concede the cross-family confound and promote the within-SAE selector audit as a partial control: even inside one candidate pool/operator, metric-specific selectors fail the behavioral endpoint. Land in §Localization synthesis and L2/L3 limitation rows.

3. **“The SAE null is probably a bad selector.”**  
Partly fair, but now less damaging. Readout-only selection is no longer the only tested SAE selection rule; utility and answer-span selectors move their intended margins and still miss compliance. Best response: “bad selector” remains possible outside the tested pool/operators, but not as an explanation for the original readout-selection artifact alone. Land in Appendix SAE selector table.

4. **“Your H-neuron result is modest and benchmark-specific.”**  
Fair. +4.5 pp from no-op to max is reliable, not transformative. Best response: the claim is not a new mitigation method; it is a positive control gate demonstrating that one readout family produces behavioral movement on FaithEval while comparable SAE readouts do not. Land in main result wording and conclusion.

5. **“TruthfulQA MC is not truthfulness.”**  
Fair and central. TruthfulQA’s own later discussion of multiple-choice variants reinforces that MC should not be treated as open-ended truthfulness. Best response: the paper’s externality result is precisely that constrained answer-selection gains do not license generation claims. Land in §Externality and abstract.

6. **“Wrong-entity substitution is behavioral coding, not mechanism.”**  
Fair. Best response: call it a failure taxonomy; keep the teacher-forced margin check as a consistency check, not causal localization. Land in bridge paragraph and appendix margin table.

7. **“LLM judges and jailbreak rubrics are unstable.”**  
Fair. Best response: measurement instability is part of the paper’s claim. Full-output scoring, evaluator holdout, CSV-v3/StrongREJECT comparisons, and construct-specific language should be presented as mitigation, not proof of ground truth. Land in §Measurement and evaluator table.

8. **“Anchor 3 Mistral JBB is large—why is it appendix only?”**  
Fair. Best response: effect size is large but no matched random/layer controls exist; therefore it is measurement/surface-curve evidence, not H-neuron specificity. Land in Mistral appendix table and limitations.

9. **“Why omit SIMID if it was run?”**  
Fair if reviewers see broader artifacts. Best response: SIMID failed the paper’s own claim gates: historical kappa below threshold, partial prospective effect below primary gate, MC and attempt degradation. Excluding it is a claim-discipline decision. Land only in optional omitted-diagnostics appendix row.

10. **“The paper’s novelty is crowded by probe critiques and recent SAE steering work.”**  
Fair. Best response: do not claim discovery of readout/control divergence in general. Claim a stricter empirical audit: same behavioral surface, comparable readout quality, matched intervention controls, within-SAE selector ablation, externality analysis, and measurement audit. Land in Related Work and final paragraph of Introduction.

## 7. Submission-Ready Checklist

1. Lock the headline: **Gemma-first gate-based audit, not multi-model replication.**
2. Replace abstract/contribution language with surface-local claims.
3. Update limitations: primary claims remain Gemma-local; Mistral stress tests are present but failed/incomplete, not absent.
4. Add Mistral appendix table with CP5/H1 null, anchor2 failed MC1 gate, anchor3 uncontrolled JBB curve.
5. Keep SIMID out of the main text; add at most one diagnostic appendix row.
6. Ensure SAE wording says “tested Gemma SAE pool/operators/layers,” not “SAEs cannot steer.”
7. Ensure answer-span wording says “within-metric generalization,” not “generation steering.”
8. Ensure bridge wording says “behavioral wrong-entity substitution,” not “mechanistic substitution circuit.”
9. Ensure TruthfulQA wording says “MC answer selection,” not unqualified “truthfulness.”
10. Ensure jailbreak wording says “measurement construct and evaluator choices matter,” not “settled harmfulness ground truth.”
11. Verify all numbers against `02_metric_tables_only.md`, then later direct reports `05`–`13` where they supersede the table.
12. Search for and remove forbidden phrases: “Mistral replicates,” “Mistral closes L1,” “SIMID improves truthfulness,” “SAE features do not steer,” “TruthfulQA gains transfer,” “jailbreak measurement is settled.”
13. Submit with a claim-defense ledger: claim, model, surface, metric/artifact, CI, gate, caveat.

## External literature checked for framing

This literature check affects novelty and reviewer posture, not local numeric claims. Probe critiques motivate the paper’s gate discipline: Hewitt & Liang (2019) on probe selectivity/control tasks, Elazar et al. (2021) on amnesic probing and behavioral use, and Hase et al. (2023) on localization not predicting editing. Steering context comes from ITI (Li et al., 2023/2024), Activation Addition / activation engineering (Turner et al., 2023/2024), H-Neurons (Gao et al., 2025), and the 2026 cross-domain H-neuron generalization paper reporting diagnostic transfer gaps and null activation-scaling effects on factual hallucination. SAE positioning should acknowledge Gemma Scope (Lieberum et al., 2024), “SAEs Are Good for Steering — If You Select the Right Features” (Arad et al., 2025), and interpretability–utility gap work on 90 SAEs (Wang et al., 2025). Measurement positioning should cite FaithEval, TruthfulQA, JailbreakBench, and StrongREJECT; StrongREJECT specifically supports the claim that benchmark/evaluator choice can overstate jailbreak effectiveness. The manuscript should cite these works only where it changes claim scope; do not expand into a field survey.

## Local evidence authority used

- `01_main.tex`: current manuscript claim surface, especially abstract, contribution list, localization section, externality section, measurement section, limitations, and claim ledger.
- `02_metric_tables_only.md`: primary authority for Gemma readout, FaithEval, SAE, TruthfulQA, bridge, SimpleQA, BioASQ, and jailbreak numbers.
- `03_mistral24b_l1_strategy.md`, `09_mistral_anchor3_jailbreak_measurement_review.md`, `10_mistral_anchor2_truthfulqa_mc_review.md`, and JSON `11`: authority for current Mistral state.
- `05_faitheval_sae_utility_selector_review.md`, `06_faitheval_answer_span_extension.md`, and JSON `13`: authority for SAE selector and margin/compliance claims.
- `07_simid_open_calibration_review.md`, `08_simid_prospective_partial_external_label_review.md`, and JSON `12`: authority for SIMID diagnostic-only status.

## External references checked

- Hewitt & Liang, “Designing and Interpreting Probes with Control Tasks,” EMNLP-IJCNLP 2019: https://aclanthology.org/D19-1275/
- Elazar et al., “Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals,” TACL 2021 / arXiv: https://arxiv.org/abs/2006.00995
- Hase et al., “Does Localization Inform Editing?”, arXiv 2023: https://arxiv.org/abs/2301.04213
- Li et al., “Inference-Time Intervention: Eliciting Truthful Answers from a Language Model,” NeurIPS 2023 / arXiv: https://arxiv.org/abs/2306.03341
- Turner et al., “Steering Language Models with Activation Engineering,” arXiv/OpenReview 2023–2024: https://arxiv.org/abs/2308.10248
- Gao et al., “H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs,” arXiv 2025: https://arxiv.org/abs/2512.01797
- “Do Hallucination Neurons Generalize? Evidence from Cross-Domain Transfer in LLMs,” arXiv 2026: https://arxiv.org/abs/2604.19765
- Lieberum et al., “Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2,” arXiv 2024: https://arxiv.org/abs/2408.05147
- Arad et al., “SAEs Are Good for Steering — If You Select the Right Features,” EMNLP 2025 / arXiv: https://arxiv.org/abs/2505.20063
- Wang et al., “Does higher interpretability imply better utility? A Pairwise Analysis on Sparse Autoencoders,” arXiv 2025: https://arxiv.org/abs/2510.03659
- Ming et al., “FaithEval,” ICLR 2025 / arXiv: https://arxiv.org/abs/2410.03727
- Lin et al., “TruthfulQA: Measuring How Models Mimic Human Falsehoods,” ACL 2022: https://aclanthology.org/2022.acl-long.229/
- TruthfulAI, “New, improved multiple-choice TruthfulQA,” 2025: https://truthful.ai/blog/truthfulqa-binary-choice/
- Souly et al., “A StrongREJECT for Empty Jailbreaks,” arXiv 2024: https://arxiv.org/abs/2402.10260
- Gemma Team, “Gemma 3 Technical Report,” arXiv 2025: https://arxiv.org/abs/2503.19786

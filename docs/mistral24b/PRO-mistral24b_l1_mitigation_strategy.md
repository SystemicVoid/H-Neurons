# Mistral 24B L1 Mitigation Strategy

**Strategy date:** 2026-04-28  
**Target artifact:** `mistral24b_l1_mitigation_strategy.md`

**Evidence tags.** `[observed]` = directly supported by attached artifacts or linked primary sources. `[inferred]` = strategic conclusion drawn from observed evidence. `[speculative]` = estimate or forecast. `[unsupported]` = claim type this plan refuses to make.

## 1. Verdict

[inferred] Execute **serial-strict, not parallel-aggressive**: freeze manifests/evaluators/alpha schedules first; run the Mistral tokenizer/template/activation smoke; train/evaluate the held-out H-neuron classifier; gate on FaithEval H-neuron dose-response with matched controls; then run TruthfulQA MC and TriviaQA bridge. [inferred] This overrides the April 21 memo’s paper-facing anchor order: bridge remains the cleanest Gemma anchor, but FaithEval/classifier must run first because it gates whether the Mistral migration and SAE question are even meaningful.

[inferred] SAE decision: **conditional reframe/drop on the critical path**. Do not spend claim-bearing GPU on a Mistral SAE until H-neurons replicate. No exact public Gemma-Scope-2-equivalent exists for `mistralai/Mistral-Small-24B-Instruct-2501`; the visible Mistral-family SAE candidate is a single-layer different-checkpoint artifact, not a matched substitute. [inferred] If H-neurons replicate and budget remains, train a narrow Mistral residual/MLP SAE or skip-transcoder under matched-readout controls; otherwise make SAE comparison explicitly Gemma-only.

[speculative] Biggest risk is not H100 cost; it is measurement confounding from tokenization, chat template, prompt style, and judge drift. [inferred] If the plan succeeds, the paper can claim that the H-neuron control gate and ITI externality gate are not Gemma-idiosyncratic; the cross-representational SAE dissociation remains Gemma-only unless the conditional SAE branch also passes.

## 2. Frozen Evidence Baseline

[observed] Missing attachment: `ground-truth/README.md` was not attached. This memo uses `metric_ledger.jsonl` and the five surface briefings. No Gemma numerical baselines are restated below.

| Anchor | Surface | Frozen artifact (ledger key) | MVL1 status | Mistral target metric |
|---|---|---|---|---|
| H-neuron held-out readout | Pipeline / detector | `readout.disjoint.auroc`; `readout.disjoint.accuracy`; `readout.pipeline.selected_h_neurons`; `readout.pipeline.total_ffn_neurons` | must-exact | Same disjoint split family, model-derived FFN geometry, held-out AUROC/accuracy sidecar, selected-neuron manifest |
| H-neuron target sparsity / geometry | Pipeline / detector | `readout.structure.band.early`; `readout.structure.band.middle`; `readout.structure.band.late` | must-directional | Layer distribution and target count recorded; no fixed Gemma pattern required |
| SAE readout parity | Pipeline / SAE | `readout.sae.auroc`; `readout.sae.accuracy`; `readout.sae.n_positive_features`; `readout.sae.layer_count` | Gemma-only | No MVL1 target unless conditional SAE branch runs; if run, use comparable held-out readout and same Mistral manifest |
| FaithEval H-neuron dose-response | FaithEval | `intervention.faitheval.anti.slope_pp_per_alpha`; `intervention.faitheval.anti.delta_noop_to_max_pp`; `intervention.faitheval.h_spearman` | must-exact | Same alpha grid first; slope, endpoint, monotonicity, item-level sidecar |
| FaithEval matched random specificity | FaithEval controls | `intervention.faitheval.mean_slope_difference_pp_per_alpha`; `intervention.faitheval.random_mean_slope_pp_per_alpha`; `intervention.faitheval.random_max_slope_pp_per_alpha`; `intervention.faitheval.seedwise_positive_differences` | must-exact | H-minus-random and H-minus-layer-matched contrasts on the same sample manifest |
| H-neuron-vs-SAE control dissociation | FaithEval / SAE | `intervention.faitheval_sae.neuron_minus_sae_slope_pp_per_alpha`; `intervention.faitheval_sae.h_slope_pp_per_alpha`; `intervention.faitheval_sae.random_mean_slope_pp_per_alpha` | Gemma-only | Conditional Mistral SAE branch only; otherwise scope wording says Gemma cross-representational comparison |
| Within-SAE selector ablation | FaithEval / SAE | `intervention.faitheval_sae_utility.utility_minus_readout_margin`; `intervention.faitheval_sae_utility.utility_minus_noop_compliance_pp`; `intervention.faitheval_sae_utility_positive.utility_positive_minus_noop_margin` | Gemma-only | No Mistral MVL1 target; optional if trained SAE exists and controls are pre-registered |
| FalseQA H-neuron canary | FalseQA | `intervention.falseqa.slope_pp_per_alpha`; `intervention.falseqa.delta_0_to_max_pp` | must-directional | Canary after FaithEval pass; sign/direction only, not a flagship metric |
| BioASQ surface-locality check | BioASQ | `intervention.bioasq.delta_0_to_max_pp`; `intervention.bioasq.changed_response_count`; `intervention.bioasq.slope_pp_per_alpha` | must-directional | Optional negative/flat capability check; run only after FaithEval if budget remains |
| TruthfulQA ITI source-surface gain | TruthfulQA MC | `transfer.truthfulqa_mc.mc1.iti.delta_pp`; `transfer.truthfulqa_mc.mc2.iti.delta_pp`; `transfer.truthfulqa_mc.mc1.iti.wrong_to_right`; `transfer.truthfulqa_mc.mc1.iti.right_to_wrong` | must-exact | Same held-out MC folds, alpha schedule, MC1/MC2 and flip accounting |
| TriviaQA bridge externality | TriviaQA bridge | `transfer.bridge.adjudicated_accuracy_delta_pp`; `transfer.bridge.deterministic_accuracy_delta_pp`; `transfer.bridge.base_correct_iti_wrong`; `transfer.bridge.base_wrong_iti_correct`; `transfer.bridge.mcnemar_p` | must-exact | Baseline/ITI paired generation, deterministic and adjudicated accuracy, flip table |
| Bridge failure taxonomy | TriviaQA bridge | `measurement.bridge_irr.right_to_wrong.wrong_entity_substitution`; `measurement.bridge_irr.right_to_wrong.formal_refusal`; `measurement.bridge_irr.cohen_kappa`; `measurement.bridge_irr.gwet_ac1` | must-directional | Discordant-case coding if enough flips; failure-mode proportions reported as Mistral-specific |
| Bridge log-likelihood margin check | TriviaQA bridge margins | `transfer.bridge_margins.a_rw_substitution.first3_shift_nats`; `transfer.bridge_margins.a_vs_d.first3_shift_nats_gap`; `transfer.bridge_margins.c_wr_rescue.first3_shift_nats` | must-directional | Optional teacher-forced margin audit after behavioral bridge pass |
| SimpleQA generation stress | SimpleQA | `transfer.simpleqa.compliance_delta_0_to_8_pp`; `transfer.simpleqa.attempt_delta_0_to_8_pp` | must-directional | Supporting open-generation check; bridge remains primary externality anchor |
| Jailbreak binary-vs-graded measurement sensitivity | JailbreakBench | `intervention.jailbreak.binary.delta_0_to_max_pp`; `measurement.jailbreak.v2.h_slope_csv2_yes_pp_per_alpha`; `measurement.jailbreak.v2.gap_h_minus_random_mean_pp_per_alpha`; `measurement.holdout.csv2v3.accuracy`; `measurement.holdout.sr.accuracy` | must-directional | Same-output binary/graded disagreement and evaluator-holdout sanity; not required before bridge |
| FaithEval standard prompt remap | FaithEval measurement | `measurement.faitheval.standard_raw.slope_pp_per_alpha`; `measurement.faitheval.standard_remap.parse_failures`; `measurement.faitheval.standard_remap.strict_recovered_count` | must-directional | Prompt-style audit and parser failure audit before final FaithEval claim |
| D7 causal-vs-probe panel | Jailbreak diagnostic | `mechanism.d7.current_panel.causal_vs_probe_gap_pp`; `mechanism.d7.current_panel.causal_vs_random_layer_seed1_gap_pp`; `mechanism.d7.current_panel.causal_token_cap_count` | Gemma-only | Defer; not MVL1-critical |
| Refusal overlap / swing / single-neuron diagnostics | Mechanism diagnostics | `mechanism.refusal_overlap.*`; `mechanism.swing.*`; `mechanism.neuron_4288.*`; `mechanism.verbosity.*` | Gemma-only | Defer; use only as scope caveats |

## 3. Critical Path

[inferred] The **hard gate for all H-neuron claims is Stage CP3: held-out Mistral classifier + answer-span/geometry audit**. The **empirical gate for the Mistral migration’s paper value is Stage CP5: FaithEval H-neuron specificity on the frozen full manifest**.

| Stage | Inputs | Output artifact | Decision criterion | Kill criterion | Expected H100-hours |
|---|---|---|---|---|---:|
| CP0. Pre-flight lock and adversarial audit | Current repo, registry, ledger mapping, draft outcome tree | `docs/mistral24b/preflight_lock.md`; committed manifests; DRY_RUN transcript | [observed/inferred] Every claim-bearing run has frozen sample manifest, alpha schedule, evaluator version, stop rules, and provenance sidecar | Any claim-bearing command depends on a floating prompt, floating judge, uncommitted sample set, or unspecified alpha schedule | 0 |
| CP1. Tokenizer/template/span smoke | `model_key=mistral_small_24b_instruct_2501`; `fix_mistral_regex=True`; small audited examples | `data/mistral24b/preflight/token_span_audit.jsonl`; memory trace | [observed] Answer-token spans, chat-template boundaries, and model geometry agree with registry; no quantization | Token spans fail manual audit; model loads through unsupported path; BF16 80GB run cannot complete at minimal batch | 0.2--0.8 |
| CP2. Canonical splits + activation extraction | `answer_tokens_llm.jsonl`; frozen split seed; exclusion paths | `train/dev/test_qids_llm.json`; `activations_llm_canonical/`; sidecars | [observed] Disjoint splits and activation tensors exist with model/hash provenance | Split leakage, feature-width mismatch, or missing non-answer activations | 1.5--4.5 |
| CP3. L1 classifier train/eval | CP2 activations; frozen C-selection rule | `models/mistral24b_classifier_canonical.pkl`; `classifier_canonical_*_metrics.json` | [observed/inferred] Held-out readout is nondegenerate, selected target set is finite and model-geometry-valid | AUROC/accuracy near chance, selected target set pathological, or train/dev/test inconsistency; stop H-neuron branch | 0.3--1.0 |
| CP4. FaithEval pilot + control smoke | CP3 classifier; small frozen FaithEval pilot manifest; one unconstrained and one layer-matched control | `faitheval_pilot_smoke/summary.json`; prompt/parser audit | [inferred] Intervention code, parser, and controls are not broken before full spend | Control path diverges from H path, prompt-style mismatch, evaluator/parser failure, or strong clean reversal | 0.5--1.5 |
| CP5. FaithEval full H-neuron + controls | Full frozen FaithEval manifest; Gemma alpha grid replicated first; 3 unconstrained + 3 layer-matched random controls | `data/mistral24b/intervention/faitheval/full_control_summary.json`; item-level outputs | [inferred] Same-sign H-neuron dose-response and H-minus-control specificity; CI rule sealed in CP0 | Clean null or reversal against matched controls; do not run conditional SAE as if H-neuron premise held | 2.0--6.0 |
| CP6. TruthfulQA MC ITI source-surface check | Frozen TruthfulQA folds; ITI fit path; alpha schedule replicated first | `data/mistral24b/transfer/truthfulqa_mc/summary.json` | [inferred] ITI source-surface improvement is present before claiming externality | MC source effect absent or template-confounded; bridge can be run only as exploratory, not MC-vs-generation evidence | 0.5--2.0 |
| CP7. TriviaQA bridge baseline/ITI | Frozen bridge manifest; CP6 ITI direction; adjudication protocol | `data/mistral24b/transfer/triviaqa_bridge/results.json`; flip table; adjudication sidecar | [inferred] Generation-side externality appears on paired bridge metrics and is not only refusal/format drift | Accuracy delta uninterpretable, adjudicator disagreement unresolved, or all harm explained by prompt/template artifacts | 1.5--5.0 |
| CP8. Bridge discordant coding + margin audit | CP7 discordants; precommitted rubric; optional teacher-forced margin plan | `bridge_failure_modes.jsonl`; optional `bridge_margin_summary.json` | [inferred] Failure taxonomy is reportable if discordants are sufficient and agreement rule passes | Too few discordants, unreliable coding, or margin audit contradicts behavioral framing without clear revision | 0.3--1.5 |

## 4. Parallel Tracks

| Track | Artifact produced | Decision unlocked | Earliest start | Owner-style |
|---|---|---|---|---|
| P1. Full sample-manifest construction | `manifests/{faitheval,truthfulqa,bridge,simpleqa,jailbreak}_mistral24b.lock.json` with hashes | Prevents post-hoc sample selection; enables exact reruns | Now | manifest-build |
| P2. Alpha-schedule lock | `alpha_schedule_lock.md` specifying replicate-first grids and no extension until after primary pass/fail | Blocks alpha tuning on Mistral outcomes | Now | doc-only |
| P3. Evaluator freeze and rubric diff | `judge_lock.md`; dated model IDs; rubrics; JSON schemas; tie-break rules | Allows reviewer-grade judge reproducibility and drift audit | Now | judge-engineering |
| P4. FaithEval prompt-style and parser audit | `faitheval_prompt_style_audit.md`; standard-vs-anti examples; parser failure table | Decides whether Mistral uses `standard` only or requires a reported style sensitivity appendix | Now | judge-engineering |
| P5. Mistral-specific code review | PR/audit memo over registry resolution, `fix_mistral_regex`, classifier path defaults, geometry guards, sample_manifest plumbing | Decides whether CP1 can start without hidden Gemma fallbacks | Now | code-review |
| P6. Wrapper DRY_RUN rehearsal | `runpod_dry_run_transcript.txt`; command list; environment capture template | Decides launch readiness without burning GPU | Now | code-review |
| P7. Provenance sidecar templates | `provenance_schema_mistral24b.json`; per-stage sidecar stubs | Makes each result attachable to manuscript/rebuttal | Now | doc-only |
| P8. Judge recalibration mini-set | Mistral mini holdout with human/LLM labels and CSV-v3/StrongREJECT comparison | Decides whether Gemma evaluator hierarchy transfers to Mistral outputs | After P3; no dependency on CP3 | judge-engineering |
| P9. Conditional SAE design memo | `mistral_sae_branch_spec.md` with layer choice, training corpus, matched-readout controls, dead-feature/path controls | Decides whether SAE branch is allowed after CP5 | Now | doc-only |
| P10. Human bridge-rater pilot | Human-rater instructions, blind labels, adjudication rule, cost/time log | Decides whether L4 can be downgraded before submission | Now | judge-engineering |
| P11. L5 jailbreak seed-pack prep | Frozen seed list, manifests, judge lock, kill criteria | Allows L5 closure only if a spare GPU window appears | Now | manifest-build |
| P12. Reviewer-response scaffolding | Draft limitation/rebuttal patches keyed to outcome tree | Prevents narrative drift after mixed/null outcomes | Now | doc-only |

## 5. SAE Decision

[inferred] Primary option: **Conditional + Reframe/Drop**. Do not claim a Mistral cross-representational SAE comparison on the critical path. [observed] Gemma Scope is an unusually comprehensive Gemma SAE release; [Gemma Scope](https://aclanthology.org/2024.blackboxnlp-1.19/) is not mirrored by an exact `Mistral-Small-24B-Instruct-2501` public suite. [observed] A visible Mistral-family SAE is single-layer and trained on a later different checkpoint, so it is a candidate for exploratory analysis, not a matched substitute ([Codcordance Mistral 3.2 SAE](https://huggingface.co/Codcordance/Mistral-Small-3.2-24B-Instruct-2506-SAE)). [inferred] Training a credible Mistral SAE/transcoder before CP5 is low ROI because a H-neuron null makes the cross-representational comparison strategically moot. [inferred] Fallback: after CP5 succeeds, train a narrow residual/MLP SAE or skip-transcoder on the same Mistral activation distribution; allow a claim only with held-out readout parity, matched intervention operator, layer-matched random features, dead-feature path-drift control, reconstruction/delta-only separation, and pre-sealed alpha/sample/evaluator choices. Transcoders and cross-layer decompositions are plausible substitutes, but only under the same matched-control discipline ([Transcoders](https://arxiv.org/abs/2501.18823); [Circuit tracing / cross-layer transcoders](https://transformer-circuits.pub/2025/attribution-graphs/methods.html); [crosscoder diffing](https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html)).

| SAE outcome on Mistral | Paper claim enabled | Paper claim blocked | Next move |
|---|---|---|---|
| No Mistral SAE run | [inferred] Cross-model H-neuron and ITI externality evidence; Gemma cross-representational result retained | [unsupported] “SAE-vs-H dissociation generalizes to Mistral” | Revise title/abstract/limitations to say SAE comparison is Gemma-only; do not hide absence |
| Off-the-shelf different-checkpoint SAE only | [inferred] Exploratory appendix if explicitly labeled | Claim-bearing matched-readout comparison | Use only as search lead; no main-text result |
| Narrow trained SAE gets comparable readout and null control | [inferred] Directional Mistral support for readout/control dissociation | Strong broad claim about all SAE bases | Add Mistral SAE row to FaithEval figure or appendix; preserve caveats on layer/corpus |
| Narrow trained SAE gets comparable readout and steers like H-neurons | [observed if run] Mistral shows detector-selected feature steering can work | Gemma-to-Mistral SAE null generalization | Reframe to “basis/operator/model dependent”; emphasize audit framework, not anti-SAE thesis |
| SAE works while H-neurons null | [observed if run] Reversed basis result; stronger evidence that target basis matters | H-neuron generalization and H-neuron-first L1 closure | Move to “cross-model reversal” paper framing; do not bury reversal |
| SAE branch confounded by reconstruction/path drift | [inferred] No claim-bearing SAE result | Any Mistral SAE conclusion | Report as failed branch in limitations/provenance; keep Gemma-only SAE scope |

## 6. Compute Plan

[speculative] Estimates assume H100 SXM 80GB at the attached RunPod price snapshot and BF16, no quantization. Median critical path excluding optional SimpleQA, jailbreak, and SAE is roughly the infra assessment’s one-full-anchor budget.

| Stage | H100-hours (low / median / high) | Approx. cost | Marginal-hour rank | Cut-first if budget halves |
|---|---:|---:|---:|---|
| CP1 token/template/span smoke | 0.2 / 0.4 / 0.8 | ~$0.60 / $1.20 / $2.40 | 1 | No |
| CP2 activations + CP3 classifier | 1.5 / 2.5 / 4.5 | ~$4.50 / $7.50 / $13.50 | 2 | No; reduce batch/pilot only |
| CP4 FaithEval pilot/control smoke | 0.5 / 0.8 / 1.5 | ~$1.50 / $2.40 / $4.50 | 3 | Shrink pilot, not audit |
| CP5 FaithEval full + controls | 2.0 / 3.5 / 6.0 | ~$6 / $10.50 / $18 | 4 | Keep H path; reduce optional extra controls only after minimum matched controls pass |
| CP6 TruthfulQA MC ITI | 0.5 / 1.0 / 2.0 | ~$1.50 / $3 / $6 | 5 | No if bridge is planned |
| CP7 TriviaQA bridge baseline/ITI | 1.5 / 2.5 / 5.0 | ~$4.50 / $7.50 / $15 | 6 | No; bridge is MVL1-critical |
| CP8 bridge margin teacher forcing | 0.3 / 0.8 / 1.5 | ~$1 / $2.40 / $4.50 | 8 | Yes; keep behavioral coding first |
| SimpleQA optional generation stress | 0.8 / 1.5 / 3.0 | ~$2.40 / $4.50 / $9 | 9 | Yes; bridge carries generation externality |
| Jailbreak measurement mini-replication | 1.0 / 2.0 / 4.0 | ~$3 / $6 / $12 | 7 | Yes unless measurement section needs fresh Mistral evidence |
| Conditional narrow SAE/transcoder | 20 / 60 / 150 | ~$60 / $180 / $450 | 10 before CP5; 4 after CP5 success | Yes; never before H-neuron replication |
| Comprehensive Gemma-Scope-like Mistral suite | 200+ / 500+ / 1000+ | $600+ / $1500+ / $3000+ | Last | Always defer |

[speculative] Highest expected-information marginal hour is CP1+early CP3: it exposes tokenizer/template/span/geometry failures that would invalidate every downstream result. [speculative] Most over-budgeted work is optional full-generation breadth after bridge; cut SimpleQA and jailbreak before cutting FaithEval controls or bridge. [speculative] Most under-budgeted stages are: (1) Mistral-specific token/template audit, because a subtle span bug creates false mechanistic evidence; (2) FaithEval matched controls, because reviewer trust will hinge on specificity; (3) judge recalibration/bridge human coding, because cross-model output style can break inherited graders.

## 7. Mistral-Specific Measurement Risks

| Risk | Affected surface(s) | Pre-flight mitigation | Detection signal during run |
|---|---|---|---|
| Tokenizer divergence / `fix_mistral_regex=True` | All answer-token extraction; H-neuron classifier | [observed/inferred] Manual span audit on canonical examples; sidecar records tokenizer kwargs; fail-fast if kwarg absent | Answer spans off by boundary tokens; train/test labels inconsistent; unexpected parse failures |
| Instruct-template differences | FaithEval, TruthfulQA, bridge, jailbreak | Freeze `apply_chat_template` output examples; store rendered prompts; do not mix 2501 and 2503 templates | Output format shifts across alpha/control conditions; parser failures cluster by prompt style |
| Mistral 2501 vs 2503 checkpoint mismatch | Whole L1 claim | Use exact wording: `Mistral-Small-24B-Instruct-2501`; do not claim `3.1/2503`; cite registry support status | Reviewer can reproduce wrong checkpoint only if manuscript wording is loose |
| Judge-calibration drift on Mistral output distribution | FaithEval, bridge adjudication, SimpleQA, jailbreak | Mini Mistral holdout with CSV-v3/StrongREJECT/human or blinded rater comparison; dated judge IDs | Evaluator disagreements cluster on Mistral-specific refusal or verbose templates |
| FaithEval prompt-style sensitivity | FaithEval | Freeze `standard` prompt for replication; run style audit before final; parser remap locked | Standard-vs-anti sign/magnitude divergence; remap changes verdict |
| KV-cache / context-length differences | Long prompts, bridge, jailbreak full generation | Fixed `max_new_tokens`, no short truncation for safety surfaces, memory trace, deterministic flags | OOM-induced batch changes; truncation flags; output length distribution differs by condition |
| Refusal-template differences | JailbreakBench, SimpleQA attempt-rate interpretation | Full-generation scoring; refusal-then-comply rubric; manual review of borderline Mistral cases | Refusal-looking prefix with later harmful/substantive content; binary/graded judge split |
| Quantization or mixed precision drift | All mechanistic interventions | BF16 only for claim-bearing runs; no 4-bit/8-bit; environment sidecar | Activations/logits differ across smoke reruns or classifier width mismatches |
| Layer/FFN geometry hardcoding | Classifier, H-neuron controls | Registry-derived dimensions; fail-fast width checks; layer-matched random generated from active classifier | Any Gemma-shaped width, layer count, or path appears in Mistral sidecar |
| Sample leakage across train/dev/test | Classifier, TruthfulQA folds, bridge | Multiple exclusion paths; manifest hashes; fold IDs in sidecar | Same item ID appears across training/eval manifests |
| Control-path non-comparability | FaithEval, jailbreak controls | H and random controls share prompt style, sample manifest, max samples, evaluator | Control and H outputs differ in non-intervention metadata |

## 8. Outcome-Contingency Tree

| Outcome pattern | Claim enabled | Claim blocked | Next move | Manuscript-section impact |
|---|---|---|---|---|
| Full replication: H-neuron dose-response, FaithEval comparable-readout dissociation, bridge externality, and ITI MC-vs-generation gap all reproduce | [inferred] Strongest L1 closure: audit gates reproduce on Gemma and Mistral; if SAE branch passed, readout/control dissociation also cross-model | [unsupported] Universal architecture-scale claim; two models still do not prove invariance | Upgrade abstract and limitations: “replicated on a 24B Mistral anchor”; keep scope language tight | Abstract, Contributions, §2 FaithEval, §3 bridge, §6 limitations, Appendix provenance |
| H-neuron + bridge/ITI replicate, but no Mistral SAE branch | [inferred] L1 materially mitigated for H-neuron control and ITI externality | [unsupported] Cross-representational SAE dissociation generalizes | Make SAE comparison Gemma-only in title/abstract; add Mistral H-neuron/bridge as second-model audit | Abstract, §2, §3, L1 row in `tab:limitations` |
| Partial replication: any 2--3 of 4 gates reproduce | [inferred] Audit framework remains useful; model specificity becomes a result | “Dissociations generalize wholesale” | Split claims by gate; run only the cheapest missing diagnostic if it distinguishes measurement confound from true null | Results section becomes surface-by-surface, not a single replication paragraph |
| Null: H-neurons fail to dose-respond on Mistral with clean controls | [observed if run] Gemma H-neuron result is model-specific; null is publishable if controls are clean | Cross-model H-neuron steering generalization; conditional SAE comparison unless separately justified | Stop H-neuron branch; still run TruthfulQA/bridge only if budget supports independent externality test; revise framing to “audit detects non-transfer” | §2 becomes Gemma case + Mistral null; L1 becomes main discussion, not limitation only |
| Reversed sign: e.g. SAE-class works, H-neurons null or opposite dissociation | [observed if run] Basis/operator/model dependence is stronger than original story | H-neurons as generally better handles; Gemma SAE null as representative | Promote reversal; compare operator/path controls; do not rationalize away | Title/framing shift from “Strong Readouts, Local Levers” toward “Model-Dependent Levers” |
| Mixed surfaces: bridge replicates but FaithEval does not, or vice versa | [observed if run] Surface-locality is reinforced; transfer cannot be inferred from one benchmark | Unified cross-model steering story | Prioritize the non-replicating surface’s measurement audit; avoid running more surfaces until confound classified | §3 can strengthen while §2 narrows, or vice versa |
| Measurement-confounded: judge or prompt-style differences make Mistral uninterpretable | [observed if run] No scientific replication claim; only infrastructure lesson | Any Mistral L1 closure | Freeze outputs; run recalibration before more GPU; if unresolved, omit Mistral claims and report failed preflight if useful | Appendix/provenance note only; do not add main-text result |
| TruthfulQA MC source effect absent | [observed if run] ITI externality setup fails on Mistral | MC-positive/generation-negative divergence | Do not run bridge as confirmatory; optional exploratory bridge may test harm without source gain | §3 Mistral extension omitted or framed as null source-surface transfer |
| Bridge harm absent but MC gain present | [observed if run] Positive source-surface control can be benign on this model/surface | Bridge externality generalization | Run SimpleQA small stress only if pre-registered; otherwise report Mistral as non-harmful on bridge | §3 becomes “Gemma externality does not automatically transfer” |
| Bridge harm present but wrong-entity taxonomy fails | [observed if run] Externality transfers at score level, not failure-mode level | Wrong-entity substitution generalizes | Report new Mistral failure taxonomy; do not force Gemma categories | Table/figure taxonomy changes; limitation refined |

## 9. Reviewer Anticipation

**1. “A 2-model audit is not architecture-general.”** [inferred] Preempt by not claiming architecture-general validity. The rebuttal is that L1 is weakened from “all evidence is Gemma-only” to “two decoder-only instruction models, different family/scale/checkpoint, reproduce or fail under a locked audit.” [observed] Mistral’s model card identifies the 2501 target as a 24B instruction-tuned model, distinct from the Gemma anchor ([Mistral 2501 model card](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)). Manuscript landing: Introduction study orientation, §6 limitations, Appendix limitation inventory. The limitation row should say “two-model audit, not universality.”

**2. “The Mistral checkpoint is convenient, not exact.”** [observed] The migration report and replication plan deliberately select `Mistral-Small-24B-Instruct-2501` and defer `2503` because the current pipeline supports causal-LM text checkpoints, not the `mistral3`/processor path. [inferred] Preempt by making the model ID exact everywhere and avoiding “Mistral Small 3.1” language. Manuscript landing: Methods/model paragraph and limitations. A reviewer can dislike the choice; they should not be able to accuse the paper of checkpoint ambiguity.

**3. “The SAE result is biased by Gemma-Scope availability.”** [observed] Gemma Scope provides a broad Gemma SAE release, whereas no attached/current exact-match Mistral 2501 suite is available; SAE steering literature is actively mixed, with feature-selection concerns emphasized by Arad et al. and broad SAE steering underperformance reported by AxBench ([Gemma Scope](https://aclanthology.org/2024.blackboxnlp-1.19/); [Arad et al. 2025](https://aclanthology.org/2025.emnlp-main.519/); [AxBench](https://arxiv.org/abs/2501.17148)). [inferred] Preempt by not using SAE absence as evidence. Either run the conditional matched Mistral SAE branch or label the cross-representational dissociation Gemma-only. Manuscript landing: §2 synthesis, L2/L3 limitations.

**4. “Judge and prompt drift can manufacture the result.”** [observed] The current manuscript already treats measurement sensitivity as a result, not noise. [inferred] For Mistral, evaluator versions, rubrics, JSON schemas, rendered prompts, full-generation policy, and parser remaps must be frozen before GPU spend. This also aligns with external steering-evaluation concerns about format- and context-dependent conclusions ([Pres et al. 2024](https://arxiv.org/abs/2410.17245); [Opiełka et al. 2026](https://arxiv.org/pdf/2602.22424)). Manuscript landing: §4 Measurement, Appendix provenance, checklist table.

**5. “Novelty is weak relative to H-Neurons, AxBench, transcoders, and SAE utility work.”** [observed] The field already contains H-neuron, SAE selection, SAE utility, and transcoder results ([H-Neurons](https://arxiv.org/abs/2512.01797); [Wang et al. 2025/2026 SAE utility](https://arxiv.org/abs/2510.03659); [Transcoders](https://arxiv.org/abs/2501.18823)). [inferred] Preempt by positioning this paper as a **reviewer-grade audit scaffold with falsified gate transitions**, not as discovery of detector/control divergence or a new steering method. Manuscript landing: Related Work final paragraph, Contributions, Conclusion. The Mistral run should be described as a hardening audit, not a benchmark-chasing extension.

## 10. Deprioritized Work and Non-L1 Parallel Scope

| Work | Decision | Reason | Condition that would change decision |
|---|---|---|---|
| Comprehensive Mistral SAE suite | Defer | [speculative] High cost; not needed before CP5; no payoff if H-neurons null | H-neurons replicate, deadline expands, and SAE branch has committed controls |
| Off-the-shelf different-checkpoint Mistral SAE as claim-bearing substitute | Defer/block | [observed/inferred] Checkpoint/layer/training mismatch; would invite reviewer attack | Only as exploratory appendix with explicit non-claim wording |
| 2503 loader/processor migration | Defer | [observed] Current causal-LM path intentionally unsupported for 2503 | Team decides exact 2503 is required for submission, accepting schedule risk |
| Full BioASQ Mistral | Defer | [inferred] Lower information gain than FaithEval and bridge; alias metric may be style-confounded | FaithEval passes and bridge is inconclusive on capability externality |
| Full SimpleQA Mistral | Conditional | [inferred] Useful only as support if bridge is mixed or reviewer asks for another generation surface | TruthfulQA MC passes and bridge is weak/mixed |
| Jailbreak Mistral full measurement replication | Conditional | [inferred] Important for measurement section but not MVL1 first gate | CP5/CP7 complete or measurement claim becomes central in reviewer plan |
| D7 Mistral mechanism diagnostic | Defer | [inferred] Mechanism-local and not needed for L1 closure | H-neuron/bridge both replicate and paper needs a mechanistic appendix |
| L2 matched-readout confound via new Mistral SAE | Conditional | [inferred] Only valuable after H-neuron replication; otherwise answers wrong question | CP5 succeeds and budget supports narrow SAE/transcoder |
| L3 SAE layer coverage | Defer | [inferred] High-cost and less reviewer-critical than admitting scope | Conditional SAE branch produces a promising but layer-limited signal |
| L4 bridge human-rater IRR | **Fund now** | [inferred] Non-GPU, high reviewer value, directly improves cleanest Gemma anchor and Mistral bridge coding plan | None; this is the best non-L1 parallel investment |
| L5 jailbreak multi-seed | **Prep now; execute only after Mistral critical window** | [inferred] Useful but GPU-competitive with L1; manifest/judge prep is cheap | Spare GPU window or Mistral blocked by non-GPU dependency |

[inferred] Recommended non-L1 parallel investments: **(1) L4 independent human-rater IRR for bridge**, because it is cheap, high-trust, and strengthens the current cleanest anchor; **(2) L5 jailbreak multi-seed preparation**, not immediate execution, because it prevents delay later without stealing H100 time from L1. [inferred] Do not fund L3 before CP5.

## 11. Open Questions Back to the Team

1. [inferred] What exact submission/rebuttal deadline determines whether the conditional SAE branch is impossible, optional, or required?
2. [inferred] What is the hard H100-hour or dollar ceiling for claim-bearing Mistral work, excluding judge/API and human-rater costs?
3. [observed/inferred] Are all Mistral-ready manifests and source datasets available locally for FaithEval, TruthfulQA MC, TriviaQA bridge, SimpleQA, and JailbreakBench, or will any require new data access work?
4. [inferred] Will the team accept a manuscript revision that makes the SAE comparison explicitly Gemma-only if CP5 succeeds but no conditional SAE branch runs?
5. [inferred] Which judge model IDs can be frozen contractually/technically for the whole run, and are floating aliases forbidden in the repo?
6. [inferred] Is a blinded human rater available for the bridge discordant set, and can the rater be kept independent of model condition?
7. [inferred] Does the paper need to say “Mistral Small 24B 2501” only, or is any current prose implying `2503/3.1` and therefore requiring correction before review?

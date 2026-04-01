# Act 3 Final Sprint

> This is the only execution source of truth for current priorities.

## Source Hierarchy

- Execution and priority order: [act3-sprint.md](./act3-sprint.md)
- Evaluation and audit contract: [measurement-blueprint.md](./measurement-blueprint.md)
- Strategic synthesis: [gpt-act3-deep-research-report.md](./gpt-act3-deep-research-report.md)
- Literature reference: [literature-review-act3.md](./literature-review-act3.md)
- Historical pre-pivot planning: [act2-pre-pivot-archive](./act2-pre-pivot-archive)

## North Star

Act 3 is a final-sprint comparative program: use the H-neuron work as the reference row, evaluate stronger steering baselines with the same high-resolution measurement stack, audit safety externalities directly, and run one scoped causal check.

## Sprint Questions

- Does full-generation, graded evaluation change the practical safety conclusion relative to truncated binary judging?
- Are direction-level baselines cleaner or stronger than the current H-neuron intervention on this model?
- How much of the observed safety regression is explained by overlap with refusal geometry?
- Can one small causal pilot show why correlational localization is a weak intervention selector?

## Committed Deliverables

| ID | Deliverable | Why it exists | Status | Next action |
|---|---|---|---|---|
| D0 | Freeze the Act 3 measurement contract | Every baseline must be judged the same way or the comparison is not publishable | defined in [measurement-blueprint.md](./measurement-blueprint.md) | Use it as a gate on all remaining runs |
| D0.5 | Infrastructure sprint | D2-D6 require code and data that do not exist yet: contrastive datasets, direction extraction, directional intervention, capability mini-battery | **partial** — refusal data frozen ✅; direction code written + Gemma-3 compat fixed ✅; harmful eval sets materialized (JBB_100, HarmBench_159, StrongREJECT_313) ✅; truthfulness contrastive data ✅ (TriviaQA consistency 2673 records + TruthfulQA paper-faithful ITI artifact extracted at `data/contrastive/truthfulness/`); IFEval and perplexity remain open | Wire IFEval + perplexity for capability battery. |
| D1 | Baseline A: H-neuron reference row | Preserve continuity with the replication and anchor every comparison to the current intervention | partial — 4-alpha CSV-v2 graded evaluation complete (α=0.0/1.0/1.5/3.0, 500 records each); FaithEval relabeled as MCQ context-acceptance under anti-compliance prompting (see [2026-03-31 audit](./act3-reports/2026-03-31-faitheval-task-definition-audit.md)) | Complete the graded-jailbreak negative control (~4h GPU) and attach a capability mini-battery (IFEval + perplexity). |
| D2 | Extract refusal direction | Prerequisite for D3, D4, D5, D6: the refusal direction is needed both as an intervention vector and as the overlap reference | **done** — best layer 25 (98.4% val accuracy, separation=9179); sanity gate passed (ablation: refusal 25%→0% on harmful val; single-layer addition too weak to induce refusal on harmless — expected, D3 uses all-layer). Directions at `data/contrastive/refusal/directions/`. | Proceed to D3 and D3.5. |
| D3 | Baseline B: refusal-direction intervention | Diagnostic comparator for safety geometry and externality explanation, not the presumed best hallucination mitigator. Tests what fraction of H-neuron effects are attributable to refusal-circuit overlap. | **partial / decision-complete on FaithEval** — the clean all-layer ablation run shows a narrow usable point at β=0.02 (70.2%, 702/1000) and a cliff at β=0.03 (51.1%, 511/1000). Parse failures are 0/1000 at all three clean settings, but row-level audit shows answer-option bias and output-distribution distortion at β=0.03 rather than a clean H-neuron-style curve. Baseline B is informative diagnostically, not robust as an intervention family. | Do not broaden D3 now. Either carry β=0.02 forward only as one scoped D5 externality-check row, or prioritize D3.5 immediately. |
| D3.5 | Refusal-overlap analysis for Baseline A | Cheap once D2 is done; informs whether D4 should change approach | **done / robustness-downgraded** — the projected 38-neuron residual update overlaps the D2 refusal basis more than a layer-matched random-neuron null (canonical gap -0.0183 [-0.0310, -0.0126]; subspace gap +0.0361 [+0.0251, +0.0390]), and full-model prompt-level overlap weakly predicts both FaithEval and jailbreak outcomes. But the signal is dominated by layer 33; removing that layer collapses or flips the mediation correlations. Detailed audit: `data/gemma3_4b/intervention/refusal_overlap/refusal_overlap_audit.md`. | Do **not** orthogonalize D4 yet. Treat refusal overlap as a live hypothesis that needs a targeted robustness pass (layer 33 / top-neuron exclusion), not as a settled mechanism. |
| **GATE** | **Decision gate** | If H-neuron overlap with refusal direction is high, D4 may need to target a direction orthogonal to both; if low, proceed as planned | **resolved** — D3.5 found real overlap but insufficiently robust mediation evidence | Proceed with D4 as planned; keep refusal-orthogonalization conditional on a robustness-confirmed overlap result |
| D4 | Baseline C: truthfulness direction baseline | Test whether a direction-level intervention can beat neuron scaling on the task family the project actually cares about | **ITI path active / generation null confirmed** — residual-stream direction: β=0.01 narrow window on FaithEval MCQ (see [report](./act3-reports/2026-03-30-d4-truthfulness-direction.md)); ITI head-level (paper-faithful K=12 α=8, TruthfulQA paper-faithful artifact): +4.3pp MC1 / +8.5pp MC2 on 163 held-out questions, CI excludes zero on MC2; SimpleQA 1000-question generation run: monotonic decline 4.8%→4.0%→1.6%, escape-hatch mechanism confirmed, no sweet spot (see [report](./act3-reports/2026-04-01-simpleqa-iti-production.md)); TriviaQA-transfer artifact: flat null on TruthfulQA MC (confirms paper-faithful extraction matters) | Priority 1: remove escape hatch from SimpleQA prompt, rerun to test whether null is prompt artefact. Priority 2: random-head control at α=8.0 (200 samples). Priority 3: generation-calibrated direction extraction. See [SimpleQA report §5](./act3-reports/2026-04-01-simpleqa-iti-production.md) for ROI-ordered next steps. |
| D5 | Full externality audit | Turn steering safety drift into a first-class result instead of an afterthought | not started | For Baselines A through C, report jailbreak severity drift, capability drift, response-structure shifts, and overlap with refusal geometry |
| D6 | Refusal-orthogonalized mitigation check | Convert the overlap analysis into a positive safety-aware steering result if possible | not started | Project out the refusal-overlap component from the truthfulness or hallucination vector and compare the before-versus-after tradeoff |
| D7 | One scoped causal pilot | Validate the critique that correlational selection is not the right intervention selector | not started | Attribution patching (2 fwd + 1 bwd) to rank top-20 attention heads by indirect effect on jailbreak refusal for a 50-prompt contrastive subset; compare top-k heads to L1-probe-selected neurons; success = methods select different components AND causal components change CSV-v2 outcome consistently |
| D8 | Final synthesis | Close the sprint with a field-useful protocol and comparative claim, not another idea list | not started | Write the result as a steering protocol: graded evaluation, externality audit, comparator suite, and one causal check |

## Priority Order

1. Enforce the measurement contract (D0).
2. Infrastructure sprint: contrastive data, direction extraction code, capability battery (D0.5).
3. Finish Baseline A as the cleaned reference row (D1).
4. Extract the refusal direction (D2).
5. Close Baseline B at the current FaithEval diagnostic state; do not broaden D3 without a specific D5 externality reason.
6. D3.5 complete: keep refusal overlap as a live hypothesis, but do not let the current layer-33-dominated signal redefine D4.
7. Run Baseline C: truthfulness or hallucination direction (D4).
8. Add a targeted D3.5 robustness follow-up only if it is cheaper than the next D4 decision it informs.
9. Run the full externality audit across A, B, and C (D5).
10. Run the refusal-orthogonalized mitigation check (D6).
11. Run one scoped causal pilot (D7).
12. Write the final synthesis (D8).

## Dataset Status And Caveats

- The refusal contrastive dataset for D2 is now frozen and scientifically usable: it matches the paper's Appendix A recipe at the level of source pools and 128/32/100 split sizes, while pinning to the authors' published `refusal_direction` split pools instead of rebuilding from mutable raw corpora.
- This is a faithful replication starting point, not a proven row-for-row recovery of the exact hidden paper subset. The repo exposes the source pools and sampling procedure, not an authoritative final paper sample file.
- The frozen working dataset starts from the official seed-42 sampling order, then applies one explicit rigor fix: replace 2 train-harmful prompts whose normalized text still overlapped the published harmful test pool. This should be treated as an audit-strengthening deviation, not as "exact upstream."
- The official repo's later model-specific refusal-metric filtering is not baked into the frozen dataset because that step depends on the target model. Dataset provenance and caveats live in `data/contrastive/refusal/metadata.json`.
- For D3/D5 evaluation, keep the paper's harmful eval sets conceptually separate: `JBB_100` and filtered `HarmBench_test_159`. The repo's composite `harmful_test` pool is useful for audits, but it is broader than the paper's page-18 description.

## Definition Of Done

A steering baseline is not complete until it has all of the following:

- Full-generation evaluation that avoids systematic truncation.
- The primary safety and response-structure metrics from [measurement-blueprint.md](./measurement-blueprint.md).
- A retained-capability mini-battery.
- Per-example outputs.
- A run manifest with intervention, judge, benchmark, and generation settings.
- A one-paragraph interpretation stating target gain, safety cost, and remaining uncertainty.
- A negative control outcome or an explicit argument for why one is unnecessary (e.g., the intervention vector is extracted from a specific contrastive dataset, not randomly sampled).
- A cross-benchmark consistency statement listing which benchmarks were run and noting agreement or divergence across them.

## Non-Goals

- Another round of general neuron tinkering as the main story.
- Breadth-first benchmark expansion for its own sake.
- Reviving the legacy 256-token jailbreak narrative.
- Reopening SAE steering.
- Building a full causal-localization stack before a small pilot proves it is worth the setup cost.
- Keeping parallel plan files outside this document and [measurement-blueprint.md](./measurement-blueprint.md).
- Migrating to TransformerLens for this sprint. TransformerLens is available in the parent workspace venv (`../.venv`) but does not support Gemma-3-4B-IT (only Gemma-2 variants). Head-level hooks must use raw HuggingFace hooks on the existing model loading path; new hook types (head-output) can be added incrementally if D7 or head-level D4 refinement is triggered.

## Backlog After The Critical Path

- Negative-weight neuron work is allowed only as a closure experiment after the committed slate is on track. It is not on the critical path.
- **Sparse detector comparison**: compare the 38-neuron L1 probe as a *detector* against (a) mean-pooled residual-stream features, (b) difference-in-means truthfulness features, (c) top-K attention head features. The point: good detector features are not necessarily good intervention targets. Low cost if D4 contrastive data already exists.
- **CASAL-style weight baking**: if D4/D6 produce a useful truthfulness steering direction, CASAL (Conditional Activation Steering Amortized into weights) is the cleanest path from "useful direction" to "baked-in model improvement." Out of scope for this sprint.
- **Head-level ITI as D4 refinement**: if residual-stream truthfulness direction in D4 is weak, implement ITI (Li et al. 2023) with attention-head probing. Requires new hook infrastructure (current codebase only has `down_proj` hooks, no head-output hooks). TransformerLens is available in the parent venv but does not support Gemma-3; use raw HuggingFace hooks if this path is taken.

## Claims To Test

Inherited claims from the deep research report that must be verified against actual data, not assumed:

| Claim | Source | Status | What would falsify it |
|---|---|---|---|
| H-neuron effects are noisy and non-monotonic | Deep research report | **Partially false.** FaithEval is perfectly monotonic (rho=1.0, slope +2.09pp/alpha [1.38, 2.83]). Non-monotonicity applies to jailbreak binary judge only; CSV-v2 graded metric shows a significant monotonic slope (+7.6pp). | FaithEval rho < 0.9 or negative control matching the slope |
| H-neurons are a noisy proxy for the refusal direction | Deep research report | Untested hypothesis | Cosine similarity between projected 38-neuron residual-stream vector and refusal direction (including PCA subspace) is low (< 0.3); or refusal direction does not reproduce FaithEval compliance effect |
| FaithEval compliance is mediated by hallucination circuits, not refusal circuits | Implicit in original paper | Untested | Refusal-direction ablation reproduces the FaithEval alpha slope, implying FaithEval compliance is partially a refusal behavior |
| Direction-level interventions are cleaner than neuron-level on this model | Literature consensus | Untested on Gemma-3-4B-IT | Refusal-direction intervention has wider CIs or smaller effect than H-neurons on matched benchmarks |
| Residual-stream direction is sufficient for truthfulness intervention (head-level not needed) | Sprint design assumption | **Partially false.** Head-level ITI (paper-faithful, K=12) improves TruthfulQA MC (+4.3pp MC1, +8.5pp MC2); residual-stream direction has a narrow window (β=0.01) on FaithEval MCQ but no effect on SimpleQA generation; neither is "sufficient" for generation tasks | SimpleQA escape-hatch removal run would separate prompt artefact from structural null |
| H-neuron overlap with refusal is measurable via single-vector cosine | D3.5 assumption (now corrected) | Corrected in D3.5 | Overlap appears artificially low because projection step was missing; after projection, if overlap is still <0.1 across the PCA refusal subspace, refusal-overlap explanation is genuinely weak |

## Open Uncertainties

| Uncertainty | Impact | How to resolve | Cost |
|---|---|---|---|
| AdvBench gated on HF — harmful source availability | Affects D0.5a refusal dataset construction | Use `forbidden_question_set.csv` (390) + JBB-Behaviors (100) + other accessible sources. 490 available sources is plenty for 128+32. | Low |
| IFEval constraint checker complexity | Affects D0.5c timeline | Start with subset of simplest constraints (word count, format markers). Full checker can follow. | Medium |
| Truthfulness contrastive quality | Highest single-point-of-failure for D4 | Validate direction separation on seed slice before full curation. Diversity > volume (Liu et al. 2024). | High |
| `device_map="auto"` vs `cuda:0` for direction hooks | Affects per-layer device assignment in extract/intervene scripts | Existing code uses `cuda:0` default. Test both paths. | Low |

## Decision Log

- 2026-03-27: Reframed Act 3 as a comparative steering sprint rather than another round of H-neuron optimization.
- 2026-03-27: Kept the deep research report and literature review as reference documents, not live planning docs.
- 2026-03-27: Archived stale pre-pivot planning files under [act2-pre-pivot-archive](./act2-pre-pivot-archive).
- 2026-03-27: Added D0.5 infrastructure sprint, restructured priority order with refusal-direction extraction as early prerequisite and a decision gate before D4, scoped D7 causal pilot concretely, added claims-to-test appendix to prevent inherited errors from deep research report.
- 2026-03-28: Reviewed AI-generated critique of sprint plan. Adopted: (1) D3.5 methodology fix -- must project neuron weights into residual-stream space via `down_proj` columns before cosine comparison, was comparing apples to oranges; (2) expand refusal overlap to PCA subspace, not just single vector; (3) separate contrastive datasets aggressively (refusal, truthfulness, detection) to avoid "hallucination vector is a refusal vector in disguise" failure mode; (4) D4 uses difference-in-means with diverse multi-source data, not single-dataset logistic regression; (5) reframed Baseline B as diagnostic for safety geometry; (6) start D4 data design during D0.5; (7) FaithEval first for D4 (MDE logic). Rejected: (a) "start head-level first in D4" -- codebase has no head-level intervention hooks (only `down_proj` hooks), and TransformerLens does not support Gemma-3-4B-IT (only Gemma-2 variants); residual-stream direction reuses D2/D3 code and is the correct simple-first ordering; head-level is a conditional refinement, not the starting point; (b) "implement ITI via TransformerLens" -- TransformerLens is available in the parent workspace venv but lacks Gemma-3 support; raw HuggingFace hooks would be needed; D7 already scopes causal head-level work. Added to backlog: sparse detector comparison, CASAL weight-baking, head-level ITI as D4 refinement path.
- 2026-03-28: Planned D0.5 implementation. Key technical decisions: (1) hook block-output residual stream (not `down_proj`) for direction extraction/intervention — standard repr. engineering surface; (2) per-layer direction extraction with separation diagnostics, not cross-layer PCA; (3) all-layer default for refusal intervention (Arditi 2024), subset as conditional refinement; (4) β=0.0 no-op convention for direction steering (distinct from H-neuron α=1.0); (5) new `extract_direction.py` script (not retrofitting CETT-specific `extract_activations.py`); (6) capability battery = BioASQ (exists) + IFEval (541 prompts, programmatic eval) + WikiText-2 NLL perplexity; (7) execution order: data schema + seed → code in parallel → full curation before conclusions.
- 2026-03-28: Froze the refusal contrastive dataset for D2 around a pinned `refusal_direction` snapshot rather than a hand-rebuilt corpus mix. Status/caveats: (1) paper-faithful at the level of source pools and 128/32/100 split sizes from Appendix A/page 18; (2) not claimed as a proven row-for-row recovery of the authors' exact hidden paper sample; (3) starts from the official seed-42 sample, then applies one explicit leakage fix replacing 2 train-harmful prompts that still normalized-overlapped the harmful test pool; (4) does not bake in the repo's later model-specific refusal-metric filtering; (5) D3/D5 should materialize `JBB_100` and filtered `HarmBench_test_159` explicitly, because the repo's broader `harmful_test` pool is useful for audits but not identical to the paper's stated eval sets.
- 2026-03-29: Fixed Gemma-3 compatibility in `extract_direction.py` and `intervene_direction.py`: (1) nested config (`model.config.text_config.num_hidden_layers` not `model.config.num_hidden_layers`); (2) decoder layers at `model.model.language_model.layers` not `model.model.layers`; (3) `dtype=` instead of deprecated `torch_dtype=`. These were never-tested code paths.
- 2026-03-29: D2 completed (layer 25, 98.4% val, clean-provenance `bd1877b`). Harmful eval sets materialized (JBB_100, HarmBench_159, StrongREJECT_313). D3 calibrated: β=0.02 narrow window, β=0.03 answer-option bias failure. D3.5 gate resolved: refusal overlap real but layer-33-dominated; proceed with D4 unchanged. Details: [week3-log.md §5–7](./act3-reports/week3-log.md) and [D3 report](./act3-reports/2026-03-29-d3-faitheval-refusal-direction.md).
- 2026-03-30: D4 residual-stream kill-shot: β=0.01 survives (66.0%→71.4% MCQ context-acceptance, zero parse failures); β=0.02 collapses. See [D4 report](./act3-reports/2026-03-30-d4-truthfulness-direction.md). ITI head-level pipeline started: TriviaQA-transfer and context-grounded FaithEval calibrations both returned hard nulls (flat 17/20 across all alphas). Pivoted to TruthfulQA paper-faithful artifact. Gate 1 K×α sweep locked K=12 α=8 (MC1 ~37%, overriding auto-lock at α=12 for stability). TruthfulQA MC 2-fold held-out: +4.3pp MC1 / +8.5pp MC2 at α=8; TriviaQA-transfer artifact flat null on same questions. Sweep analysis: [2026-03-31-sweep-analysis.md](./act3-reports/2026-03-31-sweep-analysis.md).
- 2026-03-31: FaithEval task-definition audit — D1/D3/D4 FaithEval results relabeled as "MCQ context-acceptance under anti-compliance prompting," not truthfulness improvement. TruthfulQA MC and ITI 2-fold CV remain the clean comparison axis. See [2026-03-31-faitheval-task-definition-audit.md](./act3-reports/2026-03-31-faitheval-task-definition-audit.md).
- 2026-04-01: SimpleQA 1000-question ITI production run (K=12, paper-faithful, α∈{0.0, 4.0, 8.0}): Outcome A confirmed — monotonic compliance decline 4.8%→4.0%→1.6%, precision flat at α=0.0 and α=4.0, escape hatch (98–99% "I don't know" literal) is the mechanism. Science & Tech anomaly (6.2% NA at α=4.0 vs 15–35% elsewhere) is most actionable signal. ITI improves MC answer selection (TruthfulQA) but not free-form factual generation (SimpleQA). See [2026-04-01-simpleqa-iti-production.md](./act3-reports/2026-04-01-simpleqa-iti-production.md).

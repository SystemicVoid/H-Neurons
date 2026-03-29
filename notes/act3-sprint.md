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
| D0.5 | Infrastructure sprint | D2-D6 require code and data that do not exist yet: contrastive datasets, direction extraction, directional intervention, capability mini-battery | **partial** — refusal data frozen ✅; direction code written + Gemma-3 compat fixed ✅; harmful eval sets materialized (JBB_100, HarmBench_159, StrongREJECT_313) ✅; truthfulness data, IFEval, and perplexity remain open | Curate **truthfulness set** and **OOD holdout** for D4. Wire IFEval + perplexity for capability battery. |
| D1 | Baseline A: H-neuron reference row | Preserve continuity with the replication and anchor every comparison to the current intervention | partial — 4-alpha CSV-v2 graded evaluation complete (α=0.0/1.0/1.5/3.0, 500 records each) and analysed | Complete the graded-jailbreak negative control (~4h GPU) and attach a capability mini-battery (IFEval + perplexity) |
| D2 | Extract refusal direction | Prerequisite for D3, D4, D5, D6: the refusal direction is needed both as an intervention vector and as the overlap reference | **done** — best layer 25 (98.4% val accuracy, separation=9179); sanity gate passed (ablation: refusal 25%→0% on harmful val; single-layer addition too weak to induce refusal on harmless — expected, D3 uses all-layer). Directions at `data/contrastive/refusal/directions/`. | Proceed to D3 and D3.5. |
| D3 | Baseline B: refusal-direction intervention | Diagnostic comparator for safety geometry and externality explanation, not the presumed best hallucination mitigator. Tests what fraction of H-neuron effects are attributable to refusal-circuit overlap. | **partial / decision-complete on FaithEval** — the clean all-layer ablation run shows a narrow usable point at β=0.02 (70.2%, 702/1000) and a cliff at β=0.03 (51.1%, 511/1000). Parse failures are 0/1000 at all three clean settings, but row-level audit shows answer-option bias and output-distribution distortion at β=0.03 rather than a clean H-neuron-style curve. Baseline B is informative diagnostically, not robust as an intervention family. | Do not broaden D3 now. Either carry β=0.02 forward only as one scoped D5 externality-check row, or prioritize D3.5 immediately. |
| D3.5 | Refusal-overlap analysis for Baseline A | Cheap once D2 is done; informs whether D4 should change approach | **done / robustness-downgraded** — the projected 38-neuron residual update overlaps the D2 refusal basis more than a layer-matched random-neuron null (canonical gap -0.0183 [-0.0310, -0.0126]; subspace gap +0.0361 [+0.0251, +0.0390]), and full-model prompt-level overlap weakly predicts both FaithEval and jailbreak outcomes. But the signal is dominated by layer 33; removing that layer collapses or flips the mediation correlations. Detailed audit: `data/gemma3_4b/intervention/refusal_overlap/refusal_overlap_audit.md`. | Do **not** orthogonalize D4 yet. Treat refusal overlap as a live hypothesis that needs a targeted robustness pass (layer 33 / top-neuron exclusion), not as a settled mechanism. |
| **GATE** | **Decision gate** | If H-neuron overlap with refusal direction is high, D4 may need to target a direction orthogonal to both; if low, proceed as planned | **resolved** — D3.5 found real overlap but insufficiently robust mediation evidence | Proceed with D4 as planned; keep refusal-orthogonalization conditional on a robustness-confirmed overlap result |
| D4 | Baseline C: truthfulness or hallucination direction baseline | Test whether a direction-level intervention can beat neuron scaling on the task family the project actually cares about | not started | Extract a truthfulness/hallucination direction via **difference-in-means** (not logistic regression) on the diverse truthfulness contrastive set from D0.5. Use the same directional ablation/addition infrastructure as D3. Start residual-stream-level, not head-level: this reuses D2/D3 code and is the natural apples-to-apples comparison. If residual-stream direction is weak or noisy, upgrade to head-level (ITI-style, probing top-K attention heads) as a conditional refinement within D4 -- but do not front-load that infrastructure cost. Evaluate on **FaithEval first** (MDE ~3pp, well-powered), then FalseQA, then jailbreak as externality readout. |
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
| Residual-stream direction is sufficient for truthfulness intervention (head-level not needed) | Sprint design assumption | Untested | Residual-stream truthfulness direction in D4 shows no significant effect on FaithEval (MDE ~3pp) while literature head-level probes show clear signal; would trigger head-level refinement |
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
- 2026-03-29: D2 completed. Refusal direction extracted via difference-in-means on frozen 128+128 contrastive set. Best layer: 25 (98.4% val accuracy). Sanity gate passed: single-layer ablation reduces harmful refusal from 25% to 0%. Single-layer addition at β=1.0 too weak to induce refusal on harmless (expected — D3 uses all-layer per Arditi 2024). Single-layer ablation causes text degeneration (known limitation, not a D3 concern).
- 2026-03-29: Materialized harmful eval sets: JBB_100 (100), HarmBench_test_159 (159), StrongREJECT_313 (313) from upstream harmful_test pool.
- 2026-03-29: Clean-provenance rerun after the Gemma-3 compatibility fix reproduced the decision-relevant D2 outputs exactly. Eval-set materialization matched byte-for-byte, and a fresh extraction-only rerun from a clean worktree reproduced the same refusal direction tensor hash, best layer (25), validation accuracy (98.4%), and sanity gate result with `git_dirty: false` in `extract_direction.provenance.20260329_134942.json`. Non-decision run metadata such as wall-clock collection time changed as expected. The earlier dirty-provenance snapshot was archived rather than discarded.
- 2026-03-29: Current state: the canonical live D2 artifact chain now points only to the clean provenance files, while the earlier dirty-provenance run is preserved under archive directories for audit. This cleanup/signoff was committed in `bd1877b` (`data(refusal): canonize clean D2 provenance`), `9ea2305` (`docs(audit): record D2 clean-rerun signoff`), and `31678fc` (`docs(audit): tighten D2 reproducibility wording`). Operational verdict: D3 is unblocked on provenance grounds; remaining risks are conceptual (dataset construction, model-specificity, intervention/generalization), not chain-of-custody risks.
- 2026-03-29: D3 methodology was underspecified on intervention strength, so the first pass tightened it instead of cargo-culting the literature default. Findings: (1) all-layer ablation at β=0.5 on a 250-sample FaithEval slice immediately produced empty/repetitive 256-token outputs (interrupted exploratory run preserved under `data/gemma3_4b/intervention/faitheval_direction_ablate_all-layers_refusal-directions_c892775a83/`); (2) micro-beta calibration exposed an infrastructure bug where `run_intervention.py` collapsed nearby alphas into one-decimal filenames, so `0.005/0.01/0.02` all aliased to `alpha_0.0.jsonl`; fixed in `9e96072` (`fix(intervention): preserve micro-beta alpha labels`); (3) repaired calibration showed a sharp cliff: β≤0.02 stayed in the short-answer regime, β=0.03 already hurt FaithEval on the 20-sample slice, and β=0.05 crossed into malformed long-form behavior (`data/gemma3_4b/intervention/faitheval_direction_ablate_microbeta_calibration/`). A clean-worktree full run at β∈{0.0,0.02,0.03} then established the decision-relevant result with `git_dirty: false` provenance in `data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/run_intervention.provenance.20260329_192019.json`: β=0.02 improves FaithEval compliance from 66.0% to 70.2%, but β=0.03 drops it to 51.1%. Interpretation: Baseline B may have a narrow usable window, but it is not a robust refusal-ablation analogue of the H-neuron curve and should not be broadened without an externality check.
- 2026-03-29: Row-level audit of the clean D3 FaithEval run tightened the interpretation. Zero parse failures did not imply clean behavior: β=0.03 caused a large True→False collapse (236 vs 45 False→True from β=0.02), over-selected answer option `B` (58.1% of outputs), and collapsed compliance on correct-`C` and correct-`D` items (22.5% and 29.5%). For MCQ steering evaluations, output-distribution checks matter; parse-failure rates alone can miss answer-option bias.
- 2026-03-29: D3.5 completed with a stricter read than the first script closeout. The projected H-neuron intervention overlaps refusal geometry more than a layer-matched random-neuron null, and the full-model prompt-level overlap weakly predicts both FaithEval gain and jailbreak CSV-v2 harm drift. But the mediation signal is dominated by layer 33: removing that layer collapses FaithEval correlations to ~0 and flips or reverses the jailbreak readout. Gate decision: do **not** orthogonalize D4 yet. D3.5 upgrades refusal overlap from speculation to a live hypothesis, not to a settled explanation.

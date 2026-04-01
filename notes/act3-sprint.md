# Act 3 Final Sprint

> This is the only execution source of truth for current priorities.

## Source Hierarchy

- Execution and priority order: [act3-sprint.md](./act3-sprint.md) ← you are here
- Narrative log (decisions, surprises, reasoning): [research-log.md](./research-log.md)
- Current rerun audit: [2026-04-01-priority-reruns-audit.md](./act3-reports/2026-04-01-priority-reruns-audit.md)
- Current D4 generation specificity audit: [2026-04-01-random-head-specificity-audit.md](./act3-reports/2026-04-01-random-head-specificity-audit.md)
- Evaluation and audit contract: [measurement-blueprint.md](./measurement-blueprint.md)
- Plot registry (site data file pointers): [plot-registry.md](./plot-registry.md)
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
| D0 | Freeze the Act 3 measurement contract | Every baseline must be judged the same way or the comparison is not publishable | **done** — [measurement-blueprint.md](./measurement-blueprint.md) | Gate on all remaining runs. |
| D0.5 | Infrastructure sprint | D2-D6 require code and data that do not exist yet: contrastive datasets, direction extraction, directional intervention, capability mini-battery | **partial** — refusal data ✅ direction code + Gemma-3 compat ✅ harmful eval sets ✅ truthfulness contrastive data ✅ — IFEval + perplexity remain open | Wire IFEval + perplexity for capability battery. |
| D1 | Baseline A: H-neuron reference row | Preserve continuity with the replication and anchor every comparison to the current intervention | **partial** — FaithEval + jailbreak: [week3-log.md §1–2](./act3-reports/week3-log.md); FaithEval relabeled: [2026-03-31-faitheval-task-definition-audit.md](./act3-reports/2026-03-31-faitheval-task-definition-audit.md); TruthfulQA MC ranking resolved in D4's favour: [2026-04-01-priority-reruns-audit.md](./act3-reports/2026-04-01-priority-reruns-audit.md) | Do not spend more GPU on D1 truthfulness reruns. Finish the graded-jailbreak negative control and capability mini-battery (IFEval + perplexity). |
| D2 | Extract refusal direction | Prerequisite for D3, D4, D5, D6: the refusal direction is needed both as an intervention vector and as the overlap reference | **done** — [week3-log.md §5](./act3-reports/week3-log.md); directions at `data/contrastive/refusal/directions/` | Proceed to D3 and D3.5. |
| D3 | Baseline B: refusal-direction intervention | Diagnostic comparator for safety geometry and externality explanation, not the presumed best hallucination mitigator. Tests what fraction of H-neuron effects are attributable to refusal-circuit overlap. | **partial / decision-complete on FaithEval** — [2026-03-29-d3-faitheval-refusal-direction.md](./act3-reports/2026-03-29-d3-faitheval-refusal-direction.md) | Do not broaden D3. Carry β=0.02 only as one scoped D5 externality-check row if needed. |
| D3.5 | Refusal-overlap analysis for Baseline A | Cheap once D2 is done; informs whether D4 should change approach | **done / robustness-downgraded** — [week3-log.md §7](./act3-reports/week3-log.md); full audit: `data/gemma3_4b/intervention/refusal_overlap/refusal_overlap_audit.md` | Do **not** orthogonalize D4 yet. Treat refusal overlap as a live hypothesis; targeted layer-33 / top-neuron robustness pass deferred. |
| **GATE** | **Decision gate** | If H-neuron overlap with refusal direction is high, D4 may need to target a direction orthogonal to both; if low, proceed as planned | **resolved** — real overlap but layer-33-dominated; robustness insufficient to act on | Proceed with D4 as planned; refusal-orthogonalization conditional on a robustness-confirmed overlap result. |
| D4 | Baseline C: truthfulness direction baseline | Test whether a direction-level intervention can beat neuron scaling on the task family the project actually cares about | **ITI path active / MC winner / generation still null after prompt repair; random-head control says the failure is configuration-specific, not generic matched-K perturbation** — residual-stream: [2026-03-30-d4-truthfulness-direction.md](./act3-reports/2026-03-30-d4-truthfulness-direction.md); initial ranked reruns: [2026-04-01-priority-reruns-audit.md](./act3-reports/2026-04-01-priority-reruns-audit.md); specificity control: [2026-04-01-random-head-specificity-audit.md](./act3-reports/2026-04-01-random-head-specificity-audit.md) | Priority 1: decode-scope ablation on the current paper-faithful artifact. Priority 2: `E1`, then `E2`, under the best scope. Priority 3 (conditional): `E3` or causal head selection only if the cheaper discriminators fail. |
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
| Residual-stream direction is sufficient for truthfulness intervention (head-level not needed) | Sprint design assumption | **More clearly false.** Head-level ITI (paper-faithful, K=12) improves TruthfulQA MC on the 2-fold held-out eval (+6.3 pp MC1 [95% CI +3.7, +8.9], +7.49 pp MC2 truthful mass [95% CI +5.28, +9.82]). Residual-stream direction has a narrow window (β=0.01) on FaithEval MCQ. Head-level ITI still fails to improve SimpleQA generation even after the escape hatch is removed, and the random-head control shows that this failure is specific to the ranked configuration rather than generic matched-K perturbation. | Decode-scope ablation now tests whether application policy, not intervention family alone, is the main remaining culprit |
| H-neuron overlap with refusal is measurable via single-vector cosine | D3.5 assumption (now corrected) | Corrected in D3.5 | Overlap appears artificially low because projection step was missing; after projection, if overlap is still <0.1 across the PCA refusal subspace, refusal-overlap explanation is genuinely weak |

## Open Uncertainties

| Uncertainty | Impact | How to resolve | Cost |
|---|---|---|---|
| AdvBench gated on HF — harmful source availability | Affects D0.5a refusal dataset construction | Use `forbidden_question_set.csv` (390) + JBB-Behaviors (100) + other accessible sources. 490 available sources is plenty for 128+32. | Low |
| IFEval constraint checker complexity | Affects D0.5c timeline | Start with subset of simplest constraints (word count, format markers). Full checker can follow. | Medium |
| Truthfulness contrastive quality | Highest single-point-of-failure for D4 | Validate direction separation on seed slice before full curation. Diversity > volume (Liu et al. 2024). | High |
| `device_map="auto"` vs `cuda:0` for direction hooks | Affects per-layer device assignment in extract/intervene scripts | Existing code uses `cuda:0` default. Test both paths. | Low |

## Narrative Log

Narrative, reasoning, and per-session surprises: [research-log.md](./research-log.md).

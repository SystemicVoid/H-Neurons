# Act 3 Final Sprint

> This is the only execution source of truth for current priorities.

## Source Hierarchy

- Execution and priority order: [act3-sprint.md](./act3-sprint.md)
- Evaluation and audit contract: [measurement-blueprint.md](./measurement-blueprint.md)
- Strategic synthesis: [gpt-act3-deep-research-report.md](./gpt-act3-deep-research-report.md)
- Literature reference: [literature-review-act3.md](./literature-review-act3.md)
- Historical pre-pivot planning: [act2-pre-pivot-archive](./act2-pre-pivot-archive)

## North Star

Act 3 is not a last attempt to rescue the original neuron story. It is a final-sprint comparative program: use the H-neuron work as the reference row, evaluate stronger steering baselines with the same high-resolution measurement stack, audit safety externalities directly, and run one scoped causal check.

## Sprint Questions

- Does full-generation, graded evaluation change the practical safety conclusion relative to truncated binary judging?
- Are direction-level baselines cleaner or stronger than the current H-neuron intervention on this model?
- How much of the observed safety regression is explained by overlap with refusal geometry?
- Can one small causal pilot show why correlational localization is a weak intervention selector?

## Committed Deliverables

| ID | Deliverable | Why it exists | Status | Next action |
|---|---|---|---|---|
| D0 | Freeze the Act 3 measurement contract | Every baseline must be judged the same way or the comparison is not publishable | defined in [measurement-blueprint.md](./measurement-blueprint.md) | Use it as a gate on all remaining runs |
| D0.5 | Infrastructure sprint | D2-D6 require code and data that do not exist yet: contrastive datasets, direction extraction, directional intervention, capability mini-battery | not started | (a) Curate 128+128 harmful/harmless contrastive set for Gemma-3-4B-IT, (b) build `extract_direction.py` + directional ablation/addition in the `run_intervention.py` harness, (c) wire BioASQ as accuracy slice and add one instruction-following + one fluency proxy for the capability battery |
| D1 | Baseline A: H-neuron reference row | Preserve continuity with the replication and anchor every comparison to the current intervention | partial | Complete the graded-jailbreak negative control (~4h GPU), run the alpha=1.0 CSV-v2 evaluation (in progress), and attach a capability mini-battery |
| D2 | Extract refusal direction | Prerequisite for D3, D4, D5, D6: the refusal direction is needed both as an intervention vector and as the overlap reference | not started | Difference-in-means on contrastive set from D0.5; validate with ablation sanity check (does removing it suppress refusal on a held-out set?) |
| D3 | Baseline B: refusal-direction intervention | Establish the cheapest strong comparator from the current literature | not started | Run directional ablation/addition on the same benchmarks as Baseline A; compare head-to-head |
| D3.5 | Refusal-overlap analysis for Baseline A | Cheap once D2 is done; informs whether D4 should change approach | not started | Compute cosine similarity between the 38-neuron intervention vector and the refusal direction; report overlap and correlation with CSV-v2 severity |
| **GATE** | **Decision gate** | If H-neuron overlap with refusal direction is high, D4 may need to target a direction orthogonal to both; if low, proceed as planned | — | Review D3.5 result before committing D4 scope |
| D4 | Baseline C: truthfulness or hallucination direction baseline | Test whether a direction-level intervention can beat neuron scaling on the task family the project actually cares about | not started | Build one direction-level truthfulness or hallucination baseline from diverse contrastive data; do not start with a head-level variant |
| D5 | Full externality audit | Turn steering safety drift into a first-class result instead of an afterthought | not started | For Baselines A through C, report jailbreak severity drift, capability drift, response-structure shifts, and overlap with refusal geometry |
| D6 | Refusal-orthogonalized mitigation check | Convert the overlap analysis into a positive safety-aware steering result if possible | not started | Project out the refusal-overlap component from the truthfulness or hallucination vector and compare the before-versus-after tradeoff |
| D7 | One scoped causal pilot | Validate the critique that correlational selection is not the right intervention selector | not started | Attribution patching (2 fwd + 1 bwd) to rank top-20 attention heads by indirect effect on jailbreak refusal for a 50-prompt contrastive subset; compare top-k heads to L1-probe-selected neurons; success = methods select different components AND causal components change CSV-v2 outcome consistently |
| D8 | Final synthesis | Close the sprint with a field-useful protocol and comparative claim, not another idea list | not started | Write the result as a steering protocol: graded evaluation, externality audit, comparator suite, and one causal check |

## Priority Order

1. Enforce the measurement contract (D0).
2. Infrastructure sprint: contrastive data, direction extraction code, capability battery (D0.5).
3. Finish Baseline A as the cleaned reference row (D1).
4. Extract the refusal direction (D2).
5. Run Baseline B: refusal-direction intervention (D3).
6. Compute refusal-overlap for Baseline A (D3.5).
7. **Decision gate**: review overlap result before committing D4 scope.
8. Run Baseline C: truthfulness or hallucination direction (D4).
9. Run the full externality audit across A, B, and C (D5).
10. Run the refusal-orthogonalized mitigation check (D6).
11. Run one scoped causal pilot (D7).
12. Write the final synthesis (D8).

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

## Backlog After The Critical Path

- Negative-weight neuron work is allowed only as a closure experiment after the committed slate is on track. It is not on the critical path.

## Claims To Test

Inherited claims from the deep research report that must be verified against actual data, not assumed:

| Claim | Source | Status | What would falsify it |
|---|---|---|---|
| H-neuron effects are noisy and non-monotonic | Deep research report | **Partially false.** FaithEval is perfectly monotonic (rho=1.0, slope +2.09pp/alpha [1.38, 2.83]). Non-monotonicity applies to jailbreak binary judge only; CSV-v2 graded metric shows a significant monotonic slope (+7.6pp). | FaithEval rho < 0.9 or negative control matching the slope |
| H-neurons are a noisy proxy for the refusal direction | Deep research report | Untested hypothesis | Cosine similarity between 38-neuron intervention vector and refusal direction is low (< 0.3); or refusal direction does not reproduce FaithEval compliance effect |
| FaithEval compliance is mediated by hallucination circuits, not refusal circuits | Implicit in original paper | Untested | Refusal-direction ablation reproduces the FaithEval alpha slope, implying FaithEval compliance is partially a refusal behavior |
| Direction-level interventions are cleaner than neuron-level on this model | Literature consensus | Untested on Gemma-3-4B-IT | Refusal-direction intervention has wider CIs or smaller effect than H-neurons on matched benchmarks |

## Decision Log

- 2026-03-27: Reframed Act 3 as a comparative steering sprint rather than another round of H-neuron optimization.
- 2026-03-27: Kept the deep research report and literature review as reference documents, not live planning docs.
- 2026-03-27: Archived stale pre-pivot planning files under [act2-pre-pivot-archive](./act2-pre-pivot-archive).
- 2026-03-27: Added D0.5 infrastructure sprint, restructured priority order with refusal-direction extraction as early prerequisite and a decision gate before D4, scoped D7 causal pilot concretely, added claims-to-test appendix to prevent inherited errors from deep research report.

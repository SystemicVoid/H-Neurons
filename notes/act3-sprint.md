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
| D1 | Baseline A: H-neuron reference row | Preserve continuity with the replication and anchor every comparison to the current intervention | partial | Complete the graded-jailbreak negative control, add the missing `alpha=1.0` 5000-token baseline, and attach a capability mini-battery |
| D2 | Baseline B: refusal-direction baseline | Establish the cheapest strong comparator from the current literature | not started | Build a harmful-versus-harmless contrastive set and extract a difference-in-means refusal direction |
| D3 | Baseline C: truthfulness or hallucination direction baseline | Test whether a direction-level intervention can beat neuron scaling on the task family the project actually cares about | not started | Build one direction-level truthfulness or hallucination baseline from diverse contrastive data; do not start with a head-level variant |
| D4 | Externality audit plus refusal-overlap analysis | Turn steering safety drift into a first-class result instead of an afterthought | not started | For Baselines A through C, report jailbreak severity drift, capability drift, response-structure shifts, and overlap with refusal geometry |
| D5 | Refusal-orthogonalized mitigation check | Convert the overlap analysis into a positive safety-aware steering result if possible | not started | Project out the refusal-overlap component from the truthfulness or hallucination vector and compare the before-versus-after tradeoff |
| D6 | One scoped causal pilot | Validate the critique that correlational selection is not the right intervention selector | not started | Run one small attribution-patching or GCM-style pilot on a single behavior slice; keep it narrow |
| D7 | Final synthesis | Close the sprint with a field-useful protocol and comparative claim, not another idea list | not started | Write the result as a steering protocol: graded evaluation, externality audit, comparator suite, and one causal check |

## Priority Order

1. Enforce the measurement contract.
2. Finish Baseline A as the cleaned reference row.
3. Run Baseline B.
4. Run Baseline C.
5. Run the externality and refusal-overlap audit.
6. Run the refusal-orthogonalized mitigation check.
7. Run one scoped causal pilot.
8. Write the final synthesis.

## Definition Of Done

A steering baseline is not complete until it has all of the following:

- Full-generation evaluation that avoids systematic truncation.
- The primary safety and response-structure metrics from [measurement-blueprint.md](./measurement-blueprint.md).
- A retained-capability mini-battery.
- Per-example outputs.
- A run manifest with intervention, judge, benchmark, and generation settings.
- A one-paragraph interpretation stating target gain, safety cost, and remaining uncertainty.

## Non-Goals

- Another round of general neuron tinkering as the main story.
- Breadth-first benchmark expansion for its own sake.
- Reviving the legacy 256-token jailbreak narrative.
- Reopening SAE steering.
- Building a full causal-localization stack before a small pilot proves it is worth the setup cost.
- Keeping parallel plan files outside this document and [measurement-blueprint.md](./measurement-blueprint.md).

## Backlog After The Critical Path

- Negative-weight neuron work is allowed only as a closure experiment after the committed slate is on track. It is not on the critical path.

## Decision Log

- 2026-03-27: Reframed Act 3 as a comparative steering sprint rather than another round of H-neuron optimization.
- 2026-03-27: Kept the deep research report and literature review as reference documents, not live planning docs.
- 2026-03-27: Archived stale pre-pivot planning files under [act2-pre-pivot-archive](./act2-pre-pivot-archive).

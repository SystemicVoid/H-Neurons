# Active Plan

This is the canonical execution document for the current phase of the project.

- Put decisions, committed work, and explicit deferrals here.
- Treat `notes/` and `prompts/` as working material, not source of truth.
- When an idea becomes an actual commitment, promote it here.

## Current Phase

Turn the replication-plus-critique work into a hard-to-kill claim package, then use that stronger base to choose one comparative Act 3 experiment. The immediate bottleneck is measurement trustworthiness and capability hygiene, not adding more benchmark surface area.

## Accepted Findings

- H-neuron amplification increases over-compliance on epistemic benchmarks with benchmark-specific negative-control support: FaithEval anti-compliance changes by `+6.3` percentage points with 95% CI `[4.2, 8.5]`, and FalseQA changes by `+4.8` percentage points with 95% CI `[1.3, 8.3]`.
- The corrected 5000-token jailbreak binary effect is inconclusive at current power: `+3.0` percentage points with 95% CI `[-1.2, 7.2]`.
- The graded jailbreak evaluation does recover a safety-relevant slope: CSV-v2 `csv2_yes` changes by `+7.6` percentage points with 95% CI `[3.6, 11.6]`, but this claim is not yet specificity-hardened because the graded metric still lacks a negative control.
- FaithEval standard has a known evaluator-format artifact at high alpha and should not be cited as a raw parser result without remap-aware handling.
- SAE steering is closed as a practical intervention direction for this project.
- The main unresolved risks are evaluator trustworthiness, missing capability-degradation measurement, and a few claim-critical bookkeeping gaps.

## Priority Queue

### P0 Claim Hygiene

| Work item | Why it matters | Planned action | Status |
|---|---|---|---|
| Sentinel sets for judged benchmarks | Protects against silent evaluator drift on known hard cases | Finalize small fixed hard-case sets for FaithEval, FalseQA, and jailbreak; fill remaining `human_label` fields where needed | not started |
| Cross-evaluator audit for headline claims | Prevents one scorer from silently defining the result | Keep raw parser and remap-aware scores where applicable; retain hand-review slices for disputed cases | not started |
| Judge regression fixtures | Locks down already-known brittle behaviors before more benchmark growth | Add regression fixtures for `extract_mc_answer`, `judge_falseqa`, and `judge_jailbreak`, plus at least one metamorphic formatting check | not started |
| Capability mini-battery | Tests whether alpha gains are just collateral damage to general ability | Add a minimal retained-capability assay and use it as a gate on new steering claims | not started |

### P1 Data and Artifact Integrity

| Work item | Why it matters | Planned action | Status |
|---|---|---|---|
| Frozen run manifests | Makes result artifacts auditable and comparable | Record model, tokenizer, classifier identity, benchmark version, judge version, alpha grid, and commit for each claim-relevant run | not started |
| Schema and invariant checks | Catches silent row loss and pairing errors before they reach the site or write-up | Validate sample ID consistency, required fields, and paired-comparison assumptions automatically | not started |
| Per-example output retention | Keeps surprising aggregate results debuggable | Preserve row-level outputs for every headline metric instead of aggregate-only JSON | in progress in parts, not yet complete |

### P2 Jailbreak-Specific Repairs

| Work item | Why it matters | Planned action | Status |
|---|---|---|---|
| Graded jailbreak negative control | Specificity is still missing for the strongest remaining jailbreak claim | Run the random-neuron control for the graded severity metric, not the deprecated 256-token count story | not started |
| Alpha `1.0` 5000-token baseline | Needed for cleaner interpretation of the corrected jailbreak sweep | Add the missing default-activation baseline to the long-generation setup | not started |
| Judge calibration slice | Checks whether disclaimer-heavy harmful outputs still fool the rubric | Keep a small blind human-review slice across truncation-sensitive and disclaimer-heavy cases | partially done, needs formalization |

### P3 Throughput Instrumentation

| Work item | Why it matters | Planned action | Status |
|---|---|---|---|
| Safe mid-run timing | Gives visibility without changing generation outputs | Log per-sample wall time, per-sample W&B timing, and per-alpha summaries | not started |
| Hook-tax measurement | Tests whether hook overhead is worth optimizing at all | Add cumulative hook timing and one small A/B after the current run shape is stable | not started |

## Explicit Deferrals

- Breadth-first benchmark expansion without stronger claim hygiene.
- Full paper-faithful replication extras that do not change a current decision.
- Large infrastructure cleanups that are not directly connected to measurement trust or experiment selection.

## Closed Or Not For Now

- SAE steering revisits: closed for this project unless a later result changes the underlying causal picture.
- Full circuit-discovery stacks as a primary next step: not for now; too much setup cost relative to immediate decision value.
- Another 256-token jailbreak-heavy rerun: not useful now that truncation is a known measurement artifact.

## Working Rules

- If a task mostly reduces the chance that an evaluator or bookkeeping bug survives into a claim, it beats a task that only adds another benchmark.
- Every new intervention claim should ship with a retained-capability check or an explicit note that the claim is still missing one.
- Promote only agreed work into this file; keep open-ended exploration in [experiment-portfolio.md](./experiment-portfolio.md).

## Decision Log

- 2026-03-27: Established this file as the canonical execution plan for the current phase.
- 2026-03-27: Left `ROADMAP.md`, `notes/nextsteps.md`, `notes/scratchpad.md`, and `prompts/extensions-discussion.md` unchanged as source material and historical context.
- 2026-03-27: Prioritized measurement hardening and capability hygiene before choosing one broader Act 3 comparison experiment.

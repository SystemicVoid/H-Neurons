# Measurement Blueprint For Act 3

> This is the canonical evaluation and audit contract for every Act 3 claim.

## Scope

This contract applies to Baselines A through C, the refusal-overlap audit, the refusal-orthogonalized mitigation check, and the scoped causal pilot.

## Headline Rules

- Do not use systematically truncated generations as headline evidence.
- For jailbreak-style safety claims, graded severity is the primary endpoint; binary harmful versus safe is diagnostic only.
- Every quantitative headline claim must include uncertainty.
- Every steering claim must report both target behavior and safety externality.
- Do not redefine metrics ad hoc inside experiment notes; update this file first.

## Required Generation Policy

- Use generation settings that eliminate systematic truncation for the evaluated population.
- Do not cite the legacy 256-token jailbreak setup as current evidence.
- Keep generation settings fixed across compared interventions unless the comparison is explicitly about generation settings.

## Standard Battery

### Target-Behavior Metrics

- Report the primary task metric on the benchmark the intervention is meant to improve.
- Keep the benchmark-side metric identical across the compared baselines.

### Primary Safety Metrics

- Use the current graded jailbreak pipeline as the primary safety readout.
- Keep `csv2_yes` or its direct successor as the top-line strict harmful metric.
- Track severity-sensitive components such as strong harmful content and high-specificity harmful outputs whenever the current rubric supports them.

### Response-Structure Metrics

- Keep harmful payload share.
- Keep pivot position.
- Keep disclaimer-free or disclaimer-persistence rate.
- Keep per-prompt churn when a comparison is sensitive to prompt-level instability.
- Use the current CSV-v2 pipeline definitions for these metrics; do not redefine them per report.

### Capability Mini-Battery

Every steering comparison must include all three:

- One unrelated epistemic accuracy slice.
- One harmless instruction-following slice.
- One lightweight fluency or language-model-quality proxy.

## Steering Externality Audit

Every steering method must report:

1. Target-behavior gain.
2. Graded jailbreak or refusal robustness drift.
3. Capability mini-battery outcome.
4. Response-structure shifts.
5. Refusal-overlap geometry when a vector-based intervention is available.

## Judge Trustworthiness Requirements

- Maintain sentinel sets for FaithEval, FalseQA, and jailbreak-style judged outputs.
- Add or keep regression fixtures for `extract_mc_answer`, `judge_falseqa`, and `judge_jailbreak`.
- Keep a blind human-review slice for truncation-sensitive and disclaimer-heavy cases.
- Run a cheap negative control whenever a new headline steering claim would otherwise lack specificity support.
- Retain per-example outputs for every headline metric.

## Run Manifest Requirements

Each claim-relevant artifact must record:

- Model and tokenizer identifiers.
- Intervention method and intervention strength.
- Component-selection method or source dataset.
- Benchmark version and prompt style.
- Generation settings.
- Judge model and judge prompt version.
- Sample IDs and pairing assumptions.
- Git commit or equivalent provenance identifier.
- Paths to row-level outputs and summaries.

## Reporting Template

Each result table, JSON summary, or write-up section must state:

- Which metric is primary and which are diagnostic.
- Sample count.
- Point estimate and uncertainty.
- Whether the result is target gain, safety cost, or both.
- Known judge blind spots that matter to interpretation.
- Whether the capability mini-battery passed, failed, or remains missing.

## Working Rule

If a result cannot survive this measurement contract, it is not yet a sprint-level claim.

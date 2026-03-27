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

## Minimum Detectable Effect By Benchmark

Before running a new intervention on a benchmark, check that the expected effect size exceeds the minimum detectable effect (MDE) for that sample size. If it does not, either increase the sample size or choose a more sensitive metric.

Estimates below are based on existing data (paired bootstrap, 80% power, alpha=0.05):

| Benchmark | N | Metric | Observed H-neuron effect | Approximate MDE (paired, 80% power) | Notes |
|---|---|---|---|---|---|
| FaithEval | 1000 | Compliance rate | +6.3pp [4.2, 8.5] | ~3pp | Well-powered for effects comparable to H-neurons |
| FalseQA | 687 | Compliance rate | +4.8pp [1.3, 8.3] | ~4pp | Borderline — effects below 4pp may not reach significance |
| Jailbreak | 500 | csv2_yes (graded) | +7.6pp [3.6, 11.6] | ~5pp | Binary judge is underpowered (MDE ~6pp); use CSV-v2 |
| Jailbreak | 500 | Binary compliance | +3.0pp [-1.2, 7.2] | ~6pp | Insufficient — binary metric does not detect the real effect at n=500 |
| BioASQ | 1600 | Compliance rate | -0.06pp [-1.5, 1.4] | ~2pp | Well-powered null; can detect effects >2pp |

If a new intervention is expected to produce effects smaller than the H-neuron baseline (e.g., a subtle truthfulness direction), consider running on FaithEval (n=1000, MDE ~3pp) first rather than FalseQA or jailbreak.

## Negative Control Requirements

- Every new intervention method requires at least one negative control: a random-direction or shuffled-component baseline run through the same pipeline.
- Exception: if the intervention vector is extracted from a specific contrastive dataset (not randomly sampled), state why specificity is established by construction and what alternative confound remains (e.g., dataset leakage, geometric coincidence).
- For neuron-level interventions, the standard is multi-seed random-neuron sets (as done for FaithEval with 5 seeds, FalseQA with 3 seeds).
- For direction-level interventions, the standard is at least one random-direction baseline of the same norm.

## Cross-Benchmark Consistency

Every steering comparison must state:

- Which benchmarks the intervention was evaluated on.
- Whether the effect direction (positive, negative, null) is consistent across benchmarks.
- If inconsistent, which benchmark is primary and why.
- Known reasons for divergence (e.g., BioASQ is a factoid-QA format where neuron scaling has no effect because it is not a compliance task).

## Working Rule

If a result cannot survive this measurement contract, it is not yet a sprint-level claim.

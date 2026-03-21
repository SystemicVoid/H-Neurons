# Roadmap

## Research

### Measurement Hardening — highest ROI next

The main sprint risk is not "we cannot run enough evals." It is "we may believe the wrong eval." The repo has already surfaced real evaluator artifacts and bookkeeping edge cases, so the highest-return work is calibration and invariant-checking around the existing harness before expanding benchmark/tooling scope.

#### 1. Hand-labeled sentinel sets for evaluator regression

Keep a small fixed set of hard cases for each judged benchmark:

- `FaithEval standard` parse-failure cases
- `FalseQA` false-premise edge cases
- future `jailbreak` borderline safe-vs-harmful responses

Why this is first:

- `FaithEval standard` already showed a genuine evaluator artifact: raw compliance at `alpha=3.0` is `63.6%`, but strict answer-text remapping lifts it to `72.1%` by recovering `140/150` parse failures.
- `FalseQA` is already tracked as having judge-model error outside CI accounting, so judge drift is an active methodological risk.

Evidence:

- `data/gemma3_4b/intervention/faitheval_standard/experiment/results.json`
- `site/data/intervention_sweep.json`
- `docs/ci-rollout-status.local.md`

Analogy:

- This is the calibration-weight set for the scale. You trust the evaluator because it still gets the fixed hard cases right, not because it looks tidy.

#### 2. First-class cross-evaluator audits for headline claims

Do not rely on one scorer when a second operationalization is cheap.

Minimum standard:

- keep raw parser score
- keep text-remap score
- keep a hand-review slice for disputed cases
- report when the conclusion depends on scorer choice

Why this is high ROI:

- The repo already found the exact failure mode this is meant to catch.
- Current scoring is split across inline rules, parser extraction, and LLM-as-judge; that is acceptable only if disagreement is measured rather than hidden.

Evidence:

- `scripts/run_intervention.py`
- `scripts/evaluate_intervention.py`
- `site/data/intervention_sweep.json`

Analogy:

- One thermometer is a reading. Two thermometers are an instrument check.

#### 3. Freeze run manifests plus schema/invariant checks

Every committed result artifact that might feed a claim should record:

- model/tokenizer identifiers
- classifier path or hash
- benchmark version and prompt style
- judge model and judge prompt version
- alpha grid
- code commit

Then validate invariants automatically:

- no duplicate sample IDs
- same IDs across alpha sweeps when paired comparisons are claimed
- required fields present with expected types
- no silent row loss for headline summaries unless explicitly documented

Why this is high ROI:

- The repo already benefits from partial invariant checks. For example, `export_site_data.py` checks anti-compliance sample ID consistency across alphas before computing population-level summaries.
- The CI tracker already records subset drift issues, e.g. the disjoint classifier summary reflects `780` evaluated examples despite `782` sampled IDs.

Evidence:

- `scripts/export_site_data.py`
- `docs/ci_manifest.json`
- `docs/ci-rollout-status.local.md`

Analogy:

- This is the chain-of-custody layer for experiments. It does not improve the science directly; it makes it much harder for silent bookkeeping bugs to impersonate science.

#### 4. Preserve per-example outputs for every headline metric

Do not let important claims collapse to aggregate-only JSON.

Why this is high ROI:

- Surprising results can only be debugged from rows, not from aggregates.
- The CI tracker already records a concrete failure mode: some historical aggregates cannot be bootstrapped because per-example predictions were not retained.

Evidence:

- `docs/ci-rollout-status.local.md`
- `scripts/classifier.py`
- `scripts/export_site_data.py`

Analogy:

- Aggregates are the paper table. Rows are the lab notebook.

#### 5. Judge regression tests before adding more benchmark surface area

Before broadening the eval stack, lock down the tricky judge behavior already known to be brittle.

Minimum scope:

- regression fixtures for `extract_mc_answer`
- regression fixtures for `judge_falseqa`
- regression fixtures for `judge_jailbreak`
- at least one metamorphic check where semantically irrelevant formatting changes should not flip the label

Why this outranks framework migration:

- The current bottleneck is evaluator trustworthiness, not missing eval runner features.
- `sycophancy` and `jailbreak` are scaffolded already; the current blocker is result quality and judged data, not lack of framework support.

Evidence:

- `tests/test_utils.py`
- `scripts/evaluate_intervention.py`
- `docs/ci-rollout-status.local.md`

Working rule:

- If a task reduces the chance that a subtle evaluator or bookkeeping bug survives into a claim, it should usually beat a task that merely makes it easier to run one more benchmark.

### FalseQA Negative Control — done

H-neuron specificity confirmed. Full audit: [`data/gemma3_4b/intervention/falseqa/falseqa_negative_control_audit.md`](data/gemma3_4b/intervention/falseqa/falseqa_negative_control_audit.md). Integrated into `intervention_findings.md` §1.6–1.7.

### BioASQ Pipeline Audit — done

Completed. Full audit: [`data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md`](data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md). Integrated into `intervention_findings.md` as a linked side report rather than a new headline causal benchmark.

### SAE Decomposition Investigation — done (negative result, definitively closed)

All 5 phases completed (feasibility, extraction, probe training, interpretability, steering). Gate 3 outcome: **FAIL**. Full-replacement SAE steering slope 0.16 pp/α (CI contains zero) vs neuron baseline 2.09 pp/α. Delta-only falsification test (2026-03-21) confirmed the failure is fundamental feature-space misalignment, not reconstruction noise: delta-only H-feature slope 0.12 pp/α, random -0.09 pp/α — both indistinguishable from zero while the neuron baseline on the same alphas is 2.12 pp/α. SAE detection (AUROC 0.848) marginally matches CETT (0.843) but uses 7× more features and does not translate to causal control under either steering architecture. Full plan: [`docs/sae_investigation_plan.md`](docs/sae_investigation_plan.md). Pipeline audit: [`data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md`](data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md). Integrated into `intervention_findings.md` §1.9–1.10 and Finding 6.

---

Infrastructure improvements deferred for future consideration.

## Infrastructure

### Structured Logging

Replace ad-hoc `print()` calls across pipeline scripts with Python `logging` at
INFO/DEBUG levels. Long-running intervention and activation-extraction jobs
benefit from timestamped, level-filtered output -- especially with the
suspend/resume workflow on Pop!_OS where post-wake debugging currently relies on
scrollback grep.

### AGENTS.md Staleness Detection

AGENTS.md goes stale when codebase behaviour changes --
e.g. a new shared module is extracted, test infrastructure is added, or a
workflow step is removed -- but nobody updates the guidelines.

Possible approaches:
- A pre-commit hook that flags when files in `scripts/` or `tests/` change but
  AGENTS.md does not (noisy, but catches the obvious case).
- A periodic `/readiness-report`-style agent task that diffs AGENTS.md claims
  against the actual file tree and import graph.
- Convention: any commit that changes project structure or workflow must include
  a corresponding AGENTS.md update (enforced by review, not tooling).

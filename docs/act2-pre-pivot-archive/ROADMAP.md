# Roadmap

## Research

### Measurement Hardening — highest ROI next


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


NOT done :
 Full remap-sweep across all FaithEval standard alphas (separate work)
   •  Rewriting evaluate_intervention.py (no changes to existing evaluation pipeline)
   •  Metamorphic formatting checks (roadmap item #5 territory)

   Next step for you: Review the FalseQA and jailbreak candidate files and fill in human_label fields. The validation
   script will skip cases with null labels until you do.



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

### Throughput Instrumentation — next after current run

The canonical jailbreak run (5000-token, 3 alphas x 500 samples, ~15 hr) exposed sensor gaps that block all further optimisation decisions. See `docs/throughput-assessment.md` §Sensor Gap Analysis and §Revised Ranked Next Steps for full details.

**Safe to add mid-run** (do not affect generation outputs):
- Wall-clock timestamps per sample (`time.time()` at loop boundaries)
- Per-sample W&B streaming (`wandb.log()` with `generate_s`, tok/s)
- Per-alpha summary (wall time, samples/sec, mean tok/s)

**Requires A/B after run completes:**
- Cumulative hook timer in `HNeuronScaler` to measure the 23-callbacks-per-step tax
- 50-sample A/B: hooks installed vs hooks removed vs wrapped `down_proj` forward

Key finding from current data: `model.generate()` is 100% of measured time at 34.2 tok/s with zero drift. The hook tax exists but is invisible without per-step instrumentation — cannot justify any optimisation until it is measured.

## What the paper actually says

For **jailbreak**, the paper explicitly uses the *forbidden question set* with **390** test cases, pairs each harmful query with **a jailbreak template**, and generates with **temperature=0.7, top_k=20, top_p=0.8, maximum output length 256 tokens**, then evaluates with a **GPT-4o judge** guided by **15 benchmark examples**. So yes: the 256-token cap is in the paper text, not an inference. 

But the paper’s **core H-neuron identification** is **not** built on jailbreak at all. They identify neurons from **TriviaQA** using a sparse linear classifier over answer-token neuron contributions, and they tune the sparsity tradeoff using held-out classification accuracy plus **TriviaQA performance under suppression**. Jailbreak is only one downstream perturbation benchmark among four.

The paper’s main behavioral claim is broader than jailbreak: amplifying H-neurons tends to raise over-compliance across **FalseQA, FaithEval, Sycophancy, and Jailbreak**, though they explicitly note the response is **not strictly monotonic** and that Jailbreak can fluctuate at intermediate scaling factors. They also say later that **simple suppression or amplification is insufficient for effective control** and future work needs more sophisticated interventions.

That matters a lot, because it means your jailbreak truncation issue threatens **one downstream measurement**, not the paper’s whole neuron story. The paper itself is basically saying: “these neurons correlate with a broad over-compliance axis; jailbreak is one readout.”


## Revised recommendation


### 4) Add a **judge calibration audit**

This is the part I underemphasized before, and I now think it is essential.

Take **30–50 cases**, stratified toward:

* originally SAFE but truncated
* disclaimer-heavy continuations
* borderline cases near truncation point

Then do **blind human relabeling** on:

* the original 256-token outputs
* the 1024-token outputs

You do not need a giant annotation project. Even 30–50 carefully chosen examples is enough to estimate whether the GPT-4o judge is systematically fooled by “moral throat-clearing followed by actionable harm.”

Because right now your problem is probably:

* **truncation bias**
* plus **judge framing bias**

If you only fix the first, you may still be measuring through dirty glass.

## What to do about the negative control

Here is the corrected version of my earlier advice:

If the negative control is mostly about **mechanistic specificity** or the non-jailbreak over-compliance story, I would still run it. The paper’s main contribution is the broader over-compliance link, not just jailbreak ASR.

If the negative control is specifically another **256-token jailbreak-heavy** run, pause it. No point tuning a thermometer you already know is in the oven.

My revised recommendation is:

**Keep the current full 256-token sweep.
Do not full-rerun jailbreak yet.
Run:**

1. **1024-token audit at alpha 0.0**
2. **1024-token matched audit at one additional alpha**
3. **small human judge-calibration audit**
4. **continue non-jailbreak negative control if it tests mechanism rather than 256-token jailbreak**

That is the highest signal-per-hour move, and it matches the paper better than my earlier advice because it respects where the paper’s core claims actually live. The paper’s own discussion also warns that simple activation scaling is a crude control knob, so the goal here is not to perfect every jailbreak datapoint; it is to establish whether your downstream claim survives measurement repair.



1. Probe-family robustness

Train several detectors:

dense linear
sparse linear at several C values
maybe logistic vs linear SVM style equivalent
different train/test seeds or folds

Ask:

what signal is stable across models?

That tells you what is real versus optimizer bookkeeping.

2. Directional intervention

Use the probe vector itself as the object.

Test:

move activations slightly against the jailbreak direction
measure jailbreak suppression
measure retention on harmless tasks

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

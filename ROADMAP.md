# Roadmap

## Research

### FalseQA Negative Control (complete)

**Scientific question:** Is the H-neuron compliance effect on FalseQA specific to H-neurons, or does scaling any 38 random neurons produce a similar trend?

**Status:** Complete. **H-neuron specificity confirmed on FalseQA.**

**Results** (`data/gemma3_4b/intervention/falseqa/control/comparison_summary.json`):
- H-neurons: 69.6% → 71.9% → 74.4% (slope +1.55%/α, ρ=1.0, monotonic)
- Random mean: 72.3% → 72.2% → 72.3% (slope 0.00%/α, flat)
- Random 95% slope interval: [-0.40, +0.38] pp/α — H-neuron slope of +1.55 is well outside
- H-neuron α=3.0 compliance (74.4%) is above the random 95% interval [71.8%, 72.9%]
- Plot: `data/gemma3_4b/intervention/falseqa/control/negative_control_comparison.png`

This is the second benchmark (after FaithEval) confirming H-neuron specificity, using a completely different evaluator (GPT-4o judge vs regex letter extraction) and task type (open-ended false-premise rejection vs MC counterfactual context).

**Pipeline:**
1. **Generation** (`scripts/run_negative_control.py --benchmark falseqa --quick`): 3 unconstrained random seeds x 3 alphas (0.0, 1.0, 3.0) x 687 FalseQA samples = 6,183 responses.
2. **GPT-4o judging** (`scripts/evaluate_intervention.py --benchmark falseqa`): Same judge prompt as the H-neuron FalseQA baseline for apples-to-apples comparison. ~$2.50 API cost.
3. **Analysis** (`scripts/run_negative_control.py --benchmark falseqa --quick --analysis_only`): Comparison summary, slope intervals, and plot.

**Data:**
- Generated responses: `data/gemma3_4b/intervention/falseqa/control/seed_{0,1,2}_unconstrained/alpha_{0.0,1.0,3.0}.jsonl`
- Per-seed results: `data/gemma3_4b/intervention/falseqa/control/seed_{0,1,2}_unconstrained/results.json`
- H-neuron FalseQA baseline: `data/gemma3_4b/intervention/falseqa/experiment/results.json` (7 alphas, 687 samples, GPT-4o judged)

**Reviewer note:** The quick-mode run only tested 3 alphas (0.0, 1.0, 3.0) with 3 unconstrained seeds. The FaithEval control used 7 alphas with 5 unconstrained + 3 layer-matched seeds. A full FalseQA sweep (8 seeds x 7 alphas) would cost ~$16 API + ~8h GPU and is warranted if the result needs to go into a publication figure.

**Scripts modified:**
- `scripts/run_negative_control.py` — extended with `--benchmark {faitheval,falseqa,bioasq}` flag, benchmark-specific generation loops (`_run_faitheval_alphas`, `_run_falseqa_alphas`, `_run_bioasq_alphas`), deferred-judging workflow for FalseQA, benchmark-aware `build_comparison_summary()` and `plot_comparison()`
- `scripts/run_intervention.py` — added `load_bioasq()`, `run_bioasq()`, BioASQ CLI wiring

### BioASQ H-Neuron Intervention (in progress)

**Scientific question:** Does amplifying H-neurons causally change factoid accuracy on an out-of-distribution biomedical QA benchmark?

**Context:** The probe transfer audit (`data/gemma3_4b/probing/bioasq13b_factoid/probe_transfer_audit.md`) shows H-neurons are *detectable* on BioASQ (AUROC 0.82, accuracy 0.70). The intervention tests whether they are *causally active* for factoid accuracy — a stronger claim.

**Status:** H-neuron baseline complete, negative control running.

**H-neuron baseline results** (`data/gemma3_4b/intervention/bioasq/experiment/results.json`):
- α=0.0: 16.9% accuracy (270/1600)
- α=1.0: 18.6% accuracy (297/1600)
- α=3.0: 16.8% accuracy (269/1600)
- **Flat curve — no dose-response.** H-neuron scaling does not affect BioASQ factoid accuracy despite neurons being detectable. This dissociation between detection and causal intervention is scientifically interesting.

**Negative control results** (`data/gemma3_4b/intervention/bioasq/control/comparison_summary.json`):
- Random neurons: 18.6% → 18.6% → 18.7% across α (perfectly flat, slope +0.04%/α)
- H-neurons: 16.9% → 18.6% → 16.8% (non-monotonic V-shape, slope -0.14%/α)
- Both are flat and indistinguishable from noise at n=1600 (Wilson 95% CI width ~3.8pp)
- The H-neuron α=0.0 ablation dip to 16.9% vs random 18.6% is the only hint of a signal but does not survive the CI

**Key anomaly: detection-without-intervention dissociation.** The probe audit shows H-neurons carry hallucination-predictive signal on BioASQ (AUROC 0.82). But scaling them 0x--3x does nothing to factoid accuracy. This is the opposite of FaithEval, where H-neurons both predict and causally drive compliance. Something is structurally different about how H-neurons relate to BioASQ factoid behavior.

**Data:**
- BioASQ benchmark: `data/benchmarks/bioasq13b_factoid.parquet`
- BioASQ samples (1600 questions, GPT-4o judged): `data/gemma3_4b/probing/bioasq13b_factoid/samples.jsonl`
- H-neuron baseline: `data/gemma3_4b/intervention/bioasq/experiment/results.json`
- Negative control: `data/gemma3_4b/intervention/bioasq/control/comparison_summary.json`
- Negative control plot: `data/gemma3_4b/intervention/bioasq/control/negative_control_comparison.png`
- Probe transfer audit: `data/gemma3_4b/probing/bioasq13b_factoid/probe_transfer_audit.md`

**Evaluation method:** Inline `normalize_answer()` substring matching against BioASQ alias lists — no GPT-4o cost for judging.

### BioASQ Dissociation: Next Steps to Investigate

The detection-without-intervention result on BioASQ needs explanation. It could be a genuine mechanistic finding (H-neurons encode hallucination status but don't gate factoid retrieval on OOD domains) or an artifact of the evaluation setup. The following investigations are ordered by cost and diagnostic value.

**1. Check whether the evaluation metric is too coarse (zero GPU cost, ~1h)**
BioASQ factoid accuracy is ~18% — the model gets most questions wrong regardless of alpha. If the "floor" is already at random-guess level, H-neuron scaling might be changing *how* the model is wrong (e.g., shifting from confident wrong answers to hedged wrong answers) without changing the binary correct/incorrect count. Concrete steps:
- Load `data/gemma3_4b/intervention/bioasq/experiment/alpha_{0.0,1.0,3.0}.jsonl` and compare response *text* across alphas for the same questions
- Classify responses into: correct, wrong-but-confident, wrong-and-hedged, refusal, off-topic
- Check if the distribution across these categories shifts with alpha even when binary accuracy doesn't
- Check average response length per alpha — format degradation would show up here

**2. Per-question alpha sensitivity analysis (zero GPU cost, ~1h)**
Replicate the swing-sample methodology from FaithEval (`data/gemma3_4b/swing_characterization/`):
- For each of the 1600 questions, track correctness across α=0.0, 1.0, 3.0
- Identify any "swing" questions that flip correct↔incorrect with alpha
- If swing questions exist but cancel out (some flip correct→wrong, others wrong→correct), the aggregate metric hides a real effect
- If no swing questions exist, the intervention genuinely has no effect at the per-item level
- Compare swing rate to FaithEval's 14% swing subpopulation

**3. Check task-type mismatch (zero GPU cost, ~30min)**
FaithEval tests *compliance with misleading context* — a decision the model makes at generation time. BioASQ tests *factual recall from parametric knowledge* — whether the model ever learned the answer during pre-training. H-neurons may specifically gate the compliance/refusal decision rather than the factual retrieval pathway. To probe this:
- Cross-reference BioASQ questions with the probe audit's false-positive analysis: the detector overcalls verbose faithful answers. Are the questions where the model is correct at α=1.0 also the ones where the probe predicts "truthful"?
- Check whether H-neuron activation magnitudes at BioASQ answer positions correlate with correctness as strongly as they do on TriviaQA (the probe says yes with AUROC 0.82, but the intervention says the causal arrow is missing)

**4. Run the intervention with a compliance-framed BioASQ prompt (GPU cost: ~2h)**
The current BioASQ prompt is bare factoid QA (`{question} Respond with the answer only...`). The FaithEval prompt explicitly frames a conflict between context and knowledge. Hypothesis: H-neurons only causally activate when there's a compliance *decision* to make. Test:
- Construct a counterfactual-context version of BioASQ: prepend a misleading context paragraph (e.g., "According to recent research, the answer is X" where X is wrong) and ask the model to answer
- Run the same alpha sweep on this compliance-framed version
- If H-neuron scaling now shows a dose-response, the dissociation is explained: these neurons gate compliance decisions, not factual retrieval

**5. Inspect the H-neuron activation distributions on BioASQ directly (GPU cost: ~30min)**
The probe transfer audit used saved activations from single responses. Run `scripts/extract_activations.py` on the intervention JSONL files to get activations at each alpha value, then:
- Check whether H-neuron activations at α=1.0 on BioASQ have similar magnitude/variance to TriviaQA
- Check whether the scaling hook is actually reaching the right tokens (answer tokens vs prompt tokens)
- This would catch a bug where the hook scales neurons during prompt processing but the model's answer is determined before the scaled tokens matter

### Summary of Existing Negative Control Results

For reference, the FaithEval anti-compliance negative control is complete and conclusive:
- **H-neuron slope:** 2.09%/α (monotonic, ρ=1.0, 6.3pp total swing)
- **Random neuron slope:** 0.02%/α mean (5 unconstrained seeds), 0.17%/α mean (3 layer-matched seeds)
- **Separation:** t-test p < 10⁻⁵ on both compliance and slope
- **Full analysis:** `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`
- **Detailed write-up:** `data/gemma3_4b/intervention_findings.md` (sections 1.4, 1.5)

---

Infrastructure improvements deferred for future consideration.

## Ifrastructure

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

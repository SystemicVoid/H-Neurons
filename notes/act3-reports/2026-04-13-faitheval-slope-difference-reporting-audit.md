# FaithEval Slope-Difference Reporting Audit — 2026-04-13

> **Verdict: queue-ready for the narrow FaithEval slope-difference reporting path.**
> The committed pipeline outputs are internally consistent, independently
> reproducible from raw JSONL rows, correctly exported into
> `site/data/intervention_sweep.json`, and supported by targeted tests.
> The strongest publication-safe reading is:
> **on the committed full FaithEval anti-compliance sweep, H-neuron scaling has a
> steeper compliance slope than both random-neuron controls and the full-replacement
> SAE readout.**
>
> The stronger causal claim
> ("SAE features fail even after removing reconstruction confounding")
> still belongs to the earlier delta-only SAE audit and should not be silently
> replaced by this reporting-path result.

## Source Hierarchy

- H-neuron reference report: [intervention_findings.md](../../data/gemma3_4b/intervention_findings.md)
- SAE closure audit: [sae_pipeline_audit.md](../../data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md)
- Sprint context: [act3-sprint.md](../act3-sprint.md)
- Strategic framing: [2026-04-11-strategic-assessment.md](../2026-04-11-strategic-assessment.md)

This report owns only the **paired slope-difference reporting-path audit** for the
FaithEval anti-compliance benchmark. It does not supersede the broader SAE
closure audit or the project-wide intervention synthesis.

## Files Reviewed

### Pipeline code

- `scripts/compute_faitheval_slope_difference.py`
- `scripts/export_site_data.py`
- `scripts/uncertainty.py`

### Tests

- `tests/test_compute_faitheval_slope_difference.py`
- `tests/test_export_site_data.py`
- `tests/test_uncertainty.py`

### Generated artifacts

- `data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json`
- `data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json`
- `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json`
- `site/data/intervention_sweep.json`
- `logs/faitheval_slope_pipeline_20260413T180806Z.log`

### Raw trajectory sources spot-checked directly

- `data/gemma3_4b/intervention/faitheval/experiment/alpha_{0.0..3.0}.jsonl`
- `data/gemma3_4b/intervention/faitheval_sae/experiment/alpha_{0.0..3.0}.jsonl`
- `data/gemma3_4b/intervention/faitheval/control/seed_*/alpha_{0.0..3.0}.jsonl`

## 1. Data

### 1.1 Pipeline completion and artifact validation

The logged three-step pipeline completed with exit code 0 and no mid-run code
changes:

1. `uv run python scripts/compute_faitheval_slope_difference.py`
2. `uv run python scripts/export_site_data.py`
3. `uv run python scripts/audit_ci_coverage.py`

The log confirms the intended outputs were written and that CI audit warnings
were unrelated to the FaithEval slope-difference claim.

### 1.2 Independent raw-data recheck

I independently reloaded the committed JSONL trajectories and recomputed the
simple OLS slopes from the per-alpha compliance rates.

Verified exactly:

| Condition | n items | Compliance rates (%) | Slope (pp/α) |
|---|---:|---|---:|
| H-neurons | 1000 | 64.2, 65.4, 66.0, 67.0, 68.2, 69.5, 70.5 | 2.092857 |
| SAE h-features | 1000 | 72.3, 74.7, 66.0, 75.0, 75.1, 74.9, 69.9 | 0.164286 |

All 8 random-neuron controls also had:

- 1000 items each
- full item-ID overlap with the H-neuron run
- slopes matching the saved summaries:
  - unconstrained: +0.17, -0.00, -0.07, -0.11, +0.11 pp/α
  - layer-matched: +0.21, +0.16, +0.15 pp/α

This matters because it rules out the easy failure mode where the new summaries
are numerically correct only relative to a misaligned sample order. They are not:
the trajectories align 1000/1000 by sample ID.

### 1.3 Saved summary numbers

The committed summary artifacts report:

#### Matched readout comparison: neuron minus SAE

- H-neuron slope: **+2.09 pp/α**
- SAE slope: **+0.16 pp/α**
- Paired slope difference: **+1.93 pp/α**
- 95% bootstrap CI: **[+0.94, +2.92]**
- Directional permutation p-value: **9.9998e-05**
- `n_items = 1000`

#### Random-neuron specificity comparison

- 8/8 seed-specific paired slope differences are positive
- Mean paired slope difference: **+2.01 pp/α**
- Seed range: **+1.89 to +2.20 pp/α**
- Every seed-specific 95% bootstrap CI excludes zero
- Every seed-specific directional permutation test returns the minimum non-zero
  p-value achievable with 50,000 permutations under this implementation:
  **1.99996e-05**

### 1.4 Export wiring

The site payload is wired correctly.

Verified fields in `site/data/intervention_sweep.json`:

- `matched_readout_comparison.comparison == "neuron_minus_sae"`
- `matched_readout_comparison.n_items == 1000`
- `negative_control.paired_slope_difference.aggregate.n_seeds == 8`
- both slope-difference summary JSONs appear in
  `provenance.source_files`

### 1.5 Targeted regression tests

`uv run pytest tests/test_compute_faitheval_slope_difference.py tests/test_export_site_data.py tests/test_uncertainty.py`

Result: **15 passed**

What those tests cover:

- missing-control and misaligned-ID failures raise hard errors
- slope-difference logic detects known divergence and null cases
- the site export contract includes the new slope-difference summaries

## 2. Interpretation

### 2.1 What clearly withstands scrutiny

#### A. H-neuron specificity on FaithEval is strengthened, not weakened

The older FaithEval control claim already showed separation by empirical random
intervals. The new paired analysis is stronger because it compares trajectories
item-by-item rather than only comparing aggregate seed means.

Plainly: the random controls are not just "flat on average." Each one loses to
the H-neuron trajectory on the same 1000 questions.

#### B. The reporting-path statement "neuron beats committed SAE readout" is supported

For the specific pipeline now exported to the site, the correct statement is:

> On the committed full FaithEval anti-compliance sweep, the H-neuron
> intervention has a steeper compliance slope than the full-replacement SAE
> h-feature intervention by **+1.93 pp/α [0.94, 2.92]** on matched items.

That statement is directly backed by the saved JSON, the raw JSONL recheck, and
the export contract.

#### C. The narrow release decision is justified

For the **reporting path** only, the user's "queue-ready" assessment is correct.
The pipeline produced the intended artifacts, the artifacts are coherent, and
the site export consumes them as intended.

### 2.2 What is supported but needs careful wording

#### A. This is not the whole SAE dissociation by itself

The new matched-readout comparison uses the committed
**full-replacement SAE run**, not the later delta-only corrective experiment.
That means the result is excellent evidence for the reporting-path comparison,
but it is not by itself the cleanest causal answer to "do SAE features still
fail after removing reconstruction confounding?"

That stronger claim remains anchored by the older delta-only audit:

- full-replacement SAE: reporting-path evidence
- delta-only SAE: confound-removal evidence

The two reports should be cited together when making the strongest paper claim.

#### B. The permutation p-values are directional

The implementation reports `alternative = "one_sided_greater"`.
That is acceptable if the hypothesis was prespecified as
"H-neurons have a steeper slope than controls/SAE."
It is not the right framing if someone later tries to repurpose the number as a
generic symmetric model-comparison test.

The safest publication habit is:

- lead with the point estimate and 95% CI
- treat the permutation p-value as directional supporting evidence

#### C. The random-seed aggregate is descriptive, not a pooled inferential CI

`mean_slope_difference = +2.01 pp/α` is a useful summary, but it is not a
bootstrap confidence interval over the full random-neuron universe. It is the
mean over 8 sampled control seeds, with min/max range retained for context.

That is the right cautious choice here. It should stay described as an
**empirical seed summary**, not as a population estimate with asymptotic
pretensions.

### 2.3 What does not follow from this audit

This audit does **not** establish any of the following on its own:

- that H-neurons are the unique or optimal steering target in a broad mechanistic
  sense
- that the SAE line failed only because of feature-space mismatch
- that the same level of specificity is already established for jailbreak
- that `site/data/intervention_sweep.json` is a benchmark index

That last point is a small but real structural note: despite its filename, the
artifact behaves like a **single FaithEval report card**, not a top-level map of
benchmark payloads. That is fine operationally, but future code should not infer
benchmark multiplexing from the filename.

## 3. Balanced Bottom Line

### What survived the audit

- The paired FaithEval slope-difference result is real.
- The export contract is correct.
- The specificity claim against random-neuron controls is stronger after this
  audit than before.
- The narrow claim "H-neurons outperform the committed SAE readout on matched
  FaithEval trajectories" is publication-safe.

### What remains uncertain

- The paired slope-difference result does not replace the delta-only SAE audit.
- The directional p-values should not be over-interpreted as two-sided tests.
- The random-control aggregate remains an empirical 8-seed summary, not a
  universe-level null distribution.

## 4. Recommended Citation Pattern

To avoid drift across docs, use this split:

- **This report** for the 2026-04-13 reporting-path audit, export wiring, and
  paired slope-difference numbers
- [sae_pipeline_audit.md](../../data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md)
  for the broader SAE closure argument and delta-only confound-removal result
- [intervention_findings.md](../../data/gemma3_4b/intervention_findings.md)
  for the project-level intervention synthesis

## 5. Highest-Value Next Steps

1. In paper/site prose, cite the paired slope-difference number for the narrow
   matched-readout statement and cite the delta-only report for the stronger
   "not just reconstruction noise" interpretation.
2. Keep the random-control result phrased as an empirical 8-seed specificity
   check, not as a fully modeled null over all zero-weight neuron sets.
3. If the site payload is ever generalized beyond FaithEval, rename or wrap
   `intervention_sweep.json` so its structure matches its filename.

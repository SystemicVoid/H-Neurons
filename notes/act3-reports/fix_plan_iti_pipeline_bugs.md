# Fix Plan: ITI Paper-Faithful Pipeline Bugs

**Source plan:** `~/.claude/plans/agile-soaring-dragon.md`
**Files built:** `report_iti_2fold.py`, `iti_paperfaithful_rerun_pipeline.sh`, `run_calibration_sweep.py`, `build_truthfulqa_calibration_splits.py` + modifications to `intervene_iti.py`, `run_intervention.py`, `evaluate_intervention.py`

---

## Confirmed Issues (from code review)

### FIX 1 — [P1] `report_iti_2fold.py` miscomputes MC2 by reading `compliance` instead of `metric_value`

**File:** `scripts/report_iti_2fold.py`, lines 51–59 (`extract_correctness`)

**Root cause:** `extract_correctness()` always reads `bool(r["compliance"])` regardless of variant. For MC1 this is correct (compliance = top choice is truthful). For MC2, `compliance` is still the binary "top choice is truthful" flag, but the actual MC2 metric is `metric_value` = truthful mass (a continuous 0–1 value). The plan explicitly says MC2 uses truthful mass, and `run_intervention.py:1680-1684` confirms `metric_value` stores truthful_mass for MC2.

**Fix:**
- For MC1: keep reading `compliance` (binary correctness).
- For MC2: read `float(r["metric_value"])` — this is the truthful mass.
- Since MC2 is continuous (not binary), `paired_bootstrap_binary_rate_difference` is wrong for MC2. Replace with a continuous paired bootstrap on the raw truthful mass values. Add `paired_bootstrap_continuous_mean_difference()` to `uncertainty.py`, or inline a simple bootstrap of means. The bootstrap resamples question-level (metric_value_intervened − metric_value_baseline) differences.
- Wilson CI also doesn't apply to MC2 (it's for proportions). Use bootstrap percentile CI on the mean instead.
- McNemar is binary-only, so only report it for MC1 (already the case — no MC2 change needed here).

**Impact:** Without this fix, every MC2 result from the 2-fold report is a binary compliance stat, not the planned truthful-mass metric. Any calibration or paper claim citing MC2 is wrong.

**Changes:**
1. `scripts/report_iti_2fold.py` — rewrite `extract_correctness` → `extract_metric_values`; bifurcate compute_fold_report and compute_pooled_report logic by variant type.
2. `scripts/uncertainty.py` — add `paired_bootstrap_continuous_mean_difference()`.

---

### FIX 2 — [P1] Pooled report reconstructs arrays from rounded rates, destroying pairing

**File:** `scripts/report_iti_2fold.py`, lines 148–159 (`compute_pooled_report`)

**Root cause:** The pooled section does:
```python
b_count = round(fr["baseline"]["rate"] * n)
all_baseline.extend([True] * b_count + [False] * (n - b_count))
```
This reconstructs approximate boolean arrays from rounded per-fold summary statistics. The order is lost (all True first, then all False), so when `paired_bootstrap_binary_rate_difference` resamples indices, it draws from a non-paired order. The CI and McNemar become scientifically invalid because they assume index i in baseline corresponds to index i in intervened.

**Fix:** Thread the actual per-question raw data through to the pooled report. Change the flow so `compute_fold_report` also returns the aligned arrays (or the raw records), and `compute_pooled_report` concatenates the original arrays from both folds.

**Changes:**
1. `scripts/report_iti_2fold.py` — `compute_fold_report` returns the aligned baseline/intervened arrays alongside the report dict. `compute_pooled_report` accepts these arrays, concatenates them, and computes true paired statistics.

---

### FIX 3 — [P1] Random-direction controls collide in same output directory (3 seeds → 1 run)

**File:** `scripts/infra/iti_paperfaithful_rerun_pipeline.sh`, lines 149–158; `scripts/run_intervention.py`, `build_iti_output_suffix` lines 95–121

**Root cause:** `build_iti_output_suffix` hashes `{iti_head_path}|{iti_family}|{iti_k}|{iti_selection_strategy}|{iti_random_seed}` — but does NOT include `iti_direction_mode` or `iti_direction_random_seed`. When the pipeline runs `--iti_direction_mode random --iti_direction_random_seed 1/2/3`, the default output directory is identical for all three seeds. Since `run_intervention.py` skips sample IDs already present in `alpha_*.jsonl`, seeds 2 and 3 silently no-op.

**Fix:** Include `direction_mode` and `direction_random_seed` in `build_iti_output_suffix` and its config hash. The function signature needs two new parameters; `resolve_output_dir` must pass them.

**Changes:**
1. `scripts/run_intervention.py` — add `direction_mode: str = "artifact"` and `direction_random_seed: int | None = None` to `build_iti_output_suffix()`. Include them in the config hash string and in the steering label when `direction_mode != "artifact"`. Update `resolve_output_dir` to pass the new args.

---

### FIX 4 — [P2] Pipeline does not run judging/reporting; manual steps required

**File:** `scripts/infra/iti_paperfaithful_rerun_pipeline.sh`, lines 215–218

**Root cause:** The plan says this is the "end-to-end rerun chain" but it ends with `echo "Next steps"` instead of actually running `report_iti_2fold.py`, `evaluate_intervention.py --benchmark simpleqa`, and `evaluate_intervention.py --benchmark falseqa`. The operator must manually do the main deliverables.

**Fix:** Add the missing steps at the end of the pipeline:
1. Run `report_iti_2fold.py` for MC1 and MC2 (need to compute output dirs from locked config).
2. Run `evaluate_intervention.py --benchmark simpleqa --api-mode batch` for the SimpleQA output dir.
3. Run `evaluate_intervention.py --benchmark falseqa --api-mode batch` for the FalseQA output dir.
4. Append to `notes/runs_to_analyse.md`.

**Note:** The `evaluate_intervention.py` calls are API calls (OpenAI batch), not GPU. They may take time but should still be in the chain per plan Phase 5 spec. Per `scripts/AGENTS.md`, always use `--api-mode batch`.

**Changes:**
1. `scripts/infra/iti_paperfaithful_rerun_pipeline.sh` — replace the "Next steps" echo block with actual invocations + runs_to_analyse.md append.

---

## Additional Issues Found (not in original review)

### FIX 5 — [P2] `report_iti_2fold.py` is single-variant per invocation but plan says "for each metric (MC1, MC2)"

**Root cause:** The script accepts `--variant mc1` or `--variant mc2` and produces a single report. The plan (Phase 2.3) says the report should cover both MC1 and MC2 together. The pipeline (FIX 4) would need to call it twice, but the output file naming `iti_2fold_{variant}_report.json` is at least correct for separate files.

**Assessment:** This is actually fine as-is — calling it twice with `--variant mc1` and `--variant mc2` produces both reports. Just ensure FIX 4 calls it twice.

**No code change needed** — just make sure the pipeline calls both variants.

---

### FIX 6 — [P2] `run_calibration_sweep.py` doesn't clean up hooks between K values properly

**File:** `scripts/run_calibration_sweep.py`, lines 221–260

**Root cause:** The loop creates a new `ITIHeadScaler` for each K but only calls `scaler.remove()` at the end of the inner α loop. This is actually correct per the current code flow — remove is called before the next K's scaler is built. However, the new scaler is constructed BEFORE the old one is removed (the `scaler.remove()` is at the bottom of the K loop, line 260). This means during construction of the new scaler for K=12, the K=8 hooks are still active. The `ITIHeadScaler.__init__` calls `self._install(model)` which adds NEW hooks without removing old ones.

**Fix:** Move `scaler.remove()` to the beginning of the K loop (guarded by `if scaler is not None`), or explicitly remove before constructing the next scaler.

**Changes:**
1. `scripts/run_calibration_sweep.py` — restructure the loop to `remove()` before creating the next scaler.

---

### FIX 7 — [P3] `report_iti_2fold.py` does not record locked K or artifact paths in the output JSON

**Root cause:** The plan (Phase 2.3) says "Locked K, α noted in report." The script records `locked_alpha` but not `locked_K`, nor the artifact paths or fold directories. For reproducibility and provenance, the report should include the full locked config reference.

**Fix:** Add `--locked_k` CLI arg and record it plus fold dirs in the output JSON. Alternatively, accept `--locked_config` path and embed its contents.

**Changes:**
1. `scripts/report_iti_2fold.py` — add `--locked_k` arg; include it and fold dir paths in the output report.

---

### FIX 8 — [P3] Pipeline doesn't use `systemd-inhibit` for the ~6h GPU chain

**Root cause:** Per `AGENTS.md`: "Use `systemd-inhibit` for any GPU job longer than ~20 minutes." The pipeline is estimated at ~6h total. The script does not wrap itself in `systemd-inhibit`.

**Fix:** Add `systemd-inhibit --what=sleep:idle --why="ITI paperfaithful pipeline"` wrapper or document it in the usage header.

**Changes:**
1. `scripts/infra/iti_paperfaithful_rerun_pipeline.sh` — add `systemd-inhibit` invocation note in header, or wrap the whole script.

---

## Execution Order

Fixes are ordered by dependency, not severity:

1. **FIX 3** (output dir collision) — standalone change in `run_intervention.py`
2. **FIX 6** (sweep hook cleanup) — standalone change in `run_calibration_sweep.py`
3. **FIX 1 + FIX 2** (MC2 metric + pooled pairing) — coupled changes in `report_iti_2fold.py` + `uncertainty.py`
4. **FIX 7** (locked_k in report) — small addition to `report_iti_2fold.py`
5. **FIX 4** (pipeline completeness) — depends on knowing correct output dir patterns (FIX 3)
6. **FIX 8** (systemd-inhibit) — cosmetic, do last

## Verification

After all fixes:
```bash
uv run pytest
ruff check scripts && ruff format scripts
ty check
```

Smoke-test `build_iti_output_suffix` with direction_mode="random" and different seeds → confirm distinct paths.

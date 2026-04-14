# D7 Control And Ruler Audit — mixed-ruler reconciliation

> **Status: historical pre-probe reconciliation note; superseded for current D7 interpretation by**
> [2026-04-14-d7-full500-current-state-audit.md](./2026-04-14-d7-full500-current-state-audit.md).
>
> This note preserves the April 14 evidence state **before** `probe_locked` was fully scored. Keep using it for the pre-probe control/ruler reconciliation only, not as the live D7 headline.
>
> **Historical report preserved:** [2026-04-08-d7-full500-audit.md](./2026-04-08-d7-full500-audit.md)
>
> **Important exclusion:** `probe_locked` is not part of the evidence base here. This note captures the April 14, 2026 pre-probe state, when `probe_locked` was only partially generated and had no judge/CSV2 outputs, so it was non-evidential at the time.

**Date:** 2026-04-14  
**Data root:** `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/`  
**Structured summary:** `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_control_and_ruler_summary.json`  
**Historical structured summary:** `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_csv2_report.json`  
**Model:** `google/gemma-3-4b-it`  
**Generation surface:** canonical (`do_sample=True`, `temperature=0.7`, `top_k=20`, `top_p=0.8`, `max_new_tokens=5000`)  
**Primary task of this note:** reconcile historical April 8 D7 conclusions with the April 14 layer-matched random-head control and the current normalization stack, without paying to rescore old branches.

## 0. Bottom Line

### Data

- The April 8, 2026 legacy-ruler panel reported:
  - baseline: **23.4%** `csv2_yes` (117/500)
  - L1: **27.4%** (137/500), **+4.0 pp** vs baseline **[+0.6, +7.6]**
  - causal: **14.4%** (72/500), **-9.0 pp** vs baseline **[-12.2, -5.8]**
- The April 14, 2026 current-normalization reconciliation over the stored artifacts gives a very different panel:
  - baseline: **51.6%** strict harmfulness (258/500)
  - L1: **46.8%** (234/500), **-4.8 pp** vs baseline **[-8.8, -1.0]**
  - causal: **24.8%** (124/500), **-26.8 pp** vs baseline **[-31.0, -22.6]**
  - layer-matched random seed 1: **37.2%** (186/500), **-14.4 pp** vs baseline **[-19.0, -9.8]**
- The layer-matched random-head control is **not literally missing anymore**, but it is **single-seed, mixed-ruler, and error-bearing**:
  - 500/500 prompt IDs match baseline
  - 8 CSV2 rows carry explicit errors (`7 invalid_evidence_spans`, `1 parse_failed`)
  - excluding those 8 rows barely changes the random rate (**37.2% -> 37.0%**) or the causal-vs-random gap (**12.4 pp -> 12.0 pp**)
- `probe_locked` was still incomplete and excluded from conclusions at the time of this pre-probe note.

### Interpretation

The strongest surviving D7 claim is now narrower and cleaner:

> **On this benchmark surface, the locked causal intervention still looks like the strongest completed D7 condition.**

What is **not** earned is the stronger claim:

> **D7 is now fully closed as a selector-specific or mechanism-clean result.**

The random-head control now provides **partial** support for head-identity specificity, but the full D7 story remains mixed-ruler, one-seed, and incomplete without the full-500 probe branch.

## 1. Data

### 1.1 Evidence state

| Condition | Status | Rows | Judge/CSV2 status | Use in this report |
|---|---|---:|---|---|
| `baseline_noop` | complete | 500 | judged + CSV2-scored | yes |
| `l1_neuron` | complete | 500 | judged + CSV2-scored | yes |
| `causal_locked` | complete | 500 | judged + CSV2-scored | yes |
| `causal_random_head_layer_matched/seed_1` | complete | 500 | judged + CSV2-scored | yes |
| `causal_random_head_layer_matched/seed_2` | absent | 0 | not run | no |
| `probe_locked` | incomplete at time of note | partial generation only | generation only; no judge/CSV2 | no |

### 1.2 Prompt parity

- `baseline_noop`, `l1_neuron`, `causal_locked`, and `causal_random_head_layer_matched/seed_1` all contain **500/500** rows.
- All four scored conditions have **exact prompt-ID parity**.
- This means the current paired comparisons are numerically well-formed, even though they are not all on the same ruler.

## 2. Historical April 8 Panel

This is the historical legacy-ruler panel from the stored April 8 summary. It is preserved for provenance and comparison; it is **not** the active D7 headline anymore.

| Condition | Legacy `csv2_yes` | 95% CI | Paired delta vs baseline |
|---|---:|---:|---:|
| baseline | 23.4% (117/500) | [19.9, 27.3] | — |
| L1 | 27.4% (137/500) | [23.7, 31.5] | **+4.0 pp** [+0.6, +7.6] |
| causal | 14.4% (72/500) | [11.7, 17.8] | **-9.0 pp** [-12.2, -5.8] |

On that historical ruler, the story was straightforward: causal beat baseline and L1, while L1 worsened baseline.

## 3. Current Reconciliation / Control Panel

This panel applies the **current** normalization stack to the stored artifacts. It is useful for reconciliation and for incorporating the April 14 random-head control, but it is **not the same ruler** as the April 8 panel.

### 3.1 Current normalized rates

| Condition | Strict harmfulness | 95% CI | Binary harmful | 95% CI | Token-cap hits |
|---|---:|---:|---:|---:|---:|
| baseline | 51.6% (258/500) | [47.2, 55.9] | 30.0% (150/500) | [26.2, 34.2] | 0 |
| L1 | 46.8% (234/500) | [42.5, 51.2] | 33.2% (166/500) | [29.2, 37.5] | 0 |
| causal | 24.8% (124/500) | [21.3, 28.8] | 19.4% (97/500) | [16.2, 23.0] | 112 |
| random layer seed 1 | 37.2% (186/500) | [33.1, 41.5] | 36.6% (183/500) | [32.5, 40.9] | 0 |

### 3.2 Paired deltas versus baseline

| Condition | Strict harmfulness delta | 95% CI | Binary harmful delta | 95% CI |
|---|---:|---:|---:|---:|
| L1 | **-4.8 pp** | [-8.8, -1.0] | +3.2 pp | [-0.2, +6.6] |
| causal | **-26.8 pp** | [-31.0, -22.6] | **-10.6 pp** | [-14.0, -7.2] |
| random layer seed 1 | **-14.4 pp** | [-19.0, -9.8] | **+6.6 pp** | [+2.8, +10.4] |

This is the most important current-panel fact: the layer-matched random-head control looks **better than baseline on normalized strict harmfulness**, but **worse than baseline on the binary judge**.

### 3.3 Severity diagnostics

| Condition | Mean `C` | Mean `S` | Mean harmful payload share | Primary outcome mix |
|---|---:|---:|---:|---|
| baseline | 0.890 | 1.018 | 0.2528 | 142 partial, 116 substantive, 231 refusal |
| L1 | 0.938 | 1.072 | 0.2623 | 98 partial, 136 substantive, 255 refusal |
| causal | 0.490 | 0.562 | 0.1031 | 50 partial, 74 substantive, 364 refusal |
| random layer seed 1 | 1.102 | 1.162 | 0.2522 | 16 partial, 170 substantive, 220 refusal |

The random-head control is not a clean null on the current panel. Relative to baseline, it produces:

- fewer total normalized harmful rows,
- **more** binary-harmful rows,
- essentially unchanged mean harmful payload share,
- and a sharp shift from **partial compliance** toward **substantive compliance**.

That is exactly the kind of mixed signal that should prevent overclaiming.

### 3.4 Direct causal-versus-random comparison

| Comparison | Estimate | 95% CI | Transition counts |
|---|---:|---:|---|
| random minus causal, strict harmfulness | **+12.4 pp** | [+8.0, +16.8] | 98 random-only harmful vs 36 causal-only harmful |
| random minus causal, strict harmfulness (clean rows only) | **+12.0 pp** | [+7.5, +16.3] | 95 random-only harmful vs 36 causal-only harmful |
| random minus causal, binary harmful | **+17.2 pp** | [+13.0, +21.4] | 105 random-only harmful vs 19 causal-only harmful |
| random minus causal, payload share | **+0.1490** | [+0.1218, +0.1768] | paired mean delta |

On every completed paired comparison against the available random-head seed, the causal branch looks better.

### 3.5 Random-head CSV2 error burden

The `seed_1` random-head CSV2 file contains 8 explicit errors:

- `invalid_evidence_spans`: 7 rows
- `parse_failed`: 1 row

This is enough to make the existing strict D7 paired-reporting script reject the file, but not enough to explain away the causal-vs-random gap:

- random strict harmfulness, all rows: **37.2%**
- random strict harmfulness, clean rows only: **37.0%**
- random minus causal strict harmfulness, all rows: **+12.4 pp**
- random minus causal strict harmfulness, clean rows only: **+12.0 pp**

So the error burden is scientifically relevant, but not numerically decisive.

## 4. Ruler Audit

### 4.1 The drift is large enough to change conclusions

| Condition | April 8 legacy rate | Current normalized rate | Drift | Borderline -> yes |
|---|---:|---:|---:|---:|
| baseline | 23.4% | 51.6% | **+28.2 pp** | 141/146 |
| L1 | 27.4% | 46.8% | **+19.4 pp** | 97/100 |
| causal | 14.4% | 24.8% | **+10.4 pp** | 52/54 |

This is not a cosmetic difference. It is large enough to reverse the L1-vs-baseline conclusion:

- **April 8 legacy panel:** L1 looked worse than baseline.
- **Current normalized panel:** L1 looks better than baseline on strict harmfulness, but still worse on binary harmfulness.

### 4.2 What changed in the ruler

The legacy April 8 files are unversioned CSV2 artifacts with a three-way `harmful_binary` label:

- `yes`
- `borderline`
- `no`

The current normalization path does **not** preserve `borderline` as a third state. It derives:

1. `primary_outcome`
2. `intent_match`
3. a final binary harmfulness label (`yes` / `no`)

That means most legacy `borderline` rows are now forced into `yes`:

- baseline: **141/146**
- L1: **97/100**
- causal: **52/54**

This is the main source of ruler drift. The current panel is therefore best understood as a **compatibility reinterpretation** of the old files, not as a fresh April 8 truth.

### 4.3 Why the random-head branch is different again

The April 14 random-head control is already in a `csv2_v3`-style schema with:

- explicit `primary_outcome`
- explicit `intent_match`
- explicit span validation errors

So the random-head branch is not just “new data.” It is new data on a newer measurement surface. That is why a uniform current panel is possible only by **reinterpreting the old files**, not by simply reading four like-for-like outputs.

## 5. What Withstands Scrutiny

1. **Causal still looks best among the completed D7 branches.** That survives the legacy panel, the current normalized panel, the binary judge, and the direct causal-versus-random comparison.
2. **The random-head control is no longer missing.** Any document still saying “D7 random-head control missing” is now stale.
3. **The random-head control does not close D7 cleanly.** It is one seed, mixed-ruler, and carries 8 explicit CSV2 errors.
4. **The strongest current claim is benchmark-local.** The data support “causal is the strongest completed D7 intervention on this jailbreak surface,” not “gradient ranking is now cleanly proven mechanism-specific.”
5. **Measurement choice materially changes the comparator story.** The L1 baseline comparison reverses under current normalization, which means that part of the old April 8 narrative does not survive ruler audit.

## 6. What Does Not Withstand Scrutiny

1. **“L1 worsens baseline” is no longer stable.** It is true on the April 8 legacy panel and false on current normalized strict harmfulness.
2. **“The random-head control is missing” is false.** The right statement is now “single-seed layer-matched random-head control exists, but D7 is still not fully closed.”
3. **“Selector specificity is solved” is not earned.** The random control is supportive, but not decisive enough for a mechanism-clean claim.
4. **“Full-500 probe null versus causal positive” is not available in this pre-probe note.** At the time, the probe branch was still incomplete and had to stay out of the conclusions recorded here.

## 7. Interpretation

### 7.1 What the data do support

The cumulative evidence now supports a narrower but stronger D7 sentence than either extreme:

- not “the random-head control is still missing,”
- and not “D7 is now fully closed.”

The best reading is:

> **D7 now has partial control support: the causal branch outperforms the available layer-matched random-head control, but the control evidence is still mixed-ruler, single-seed, and not sufficient to make the selector-specific claim mechanism-clean.**

### 7.2 Why the random-head result is interesting, not null

The random-head seed is not a flat null in the simple sense. It appears to reduce normalized strict harmfulness by collapsing many **partial-compliance** cases, while simultaneously increasing:

- binary harmfulness,
- mean severity (`C`, `S`),
- and the share of **substantive compliance** among the harmful rows that remain.

That means the random-head seed is acting more like a **distribution-shifting perturbation** than a clean do-nothing baseline. This makes the causal-vs-random gap informative, but not interpretable as a pristine selector-specificity closure.

### 7.3 Where D7 should sit after this audit

Scientifically, D7 should now be framed as:

- **supporting evidence that selector choice matters on this benchmark surface,**
- **partially strengthened by a single-seed layer-matched random-head control,**
- **still limited by ruler mismatch, missing full-500 probe scoring, and visible quality debt.**

That is stronger than the April 8 “control missing” framing, but still below a flagship mechanism claim.

## 8. Uncertainties

| Uncertainty | Size | Why it matters |
|---|---|---|
| Mixed-ruler comparison | high | Legacy unversioned April 8 files and April 14 `csv2_v3` control are not truly like-for-like |
| Single-seed random-head control | moderate to high | One seed can support directionality but not robust null-distribution claims |
| Random-head CSV2 errors | low to moderate | Only 8 rows, but enough to invalidate the old strict report script and force an audit path |
| Missing full-500 probe scoring | high | Prevents a complete full-scale causal-versus-probe-versus-random closure |
| Causal token-cap debt | moderate | Causal remains the strongest branch, but 112/500 cap hits are still visible quality debt |

## 9. Most Valuable Next Steps

1. **Score `probe_locked` and append an additive update to this report.** Do not rewrite the current body; add a dated update section once probe scoring exists.
2. **If D7 later needs to become more central, decide whether a second layer-matched random seed is worth it.** That is a better spend target than duplicative rescoring of April 8 branches.
3. **Keep the April 8 report frozen as historical provenance.** Any downstream note should cite this report for current D7 interpretation and link the April 8 report only for legacy-ruler context.

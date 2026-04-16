# D7 Full-500 Two-Seed Current-State Audit — canonical mixed-ruler interpretation

> **Status: canonical source of truth for current D7 interpretation.**
>
> This report supersedes [2026-04-14-d7-full500-current-state-audit.md](./2026-04-14-d7-full500-current-state-audit.md) for **current** D7 claims because the second layer-matched random-head control seed is now fully generated, judged, and CSV2-scored.
>
> **Historical reports preserved:**
> [2026-04-14-d7-full500-current-state-audit.md](./2026-04-14-d7-full500-current-state-audit.md),
> [2026-04-14-d7-control-and-ruler-audit.md](./2026-04-14-d7-control-and-ruler-audit.md),
> [2026-04-08-d7-full500-audit.md](./2026-04-08-d7-full500-audit.md)

**Date:** 2026-04-16  
**Data root:** `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/`  
**Structured summary:** `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_full500_current_state_summary.json`  
**Historical structured summary:** `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_csv2_report.json`  
**Model:** `google/gemma-3-4b-it`  
**Generation surface:** canonical (`do_sample=True`, `temperature=0.7`, `top_k=20`, `top_p=0.8`, `max_new_tokens=5000`)  
**Primary task of this note:** state the strongest honest current D7 claim after integrating the April 8 legacy panel, the full-500 `probe_locked` branch, and both completed layer-matched random-head control seeds.

## 0. Bottom Line

### Data

- The April 8 legacy-ruler panel remains:
  - baseline: **23.4%** `csv2_yes` (117/500)
  - L1: **27.4%** (137/500), **+4.0 pp** vs baseline **[+0.6, +7.6]**
  - causal: **14.4%** (72/500), **-9.0 pp** vs baseline **[-12.2, -5.8]**
- The current normalized mixed-ruler panel over the stored full-500 artifacts is:
  - baseline: **51.6%** strict harmfulness (258/500)
  - L1: **46.8%** (234/500), **-4.8 pp** vs baseline **[-8.8, -1.0]**
  - causal: **24.8%** (124/500), **-26.8 pp** vs baseline **[-31.0, -22.6]**
  - random seed 1: **37.2%** (186/500), **-14.4 pp** vs baseline **[-19.0, -9.8]**
  - random seed 2: **38.8%** (194/500), **-12.8 pp** vs baseline **[-17.4, -8.4]**
  - probe: **34.8%** (174/500), **-16.8 pp** vs baseline **[-20.8, -12.8]**
- The non-causal comparator branches remain scientifically usable but not pristine:
  - random seed 1: **8** explicit CSV2 errors (`7 invalid_evidence_spans`, `1 parse_failed`)
  - random seed 2: **14** explicit CSV2 errors (`14 invalid_evidence_spans`)
  - probe: **12** explicit CSV2 errors (`12 invalid_evidence_spans`)
- Direct current-panel comparisons continue to favor causal over every available comparator:
  - causal vs random seed 1: **-12.4 pp** strict harmfulness **[-16.8, -8.0]**
  - causal vs random seed 2: **-14.0 pp** strict harmfulness **[-18.2, -10.0]**
  - causal vs probe: **-10.0 pp** strict harmfulness **[-14.0, -6.2]**
- The two random-head seeds are directionally consistent within uncertainty:
  - seed 2 minus seed 1: **+1.6 pp** strict harmfulness **[-2.4, +5.4]**

### Interpretation

The strongest surviving D7 sentence is now:

> **On the current mixed-ruler full-500 evidence base, the locked causal intervention is the strongest completed D7 branch.**

What is still **not** earned:

> **D7 is fully closed as a mechanism-clean selector-specificity result.**

The single-seed caveat is no longer the bottleneck. The remaining limiting factors are that the panel is still **mixed-ruler**, the probe/random branches are **error-bearing**, and the causal branch still carries visible **token-cap debt**.

## 1. Data

### 1.1 Evidence state

| Condition | Status | Rows | Judge/CSV2 status | Use in this report |
|---|---|---:|---|---|
| `baseline_noop` | complete | 500 | judged + CSV2-scored | yes |
| `l1_neuron` | complete | 500 | judged + CSV2-scored | yes |
| `causal_locked` | complete | 500 | judged + CSV2-scored | yes |
| `causal_random_head_layer_matched/seed_1` | complete, error-bearing | 500 | judged + CSV2-scored | yes |
| `causal_random_head_layer_matched/seed_2` | complete, error-bearing | 500 | judged + CSV2-scored | yes |
| `probe_locked` | complete, error-bearing | 500 | judged + CSV2-scored | yes |

### 1.2 Prompt parity

- All six scored conditions contain **500/500** rows.
- All six scored conditions have **exact prompt-ID parity**.
- This means every paired comparison in the current panel is numerically well-formed.

## 2. Historical April 8 Panel

This panel is preserved for provenance only. It is the original April 8 structured summary and should not be used as the live D7 headline without this current-state report.

| Condition | Legacy `csv2_yes` | 95% CI | Paired delta vs baseline |
|---|---:|---:|---:|
| baseline | 23.4% (117/500) | [19.9, 27.3] | — |
| L1 | 27.4% (137/500) | [23.7, 31.5] | **+4.0 pp** [+0.6, +7.6] |
| causal | 14.4% (72/500) | [11.7, 17.8] | **-9.0 pp** [-12.2, -5.8] |

On that historical ruler, the story was simple: causal beat baseline and L1, while L1 worsened baseline.

## 3. Current-State Mixed-Ruler Panel

This panel applies the current normalization stack to the stored artifacts. It is the best available **current** D7 interpretation, but it is not a like-for-like rerun on one stable ruler.

### 3.1 Current normalized rates

| Condition | Strict harmfulness | 95% CI | Binary harmful | 95% CI | Token-cap hits |
|---|---:|---:|---:|---:|---:|
| baseline | 51.6% (258/500) | [47.2, 55.9] | 30.0% (150/500) | [26.1, 34.2] | 0 |
| L1 | 46.8% (234/500) | [42.5, 51.2] | 33.2% (166/500) | [29.2, 37.4] | 0 |
| causal | 24.8% (124/500) | [21.2, 28.8] | 19.4% (97/500) | [16.2, 23.1] | 112 |
| random seed 1 | 37.2% (186/500) | [33.1, 41.5] | 36.6% (183/500) | [32.5, 40.9] | 0 |
| random seed 2 | 38.8% (194/500) | [34.6, 43.1] | 35.2% (176/500) | [31.1, 39.5] | 0 |
| probe | 34.8% (174/500) | [30.8, 39.1] | 32.4% (162/500) | [28.4, 36.6] | 0 |

### 3.2 Paired deltas versus baseline

| Condition | Strict harmfulness delta | 95% CI | Binary harmful delta | 95% CI |
|---|---:|---:|---:|---:|
| L1 | **-4.8 pp** | [-8.8, -1.0] | +3.2 pp | [-0.2, +6.6] |
| causal | **-26.8 pp** | [-31.0, -22.6] | **-10.6 pp** | [-14.0, -7.2] |
| random seed 1 | **-14.4 pp** | [-19.0, -9.8] | **+6.6 pp** | [+2.8, +10.4] |
| random seed 2 | **-12.8 pp** | [-17.4, -8.4] | +5.2 pp | [+1.8, +8.6] |
| probe | **-16.8 pp** | [-20.8, -12.8] | +2.4 pp | [-0.8, +5.8] |

The critical comparison fact is unchanged: **all non-baseline interventions look better than baseline on normalized strict harmfulness, but only causal also improves the binary judge**.

### 3.3 Severity diagnostics

| Condition | Mean `C` | Mean `S` | Mean harmful payload share | Primary outcome mix |
|---|---:|---:|---:|---|
| baseline | 0.890 | 1.018 | 0.2528 | 142 partial, 116 substantive, 231 refusal |
| L1 | 0.938 | 1.072 | 0.2623 | 98 partial, 136 substantive, 255 refusal |
| causal | 0.490 | 0.562 | 0.1031 | 50 partial, 74 substantive, 364 refusal |
| random seed 1 | 1.102 | 1.162 | 0.2522 | 16 partial, 170 substantive, 220 refusal |
| random seed 2 | 1.154 | 1.196 | 0.2348 | 29 partial, 165 substantive, 194 refusal |
| probe | 1.024 | 1.066 | 0.2078 | 26 partial, 148 substantive, 216 refusal |

Two patterns matter here:

1. **Causal is still the only branch that is cleaner on every available surface.**
2. **Neither random branch is a clean null.** Both reduce normalized strict harmfulness without reducing binary harmfulness, and both remain substantially harsher than causal on severity-sensitive metrics.

### 3.4 Direct comparator panel

| Comparison | Strict harmfulness delta | 95% CI | Binary harmful delta | 95% CI | Payload-share delta |
|---|---:|---:|---:|---:|---:|
| causal vs random seed 1 | **-12.4 pp** | [-16.8, -8.0] | **-17.2 pp** | [-21.4, -13.0] | **-0.1490** [-0.1768, -0.1218] |
| causal vs random seed 2 | **-14.0 pp** | [-18.2, -10.0] | **-15.8 pp** | [-19.4, -12.2] | **-0.1317** [-0.1561, -0.1075] |
| causal vs probe | **-10.0 pp** | [-14.0, -6.2] | **-13.0 pp** | [-16.6, -9.4] | **-0.1046** [-0.1278, -0.0816] |
| probe vs random seed 1 | -2.4 pp | [-6.8, +1.8] | -4.2 pp | [-8.4, 0.0] | **-0.0444** [-0.0726, -0.0168] |
| probe vs random seed 2 | **-4.0 pp** | [-7.8, -0.2] | -2.8 pp | [-6.4, +0.6] | **-0.0271** [-0.0492, -0.0045] |
| random seed 2 vs random seed 1 | +1.6 pp | [-2.4, +5.4] | -1.4 pp | [-5.2, +2.4] | -0.0173 [-0.0446, +0.0088] |

The interpretation is asymmetric:

- **causal > random seed 1** is robust,
- **causal > random seed 2** is robust,
- **causal > probe** is robust,
- **probe > random seed 2** is supported on strict harmfulness,
- **probe vs random seed 1** remains suggestive rather than closed,
- **seed 1 vs seed 2** shows modest heterogeneity but no sign flip.

### 3.5 Error burden and clean-row sensitivity

| Condition | Error rows | Error types | Strict harmfulness all rows | Strict harmfulness clean rows |
|---|---:|---|---:|---:|
| random seed 1 | 8 | `7 invalid_evidence_spans`, `1 parse_failed` | 37.2% | 37.0% |
| random seed 2 | 14 | `14 invalid_evidence_spans` | 38.8% | 38.3% |
| probe | 12 | `12 invalid_evidence_spans` | 34.8% | 34.8% |

Paired clean-row sensitivity:

- baseline vs random seed 1: **-14.4 pp -> -14.2 pp**
- baseline vs random seed 2: **-12.8 pp -> -13.3 pp**
- baseline vs probe: **-16.8 pp -> -16.6 pp**
- random seed 1 vs causal: **+12.4 pp -> +12.0 pp**
- random seed 2 vs causal: **+14.0 pp -> +14.0 pp**
- probe vs random seed 2: **-4.0 pp -> -4.4 pp**

So the error burden matters for rigor, but it is not driving the sign of the main conclusions.

## 4. What Withstands Scrutiny

1. **Causal is still the strongest completed D7 branch.** That survives the legacy panel, the current panel, the binary judge, and every direct comparator panel.
2. **The layer-matched random-head control is now a two-seed family, not a one-seed anecdote.** Any live note still calling it single-seed or saying `seed_2` is absent is stale.
3. **The full-500 probe branch is no longer missing.** Any live note still calling `probe_locked` incomplete is stale.
4. **Selector choice matters on this benchmark surface.** The current full-500 evidence no longer rests on probe-null bookkeeping; it rests on causal beating both available probe/random alternatives on the normalized panel.
5. **Measurement choice still materially changes the comparator story.** The L1 baseline comparison remains ruler-sensitive, so April 8 L1 claims must stay historical.

## 5. What Does Not Withstand Scrutiny

1. **“The random-head control is missing” is false.**
2. **“The random-head control is still single-seed” is false.**
3. **“L1 worsens baseline” is not stable enough for current copy.** It is true on the April 8 legacy panel and false on current normalized strict harmfulness.
4. **“D7 is now mechanism-clean” is still not earned.** The panel remains mixed-ruler, and the probe/random branches remain error-bearing.
5. **“Probe is null at full-500” is not supported.** On the current normalized panel, probe is clearly non-null against baseline on strict harmfulness and beats one of the two random seeds.

## 6. Interpretation

### 6.1 Best current reading

The right D7 sentence is now:

> **D7 provides benchmark-local supporting evidence that selector choice matters: the locked causal branch outperforms the available probe and both layer-matched random branches on the current normalized full-500 panel, but the result is still not a mechanism-clean selector-specificity closure.**

This is stronger than the April 14 single-seed framing, but it still falls short of a clean mechanistic claim.

### 6.2 What seed 2 changes

`seed_2` changes the scientific state in three ways.

First, it removes the bookkeeping caveat that the layer-matched random family is only one seed deep.

Second, it shows the random family is directionally consistent within uncertainty: both seeds sit well above causal on strict harmfulness and both remain worse than causal on every available severity-sensitive surface.

Third, it sharpens one ordering that was previously only suggestive: probe now looks cleaner than random seed 2 on strict harmfulness with a CI excluding zero.

### 6.3 Why this still is not closure

The evidence is still below a mechanism-clean selector-specificity result because:

- the panel remains mixed-ruler,
- the probe and both random branches carry explicit CSV2 error debt,
- and the causal branch still has **112/500** token-cap hits.

So the update strengthens the benchmark-local comparator claim without making the broader mechanistic claim safe.

## 7. Uncertainties

| Uncertainty | Size | Why it matters |
|---|---|---|
| Mixed-ruler comparison | high | Legacy baseline/L1/causal and newer probe/random outputs are not strictly like-for-like |
| Probe / random CSV2 errors | moderate | Clean-row sensitivity is stable, but the audit path is still necessary |
| Random-family size | moderate | Two seeds are better than one, but still not a rich null-distribution estimate |
| Causal token-cap debt | moderate | Causal remains best, but 112/500 cap hits are visible quality debt |
| Probe vs random ordering | low to moderate | Probe clearly beats seed 2, but probe vs seed 1 remains close |

## 8. Most Valuable Next Steps

1. **Keep using D7 as benchmark-local supporting evidence, not a mechanism-clean flagship result.**
2. **Keep the April 8 report frozen as legacy-ruler provenance.**
3. **Keep the April 14 control/ruler audit frozen as the pre-probe reconciliation note.**
4. **If D7 later needs to become more central, the highest-value cleanup is not another random seed first; it is reducing the mixed-ruler and CSV2-error debt on the current panel.**

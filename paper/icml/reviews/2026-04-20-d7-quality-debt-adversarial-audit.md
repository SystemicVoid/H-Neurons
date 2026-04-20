# D7 reassessment against ground-truth data

**Date:** 2026-04-20  
**Scope:** test the specific claim that D7 remains too token-cap-heavy and mixed-ruler to support a serious paper-facing claim.  
**Method:** row-level recomputation from on-disk JSONL only. No summary note is treated as authoritative.

## Executive verdicts

1. **The narrow same-ruler D7 claim is real.** Under a single v3-style harmful / not-harmful ruler, the causal branch is materially better than baseline and both available random-head controls:
   - baseline: `34.2%`
   - causal: `20.0%`, paired `-14.2 pp`
   - random seed 1: `37.2%`, random-minus-causal `+17.2 pp`
   - random seed 2: `38.4%`, random-minus-causal `+18.4 pp`
2. **Token-cap debt does not drive that result.** The causal-vs-baseline gap stays large on both uncapped rows (`-13.7 pp`) and capped rows (`-16.1 pp`).
3. **Mixed-ruler debt is real, but it is a magnitude problem, not a sign-flip problem for the causal-vs-baseline/random claim.** Legacy raw, mixed-ruler normalized, and v3-only panels all disagree on effect size; they do not erase the narrower causal advantage.
4. **The stronger "D7 is now a clean selector-specific flagship" claim is still too broad.** Probe does not yet have the same v3 rescore path as baseline / L1 / causal, so the full selector hierarchy is not closed on one ruler.
5. **D7 is not a current `paper/icml/main.tex` blocker, because the current draft does not use D7.** This is a framing / future-positioning audit, not a correction to a claim already made in the ICML TeX.

This file now stays strictly on D7. D4 / FaithEval contract issues are handled separately in `2026-04-20-d4-and-anti-compliance-adversarial-audit.md`.

---

## 1. Ground truth

### 1.1 Raw sources used

Generation rows:

- `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/baseline_noop/experiment/alpha_1.0.jsonl`
- `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/l1_neuron/experiment/alpha_3.0.jsonl`
- `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_locked/experiment/alpha_4.0.jsonl`
- `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_1/experiment/alpha_4.0.jsonl`
- `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/causal_random_head_layer_matched/seed_2/experiment/alpha_4.0.jsonl`

Judgments:

- `baseline_noop/csv2_v3_evaluation/alpha_1.0.jsonl`
- `l1_neuron/csv2_v3_evaluation/alpha_3.0.jsonl`
- `causal_locked/csv2_v3_evaluation/alpha_4.0.jsonl`
- `causal_random_head_layer_matched/seed_1/csv2_evaluation/alpha_4.0.jsonl`
- `causal_random_head_layer_matched/seed_2/csv2_evaluation/alpha_4.0.jsonl`

Important boundary:

- `probe_locked` has `csv2_evaluation/alpha_1.0.jsonl`
- it does **not** have a matching `csv2_v3_evaluation/` rescore directory

So the same-ruler closure available on disk is:

> baseline / L1 / causal / random seed 1 / random seed 2

not

> baseline / L1 / causal / probe / random 1 / random 2 all on one ruler

### 1.2 Computation

For the v3-style panel, I treated `harmful_binary == "yes"` as harmful and computed paired deltas on shared prompt IDs with percentile bootstrap CIs.

This is intentionally narrower than the repo's mixed-ruler current-state summaries. The goal here is to isolate what survives when the ruler is held fixed.

---

## 2. What the same-ruler data actually says

Common prompt parity is exact at `n = 500`.

| Condition | v3 harmful yes-rate | Paired delta vs baseline |
|---|---:|---:|
| baseline | 34.2% | — |
| L1 | 36.4% | `+2.2 pp` `[-1.6, +6.2]` |
| causal | 20.0% | `-14.2 pp` `[-18.0, -10.6]` |
| random seed 1 | 37.2% | `+3.0 pp` `[-1.2, +7.2]` |
| random seed 2 | 38.4% | `+4.2 pp` `[+0.4, +8.0]` |

Direct paired comparisons against causal:

| Contrast | Delta |
|---|---:|
| random seed 1 minus causal | `+17.2 pp` `[+13.0, +21.4]` |
| random seed 2 minus causal | `+18.4 pp` `[+14.6, +22.2]` |

### 2.1 What is genuinely supported

The safe D7 sentence is:

> On the available same-ruler full-500 evidence, the causal branch beats baseline and both available layer-matched random-head controls.

That is substantially stronger than the old "too dirty to use" objection.

### 2.2 What is not yet supported

The safe sentence is **not**:

> D7 now cleanly proves causal selector superiority over correlational selectors in general.

Why not:

1. probe is not part of the same-ruler closure,
2. the full mixed-ruler hierarchy still changes magnitude substantially depending on ruler,
3. this is still one benchmark surface with one judge family.

---

## 3. Token-cap debt: real behavior, weak confound

From `causal_locked/experiment/alpha_4.0.jsonl`:

- token-cap hits: `112 / 500 = 22.4%`

Same-ruler sensitivity check, baseline vs causal:

| Subset | n | baseline | causal | delta |
|---|---:|---:|---:|---:|
| uncapped rows | 388 | 36.3% | 22.7% | `-13.7 pp` |
| capped rows | 112 | 26.8% | 10.7% | `-16.1 pp` |
| full panel | 500 | 34.2% | 20.0% | `-14.2 pp` |

So:

- the token-cap phenomenon is real and visible,
- but it is **not** the reason the causal branch looks better than baseline,
- and it is **not** enough to dismiss the causal-vs-random gap.

The older version of this audit was right to push back on "token-cap means scientifically dirty." That part stands.

---

## 4. Mixed-ruler debt: real and large, but narrower than the old objection

Three different panels exist for overlapping D7 evidence:

| Panel | Causal vs baseline |
|---|---:|
| legacy raw yes | `-9.0 pp` |
| mixed-ruler normalized current-state panel | `-26.8 pp` |
| single-ruler v3 panel | `-14.2 pp` |

This drift is **not small**. The previous version of this audit over-compressed this point when it said both confounds were quantitatively small. That was too broad.

The accurate statement is:

- **token-cap confounding is small for the paired effect**
- **ruler drift is large for absolute magnitudes**
- **but the same-ruler causal-vs-baseline/random claim survives**

That is the right synthesis.

---

## 5. Comparator-quality corrections

The current files do **not** support some of the larger stale error counts that had started circulating in adjacent D7 notes.

From the current `csv2_evaluation/*.jsonl` files:

| Condition | rows with `csv2.error` | rows with validation errors |
|---|---:|---:|
| random seed 1 | 4 | 3 |
| random seed 2 | 2 | 2 |
| probe | 2 | 2 |

Examples:

- `random seed 1`: `jbb_harmful_18_t4`, `jbb_harmful_26_t2`, `jbb_harmful_38_t0`, `jbb_harmful_91_t1`
- `random seed 2`: `jbb_harmful_41_t1`, `jbb_harmful_74_t4`
- `probe`: `jbb_harmful_49_t1`, `jbb_harmful_52_t0`

So the correct comparator-quality language is:

> probe and random branches are still non-pristine, but the visible row-level error burden in the current files is small, not massive.

That does **not** rescue the broader selector-closure claim; it only means the old bookkeeping caveat had started drifting beyond what the files now show.

---

## 6. What this means for paper-facing use

### 6.1 For the current ICML TeX

No immediate change is forced, because `paper/icml/main.tex` does not currently make a D7 claim.

### 6.2 For any future D7 promotion

If D7 is brought into the paper or into framing memos, the safe wording is:

> On a single v3-style harmful / not-harmful ruler, the causal D7 branch outperforms baseline and both available layer-matched random-head controls on the full-500 benchmark. Token-cap behavior is visible but does not drive the paired effect. Probe is not yet part of a same-ruler full-panel closure, so D7 should still be presented as benchmark-local supporting evidence rather than a clean selector theorem.

Unsafe wording:

- "D7 is still too dirty to use"
- "D7 is now mechanism-clean"
- "D7 proves causal selectors beat correlational selectors in general"
- "probe is null on the full 500 in a same-ruler comparison"

The old version of this audit was too strong in the first direction; some later framing memos were too strong in the second.

---

## Bottom line

The data no longer supports the blanket objection that D7 is too dirty to support any serious claim. A narrow, same-ruler causal-vs-baseline/random result is real and robust to the token-cap concern.

But the data still does **not** support promoting D7 as a clean flagship selector-specificity result. Mixed-ruler magnitude drift is large, and probe is not yet in the same-ruler closure.

So the correct D7 status is:

> stronger than "appendix-only debt," weaker than "clean flagship."

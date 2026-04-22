# Bridge logprob-margin mechanism check — post-run analysis (2026-04-21)

> **Verdict (data).** On 257 adjudicated-plus-control bridge cases
> (A=31 R→W substitution, B=12 R→W non-substitution, C=14 W→R rescue,
> D=200 random wrong-entity controls), ITI α=8.0 with `first_3_tokens`
> decode scope compresses the gold-vs-wrong log-likelihood margin on *all*
> R→W cohorts (A first-3 shift = −10.16 nats [−12.81, −7.60]; B =
> −40.06 [−50.09, −30.37]) and expands it on W→R rescues (C = +4.73
> [+2.27, +7.25]). A is significantly more negative than the random
> control D (ΔA−D = −5.10 nats [−8.22, −2.16], one-sided p = 0.0081), but
> is **~30 nats less negative than B**, not more (ΔA−B = +29.90 [+19.91,
> +40.54], one-sided p = 1.0). Per-position decomposition shows B's large
> shift is concentrated — but not exclusively — at the first continuation
> token (B position-0 = −21.74 nats vs. A position-0 = −3.49).
>
> **Verdict (interpretation).** The broad confirmatory claim survives:
> ITI causally shifts gold-vs-wrong margins in the direction behavioral
> outcomes demand (compression on R→W, expansion on W→R). The narrower
> *substitution-specific* mechanism claim — the "clean result" that
> Priority-3 in `TODO_Limitations_Fixes.md` was designed to test — does
> **not** survive: non-substitution cases show larger margin compression
> than substitution cases, not smaller. The precommit decision tree's
> fallback reading therefore applies: *the behavioral wrong-entity-
> substitution taxonomy is reliable (κ = 0.90, AC1 = 0.96 per the sibling
> [bridge-IRR review](./2026-04-21-bridge-irr-review.md)) but does not
> index a distinct margin-shift signature*. This is an informative
> negative result that a locked precommit protects; the paper's §4
> externality claim does not need rewording, but the margin-shift story
> cannot be framed as a "mechanistic upgrade" of the substitution
> taxonomy. It should be framed as a refinement: *ITI compresses
> answer-commitment margins on R→W discordant cases in general, with a
> disproportionate contribution from the first-token answer frame in
> evasion/refusal cases.*

## Source hierarchy

This report is the authoritative post-run prose analysis for the bridge
logprob-margin investigation. Numerical source-of-truth is
`results.json`; this report interprets it against the precommit
decision tree.

- Machine-readable summary: [`data/gemma3_4b/analysis/bridge_margins/test/results.json`](../../../data/gemma3_4b/analysis/bridge_margins/test/results.json)
- Per-case records (257 rows): [`data/gemma3_4b/analysis/bridge_margins/test/margins.jsonl`](../../../data/gemma3_4b/analysis/bridge_margins/test/margins.jsonl)
- Gate 2 sanity peek (5 rows): [`data/gemma3_4b/analysis/bridge_margins/test_sanity/margins.jsonl`](../../../data/gemma3_4b/analysis/bridge_margins/test_sanity/margins.jsonl)
- Scoring provenance: `data/gemma3_4b/analysis/bridge_margins/test/score_bridge_margins.provenance.20260421T165055Z.json`
- Analysis provenance: `data/gemma3_4b/analysis/bridge_margins/test/analyze_bridge_margins.provenance.20260421T165100Z.json`
- Plots: `data/gemma3_4b/analysis/bridge_margins/test/margin_shift_by_cohort_{first3,full}.{pdf,png}`
- Scorer: [`scripts/score_bridge_margins.py`](../../../scripts/score_bridge_margins.py)
- Analyzer: [`scripts/analyze_bridge_margins.py`](../../../scripts/analyze_bridge_margins.py)
- ITI scaler: [`scripts/intervene_iti.py`](../../../scripts/intervene_iti.py)
- Tests: [`tests/test_bridge_margins.py`](../../../tests/test_bridge_margins.py)
- Sibling IRR analysis (behavioral taxonomy): [`./2026-04-21-bridge-irr-review.md`](./2026-04-21-bridge-irr-review.md)
- Original task spec: [`../reviews/TODO_Limitations_Fixes.md`](../reviews/TODO_Limitations_Fixes.md) §Priority 3
- Precommit protocol (sealed in §A below; this file's prior content)

The precommit decision tree that this run executes against is preserved
verbatim as **Appendix A** so a reviewer can verify no endpoint was
reshaped after the numbers landed.

---

## 1. Context and question

Priority-3 in [`TODO_Limitations_Fixes.md`](../reviews/TODO_Limitations_Fixes.md)
asked whether the bridge wrong-entity-substitution taxonomy (the paper's
§4 externality claim, supported by Rater-A/Rater-B IRR with κ = 0.90) is
merely a post-hoc behavioral description or reflects an underlying
probability-mass shift the intervention causes. The TODO specified a
canonical "clean result" shape:

> On substitution-coded R→W flips, ITI reduced the gold-vs-wrong-entity
> log-likelihood margin by X nats/token, whereas non-substitution flips
> and random wrong-entity controls showed smaller or no margin shifts.

This is the hypothesis the confirmatory A-vs-B and A-vs-D tests were
designed to adjudicate. The Gate 2 sanity peek (`test_sanity/`, 5 cases)
surfaced two confounds — large first-token magnitudes and an unexpectedly
large negative shift on a B-evasion case — that tempted a post-hoc
endpoint change. The 2026-04-21 precommit (see Appendix A) locked the
analysis before the full 57+200 run executed. This report applies that
locked plan to the full-run numbers.

## 2. Pipeline audit

### 2.1 What the scorer does (data)

For each case the scorer computes four teacher-forced log-probability sums
on the continuation after the prompt:
`ell_base_gold`, `ell_base_wrong`, `ell_iti_gold`, `ell_iti_wrong`. It
then reports `delta_base = ell_base_gold − ell_base_wrong`,
`delta_iti = ell_iti_gold − ell_iti_wrong`, and
`shift_nats = delta_iti − delta_base` on two windows: `first3` (sum of
the first 3 continuation token logprobs) and `full` (whole target).
Baseline uses `scaler.alpha = 0.0` and ITI uses `scaler.alpha = 8.0`; the
same scaler object is reset (`reset_sample_stats`) and rearmed
(`arm_first_decode_token`) between targets so the token-index gating is
correct. Gold alias is chosen per-case as the alias with maximum
per-token-averaged baseline logprob (`pick_gold_alias`). Wrong-entity
text is the baseline model's own incorrect response for A/B/C and a
length-matched alias from an unrelated question for D (with
`test_control_pool_excludes_self_question` enforcing non-leakage).

### 2.2 What the analyzer does (data)

The analyzer reads `margins.jsonl` and emits per-cohort bootstrap CIs
(10,000 resamples, seed=42, percentile interval, 95%) on:
per-case `shift_nats` and `shift_per_token` in both windows;
per-position shifts at positions 0, 1, 2 (ragged lengths handled
explicitly by filtering `None`); the exploratory tokens-2–3 window
(positions 1+2 sum); and the A_pos/A_neg baseline-sign split.
Between-cohort tests (A-vs-B and A-vs-D, at first3 and tokens-2–3)
use independent-group bootstrap on the mean difference plus a one-sided
permutation test with Laplace-smoothed p-values
`p = (n_extreme + 1) / (n_perms + 1)`. The permutation H1 is
`mean(A) < mean(other)` (i.e., A more negative than other), matching the
precommit's directional hypothesis.

### 2.3 What passes muster (interpretation of audit)

1. **Baseline is a genuine no-op.** With `alpha = 0.0` the hook in
   `intervene_iti.py` short-circuits inside `_scope_allows_token` /
   `_apply_iti` without touching activations (lines 310–317).
2. **Intervention scope and measurement window coincide.**
   `_decode_scope_limit = 3` gates ITI to generated indices 1–3
   (continuation positions 0, 1, 2), which is exactly the `first3`
   measurement window. No scope/window mismatch.
3. **Cohort partition is disjoint-and-exhaustive** (`cohort_for_case`,
   `scripts/score_bridge_margins.py:119–126`) and tested
   (`tests/test_bridge_margins.py:50–114`).
4. **Controls are not contaminated.**
   `test_control_pool_excludes_self_question` enforces D's wrong-entity
   draw excludes the source question's aliases.
5. **Ragged-length handling is explicit and transparent.**
   `per_position_shift` returns `None` when either target is too short;
   `tokenwise_cohort_summary` filters before bootstrapping. Sample sizes
   shrink with position honestly (A pos-2 n=14/31, B pos-2 n=5/12, etc.)
   rather than silently coercing to zero.
6. **Provenance is complete.** Git commit
   `76d7ca360a6b98a3500b87b97f50312d7f0a0a88`, ITI artifact SHA256
   `5d57eeba…c7ae023c`, seeds, α, k, scope, resamples, and confidence
   level are recorded in both scorer and analyzer sidecars. Sanity and
   full runs pin the same code and artifact.
7. **Precommit adherence.** One-sided permutation alternative matches the
   precommit H1 (`mean(A) < mean(other)`), endpoints match the locked
   windows, no post-hoc metric was added.

### 2.4 Concrete red flags (severity-ranked)

| Severity | Flag | Source |
|---|---|---|
| Medium | No regression test that `α=0` is a bit-exact no-op. The hook short-circuits by code inspection but not by test. | `tests/test_bridge_margins.py` lacks an alpha-sweep case |
| Medium | Position-2 n is asymmetric across cohorts (A=14/31, B=5/12, C=4/14, D=92/200) because not every gold alias spans 3 tokens. Expected, but B's position-2 estimate rests on n=5; report should always co-cite n. | `results.json .cohorts.*.tokenwise.position_2` |
| Medium | A-vs-B tokens-2–3 uses A n=14, B n=5. Adequately powered to *measure*, underpowered to *reject* the precommit's "does B attenuate?" question with confidence. | `results.json .between_cohort.A_vs_B.tokens_2_3` |
| Low | No explicit NaN/Inf guard on per-token logprobs. `torch.log_softmax` could in principle produce NaN; would propagate silently through bootstrap. Unlikely in practice. | `scripts/score_bridge_margins.py` `score_continuations` |
| Low | ITI artifact SHA256 is recorded but not runtime-validated. A future audit that re-runs and re-hashes could miss an intervening modification. | `scripts/score_bridge_margins.py` provenance write |
| Low | `wrong_entity_text` is tokenized as-is from `incorrect_response`; leading/trailing whitespace is not stripped. Would matter only if the baseline queue emits heterogeneous whitespace. | `scripts/score_bridge_margins.py:519` |
| Low | `pick_gold_alias` uses per-token-averaged logprob; for 1-token aliases this reduces to a single-token logprob and is noisier. | `scripts/score_bridge_margins.py:330–366` |
| Negligible | Permutation RNG seeded at `seed + 1`; trivial correlation with bootstrap. Common practice. | `scripts/analyze_bridge_margins.py:167` |

### 2.5 Test coverage gaps (known unknowns)

Not covered by `tests/test_bridge_margins.py`:
`α=0` no-op equivalence; end-to-end scoring on a real case with verified
numbers; KV-cache reset behavior between conditions; NaN/Inf logprob
propagation; tokenization differences when `gold` and `wrong` share a
leading token.

None of these are blockers for the current numerical results (they would
manifest as implausible values, which we do not see), but they are the
natural next additions if this surface is depended on more heavily.

---

## 3. Results — data

All numbers trace to `results.json` at the JSONPaths given in the right-
hand column. This section is data only; interpretation lives in §4–§8.

### 3.1 Cohort sizes and baseline preference

| Cohort | n | baseline `delta_base` mean (first3) | fraction with `delta_base > 0` |
|---|---:|---:|---:|
| A — R→W substitution | 31 | +3.41 nats | 0.71 |
| B — R→W non-substitution | 12 | **+27.55 nats** | **1.00** |
| C — W→R rescue | 14 | (negative by construction) | — |
| D — random control | 200 | — | — |

*Source:* `.cohort_sizes`, `.baseline_sign_check.*`, `.cohorts.A_rw_substitution.first3.delta_base_mean`.

**Non-obvious:** B cases start with a *much* stronger baseline gold
preference than A cases (+27.55 vs +3.41 nats, all 12/12 positive). B is
not "the model thought the wrong answer was right"; it is "the model
knew the right answer but emitted a non-entity evasion/refusal/dilution
response at temperature/sampling." This selection-by-behavior shapes
§4's interpretation of the A-vs-B direction.

### 3.2 Primary endpoint — first-3-token `shift_nats`

| Cohort | n | estimate (nats) | 95% CI | JSONPath |
|---|---:|---:|---|---|
| A | 31 | **−10.16** | [−12.81, −7.60] | `.cohorts.A_rw_substitution.first3.shift_nats` |
| B | 12 | **−40.06** | [−50.09, −30.37] | `.cohorts.B_rw_nonsubstitution.first3.shift_nats` |
| C | 14 | **+4.73** | [+2.27, +7.25] | `.cohorts.C_wr_rescue.first3.shift_nats` |
| D | 200 | **−5.05** | [−6.55, −3.60] | `.cohorts.D_random_control.first3.shift_nats` |

Per-token normalization (`shift_per_token`) — same sign, same ordering:
A = −3.69 [−4.67, −2.74], B = −16.15 [−20.65, −12.13], C = +2.06 [+0.99,
+3.16], D = −1.83 [−2.38, −1.30].

### 3.3 Secondary endpoint — full-continuation `shift_nats`

| Cohort | n | estimate (nats) | 95% CI |
|---|---:|---:|---|
| A | 31 | −11.32 | [−14.28, −8.42] |
| B | 12 | −44.32 | [−55.22, −33.79] |
| C | 14 | +5.09 | [+2.52, +7.82] |
| D | 200 | −5.42 | [−6.97, −3.95] |

Full continuation reproduces the first-3 ordering and signs with slightly
larger magnitudes; no qualitative change.

### 3.4 Diagnostic — per-position decomposition

Shift in nats at continuation position *p* (0-indexed), bootstrap mean
and 95% CI. Sample size is smaller at higher positions because not every
gold/wrong pair spans 3 tokens.

| Cohort | pos 0 (n) | pos 1 (n) | pos 2 (n) |
|---|---|---|---|
| A | −3.49 (31), CI [−5.14, −1.97] | −3.15 (30), CI [−5.06, −1.47] | −5.39 (14), CI [−8.28, −2.87] |
| B | **−21.74** (12), CI [−29.48, −14.85] | −8.19 (8), CI [−16.51, −2.14] | −15.64 (5), CI [−24.23, −7.08] |
| C | +3.34 (14), CI [+1.70, +4.84] | +1.18 (14), CI [−1.39, +4.59] | +0.16 (4), CI [−6.58, +6.42] |
| D | −2.61 (200), CI [−3.65, −1.60] | −1.35 (193), CI [−2.14, −0.53] | −1.29 (92), CI [−2.64, −0.07] |

*Source:* `.cohorts.*.tokenwise.position_{0,1,2}`.

**Structural feature:** cohort B's position-0 shift (−21.74 nats) is 6×
cohort A's (−3.49) and ~8× cohort D's (−2.61). The gap between A and B
at positions 1 and 2 is narrower in absolute terms but still favors B
being more negative.

### 3.5 Between-cohort tests (precommitted)

One-sided permutation test, H1: `mean(A) < mean(other)`. p-value is
Laplace-smoothed `(n_extreme + 1) / (n_perms + 1)`; n_perms = 10,000.

| Comparison | window | Δ = mean(A) − mean(other) | 95% CI on Δ | one-sided p | matches H1? |
|---|---|---:|---|---:|---|
| A vs. B | first3 | **+29.90** | [+19.91, +40.54] | 1.000 | **no — reversed** |
| A vs. D | first3 | **−5.10** | [−8.22, −2.16] | **0.0081** | yes |
| A vs. B | tokens 2–3 | +17.43 | [+7.97, +26.20] | 0.997 | no — reversed, narrower |
| A vs. D | tokens 2–3 | −4.95 | [−9.90, −0.46] | 0.0273 | yes, survives |

*Source:* `.between_cohort.A_vs_{B,D}.{first3,tokens_2_3}.shift_nats`.

### 3.6 Exploratory — baseline-sign subgroup within A

| Subgroup | n | first3 `shift_nats` | 95% CI | tokens 2–3 (n) |
|---|---:|---:|---|---|
| A_pos — baseline prefers gold (`delta_base > 0`) | 22 | −12.06 | [−15.01, −9.34] | −9.32 (12), CI [−14.67, −4.60] |
| A_neg — baseline ambivalent/wrong (`delta_base ≤ 0`) | 9 | −5.51 | [−10.01, −1.11] | −4.50 (2), CI [−6.19, −2.81] |

*Source:* `.within_A_baseline_sign.*`.

Both subgroups exclude zero on first3. A_pos is ~2× the magnitude of
A_neg; CIs overlap at the boundary.

### 3.7 Spot-check — extremes within A

Three cases with largest negative shift and three with least-negative /
positive shift in cohort A (first3), to make the distribution legible:

| case_id | question (truncated) | gold | wrong_entity_text (truncated) | first3 shift | first3 `delta_base` |
|---|---|---|---|---:|---:|
| `bridge_test_case_195dbe046184` | Give a year in the life of St Ignatius Loyola… | `1491-1556` | `1516-1591: From Basque nobleman…` | −30.21 | +27.12 |
| `bridge_test_case_127c6afb83c8` | …youngest person to hold the position of Chief Scout? | `Bear Grylls` | `James Palmer was the youngest.` | −20.77 | +6.53 |
| `bridge_test_case_1c2fb525fd6c` | Which Christmas condiment…marshworts? | `Cranberry sauce` | `Mulled wine` | −4.05 | −2.44 |
| `bridge_test_case_32404330f6ba` | Hans Langsdorff commanded which pocket battleship…? | `Graf Spee` | `SMS Bayern` | +0.18 | −14.82 |
| `bridge_test_case_984567b64819` | In physics, what is a substance that continually deforms…? | `Fluids` | `Viscoelastic material` | +5.86 | −13.18 |

*Source:* `.margins.jsonl`, filtered by `cohort == "A_rw_substitution"`.

**Pattern:** the two positive-shift A cases are both A_neg (baseline
ambivalent, `delta_base < 0`) — ITI marginally *helped* them. The three
most-negative A cases include two with strong positive baseline (A_pos)
and one A_neg. No A case has a shift that warrants exclusion as a
data-quality error.

---

## 4. Applying the precommit decision tree

The precommit (Appendix A) laid out explicit readout rules. Working
through them with the full-run numbers:

### 4.1 Broad confirmatory claim

> "ITI harms the gold-vs-wrong log-likelihood margin on the manipulated
> prefix in R→W cases and reverses the sign of that shift in W→R
> rescue cases."

Requires: A first3 < 0 with CI excluding zero; C first3 > 0 with CI
excluding zero.

- A: −10.16 [−12.81, −7.60] — **holds**
- C: +4.73 [+2.27, +7.25] — **holds**

**The broad claim survives.** ITI has a directionally-signed causal
effect on gold-vs-wrong logprob margins that tracks the behavioral
transition.

### 4.2 Narrower A-vs-B mechanism claim

The precommit offered three branches:

- **(i)** A stays more negative than B on tokens 2–3 → substitution-specific story survives.
- **(ii)** B strongly negative on first3 but attenuates on tokens 2–3 → early answer-framing confound at token 0 plus a weaker content-token effect.
- **(iii)** B remains as negative as A on tokens 2–3 → drop the A<B headline.

Observed on the full run:

- First3: B = −40.06, A = −10.16. B is ~4× more negative. Branch (i) is ruled out.
- Tokens 2–3: B = −26.07, A = −8.64. B is still ~3× more negative. Not attenuation; still B < A.
- Position 0 alone: B = −21.74, A = −3.49 — so position 0 carries roughly half of B's tokens-0–2 total but is not the entire story.

This is a **hybrid of branches (ii) and (iii)**: the position-0 answer-
frame *is* a disproportionate contributor in B (consistent with
evasion/refusal openers such as "The bombing's perpetrators remain
officially unidentified"), but even when position 0 is dropped, B
remains more compressed than A. The honest reading is branch (iii):
**drop the A<B headline**, with a branch-(ii) diagnostic footnote on the
first-token contribution.

### 4.3 Baseline-sign subgroup within A

Precommit: both subgroups strongly negative → single mechanism; only
A_pos negative → two phenomena.

- A_pos: −12.06 [−15.01, −9.34] — clearly negative.
- A_neg: −5.51 [−10.01, −1.11] — negative, CI just excludes zero.

Both subgroups exclude zero. Per precommit, **single-mechanism reading
survives**, with amplitude modulated by baseline preference (A_pos is ~2×
A_neg). I will not split A into two mechanisms for the paper.

### 4.4 Language / framing (precommit §"Language / framing")

Precommit required (a) reporting `shift_per_token` or per-position
decomposition alongside nats totals and (b) prose that says
"log-likelihood margin" and "teacher-forcing" rather than "generation
probability." Both satisfied: §3.2–3.4 above.

### 4.5 Summary of precommit adherence

| Precommit item | Status |
|---|---|
| Confirmatory primary = first3 `shift_nats` A-vs-B, A-vs-D, one-sided permutation | Executed unchanged |
| Locked secondary = full continuation, same comparisons | Executed (§3.3) |
| Mandatory diagnostic = per-position shifts at 0, 1, 2 with bootstrap CI | Executed (§3.4) |
| Exploratory tokens-2–3 window | Executed (§3.5, §3.6) |
| Exploratory A_pos / A_neg split | Executed (§3.6) |
| Don't retune window after seeing data | Honored — no endpoint change |
| Report per-token or per-position alongside totals | Honored (§3.2, §3.4) |

---

## 5. What stands up under scrutiny

1. **Directional causal effect of ITI on gold-vs-wrong margins.** A, B,
   C all have CIs that exclude zero and point in the precommit-predicted
   direction for their transition. The sign of the shift is not an
   artifact of noise — C (+4.73 nats) is cleanly opposite A/B. n=14 is
   modest for C but the CI is tight enough to be convincing for the
   directional claim.
2. **A significantly more compressed than random-wrong controls.**
   A-vs-D first3 ΔA−D = −5.10 nats, 95% CI [−8.22, −2.16], one-sided
   p = 0.0081. This survives dropping position 0 (tokens 2–3 ΔA−D =
   −4.95, CI [−9.90, −0.46], p = 0.027). Substitution cases carry a
   margin-shift signal that random-wrong controls do not — but the
   magnitude is modest (~5 nats of additional compression).
3. **The pipeline is sound under audit.** §2 red flags are all either
   medium (test coverage, n at position 2) or low (defensive hardening).
   None alter the numbers reported here.
4. **Precommit protocol worked.** The Gate 2 sanity peek had already
   noticed the B-evasion outlier and the large first-token magnitudes,
   and would have tempted an endpoint change. The lock held. The
   surprising A > B direction lands as an evidentially clean negative
   result.
5. **Sanity ↔ full-run agreement.** Each of the five locked sanity cases
   falls within the full-run cohort CI at the first3 window (§11 below).

## 6. What does not stand up

1. **The Priority-3 "clean result" shape is contradicted.** The TODO
   predicted "non-substitution flips and random wrong-entity controls
   showed smaller or no margin shifts" than substitution flips. What we
   measure: B > A > D > C (in compression magnitude on first3); B shows
   ~4× A's compression. The directional hypothesis A < B is formally
   rejected by the one-sided permutation test (p = 1.0; the sign of the
   observed difference is positive, not negative).
2. **The substitution taxonomy does not mechanistically outrank the
   non-substitution taxonomy** at the logprob-margin level. Both
   categories show margin compression; if anything, evasion/refusal
   (B) shows *stronger* compression. The paper's behavioral taxonomy
   (κ = 0.90) is reliable as a *description* of what the model emits,
   but this analysis does not upgrade it to a *mechanistic* label.
3. **Any paper prose implying "ITI specifically suppresses the gold
   entity in substitution cases" overstates the evidence.** The signal
   is "ITI suppresses the gold-vs-wrong margin on R→W discordants
   generally, more strongly on non-substitution, with a large first-
   token component in non-substitution cases."

## 7. Uncertainties (calibrated)

**High confidence.** The sign and approximate magnitude of shifts in
A, B, C, D. The A-vs-B ordering (B more compressed). The A-vs-D
difference and its modest size. The precommit branch that fires
(branch iii, with branch-ii flavoring).

**Moderate confidence.** That position 0 is the *primary* driver of B's
gap over A (it accounts for ≈ 54% of B's first3 shift by magnitude; the
residual tokens-2–3 gap narrows from 29.9 to 17.4 nats but does not
close). That A_pos / A_neg reflects a single mechanism with amplitude
modulation rather than two phenomena (CIs overlap at the boundary;
A_neg n = 9 is small).

**Lower confidence.** Whether B's extra compression reflects a deeper
ITI mechanism on evasion/refusal cases or a prompt-structure artifact
of evasion-case continuations that forces particular answer frames.
Whether the substitution taxonomy fails to index a distinct margin-
shift signature because the mechanism is domain-general or because our
cohorts are too small to resolve a real but smaller-than-expected
difference. The IRR work (κ = 0.90, AC1 = 0.96) argues the taxonomy is
reliable, so "noisy labeling" is not the dominant explanation.

**Unknown (out of scope).** Whether the picture holds on a second model
(only Gemma-3-4B-IT was run). Whether it holds at other α (only α=8.0).
Whether it holds on a different benchmark (TriviaQA-bridge only). The
position-2 estimates for B (n=5) and C (n=4) are single-digit samples
and would move materially under outlier removal; treat those specific
numbers as indicative, not confirmed.

## 8. Insights (the non-obvious scientific findings)

The target outcome is contradicted; the *actual* findings are worth
stating explicitly, because they re-shape the mechanistic story more
informatively than a confirmation would have.

1. **The behavioral substitution taxonomy, though inter-rater reliable
   (κ = 0.90), does not index a distinct mechanistic signature at the
   logprob-margin level.** Non-substitution (evasion/refusal/dilution)
   R→W cases show ~4× the margin compression of substitution cases on
   the first-3-token window, and ~3× on the tokens-2–3 window. If the
   substitution taxonomy tagged a distinct neural failure mode, the
   margin-shift signal should be at least as large in A as in B. It is
   not.
2. **The effect appears to live substantially in an "answer-frame"
   commitment, not in per-entity retrieval.** B's first-token shift
   (−21.74 nats) is 6× A's. Evasion-case wrong outputs often begin with
   a generic disclaimer frame ("The bombing's perpetrators remain…",
   "Roe is not typically used…", "He did not complete…"). ITI reduces
   the margin between that frame and the gold answer drastically at
   position 0, consistent with ITI steering affecting a
   generic answer-commitment axis rather than a targeted wrong-entity
   insertion. This matches the ITI literature's picture of truthfulness
   steering as a direction in activation space, not a per-claim edit.
3. **B cases have much stronger baseline gold preference than A cases
   (+27.55 vs +3.41 nats; 100% vs 71% positive).** Non-substitution
   wrong answers are behaviorally cases where the model *knew* the
   right answer and emitted a non-entity response anyway. This is a
   selection-on-behavior artifact of how cohorts are defined and
   partially explains *why* ITI has so much more margin to compress
   in B: there is more standing delta to shrink.
4. **The precommit protocol prevented a quiet narrative flip.** The
   sanity peek had already surfaced the confounds; without the
   precommit, it would be tempting to report "A vs. D" as the
   confirmatory test and demote A-vs-B to secondary, which would hide
   the fact that the taxonomy does not index a mechanistic signature.
   The lock forced the negative result to land as a result.

## 9. Paper framing implications

- **Do not** write the bridge-margin story as "ITI reduces the
  gold-vs-wrong margin by X nats/token *on substitution cases*." That
  framing is contradicted.
- **Acceptable framing** for a supplement or short §4 paragraph:
  "Teacher-forced log-likelihood margins confirm that ITI compresses
  the gold-vs-wrong margin on R→W discordant cases (A = −10.2 nats
  [−12.8, −7.6]; B = −40.1 [−50.1, −30.4]) and expands it on W→R
  rescues (C = +4.7 [+2.3, +7.3]). Substitution-coded cases are
  distinguishable from length-matched random-wrong controls
  (A−D = −5.1 [−8.2, −2.2], p = 0.008) but show *smaller* margin shift
  than non-substitution R→W cases, with the latter's shift
  concentrated at the first-token answer frame. The behavioral
  wrong-entity-substitution taxonomy is therefore a reliable
  description of what the model emits (κ = 0.90), but does not index a
  distinct margin-shift signature at the logprob level."
- The paper's main §4 externality claim (dominant failure mode is
  wrong-entity substitution, 72.1% [57.3, 83.3]) is **unaffected** by
  this finding. The margin-shift analysis refines rather than
  contradicts the externality story.
- Suggested placement: one-line footnote in §4 pointing to a short
  supplement paragraph, not a main-text paragraph. Do not let the
  margin-shift story pull rhetorical weight away from the 72.1%
  behavioral claim, which is what the paper actually earned.
- The L4 limitation row can remain as written; this analysis neither
  closes nor widens L4.

## 10. Next steps (ranked by value/cost)

| Priority | Action | Value | Cost | Risk |
|---|---|---|---|---|
| High | Update [`TODO_Limitations_Fixes.md`](../reviews/TODO_Limitations_Fixes.md) Priority-3 entry to mark executed; record directional outcome; link this report | Closes the audit trail | 10 min | Nil |
| High | Remove stale [`notes/runs_to_analyse.md`](../../../notes/runs_to_analyse.md) entry per notes-folder policy | Hygiene | 5 min | Nil |
| High | Add a cross-link from the sibling [IRR review](./2026-04-21-bridge-irr-review.md) §7.2 to this report | Readers landing on IRR first find the mechanism outcome | 5 min | Nil |
| Medium | Decompose position 0 vs. positions 1–2 as separate panels in the margin-by-cohort figure, not a single first-3 sum | Clarifies the answer-frame story visually | 1 hr | Nil |
| Medium | Add a regression test that `α=0` is a bit-exact no-op for `score_continuations` | Closes the §2.4 medium-severity gap | 30 min | Nil |
| Medium | Run the same scorer at α ∈ {4, 16} and (budget permitting) on the Mistral anchor to test whether B > A is model/α-robust | Tests generalization of the §8 insights | 2–4 hr GPU | Low |
| Low | NaN/Inf guard in scorer, surface any flagged cases in provenance | Defensive | 30 min | Nil |
| Low | Whitespace-strip `wrong_entity_text` at tokenization; spot-check existing records | Defensive | 20 min | Nil |
| Low (skip) | Expand B cohort via new discordant set | Would tighten position-2 CIs; unlikely to change direction | High GPU cost | Low |

## 11. Sanity ↔ full-run agreement check

All five Gate 2 sanity cases fall within the full-run cohort first3 CI.

| case_id | cohort | gold | wrong (trunc) | sanity shift | full-run cohort CI |
|---|---|---|---|---:|---|
| `bridge_test_case_127c6afb83c8` | A | Bear Grylls | James Palmer was the youngest. | −20.77 | within A range [−30.21, +5.86] |
| `bridge_test_case_195dbe046184` | A | 1491-1556 | 1516-1591: From Basque nobleman… | −30.21 | within A range |
| `bridge_test_case_1c2fb525fd6c` | A | Cranberry sauce | Mulled wine | −4.05 | within A range |
| `bridge_test_case_0bf0a66035e1` | B | A Libyan national | The bombing's perpetrators remain… | −34.85 | within B CI [−50.09, −30.37] |
| `bridge_test_case_0b87bdfc8ef2` | C | Henry Cooper | Rocky Marciano | +5.33 | within C CI [+2.27, +7.25] |

The sanity cases agreed qualitatively with the full run on every
directional expectation, including the B-evasion case's large magnitude.
No evidence of data corruption between sanity and full runs.

---

## Appendix A. Sealed precommit protocol (2026-04-21, preserved verbatim)

*This appendix reproduces the original content of this file as written
on 2026-04-21, before the full 57+200 run executed. It is preserved
verbatim so reviewers can verify the decision tree used in §4 was
locked before the data landed. Only formatting has been preserved;
the text of the precommit is frozen.*

> # Bridge logprob-margin mechanism check — **precommit**
>
> **Date:** 2026-04-21 (written *before* the full 57+200 run executes).
>
> **Purpose of this note.** The Gate 2 sanity peek (5 cases) surfaced two
> phenomena that invite post-hoc window retuning: (i) magnitudes at first-3
> tokens are large (5–35 nats, not 0.5–5), and (ii) one B-evasion case showed
> a larger negative shift than any A-substitution case, apparently because
> token 1 under ITI raises `p("The …")` as a generic answer-frame rather
> than as an entity commitment.
>
> Changing the primary endpoint after seeing that data would be exactly the
> kind of researcher-degrees-of-freedom move that confirmatory analyses are
> supposed to exclude. This note locks what is confirmatory, what is
> secondary, what is diagnostic, and what is exploratory, **before** any
> further data lands, and pins the interpretation logic for every plausible
> outcome so nothing has to be decided by eye after the fact.
>
> ---
>
> ## Locked analysis plan
>
> ### Confirmatory primary
>
> **First-3-token shift_nats** per case, cohort A vs. cohort B and
> cohort A vs. cohort D, two-sample bootstrap mean difference with a
> one-sided permutation test (H1: `mean(A) < mean(B or D)`).
>
> Why: first-3 tokens is exactly the window where the ITI hook fires
> (`decode_scope=first_3_tokens`). Any shift attributable to the
> intervention must be measurable there.
>
> ### Locked secondary (sensitivity)
>
> **Full-continuation shift_nats** per case, same cohort comparisons.
>
> Role: confirm the finding is not an artefact of the 3-token boundary.
>
> ### Mandatory diagnostic
>
> **Per-position shift decomposition** at positions 0, 1, 2 (0-indexed) for
> every cohort, with bootstrap mean + 95% CI. Answers "where inside the
> manipulated prefix does the shift live?"
>
> ### Exploratory (labeled as such in the paper)
>
> 1. **Tokens-2–3 window**: sum of shifts at positions 1 and 2 (0-indexed).
>    Still entirely inside the hooked surface; drops position 0, which the
>    sanity run suggests may be dominated by answer-frame initiation
>    (`"The …"`) rather than entity commitment. **Token 4 is explicitly not
>    added**: it is outside the intervention scope, so a 2–4 window is
>    neither a clean confirmatory endpoint nor a clean causal readout.
>
> 2. **Within-A baseline-sign split**: partition cohort A by
>    `sign(first3.delta_base)`. The two subgroups potentially mix two
>    mechanisms:
>    - `A_pos` (baseline already prefers gold): clean ITI-induced reversal.
>    - `A_neg` (baseline ambivalent/wrong-leaning): ITI amplifies an already
>      shaky baseline preference.
>    Both are real, but the story is different.
>
> ---
>
> ## Precommitted interpretation logic (decide reading *before* the numbers land)
>
> ### Broad confirmatory claim
>
> > ITI harms the gold-vs-wrong log-likelihood margin on the manipulated
> > prefix in R→W cases and reverses the sign of that shift in W→R rescue
> > cases.
>
> This claim stands on: A first-3 shift is negative and its CI excludes
> zero; C first-3 shift is positive and its CI excludes zero.
>
> ### Narrower mechanism claim (A vs. B)
>
> Readout logic (decided now, so it cannot be reshaped by the numbers):
>
> - If **A stays more negative than B on tokens 2–3**: the substitution-
>   specific story survives the opener confound. Headline "bridge fails via
>   wrong-entity substitution" holds, with the caveat that token 0 carries
>   additional generic-opener variance.
> - If **B is strongly negative on first-3 but that attenuates on
>   tokens 2–3**: report as "early answer-framing confound at token 0 plus a
>   weaker substitution-specific content-token effect at tokens 1–2." This
>   is a narrower, honest version of the mechanism claim.
> - If **B remains as negative as A on tokens 2–3**: drop the A<B headline.
>   Broaden to "ITI compresses the gold-vs-wrong margin on R→W flips
>   generally; the behavioral taxonomy does not correspond to a distinct
>   margin-shift signature."
>
> ### Baseline-sign subgroup
>
> - If `A_pos` and `A_neg` are both strongly negative: a single mechanism
>   hypothesis survives. Report as primary.
> - If only `A_pos` is strongly negative while `A_neg` is near zero: the
>   substitution headline is driven by clean ITI-induced reversals; the
>   `A_neg` subgroup is amplification of baseline ambivalence and should be
>   reported as a distinct phenomenon, not merged.
>
> ### Language / framing
>
> - Report **shift_per_token** or the per-position decomposition alongside
>   total nats. Do not characterise these as "small margin compression";
>   with 3-token windows, per-position shifts of 1–5 nats are large early
>   log-likelihood effects on the manipulated surface.
> - The prose must say "log-likelihood margin" and "teacher-forcing", not
>   "generation probability". We measure what the model *would* assign, not
>   the sampled trajectory.
>
> ---
>
> ## What this note is *not*
>
> - It is not a hypothesis test over the diagnostic / exploratory analyses.
>   Those are reported transparently, labeled exploratory, with CIs, and
>   they do not carry the main claim.
> - It is not permission to rewrite the primary after the full run. Only
>   numerical errors in the scoring pipeline can change the primary.
>
> ## Pointer
>
> - Scorer: `scripts/score_bridge_margins.py`
> - Analyzer: `scripts/analyze_bridge_margins.py`
> - Outputs: `data/gemma3_4b/analysis/bridge_margins/test/`
> - Gate 2 sanity: `data/gemma3_4b/analysis/bridge_margins/test_sanity/margins.jsonl`

## Appendix B. Provenance and reproducibility

- **Git commit** (both scorer and analyzer runs): `76d7ca360a6b98a3500b87b97f50312d7f0a0a88`
- **Model**: `google/gemma-3-4b-it`
- **ITI artifact**: `iti_truthfulqa_paperfaithful_k12_alpha8.0` artifact; SHA256 `5d57eebab05865f5caa1c6a9036fc2c6e7ea013a8014a0f0a3cd3a40c7ae023c`
- **ITI config**: α=8.0, k=12, selection `ranked`, decode scope `first_3_tokens`, ITI seed 42
- **Bootstrap / permutation**: 10,000 resamples, seed 42, 95% percentile CI, Laplace-smoothed p-values
- **Control cohort (D)**: n=200 length-matched random aliases from unrelated questions, control seed 42
- **Timestamps**: scoring 2026-04-21T16:50:55Z (246.6 s wall), analysis 2026-04-21T16:51:00Z
- **Sanity run**: scoring 2026-04-21T16:26:35Z (8.3 s, n=5, dry-run), analysis 2026-04-21T16:46:28Z

Full machine-readable sidecars:
`data/gemma3_4b/analysis/bridge_margins/test/{score,analyze}_bridge_margins.provenance.*.json`.

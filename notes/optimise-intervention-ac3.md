# Optimising The Truthfulness Intervention — Act 3 Strategy

**Date:** 2026-04-01
**Status:** All five experimental stages reached terminal or near-terminal status (2026-04-01 to 2026-04-16). Stage 1 (cheap discriminators): specificity confirmed, scope locked to `first_3_tokens`. Stage 2 (artifact improvements): E1 tradeoff, E2 null, E3 gate not met. Stage 3 (bridge benchmark): built and validated, E0 ITI informative null. Stage 4 (chooser work): gate not met. Stage 5 (causal pilot): D7 two-seed current-state audit complete. The current D7 story is benchmark-local and mixed-ruler: causal remains the strongest completed branch, but selector specificity is still not mechanism-clean. See [2026-04-16-d7-full500-two-seed-current-state-audit.md](./act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md) and §10 for the frozen terminal summary.
**Model:** Gemma-3-4B-IT (`google/gemma-3-4b-it`)  
**Purpose:** Update the intervention plan so it is driven by what the repo and the primary papers actually support, not by plausible-but-loose extrapolation.

> This file is the strategy note for "what to try next."
> Current result facts live in:
> - [2026-04-16-d7-full500-two-seed-current-state-audit.md](./2026-04-16-d7-full500-two-seed-current-state-audit.md)
> - [2026-04-08-d7-full500-audit.md](./2026-04-08-d7-full500-audit.md)
> - [2026-04-07-d7-causal-pilot-audit.md](./2026-04-07-d7-causal-pilot-audit.md)
> - [2026-04-01-priority-reruns-audit.md](./2026-04-01-priority-reruns-audit.md)
> - [2026-04-01-random-head-specificity-audit.md](./2026-04-01-random-head-specificity-audit.md)
> - [2026-04-02-e1-truthfulqa-modernized-audit.md](./2026-04-02-e1-truthfulqa-modernized-audit.md)
> - [2026-04-04-e2-triviaqa-transfer-synthesis.md](./2026-04-04-e2-triviaqa-transfer-synthesis.md)
> - [2026-04-04-bridge-phase2-dev-results.md](./2026-04-04-bridge-phase2-dev-results.md)
> - [act3-sprint.md](../act3-sprint.md)
> - [measurement-blueprint.md](../measurement-blueprint.md)

### Update — 2026-04-16 (D7 full-500 two-seed current-state audit)

The canonical D7 report is now [2026-04-16-d7-full500-two-seed-current-state-audit.md](./2026-04-16-d7-full500-two-seed-current-state-audit.md). The machine-readable source of truth is `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_full500_current_state_summary.json`.

- **Current normalized strict harmfulness panel:** baseline 51.6%, L1 comparator 46.8%, random layer-matched seeds 1/2 at 37.2% and 38.8%, probe 34.8%, causal 24.8%.
- **Paired deltas vs baseline:** L1 `-4.8 pp` `[-8.8, -1.0]`; random seed 1 `-14.4 pp` `[-19.0, -9.8]`; random seed 2 `-12.8 pp` `[-17.4, -8.4]`; probe `-16.8 pp` `[-20.8, -12.8]`; causal `-26.8 pp` `[-31.0, -22.6]`.
- **Current interpretation:** causal is still the strongest completed branch, but the panel is mixed-ruler rather than mechanism-clean. Probe and both random branches carry explicit CSV2 span-validation errors, and the causal branch still has 112/500 token-cap hits.
- **Most defensible use:** D7 now supports a stronger benchmark-local selector-choice claim than the April 8 trimmed audit did, but it still should not be presented as clean proof that gradient-based selection is specifically causal in a mechanism-clean sense.

### Update — 2026-04-08 (D7 full-500 trimmed audit)

Historical note only. Current D7 claims should cite [2026-04-16-d7-full500-two-seed-current-state-audit.md](./2026-04-16-d7-full500-two-seed-current-state-audit.md).

- **Shared baseline:** 23.4% `csv2_yes` (117/500).
- **L1-neuron comparator:** 27.4% `csv2_yes`, paired **+4.0 pp** **[+0.6, +7.6]** versus baseline.
- **Causal locked intervention:** 14.4% `csv2_yes`, paired **-9.0 pp** **[-12.2, -5.8]** versus baseline.
- **Binary judge diagnostic:** same direction; causal **-10.6 pp** **[-14.0, -7.2]**.
- **Important caveat:** this was a **trimmed** confirmatory run. `probe_locked` remained incomplete and the planned `causal_random_head` negative control was skipped.

Interpretation update: D7 is now good enough to support the practical claim that the locked causal head intervention is a better jailbreak mitigation than both no-op and the current L1-neuron baseline on this benchmark surface. It is **not** yet strong enough to support a clean selector-specific mechanistic claim.

### Update — 2026-04-07 (D7 pilot result)

Pilot100 canonical run complete for both causal and probe families. Full audit: [2026-04-07-d7-causal-pilot-audit.md](./2026-04-07-d7-causal-pilot-audit.md).

- **Causal selector (gradient-based):** α=4.0 locked, **-13pp csv2_yes [CI: -21, -6]**. Statistically significant. Harmful payload share decreases monotonically. But 24% of samples hit the 5000-token cap at α=4.0 — degeneration is a concern.
- **Probe selector (AUROC-based):** α=1.0 locked, **-2pp csv2_yes [CI: -10, +6]**. Null at every alpha. At α=8.0, the probe *increases* harmful compliance (+12pp [+3, +21]).
- **Head overlap:** Jaccard = 0.11 on top-20. The methods select genuinely different heads.
- **Template heterogeneity:** The causal effect is carried by templates 1 and 2 (instruction-following and fiction framing); DAN persona (template 0) shows zero effect.

The pilot justifies proceeding to the full-500 confirmatory run. See audit report §8 for full-500 recommendations. Baseline_noop generation is in progress.

### Update — 2026-04-05 (D7 pipeline implementation ready)

The D7 execution stack is now implemented and validated at code/test level:

- Deterministic JBB paired manifests (`scripts/build_d7_jbb_manifests.py`) with disjoint extraction/pilot splits and parity metadata.
- New extraction families in [`scripts/extract_truthfulness_iti.py`](../../scripts/extract_truthfulness_iti.py):
  - `iti_refusal_probe` (probe-ranked heads on paired harmful/benign labels)
  - `iti_refusal_causal` (paired causal head ranking from harmful-prompt NLL attribution deltas)
- Deterministic jailbreak decoding controls in [`scripts/run_intervention.py`](../../scripts/run_intervention.py) with backward-compatible defaults.
- Pilot alpha lock utility (`scripts/lock_d7_alpha.py`) and paired CSV2 report utility (`scripts/report_d7_csv2.py`).
- Staged, resumable orchestrator (`scripts/infra/d7_causal_pilot.sh`) for 100→500 execution.

Implementation-only status: long GPU/judge execution has not been launched in this change set. D7 remains execution-pending, not result-complete.

---

## 0. Bottom Line

> **Status (2026-04-11):** The five stage questions in §10 have been addressed by completed experiments. See §10 for the summary table linking each question to its terminal outcome and canonical report.

1. We already have a D4-class intervention that beats H-neurons on the clean answer-selection axis.
2. We do **not** yet have a D4-class intervention that is useful for free-form factual generation.
3. The highest-value next question is therefore **not** "can we find any stronger truth vector?" It is:
   **can we make truth steering more selective at generation time, now that the first specificity control has ruled out generic matched-`K` perturbation?**
4. That means the next work should prioritize:
   - **decode-scope ablation**
   - only then **artifact improvements** (`E1`, `E2`, then conditional `E3`)
   - optional **random-direction decomposition** only if scope remains ambiguous
5. A bridge generation benchmark is a good idea, but it should be added **before chooser work**, not before the cheap control/scope discriminators.

---

## 1. What The Repo Actually Establishes

### 1.1 D4 already beats H-neurons on the clean truthfulness axis

From raw held-out TruthfulQA MC runs summarized in
[2026-04-01-priority-reruns-audit.md](./2026-04-01-priority-reruns-audit.md):

- **MC1**:
  - D1 H-neurons: `+0.9 pp` with 95% CI `[-1.7, +3.5]`
  - D4 ITI: `+6.3 pp` with 95% CI `[+3.7, +8.9]`
- **MC2 truthful mass**:
  - D1 H-neurons: `+0.03 pp` with 95% CI `[-1.54, +1.62]`
  - D4 ITI: `+7.49 pp` with 95% CI `[+5.28, +9.82]`

So the project no longer has an "is there any truth signal at all?" problem.
On answer selection, D4 is already the better intervention family.

### 1.2 D4 still fails on generation — now confirmed on two benchmarks

From the forced-commitment SimpleQA rerun in
[2026-04-01-priority-reruns-audit.md](./2026-04-01-priority-reruns-audit.md):

- Baseline (`α=0.0`): compliance `4.6%` with Wilson 95% CI `[3.5, 6.1]`
- `α=4.0`: compliance `3.7%`, delta `-0.9 pp` with 95% CI `[-1.8, +0.0]`
- `α=8.0`: compliance `2.8%`, delta `-1.8 pp` with 95% CI `[-3.1, -0.6]`
- At `α=8.0`, attempt rate still drops hard: `99.7% -> 67.0%`
- Precision remains flat within uncertainty: `4.6% -> 4.2%`, delta `-0.4 pp` with 95% CI `[-1.9, +1.1]`

Removing the explicit `"I don't know."` escape hatch helped diagnose the
surface form, but it did **not** rescue generation usefulness.

From the TriviaQA Bridge Phase 2 dev run in
[2026-04-04-bridge-phase2-dev-results.md](./2026-04-04-bridge-phase2-dev-results.md):

- Baseline (α=1.0, n=100): adjudicated `47.0%` CI [37.5%, 56.7%]
- ITI α=4.0: adjudicated `46.0%`, Δ = `-1.0 pp` CI [-6%, +4%]
- ITI α=8.0: adjudicated `40.0%`, Δ = `-7.0 pp` CI [-14%, 0%]
- Flip asymmetry at α=8: 10 right→wrong vs 3 wrong→right (McNemar p=0.096)
- Dominant failure mode: **confident wrong substitution** (5/10 right→wrong
  flips replace correct entity with plausible-but-wrong alternative)
- NOT_ATTEMPTED growth: 1 → 2 → 3 (evasion is secondary, not primary)

The bridge result confirms on a higher-headroom benchmark (47% vs SimpleQA 4.6%)
that the generation failure is not a floor effect. E0 ITI genuinely cannot improve
factual generation — the truthfulness direction redistributes probability mass among
the model's existing candidates, which helps when the wrong candidate is a
TruthfulQA-style misconception but hurts when the model simply lacks the knowledge.

### 1.3 The current ITI implementation is already decode-only

This matters because several external suggestions implicitly assume that the
repo is intervening on the full prompt.

The current code path says otherwise:

- [`scripts/intervene_iti.py`](../../scripts/intervene_iti.py):
  the scaler edits decode steps plus the final prompt position that produces
  the first generated token logits
- [`scripts/run_intervention.py`](../../scripts/run_intervention.py):
  `generate_response()` arms the scaler only for decode-time generation

So the meaningful scope question is **not** "full prompt vs decode-only."
That is already answered. The open question is **which decode tokens should be
steered**:

- all generated tokens (current behavior)
- token 1 only
- first few generated tokens only
- some other short prefix

### 1.4 Mixed-source and transfer paths are now closed

What exists locally:

- a clean paper-faithful TruthfulQA ITI artifact (E0: MC winner)
- a TriviaQA-based ITI artifact (E2: null on both selection and generation)
- evidence that the TriviaQA artifact selects different heads/positions:
  [iti_audit_baseline.md](./iti_audit_baseline.md)

What has been tested and failed:

- **E2-A** (TriviaQA source-isolated): near-inert on TruthfulQA MC and SimpleQA
- **E2-B** (TriviaQA family-default): calibration overfitting, held-out null
- **E0 on TriviaQA bridge**: informative null on generation (Phase 2 dev)
- **E3 gate**: not met (no complementary profile between E1 and E2)

The transfer/artifact-improvement lane is exhausted. The remaining value is in
D5 (externality audit) and D7 (causal head selection), not further artifact
iteration. See [E2 synthesis](./2026-04-04-e2-triviaqa-transfer-synthesis.md)
and [Bridge Phase 2 results](./2026-04-04-bridge-phase2-dev-results.md).

### 1.5 One stale reporting edge needed correction

The earlier plan note and one audit summary still used stale binary MC2 wording.
The intended MC2 metric in this repo is **continuous truthful mass**, stored in
`metric_value` by [`scripts/run_intervention.py`](../../scripts/run_intervention.py).
The aggregation path in
[`scripts/report_iti_2fold.py`](../../scripts/report_iti_2fold.py) now reads
that correctly.

Implication:

- Do **not** cite old binary MC2 summaries as if they were canonical.
- Do use MC2 truthful mass as a **secondary clean axis**, with continuous paired
  bootstrap CIs.

---

## 2. What The Literature Actually Supports

### 2.1 ITI and LITO do not justify the claim "last-token-only intervention"

From the local paper copies:

- [Inference-Time Intervention](../../papers/Inference-Time%20Intervention:Eliciting%20Truthful%20Answers%20from%20a%20Language%20Model2306.03341v6.md)
- [LITO](../../papers/Enhanced%20Language%20Model%20Truthfulnesswith%20Learnable%20Intervention%20and%20Uncertainty%20Expression2405.00301v3.md)

Both make the same distinction:

- the **probe training representation** is taken from the **answer's last token**
- the **intervention** is then applied **for each next-token prediction**
  autoregressively during generation

That is a crucial difference. The papers support a decode-scope ablation as an
open engineering question, but they do **not** support the stronger claim that
"LITO applies intervention to the last token only and therefore our bug is
full-decode steering."

### 2.2 The Universal Hyperplane paper supports diversity for probes, not yet for steering

From
[On the Universal Truthfulness Hyperplane Inside LLMs](../../papers/On%20the%20Universal%20Truthfulness%20Hyperplane%20Inside%20LLMs-2407.08582v3.md):

- TruthfulQA-only probes generalize poorly OOD
- diverse training datasets improve **probe detection** substantially
- attention-head features are stronger than layer-residual features

This is solid motivation for `E2` and `E3`.

What it does **not** show:

- that mixed-source **mass-mean steering** will fix generation-time selectivity
- that better detection automatically yields better intervention behavior
- that a mixed direction is better than a strong single-source direction on
  Gemma-3-4B-IT

So it supports "try diverse extraction," not "promote E3 to the flagship bet
before cheaper causal tests."

### 2.3 LITO supports adaptive intensity only in a higher-headroom regime

From the same LITO paper:

- context-dependent α is real
- TriviaQA-trained LITO transfers better than TruthfulQA-trained LITO
- the selector is evaluated on models and tasks with much higher baseline QA
  accuracy than our current SimpleQA surface

That supports two limited conclusions:

1. `E2` (TriviaQA-only) deserves to be treated seriously.
2. chooser/adaptive-α work should be **conditional on headroom**, not assumed.

It does **not** support skipping the cheaper control and scope tests.

### 2.4 GCM supports causal localization as a fallback, not as the next cheap move

From
[Surgical Activation Steering via Generative Causal Mediation](../../papers/Surgical%20Activation%20Steering%20via%20Generative%20Causal%20Mediation-2602.16080v1.md):

- causal mediation can outperform probe-based head selection for sparse steering
- this is especially relevant for concepts diffused across long-form outputs

This is a strong fallback if probe-selected heads still behave like a blunt
commitment damper. It is not the right **first** move because the repo still has
two cheaper discriminators:

- specificity control
- decode-scope ablation

---

## 3. Assessment Of The Outside Recommendation

| Claim | Assessment | Why |
|---|---|---|
| Shift from "better vector" to "selective generation-time steering" | **Supported** | This matches the current repo state: D4 already wins on clean MC, but fails on free-form generation. |
| Random-head control should be first | **Supported** | Required by the measurement contract and still the cleanest specificity test. |
| Decode-scope ablation is high ROI | **Supported, but rationale must be corrected** | Good idea locally; not because LITO proved last-token-only steering, but because current failure may come from steering too many generated tokens. |
| E1 and E2 should come before E3 | **Supported** | Cheaper, more interpretable, and better matched to what the literature actually validates. |
| Mixed-source E3 should not be the hero shot yet | **Supported** | Universal Hyperplane motivates it, but does not validate it as the best immediate bet. |
| FaithEval should not sit inside the flagship truthfulness score | **Supported** | Repo audits already relabeled it as diagnostic anti-compliance MCQ behavior under this harness. |
| Add a bridge generation benchmark | **Partly supported** | Good idea before chooser work, but not worth delaying the cheaper control/scope discriminators. |
| MC2 is unusable until fixed | **Too strong / outdated** | Old summaries were wrong; the raw metric and current aggregation path are now correct. |
| LITO uses last-token-only intervention | **Incorrect** | The papers use last-token activations for probe extraction, but intervene autoregressively for next-token prediction. |
| Causal head selection should move up if probe heads stay blunt | **Supported as fallback** | Reasonable after the cheap tests fail, not before. |

---

## 4. Updated Research Question

> **Status (2026-04-11):** This question has been addressed. Sub-hypotheses: (1) Specificity — confirmed direction-specific, not generic perturbation. (2) Scope — `first_3_tokens` locked; useful regularizer but does not fix generation (95% of rescued attempts are INCORRECT). (3) Artifact — E1 tradeoff, E2 null on both selector policies, E3 gate not met. (4) Selection — D7 causal heads outperform probe heads on jailbreak; no generation-side transfer tested. The overarching question ("can D4-style steering be made selective enough for generation?") is answered negatively within the ITI mass-mean framework tested here.

The right question for the next sprint slice is:

**Can D4-style truth steering be made selective enough during generation to
produce a better overall intervention than H-neurons, rather than only a better
answer-selection intervention?**

That breaks into four falsifiable hypotheses:

1. **Specificity hypothesis:** the current generation failure is direction-specific, not generic perturbation.
2. **Scope hypothesis:** the current generation failure is driven by steering too many decode steps.
3. **Artifact hypothesis:** if scope is fixed, a modestly improved artifact (`E1` or `E2`) can outperform the paper-faithful artifact.
4. **Selection hypothesis:** if probe-selected heads remain blunt after 1–3, then causal head selection is the more promising path.

---

## 5. Updated Plan

## Stage 1: Cheap Discriminators Before Any New Artifact Campaign

### 5.1 Forced-commitment random-head control

**Why first**

- Required by [measurement-blueprint.md](../measurement-blueprint.md)
- cheapest way to test whether the SimpleQA failure is a truth-direction effect
  or generic head perturbation

**Outcome**

- **Completed.**
- Canonical audit:
  [2026-04-01-random-head-specificity-audit.md](./2026-04-01-random-head-specificity-audit.md)
- Result:
  the ranked `α=8.0` configuration loses `31.0` attempt-rate points on the
  shared 200-ID slice, while all three random-head seeds stay essentially at
  baseline attempt rate.
- Interpretation:
  the current failure is **not** generic matched-`K` perturbation. It is
  specific to the ranked configuration, meaning the selected head set and/or
  its coupling to the learned directions.

**Protocol**

- Benchmark: forced-commitment SimpleQA
  (`--simpleqa_prompt_style factual_phrase`)
- Sample set: fixed 200-question topic-stratified manifest
  [`data/manifests/simpleqa_verified_control200_seed42.json`](../../data/manifests/simpleqa_verified_control200_seed42.json)
  so every control and every later generation pilot stays paired
- Artifact: existing paper-faithful production ITI artifact
  [`data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt`](../../data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_production/iti_heads.pt)
- Intervention family: current D4 head-level ITI, `K=12`
- Control mode: `--iti_selection_strategy random`
- Direction mode: `--iti_direction_mode artifact`
- Random-head seeds: `1`, `2`, `3`
- Alpha grid: `4.0`, `8.0`
- Comparator: do **not** rerun ranked D4. Reuse the existing canonical
  forced-commitment ranked run, filtered to the same 200 IDs

**What to report**

- primary metrics: attempt rate, precision, compliance
- uncertainty: Wilson 95% CIs for rates and paired bootstrap deltas vs the
  shared `α=0.0` baseline on the same 200 IDs
- secondary diagnostics: `NOT_ATTEMPTED` language profile and per-example grade
  transitions
- presentation shape: one table for the ranked subset, one per-seed table for
  random-head, plus a short seed-range summary

**Interpretation bands**

- **Direction-specific failure**:
  the ranked subset shows the known collapse, while all three random-head seeds
  stay materially closer to baseline on attempt rate and compliance
- **Generic perturbation failure**:
  at least two of three random-head seeds reproduce the ranked direction on
  attempt rate and compliance, with overlapping delta uncertainty
- **Ambiguous**:
  anything in between; if ambiguous, expand the control before moving on

**Stop/go rule**

- Only proceed to decode-scope ablation if this control does **not** already
  collapse the hypothesis into "generic perturbation at this intervention
  scale."
- If ambiguous, expand the control rather than moving to scope or artifact work.

**Status after completion**

- The control did **not** support the generic-perturbation hypothesis.
- Proceed to **5.2 Decode-scope ablation**.

### 5.2 Decode-scope ablation on the existing paper-faithful artifact

**Why second**

- directly targets the current failure mode
- cheaper than building new artifacts
- grounded in repo code semantics, not in a misread of LITO

**Implement as explicit decode-token scopes**

Avoid ambiguous language like "answer-onset only" or "last-token-only"
for generation. In the current harness, the answer starts at token 1 of the
generated continuation.

Test these scopes:

1. `full_decode` — current behavior
2. `first_token_only`
3. `first_3_tokens`
4. `first_8_tokens`

**Evaluation**

- TruthfulQA MC1 cal-val: retain a clean answer-selection gate
- forced-commitment SimpleQA pilot: primary generation readout

**Status after Gate 1 review**

- Canonical audit:
  [2026-04-01-decode-scope-gate1-audit.md](./2026-04-01-decode-scope-gate1-audit.md)
- What now withstands scrutiny:
  `first_token_only` is too weak, while `first_3_tokens` and `first_8_tokens`
  both retain most of the observed MC1 gain.
- What does **not** yet withstand scrutiny:
  a claim that either narrower surviving scope is already better than
  `full_decode`.
- Operational consequence:
  carry `full_decode`, `first_3_tokens`, and `first_8_tokens` into the 200-ID
  forced-commitment SimpleQA pilot; do not promote a new locked scope from the
  cal-val gate alone.

**Status after 200-ID generation review**

- Canonical audit:
  [2026-04-01-decode-scope-simpleqa-pilot-audit.md](./2026-04-01-decode-scope-simpleqa-pilot-audit.md)
- What now withstands scrutiny:
  `first_3_tokens` materially reduces raw meta-evasive spillover and restores
  more concise phrase-like outputs relative to `full_decode`, while
  `first_8_tokens` is intermediate but much closer to `full_decode`.
- What still does **not** withstand scrutiny:
  any claim that `first_3_tokens` is already more correct or already the locked
  best scope.
- Operational consequence:
  proceed to judged comparison for all three surviving scopes on the shared
  200-ID slice; treat `first_3_tokens` as the main narrowed candidate, but do
  not prune `first_8_tokens` post hoc from raw surface form alone.

**Promotion rule**

Promote a narrower scope if it:

- keeps a material fraction of the MC1 gain, and
- improves SimpleQA attempt/compliance relative to `full_decode`, without
  reducing precision further

If no narrower scope helps, that is strong evidence that the current artifact
problem is not just "too many decode tokens."

**Status after batch judging — STAGE 5.2 COMPLETE (2026-04-02)**

- Canonical results:
  [2026-04-02-decode-scope-simpleqa-judge-results.md](./2026-04-02-decode-scope-simpleqa-judge-results.md)
- Summary:

| Scope | α=8.0 compliance | Δ vs baseline | Bootstrap 95% CI | Attempt rate | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (α=0.0) | 5.5% (11/200) | — | — | 99.0% | 5.6% |
| `full_decode` | 2.5% (5/200) | −3.0 pp | [−6.0, 0.0] | 68.0% | 3.7% |
| `first_8_tokens` | 3.5% (7/200) | −2.0 pp | [−5.0, +1.0] | 84.5% | 4.1% |
| `first_3_tokens` | 4.0% (8/200) | −1.5 pp | [−4.0, +0.5] | 89.5% | 4.5% |

- Scope hypothesis **falsified as a complete fix.** Narrowing scope converts
  NOT_ATTEMPTED into INCORRECT (95% of rescued attempts), not into CORRECT.
  All three scopes remain below unsteered baseline compliance.
- `first_3_tokens` clears all three promotion criteria (MC1 retention ~90%;
  attempt/compliance above `full_decode`; precision not worse) and is
  **locked as the canonical default scope** for all subsequent experiments.
- Decode-scope is a useful regularizer, not a solution. The direction-quality
  hypothesis is now the primary candidate.
- E1 (5.3) has now completed with mixed outcomes (see
  [2026-04-02-e1-truthfulqa-modernized-audit.md](./2026-04-02-e1-truthfulqa-modernized-audit.md)).
  Proceed to **5.4 E2 (TriviaQA-only)** under `first_3_tokens`.

---

## Stage 2: Artifact Improvements Under The Best Scope

Only start this after Stage 1, so we do not confound "better vector" with
"better application policy." Scope is now fixed (`first_3_tokens`), so the
current live question is artifact/source quality under that scope.

### 5.3 E1 first: TruthfulQA-modernized

**Why first**

- cheapest artifact change
- tests prompt/ranking/position cleanup directly
- highest interpretability

**What changes**

- chat-template-matched extraction
- AUROC-based head ranking
- best-of first / mean / last answer positions
- same family objective: TruthfulQA-only truth steering

**What it tests**

Whether the current paper-faithful artifact is being limited by avoidable
extraction mismatch rather than by source-data limitations.

**Status (2026-04-02) — E1 EXECUTION COMPLETE**

- Canonical report:
  [2026-04-02-e1-truthfulqa-modernized-audit.md](./2026-04-02-e1-truthfulqa-modernized-audit.md)
- Outcome summary:
  E1 is a successful diagnosis of a tradeoff, not a successful solution. It
  improves SimpleQA attempt/compliance relative to the paper-faithful comparator
  on the paired 200-ID panel, but regresses both MC1 and MC2 on the paired
  2-fold TruthfulQA held-out comparison. This is a structured MC↓ /
  SimpleQA-behavior↑ tradeoff, currently on the wrong side of the sprint's
  cross-benchmark consistency requirement.
- Methodology note:
  lock-selection metadata drift and sweep auditability gaps were patched in
  pipeline code (`scripts/run_calibration_sweep.py`, `scripts/lock_config.py`)
  for future runs; completed E1 numerical outputs remain unchanged.
- **Operational next step: run E2 (TriviaQA-only) under `first_3_tokens`.**

### 5.3b E1b (conditional): entity-span position targeting

**Hypothesis:** `mean_answer_span` is diluted by syntactic filler tokens (e.g. "The answer is…").
Restricting activation extraction to the core factual entity tokens could yield a cleaner probe
direction. This is the token-position logic from the H-Neurons paper applied to ITI extraction.

**Gate:** run only if E2 also fails to produce a clean MC+generation profile, or if we
choose to continue iterating the TruthfulQA-source lane despite E1's mixed tradeoff.

**What exists in the repo**

- [`scripts/extract_answer_tokens.py`](../../scripts/extract_answer_tokens.py): GPT-4o pipeline that
  identifies minimal contiguous entity-token spans in model responses. Output format:
  `{qid: {answer_tokens: [...], judge: true/false}}`.
- [`scripts/extract_activations.py`](../../scripts/extract_activations.py): CETT (MLP `down_proj`)
  activations at those positions — **not** reusable for ITI directly (wrong architectural site;
  ITI uses `self_attn.o_proj` attention head projections).

**What would be needed**

- Re-run `extract_answer_tokens.py` on TruthfulQA best-answer strings in the ITI forced-answer
  prompt format (entity spans from free-form generation do not transfer; token positions differ).
- Add an `entity_span` position summary to
  [`scripts/extract_truthfulness_iti.py`](../../scripts/extract_truthfulness_iti.py) alongside
  the existing `first_answer_token` / `mean_answer_span` / `last_answer_token`.
- For E2 (TriviaQA), the existing annotations have partial overlap in QID space but still require
  re-identification in the forced-answer context.

**Disclaimer:** this path has not been validated. The expected gain on short TruthfulQA
best-answers is unclear — for compact factual strings, entity-span ≈ full answer span and
`mean_answer_span` already captures most of the signal. Treat as a live hypothesis pending E1
results, not a committed experiment.

### 5.4 E2 second: TriviaQA-only

**Why next**

- best-supported transfer candidate from the literature
- especially relevant once the problem is "generation usefulness," not
  "find any truth signal"

**Why not first**

- current bottleneck is still application selectivity

**Status (2026-04-03) — E2-A EXECUTION COMPLETE: NEAR-INERT**

- Canonical report:
  [2026-04-02-e2-triviaqa-source-isolated-audit.md](./2026-04-02-e2-triviaqa-source-isolated-audit.md)
- Outcome summary:
  E2-A is a valid, well-executed null for the source-isolated TriviaQA
  artifact under paper-faithful override selectors (val_accuracy ranking +
  last_answer_token). The locked config (K=40, α=6.0) produces no detectable
  effect on TruthfulQA MC1 (-0.92 pp, CI [-3.36, +1.53]),
  MC2 (+0.73 pp, CI [-1.21, +2.65]), or SimpleQA compliance (-0.50 pp,
  CI [-3.50, +2.50]). All CIs include zero. This is a negative result for
  E2-A specifically — the family-default selectors (AUROC + all_answer_positions)
  remain untested.
- Key finding: TriviaQA-derived probes are substantially weaker than
  TruthfulQA-derived probes (best E2 head val_accuracy = 0.696 < worst
  E0 head = 0.739). Direction cosines on the 4 shared selected heads are
  negative (mean = -0.163), which is suggestive but based on too small a
  sample for a strong conclusion. Rank agreement across 272 shared heads
  is moderate (ρ = 0.543).
- Calibration signal (+3.70 pp on 81 questions) did not replicate in the
  655-question held-out evaluation. All four extraction artifacts are
  bit-for-bit identical (expected for source-isolated families).
- E2-B (family-default selectors) subsequently tested and also null:
  [2026-04-03-e2b-triviaqa-familydefault-diagnostic.md](./2026-04-03-e2b-triviaqa-familydefault-diagnostic.md).
  Calibration overfitting (+6.2pp → +0.0pp held-out). Classification:
  `wrong_source_still_likely`.
- Full transfer synthesis:
  [2026-04-04-e2-triviaqa-transfer-synthesis.md](./2026-04-04-e2-triviaqa-transfer-synthesis.md).
- **TriviaQA transfer lane closed: both selector policies null.**
- **Operational next step: evaluate E3 conditional gate.** → **Gate not met (see §5.5).**

### 5.5 E3 third and conditional: mixed-source

Run mixed-source only if at least one of these is true:

- E1 helps MC but not generation
- E2 helps generation but regresses MC
- E1 and E2 appear complementary in head selection or benchmark profile

That is the right moment to ask whether mixing sources combines useful geometry.
Before that, E3 is too underconstrained.

**Status (2026-04-03) — E3 CONDITIONAL GATE NOT MET**

- E1 shows a tradeoff (MC↑ relative to own baseline, but MC↓ and
  generation-behavior↑ relative to paper). E2 shows near-inert results.
- E1 and E2-A are not complementary: E1 actively steers (wrong direction),
  E2-A does nothing under paper-faithful override selectors. Mixing an
  active-but-imperfect signal with an inert signal is unlikely to produce
  useful geometry.
- The E3 gate conditions are evaluated:
  - "E1 helps MC but not generation": **Partially met** (E1 helps MC vs own
    baseline, but not vs paper; E1 does not help generation vs own baseline).
  - "E2 helps generation but regresses MC": **Not met** (E2-A is inert on both under paper-faithful selectors).
  - "E1 and E2 appear complementary in head selection or benchmark profile":
    **Not met** (head overlap is 2.1% Jaccard; no complementary profile).
- **Decision: E3 is deprioritized.** The ITI artifact-improvement lane
  (Stage 2) has likely reached diminishing returns. Shift priority to
  D5 (externality audit), D7 (causal head selection), or the bridge
  benchmark (§5.6).

---

## Stage 3: Add A Bridge Generation Benchmark Before Any Chooser Work

The outside recommendation is right that SimpleQA alone is a poor surface for
adaptive chooser work. A chooser cannot select correct answers the model never
knows.

But this should be added **after** the Stage 1 discriminators, not before them.

### 5.6 Bridge benchmark requirement

Before any Stage B chooser / adaptive-α campaign:

- add a held-out open-ended factual QA benchmark with more headroom
- likely candidates: open-ended TriviaQA or NQ

**Status (2026-04-04) — BRIDGE BENCHMARK BUILT AND VALIDATED**

- Benchmark plan:
  [2026-04-03-bridge-benchmark-plan.md](./2026-04-03-bridge-benchmark-plan.md)
- Phase 1 (pilot): passed — IT headroom 47.3% adjudicated on 150 questions,
  grader calibrated (§3.4.1), two-metric policy locked
- Phase 2 (dev): executed — E0 ITI informative null on 100 questions
  ([2026-04-04-bridge-phase2-dev-results.md](./2026-04-04-bridge-phase2-dev-results.md))
- Phase 3 (test): not yet run — awaiting a candidate intervention that warrants
  the single-shot 500-question test. Baseline-only run can establish published
  headroom independently.
- The bridge benchmark confirmed that the generation failure is not a headroom
  problem. E0 ITI has 47% baseline headroom to work with — and still cannot
  improve accuracy. The failure is in the intervention, not the surface.

**Role of each benchmark (confirmed)**

- **TruthfulQA MC1**: primary clean answer-selection axis
- **MC2 truthful mass**: secondary clean axis
- **TriviaQA bridge**: primary development surface for generation usefulness
- **SimpleQA**: hard OOD stress test, not the sole generation benchmark
- **FalseQA / FaithEval**: diagnostics, not flagship truthfulness score

---

## Stage 4: Adaptive Candidate Selection Is Conditional

Do not jump to LITO-style chooser work unless the best single-vector artifact,
under the best decode scope, shows clear candidate headroom on the bridge
benchmark.

### 5.7 Gate for chooser work

Run a small candidate-lattice study only if:

- the single-vector intervention has a live generation effect on the bridge benchmark
- oracle best-of-K materially beats the best single α

**Status (2026-04-04) — GATE NOT MET**

The first condition fails: E0 ITI has no live generation effect on the bridge
benchmark (α=4 flat, α=8 harmful). There is no signal for a chooser to amplify.
Chooser/adaptive-α work is definitively deprioritized.

**Escalation order** (retained for reference if a future intervention creates
signal):

1. max-confidence chooser
2. simple linear verifier
3. LSTM-style verifier only if the simpler methods leave real oracle gap

---

## Stage 5: Causal Head Selection As The Fallback If Probe Heads Stay Blunt

Escalate to a small GCM-inspired pilot if:

- the completed random-head control still leaves the harmful effect specific to
  the current ranked configuration, and
- decode-scope ablation and `E1`/`E2`/`E3` still look like commitment damping

That would shift the diagnosis from "wrong scope" or "wrong dataset source" to
"wrong components."

At that point, a scoped causal head-selection pilot is more justified than
another round of dataset mixing.

---

## 6. Evaluation Rules For This Plan

These come from [measurement-blueprint.md](../measurement-blueprint.md) and are
part of the plan, not optional polish.

### 6.1 Primary result surfaces

- report uncertainty on every headline claim
- keep negative controls for each new intervention family
- report target gain **and** safety externality
- keep per-example outputs

### 6.2 Flagship claim bundle

Do **not** use a flagship macro-average that blends:

- SimpleQA
- FalseQA
- FaithEval

Those are different phenomena.

Instead:

- use TruthfulQA MC as the clean answer-selection claim
- use the future bridge benchmark as the main generation-development claim
- keep SimpleQA as the hard OOD stress test
- keep FaithEval and FalseQA as diagnostics

### 6.3 Stop conditions

Kill or pause a branch if:

- a stronger decomposition control reproduces the main effect after scope work
- a narrower decode scope does not improve generation usefulness
- `E1` and `E2` both fail under the best scope
- bridge-benchmark oracle headroom is too small for chooser work

**Status (2026-04-04):** conditions 3 and 4 are now met. E1 shows a tradeoff
(MC↓ vs paper, SimpleQA-behavior↑). E2 is null on both selector policies.
The bridge benchmark has ample headroom (47%) but E0 ITI shows no signal to
amplify. The artifact-improvement and chooser branches are both stopped.
The remaining active branches are D5 (externality audit) and D7 (causal pilot).

---

## 7. Concrete Priority Order

1. ~~Forced-commitment random-head control~~ **done:**
   [2026-04-01-random-head-specificity-audit.md](./2026-04-01-random-head-specificity-audit.md)
2. ~~Decode-scope Gate 1~~ **done:**
   [2026-04-01-decode-scope-gate1-audit.md](./2026-04-01-decode-scope-gate1-audit.md)
3. ~~Decode-scope SimpleQA pilot generation review~~ **done:**
   [2026-04-01-decode-scope-simpleqa-pilot-audit.md](./2026-04-01-decode-scope-simpleqa-pilot-audit.md)
4. ~~Decode-scope batch judging — scope `first_3_tokens` locked~~ **done:**
   [2026-04-02-decode-scope-simpleqa-judge-results.md](./2026-04-02-decode-scope-simpleqa-judge-results.md)
5. ~~E1 (TruthfulQA-modernized)~~ **done (tradeoff, dominated):**
   [2026-04-02-e1-truthfulqa-modernized-audit.md](./2026-04-02-e1-truthfulqa-modernized-audit.md)
6. ~~E2 (TriviaQA-only)~~ **done (null on both selector policies):**
   [2026-04-04-e2-triviaqa-transfer-synthesis.md](./2026-04-04-e2-triviaqa-transfer-synthesis.md)
7. ~~E3 (mixed-source)~~ **gate not met (no complementarity).** Deprioritized.
8. ~~Build the bridge generation benchmark~~ **done (TriviaQA bridge, 3-stage):**
   [2026-04-03-bridge-benchmark-plan.md](./2026-04-03-bridge-benchmark-plan.md)
9. ~~Bridge Phase 2 dev validation~~ **done (E0 ITI informative null):**
   [2026-04-04-bridge-phase2-dev-results.md](./2026-04-04-bridge-phase2-dev-results.md)
10. ~~Chooser work~~ **gate not met** (no live generation signal to amplify).
11. ~~**Run externality audit (D5).**~~ Deferred: existing cross-benchmark data may suffice.
12. ~~Run causal head-selection pilot (D7).~~ **current-state audit complete:**
    [2026-04-16-d7-full500-two-seed-current-state-audit.md](./2026-04-16-d7-full500-two-seed-current-state-audit.md) — causal is the strongest completed branch on the normalized strict-harmfulness panel, but the result remains benchmark-local and mixed-ruler rather than mechanism-clean.
13. ~~**Final synthesis (D8).**~~ Strategic assessment produced ([2026-04-11-strategic-assessment.md](./2026-04-11-strategic-assessment.md)); paper drafting not yet started.

---

## 8. What This Plan Explicitly Rejects

- treating mixed-source `E3` as the default hero shot
- citing LITO as evidence for "last-token-only intervention"
- treating SimpleQA alone as sufficient evidence for chooser upside
- using FaithEval as a flagship truthfulness metric
- relying on stale binary MC2 summaries

---

## 9. Final Judgment

> **Status (updated 2026-04-14):** The three unknowns this judgment deferred to have been substantially addressed. **Specificity:** direction-specific (random-head control confirmed on SimpleQA). **Decode scope:** useful regularizer, not a fix (`first_3_tokens` locked; INCORRECT rate unchanged). **Component choice:** the D7 current-state panel now includes probe and random full-500 branches, and causal remains the strongest completed branch on normalized strict harmfulness (24.8% vs probe 34.8%, random 37.2%, baseline 51.6%). But because the panel is mixed-ruler, error-bearing, and quality-costly, it still supports only a benchmark-local selector-choice claim. All stage results are summarized in §10.

The most defensible update is:

**deprioritize broad vector-search complexity until we know whether the current
D4 signal fails because of specificity, decode scope, or component choice.**

That does **not** kill `E2` or `E3`.
It means they should be tried in the right order:

- first fix or falsify the application hypothesis
- then test the cheapest artifact improvements
- then add the bridge benchmark
- only then pay the cost of adaptive choosers or causal localization

That is the shortest path to an intervention that is genuinely better than
H-neurons on the full sprint standard, not just on one clean MC benchmark.

---

## 10. Terminal Status Summary (2026-04-11)

> Historical snapshot only. The D7 row below reflects the 2026-04-11 state; current D7 claims are superseded by [2026-04-16-d7-full500-two-seed-current-state-audit.md](./2026-04-16-d7-full500-two-seed-current-state-audit.md).

This strategy note guided the experiment-discovery phase from 2026-04-01 to 2026-04-08. All five stages reached terminal or near-terminal status:

| Stage | Question | Terminal Status | Canonical Report |
|---|---|---|---|
| 1: Specificity + scope | Is the generation failure direction-specific? Is it scope-dependent? | Specificity confirmed. Scope is a regularizer, not a fix. `first_3_tokens` locked. | [Specificity](./2026-04-01-random-head-specificity-audit.md), [Scope](./2026-04-02-decode-scope-simpleqa-judge-results.md) |
| 2: Artifact improvements | Can a better artifact fix generation? | E1 tradeoff, E2 null, E3 gate not met. Lane exhausted. | [E1](./2026-04-02-e1-truthfulqa-modernized-audit.md), [E2](./2026-04-04-e2-triviaqa-transfer-synthesis.md) |
| 3: Bridge benchmark | Does a higher-headroom benchmark change the picture? | Built and validated. E0 ITI informative null (47% baseline, Δ = -1pp / -7pp). E1 α=8.0 formally significant (Δ = -9pp, p=0.016) — confirms method-level failure. | [Plan](./2026-04-03-bridge-benchmark-plan.md), [Phase 2](./2026-04-04-bridge-phase2-dev-results.md) |
| 4: Chooser work | Is there signal for adaptive-α? | Gate not met. No live generation signal to amplify. | [Bridge Phase 2](./2026-04-04-bridge-phase2-dev-results.md) (§5.7 above) |
| 5: Causal pilot | Does causal head selection outperform correlational? | D7 causal -9.0pp vs baseline on jailbreak. Probe null at every alpha (pilot100 scale; not completed at full-500). Random-head control not yet run. | [D7 full-500](./2026-04-08-d7-full500-audit.md), [Pilot](./2026-04-07-d7-causal-pilot-audit.md) |

### Remaining open items (not addressed by this strategy note)

- D7 random-head negative control — highest-value optional experiment for selector-specificity claim.
- Seed 0 jailbreak control scoring — pending, already generated.
- CSV v3 judge recalibration — smoke test identified calibration gap; 4th few-shot example needed before scaling.
- D5 externality audit — deferred; assess whether D1/D4/D7 cross-benchmark divergences provide sufficient externality coverage before committing GPU time.
- Capability mini-battery — not yet run for any intervention.

### Pointer

A strategic assessment for the BlueDot submission was produced on 2026-04-11: [2026-04-11-strategic-assessment.md](./2026-04-11-strategic-assessment.md). It proposes a paper framing and structure; that framing has not yet been reviewed against the full evidence base and should not be treated as settled.

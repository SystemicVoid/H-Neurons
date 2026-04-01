# Optimising The Truthfulness Intervention — Act 3 Strategy

**Date:** 2026-04-01  
**Status:** Revised after repo audit, raw-data check, code-path audit, and literature review  
**Model:** Gemma-3-4B-IT (`google/gemma-3-4b-it`)  
**Purpose:** Update the intervention plan so it is driven by what the repo and the primary papers actually support, not by plausible-but-loose extrapolation.

> This file is the strategy note for "what to try next."  
> Current result facts live in:
> - [2026-04-01-priority-reruns-audit.md](./2026-04-01-priority-reruns-audit.md)
> - [act3-sprint.md](../act3-sprint.md)
> - [measurement-blueprint.md](../measurement-blueprint.md)

---

## 0. Bottom Line

The outside recommendation is directionally useful but materially overstated in a few places.

The grounded conclusion is:

1. We already have a D4-class intervention that beats H-neurons on the clean answer-selection axis.
2. We do **not** yet have a D4-class intervention that is useful for free-form factual generation.
3. The highest-value next question is therefore **not** "can we find any stronger truth vector?" It is:
   **can we make truth steering more selective at generation time, and can we show the effect is specific rather than generic perturbation?**
4. That means the next work should prioritize:
   - forced-commitment **random-head control**
   - **decode-scope ablation**
   - only then **artifact improvements** (`E1`, `E2`, then conditional `E3`)
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

### 1.2 D4 still fails on generation, even after prompt repair

From the forced-commitment SimpleQA rerun in
[2026-04-01-priority-reruns-audit.md](./2026-04-01-priority-reruns-audit.md):

- Baseline (`α=0.0`): compliance `4.6%` with Wilson 95% CI `[3.5, 6.1]`
- `α=4.0`: compliance `3.7%`, delta `-0.9 pp` with 95% CI `[-1.8, +0.0]`
- `α=8.0`: compliance `2.8%`, delta `-1.8 pp` with 95% CI `[-3.1, -0.6]`
- At `α=8.0`, attempt rate still drops hard: `99.7% -> 67.0%`
- Precision remains flat within uncertainty: `4.6% -> 4.2%`, delta `-0.4 pp` with 95% CI `[-1.9, +1.1]`

Removing the explicit `"I don't know."` escape hatch helped diagnose the
surface form, but it did **not** rescue generation usefulness.

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

### 1.4 The repo does not yet justify a mixed-source hero shot

What exists locally:

- a clean paper-faithful TruthfulQA ITI artifact
- a TriviaQA-based ITI artifact
- evidence that the TriviaQA artifact selects different heads/positions:
  [iti_audit_baseline.md](./iti_audit_baseline.md)

What does **not** yet exist cleanly:

- a valid cross-benchmark win from `iti_triviaqa_transfer`
- any valid mixed-source ITI artifact result
- evidence that a better direction alone fixes the generation failure mode

So mixed-source remains a live bet, but it is not the best-supported **first**
bet.

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

**Run**

- Benchmark: forced-commitment SimpleQA (`factual_phrase`)
- Samples: 200 as a pilot
- Artifact family: current paper-faithful ITI
- Selection: `iti_selection_strategy=random`
- Match current D4 operating point at minimum for `α=8.0`
- If budget permits, also include `α=4.0`

**Decision rule**

- If random heads reproduce the attempt/compliance collapse, the current
  generation failure is mostly generic perturbation.
- If they do not, the failure is more likely tied to the selected heads or
  direction.

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

**Promotion rule**

Promote a narrower scope if it:

- keeps a material fraction of the MC1 gain, and
- improves SimpleQA attempt/compliance relative to `full_decode`, without
  reducing precision further

If no narrower scope helps, that is strong evidence that the current artifact
problem is not just "too many decode tokens."

---

## Stage 2: Artifact Improvements Under The Best Scope

Only start this after Stage 1, so we do not confound "better vector" with
"better application policy."

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

### 5.4 E2 second: TriviaQA-only

**Why next**

- best-supported transfer candidate from the literature
- especially relevant once the problem is "generation usefulness," not
  "find any truth signal"

**Why not first**

- current bottleneck is still application selectivity
- the repo does not yet have clean local behavioral evidence that E2 helps

### 5.5 E3 third and conditional: mixed-source

Run mixed-source only if at least one of these is true:

- E1 helps MC but not generation
- E2 helps generation but regresses MC
- E1 and E2 appear complementary in head selection or benchmark profile

That is the right moment to ask whether mixing sources combines useful geometry.
Before that, E3 is too underconstrained.

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

**Why later, not now**

- the repo does not yet have a production-ready judged open-ended TriviaQA/NQ
  intervention harness
- building that harness before the control/scope tests would slow the highest-ROI
  uncertainty reduction

**Role of each benchmark after that**

- **TruthfulQA MC1**: primary clean answer-selection axis
- **MC2 truthful mass**: secondary clean axis
- **Bridge benchmark**: primary development surface for generation usefulness
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

**Escalation order**

1. max-confidence chooser
2. simple linear verifier
3. LSTM-style verifier only if the simpler methods leave real oracle gap

This matches both the literature and best practice: start with the simplest
thing that could work.

---

## Stage 5: Causal Head Selection As The Fallback If Probe Heads Stay Blunt

Escalate to a small GCM-inspired pilot if:

- random-head control shows the current effect is direction-specific, but
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

- random heads reproduce the main effect
- a narrower decode scope does not improve generation usefulness
- `E1` and `E2` both fail under the best scope
- bridge-benchmark oracle headroom is too small for chooser work

That is how we avoid spending the sprint on elegant but low-yield complexity.

---

## 7. Concrete Priority Order

1. Run forced-commitment random-head control.
2. Implement and run decode-scope ablation on the current paper-faithful artifact.
3. If a better scope exists, lock it and carry it forward.
4. Run `E1` under the locked scope.
5. Run `E2` under the locked scope.
6. Run `E3` only if `E1` and `E2` create a real complementarity story.
7. Build the bridge generation benchmark before any chooser/LITO-style work.
8. Run chooser work only if the bridge benchmark shows real oracle headroom.
9. Escalate to a small causal head-selection pilot if probe-selected variants
   still behave like truth-flavored refusal.

---

## 8. What This Plan Explicitly Rejects

- treating mixed-source `E3` as the default hero shot
- citing LITO as evidence for "last-token-only intervention"
- treating SimpleQA alone as sufficient evidence for chooser upside
- using FaithEval as a flagship truthfulness metric
- relying on stale binary MC2 summaries

---

## 9. Final Judgment

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

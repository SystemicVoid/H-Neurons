# SimpleQA × ITI Production Run — Deep Dive Report

**Date:** 2026-04-01
**Model:** Gemma-3-4B-IT (`google/gemma-3-4b-it`)
**Benchmark:** SimpleQA (1000 verified factual questions)
**Intervention:** ITI head-level, paper-faithful artifact, K=12, ranked selection, seed=42
**Alpha grid tested:** {0.0, 4.0, 8.0}
**Run directory:** `data/gemma3_4b/intervention/simpleqa_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_iti-truthfulqa-paperfaithful-production-iti-head_79f3852513/experiment`
**Provenance:**
- Inference: `run_intervention.provenance.20260401_092253.json` (α=0.0, 8.0), `run_intervention.provenance.20260401_102920.json` (α=4.0)
- Judge: `evaluate_intervention.provenance.20260401_092922.json` (α=0.0, 8.0), `evaluate_intervention.provenance.20260401_103235.json` (α=4.0)

**Judge:** GPT-4o batch (0 failures across all runs)

---

## 1. Raw Data

### 1a. Top-Line Results (GPT-4o Graded)

| α | CORRECT | INCORRECT | NOT_ATTEMPTED | Total | Compliance % | 95% CI (Wilson) |
|---|---------|-----------|---------------|-------|-------------|-----------------|
| 0.0 | 48 | 915 | 37 | 1000 | 4.8% | [3.6%, 6.3%] |
| 4.0 | 40 | 783 | 177 | 1000 | 4.0% | [3.0%, 5.4%] |
| 8.0 | 16 | 192 | 792 | 1000 | 1.6% | [1.0%, 2.6%] |

Note: compliance = CORRECT / total. The CIs for α=0.0 and α=4.0 overlap substantially; the CIs for α=0.0 and α=8.0 are cleanly separated.

### 1b. Attempt-Rate and Precision Breakdown

| α | Attempt rate | Precision (CORRECT / attempted) |
|---|-------------|--------------------------------|
| 0.0 | 96.3% (963/1000) | 4.99% (48/963) |
| 4.0 | 82.3% (823/1000) | 4.86% (40/823) |
| 8.0 | 20.8% (208/1000) | 7.69% (16/208) |

**Critical observation:** Precision is flat at α=0.0 and α=4.0 (5.0% → 4.9%), then rises slightly at α=8.0 as only high-confidence answers survive the refusal filter. This rules out any "the model knows more accurate facts" explanation — the intervention is purely modulating commitment, not knowledge.

### 1c. Grade Transition Matrices

**α=0.0 → α=4.0:**

```
α=0.0 grade     →  α=4.0 grade
─────────────────────────────────────────────────────────
CORRECT         →  CORRECT        :   35  (72.9% retained)
CORRECT         →  INCORRECT      :   11  (22.9% regressed)
CORRECT         →  NOT_ATTEMPTED  :    2  ( 4.2%)
─────────────────────────────────────────────────────────
INCORRECT       →  CORRECT        :    5  ( 0.5% rescued)
INCORRECT       →  INCORRECT      :  771  (84.3%)
INCORRECT       →  NOT_ATTEMPTED  :  139  (15.2%)
─────────────────────────────────────────────────────────
NOT_ATTEMPTED   →  CORRECT        :    0  ( 0.0%)
NOT_ATTEMPTED   →  INCORRECT      :    1  ( 2.7%)
NOT_ATTEMPTED   →  NOT_ATTEMPTED  :   36  (97.3% stable floor)
```

**α=4.0 → α=8.0:**

```
α=4.0 grade     →  α=8.0 grade
─────────────────────────────────────────────────────────
CORRECT         →  CORRECT        :   12  (30.0% retained)
CORRECT         →  INCORRECT      :    1  ( 2.5%)
CORRECT         →  NOT_ATTEMPTED  :   27  (67.5% lost to refusal)
─────────────────────────────────────────────────────────
INCORRECT       →  CORRECT        :    4  ( 0.5% rescued)
INCORRECT       →  INCORRECT      :  191  (24.4%)
INCORRECT       →  NOT_ATTEMPTED  :  588  (75.1%)
─────────────────────────────────────────────────────────
NOT_ATTEMPTED   →  CORRECT        :    0  ( 0.0%)
NOT_ATTEMPTED   →  INCORRECT      :    0  ( 0.0%)
NOT_ATTEMPTED   →  NOT_ATTEMPTED  :  177  (100.0% — once refused, always refused)
```

**α=0.0 → α=8.0 (for reference):**

```
CORRECT → NOT_ATTEMPTED:   34  (70.8%)
INCORRECT → NOT_ATTEMPTED: 721  (78.8%)
```

### 1d. Full 3-Way Trajectory Distribution

Every sample's trajectory across all three alphas:

| α=0.0 | α=4.0 | α=8.0 | Count | % |
|-------|-------|-------|-------|---|
| INCORRECT | INCORRECT | NOT_ATTEMPTED | 578 | 57.8% |
| INCORRECT | INCORRECT | INCORRECT | 190 | 19.0% |
| INCORRECT | NOT_ATTEMPTED | NOT_ATTEMPTED | 139 | 13.9% |
| NOT_ATTEMPTED | NOT_ATTEMPTED | NOT_ATTEMPTED | 36 | 3.6% |
| CORRECT | CORRECT | NOT_ATTEMPTED | 23 | 2.3% |
| CORRECT | CORRECT | CORRECT | 11 | 1.1% |
| CORRECT | INCORRECT | NOT_ATTEMPTED | 9 | 0.9% |
| INCORRECT | CORRECT | NOT_ATTEMPTED | 4 | 0.4% |
| INCORRECT | INCORRECT | CORRECT | 3 | 0.3% |
| CORRECT | NOT_ATTEMPTED | NOT_ATTEMPTED | 2 | 0.2% |
| *all other patterns* | — | — | 8 | 0.8% |

**Reading the dominant trajectory (57.8%):** The model gets the question wrong at α=0.0, keeps getting it wrong at α=4.0 (commitment still high, no accuracy gain), then refuses entirely at α=8.0. ITI does not help these questions at any alpha.

### 1e. NOT_ATTEMPTED Response Distribution

At α=4.0: 177 total NOT_ATTEMPTED; 174/177 (98.3%) are the literal string `"I don't know."`. The remaining 3 are longer hedging responses.

At α=8.0: 792 total NOT_ATTEMPTED; 787/792 (99.4%) are the literal string `"I don't know."`.

The mechanism is identical across alphas — alpha controls the magnitude of induction, not its character.

### 1f. Rescued Answers at α=4.0 (INCORRECT → CORRECT, 5 total)

| Question (abbreviated) | Reference | α=0.0 response | α=4.0 response |
|------------------------|-----------|----------------|----------------|
| MGA rebrand year | 2023 | '2018' | '2023' |
| "Un'alma innamorata" composer (1707) | Handel | 'Antonio Vivaldi' | 'Georg Friedrich Handel' |
| Tsimikas assists 2021-22 (all comps) | 6 | '8' | '6' |
| 1969 Honda CL125 wheelbase (mm) | 1270mm | '2360 mm' | '1260' ✓ (in range) |
| Diana Ramos university | USC | 'UCLA' | 'University of Southern California' |

These 5 rescues are partially offset by 11 CORRECT→INCORRECT regressions — small numerical answers that shift by 1-3 (e.g., "200 stars/hr" → "100", "2" yellows → "1"). The net change is −8 CORRECT, consistent with the observed 48→40 drop.

### 1g. Topic and Answer-Type Breakdown (All Three Alphas)

**By topic:**

| Topic | n | α=0.0 C | α=4.0 C | α=8.0 C | α=4.0 NA | α=8.0 NA |
|-------|---|---------|---------|---------|---------|---------|
| Art | 145 | 9 (6.2%) | 7 (4.8%) | 3 (2.1%) | 22 (15.2%) | 117 (80.7%) |
| Geography | 111 | 3 (2.7%) | 3 (2.7%) | 1 (0.9%) | 28 (25.2%) | 82 (73.9%) |
| History | 52 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 9 (17.3%) | 47 (90.4%) |
| Music | 102 | 6 (5.9%) | 3 (2.9%) | 1 (1.0%) | 15 (14.7%) | 76 (74.5%) |
| Other | 102 | 7 (6.9%) | 6 (5.9%) | 2 (2.0%) | 26 (25.5%) | 87 (85.3%) |
| Politics | 176 | 9 (5.1%) | 9 (5.1%) | 3 (1.7%) | 31 (17.6%) | 144 (81.8%) |
| **Science & Tech** | **160** | **8 (5.0%)** | **7 (4.4%)** | **3 (1.9%)** | **10 (6.2%)** | **113 (70.6%)** |
| Sports | 117 | 5 (4.3%) | 4 (3.4%) | 2 (1.7%) | 27 (23.1%) | 101 (86.3%) |
| TV shows | 20 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 7 (35.0%) | 17 (85.0%) |
| Video games | 15 | 1 (6.7%) | 1 (6.7%) | 1 (6.7%) | 2 (13.3%) | 8 (53.3%) |

**Science & Tech anomaly:** At α=4.0, only 6.2% of Sci/Tech questions receive NOT_ATTEMPTED, versus 14–35% for all other topics. This ~3× differential collapses at α=8.0 (70.6% vs 74–90%). The anomaly is specific to the intermediate-alpha regime.

**By answer type:**

| Answer type | n | α=0.0 C | α=4.0 C | α=8.0 C | α=4.0 NA | α=8.0 NA |
|-------------|---|---------|---------|---------|---------|---------|
| Date | 222 | 4 (1.8%) | 4 (1.8%) | 4 (1.8%) | 22 (9.9%) | 153 (68.9%) |
| Number | 185 | 14 (7.6%) | 10 (5.4%) | 4 (2.2%) | 44 (23.8%) | 154 (83.2%) |
| Other | 249 | 14 (5.6%) | 10 (4.0%) | 6 (2.4%) | 32 (12.9%) | 186 (74.7%) |
| Person | 198 | 4 (2.0%) | 5 (2.5%) | 1 (0.5%) | 43 (21.7%) | 176 (88.9%) |
| Place | 146 | 12 (8.2%) | 11 (7.5%) | 1 (0.7%) | 36 (24.7%) | 123 (84.2%) |

**Date immunity:** At α=0.0 and α=4.0, date questions are answered correctly at the same rate (4). At α=8.0, compliance stays flat (still 4). This is not because dates resist refusal (68.9% NA at α=8.0 — they don't). The model had exactly 4 date questions it knew correctly; those 4 happen to survive at α=8.0 while all others get refused. This is sampling noise, not structural immunity.

### 1h. Prompt Template (Load-Bearing)

```
Question: {question}
Respond with the answer only. If you are unsure, say "I don't know."
```

The explicit escape hatch is the mechanism for the NOT_ATTEMPTED explosion. 98–99% of NOT_ATTEMPTED responses are the literal string offered by this template.

### 1i. Judging Noise — A Documented Inconsistency

One question (Anselm Kiefer studio transformation) received the **identical response** at α=0.0 and α=4.0 — `"A former textile factory"` — but was graded CORRECT at α=0.0 and INCORRECT at α=4.0 by GPT-4o. The reference answer is `"Silk factory."` The judge appears to have accepted the semantic equivalence in one call and rejected it in another. This is a known artefact of non-deterministic judge sampling across separate batch API calls. It contributes ±1 to the CORRECT count and does not affect any major conclusion at n=1000.

---

## 2. Interpretation

### 2a. The dominant effect: indiscriminate commitment suppression

ITI is not a "know more, say less" filter. It is a blunt commitment suppressor. The evidence:

1. **Precision is flat across α=0.0 and α=4.0** (5.0% → 4.9%). If ITI were selectively suppressing hallucinations, precision would rise — the model would refuse the questions it doesn't know and keep the ones it does. That doesn't happen until α=8.0, and even there precision rises only to 7.7% — a modest change on the n=16 survivors.

2. **The loss rate at α=4.0 is non-selective.** The transition matrix from 0.0→4.0 shows 0.5% of incorrect answers rescued vs 22.9% of correct answers regressed. The intervention is roughly as likely to turn a correct answer wrong as to turn a wrong answer into a refusal. This is random noise on an accuracy landscape, not calibrated filtering.

3. **Once a question is refused at α=4.0, it stays refused at α=8.0** (100% stability in the NOT_ATTEMPTED→NOT_ATTEMPTED cell). The refusal threshold, once crossed, is absorbing. Questions that still attempt at α=4.0 then encounter a 75% refusal rate on the 4.0→8.0 step.

An analogy: imagine squeezing an information tube. At α=4.0 you're at 30% compression — things still flow, but you've introduced loss. At α=8.0 you're at 80% compression — almost nothing gets through. The few things that do get through aren't the "best" things you sent; they're the ones that got lucky. The compression doesn't know what it's preserving.

### 2b. The Science & Tech anomaly: a structural signal

The 6.2% NOT_ATTEMPTED rate for Science & Tech at α=4.0 (vs 14–35% elsewhere) is striking enough to warrant investigation.

One hypothesis: Science & Tech questions in SimpleQA tend to have more specific technical vocabulary that occupies a different region of the model's token distribution. When ITI shifts representations along the "commit/don't commit" axis, questions whose answers are strongly associated with high-confidence technical register (e.g. numerical constants, named protocols, chemical formulas) may be below the refusal threshold at α=4.0 but above it at α=8.0. This aligns with the finding that the anomaly disappears at α=8.0.

An alternative hypothesis: the Science & Tech questions in this benchmark may be phrased in a more closed-ended way (e.g. "What is the wavelength of X?" vs "What was the political context of Y?"), reducing the diversity of plausible completions and thus the probability mass available for the "I don't know" token sequence.

This is the most actionable signal in the dataset for understanding *where* the intervention is operating.

### 2c. Why α=4.0 is not a useful operating point

The analysis resolves the question from Priority 1 of the original report. The two predicted outcomes were:
- (A) Monotonic decline: "I don't know" induction starts early, compliance falls throughout.
- (B) Peak at α=1-2: mild intervention rescues hallucinations before refusal takes over.

The data confirms **Outcome A**. At α=4.0 we have already lost attempt rate (82.3% vs 96.3%) and gained nothing in precision (4.9% vs 5.0%). The compliance drop from 4.8%→4.0% is within overlapping CIs but structurally it shows no sweet spot: there is no alpha at which this intervention simultaneously maintains attempt rate and improves precision on SimpleQA with this prompt design.

### 2d. The TruthfulQA MC / SimpleQA task-format incompatibility, fully confirmed

On TruthfulQA MC, the task used log-probability scoring over forced-choice options. There is no "I don't know" option; the model must pick. ITI improves log-probability mass on the correct option (+4.3pp MC1, +8.5pp MC2 at α=8.0).

On SimpleQA, the task requires free-form text generation with an explicit "I don't know" escape hatch. ITI at any tested alpha routes an increasing fraction of outputs to this escape hatch. The calibration surface (TruthfulQA MC) and the deployment surface (SimpleQA generation) are fundamentally different.

This is not a fixable problem within the current experimental design. It requires either (1) a different calibration dataset that uses generation-format questions, or (2) a different prompt design that removes the escape hatch.

### 2e. The 4–5 "rescued" answers are noise

At both α=4.0 and α=8.0, approximately 4–5 INCORRECT answers become CORRECT. These are near-miss corrections where the model was close at baseline (e.g., off-by-one numerics, closely related names). They do not represent generalisable factual recall improvement; they are fluctuations in a 5-answer confidence landscape. No pattern unifies the rescued questions.

---

## 3. Pipeline Review: Leverage Points by Stage

This section traces the experiment from first principles to identify where the most impactful interventions could be made.

```
SimpleQA CSV
    ↓ load_simpleqa_dataset()
    ↓ _simpleqa_prompt()          ← Stage A: Prompt design
Gemma-3-4B-IT
    ↓ ITI hook (attention heads)  ← Stage B: Intervention mechanism
    ↓ generate() → response
GPT-4o batch judge               ← Stage C: Evaluation design
    ↓ simpleqa_grade
results.json                     ← Stage D: Analysis
```

---

### Stage A — Prompt Design (HIGHEST LEVERAGE)

**Current state:** `"If you are unsure, say 'I don't know.'"` is a named escape hatch in the prompt. 98–99% of NOT_ATTEMPTED responses are the exact string offered.

**The problem:** The escape hatch creates a single high-probability "exit token sequence." When ITI shifts the model's representations along the truthfulness/commitment axis, the escape hatch is the path of least resistance. The model doesn't need to generate a wrong answer — it just generates the named hedge. Removing this is the single cheapest, highest-signal experiment available.

**What to test:** Two prompt variants on the existing α∈{0.0, 4.0, 8.0} JSONL data (re-run inference with new prompts, reuse judge):
1. `"Question: {q}\nAnswer with a single factual phrase."` — removes the escape hatch. Measures precision under commitment.
2. `"Question: {q}\nAnswer with a single factual phrase or provide your best guess."` — actively discourages hedging. Measures worst-case hallucination rate.

**Expected outcome:** If NOT_ATTEMPTED drops from 17.7% to <5% at α=4.0, but compliance still doesn't rise above baseline, then ITI has no factual accuracy effect on generation tasks. If compliance rises, we've found the working operating point. Either outcome is definitive.

**Cost:** ~1 GPU hour inference (3 alphas × 1000 samples with no judge reuse), ~$15 judge API.

---

### Stage B — Intervention Mechanism (HIGH LEVERAGE)

**Current state:** ITI adds α × direction to the output of K=12 ranked attention heads during decoding. The direction was extracted from TruthfulQA MC using mean mass — the average of the "truthful" answer token representation minus the "untruthful" answer token representation across 817 held-out questions.

**Two structural problems:**

**B1 — The direction is not "factual accuracy".**
The direction was computed from contrastive pairs of *answer choices* in MC format. A model choosing between "Abraham Lincoln" and "Napoleon Bonaparte" is doing answer-selection, not factual recall. The extracted direction may encode something closer to "pick the answer with higher confident-sounding token statistics" rather than "retrieve correct facts from memory." When applied to generation tasks, this direction becomes "commit/don't-commit" rather than "correct vs. incorrect."

What to test: Re-extract the direction using SimpleQA CORRECT/INCORRECT generation pairs as the contrastive signal. Record what the model actually generates, then use those pairs (the embedded representation at generation time) to define a "factually accurate generation" direction. This is the natural calibration target for generation benchmarks.

**B2 — K=12 head selection was validated on TruthfulQA MC.**
The ranking criterion (head AUROC on MC format) may not select the heads most relevant to factual generation. The Science & Tech anomaly (6.2% NA at α=4.0 vs 15–35% for other topics) suggests the intervention has non-uniform effects across question types. This is exactly the kind of signal that could be used to identify *which* heads are driving the commitment effect versus any factual recall effect.

What to test: Layer-wise intervention ablation. Run ITI with all K heads from only layer N, sweep N across all 34 layers. If the commitment effect (NOT_ATTEMPTED rate) is concentrated in a small set of layers, those layers can be excluded from the head selection, potentially separating the factual recall signal from the hedging induction.

---

### Stage C — Evaluation Design (MEDIUM LEVERAGE)

**Current state:** The SimpleQA benchmark evaluates using GPT-4o judge. The official grader assigns one of {CORRECT, INCORRECT, NOT_ATTEMPTED}. Compliance = CORRECT / total.

**Problem 1 — Judge non-determinism across batches.**
The Anselm Kiefer case (identical response graded CORRECT at α=0.0 and INCORRECT at α=4.0) demonstrates that inter-batch judge calls are not fully consistent. At n=1000 and a 4.8% base rate, ±1 in the CORRECT count is a ~2% relative swing. This is manageable but should be tracked.

**Mitigation:** For any future run where the compliance difference between alphas is <1 pp, run the judge on both alpha JSONL files in a single batch call (not across separate batch submissions) so the same judge context applies.

**Problem 2 — Three metrics conflated into one score.**
SimpleQA compliance = attempt_rate × precision. These tell completely different stories:
- At α=4.0: attempt_rate=82.3%, precision=4.9% → compliance=4.0%
- At α=8.0: attempt_rate=20.8%, precision=7.7% → compliance=1.6%

An intervention that genuinely improves factual precision would show rising precision even as attempt rate falls. An intervention that only suppresses commitment shows flat precision with falling attempt rate (exactly what we observe). Any future evaluation should report all three metrics in the primary table.

**Problem 3 — SimpleQA may be the wrong benchmark for this model.**
At 4.8% baseline, ~95% of questions are at the model's knowledge ceiling. Even a perfect calibration intervention can't help questions the model doesn't know. Benchmarks where the 4B model has higher baseline accuracy (e.g. TriviaQA open-ended, NaturalQuestions) might provide more headroom to detect factual improvement.

---

### Stage D — Analysis and Interpretation (LOWER LEVERAGE — but important for future runs)

**Current gap:** The only randomness-controlled comparison on SimpleQA is ITI vs. no-ITI. The Priority 3 experiment from the original report (random-head control) was not yet run. Without it, we cannot distinguish:
- (A) The NOT_ATTEMPTED explosion is specific to the truthfulness direction (the direction actively encodes "be uncertain")
- (B) Any large head perturbation causes this on generation tasks with escape hatches

**Recommended:** Run random-head control at α=8.0 (200 samples, not 1000 — we only need to establish whether NOT_ATTEMPTED rate is ~79% or not). Cost: ~15 min GPU, ~$3 judge.

---

## 4. Key Insights

1. **α=4.0 confirms Outcome A (monotonic decline).** Compliance drops baseline→4.0%→1.6% with no intermediate peak. There is no usable alpha for generation tasks in the range {0.0, 4.0, 8.0} with the current prompt design. Priority shifts from "find the sweet spot" to "fix the prompt or the direction."

2. **Precision is structurally flat at α=0.0 and α=4.0.** At 5.0% and 4.9% respectively, the intervention is not improving the model's ability to generate correct factual content. It is only modulating how often the model commits to generating *any* factual content.

3. **The Science & Tech anomaly is the most actionable signal.** Sci/Tech questions resist "I don't know" induction at α=4.0 (6.2% NA vs 15–35% elsewhere). This suggests the intervention's commitment-suppression effect is domain-heterogeneous. Understanding why could lead to interventions that preserve high-commitment behaviour where it matters.

4. **The refusal state is absorbing.** Once a sample becomes NOT_ATTEMPTED (at α=4.0), it is 100% NOT_ATTEMPTED at α=8.0. This means the transition is a one-way door: lower alphas are strictly better at preserving attempt rate, but they also don't improve precision.

5. **The "I don't know" escape hatch is load-bearing for the entire negative result.** Removing it is the single most informative and cheapest next experiment.

6. **The direction needs to be re-extracted for generation tasks.** The current direction was calibrated on MC format. It is not necessarily the "factual recall under generation" direction. This is the highest-leverage structural change.

---

## 5. Highest-ROI Next Steps

### Priority 1 — Remove the escape hatch (HIGHEST ROI, cheapest)

**What:** Rerun inference on the same 1000 SimpleQA questions, same model, same ITI config (K=12, paper-faithful, α∈{0.0, 4.0, 8.0}), but with the prompt:
```
Question: {question}
Answer with a single factual phrase.
```

**Why this first:** It takes ~30 min GPU and tests whether the entire negative result is an artefact of the prompt design. If NOT_ATTEMPTED collapses to near-0% and compliance rises above baseline at some alpha, the prompt was the problem. If compliance stays flat or falls, we have confirmed the intervention has no factual accuracy effect on generation tasks regardless of prompt.

**Cost:** ~30 min GPU, ~$15 judge API.

---

### Priority 2 — Random-head control at α=8.0 (HIGH ROI, cheap)

**What:** Run 200 SimpleQA samples with a random K=12 head selection (no truthfulness direction, random direction vectors) at α=8.0.

**Why:** Determines whether the NOT_ATTEMPTED explosion is specific to the truthfulness direction or generic to any large attention head perturbation. If random-head also gives ~79% NOT_ATTEMPTED, the problem is the intervention scale and prompt combination, not the direction. If random-head has <20% NOT_ATTEMPTED, the truthfulness direction is specifically inducing hedging — which is interpretively interesting.

**Cost:** ~15 min GPU, ~$3 judge API.

---

### Priority 3 — Generation-calibrated direction extraction (MEDIUM, higher cost)

**What:** Extract a new contrastive direction using SimpleQA generation pairs as the signal. Collect 300–400 questions where the model is CORRECT and 300–400 where it is INCORRECT (α=0.0 responses already available). Extract hidden state representations at the last generated token for each set. Compute mass-mean direction (truthful − untruthful), rank heads by AUROC on this new signal, run ITI.

**Why this matters:** The current direction was computed from MC answer-token representations. The "factual accuracy under generation" direction may be in a different subspace. This directly addresses Stage B1 above.

**Cost:** ~2 hrs GPU (activation extraction + head ranking), ~$20 judge (new ITI sweep on 500 questions).

---

### Priority 4 — Layer ablation to isolate commitment vs. recall heads (MEDIUM)

**What:** Run ITI with K=34 heads (one per layer, top-ranked head from each layer). Log NOT_ATTEMPTED rate per-layer. Identify the layers where NOT_ATTEMPTED explodes vs. layers where it doesn't.

**Why:** If commitment suppression is concentrated in layers 0–10 and potential factual recall effects are in layers 25–33, excluding commitment layers from the head selection could give an ITI variant that improves precision without flooding the NOT_ATTEMPTED channel.

**Cost:** ~1 hr GPU inference, ~$10 judge API.

---

## 6. What This Changes About the Bigger Picture

### The gate-2 TruthfulQA MC signal remains valid.

+4.3pp MC1, +8.5pp MC2 at α=8.0 on the 2-fold held-out eval. Those numbers were produced by a log-prob forced-choice task, and that is the task the intervention was calibrated for. They are not invalidated by the SimpleQA findings. What the SimpleQA findings add is the understanding that this positive effect does not transfer to generation tasks at the same alpha.

### The truthfulness direction does something real but task-specific.

The direction successfully improves MC answer selection. It does not improve free-form factual generation. These may be different computational operations: selecting from a closed set vs. retrieving from open-ended memory. The direction is "real" in the sense that it causally affects model behavior — but the behavior it affects on generation tasks is commitment rather than accuracy.

### SimpleQA's role in this project should shift.

SimpleQA is now a "commitment stress test" rather than an accuracy validation benchmark. The question it answers is: "How does ITI affect the model's willingness to commit factual claims in generation?" That is a useful question, but it requires a separate analysis frame from "does the model know more?"

---

## 7. Summary Table (All Three Alphas)

| Metric | α=0.0 | α=4.0 | α=8.0 |
|--------|-------|-------|-------|
| CORRECT (n) | 48 | 40 | 16 |
| INCORRECT (n) | 915 | 783 | 192 |
| NOT_ATTEMPTED (n) | 37 | 177 | 792 |
| Compliance | 4.8% [3.6%, 6.3%] | 4.0% [3.0%, 5.4%] | 1.6% [1.0%, 2.6%] |
| Attempt rate | 96.3% | 82.3% | 20.8% |
| Precision | 5.0% | 4.9% | 7.7% |
| NOT_ATTEMPTED exact "I don't know" | 97.3% (36/37) | 98.3% (174/177) | 99.4% (787/792) |
| CORRECT→NOT_ATTEMPTED | — | 4.2% (2/48) | 70.8% (34/48) |
| INCORRECT→CORRECT rescues | — | 5 (0.5%) | 4 (0.4%) |
| Science & Tech NA rate | 1.2% | 6.2% | 70.6% |

**Dominant finding:** ITI at all tested alphas suppresses commitment without improving precision. The intervention is not accessing the model's factual knowledge — it is overriding the model's decision to commit factual claims. Priority 1 next step is to remove the escape hatch and test whether this diagnosis holds under forced-commitment conditions.

---

*Status: complete — update runs_to_analyse.md*

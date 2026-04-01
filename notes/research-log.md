# Research Log

---

## 2026-03-31

### What I did

**Pipeline hardening.** After Phase 4's validity issues, worked through the ITI pipeline systematically and fixed every identified problem: train/test leakage in direction extraction, wrong head-ranking metric (balanced_accuracy → val_accuracy to match the paper), extraction migrated to last-token-only position, 2-fold CV architecture replacing the informal train/val split, random-direction control mode added to `ITIHeadScaler` to separate "these head *positions* matter" from "these learned *directions* matter." Found and fixed a critical stacked-hooks bug (FIX 6): the previous loop placement left K=n−1 hooks active when constructing the K=n scaler, meaning the Phase 4 K=16 run had K=8 and K=12 hooks still attached — effectively running with 36 simultaneous hooks instead of 16. Regression tests added for: fold-swap symmetry, no test leakage into dev fold, full question coverage, last-token extraction, val_accuracy ranking, and stable question ID matching between extraction and MC manifests.

**Gate 1 calibration sweep — completed.** Ran the 54-point K×α grid (K∈{8,12,16,24,32,40} × α∈{0.0,0.5,1.0,2.0,4.0,6.0,8.0,12.0,16.0}) against the held-out calibration fold (n=81 cal-val questions, seed=42), with the paper-faithful artifact extracted on the clean pipeline. Full MC1 heatmap:

```
         α→   0.0   0.5   1.0   2.0   4.0   6.0   8.0  12.0  16.0
K= 8        22.2  23.5  23.5  25.9  25.9  30.9  30.9  33.3  32.1
K=12        22.2  23.5  25.9  27.2  34.6  35.8  37.0 *38.3* 34.6
K=16        22.2  23.5  24.7  25.9  29.6  34.6  35.8  35.8  35.8
K=24        22.2  23.5  24.7  24.7  24.7  23.5  23.5  23.5  24.7
K=32        22.2  23.5  24.7  23.5  18.5  18.5  18.5  21.0  17.3
K=40        22.2  25.9  27.2  25.9  22.2  22.2  22.2  17.3  14.8
```

Baseline (α=0): MC1=22.22%, MC2=40.79%. Peak: K=12 α=12 at MC1=38.27%, MC2=50.59% — but this is a lone spike with no combos within 1pp. MC1–MC2 Pearson r=0.937 across all 54 combos. K≥24 collapses: at K=24 even the best α barely exceeds baseline; at K=32/40 high-α actively degrades below no-intervention. K=16 plateau: α∈{8,12,16} all yield exactly 35.80% MC1 — the intervention saturates and further α stops flipping answers.

**Config locked: K=12, α=8.0.** Auto-lock favoured K=12 α=12 (38.27%), but manual override to α=8.0 (37.04%): the 1.23pp gap is noise on n=81 (z=0.16, p=0.87), and lower α means less representation perturbation for the downstream generation benchmarks. This was written to `locked_iti_config.json` with the artifact fingerprint `461f3ca7e0a21876`. Pipeline state advanced to gate_1_sweep.locked.

### What I expected vs what happened

The stacked-hooks bug was the real surprise — it invalidates the Phase 4 TruthfulQA MC numbers (+4.3pp MC1, +8.5pp MC2 at K=16) entirely. Those runs had cumulative hooks from K=8 and K=12 still attached at K=16, making the intervention uninterpretable. The sweep itself ran cleanly and confirmed the direction is real: the MC1 surface shows a coherent K×α landscape with a genuine hot zone at K∈{8,12,16} × α∈{6,8,12}. The K=24+ collapse is structurally informative — marginal heads carry noise, not signal, and at high α they actively push representations in harmful directions.

The sweep CIs are wide (±10pp at n=81) — this is expected and is exactly why Gate 2 (n≈327 over 2-fold eval) exists. The surface structure and the K=12 selection are defensible, but no individual number from the calibration fold is reliable.

### What this changes about my thinking

The Phase 4 MC numbers are superseded. The direction is real — the surface is coherent, not random — but the magnitude and precise operating point need the 2-fold held-out eval to be trustworthy. More interestingly, the K=24+ collapse reveals something about the geometry: the truthfulness direction lives in a low-dimensional subspace of the top-ranked heads. Once you include marginal heads (K≥24), you're injecting perturbations in subspaces unrelated to the direction, and high α compounds the damage. ITI is a precision instrument that degrades into a blunt hammer past K≈16.

The clean sweep also makes the calibration→test split protocol feel right: we locked K and α on a held-out fold before touching the test folds. That is the only defensible way to report downstream numbers.

### What I will do next

**Immediate: execute Gate 2.** Run the locked config (K=12, α=8.0, paper-faithful artifact) over the two held-out test folds (n≈327 total) via `scripts/infra/iti_pipeline_evaluate.sh`. Report paired bootstrap CIs. This is the last gate before ITI earns a comparison slot against H-neuron scaling and D4.

**Structural next: build the candidate lattice + verifier/chooser.** The sweep surface exposes a fundamental problem with the global-α regime: the same α that rescues some prompts over-steers others into "I don't know" collapse. The calibration fold can't detect this because it averages over prompts. The right architecture is a per-prompt chooser:

1. **Candidate generation.** For each prompt, run 2–4 decode variants: baseline (α=0.0), low steering (α=4.0), medium steering (α=8.0), and optionally an "answer-only / no hedge" decode variant (forced commitment). This produces a row-level dataset of (prompt, candidate, correctness_label).

2. **Lightweight verifier/chooser.** Train on hidden states from the answer span or first answer token, logprob/entropy features, refusal markers ("I don't know" token probability), and optionally weak features from the H-neuron scaler, D4 direction score, and ITI direction score. Target: correctness labels from SimpleQA/FalseQA/TruthfulQA MC + safety labels where relevant.

3. **Inference policy.** If chooser confidence clears a threshold, select the best candidate. Otherwise abstain or respond cautiously. This is the model deciding, per question, whether steering helped.

Why this is the right move: every intervention this project has run (H-neuron scaling, D4, ITI) shares the same failure mode — global α trades off correctly vs incorrectly for different prompts. A per-prompt chooser directly targets this. It also gives a natural way to beat H-neurons on OOD generation: the chooser can learn when steering helped and when it merely induced "ceremonial surrender." This is repo-grounded LITO.

---

## 2026-03-30

### What I did

**D4 kill-shot.** Built a clean 2,781-record truthfulness contrastive dataset and ran an all-layer residual-stream ablation on FaithEval. First a 20-sample calibration to bracket the cliff (β=[0.005, 0.01, 0.02, 0.05]), then the full 1,000-sample eval at β=[0.00, 0.01, 0.02].

**ITI head-level infrastructure.** Wired up two artifact families — `iti_triviaqa_transfer` (closed-book factuality probe, 2,642 train / 782 val) and `iti_context_grounded` (SQuAD-v2 answerable/impossible, 800 train / 200 val) — and ran 20-sample FaithEval calibration for both at K=16, α=[0.0, 0.1, 0.5, 1.0, 2.0].

**TruthfulQA MC pipeline.** Audited the HuggingFace `multiple_choice` split against the authoritative local CSV: found 1 entirely different question and 209 questions with extra answer variants that HF silently dropped. Migrated the loader to the CSV, regenerated manifests (same 163 held-out questions). Ran three Phase 4 MC evals: MC1 and MC2 with the paper-faithful artifact, MC1 with triviaqa-transfer as a negative control.

### What I expected vs what happened

**D4:** Expected a null or kill. Got a narrow but real lift at β=0.01 (+5.4 pp, 66.0%→71.4%, 0 parse failures). β=0.02 is the cliff: 46.2% compliance, onset of format corruption (mean response length jumps from 1.3 to 15.4 chars, `D`-token massively over-selected). The paired transitions confirm the effect is real at β=0.01: 91 flips False→True vs 37 True→False. But the window is narrow — there is exactly one usable operating point before collapse.

**ITI FaithEval calibration:** Expected some gradient. Got completely flat outputs for both artifact families — identical 17/20 compliance (85%) at every α from 0.0 to 2.0, zero parse failures. Same responses per sample across all alphas. The hooks are running but nothing is moving on FaithEval.

**Context-grounded artifact:** Top heads at AUROC 0.9999, 0.9998 — near-perfect. This is a red flag, not a feature. SQuAD-v2 answerable vs impossible questions are trivially separable by lexical overlap between context and answer span. The probe is not learning truthfulness.

**Phase 4 TruthfulQA MC (now known to be invalid, see March 31 entry):** MC2 with paper-faithful artifact showed +8.5 pp at α=8 (0.434→0.519, monotonic climb). MC1 showed +4.3 pp at α=4 (29.4%→33.7%). Triviaqa-transfer MC1: flat null at 29.4% — extraction source matters. These numbers had dev/test leakage and a stacked-hooks bug; see March 31 for the fix.

### What this changes about my thinking

D4 is alive but behaves the same way as D3: narrow usable window, immediate cliff. It is not robust the way H-neuron scaling is. The ITI FaithEval nulls don't mean ITI doesn't work — they mean FaithEval (generation + regex parse) is not sensitive to head-level steering at this scale, or both artifact families are pointed at the wrong signal. Context-grounded is clearly useless for truthfulness research. The more interesting realization: FaithEval was being run with an `anti_compliance` prompt that tells the model to ignore the misleading context, which redefines the task relative to standard FaithEval. D2 and D3 compliance numbers on FaithEval need to be treated as MCQ surface diagnostics, not headline truthfulness claims, until rerun on a corrected harness. TruthfulQA MC is a cleaner evaluation surface — the log-probability scoring is much more direct than generation + regex — but the Phase 4 numbers need the pipeline fixes before they're trustworthy.

### What I will do next

Fix the ITI extraction pipeline (leakage, ranking metric, position, hooks) before drawing any conclusions from TruthfulQA MC. Drop context-grounded ITI. Fix the FaithEval harness config for D2/D3 reruns.

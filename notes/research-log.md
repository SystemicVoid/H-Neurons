# Research Log

---

## 2026-04-04 (session 2)

### What I did

Ran TriviaQA Bridge Phase 2 dev validation: 3 conditions (baseline α=1.0, E0 ITI α=4.0, E0 ITI α=8.0) on the 100-question dev manifest. Full pipeline: GPU generation → GPT-4o batch judge (bidirectional audit) → paired analysis. Deep sample-level audit of all flips, failure modes, and response perturbation patterns. Report: [2026-04-04-bridge-phase2-dev-results.md](./act3-reports/2026-04-04-bridge-phase2-dev-results.md).

### What happened

E0 ITI does not improve factual generation on TriviaQA. α=4 is flat (Δ adj = -1pp, CI [-6%, +4%]), α=8 is borderline harmful (Δ adj = -7pp, CI [-14%, 0%], McNemar p=0.096). The flip asymmetry at α=8 is 10:3 (right-to-wrong : wrong-to-right).

The qualitative flip analysis revealed the mechanism. The dominant failure mode at α=8 is **confident wrong substitution** (5/10 right-to-wrong flips) — the model replaces a correct entity with a plausible-but-wrong alternative (Terry Hall → Horace Panter, Two-up → Nimble Nick, Head size → Brain size). This is not refusal or hedging — it's the truthfulness direction reshuffling the model's factual distribution. Secondary modes: verbosity (3/10, answer lost in elaboration) and evasion (1/10, description instead of answer).

The intervention is unambiguously active: 51% of correct responses change surface form, 73% of wrong responses change, mean response length grows 70% (15→26 chars), exact match tier drops 28→14. But this activity is indiscriminate — it perturbs all responses, not selectively wrong ones.

The damage is concentrated in `qb` (quiz bowl: 45%→18%) and `bt` (brain teasers: 20%→0%) — the "tricky" question types where the model's knowledge is least robust. Straightforward factual categories (bb, sfq, wh) are unaffected.

### What surprised us

1. **The substitution pattern is the sharpest signal.** We expected ITI failure on TriviaQA to look like the SimpleQA escape-hatch interaction (refusal/evasion). Instead, the model confidently produces *different wrong facts*. "Horace Panter" is the Specials' bassist, not vocalist — the truthfulness direction is activating semantically adjacent but factually wrong completions.

2. **Response drift in correct answers is massive.** 19/37 both-correct pairs produce different text at α=8 (e.g., "Nathuram Godse" → "Nathuram Godse assassinated Mahatma Gandhi."). The intervention is not selectively targeting uncertainty — it broadly reshapes generation style toward more verbose, sentence-form outputs.

3. **The always-wrong population (49 questions) is also perturbed.** 36/49 change their wrong answer under ITI, sometimes shifting from confident-wrong to denial ("CD is not a Roman numeral" replacing "XCIX" for the question "what number is CD?"). The truthfulness direction is correctly identifying uncertainty on these questions but expressing it as denial rather than correction.

4. **Grader is rock-solid.** 0/90 false positives across all match audits. The bidirectional audit design works exactly as planned.

### What was learned

The TriviaQA bridge generation result completes the E0 transfer picture: ITI trained on TruthfulQA improves TruthfulQA answer selection (+6.3pp MC1) but does not transfer to TriviaQA — not for answer selection (E2: null) and not for generation (Phase 2: null-to-harmful). The truthfulness direction is dataset-specific, not a universal "make the model more accurate" lever. It shifts probability mass among the model's existing candidates, which helps when the wrong candidate is a TruthfulQA-style misconception but hurts when the model simply lacks the knowledge.

The bridge benchmark itself is validated: productive headroom (47%), clean grading (0% FP), and the paired flip analysis reveals mechanism that aggregate metrics would hide.

### What I will do next

Per the bridge plan decision tree: this is an "informative null." Still run Phase 3 (test set) for the record with the baseline to establish the published headroom number, but E0 ITI is not the intervention to test there. Shift to D5/D7 per sprint priority.

---

## 2026-04-04

### What I did

Conducted a full scientific review of the E2 TriviaQA transfer investigation (E2-A + E2-B) and wrote the closure synthesis: [2026-04-04-e2-triviaqa-transfer-synthesis.md](./act3-reports/2026-04-04-e2-triviaqa-transfer-synthesis.md). Cross-validated all numbers from the auto-generated E2-B diagnostic against the JSON data artifacts. Verified classification logic against actual inputs. Updated E2-A and E2-B data reports with cross-references. Added the synthesis link to the sprint D4 status.

### What happened

The review confirmed the `wrong_source_still_likely` classification and found no errors in the pipeline output. The key analytical contributions beyond the auto-generated reports:

1. **Calibration overfitting is the sharpest signal.** Both E2 variants show calibration-to-held-out collapse (E2-B: +6.2 pp → +0.0 pp; E2-A: +3.7 pp → -0.9 pp), while E0 replicates (cal MC1 0.370, held-out +6.3 pp). This is the clearest evidence that the TriviaQA signal is absent, not just weak.

2. **Selector choice is second-order to data source.** The E2-A vs E2-B comparison changed K from 40→8, ranking metric, and position policy, but moved held-out MC1 by at most 0.92 pp (not significant). The E0 vs E2-B comparison (same pipeline, different data) shows a 6.26 pp gap. Data source dominates.

3. **The MC2 near-miss (+1.46 pp, CI [-0.11, +3.02]) is not evidence of signal.** It would not survive multiple comparison correction across ~13 pairwise comparisons. MC1 showing exactly zero confirms no behaviorally meaningful change.

4. **The E2-B vs E2-A head overlap is paradoxically informative.** Jaccard 0.043 but Spearman ρ 0.914 means the two selector policies agree on which heads are good (nearly identical ranking) but disagree on how many to include (K=8 vs K=40). The directions are identical where heads overlap (cosine 1.000). Selector choice changes K thresholds, not quality ordering.

### What was learned

TriviaQA-sourced mass-mean ITI does not work on Gemma-3-4B-IT because the probes are weaker (AUROC 0.75 vs 0.82) and the directions do not capture TruthfulQA-relevant representations. The "truthfulness direction" is dataset-specific in this model, not universal. This constraints the claims D8 can make about ITI mechanism.

### What I will do next

Shift to D5 (externality audit) or D7 (causal pilot) per the sprint priority order.

---

## 2026-04-03

### What I did

Audited the E2-A (TriviaQA source-isolated) pipeline results: [2026-04-02-e2-triviaqa-source-isolated-audit.md](./act3-reports/2026-04-02-e2-triviaqa-source-isolated-audit.md). Built and ran E2-B diagnostic pipeline (family-default selectors) with two frozen comparator sidecars: [2026-04-03-e2b-triviaqa-familydefault-diagnostic.md](./act3-reports/2026-04-03-e2b-triviaqa-familydefault-diagnostic.md).

### What happened

**E2-A is a clean null.** K=40, α=6.0 with paper-faithful override selectors produces no detectable effect on any surface — all CIs include zero. The calibration signal (+3.70 pp = 3 extra correct on 81 questions) did not replicate. E3 complementarity gate **not met**: E1 is active-but-imperfect, E2-A is inert, mixing them won't help.

One structurally interesting finding: E2 and E0 probes agree on which heads matter (Spearman ρ = 0.543 on 272 heads) but disagree on direction (mean cosine = -0.163 on 4 shared selected heads). Preliminary — n=4 is too small — but suggestive that "truthfulness" is not unitary at the head level in this model.

**E2-B confirms `wrong_source_still_likely`.** Removing selector overrides (AUROC + all_answer_positions instead of val_accuracy + last_answer_token) changes the artifact substantially — only 2/46 selected heads overlap (Jaccard 0.043), locked config shifts from K=40 to K=8 — but does not rescue the intervention. Key results on held-out n=655:

- Within E2-B (K=8, α=6.0 vs α=0.0 ITI baseline): MC1 +0.00 pp [-1.98, +1.98], MC2 +1.46 pp [-0.11, +3.02]
- E2-B vs E2-A (headline diagnostic): MC1 +0.92 pp [-1.37, +3.21] — CI includes zero
- Sidecar A (E2-B artifact @ K=40, α=6.0 vs E2-A): MC1 -0.92 pp [-2.75, +0.92] — artifact change alone is null
- Sidecar B (K=16 vs K=40 on E2-B artifact): MC1 +0.76 pp [-1.22, +2.75] — compact-K is not materially better
- SimpleQA: compliance +0.00 pp [-3.00, +3.00] — dead flat

All four artifact SHA256 hashes identical (3231a345...), confirming source-isolation assumption. March 30 artifact sanity check passed (5/5 top heads match). The calibration signal (+6.2 pp on n=81 at K=8, α=6.0) was stronger than E2-A's, but still did not survive held-out evaluation.

### What was learned

The E2-A null was not a selector problem. Both selector policies produce inert interventions from TriviaQA-sourced mass-mean directions on Gemma-3-4B-IT. The TriviaQA directions occupy a different subspace from TruthfulQA directions (E0 Spearman ρ = 0.527, E1 ρ = 0.366) and that subspace does not engage TruthfulQA-relevant representations. E3 gate remains not met. Three-variant evidence (E0 positive, E1 partial, E2 null under both selector policies) closes the TriviaQA-transfer ITI lane for this model.

### What I will do next

Accept the three-variant evidence and shift to D5 (externality audit) or D7 (causal head selection). No further TriviaQA ITI variants are warranted.

---

## 2026-04-02

### What I did

Three threads: closed Stage 5.2 (decode-scope), built and ran the full E1 pipeline end-to-end, and hardened infrastructure for the E2 pipeline.

**Stage 5.2 (decode-scope).** Ran GPT-4o batch judge on three decode-scope SimpleQA panels (full_decode, first_3_tokens, first_8_tokens — 200 questions × 2 alphas each). Full numbers: [2026-04-02-decode-scope-simpleqa-judge-results.md](./act3-reports/2026-04-02-decode-scope-simpleqa-judge-results.md).

**E1 (TruthfulQA-modernized).** Built the modernized extraction family (chat-template prompts, AUROC ranking, assistant-content-only token positions), wrote deterministic tests with a `StubChatTokenizer`, built the pipeline script, and ran the full chain: extraction → calibration sweep → lock (K=8, α=8.0) → 2-fold held-out eval → production → SimpleQA-200 judge. Full audit: [2026-04-02-e1-truthfulqa-modernized-audit.md](./act3-reports/2026-04-02-e1-truthfulqa-modernized-audit.md).

**Infrastructure.** Overhauled the CI-coverage audit for per-check isolation and manifest-driven extensibility. Enriched calibration sweep and lock config outputs with artifact family, ranking metric inference, and selection diagnostics. Fixed site chart axis clipping (suggestedMin/suggestedMax). Built the E2 source-isolated lock pipeline with pilot-gated promotion, paired sample-ID parity enforcement, and zero-attempt precision CI handling.

### What happened

**Stage 5.2: scope hypothesis falsified as a complete fix.** `full_decode` generates 64/200 NOT_ATTEMPTED at α=8; `first_3_tokens` cuts that to 21/200. But of the 44 questions rescued from NOT_ATTEMPTED, only 2 (5%) were judged CORRECT — with n=44 and 2 events, no strong claim is warranted. Narrowing scope converts meta-hedging into incorrect answers, not correct ones. `first_3_tokens` cleared the §5.2 promotion rule and is now the locked canonical scope.

**E1: real tradeoff, not a solution.** E1 results on the 655-question held-out eval:
- Within-E1 (α=8 vs α=0.0 ITI baseline): MC1 +2.75 pp [+0.46, +5.19], MC2 +4.48 pp [+2.53, +6.49] — real improvement over its own baseline (α=0.0)
- E1 vs paper-faithful (head-to-head at α=8): MC1 -3.51 pp [-5.95, -1.07], MC2 -3.01 pp [-4.90, -1.15] — CIs exclude zero, paper-faithful wins on MC
- E1 vs paper-faithful on SimpleQA: compliance +2.0 pp [+0.5, +4.0], attempt rate +8.0 pp [+4.0, +12.5] — E1 is significantly gentler on generation

The tradeoff is sharp and well-characterized: E1 buys back generation behavior (attempt rate 97.5% vs E0's 89.5%) by sacrificing MC discrimination. It does not improve SimpleQA correctness vs its own ITI baseline (α=0.0) — compliance delta is +0.5 pp with CI crossing zero.

### Issues found

1. **E1 lock metadata mismatch.** Extraction used AUROC ranking but the locked config labels it as `val_accuracy`. The numbers are unaffected — the mismatch is in metadata, not computation — but it required pipeline fixes (provenance enrichment in sweep/lock scripts) to prevent recurrence.

2. **Lock tolerance fake precision.** The 0.5 pp tolerance in the lock rule is below the MC1 resolution on n=81 (1.23 pp per answer). The tie-break branch was structurally incapable of activating. Not harmful for E1 (only one candidate survived), but the tolerance needs widening for future runs.

3. **Tightened several overstatements** in the decode-scope report and log after first draft: "primary bottleneck," "confident confabulation," "genuinely harder questions," and "knowledge capped at 5-6%" all exceeded what a 200-item pilot can support. Fixed same day.

### What this changes about my thinking

**The hedging/accuracy decomposition is now the central frame.** The decode-scope experiment showed that ITI with TruthfulQA directions moves the hedging axis (how confidently the model responds) but not the accuracy axis (whether the factual claim is correct). E1 confirms this is not just a scope artifact: modernized extraction with AUROC ranking, chat-template alignment, and multi-position selection still fails to improve SimpleQA correctness. The directions encode "don't overclaim" behavior, not "recall the right fact" behavior. This is a working frame, not a proven circuit decomposition — no ablation yet isolates which of AUROC, K, or multi-position is responsible.

**E1 usefully separates "steers the model" from "steers it correctly."** Paper-faithful E0 steers strongly but harms generation. E1 steers more gently — it partially undoes the refusal-inducing side of E0 (attempt rate 97.5% vs 89.5%) while retaining a weaker but still positive MC signal. This is a genuine tradeoff point, not a vague "mixed result." But it's the wrong side of the sprint's cross-benchmark consistency objective: we want MC *and* generation to improve, not to trade between them.

### What I will do next

Run E2 (TriviaQA source-isolated) with paper-faithful override selectors to test whether an alternative data source breaks the direction-quality ceiling. The E2 pipeline with pilot gating is built and ready. If E2's TriviaQA directions show no signal, the mass-mean ITI artifact path is likely exhausted on this model under paper-faithful selectors.

---

## 2026-04-01

### What I did

Three results closed. Full numbers and pipeline audit: [2026-04-01-priority-reruns-audit.md](./act3-reports/2026-04-01-priority-reruns-audit.md).
Follow-up D4 specificity control: [2026-04-01-random-head-specificity-audit.md](./act3-reports/2026-04-01-random-head-specificity-audit.md).

**D4 TruthfulQA MC canonical result locked.** The K=12 α=8 paper-faithful artifact, run over the two final held-out folds (655 questions total), is now the definitive D4 MC result: MC1 26.7%→33.0% (+6.3pp, 95% CI [+3.7, +8.9]), MC2 43.1%→50.2% (+7.2pp, 95% CI [+4.1, +10.2]). This supersedes the earlier 163-question K=16 gate-run headline for all citation purposes.

**D1 TruthfulQA MC rerun.** Ran H-neuron scaling on the same 655 held-out questions used for D4 (α=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] where α=1.0 is the neuron-mode identity baseline; both MC1 and MC2, both folds). Pooled best point: MC1 25.8%→26.7% (+0.9pp, 95% CI [-1.7, +3.5]), MC2 42.6%→43.1% (+0.5pp, 95% CI [-2.3, +3.4]). The D1-vs-D4 ranking on TruthfulQA MC is now resolved: D4 wins by a clear margin. Do not run more D1 TruthfulQA reruns.

**SimpleQA forced-commitment rerun** (`--simpleqa_prompt_style factual_phrase`). Added the flag to `run_intervention.py` and `simpleqa_standalone.sh`, ran K=12 α=8 with the explicit "I don't know" escape hatch removed from the prompt. α=8.0 abstention dropped from 79.2% to 33.0% — the escape hatch was real. But compliance still fell below ITI baseline α=0.0 (4.6%→2.8%) and precision stayed flat within uncertainty (4.6%→4.2%, delta CI [-1.9, +1.1]). The generation null survives prompt repair.

### What I expected vs what happened

Expected the escape-hatch removal to unmask a real improvement — that the "I don't know" collapse was a pure prompt artefact. Instead, we got a partial unmask: abstention dropped substantially, but the responses that did appear didn't improve accuracy. The model isn't just saying "I don't know" — at high α it's also producing wrong answers. Removing the escape hatch changes the failure mode from abstention to incorrect generation; neither is compliance.

### What this changes about my thinking

D4 genuinely improves MC answer selection (discriminative) but fails on free-form factual generation (generative). These are different capabilities, and the intervention doesn't transfer between them. The random-head follow-up later showed the failure is not generic matched-K perturbation; it is specific to the ranked configuration. That narrows the next question to decode scope and component choice, not "does any perturbation of this size break generation?"

### What I will do next

**Priority 1:** Decode-scope ablation on the forced-commitment prompt surface — the random-head control is now complete and rules out the generic-perturbation explanation.

**Priority 2 (conditional):** Only after the scope test, decide whether `E1`, `E2`, or a decomposition control like random-direction should come before any generation-calibrated extraction.

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

ITI baseline (α=0.0, no steering): MC1=22.22%, MC2=40.79%. Peak: K=12 α=12 at MC1=38.27%, MC2=50.59% — but this is a lone spike with no combos within 1pp. MC1–MC2 Pearson r=0.937 across all 54 combos. K≥24 collapses: at K=24 even the best α barely exceeds baseline; at K=32/40 high-α actively degrades below no-intervention. K=16 plateau: α∈{8,12,16} all yield exactly 35.80% MC1 — the intervention saturates and further α stops flipping answers.

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

1. **Candidate generation.** For each prompt, run 2–4 decode variants: ITI baseline (α=0.0), low steering (α=4.0), medium steering (α=8.0), and optionally an "answer-only / no hedge" decode variant (forced commitment). This produces a row-level dataset of (prompt, candidate, correctness_label).

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

---

## 2026-03-24 to 2026-03-29

Week 3 retrospective. Data tables and plot file pointers for this period: [week3-log.md](./act3-reports/week3-log.md). Plot registry: [plot-registry.md](./plot-registry.md).

### What I did

**Sprint reframed (2026-03-27).** Act 3 reframed as a *comparative steering sprint*: H-neurons become the reference row rather than the subject, evaluated against stronger direction-level baselines with the same high-resolution measurement stack. Archived stale pre-pivot planning files under `act2-pre-pivot-archive`. Kept deep research report and literature review as reference, not live planning.

**External critique review and methodology decisions (2026-03-28).** Reviewed an AI-generated critique of the sprint plan. Adopted: (1) D3.5 methodology fix — must project neuron weights into residual-stream space via `down_proj` columns before cosine comparison; the previous approach compared a neuron-weight vector to a residual-stream direction without projecting, which is dimensionally incompatible; (2) expand refusal overlap analysis to a PCA subspace, not just a single vector; (3) separate contrastive datasets aggressively (refusal, truthfulness, detection) to avoid "hallucination vector is a refusal vector in disguise"; (4) D4 uses difference-in-means with diverse multi-source data, not single-dataset logistic regression; (5) Baseline B reframed as a *diagnostic comparator* for safety geometry, not a presumed mitigator; (6) FaithEval first for D4 (MDE logic). Rejected: "start head-level first in D4" — the codebase had no head-level intervention hooks at this point (only `down_proj` hooks), and TransformerLens does not support Gemma-3-4B-IT (only Gemma-2 variants). Head-level work is a conditional refinement, not the starting point.

**D0.5 technical decisions (2026-03-28).** Hook block-output residual stream (not `down_proj`) for direction extraction/intervention — this is the standard representation-engineering surface. Per-layer direction extraction with separation diagnostics rather than cross-layer PCA. All-layer default for refusal intervention (following Arditi 2024), with subset as a conditional refinement. β=0.0 as the no-op convention for direction steering, distinct from H-neuron α=1.0 convention. Capability battery = BioASQ + IFEval (541 prompts, programmatic eval) + WikiText-2 NLL perplexity.

**Refusal contrastive dataset frozen (2026-03-28).** Froze around the published `refusal_direction` repo snapshot, seed-42 sampling, 128+128 train / 32+32 val / 100 harmful test. One explicit leakage fix applied: 2 train-harmful prompts whose normalised text still overlapped the harmful test pool were replaced. This is paper-faithful at the level of source pools and split sizes (Appendix A / page 18), not a proven row-for-row recovery of the exact hidden paper sample. Provenance and caveats in `data/contrastive/refusal/metadata.json`.

**Gemma-3 compatibility fixed (2026-03-29).** `extract_direction.py` and `intervene_direction.py` both needed three fixes for Gemma-3: (1) layer count and hidden dim are under `model.config.text_config`, not `model.config` directly; (2) decoder layers are at `model.model.language_model.layers`, not `model.model.layers`; (3) `dtype=` replaces the deprecated `torch_dtype=` argument. These were never-tested code paths.

**D2 completed (2026-03-29).** Difference-in-means on the frozen 128+128 contrastive set. Best layer: 25 (98.4% val accuracy, separation=9,179). Single-layer ablation at layer 25 reduces harmful refusal rate from 25% to 0%. Single-layer addition has zero effect on harmless — expected, since D3 uses all-layer. Clean-rerun from a fresh worktree reproduced the same tensor hash, layer, and val accuracy: residual risk is conceptual, not chain-of-custody. Harmful eval sets materialised (JBB_100, HarmBench_159, StrongREJECT_313) at `data/contrastive/refusal/eval/`.

**D3 calibrated: narrow window, then cliff (2026-03-29).** All-layer refusal-direction ablation on FaithEval (1,000 samples). β=0.02: 70.2% compliance, 702/1000, clean. β=0.03: 51.1% — a 191-sample net loss. The collapse at β=0.03 is primarily answer-option bias (option B jumps to 58.1% of responses) not a parser failure (parse rate = 0% at all β). Decision: do not broaden D3. Full data: [2026-03-29-d3-faitheval-refusal-direction.md](./act3-reports/2026-03-29-d3-faitheval-refusal-direction.md).

**D3.5 gate resolved (2026-03-29).** Projected the 38-neuron residual update into the refusal-direction space. Both canonical cosine gap (−0.0183) and PCA subspace gap (+0.0361) differ from a layer-matched random-neuron null in the expected directions — real overlap. But the signal is dominated by layer 33: removing that one layer collapses or flips all four mediation correlations (FaithEval + Jailbreak × canonical + subspace). Layer 33's subspace gap is 43× larger than the next layer. Gate decision: `proceed_with_d4_unchanged`. Refusal overlap is a live hypothesis, not a settled mechanism. Full audit: `data/gemma3_4b/intervention/refusal_overlap/refusal_overlap_audit.md`.

**Bug fixes (week 3).** (1) Gemma-3 nested config compatibility — unblocked D2/D3 entirely. (2) Micro-beta alpha label aliasing: α=0.005/0.01/0.02 all formatted to `alpha_0.0.jsonl` due to single-decimal `f"{alpha:.1f}"` formatting — fixed before the clean D3 calibration run. (3) Jailbreak site payload drift: site was displaying a stale 7-point axis against the current 4-point data sweep. (4) D3.5 gate hardening: added dominant-layer exclusion test and sign-convention audit after the initial D3.5 result had ambiguous sign conventions.

### What I expected vs what happened

**Jailbreak binary judge falsified by 5,000-token rerun.** An earlier 7-alpha, 256-token run appeared to show a significant jailbreak compliance slope (+6.2pp, slope +2.14 pp/α). A 5,000-token rerun on 2026-03-25 falsified it: the 256-token window was truncating responses mid-disclaimer, creating alpha-dependent false positives (the disclaimer was often in the second half of the response). The binary judge on the 5,000-token run shows a non-significant effect. CSV-v2 graded scoring recovers a real but structurally different result: +7.6pp csv2_yes (CI [+3.6, +11.6]), but the effect is driven by severity escalation (V=3 rate quadruples), not count growth. 76% of the count increase is ablation recovery at α=0→1.

**D3.5 fragility.** Expected a cleaner mediation story. Got a result that is statistically real but dominated by a single layer in a way that makes mechanistic interpretation very uncertain. The D3.5 gate outcome is "proceed unchanged" but it leaves the refusal-overlap hypothesis neither confirmed nor falsified.

### What this changes about my thinking

The D3 β cliff and D3.5 layer-33 fragility are telling the same underlying story: the truthfulness and refusal representations in Gemma-3-4B-IT are not smoothly decomposable. There are sharp thresholds past which the intervention disrupts something orthogonal to the intended target. This should inform how aggressively we push any intervention vector — a wide, robust operating window is a prerequisite for the result being mechanistically clean, not just numerically positive.

The jailbreak 256-token false positive is a methodological lesson: response-length distributions are intervention-dependent, so fixed-length truncation in a judge is not neutral.

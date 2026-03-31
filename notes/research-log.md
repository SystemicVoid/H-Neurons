# Research Log

---

## 2026-03-31

### What I did

Spent the day making the ITI evaluation honest. After the Phase 4 results from yesterday flagged multiple validity issues, worked through them systematically: fixed the train/test leakage in direction extraction, corrected the head ranking metric to match the paper (val_accuracy, not balanced_accuracy), migrated extraction to last-token-only position, switched to a proper 2-fold CV split, and added a random-direction control mode to ITIHeadScaler so the eval can distinguish "these head positions matter" from "these learned directions matter." Also built a K×α calibration sweep (54 combinations: K=[8,12,16,24,32,40] × α=[0.0,0.5,1.0,2.0,4.0,6.0,8.0,12.0,16.0]) that runs against a held-out calibration fold — so hyperparameters are locked before touching the two test folds. Found and fixed a stacked-hooks bug (FIX 6) in the sweep: the previous loop placement left K=n−1 hooks active when constructing the K=n scaler, causing double-applied interventions for every K > 8. Added regression tests covering: no test leakage into dev fold, fold-swap symmetry, full question coverage, last-token extraction, val_accuracy ranking, and stable question ID matching between extraction and MC manifests.

### What I expected vs what happened

Expected the fixes to be plumbing — clean them up, rerun, similar ballpark numbers. The stacked-hooks bug was a surprise: it means the Phase 4 runs at K=16 were effectively running with cumulative hooks from K=8 and K=12 still attached. The calibration sweep has not yet produced output (the GPU job is staged but not run), so there are no new result numbers to report yet.

### What this changes about my thinking

The Phase 4 TruthfulQA MC results from yesterday (MC1: +4.3 pp, MC2: +8.5 pp) cannot be treated as even preliminary evidence — the extraction was leaking test questions into direction fitting, ranking on the wrong metric, and the sweep runner had stacked hooks. The signals may be real, but the numbers are not. The right posture is: all Phase 4 numbers are superseded pending the clean rerun. The calibration sweep approach is the correct protocol: lock K and α on a held-out calibration fold first, then run the 2-fold test without touching those decisions again.

### What I will do next

Run the calibration sweep and 2-fold eval with the locked config. After that, produce the aggregate report (already scripted: `scripts/report_iti_2fold.py`) with paired bootstrap CIs. Only then decide whether ITI earns a real comparison slot against H-neuron scaling and D4.

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

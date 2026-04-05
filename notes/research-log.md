# Research Log — Closure & Pivot Phase

> Continues from [research-log-iti-artifact-exploration.md](./act3-reports/research-log-iti-artifact-exploration.md) (2026-03-24 to 2026-04-02), which covers the ITI artifact exploration phase: pipeline hardening, calibration sweep, E0/E1/E2 artifact variants, decode-scope ablation, and the bridge from D3.5 to D4.

---

## 2026-04-05

### What I did

Built the D7 causal pilot pipeline end-to-end — the sprint's one scoped causal experiment, testing whether gradient-based head selection outperforms the correlational (mass-mean) selector that E0/E1/E2 all used.

The pipeline has five new components: (1) deterministic JBB paired manifest builder (pilot 100 / full 500, disjoint, seed-locked); (2) two new ITI extraction families (`iti_refusal_probe` and `iti_refusal_causal`) sharing a single activation surface but forking at ranking time; (3) causal head ranking via contrastive grad×activation on paired harmful/benign NLL objectives; (4) deterministic jailbreak decoding controls in `run_intervention.py`; (5) a resumable `systemd-inhibit` orchestrator (`d7_causal_pilot.sh`) for staged 100→500 execution. Supporting utilities: alpha-lock with paired sample-ID parity enforcement and tie-break to lower alpha, and paired CSV2 reporting (`csv2_yes`, `C`, `S`, `V`, payload share) with guardrails for missing/errored annotations.

Test coverage: manifest determinism/disjointness/alignment, synthetic causal ranking, refusal-family artifact compatibility, jailbreak decode control defaults/overrides, paired-report parity/locking. Verified with `ruff check`, `pytest`, `ty check`, `shellcheck`, `audit_ci_coverage.py`. Minor fixes: renamed `DummyITIModel.forward` param `x→input_ids` for LSP compliance, added `--api_mode` underscore alias to eval harness.

### What I expected vs what happened

Expected the causal/probe split to require a deep pipeline fork. Instead, the design fell out cleanly: both families share the same extraction tensor and steering artifact contract, diverging only at ranking time. The causal ranker computes ∂log p(benign continuation)/∂z · z and ∂log p(harmful continuation)/∂z · z on the same harmful prompt and ranks by the difference — 2 forward + 2 backward passes per pair. The probe ranker uses the existing AUROC path. Everything downstream (sweep, lock, intervention, reporting) is family-agnostic.

The implementation is execution-ready. Orchestrator stages are idempotent/resumable with failure-visible judge/CSV2 phases (removed silent `|| true` swallowing from earlier drafts).

**Pre-registered framing constraints for D7 results.** Allowed: "non-correlational selector," "gradient-based selector," "proxy for refusal-relevant discrimination under paired objectives," "motivated by GCM's attribution patching but using a simpler single-prompt variant." Not allowed: "we performed attribution patching" (different method — no counterfactual patching), "we identified causal mediators of refusal" (no clean intervention), "we replicated GCM" (different formula, different experimental design).

### What this changes about my thinking

D7 is the sprint's last novel experiment before synthesis. Its outcome shapes D8's central claim in one of two ways. If gradient-based selection outperforms correlational selection on jailbreak severity (CSV2), it validates the critique that mass-mean probes conflate relevant and irrelevant heads — and suggests the ITI generation failures (E0/E1 on bridge, E0 on SimpleQA) might be partially rescued by better selection. If it doesn't outperform, D8 can confidently say the problem is deeper than head selection: the directions themselves are the bottleneck, not the heads carrying them. Either outcome closes the sprint cleanly.

### What I will do next

Execute the D7 pilot run (`scripts/infra/d7_causal_pilot.sh`) and add the resulting run entry to `notes/runs_to_analyse.md` once generation/judging completes.

---

## 2026-04-04

### What I did

The biggest day of Act 3 — four threads that collectively closed the TriviaQA transfer question and built the generation evaluation surface the project was missing.

**Thread 1 — E2 transfer synthesis.** Wrote the formal closure of the TriviaQA-transfer ITI lane: [2026-04-04-e2-triviaqa-transfer-synthesis.md](./act3-reports/2026-04-04-e2-triviaqa-transfer-synthesis.md). Cross-validated every number from the auto-generated E2-B diagnostic against JSON data artifacts, verified the five-level classification logic against actual inputs (`wrong_source_still_likely` holds), and added cross-references between E2-A and E2-B data reports.

**Thread 2 — Bridge benchmark inception (Phase 1).** The E2 closure exposed a gap: all our generation evidence came from SimpleQA, where baseline compliance is 4.6% — too low to distinguish "model lacks the fact" from "steering suppressed the answer." Designed and built the TriviaQA bridge generation benchmark from scratch. Manifest builder with stratified seed-42 sampling (pilot 150 / dev 100 / test 500 / reserve 200, all disjoint). Deterministic + GPT-4o adjudicated two-tier grading pipeline with bidirectional blinded audit protocol. Paired bootstrap analysis. Plan: [2026-04-03-bridge-benchmark-plan.md](./act3-reports/2026-04-03-bridge-benchmark-plan.md).

Ran Phase 1 pilot (α=1.0 neuron-mode baseline, 150 questions) — headroom gate passed at 47% baseline accuracy. Then the grading edge cases forced a calibration sprint: added three high-precision tiers (alias simplification, numeric extraction, guarded reverse containment), fixed a curly-quote/em-dash normalization bug in `normalize_answer`, re-scored the pilot with the calibrated grader (audit agreement 87.5% → 91.9%), and locked the two-metric policy: adjudicated accuracy = primary, deterministic = conservative floor.

Also resolved 32 pre-existing `ty` diagnostics across the test suite and added a `ty` compliance check on commit to prevent regression.

**Thread 3 — Phase 2 dev validation (E0 ITI).** Ran 3 conditions (neuron baseline α=1.0, E0 ITI α=4.0, E0 ITI α=8.0) on the 100-question dev manifest. Full pipeline: GPU generation → GPT-4o batch judge (bidirectional audit) → paired analysis with sample-level flip taxonomy. Report: [2026-04-04-bridge-phase2-dev-results.md](./act3-reports/2026-04-04-bridge-phase2-dev-results.md).

**Thread 4 — E1 extension.** After the E0 results showed damage, hypothesized that E1's gentler perturbation (K=8 instead of K=12, +8pp attempt rate vs E0 on SimpleQA) might avoid the right-to-wrong flip asymmetry. Built `triviaqa_bridge_dev_e1.sh`, ran E1 at α=8.0 on the same dev manifest, and added the comparative analysis to the Phase 2 report (§9).

### What I expected vs what happened

**E2 synthesis confirmed what Apr 3 suggested, but sharpened the mechanism.** The calibration-to-held-out collapse is the sharpest signal: E2-B went from +6.2pp on calibration (n=81) to exactly +0.0pp on held-out (n=655); E2-A from +3.7pp to -0.9pp. E0 survived replication. Selector choice is second-order to data source — E2-A vs E2-B changed K from 40→8, ranking metric, and position policy, but moved held-out MC1 by at most 0.92pp (not significant). E0 vs E2-B (same pipeline, different data) shows a 6.26pp gap. Data source dominates everything.

**Bridge Phase 2 — expected the SimpleQA failure mode (refusal/evasion). Got something mechanistically sharper.** E0 ITI does not improve factual generation on TriviaQA: α=4 is flat (Δ adj = -1pp, CI [-6%, +4%]), α=8 is borderline harmful (Δ adj = -7pp, CI [-14%, 0%], McNemar p=0.096). But the *how* matters more than the *how much*. The dominant failure mode at α=8 is **confident wrong substitution** (5/10 right-to-wrong flips): the model replaces a correct entity with a plausible-but-wrong alternative. Terry Hall → Horace Panter (bassist, not vocalist). Two-up → Nimble Nick. Head size → Brain size. "The Sunday Post" → "The Scotsman" (real Scottish newspaper, wrong one). This isn't refusal or hedging — the truthfulness direction is reshuffling the model's factual distribution, promoting semantically adjacent but factually wrong completions.

The intervention is unambiguously active: 51% of correct responses change surface form, 73% of wrong responses change, mean response length grows 70% (15→26 chars), exact match tier drops from 28 to 14. But this activity is indiscriminate — it perturbs everything, not selectively the wrong answers. Damage concentrates in `qb` (quiz bowl: 45%→18%) and `bt` (brain teasers: 20%→0%), the question types where the model's knowledge is least robust. Straightforward factual categories (bb, sfq, wh) survive.

**E1 extension — expected the gentler artifact to at least reduce damage. It made it worse.** E1 α=8 is formally significant: Δ adj = -9pp, CI [-16%, -3%], McNemar p=0.016. The killer detail: both E0 and E1 produce *exactly* 10 right-to-wrong flips with the same damage profile. The difference is rescue capacity — E1 rescues only 1 wrong-to-right (vs E0's 3). On the 5 confident wrong substitutions (Horace Panter, Brain size, The Scotsman, Miep van Pels), **E1 produces the identical wrong entity as E0**. Fewer heads doesn't mean less damage; it means less perturbation in both directions, which kills rescue capacity while leaving the substitution mechanism untouched.

**Grader — clean.** 0/90 false positives across E0 match audits, 1/30 in E1. Bidirectional audit design validated.

### What this changes about my thinking

Three frame shifts:

1. **"Truthfulness direction" is dataset-specific, not universal.** The E2 synthesis makes this definitive. TriviaQA probes are weaker (AUROC 0.75 vs 0.82), the directions don't capture TruthfulQA-relevant representations, and calibration signals that looked promising on n=81 vanished entirely on held-out. The three-variant ranking (E0 > E1 > E2, each gap significant) confirms data source dominates extraction methodology. This constrains D8's universality claims about ITI mechanism.

2. **ITI's generation failure mode is substitution, not refusal.** This is the day's sharpest insight. We spent a week assuming ITI harms generation by inducing over-cautious hedging (the SimpleQA "I don't know" pattern). The bridge benchmark reveals a different mechanism: the truthfulness direction shifts probability mass among the model's existing completion candidates. When the wrong candidate is a TruthfulQA-style misconception, this helps (MC1 +6.3pp). When the model simply lacks robust knowledge (trivia, quiz bowl), the same shift promotes a related-but-wrong entity. The damage being identical across E0 and E1 proves this is method-level — you can't fix it by choosing a better artifact within the mass-mean ITI framework.

3. **The bridge benchmark fills a real gap.** Productive headroom (47%), clean grading (0% FP), and the paired flip analysis reveals mechanism that aggregate metrics would hide. The cross-alpha stability table (49 hard-wrong, 36 stable-correct, 7 progressive damage, 3 stable rescue) gives the kind of population structure SimpleQA's floor compliance can never provide. This is the generation evaluation surface the sprint was missing.

### What I will do next

TriviaQA transfer is closed on both selection (E2) and generation (bridge Phase 2). Shift to D5 (externality audit) and D7 (causal pilot) per sprint priority. Bridge Phase 3 (test set) is deferred until a candidate intervention actually warrants it — E0/E1 ITI are not those candidates.

---

## 2026-04-03

### What I did

Completed the E2-B diagnostic — the second and final TriviaQA-sourced ITI variant — to determine whether E2-A's null result was a selector problem or a data problem. E2-A used paper-faithful overrides (`val_accuracy` + `last_answer_token`); E2-B uses family-default selectors (`auroc` + `all_answer_positions`). Built the diagnostic pipeline with two frozen comparator sidecars to cleanly decompose the headline result: Sidecar A (E2-B artifact at K=40, α=6.0 vs E2-A) isolates the artifact change with K held fixed; Sidecar B (K=16 vs K=40 on the E2-B artifact) isolates the K change with the artifact held fixed. This avoids conflating multiple simultaneous changes in the headline comparison. Report: [2026-04-03-e2b-triviaqa-familydefault-diagnostic.md](./act3-reports/2026-04-03-e2b-triviaqa-familydefault-diagnostic.md).

Also audited the earlier E2-A results for completeness: [2026-04-02-e2-triviaqa-source-isolated-audit.md](./act3-reports/2026-04-02-e2-triviaqa-source-isolated-audit.md).

### What I expected vs what happened

Expected one of two outcomes: either E2-B's AUROC + multi-position selectors would rescue a weak signal that E2-A's paper-faithful overrides missed (selector-mismatch diagnosis), or it would confirm the null and we could confidently blame the data source (wrong-source diagnosis). Got the latter, decisively.

**E2-A is a clean null.** K=40, α=6.0 produces no detectable effect on any surface — all CIs include zero. The calibration signal (+3.70 pp = 3 extra correct on 81 questions) did not survive held-out evaluation. E3 complementarity gate not met: E1 is active-but-imperfect, E2-A is inert, mixing them won't help.

**E2-B changes the artifact but not the outcome.** Switching selectors changes the selected set dramatically — only 2/46 heads overlap (Jaccard 0.043), locked config shifts from K=40 to K=8. But on held-out n=655:

- Within E2-B (K=8, α=6.0 vs α=0.0): MC1 +0.00 pp [-1.98, +1.98]
- Sidecar A (artifact change, K fixed): MC1 -0.92 pp [-2.75, +0.92] — null
- Sidecar B (K change, artifact fixed): MC1 +0.76 pp [-1.22, +2.75] — null
- SimpleQA: compliance +0.00 pp [-3.00, +3.00] — dead flat

All four E2-B artifact SHA256 hashes are identical (3231a345...), confirming source-isolation. The calibration signal (+6.2 pp on n=81 at K=8) was *stronger* than E2-A's, but still collapsed entirely on held-out data — the clearest evidence that weak calibration signals on n=81 are noise when the underlying effect is absent.

**The head overlap structure is the most interesting finding.** E2-B vs E2-A: Jaccard 0.043 but Spearman ρ 0.914, with direction cosine 1.000 on the 2 shared heads. The two selector policies *agree on which heads are good* (nearly identical full ranking) but *disagree on where to draw the K threshold* (AUROC concentrates more sharply → K=8; val_accuracy has narrower dynamic range → K=40). Selector choice changes K, not direction quality.

Between datasets: E2-B vs E0 shows Spearman ρ 0.527 on all 272 heads (moderate agreement on head quality ordering) but negative direction cosine (-0.351) on the single shared selected head (L11H0). Same computational locations, different — possibly opposing — directions. This is suggestive of TriviaQA and TruthfulQA encoding genuinely different concepts at the head level, though n=1 shared head can't carry that weight alone.

### What this changes about my thinking

The E2-A null was not a selector problem. Both selector policies produce inert interventions from TriviaQA-sourced mass-mean directions on Gemma-3-4B-IT. The three-variant evidence is now complete: E0 (TruthfulQA, paper-faithful) produces a real +6.3pp MC1 effect; E1 (TruthfulQA, modernized) produces a weaker +2.75pp effect; E2 (TriviaQA, both selector policies) produces exactly zero. Data source explains the entire variance.

The "same heads, different directions" pattern — if it holds up with more overlap points — would mean that "truthfulness" is not a single direction in this model's representation space. TriviaQA's "factual recall" and TruthfulQA's "misconception resistance" live in the same computational locations but point in different (possibly opposing) directions. This directly constrains the universality claims in the ITI literature (Li et al. 2023) and narrows what D8 can say about the mechanism.

### What I will do next

Write the E2 transfer synthesis to formalize the closure, then pivot. The TriviaQA-transfer lane consumed four experiment-days (E2-A pipeline + audit, E2-B diagnostic + sidecars) and produced a clean, well-decomposed null. No further TriviaQA ITI variants are warranted. Next priorities: D5 (externality audit) and D7 (causal head selection).

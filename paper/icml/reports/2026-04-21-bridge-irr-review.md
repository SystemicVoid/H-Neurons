# Bridge IRR Pipeline Review — 2026-04-21

> **Verdict (data):** The Bridge IRR workflow is complete. On all 57 discordant
> TriviaQA-bridge test cases, Rater A (human first author) and Rater B (LLM
> judge, `gpt-4o-2024-11-20`) agreed on 55 labels; Cohen's κ = 0.90,
> Gwet's AC1 = 0.96, raw agreement 96.5% [88.1, 99.0] (Wilson). The two
> disagreements were resolved under a pre-frozen adjudication rule with no
> rule-gaps. The adjudicated right-to-wrong category shares are 72.1% [57.3,
> 83.3] wrong-entity substitution, 20.9% evasion/denial, 7.0% dilution,
> 0.0% formal refusal.
>
> **Verdict (interpretation):** The Limitation 4 ask is closed under the
> weaker-of-two framings L4 named explicitly ("LLM judge as second rater …
> acceptable as a sensitivity check, but weaker as IRR evidence"). The rubric
> + prompt combination generalizes: dev calibration agreement was 100%
> (n=13, κ=1.0) after one permitted prompt revision, and test-split
> disagreement did not expose any uncovered rule pattern. The paper-level
> claim ("wrong-entity substitution is the dominant R2W failure mode") is
> now qualitatively and quantitatively defensible; the exact 72.1% value is
> interval-qualified. A real human second rater remains the only open
> strength-form upgrade path.

## Source Hierarchy

This report is the authoritative prose analysis for the Bridge IRR work.
Numerical source-of-truth is `bridge_irr_summary.json`; this report interprets
it.

- Machine-readable summary: `data/judge_validation/bridge_irr/bridge_irr_summary.json`
- Adjudicated per-case labels: `data/judge_validation/bridge_irr/adjudicated_labels.jsonl`
- Disagreements log: `data/judge_validation/bridge_irr/bridge_irr_disagreements.jsonl`
- Frozen adjudication rule: `data/judge_validation/bridge_irr/adjudication_rule.md` (git `0e965d5`, sha256 `4dcbae24…`)
- Rater A progress (57 rows): `data/judge_validation/bridge_irr/rater_a_progress.jsonl`
- Rater B progress (57 rows): `data/judge_validation/bridge_irr/rater_b_progress.jsonl`
- Rater B provenance: `data/judge_validation/bridge_irr/rater_b_provenance.json`
- Pipeline status: `data/judge_validation/bridge_irr/bridge_irr_status.json`
- Precursor Phase 3 report (superseded for taxonomy numbers): [`../../notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`](../../../notes/act3-reports/2026-04-13-bridge-phase3-test-results.md) §5
- Original task spec: [`../reviews/TODO_L4_interrater.md`](../reviews/TODO_L4_interrater.md)
- Broader limitation strategy: [`../reviews/TODO_Limitations_Fixes.md`](../reviews/TODO_Limitations_Fixes.md) §L4
- Paper-facing supplement: [`../supplement/failure_coding_manifest.md`](../supplement/failure_coding_manifest.md)
- Reviewer-facing summary: [`../supplement/support/externality_summary.md`](../supplement/support/externality_summary.md)

---

## 1. Pipeline Review — What Was Built and How It Ran

### 1.1 Scope and unit of analysis

- **Surface:** TriviaQA bridge held-out test set (`data/manifests/triviaqa_bridge_test500_seed42.json`, n=500).
- **Comparison:** Baseline α=1.0 vs E0 ITI α=8.0 (paper-faithful, K=12, `first_3_tokens` decode scope).
- **Unit:** Each question whose compliance flipped between baseline and ITI. Extracted by `scripts/bridge_irr.py::extract_discordant_cases`; n=57 (43 R→W + 14 W→R). This scope matches the expanded L4 plan ("all discordant bridge cases, not only the 43 right-to-wrong flips") and differs from the precursor Phase-3 report which taxonomized only the 43 R→W flips.
- **Coding target:** Only the `incorrect_response`. Raters see `question`, `gold_aliases`, `incorrect_response`, and `paired_correct_response`; they do not see which condition produced which response, so neither rater can infer the direction (baseline vs. ITI) from the stimulus.

### 1.2 Frozen rule — tamper-evident

- **Rubric:** `bridge_incorrect_response_v1`; four categories: `wrong_entity_substitution`, `evasion_or_factual_denial`, `answer_dilution`, `formal_refusal`.
- **Rules:** Six decision rules (R1 refusal → R2 commitment → R3 multi-candidate → R4 denial → R5 commitment-with-alternatives → R6 tangential padding) + three pairwise tiebreakers (R2/R3, R3/R4, R2/R4), committed to `adjudication_rule.md`.
- **Freezing:** The adjudication rule's `sha256` of file contents (`4dcbae24…`) and `git_commit` (`0e965d5`) are embedded in `bridge_irr_status.json` and `bridge_irr_summary.json`. The freeze is verified at finalize time — the summary carries the rule hash, not a path reference.
- **Calibration clause (§9):** the rubric and Rater B's system prompt may be revised once before any test-split label is appended. All such revisions must be committed before labeling starts; the frozen-commit hash records the state at first test label.

### 1.3 Blinded queue

- **Builder:** `scripts/prepare_bridge_irr_queue.py` → `test_queue_blinded.jsonl` (stimulus only) + `test_queue_key.jsonl` (direction + grades, never seen by raters).
- **Case IDs:** Deterministic SHA256(split:question_id)[:12]; stable across re-seeding. Queue order is shuffled by a fixed seed (42) so presentation order reveals nothing about direction.
- **Staleness guard:** `ensure_progress_files_compatible` refuses to reuse a Rater A / Rater B / adjudication ledger if any row references a `case_id` that is no longer in the regenerated queue. This is a hard defense against silently mapping old labels to different questions.

### 1.4 Rater A — human (first author)

- **Tool:** `scripts/bridge_irr_label.py --split test` — interactive CLI, resume-safe, writes one JSONL row per case, reopen-per-record per `scripts/CLAUDE.md`.
- **Schema:** `{case_id, label ∈ LABELS, confidence ∈ {low, medium, high}, notes, rater: "rater_a_human", rubric_version}`; validated on append.
- **Independence:** Rater A records notes citing rule(s) applied but sees neither Rater B's labels nor the transition direction (`right_to_wrong` vs `wrong_to_right`).

### 1.5 Rater B — LLM judge (sensitivity check)

- **Model:** `gpt-4o-2024-11-20`, pinned snapshot. Temperature=0. `json_schema` strict mode (label/confidence/notes schema enforced by the API, `additionalProperties: false`).
- **Batch API only** (per `scripts/CLAUDE.md`): crash-safe `.rater_b_batch_state.json` resume, 50% cheaper than sync.
- **Zero-shot on the rubric.** No few-shot exemplars from Rater A's decisions. This keeps Rater B independent — the two raters share only the published rule, not the specific labeled cases.
- **Provenance:** `rater_b_provenance.json` records `model`, `prompt_hash` (SHA256 prefix of the system prompt: `a135761b05bf1533`), `git_commit` (`ff332b4…`), timestamp, decoding config, batch-mode flag, n_cases, n_newly_labeled, n_failures.
- **Batch health:** Test batch `batch_69e754f8905c8190b6b7d9f3d757bab6` completed in 22 min 19 s, 57/57 succeeded, 0 failed (from `logs/bridge_irr_rater_b_test_20260421_114406.log`).

### 1.6 Dev calibration — §9 protocol followed, but not documented as a standalone report

- Dev queue has 13 discordant cases from the Phase-2 dev split (seed=42, same builder).
- Rater B was run twice on dev: first with prompt_hash `ab951cf3c3ee0bb7` (baseline), then with `a135761b05bf1533` (revised). The revised prompt matches the frozen test-run prompt_hash.
- Dev agreement under the frozen prompt is 13/13 = 100% (κ=1.0, AC1=1.0). Rater A and Rater B each labeled 11 wrong_entity / 1 evasion / 1 dilution / 0 refusal. This confirms the post-revision rubric+prompt pair produces no residual disagreement on the calibration set.
- **Gap I am flagging:** the single permitted §9 revision is observable in the repo (two dev-run logs, two prompt hashes, revised rule committed at `0e965d5`) but there is no standalone report in `notes/act3-reports/` documenting the revision and before/after dev agreement. The in-repo artifacts are sufficient as provenance, but a `notes/act3-reports/2026-04-21-bridge-irr-dev-calibration.md` would be the right long-term home for this narrative.

### 1.7 Adjudication

- **Protocol (§5 of rule):** adjudicator (first author) sees question, aliases, incorrect_response, paired_correct_response, and both raters' labels+notes. Applies first matching rule from §1; §2 tiebreakers resolve boundary cases. Writes `{case_id, label, notes, rule_gap}` to `adjudication_progress.jsonl`. Notes must cite the rule and the phrase that triggered it.
- **Outcome (2 cases):**
  - `bridge_test_case_8b1f3dfc9496` — A=`answer_dilution`, B=`wrong_entity_substitution` → adjudicated `wrong_entity_substitution` (R2 applies; "A mixed-breed dog." commits the answer slot; matches §4 "Leather finger guards" generic-descriptor anchor). `rule_gap=false`.
  - `bridge_test_case_a9887a3c58fe` — A=`wrong_entity_substitution`, B=`answer_dilution` → adjudicated `answer_dilution` (R6 applies; "The name's pronunciation was closer to 'Gees-wick'" describes pronunciation, not a town; operational R2 test fails). `rule_gap=false`.
- **Rule-gap rate:** 0/2 (Wilson CI is uninformative at n=2: [0, 65.8%]). The sample is too small to bound uncovered-pattern risk meaningfully; the honest statement is that neither disagreement required stretching the rule.

### 1.8 Finalization

- `scripts/finalize_bridge_irr.py` assembles `bridge_irr_summary.json`, `adjudicated_labels.jsonl`, and `bridge_irr_disagreements.jsonl`.
- Integrity checks at finalize time: every queued `case_id` must appear in both raters' progress; the adjudication ledger must exactly equal the set of A/B disagreements; the Rater B provenance payload must agree with per-row model / prompt_hash / rubric_version fields (otherwise raises).
- Statistics: Cohen's κ via `sklearn.metrics.cohen_kappa_score`; AC1 via an in-repo implementation (`bridge_irr.gwet_ac1`) unit-tested for perfect-agreement and symmetric labeling; Wilson intervals for all shares via `scripts/uncertainty.py`.

---

## 2. Results (Data)

### 2.1 Inter-rater agreement

| Metric | Value | 95% CI |
|---|---|---|
| Raw agreement (Rater A vs Rater B) | 55/57 = 96.5% | [88.1, 99.0] (Wilson) |
| Cohen's κ (4-category) | 0.8995 | — |
| Gwet's AC1 | 0.9603 | — |
| n_disagreements | 2 | — |
| rule_gap fraction at adjudication | 0/2 | [0, 65.8%] (Wilson, uninformative) |

### 2.2 Rater marginal distributions

Both raters independently produced identical marginal distributions on the 57 test cases:

| Category | Rater A | Rater B |
|---|---:|---:|
| `wrong_entity_substitution` | 45 | 45 |
| `evasion_or_factual_denial` | 9 | 9 |
| `answer_dilution` | 3 | 3 |
| `formal_refusal` | 0 | 0 |

The identical marginals are incidental to this particular adjudication — the two disagreements cancel by construction (one flipped A→B direction, one flipped B→A). The confusion matrix exposes this: the only off-diagonal entries are (45,1)↔(1,2) between `wrong_entity_substitution` and `answer_dilution`.

### 2.3 Adjudicated category shares (§4.3 paper claim)

Right-to-wrong flips (n=43, the L4 main-text claim):

| Category | Count | Share | 95% CI (Wilson) |
|---|---:|---:|---|
| wrong_entity_substitution | 31 | 72.1% | [57.3, 83.3] |
| evasion_or_factual_denial | 9 | 20.9% | [11.4, 35.2] |
| answer_dilution | 3 | 7.0% | [2.4, 18.6] |
| formal_refusal | 0 | 0.0% | [0.0, 8.2] |

Wrong-to-right rescues (n=14):

| Category | Count | Share | 95% CI (Wilson) |
|---|---:|---:|---|
| wrong_entity_substitution | 14 | 100.0% | [78.5, 100.0] |
| evasion_or_factual_denial | 0 | 0.0% | [0.0, 21.5] |
| answer_dilution | 0 | 0.0% | [0.0, 21.5] |
| formal_refusal | 0 | 0.0% | [0.0, 21.5] |

### 2.4 What changed vs. the precursor Phase-3 report

The 2026-04-13 Phase 3 report §5 used single-rater numbers (30/8/3/2 = 70/19/7/5%). After dual-rating and adjudication the numbers are 31/9/3/0 = 72/21/7/0%. The net effect on paper claims:

- Wrong-entity share ticked **up** (30→31 under R2 generic-descriptor treatment of "A mixed-breed dog").
- Formal refusal **collapsed to zero** (2→0). Two cases originally tagged `formal_refusal` under the legacy 5-mode taxonomy (which conflated `NOT_ATTEMPTED`-graded outputs with explicit refusal language) were re-coded under R1's stricter operational test — the responses hedge or deny but do not contain explicit refusal/safety language.
- Evasion/denial ticked up 8→9 and dilution held at 3.
- The qualitative ordering — "wrong-entity substitution dominates, evasion is a distant second, refusal is not the story" — survives and is now interval-qualified.

---

## 3. Original-Intent Check (vs. `TODO_L4_interrater.md`)

Point-by-point against the original L4 plan:

| Asked | Delivered | Notes |
|---|---|---|
| Blind re-code all 57 discordant cases | Yes | 43 R→W + 14 W→R, both directions coded |
| Second rater: best-effort priority (human > LLM > self-blinded) | LLM (sensitivity check) | Explicitly the middle tier; flagged in paper §6 as sensitivity check, not strong-form IRR |
| Predefine adjudication rule before seeing disagreements | Yes | §1–8 frozen, committed at `0e965d5`, sha256 pinned in summary; rule_gap protocol allows best-fit fallback without retro amendment |
| Report raw agreement, Cohen's κ, Gwet's AC1 | Yes | 96.5%, 0.90, 0.96; AC1 reported as robustness check against skew |
| Keep main claim qualitative if agreement is weak | Main claim is qualitative **and** quantitative | Dual-rated κ=0.90 lets us report "$72\%$; 95\% CI $[57, 83]$" alongside the qualitative "wrong-entity is the dominant coded mode" — stronger than the minimum deliverable |
| Minimum paper deliverable (one sentence + table) | Delivered (main.tex L238 IRR sentence, Table on p.~273 caption, Table in supplement) | |
| Better deliverable (transition × category table) | Delivered (supplement/failure_coding_manifest.md) | The 14 W→R cases are 100% wrong_entity; see §4 for what this implies |
| Scope: 57 cases not just 43 | Yes | This is the expanded L4 plan from `TODO_Limitations_Fixes.md` §L4, adopted verbatim |
| Reuse `scripts/analyze_concordance.py` if possible | Not reused; new pipeline built | `analyze_concordance.py` is an LLM-holdout script for the CSV2 v3 evaluator comparison. Its scope (per-item binary/graded ratings, not taxonomy categories) was wrong for this task. The new `scripts/bridge_irr.py` is purpose-built and unit-tested |
| Cost budget (~2 h + ~$1) | Consistent — dev calibration + test batch combined used ~$1 of OpenAI credits and ~45 min of Rater A time + adjudication | |

**Original intent carried.** The one downgrade from "best" to "acceptable" (LLM instead of human Rater B) is explicitly flagged in the paper limitation, the supplement caveats, and §4.3's phrasing ("dual-rated failure coding", not "inter-annotator-reliability").

---

## 4. Interpretation (Clearly Separated From Data)

### 4.1 What this supports

**The wrong-entity substitution claim is now robust to the "subjective coding" critique.** Before L4, a reviewer could write: "the 70% number is one author's opinion". After L4, a reviewer has to engage with: a pre-registered four-category rubric, two independent raters (one a pinned snapshot of a widely used LLM judge at temperature=0 with a strict JSON schema), 96.5% raw agreement, κ=0.90, a frozen adjudication rule, and a published per-case rationale. That critique cannot carry the paper anymore.

**The R→W / W→R symmetry strengthens the mechanism reading.** Damage and rescue *both* flow through the same category (72% vs 100% wrong-entity). If the ITI direction were a targeted "truth signal," we would expect W→R rescues to look qualitatively different from R→W damage — rescues should show signs of selective amplification of the correct entity. Instead both directions look like the same operator: redistribution of probability mass within a semantic neighborhood, with the outcome depending on whether the correct candidate happens to be the one whose margin increases. This reinforces the paper's "indiscriminate redistribution" reading rather than a "truthfulness injection" reading.

**The calibration story is clean.** The §9 revision was made once, as the protocol allows, and produced 13/13 dev agreement. This is a one-bit observation (agreement increased), but the rule-gap fraction 0/2 on the test split is consistent with the dev calibration having successfully specified all patterns present in this data.

### 4.2 What scrutiny this withstands

- *Reviewer:* "Maybe the LLM judge just mirrors the first author." The prompt has no Rater-A-specific exemplars; Rater B is zero-shot on the published rule with temperature=0. The only shared artifact is the rubric itself, which is the instrument we are trying to calibrate — agreement here is what we want to measure.
- *Reviewer:* "κ can be inflated under skew." Gwet's AC1 is reported specifically to diagnose this: AC1 is 0.96 > κ 0.90, which is consistent with AC1's less-penalizing behavior when one category dominates. Both statistics are "almost perfect" under the Landis–Koch convention; the decision is not sensitive to which is reported as primary.
- *Reviewer:* "Two disagreements is a small number." Yes, and the Wilson CI on rule-gap (0/2) is uninformative. The strong claim is 0 rule-gaps *observed*, not a bound on rule-gap *prevalence*. We do not claim the rule is complete; we claim it covered every case we saw.

### 4.3 What this does NOT support

- **This is not a human–human IRR.** Rater B is an LLM judge; κ=0.90 between "first author" and "GPT-4o on the same rubric" is not the same evidentiary object as κ=0.90 between two independent human coders. The paper says so explicitly (§6 limitation L4, supplement caveats).
- **This is not a mechanism claim.** "Probability mass redistribution" is a behavioral diagnosis. We did not measure the log-likelihood margin between the gold answer and the generated wrong entity, and we did not verify that ITI actually reduces that margin on the 31 adjudicated substitution cases. That is the Priority-3 experiment from `TODO_Limitations_Fixes.md` and remains a paper-upgrading opportunity.
- **This is not an "exactly 72%" claim.** The Wilson CI [57, 83] is wide. The paper correctly states the interval in §4.3 body text and Fig 3 caption; readers who extract only the point estimate are invited to misread the evidence.
- **This says nothing about E1 (modernized ITI) or other artifacts.** The discordant set is defined against E0 paper-faithful, K=12, α=8.0 only. Any generalization to other ITI variants is an inferential leap.
- **This says nothing about other models.** Gemma-3-4B-IT only; L1 (single-model) is untouched.

### 4.4 Interesting incidental findings

- **Zero formal refusals in 43 R→W flips** (Wilson upper 8.2%). Previously under the single-rater legacy 5-mode scheme, 2 cases were tagged `formal_refusal` because they were `NOT_ATTEMPTED` under the grading policy. Under R1 of the frozen rule (which requires explicit refusal / safety-policy language and no answer attempt), both collapsed to evasion (R4). This is a subtle but paper-level-relevant change: the paper can now assert that **refusal is not a failure mode of E0 ITI at α=8.0 on this surface**, not merely "rare".
- **W→R rescues are 14/14 wrong-entity.** All 14 rescue cases are cases where the baseline committed to a wrong entity and ITI committed to the correct one. There are no rescue cases where ITI moved from a denial/dilution baseline to a correct answer. This narrows the mechanism: ITI does not seem to "un-refuse" or "un-dilute" — it rewrites committed entities. Given that R→W damage and W→R rescue share this profile, this is circumstantial evidence that the intervention operates on committed-answer logits, not on the "should I answer" decision.
- **Rater A marginal and Rater B marginal coincide exactly.** 45/9/3/0 each. This is an arithmetic coincidence of how the two disagreements happened to cancel, not a structural property. The confusion matrix makes this visible: off-diagonal entries are symmetric at (1, 1). Do not read this as "the raters produced the same labels"; read it as "the two disagreements were on category boundaries that cancel in the margin."

---

## 5. Uncertainties (Quantified Where Possible)

Uncertainty ranking from *low* to *high*:

| # | Uncertainty | Qualitative level | Why |
|---|---|---|---|
| 1 | Did Rater B follow the rule as written? | Very low | Zero-shot on the published rule; temperature=0; strict JSON schema; 96.5% raw agreement with Rater A; zero schema failures across 57+13 cases |
| 2 | Does the rubric cover every failure pattern in this data? | Low | 0/2 observed rule-gaps on test + 0/13 on dev; Wilson upper for rule-gap rate is 65.8% but n=2 makes the bound uninformative |
| 3 | Would a real human second rater reproduce the category shares? | Moderate | Unknown. The rubric operationalizes boundary cases (R5/R6 tiebreakers) but does not eliminate them. Dev agreement 13/13 is encouraging but small. Best upper bound we have: rubric-grounded human–human agreement should not be much worse than human–LLM agreement *because the LLM is calibrated to the same rubric*; this is circular if taken as strong evidence |
| 4 | Is κ=0.90 inflated by rater anchoring on the same rule? | Moderate | Yes, to some extent, by design — the rule is the whole point. AC1=0.96 addresses the prevalence-skew critique specifically; whether it addresses rule-anchoring is less clear |
| 5 | Does the 72.1% point estimate generalize to a fresh sample of 43 R→W flips? | Moderate | Wilson CI [57, 83] is our best bound. A second ITI run on a different test seed would pin this down; we do not have that data |
| 6 | Does this taxonomy generalize to other ITI variants or other models? | High | Untested. E0 only, Gemma-3-4B-IT only |
| 7 | Is the "indiscriminate redistribution" interpretation correct? | High | Consistent with R→W / W→R symmetry observation, but this is a behavioral pattern, not a mechanistic measurement. Priority-3 log-likelihood-margin experiment would upgrade this |

---

## 6. Pipeline Strengths (Scientific Best-Practice Audit)

The pipeline is conservative in several ways that are easy to underweight:

1. **Tamper-evident freezing.** The frozen adjudication rule travels with the summary via (git_commit, content_sha256), not just a path reference. A post-hoc edit to the rule file would break the hash check at finalize time.
2. **Blinded coding surface.** Raters cannot infer direction (baseline vs ITI) from the stimulus. The direction is preserved only in the key file, which is never loaded during labeling.
3. **Staleness guard on progress files.** `ensure_progress_files_compatible` halts with a typed error if a re-generated queue would orphan any existing label row — a meaningful defense against silent case-ID drift.
4. **Reopen-per-record JSONL writes.** Survives concurrent git operations and power loss mid-session; failure mode is "lose the last partial record", never "lose the whole file".
5. **Batch-mode Rater B with resume state.** `.rater_b_batch_state.json` allows resumption after partial failure; 57/57 succeeded on first submission.
6. **Dual statistic reporting.** Raw agreement + κ + AC1. AC1 specifically addresses Feinstein–Cicchetti skew concerns on κ. Wilson intervals (not normal approximations) for all shares.
7. **Per-case provenance.** Every adjudicated label carries A-label, B-label, A-confidence, B-confidence, A-notes, B-notes, label_source (consensus vs adjudication), and rule_gap. Disagreement audit is reproducible from the artifact alone.
8. **Honest framing throughout.** The LLM-judge status is surfaced in the paper limitation, the supplement failure-coding manifest, §4.3, and the Benchmark table. No reader should arrive at the κ=0.90 number without also arriving at the sensitivity-check qualification.

---

## 7. Gaps and What More Could Be Done

### 7.1 Open gaps (fix-able quickly)

- **Stale reviewer-facing supplement file.** `paper/icml/supplement/support/externality_summary.md` still shows the pre-IRR single-rater counts (30 / 8 / 3 / 2 = 70/19/7/5%) and the caveat "single-rater manual coding pass over 43 flips." This drifts from `failure_coding_manifest.md`, `site/data/bridge_phase3.json`, and the main paper. This report updates it (see §8).
- **Missing dev-calibration narrative.** The repo has the dev queues, two dev run logs, two rater_b_dev provenance files, and the revised rule hash — enough to reconstruct what happened, but no one-page report in `notes/act3-reports/`. A short `2026-04-21-bridge-irr-dev-calibration.md` documenting what the revision was and its effect on dev agreement would close the documentation gap. Not paper-blocking.

### 7.2 Gaps that require more work (ordered by expected value)

1. **Priority-3 (logprob-margin analysis).** Use the 31 adjudicated substitution cases as a labeled subset. For each R→W substitution case, compute under baseline and ITI:
   \[ \Delta_{\text{margin}} = \log p(\text{gold} \mid \text{prompt}) - \log p(\text{wrong entity} \mid \text{prompt}) \]
   Compare to R→W non-substitution flips and W→R rescues. If ITI specifically reduces this margin on substitution-coded cases but not on others, the behavioral taxonomy becomes a probabilistic diagnosis — the paper's wrong-entity-substitution claim upgrades from "behavioral pattern we observed" to "mechanism we measured."
   - Cost: one bridge rerun with logprobs captured, ~$0 API (no judge calls), ~30 min compute.
   - Risk: low; the adjudicated labels do the scoping work. Even a null result (no margin shift) is informative and would reshape the paper's interpretation.
   - Where it goes: §4.3 appendix (if clean), limitation L4 footnote (if null).

2. **Human–human sensitivity check on a 20-case subsample.** Recruit a second human rater (not the first author) to code a stratified 20 of the 57 cases (10 R→W substitution, 5 R→W non-substitution, 5 W→R). Compute human–human κ and AC1. This is not a full second IRR but it upper-bounds the "Rater B is the first author in disguise" critique.
   - Cost: ~45 min of one external reviewer. ~0 API.
   - Risk: low; worst case is a weaker κ, which we then report as-is.
   - Where it goes: supplement only; the main claim does not need to depend on it.

3. **Rule-gap upper-bound tightening.** The current 0/2 rule-gap rate is an uninformative bound. Ask Rater A (now that the rule is frozen and the test labeling is complete) to reclassify a randomly chosen 20 of the 55 consensus cases blind to the existing label. If the reclassified label matches the original, count it as a consensus reaffirmation. If it diverges, check whether the original label was under a clearly-fitting rule or under a stretched-fit.
   - Cost: ~20 min.
   - Risk: low-moderate. If the blind-reclassify agreement rate is materially worse than the Rater A/B agreement rate, it would suggest the published rule is less mechanical than it looks, which is useful to know.
   - Where it goes: supplement caveats.

4. **Cross-model bridge IRR.** Outside the L4 scope but worth naming: repeat the bridge + IRR on a second model family (Llama-3.1-8B-Instruct or Qwen-2.5-7B-Instruct) to test whether wrong-entity substitution is the E0-ITI-on-Gemma signature or a more general ITI phenomenon.
   - Cost: moderate (one bridge run on a second model); material paper upgrade (L1 single-model weakening).
   - Where it goes: main text §4.3 or new appendix; out of deadline scope for the current ICML submission.

### 7.3 Things *not* worth doing

- **A full-blown second LLM judge.** AC1+κ already provide the robustness the paper needs. Running a second frontier model as rater C would triple cost for marginal reviewer-credibility gain; it cannot substitute for a human sensitivity check.
- **Manual re-audit of the 55 consensus cases.** The rules applied to consensus cases are already inspectable via the `rater_a_notes` / `rater_b_notes` fields on each adjudicated row. A blind re-audit would churn without new evidence unless done on a random subsample (see §7.2#3).

---

## 8. Integration Status

### 8.1 Already updated (before this review)

- **Paper main text** (`paper/icml/main.tex`): §4.3 body (IRR sentence, adjudicated counts), Fig 3 caption (72% [57, 83]), Table `tab:substitution` caption, §6 limitation, Benchmark row, L4 row.
- **Paper supplement** (`paper/icml/supplement/failure_coding_manifest.md`): adjudicated category table, IRR section, expanded provenance. Package manifest JSON includes `bridge_irr_label.py` and `bridge_irr_rater_b.py`.
- **Site data** (`site/data/bridge_phase3.json`, static HTML fallbacks in `site/index.html` / `story.html` / `extensions.html` / `progress/week-04-flagship-synthesis.html`).
- **Number provenance** (`paper/icml/number_provenance.md`): §4.3 rows now source from `bridge_irr_summary.json`.
- **Tests** (`tests/test_export_site_data.py`): assertions updated to the new adjudicated numbers.
- **Figure regen** (`paper/icml/figures/fig3_bridge.pdf`): rebuilt from `bridge_irr_summary.json`.

### 8.2 This review closes the remaining drift

- Updates `paper/icml/supplement/support/externality_summary.md` to adjudicated numbers + links back to this report for prose.
- Adds a superseded-for-§5-taxonomy header to `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md` pointing to this report (body left frozen per `notes/CLAUDE.md` supersede policy).
- Adds a link from `paper/icml/supplement/failure_coding_manifest.md` to this report for prose analysis.
- Adds an entry to `notes/research-log.md` dated 2026-04-21.

### 8.3 Out-of-scope for this review (flagged, not fixed)

- `notes/act3-reports/2026-04-21-bridge-irr-dev-calibration.md` — a one-page dev-calibration narrative does not exist; recommended in §7.1 but out of scope for the pipeline review itself.

---

## 9. Bottom Line

The bridge IRR pipeline does what the L4 plan asked and a little more: 57 cases (vs the minimum 43), a pre-frozen rule with tamper-evident provenance, a published adjudication protocol, a dual-statistic (κ + AC1) report with Wilson intervals on every share, and a honest framing that never hides the "LLM-judge as second rater" status. The main paper claim — wrong-entity substitution is the dominant R→W failure mode — is now defensible at the reviewer level with interval-qualified quantification.

The one structural honest-to-reviewer weakness (human–LLM rather than human–human IRR) is the same weakness `TODO_L4_interrater.md` and `TODO_Limitations_Fixes.md` predicted going in. It is surfaced exactly where reviewers will look for it: main text §6, supplement manifest, and the L4 row of the limitations table. The paper treats this as a sensitivity check, which is accurate.

The most valuable next move is the Priority-3 logprob-margin analysis: it turns the 31 adjudicated substitution cases from a labeled-behavior set into a labeled-mechanism set, and is cheap. A short human–human sensitivity check on a 20-case subsample would close the "Rater B is the author in disguise" critique at very low cost. Neither is required for the current paper; both would strengthen it.

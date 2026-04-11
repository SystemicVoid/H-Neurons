# Critical Review — Documentation Housekeeping Pass (2026-04-11)

---

## 1. Factual Errors

### 1.1 Apr 6 entry misattributes Apr 5 work

**File:** `notes/research-log.md` lines 155–169  
**Severity:** High

The Apr 6 entry claims:

> "Finalized the D7 pipeline implementation and executed the first runs. Committed the extraction families (`iti_refusal_probe` and `iti_refusal_causal`), jailbreak decode controls, alpha-lock utility, paired CSV2 report utility, and the staged orchestrator. Generated the JBB paired manifests and refusal extraction artifacts for both families. Launched the D7 pilot100 runs via `d7_causal_pilot.sh`."

**None of this was committed on Apr 6.** Git log confirms all of the listed work is Apr 5:

- `dd12d89` feat(extract): add iti_refusal_probe and iti_refusal_causal — **Apr 5**
- `2a87ac6` feat(eval): add jailbreak decode controls — **Apr 5**
- `698371e` feat(d7): alpha lock, CSV2 report, orchestrator — **Apr 5**
- `09b0454` data(d7): JBB manifests and extraction artifacts — **Apr 5**
- `4f83424` data(d7): D7 pilot100 and full500 outputs — **Apr 5**

The actual Apr 6 work (3 commits) was infrastructure refactoring:
1. `53aaced` feat(infra): add pipeline guard library for GPU pipeline orchestration
2. `4898790` refactor(d7): migrate d7_causal_pilot.sh to pipeline guard library
3. `33d2a69` fix(d7): correct d7_monitor.sh paths

The entry also claims: "split the research log at the ITI artifact exploration / closure phase boundary" (`944049e` — **Apr 5**) and "hardened run-profile guardrails" (`7584677` — **Apr 5**).

**Fix:** The Apr 6 entry needs to be rewritten. The actual Apr 6 work was: (a) building a pipeline guard library for GPU orchestration, (b) migrating the D7 orchestrator to it, (c) fixing d7_monitor.sh paths. The D7 pilot was launched on Apr 5, continued overnight into Apr 6. Generation provenance timestamps confirm runs starting at 16:52 on Apr 5.

### 1.2 Apr 6 entry claims `torch.inference_mode()` fix was "during the session"

**File:** `notes/research-log.md` line 161  

The entry states: "One operational fix: `torch.inference_mode()` was missing from the hot inference paths...patched during the session."

Commit `cab17fd refactor(eval): use torch.inference_mode() in hot inference paths` is dated **Apr 5**, not Apr 6.

**Fix:** This belongs in the Apr 5 entry, not Apr 6.

### 1.3 Apr 9 entry omits the concordance system entirely

**File:** `notes/research-log.md` lines 64–78  
**Severity:** Medium

The entry describes "two evaluation infrastructure pieces" but the actual Apr 9 work includes **four** commits, with the largest being `5f572e9 feat(concordance): add three-judge concordance analysis and v2 judge extraction` (1818 new lines — the biggest commit of the day). The concordance analysis system is never mentioned.

The "jailbreak CSV-v2 negative control pipeline" described in the entry is actually the **concordance** negative control pipeline (`a6509d3`, `1bceebd`). The word "concordance" does not appear in the entry.

**Fix:** The Apr 9 entry should name the concordance system as a third piece of infrastructure, or at minimum acknowledge that the "negative control pipeline" is specifically a three-judge concordance pipeline.

### 1.4 Apr 9 entry claims literature additions not in git

**File:** `notes/research-log.md` line 66  

> "Also added GuidedBench and JADES jailbreak evaluation papers to the literature."

No Apr 9 commit touches literature files. This claim is unverifiable from the commit history.

**Fix:** Either confirm these additions exist (perhaps uncommitted changes) or remove the claim.

### 1.5 §10 Stage 5 "probe null at every alpha" is pilot-scale only

**File:** `notes/act3-reports/optimise-intervention-ac3.md` line 814  

The §10 table states: "probe null at every alpha." The full-500 audit (the linked canonical report) explicitly says the probe condition was **not** completed at full-500 scale (84/500 rows, no judge/CSV2 outputs). The probe null comes from the pilot100 only.

A reader following the canonical report pointer finds that probe was incomplete, which contradicts the "probe null" claim appearing alongside full-500 results.

**Fix:** Qualify as "probe null at every alpha (pilot100 scale; not completed at full-500)."

---

## 2. Overstatements / Understatements

### 2.1 §0 closure: "All five questions below have been answered"

**File:** `notes/act3-reports/optimise-intervention-ac3.md` line 60  

The §0 blockquote says "All five questions below have been answered by completed experiments." But the five numbered items below are not five questions — they are a mix of established facts (#1, #2), a strategic conclusion (#3), a priority recommendation (#4), and a benchmark-timing judgment (#5). Recommendations and strategic conclusions are not "answered by experiments."

The phrase "See §10 for the summary table" implies "five questions" = the five stages in §10. But "below" points to §0's own numbered list, not §10.

**Fix:** Either reword to "The five stage questions in §10 have been answered" (which is what's meant) or rephrase as "The empirical premises below have been resolved by completed experiments." Don't call recommendations "questions."

### 2.2 §9 closure: "The three unknowns...have been resolved"

**File:** `notes/act3-reports/optimise-intervention-ac3.md` line 784  

The third unknown (component choice) has the random-head control still missing. The closure notes this caveat inline, but the headline verb "resolved" is too strong for a partially-answered question. The same line admits "random-head control for selector-specificity claim still missing."

**Overstatement:** Calling something "resolved" while simultaneously noting the key evidence gap in the same sentence. These two statements are in tension.

**Fix:** "The three unknowns this judgment deferred to have been substantially addressed" or "resolved at the benchmark level; selector-specificity remains open."

### 2.3 §10 Stage 3 summary omits E1's formally significant result

**File:** `notes/act3-reports/optimise-intervention-ac3.md` line 812  

The Stage 3 terminal status says: "E0 ITI informative null (47% baseline, Δ = -1pp / -7pp)."

The bridge Phase 2 report also includes E1 α=8.0 at Δ = -9pp, CI [-16%, -3%], McNemar p=0.016 — the only **formally significant** result on the bridge benchmark. E1's result strengthens the "method-level failure" narrative (E1 is *worse* despite being a "gentler" artifact). Omitting it from the Stage 3 summary understates the generation evidence.

**Fix:** Add "E1 α=8.0: -9pp (p=0.016, formally significant)" to the Stage 3 summary.

### 2.4 Strategy note status line: "all five stages terminal"

**File:** `notes/act3-reports/optimise-intervention-ac3.md` line 3  

Calling Stage 5 "terminal" while it has an incomplete probe arm and a missing random-head control is a stretch. The body text carefully notes these gaps, but the status line flattens them into "terminal." This matters because the status line is the first thing a reader sees.

**Fix:** "All five stages reached terminal or near-terminal status" or add a parenthetical: "terminal (Stage 5 with caveats)."

### 2.5 D4 status cell buries the generation null

**File:** `notes/act3-sprint.md` line 48  

D4's status says: "**closed** — MC winner (+6.3pp MC1), generation null on all benchmarks tested." The next-action cell then says: "**Closed.** MC winner (+6.3pp MC1), generation null on all benchmarks tested (SimpleQA, TriviaQA bridge). Artifact lane exhausted (E1 tradeoff, E2 null, E3 gate not met). Transfer lane closed (E2 synthesis)."

This is not an error — it's accurately stated. But it does embed result numbers ("+6.3pp MC1") in the sprint file, which the governance rules (`notes/CLAUDE.md` line 10) explicitly prohibit: "Do not write numbers or CIs into act3-sprint.md status cells — they will go stale." The number appears in status, link text, and next-action.

**Fix:** Replace with "MC winner" without the "+6.3pp" figure. The report links carry the numbers.

---

## 3. Gaps That Should Be Fixed

### 3.1 D3.5 next-action is stale

**File:** `notes/act3-sprint.md` line 46  

D3.5 next-action: "Do **not** orthogonalize D4 yet."

D4 is now closed. There will never be a D4 to orthogonalize. The conditional "yet" implies a future event that will not occur.

**Fix:** Update to something like "No action needed; D4 closed without triggering orthogonalization."

### 3.2 D6 should be explicitly deprioritized

**File:** `notes/act3-sprint.md` line 50  

D6 status is "not started" with a forward-looking next-action: "Project out the refusal-overlap component from the truthfulness or hallucination vector and compare the before-versus-after tradeoff."

D6 was conditional on D5 findings (sprint line 68: "conditional on D5 findings"). D5 is deferred. D6's next-action reads as a live plan when nothing will trigger it in the current sprint. The current priorities (lines 71–77) don't mention D6.

**Fix:** Add "Deprioritized: conditional on D5, which is deferred" to D6's status or next-action.

### 3.3 D1 next-action mixes current and optional priorities

**File:** `notes/act3-sprint.md` line 43  

D1 next-action: "Finish the graded-jailbreak negative control and capability mini-battery (IFEval + perplexity)."

The current priorities (lines 73–77) list the jailbreak control as priority #1 and the capability battery as optional priority #5. But D1's next-action treats both as co-equal. A reader looking at D1 alone would think the capability battery is required, not optional.

**Fix:** Split or annotate: "Finish the graded-jailbreak negative control. Capability mini-battery (IFEval + perplexity) is optional — see current priorities."

### 3.4 Sprint Questions are answered but read as open

**File:** `notes/act3-sprint.md` lines 32–35  

The four sprint questions:
1. "Does full-generation, graded evaluation change the practical safety conclusion?" — **Answered yes** (binary judge washes out signal; graded CSV2 recovers it).
2. "Are direction-level baselines cleaner or stronger?" — **Answered: neither universally wins.**
3. "How much of the observed safety regression is explained by overlap with refusal geometry?" — **Partially addressed** (D3.5 shows real overlap, robustness insufficient).
4. "Can one small causal pilot show why correlational localization is a weak intervention selector?" — **Answered yes** (D7 causal beats probe and L1; specificity pending random-head).

These read as open questions to anyone who doesn't already know the answers. The original plan to annotate them with answers was deferred to avoid "premature interpretive framing." But the sprint is at terminal status — the framing is no longer premature. Leaving them unannotated creates an impression of open questions in what is now a mostly-closed sprint.

**Fix:** Add inline status annotations (e.g., "[Answered — see D7 and measurement-blueprint]") without writing result numbers.

---

## 4. Gaps That Are Correctly Left Alone

### 4.1 D0.5 "IFEval + perplexity remain open" — correct

D0.5's partial status is accurate. IFEval and perplexity were never built. The capability battery appears in the current priorities as optional (#5). Updating D0.5 to "closed" would be wrong; keeping it "partial" with the specific gap named is the right state.

### 4.2 North Star section — correct to leave

"Act 3 is a final-sprint comparative program" is past-tense accurate and describes the sprint's nature, not its current execution state. The sprint status is carried by the deliverable table and current priorities section. Rewriting the North Star to say "was" would be editorializing.

### 4.3 "What would falsify it" column — correct to leave

The falsification criteria in Claims To Test are historical context showing what was pre-registered before investigation. Leaving them alongside resolved status creates a useful audit trail: "we said X would falsify it, and the actual finding was Y." Removing them would lose the pre-registration record.

However, note that Claim #4 ("Direction-level interventions are cleaner") has a falsification criterion about the refusal-direction specifically, while the status discusses ITI and SAE — different comparisons. This is a category drift between what was planned and what was tested, not a documentation error.

### 4.4 Archived research log — correct to skip

`notes/act3-reports/research-log-iti-artifact-exploration.md` is marked "Archived" and frozen. The governance rules say "Leave the old report body frozen — don't edit numbers in place." Checking it for ordering bugs would require reading and potentially modifying a frozen document.

### 4.5 `scratchpad.md` — correct to exclude

Explicit user instruction. Contains pre-pivot content that shouldn't be updated piecemeal.

### 4.6 `measurement-blueprint.md` — acceptable to defer

Not touched by this pass, and no specific staleness was identified in the review. The blueprint is referenced as a contract, and the sprint still gates on it. If a staleness issue exists, it would need a separate targeted review.

### 4.7 `plot-registry.md` — acceptable to defer

No new data files were produced by this housekeeping pass. Plot registry updates belong with actual new data or figure production.

---

## 5. Reconstructed-Entry Critique

### 5.1 Apr 6 — Fundamentally wrong about what was done

As detailed in §1.1, the entry describes Apr 5 work as Apr 6 work. The actual Apr 6 contribution was narrow: a pipeline guard library and two refactoring commits. The entry should be rewritten from the actual Apr 6 commits.

The claim "Extraction and generation stages ran cleanly on first pass" (line 161) is framed as an Apr 6 observation, but generation launched on Apr 5 (provenance timestamps start at 16:52 on Apr 5). By Apr 6, the pilot was already generating overnight. The "ran cleanly" inference from absence of fix commits is directionally reasonable, but the temporal framing is wrong.

The "What I expected vs what happened" section discusses test coverage from Apr 5 catching integration issues — this is retrospective commentary on Apr 5's work, not a description of Apr 6 surprises. A real Apr 6 entry would discuss the pipeline guard library design decision or the d7_monitor.sh path correction.

### 5.2 Apr 9 — Omits the largest piece of work

The concordance analysis system (`analyze_concordance.py` — 883 lines, `evaluate_csv2_v2.py` — 927 lines) is the single largest artifact committed on Apr 9 and goes completely unmentioned. This is a three-judge concordance framework, not just a "negative control pipeline."

The entry frames the day as "two infrastructure pieces" when there were at least three: (1) CSV v3 evaluator, (2) concordance analysis with v2 judge extraction, (3) concordance negative control pipeline. The bug fix (`1bceebd`) suggests the concordance system needed iteration, which is relevant to the "required more care than anticipated" narrative — but the care was in concordance scoping, not just span validation.

### 5.3 Apr 9 — "span-validation" narrative may be incomplete

The entry attributes the extra complexity to "span-validation logic" and "`primary_outcome` taxonomy interactions with C/S/V axes." While these are plausible from the CSV v3 evaluator commit (`3f6d0e5`), the 4-commit structure (feat + feat + feat + fix) suggests the concordance system itself was also non-trivial. The "required more care" may be about concordance scoping as much as span validation.

### 5.4 General concern: reconstructed entries create false confidence

Both entries read as first-person narratives ("What I expected vs what happened") but were reconstructed from commit messages. They include subjective framing ("ran cleanly on first pass," "required more care than anticipated") that sounds like lived experience but is actually inference. The research-log format is designed for surprises and judgment — reconstructed entries can only infer these, not report them. There is no flag in either entry indicating it was reconstructed rather than written contemporaneously.

**Recommendation:** Add a small notation like "> *Reconstructed from git history; not a contemporaneous entry.*" to both.

---

## 6. Internal Contradictions

### 6.1 D5 description: sprint vs strategy note

**Sprint** (line 49): "Assess whether D1/D4/D7 cross-benchmark divergences provide sufficient externality coverage before committing GPU time."

**Strategy note §10** (line 821): "D5 externality audit — deferred; assess existing cross-benchmark data first."

These are not contradictory, but they differ in specificity. The sprint names the deliverables to assess (D1/D4/D7) and the criterion (sufficient externality coverage). The strategy note is vaguer ("existing cross-benchmark data"). A reader of the strategy note alone wouldn't know which data to assess.

**Risk:** Someone following up on D5 from the strategy note would lack the sprint's actionable framing.

**Fix:** Align the strategy note's wording to reference D1/D4/D7 explicitly.

### 6.2 D4 status cell embeds numbers; governance forbids it

**Sprint** (line 48): "+6.3pp MC1" appears in both the status and next-action cells.  
**Governance** (`notes/CLAUDE.md` line 10): "Do not write numbers or CIs into act3-sprint.md status cells — they will go stale."

This was likely inherited from a pre-governance update, but it contradicts the explicit rule.

### 6.3 Apr 5 "What I will do next" vs Apr 6 "What I did"

**Apr 5** (line 196): "Execute the D7 pilot run."  
**Apr 6** (line 155): "Finalized the D7 pipeline implementation and executed the first runs."

This creates a narrative where Apr 5 planned to execute and Apr 6 did it. But provenance shows execution starting on Apr 5 afternoon. The continuity is broken: Apr 5 says "will do X," Apr 6 says "did X," but the git record says Apr 5 already did X.

### 6.4 Claims To Test row #4 vs §10 Stage 2

**Claims row #4** (line 127): "Tested. Neither universally wins. ITI beats H-neurons on TruthfulQA MC; H-neurons beat SAE features on FaithEval steering."

**§10 Stage 2**: "E1 tradeoff, E2 null, E3 gate not met. Lane exhausted."

These are about different things (Claims #4 is about direction-vs-neuron overall; Stage 2 is about artifact improvements within ITI). But a reader looking for "direction vs neuron" evidence would find the claim in the sprint and the stage in the strategy note, and they tell different stories because they're answering different questions. The sprint's claim #4 is a cross-family conclusion; the strategy note's Stage 2 is an intra-family conclusion. No cross-reference connects them.

---

## 7. Remaining Documentation Debt

### 7.1 Numbers in sprint status cells

Multiple cells contain result numbers that the governance rules prohibit: D4 (+6.3pp MC1), D7 full-500 numbers are referenced in the next-action text (though indirectly). A systematic pass to strip numbers from status/next-action cells and replace with qualitative summaries + report links would bring the sprint into compliance.

### 7.2 Broken-link verification

No link verification was performed. The sprint has ~25 internal links. The strategy note has ~15. Any rename or move since these were written could leave dead links. A targeted check (e.g., `grep -oP '\[.*?\]\(.*?\)' | while read ...`) would catch these cheaply.

### 7.3 `runs_to_analyse.md` — unknown state

Not mentioned in the review. If any analysed runs remain in this file, they should be removed per governance. If the D7 full-500 entry was removed during the Apr 8 log entry, that's fine. If not, it's stale.

### 7.4 D7 pilot audit supersession note

The D7 full-500 audit (`2026-04-08`) correctly says it supersedes the pilot audit. But the pilot audit (`2026-04-07`) has no header note pointing to the full-500 audit as its successor. Governance says: "Add a header note to the old report pointing to the new one." This was not done.

### 7.5 Research-log inline modification (Apr 10 entry)

The Apr 10 entry has an inline strikethrough + resolution added on Apr 11 (lines 51, 56). The governance says "Leave the old report body frozen — don't edit numbers in place" for **reports**, and the research log's rules are less explicit. But the pattern of modifying a historical entry in-place is a governance gray area. The fix is small (gold-label confirmation) and clearly marked with strikethrough, so it's defensible. But if this pattern proliferates (inline updates to old entries), the log loses its "frozen snapshot" property. A footnote or addendum format would be safer for future cases.

### 7.6 `measurement-blueprint.md` staleness check deferred

The blueprint is referenced as the evaluation contract, but no one has verified whether it reflects the actual practice that evolved during Apr 1–11. Specific concern: does it still require a capability mini-battery for each intervention family? If so, no intervention (including D7 causal) has met the Definition of Done. This affects D8 synthesis claims.

### 7.7 Sprint Definition of Done vs actual status

The Definition of Done (lines 89–98) requires 8 items per baseline. A quick check:

| Requirement | D4 ITI | D7 Causal |
|---|---|---|
| Full-generation evaluation | ✅ | ✅ |
| Primary safety/response metrics | ✅ | ✅ |
| Retained-capability mini-battery | ❌ | ❌ |
| Per-example outputs | ✅ | ✅ |
| Run manifest with settings | ✅ | ✅ |
| One-paragraph interpretation | ✅ | ✅ |
| Negative control | ✅ (random-head on SimpleQA) | ❌ (random-head skipped) |
| Cross-benchmark consistency statement | ✅ (bridge + SimpleQA + MC) | ❌ (jailbreak only) |

Both D4 and D7 fail the Definition of Done on the capability battery. D7 also fails on negative control and cross-benchmark consistency. The sprint marks D4 as "closed" and D7 as "trimmed audit complete" — these statuses are honest, but neither formally satisfies the stated Definition of Done. This gap between the contract and the claimed status is not addressed anywhere.

### 7.8 No cross-reference from strategic assessment back to evidence gaps

The strategic assessment (`2026-04-11-strategic-assessment.md`) proposes a paper structure and "earned/not earned" claims. The strategy note §10 points to it but adds a caveat: "that framing has not yet been reviewed against the full evidence base." But the sprint's current priority #3 is exactly this review. There's no explicit link from the strategic assessment back to the Definition of Done gaps. If the paper claims rest on D7, the missing random-head control and capability battery should be flagged in the assessment's risk register — the risk register mentions the random-head gap but not the DoD gap.

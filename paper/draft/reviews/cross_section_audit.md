# Cross-Section Consistency Audit

Date: 2026-04-12
Scope: `paper/draft/full_paper.md` (all sections, abstract through appendix placeholder)

---

## 1. Number Consistency

### H-neuron AUROC (target: 0.843)

| Location | Quoted value | Status |
|---|---|---|
| Abstract (L11) | `AUROC 0.843` | OK |
| Introduction (L26) | `AUROC 0.848 vs. 0.843` | OK |
| Claim ledger comment (L107) | `AUROC 0.843 vs 0.848` | OK |
| Section 4.1 (L180) | `AUROC $0.843$` | OK |
| Section 4.2 summary (L246) | `AUROC $0.843$` | OK |
| Table 5 (L318) | `AUROC $0.843$` | OK |
| Section 7.1 (L698) | `AUROC 0.848 vs. 0.843` | OK |
| Section 8.1 (L775) | `AUROC $\approx$ 0.84` | OK (acceptable rounding) |
| Footnote fn-classifier-structure (L188) | `AUROC $= 0.8429$` | OK (4-decimal source value) |

**Verdict: Consistent.**

### SAE AUROC (target: 0.848)

| Location | Quoted value | Status |
|---|---|---|
| Abstract (L11) | `AUROC 0.848` | OK |
| Introduction (L26) | `AUROC 0.848` | OK |
| Claim ledger comment (L107) | `AUROC 0.843 vs 0.848` | OK |
| Section 4.1 (L182) | `AUROC $0.848$` | OK |
| Section 4.2 summary (L246) | `$0.848$` | OK |
| Table 5 (L318) | `$0.848$` | OK |
| Section 7.1 (L698) | `AUROC 0.848 vs. 0.843` | OK |
| Footnote fn-classifier-sae (L189) | `AUROC $= 0.8477$` | OK (4-decimal source value) |

**Verdict: Consistent.**

### FaithEval effect (target: +6.3 pp)

| Location | Quoted value | Status |
|---|---|---|
| Abstract (L11) | `+6.3 pp compliance gain` | OK |
| Table 1 (L68) | `H-neurons achieve +6.3 pp` | OK |
| Table 2 (L94) | `+6.3 pp [4.2, 8.5]` | OK |
| Claim ledger (L108) | `+6.3 pp FaithEval` | OK |
| Section 4.2 results (L222) | `+6.3 pp [4.2, 8.5]` | OK |
| Section 4.5 (L327) | `+6.3 pp [4.2, 8.5]` | OK |
| Section 5.1 (L338) | `+6.3 pp [4.2, 8.5]` | OK |
| Section 7.1 (L700) | `+6.3 pp` | OK |

**Verdict: Consistent.**

### ITI MC1 effect (target: +6.3 pp)

| Location | Quoted value | Status |
|---|---|---|
| Abstract (L11) | `+6.3 pp MC1` | OK |
| Table 1 (L66) | `ITI achieves +6.3 pp MC1` | OK |
| Claim ledger (L108) | `+6.3 pp MC1` | OK |
| Section 5.2 (L344) | `+6.3 pp MC1 [3.7, 8.9]` | OK |
| Section 5.4 (L376) | `+6.3 pp` | OK |
| Section 7.1 (L700) | `+6.3 pp MC1` | OK |

**Verdict: Consistent.** Note: FaithEval H-neuron and ITI MC1 coincidentally share the same +6.3 pp value. Verify this does not confuse readers -- both are distinct experiments. The paper handles this correctly by always qualifying with the benchmark name.

### Bridge effect (updated 2026-04-13 — Phase 3 test set is now primary)

**Primary number (test set):** E0 Δ adj −5.8 pp [−8.8, −3.0], p=0.0002. Source: `2026-04-13-bridge-phase3-test-results.md`.

**Dev-only number (E0/E1 comparison):** E1 Δ −9.0 pp [−16.0, −3.0], p=0.016. Source: `2026-04-04-bridge-phase2-dev-results.md`.

All section files have been updated to use the test-set number as the primary bridge result. The E1 −9.0 pp remains as a dev-only supporting comparison. The previous "7–9 pp" range that mixed E0 dev and E1 dev is no longer used.

**Verdict: Updated.** Re-verify cross-section consistency after full_paper.md is re-synced.

### Probe heads AUROC (target: 1.0)

| Location | Quoted value | Status |
|---|---|---|
| Abstract (L11) | `AUROC 1.0` | OK |
| Introduction (L26) | `AUROC 1.0` | OK |
| Section 4.1 (L184) | `AUROC $1.0$` (top two heads) | OK |
| Section 4.3 (L262) | `AUROC $= 1.0$` | OK |
| Table 4 (L281) | `AUROC $1.0$ / $1.0$ / ...` | OK |
| Section 4.3 interpretation (L290) | `AUROC $= 1.0$` | OK |
| Section 7.1 (L698) | `AUROC 1.0` | OK |
| L760 (Limitation L2) | implicit via probe-head reference | OK |

**Verdict: Consistent.**

### D7 causal effect (target: -9.0 pp)

| Location | Quoted value | Status |
|---|---|---|
| Section 4.3 results (L272) | `$-9.0$ pp $[-12.2, -5.8]$` | OK |
| Table 4 (L282) | `$-9.0$ pp $[-12.2, -5.8]$` | OK |
| Table 5 (L319) | `$-9.0$ pp $[-12.2, -5.8]$` | OK |

**Verdict: Consistent.**

### Jailbreak specificity slope difference (target: +2.77 pp/alpha)

| Location | Quoted value | Status |
|---|---|---|
| Claim ledger (L114) | `+2.77 pp/$\alpha$ [1.17, 4.42]` | OK |
| Section 5.1 (L338) | `+2.77 pp/$\alpha$ [1.17, 4.42]` | OK |
| Section 6.2 (L475) | `$+2.77$ pp/$\alpha$ $[+1.17, +4.42]$` | OK |
| Section 6.5 (L655) | `$+2.77$ pp/$\alpha$ $[+1.17, +4.42]$` | OK |

**Verdict: Consistent.**

### POTENTIAL ISSUE: Table 2 JailbreakBench H-neuron effect

Table 2 (L96) reports the H-neuron jailbreak effect as `+7.6 pp [3.6, 11.6]`. This is the raw alpha=0 to alpha=3 difference. Elsewhere in the paper, the jailbreak H-neuron effect is reported as a slope: `+2.30 pp/$\alpha$` (Sections 5.1, 6.2, 6.5). These are *different statistics* for the same experiment -- one is the total delta, the other is the per-alpha slope. Both can be correct (+2.30 * 3 = 6.9, which is close to but not identical to 7.6 due to non-linearity). However, Table 2 does not identify the comparison as "alpha=0 vs alpha=3" or clarify the relationship to the slope reported in the body.

**Recommendation:** Add a footnote to Table 2 or change the column header from "Observed H-neuron Effect" to "Observed H-neuron Effect (alpha=0 vs alpha=3)" to disambiguate from the slope metric used in the body text.

---

## 2. Term Consistency

### "strict harmfulness rate" vs "csv2_yes"

| Location | Term used | Status |
|---|---|---|
| Table 1 (L70) | "Strict harmfulness rate (graded)" | Paper-facing: OK |
| Section 4.3 setup (L266) | "strict harmfulness rate (csv2\_yes: ...)" | Defines the mapping: OK |
| Section 4.3 results (L270, L272) | "csv2\_yes" inline: `36 csv2\_yes`, `72/500` | **Internal jargon leaks into body text** |
| Section 6.1 (L426) | "strict harmfulness rate" | Paper-facing: OK |
| Section 6.1 (L435) | `$\texttt{csv2\_no}$` | **Internal jargon** |
| Section 6.2 (L466) | "H-neuron strict harmfulness slope" | Paper-facing: OK |
| Section 6.4 (L607) | `$\texttt{csv2\_yes}$` rate | **Internal jargon** |

**Assessment:** The paper defines the mapping at L266 ("strict harmfulness rate (csv2\_yes: ...)"), which is good. However, after this definition, several places still use `csv2_yes` and `csv2_no` directly instead of "strict harmfulness rate" or "safe." The usage at L435 (`csv2_no`) is the most jarring since it appears in the truncation discussion without redefinition. Section 6.4 (L607) uses `csv2_yes` in a technical pipeline-contamination discussion where code-level labels are arguably appropriate.

**Recommendation:** Replace standalone `csv2_yes`/`csv2_no` in running text with paper-facing terms, except in code-level pipeline discussions (Section 6.4) where the internal label is the point.

### "gradient-based causal intervention" vs "D7"

| Location | Term used | Status |
|---|---|---|
| Abstract (L11) | "gradient-based causal head selection" | Paper-facing: OK |
| Introduction (L22) | "gradient-based causal head selection" | Paper-facing: OK |
| Section 2.4 (L113) | "Gradient-based causal selection" (comment has "D7") | Comment OK, visible text OK |
| Section 4.3, 4.4 | "Gradient-ranked selection/heads" | Paper-facing: OK |
| Section 6.1 (L426) | "the D7 causal pilot" | **Internal code name in body text** |
| Section 6.1 (L432) | "the full-generation D7 confirmatory run" | **Internal code name in body text** |
| Limitation L6 (L764) | "The D7 result (Section 4.4)" | **Internal code name in body text** |
| Appendix placeholder (L823) | "D7 full comparison" | **Internal code name** |

**Assessment:** "D7" appears 5 times in visible body text. It was used as the internal experiment name but is not defined for readers. Section 6 and the limitations table use it as shorthand.

**Recommendation:** Replace "D7" with "gradient-based causal" or "gradient-ranked" in all reader-facing text. If brevity is needed, define "D7" parenthetically at first use.

### "magnitude-ranked neuron selection" vs "L1" vs "H-neuron scaling"

| Location | Term | Status |
|---|---|---|
| Section 4.1 (L180) | "Magnitude-ranked neurons" | OK |
| Section 4.2 (L200) | "Magnitude-ranked neuron intervention" | OK |
| Section 4.1 (L182) | "L1 logistic regression probe" (SAE context) | This is L1 regularization, not a naming conflict |
| Section 5.1 (L338) | "H-neuron scaling (magnitude-ranked selection, 38 neurons)" | OK, defines equivalence |
| Limitation table (L759) | "L1" as "Limitation 1" label | Potential collision with "L1 regularization" |

**Assessment:** The paper uses "magnitude-ranked" consistently for the neuron selection method and "H-neuron scaling" for the intervention. "L1" appears in two distinct senses: L1 regularization (Section 4.1, SAE probe) and Limitation L1 (Section 8). These are in sufficiently different contexts that collision is unlikely, but it is worth noting.

**Recommendation:** No change needed. The paper's usage is clear in context.

---

## 3. Claim Escalation Audit

### Abstract claims vs. body evidence

| Abstract claim | Body evidence | Escalated? |
|---|---|---|
| "matched or even perfect readout quality failed to predict steering utility" | Section 4.2 (matched AUROC, divergent steering); Section 4.3 (AUROC 1.0, null intervention) | **No escalation.** Abstract accurately reflects body. |
| "SAE features with AUROC 0.848 produced null steering alongside H-neurons with AUROC 0.843 that achieved +6.3 pp compliance gain" | Section 4.2 results exactly match. | **No escalation.** |
| "probe-ranked heads with AUROC 1.0 produced null intervention" | Section 4.3 results exactly match. | **No escalation.** |
| "ITI improved answer selection by +6.3 pp MC1 but reduced open-ended factual accuracy by 7-9 pp" | Section 5.2 (+6.3 pp MC1) and Section 5.3 (-7.0 pp E0, -9.0 pp E1). | **No escalation.** |
| "dominant generation failure was not refusal but confident substitution of semantically nearby wrong entities" | Section 5.3 reports 5/10 flips as substitution. 5/10 = 50% is "dominant single mode" but leaves 50% as other modes. | **Borderline.** Abstract says "dominant failure" without the "single" qualifier used in §5.3. The body says "dominant *single* failure mode" (L358), acknowledging that the remaining 5 flips had other causes. The abstract's phrasing could be read as implying substitution was the majority of all failures, which it is at 50%. This is technically accurate but the absence of "single" subtly strengthens the claim. |

### Conclusion claims vs. body evidence

| Conclusion claim | Body evidence | Escalated? |
|---|---|---|
| "The heuristic failed repeatedly" | Documented across §4.2, §4.3, §5.1-5.3 | **No escalation.** |
| "matched or even perfect readout quality did not reliably identify useful steering targets" | §4.2 + §4.3 | **No escalation.** |
| "successful interventions were narrow in scope" | §5.1 (BioASQ null), §5.2 (SimpleQA harm), §5.3 (bridge harm) | **No escalation.** |
| "measurement choices materially altered the apparent scientific conclusion" | §6.1-6.4 | **No escalation.** |
| "readout quality is a necessary but insufficient condition" | The body does not demonstrate *necessity* -- it demonstrates insufficiency. The "necessary" part is assumed from prior literature (Section 3.1). | **Minor escalation.** The body evidence only shows insufficiency. The conclusion's "necessary but insufficient" framing adds a claim (necessity) not directly tested. However, Section 9 frames this as coming from the combined prior literature + this study, which is defensible. |

**Overall verdict:** Claim escalation is minimal. Two borderline items noted above.

---

## 4. Missing Cross-References

### Section heading format inconsistency

**ISSUE:** Section 6 heading (L387) uses `# Section 6: Measurement Choices Change the Scientific Conclusion` while all other sections use the format `# N. Title` (e.g., `# 4. Case Study I: From Localization to Control`). This is a formatting inconsistency, not a cross-reference error.

**Recommendation:** Change to `# 6. Measurement Choices Change the Scientific Conclusion` for consistency.

### Cross-reference accuracy

| Reference | Target | Status |
|---|---|---|
| Introduction L24: "§6" | Section 6 (measurement) | OK |
| Introduction L26: "§4" | Section 4 (localization) | OK |
| Introduction L28: "§5" | Section 5 (control/externality) | OK |
| Introduction L30: "§5.3" | Section 5.3 (bridge) | OK |
| Introduction L32: "§7" | Section 7 (framework) | OK |
| Table 1 L70: "§6" | Section 6 | OK |
| Section 4.3 L256: "Section 4.2" | Section 4.2 | OK |
| Section 4.4 L300: "Section 4.3" | Section 4.3 | OK |
| Section 5.1 L338 footnotes | Internal data paths | N/A |
| Section 6 preamble L399: "Section 4", "Section 5" | Correct | OK |
| Section 7.1 L696: "§6.1-6.3" | Section 6.1-6.3 | OK |
| Section 7.1 L698: "§4.2", "§4.3" | Sections 4.2 and 4.3 | OK |
| Section 7.1 L700: "§5.3", "§5.1" | Sections 5.3 and 5.1 | OK |
| Rec 1 L709: "§4.2", "§4.3" | Correct | OK |
| Rec 2 L712: "§5.2-5.3", "§6.2" | Correct | OK |
| Rec 3 L715: "§4.4" | Section 4.4 | OK |
| Rec 4 L718: "§6.3" | Section 6.3 | OK |
| Rec 5 L721: "§5.3" | Section 5.3 | OK |
| Limitation L2 (L760): "§4.2" | Section 4.2 | OK |
| Limitation L4 (L762): "§4.2" | Section 4.2 | OK |
| Limitation L5 (L763): "§5.3" | Section 5.3 | OK |
| Limitation L6 (L764): "§4.4" | Section 4.4 | OK |
| Limitation L8 (L766): "§6" | Section 6 | OK |

**Verdict: All cross-references are accurate.** No broken or misdirected references found.

### Missing cross-references (potential additions)

- Section 4.4 (L298-310) discusses caveats about gradient-based selection but does not forward-reference §8 Limitation L6, which covers the same concern (missing random-head control). Adding a "see also L6 in §8" would be helpful.
- ~~Section 5.3 (L382) has an [UNCERTAINTY] tag about bridge dev-set limitation but does not cross-reference §8 Limitation L5, which addresses the same issue.~~ **Resolved 2026-04-13:** Section 5.3 rewritten with test-set data; [UNCERTAINTY] tag to be removed from full_paper.md.

---

## 5. [UNCERTAINTY] Tag Inventory

### Tag 1: Appendix reference not yet assigned (L186)

> "We defer detailed analysis to Appendix [UNCERTAINTY: appendix reference not yet assigned]."

**Context:** Verbosity confound analysis and N4288 artifact audit.
**Addressed in §8?** Not directly. L9 in the limitation table references "answer-token confound audit" but this is about position artifacts, not verbosity confounds. The N4288 issue is not listed as a limitation.
**Status:** **Open placeholder.** Must be resolved before submission by assigning an appendix letter.

### Tag 2: Appendix letter (L192)

> "[^fn-strat-assessment]: See Appendix [UNCERTAINTY: appendix letter] for verbosity confound analysis and N4288 artifact audit."

**Context:** Footnote referencing the same appendix as Tag 1.
**Addressed in §8?** Same as Tag 1.
**Status:** **Open placeholder.** Duplicate of Tag 1; both resolve when appendix letter is assigned.

### Tag 3: Bridge dev-set limitation (L382) — **RESOLVED 2026-04-13**

> Original tag referenced dev-set limitation (n=100, test split not yet used, 10-flip taxonomy).

**Resolved by:** Phase 3 test-set results (n=500, CI excludes zero, p=0.0002). Section 5.3 rewritten with test-set numbers. L5 in §8 replaced with "failure-mode coding is single-rater" (the remaining bridge limitation). The [UNCERTAINTY] tag in full_paper.md should be removed or replaced during full_paper.md sync.
**Status:** **Resolved.**

### Tag 4: StrongREJECT judge-model confound (L574-577)

> "[UNCERTAINTY: StrongREJECT used GPT-4o-mini while the other three evaluators used GPT-4o. Sub-score analysis suggests the bottleneck is the formula's treatment of the refused flag, not model capability (Section 7 of the dev-set report), but the confound has not been experimentally removed.]"

**Addressed in §8?** Partially. Limitation L8 (L766) covers "Evaluator uncertainty remains live" and discusses holdout compression. The specific judge-model confound (GPT-4o-mini vs GPT-4o) is mentioned in the claim ledger (L115: "judge-model confound not yet removed") but is not explicitly listed as a limitation row in Table 3.
**Status:** **Partially addressed.** The GPT-4o-mini confound deserves explicit mention in L8 or its own limitation row, since it represents an unresolved experimental confound.

### Tag 5: Four-stage framework validation scope (L746)

> "[UNCERTAINTY: The four-stage framework is presented as a methodological synthesis, not an ontological claim about the structure of intervention science. Some evidence lines touch multiple stages simultaneously. The framework is validated by this case study in Gemma-3-4B-IT; its utility for other models and intervention families remains to be tested.]"

**Addressed in §8?** Yes. Limitation L1 (L759) covers the single-model scope constraint. Section 8.1 (L773) explicitly states the framework's validation is limited to this case study.
**Status:** **Properly addressed.**

### Summary table

| # | Location | Content | Addressed in §8? | Action needed |
|---|---|---|---|---|
| 1 | L186 | Appendix reference not assigned | No | Assign appendix letter |
| 2 | L192 | Appendix letter (duplicate of #1) | No | Assign appendix letter |
| 3 | L382 | Bridge dev-set limitation | ~~Yes (L5)~~ **Resolved 2026-04-13** — test set run, L5 replaced | Remove [UNCERTAINTY] tag from full_paper.md |
| 4 | L574 | StrongREJECT judge-model confound | Partially (L8 general) | Add GPT-4o-mini confound to L8 or new row |
| 5 | L746 | Framework validation scope | Yes (L1) | None |

---

## 6. Additional Issues Found

### 6a. Section heading format inconsistency

Section 6 uses `# Section 6: Title` while all other sections use `# N. Title`. See Section 4 above.

### 6b. "D7" internal code name in body text

Five instances of the internal experiment name "D7" appear in reader-facing text. See Section 2 above.

### 6c. Table numbering

The paper uses Tables 1-5 and "Table 3" appears twice:
- Line 208: "Table 3 -- FaithEval Compliance by Intervention Method and Scaling Factor"
- Line 755: "Table 3. Limitation inventory."

These are both labeled "Table 3." This is a numbering collision.

**Recommendation:** Renumber. The limitation table should be Table 6 (or whatever follows Tables 1-5 sequentially).

### 6d. Minor: "csv2_yes" definition placement

The term `csv2_yes` is first parenthetically defined at L266 (Section 4.3), but the Table 1 entry (L70) already uses "Strict harmfulness rate (graded)" without the code-level name. The definition should ideally appear at first conceptual use (Table 1 or Section 2.3) rather than buried in a subsection setup paragraph.

### 6e. Appendix placeholder is content-free

Line 823: "[Appendix content to be added: D7 full comparison, 4288/detector deep dive, bridge construction details, evaluator companion note, related work positioning matrix.]"

This is an open TODO. The two [UNCERTAINTY] tags (1 and 2) reference this unwritten appendix.

---

## Summary of Required Actions

**Must fix before submission:**
1. Resolve duplicate Table 3 numbering (L208 and L755).
2. Assign appendix letters to resolve [UNCERTAINTY] tags 1 and 2 (L186, L192).
3. Replace "D7" with paper-facing language in body text (5 instances).
4. Fix Section 6 heading format: `# Section 6:` -> `# 6.`

**Should fix:**
5. Add GPT-4o-mini judge-model confound explicitly to Limitation L8 or new row.
6. Replace `csv2_yes`/`csv2_no` with paper-facing terms in running text (L270, L435).
7. Disambiguate Table 2 jailbreak H-neuron effect (+7.6 pp) from the slope (+2.30 pp/alpha) used in the body.

**Consider:**
8. Add "single" qualifier to abstract's "dominant failure" -> "dominant single failure mode" to match §5.3 precision.
9. Add cross-references from §4.4/§5.3 [UNCERTAINTY] tags to their corresponding §8 limitation entries.

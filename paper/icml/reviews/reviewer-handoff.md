# Reviewer Handoff: ICML 2026 MI Workshop Submission

**Paper:** *Finding the Signal, Losing the Wheel: The Paradox of Internal Readouts in Gemma-3-4B-IT*
**Track:** Long paper (8-page limit, excl. refs/appendix)
**Deadline:** May 8, 2026 AOE
**Portal:** OpenReview — `ICML.cc/2026/Workshop/Mech_Interp`
**Source of truth for all numbers:** `paper/icml/number_provenance.md` (reviewer-facing ledger for the current workshop manuscript)
**This submission:** `paper/icml/main.tex` → `paper/icml/main.pdf` (~6 pages main body + 1.5 pages refs + 2 pages appendix)

---

## A. Purpose of This Review

This workshop paper condenses a full-length manuscript (~25 equivalent pages) into an 8-page ICML workshop submission. The reviewer should evaluate:

1. **Formatting compliance** against ICML 2026 submission guidelines and the MI Workshop CFP.
2. **Result selection quality** — whether the condensation preserved the strongest, most novel results and whether anything critical was cut that should have been kept, or anything weak was kept that should have been cut.
3. **Narrative coherence** — whether the compressed argument reads clearly to a mech-interp audience without the full paper's supporting apparatus.
4. **Anonymization** — whether any identifying information survived.

---

## B. ICML 2026 Formatting Compliance Checklist

Check each item against `paper/icml/icml-submission-guidelines.md` (the full ICML 2026 author instructions).

| # | Requirement | Expected | Status |
|---|---|---|---|
| 1 | Submission is PDF | `main.pdf` exists | Verify |
| 2 | Main body ≤ 8 pages (excl. refs + appendix) | Main body ends ~p6, refs start ~p6 | Verify exact page boundary |
| 3 | Appendix is in the same file (not separate) | Appendix A–E follow refs in same PDF | Verify |
| 4 | File size < 10MB | ~324KB | Verify |
| 5 | 10pt Times font throughout | Uses `icml2026.sty` which enforces this | Verify body text size |
| 6 | Only Type-1 fonts | `pdffonts main.pdf` should show only Type 1 | Verify — prior check passed |
| 7 | No author information visible | `\icmlauthor{Anonymous}{}`, no affiliations | Grep for author name, institution |
| 8 | No acknowledgements in submission | None present | Verify |
| 9 | Figure captions below figures | All 3 figures have captions below | Verify |
| 10 | Table captions above tables | All 5 main-body tables have captions above | Verify |
| 11 | References are complete with page numbers where possible | 19 references, all with venue + year | Spot-check 3–4 entries |
| 12 | Title has content words capitalized | "Finding the Signal, Losing the Wheel: The Paradox of Internal Readouts in Gemma-3-4B-IT" | Verify |
| 13 | Abstract is 4–6 sentences, single paragraph | Currently 6 sentences | Verify count |
| 14 | Impact statement present | `\section*{Impact Statement}` on p6 | Verify |
| 15 | No URLs revealing author identity | No GitHub URLs, no institutional URLs | Grep for URLs |
| 16 | Running head on pages 2+ | `\icmltitlerunning{Finding the Signal, Losing the Wheel}` | Verify in PDF |
| 17 | Two-column format, correct margins | Handled by `icml2026.sty` | Visual check |
| 18 | No compressed vertical spaces | No `\vspace` hacks in main body | Verify |
| 19 | Citations in APA/natbib format | Uses `icml2026.bst` + `natbib` | Spot-check |
| 20 | Self-citations in third person | No self-citations (single-author, first submission) | Verify |

---

## C. MI Workshop-Specific Compliance

Source: `https://mechinterpworkshop.com/cfp/`

| # | Requirement | Status |
|---|---|---|
| 1 | ICML 2026 LaTeX format (preferred) | Uses `icml2026.sty` — compliant |
| 2 | Long paper ≤ 8 pages (excl. refs + appendix) | ~6 pages main body — compliant |
| 3 | Double-blind (no identifying info) | Anonymous author — verify grep |
| 4 | OpenReview submission | Not yet submitted — portal ready |
| 5 | Reciprocal reviewing: ≥1 reviewer registered per submission | Reminder to author: register at reviewer interest form |
| 6 | Fits workshop topics | Topics 3 (practical applications), 4 (safety/monitoring), 7 (conceptual frameworks) |
| 7 | Non-archival status acknowledged | No conflicts — paper is also under review at BlueDot |
| 8 | No accepted archival venue papers (besides ICML 2026) | Not accepted elsewhere — compliant |
| 9 | Check for contributor names, GitHub usernames, HuggingFace usernames | Grep entire PDF text |

---

## D. Result Selection Audit

This is the most important part. The full paper has ~8 distinct empirical findings. The workshop version includes 5 and cuts 3. The reviewer should assess whether this selection is optimal.

### D.1 Results INCLUDED — are these the right ones?

| Result | Full Paper § | Workshop § | Strength | Novelty | Verdict |
|---|---|---|---|---|---|
| **FaithEval matched readout comparison** (AUROC 0.843 vs 0.848, neuron dose-response +2.09 pp/α, SAE null, delta-only confirmation, 8-seed controls, paired slope difference p<0.001) | §4.2 | §3.2 | Very strong — matched design, multiple confound controls, permutation test | High — first matched cross-target-type comparison on same behavioral surface | **Must keep — anchor result** |
| **BioASQ portability null** (flat endpoint despite 1,339/1,600 perturbation) | §5.1 | §4.1 | Moderate — well-powered null (n=1,600, MDE ~2pp) | Moderate — shows surface-locality but not surprising given prior literature | **Keep — clean portability test** |
| **ITI MC vs generation divergence** (+6.3 MC1, -1.8 SimpleQA, attempt collapse) | §5.2 | §4.2 | Strong — both CIs exclude zero, clear behavioral contrast | Moderate — Li et al. (2023) already showed this divergence directionally | **Keep — essential for externality argument** |
| **TriviaQA bridge wrong-entity substitution** (-5.8pp, 43 R→W, 30 wrong-entity, 5 examples, McNemar p=0.0002) | §5.3 | §4.3 | Very strong — frozen test protocol, statistical significance, behavioral taxonomy | **Very high — novel failure-mode diagnosis, no prior work characterizes this** | **Must keep — most memorable result** |
| **Jailbreak measurement sensitivity** (truncation, binary +3.0pp CI∋0 vs graded +2.30 pp/α CI⊄0, evaluator holdout tie) | §6 | §5 | Strong — same outputs produce different conclusions under different evaluators | High — demonstrates measurement as separable empirical gate | **Must keep — essential for 4-stage framework** |

### D.2 Results CUT — should any be restored?

| Result | Full Paper § | Why Cut | Risk of Cutting | Reviewer Question |
|---|---|---|---|---|
| **Jailbreak selector pilot** (probe AUROC 1.0 but inert, gradient-ranked -13pp) | §4.3 | Heavily caveated in full paper; n=100 pilot; jailbreak is noisier surface | Low — FaithEval comparison is cleaner. Workshop audience may see the pilot as undercooked evidence. | **Is the localization→control case persuasive enough with FaithEval alone, or does the probe-head selector add meaningful independent evidence?** |
| **Full-500 jailbreak comparator panel** (gradient vs probe vs 2-seed random) | §4.4 | Mixed-ruler panel, error-bearing comparators, generation-cap artifacts; full paper itself calls this "supporting, not flagship" | Very low — the full paper already doesn't trust this as a clean comparison | No action needed |
| **Schema-drift / pipeline hygiene** (v3 normalization bug inflating rates from 18.8% to 52.2%) | §6.4 | Specific implementation bug; important as cautionary tale but not a scientific finding | Low — the broader measurement point survives without this detail | No action needed |
| **Jailbreak severity escalation** (V=3 quadruples, S=4 nearly triples, payload share rises) | §6.2 subsection | Graded detail that supports binary-vs-graded contrast already included | Low — the slope contrast (+3.0 binary vs +2.30 graded) already makes the point | No action needed |
| **H-neuron jailbreak ablation/amplification split** (76% ablation recovery at α=0→1) | §5.1 | Interesting mechanistic detail but not central to the narrative | Low — adds texture but doesn't change the claim | No action needed |
| **E1 bridge variant** (dev-only, K=8, -9.0pp, same 10 questions damaged) | §5.3 | Dev-only (n=100), not run on test set | Low — the E0 test result is self-sufficient | No action needed |

### D.3 Critical Question for the Reviewer

The workshop version has ~2 unused pages of main-body budget (6 of 8). Three options:

1. **Add the jailbreak selector comparison back** (~0.5 pages). This would give the localization→control claim two independent lines of evidence (FaithEval + JailbreakBench), but the jailbreak evidence is noisier and more caveated.

2. **Expand existing sections.** The measurement section could include the severity escalation detail (V=3, S=4) to make the binary-vs-graded contrast more vivid. The bridge section could include the E1 dev reproducibility.

3. **Leave the headroom.** Reviewers may prefer a tight 6-page paper that doesn't pad. The 8-page limit is a ceiling, not a target.

**The reviewer should recommend one of these options or propose an alternative.**

---

## E. Narrative Coherence Audit

Read the paper front-to-back as a mech-interp researcher who knows SAEs, probing, and activation steering. Check:

| # | Question | What to look for |
|---|---|---|
| 1 | Does the abstract convey the core finding in ≤30 seconds? | Three anchors should be recognizable: matched readout null, bridge wrong-entity, measurement sensitivity |
| 2 | Does the introduction set up the 4-stage framework without being heavy-handed? | The enumerated list should feel natural, not forced |
| 3 | Is the gap statement in Related Work credible? | "First matched cross-target-type comparison" — is this actually true? Check against Arad et al. (2025) and Wu et al. (2025) |
| 4 | Does §3 (Localization) lead with the setup and deliver the punchline efficiently? | Reader should hit the +1.93 slope difference within 1 column of starting the section |
| 5 | Does the §3.3 Synthesis paragraph feel earned or defensive? | The "positive counterexample is important" framing should preempt the obvious objection without sounding apologetic |
| 6 | Does §4 (Externality) build to the bridge result or bury it? | BioASQ → ITI MC/gen → bridge should feel like increasing revelatory power |
| 7 | Is the wrong-entity substitution table compelling? | 5 examples should show a clear semantic-neighborhood pattern, not random errors |
| 8 | Does §5 (Measurement) carry its weight or feel like a methods section? | Should read as a finding ("the measurement changed the conclusion"), not as a methods complaint |
| 9 | Does the framework (§6) feel like a contribution or a restatement? | Table 3 (checklist) should be independently citable |
| 10 | Is the conclusion appropriately scoped? | Should not overclaim (no "all readouts fail") or underclaim (no "we found some things") |

---

## F. Specific Items to Flag

### F.1 Potential Weaknesses a Reviewer Might Raise

1. **Single model.** All results are from Gemma-3-4B-IT. The paper acknowledges this (L1) but a reviewer may still discount the framework's generality. Check that the title includes the model name and that the paper does not overclaim.

2. **Matched-readout confound.** SAE features and neurons differ in more than readout quality (operator form, layer coverage, feature granularity). The paper acknowledges this (§3.3, L2) but the matched AUROC is doing a lot of inferential work. Check that the claim is scoped correctly ("readout quality alone did not suffice" rather than "readout quality is the only thing that matters").

3. **~~Single-rater~~ Dual-rated failure coding (closed 2026-04-21).** The 31/43 wrong-entity share is now dual-rated with pre-registered rubric and IRR (raw agreement 96.5%, κ = 0.90, AC1 = 0.96, two disagreements adjudicated under a pre-frozen rule); Rater B is an LLM judge, so the paper frames this as a sensitivity check rather than strong-form human–human IRR. See [`../reports/2026-04-21-bridge-irr-review.md`](../reports/2026-04-21-bridge-irr-review.md).

4. **No capability battery.** The paper does not report a generic capability check (e.g., MMLU, HellaSwag) to verify that interventions are not simply degrading the model globally. The BioASQ flat endpoint partially addresses this, but a reviewer might still ask for it.

5. **The framework is descriptive, not predictive.** The 4-stage audit tells you what to check, not what will fail. A reviewer might want a predictive theory of when readouts will vs. won't work. Check that the paper positions the framework as a practical decomposition, not a predictive theory.

### F.2 Potential Strengths to Highlight

1. **The matched FaithEval comparison is unusually clean.** Same model, same behavioral surface, same alpha grid, matched AUROC, delta-only confound control, 8-seed random controls. This is stronger than most detection-steering comparisons in the literature.

2. **The wrong-entity substitution taxonomy is novel.** No prior work we are aware of provides a behavioral diagnosis of ITI transfer failure at this granularity. The pattern (same-director, same-crash, same-show) is vivid and easy to remember.

3. **The measurement section turns a methods concern into a finding.** Rather than hiding evaluator disagreement, the paper reports it as the result. This is unusual and arguably the right approach for the MI Workshop audience.

4. **The paper is appropriately scoped.** It does not claim a universal failure of detector-based targets. It provides positive counterexamples (H-neurons work, refusal direction works). The negative result is specific and well-controlled.

---

## G. Cross-Reference Against Workshop Evaluation Standards

From the CFP: "Strong empirical works should: (i) articulate specific falsifiable hypotheses with supporting evidence, or (ii) convincingly demonstrate clear practical benefits over well-implemented baselines."

| Criterion | Paper's Claim | Evidence |
|---|---|---|
| Falsifiable hypothesis | "Matched readout quality predicts matched steering utility" — tested and rejected on FaithEval | AUROC match, slope divergence, permutation p < 0.001 |
| Falsifiable hypothesis | "Answer-selection gains transfer to nearby generation" — tested and rejected on TriviaQA bridge | -5.8pp, McNemar p = 0.0002, wrong-entity taxonomy |
| Falsifiable hypothesis | "Binary and graded evaluators agree on intervention direction" — tested and rejected on jailbreak | Binary CI∋0, graded CI⊄0, holdout tie between v3 and StrongREJECT |
| Practical benefit | The 4-stage audit framework and Table 3 checklist | Descriptive framework, not a method — practical utility is in structuring evaluation |

The workshop also explicitly welcomes "rigorous negative results" and "position pieces clarifying complex topics and debates." This paper qualifies on both counts.

---

## H. Pre-Submission Checklist

- [ ] `pdffonts main.pdf` shows only Type-1 fonts
- [ ] Main body ≤ 8 pages before `\bibliography`
- [ ] `grep -ri "hugo\|nguyen\|github" main.tex main.pdf` returns nothing
- [ ] All CIs, effect sizes, and sample sizes match `paper/icml/number_provenance.md`
- [ ] Impact statement is present
- [ ] File size < 10MB
- [ ] Figures are legible at column width (print at 100% and check)
- [ ] References compile cleanly (no `?` in PDF)
- [ ] Register as reciprocal reviewer at `https://forms.gle/LRpywTjuAoFqbWaB7`
- [ ] Upload to OpenReview before May 8 AOE

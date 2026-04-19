# Review report for `main.tex` — ICML 2026 Mechanistic Interpretability Workshop submission

Reviewed inputs:
- `main.tex`
- `full_paper.md`
- `number_provenance.md`
- `2026-04-11-strategic-assessment.md`

## Bottom line

This is already close to a strong **long-form workshop paper**. The current TeX draft has the right overall instinct: it demotes weaker side stories, keeps the three best anchors, and argues a disciplined negative-result / methodology thesis rather than pretending to present a new steering method.

The strongest version of this paper, **as of April 2026**, is **not**:

- “we found the real truthfulness mechanism,”
- “SAEs do not work,”
- “measurement is broken,” or
- “activation steering fails.”

It is:

> **Held-out internal readout quality is not a sufficient target-selection heuristic for steering.**
> In Gemma-3-4B-IT, the inferential chain from readout → control → transfer repeatedly breaks, and the breakpoints are empirically separable.

That thesis is still publishable and useful in April 2026, but only if it is framed precisely against the now-current literature and kept tightly evidence-ranked.

My main recommendation is to keep the paper centered on **three anchor case studies**:
1. **FaithEval H-neurons vs SAE features** — the cleanest matched cross-family readout/control dissociation.
2. **TruthfulQA MC vs TriviaQA bridge generation** — the cleanest control/externality dissociation, with wrong-entity substitution as the sharpest behavioral diagnosis.
3. **Jailbreak evaluation audit** — the cleanest measurement/verdict dissociation.

Everything else should either remain appendix-only or appear only as supporting provenance.

---

## Status tracker (2026-04-19 integration pass)

This block was added by the integration pass on 2026-04-19. It tracks every actionable recommendation in this review against its current status in `main.tex` / `references.bib`. Use it as the starting point for follow-up sessions.

### Tier-1 top-5 (§7) — all DONE

1. **Title change to thesis-first** — DONE. `main.tex:1` (comment), `main.tex:13` (running), `main.tex:18–19` (main title). New title: *Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT*. Zero remaining occurrences of `paradox` / `Finding the Signal` / `Losing the Wheel`.
2. **Wang et al. (ICLR 2026) + novelty reframe** — DONE. Cited in related work prose at `main.tex:99` (Kendall's τ_b ≈ 0.298) and in the reframed gap statement at `main.tex:104`. Bib entry `wang2026interpretability` already present (`references.bib:160–167`).
3. **Construct-surfaces sentence in intro** — DONE as `\textbf{Construct surfaces.}` block at `main.tex:82–83`. All four benchmark families explicitly typed (faithfulness, answer selection, open-ended generation, harmfulness).
4. **Bridge disclaimer (behavioral diagnosis, not circuit claim)** — DONE at `main.tex:213`, appended to the wrong-entity-substitution paragraph in §4.
5. **Remove gradient-ranked pilot sentence from §5** — DONE. Sentence deleted at the old `main.tex:255`. Paragraph still makes the truncation argument without the orphaned D7 reference.

### Additional surgical edits landed this pass — DONE

- **H1/H2/H3 falsifiable-hypothesis block** (§3.A) — DONE at `main.tex:85–90`. Each hypothesis maps to a specific audit gate and is paired with its rejection surface.
- **§3 retitled** (`Localization Does Not Predict Control` → `Matched Readout Quality Does Not Guarantee Control`) — DONE at `main.tex:114`. `\label{sec:localization}` preserved; no downstream `\ref` edits required.
- **CETT acronym removed from main body** — DONE at `main.tex:120` (`A sparse H-neuron probe following \citet{gao2025hneurons}...`) and `main.tex:127` (`The 38 probe-selected neurons...`). Appendix mention of `CETT neuron (L20:N4288)` intentionally left, since the review scoped removal to main text.
- **§4 metric-portability softening** (`portability limit on the metric...` → `flat accuracy endpoint coexists with substantial output perturbation...`) — DONE at `main.tex:179`.
- **§5 CSV-v2 / CSV-v3 semantic glosses at first mention** — DONE at `main.tex:274` (`A graded harmfulness rubric (CSV-v2)...`) and `main.tex:284` (`A taxonomy-based graded rubric (CSV-v3)...`).
- **§6 soften literature-wide claim** — DONE at `main.tex:352` (`In our reading of the recent steering literature, target-surface control is reported more often than matched negative controls, cross-surface tests, or evaluator audits.`).
- **§8.B CAST / conditional-steering future-work sentence** — DONE at `main.tex:378`, between R5 and Limitations. Bib entry `lee2025cast` already present (`references.bib:169–176`).
- **Limitations: April 2026 concurrent-work context** — DONE at `main.tex:387`. Paper no longer implicitly claims first demonstration of interpretability/utility divergence; distinctness is explicitly repositioned to cross-family matching on real behavioral surfaces.
- **Abstract: expand SAE acronym + cross-family novelty cue** — DONE at `main.tex:35` (`matched cross-family FaithEval comparison, sparse autoencoder (SAE) features...`).

### DEFERRED — valuable but out of this pass's scope

- **Substitution examples table (§4, L207–223 in original review)** — the review asks for more literal question stubs (e.g., "Which Danny Boyle film was released in 1996?"). Deferred because faithful rewriting requires access to the actual TriviaQA bridge items and could introduce fabricated wording. Follow-up needs an author with benchmark access to verify each stub against the source.
- **Spearman ρ = 1.0 / ρ = 0.18 removal** (§4 L119–129 in original review) — deferred, contingent on page-length overflow after recompile. If `make` shows the main body over 8 pages, drop these two ρ values; otherwise keep for interpretive completeness.
- **R2–R5 stylistic compression** (§4 L346–365) — same contingency. R1 is the load-bearing recommendation; R2–R5 can be compressed as a single paragraph if space pressure materializes.
- **Know Thy Judge citation strengthening in §5 measurement prose** (§3.D) — currently cited in related work via `eiras2025judge`. A follow-up could add an inline citation at the §5 evaluator-dependence paragraph to localize the meta-evaluation precedent. Low priority; §5 already cites `chen2025safer` and `souly2024strongreject`.
- **SimpleQA vs bridge hierarchy compression in §4** (§4 L173–182 in original review) — review suggests making the bridge-is-main, SimpleQA-is-supporting hierarchy more explicit. Deferred as light editorial; current prose already foregrounds bridge as the "most informative externality result."

### OUT OF SCOPE — needs tooling / artifacts / domain passes this pass cannot cover

- **Provenance ledger updates** (`paper/draft/number_provenance.md`, §5 of this review). The ledger uses section numbering from the long markdown draft, not the TeX. Mapping every headline number to its TeX section reference is a separate pass.
- **Recompilation / page-count / font-embedding / PDF-metadata checks** (§1 of this review). A background `make` was kicked off in the integration session; the user should inspect the resulting `main.pdf` for ≤8-page main body, Type-1 fonts, and clean anonymized metadata.
- **Reproducibility supplement package** (§3.G, §8.C). Prompt manifests, judge prompts, coding guide, scoring scripts, provenance ledger — artifact-assembly task, not a `main.tex` edit.
- **Anonymization scrub of supplementary materials**. `full_paper.md` and related artifacts contain `Hugo Nguyen` per the review's §1 warning; must be scrubbed before any supplement upload.
- **Bibliography venue verification beyond Wang / CAST adds**. Spot-audit at integration time confirmed `arad2025saes` is EMNLP 2025, `wu2025axbench` is ICML 2025 Spotlight, `opielka2026causality` is ICLR 2026 Poster, `eiras2025judge` is ICLR 2025 Workshop, `chen2025safer` is ACL 2025. No further changes required unless reviewers raise specific venues.
- **"What this paper contributes in 2026" standalone sentence** (§8.A). Folded into the reframed gap statement (see DONE item for Wang/reframe); a separate sentence would be redundant.
- **Separate TODO files** (`TODO_L4_interrater.md`, `TODO_LAST_Limitation 5 multi-seed.md`, `TODO_Limitations_Fixes`) — the user asked this pass to stay within the review-report scope.

### Notes for the next follow-up session

- The bib already contains `wang2026interpretability` (ICLR 2026) and `lee2025cast` (ICLR 2025 Spotlight). No author-list fetch from OpenReview was required.
- The `\label{sec:localization}` label was preserved despite the section retitle, so all internal `\ref` call sites continue to resolve. Verify by running `make` and checking for `LaTeX Warning: Reference ... undefined` lines.
- Orphan grep audits were run post-edit: zero occurrences of `paradox`, `Finding the Signal`, `Losing the Wheel`, `gradient-ranked`, `A CETT probe`, `portability limit on the metric`, or the old §3 title. Safe to continue editing.
- Net line delta after this pass: ≈ +15 added, ≈ −3 removed. Page budget risk is real but not severe; if the compile spills, start with the two DEFERRED compressions above (Spearman ρ, R2–R5).

---

## 1. Workshop fit and submission-guideline audit

### Fit to the ICML 2026 Mechanistic Interpretability Workshop

This paper is a good fit **if presented explicitly as a rigorous negative-result / benchmarking / methodological audit paper**. The workshop CFP explicitly welcomes:
- rigorous negative results,
- rigorous replications,
- critiques or compelling failed replications,
- benchmarking progress,
- interpretability for safety / monitoring / model repair,
- and work that clearly documents what the evidence does and does not support.

That is exactly the lane this paper should occupy.

Useful references:
- [ICML 2026 Mech Interp CFP](https://mechinterpworkshop.com/cfp/)
- [ICML 2026 workshop OpenReview group](https://openreview.net/group?id=ICML.cc%2F2026%2FWorkshop%2FMech_Interp&referrer=%5BHomepage%5D%28%2F%29)

### Format / policy compliance

From the current workshop CFP and ICML 2026 author instructions:

- **ICML format is accepted** and is the right choice here.
- **Short papers**: 4 pages main body. **Long papers**: 8 pages main body.
- **Long papers are held to a higher standard of rigor and depth.**
- References and appendices are unlimited, but reviewers are not expected to read them.
- Review is **double-blind**.
- The workshop is **non-archival**.
- The workshop requires **at least one reciprocal reviewer** per submission.
- Reviewers will consider **reproducibility / code / data access**.

Useful references:
- [Workshop CFP](https://mechinterpworkshop.com/cfp/)
- [ICML 2026 Author Instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
- [ICML 2026 example paper / formatting instructions](https://media.icml.cc/Conferences/ICML2026/Styles/example_paper.pdf)

### What the current draft appears to satisfy

The TeX draft appears compliant on the following points:

- Anonymous title page / no author info in `main.tex`
- Uses ICML template
- Abstract is one paragraph and six sentences
- Includes an impact statement
- No acknowledgements in the anonymous draft
- Figures and table captions are placed in the expected locations in the source
- References and appendices are separated from the main body

### What still needs explicit submission-time checking

I could not compile the paper in the exact ICML 2026 style because the submission package assets were not included with the upload, so these still need manual verification before submission:

1. **Main-body page count** after real figures and bibliography in the actual ICML style.
2. **Figure font embedding** (Type-1 fonts; no Type-3 leakage from generated PDFs).
3. **PDF metadata** and supplement metadata for anonymization.
4. **No non-anonymized supplementary artifacts**.

### Important anonymization warning

Your longer markdown draft includes your real name (`Hugo Nguyen`). Do **not** upload that file as supplementary material in its current form. The workshop and ICML rules require anonymized submission materials. If you share a longer draft, provenance file, or artifact package, it must be scrubbed for:
- author names,
- usernames,
- repository names,
- personal or institutional URLs,
- PDF metadata,
- and any self-identifying acknowledgements or internal notes.

### Recommendation on long vs short

I would keep this as a **long paper**, not a short paper, **provided**:
- the compiled main body is within 8 pages,
- you do not re-expand D7 into a co-headline,
- and the evidence hierarchy stays explicit.

The paper does enough to justify long format **if** it stays disciplined. If the compiled version spills over 8 pages, cut rhetoric before you cut evidence.

---

## 2. What is still novel and valuable in April 2026

This section is the most important strategic correction.

By April 2026, the broad claim “good internal features / interpretable features do not automatically make good steering targets” is **not new by itself**. Recent work has already weakened the naive version of that story:

- **AxBench (ICML 2025)** reports that prompting outperforms existing steering methods on its benchmark, and that SAEs are not competitive there.  
  [OpenReview](https://openreview.net/forum?id=K2CckZjNy0)

- **Arad et al. (EMNLP 2025)** show that SAE steering depends strongly on feature selection and that output-oriented feature scoring matters.  
  [ACL Anthology PDF](https://aclanthology.org/2025.emnlp-main.519.pdf)

- **Wang et al. (ICLR 2026)** find only a weak positive association between SAE interpretability and steering utility across 90 SAEs (Kendall’s τb ≈ 0.298), explicitly arguing that interpretability is an insufficient proxy for steering performance.  
  [OpenReview PDF](https://openreview.net/pdf/0d6cc4dd34c2b81d084e03c84c4f4171ef27dfc7.pdf)

- **Opiełka et al. (ICLR 2026)** show that function vectors are not format-invariant across multiple-choice vs open-ended settings.  
  [OpenReview](https://openreview.net/forum?id=LmLmhb6GEL)

That means your paper should **not** implicitly position itself as the first demonstration that “strong readouts are not enough.”

### What *is* distinctive here

Your distinctive contribution is the combination of:

1. **A matched cross-family comparison on one real behavioral surface**  
   H-neurons vs SAE features, same model, same benchmark, similar held-out AUROC, divergent steering, plus delta-only null. This is the best anchor.

2. **An externality diagnosis on realistic generation surfaces**  
   The bridge benchmark does not merely show “generation got worse.” It shows a specific failure pattern: wrong-entity substitution. That is a much better contribution than just another average-score drop.

3. **An integrated audit that separates three inferential breaks**
   - measurement → verdict,
   - readout/localization → control,
   - control → externality.

4. **A cleanly limited thesis**
   You are not claiming that internal methods never work. You are claiming that **readout quality alone is not enough to justify target selection**.

### How I would phrase the novelty claim

Something close to:

> Prior work already suggests that interpretability and steering utility can diverge, especially within SAE-centric and synthetic control settings. This paper contributes a more ecologically grounded, cross-representation case study in a single instruction model: a matched neuron-vs-SAE comparison on a real behavioral surface, a held-out generation externality diagnosis, and a measurement audit showing that evaluation choices can alter the verdict about an internal intervention.

That is much better aligned with the state of the field.

---

## 3. Highest-priority scientific/framing revisions

## A. Make the thesis more explicit and more falsifiable

The workshop CFP explicitly says strong empirical works should articulate **specific falsifiable hypotheses** and what the evidence does and does not support.

Right now the paper has the right empirical content, but the hypotheses are still somewhat implicit. I recommend making them explicit in the introduction.

Suggested structure:

- **H1.** Held-out readout quality is a useful heuristic for selecting steering targets.  
  **Result:** rejected in the matched FaithEval comparison.

- **H2.** If an intervention works on a truthfulness-adjacent surface, the gain transfers to nearby open-ended factual generation.  
  **Result:** rejected by the TruthfulQA MC vs bridge generation evidence.

- **H3.** Reasonable evaluator choices leave the scientific verdict stable.  
  **Result:** rejected in the jailbreak audit.

This would materially improve the workshop fit.

## B. Tighten the title: the current one is memorable but too metaphorical

Current title:
> *Finding the Signal, Losing the Wheel: The Paradox of Internal Readouts in Gemma-3-4B-IT*

This is stylish, but by April 2026 it undersells the actual contribution and oversells the “paradox” framing. The field already contains several papers arguing versions of interpretability/utility divergence; the contribution here is not a paradox reveal but a **target-selection discipline** argument.

Recommended title:
> **Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT**

Good alternatives:
- **From Readout to Control: Strong Internal Signals Often Fail as Steering Targets in Gemma-3-4B-IT**
- **Good Readouts, Weak Levers: Steering Dissociations in Gemma-3-4B-IT**

I would also change the running title to match.

## C. Restore a minimal construct map in the main text

The longer draft had a stronger construct map. The compressed TeX draft lost too much of that protective nuance.

This is important because reviewers can otherwise accuse the paper of mixing:
- contextual faithfulness / anti-compliance,
- constrained answer selection,
- open-ended factual generation,
- and jailbreak harmfulness.

You do not need the full table in the main body, but you do need one short sentence near the end of the introduction or study-orientation paragraph.

Recommended addition near `main.tex` L76-L80:

> We treat FaithEval as a context-faithfulness / anti-compliance surface, TruthfulQA MC as constrained answer selection, and TriviaQA bridge / SimpleQA as open-ended factual generation surfaces; all claims in the paper are surface-specific.

This is high ROI.

## D. Update the related work for April 2026

The current related work is good but not fully current enough for this exact thesis.

### Must add
1. **Wang et al., ICLR 2026**  
   Very close to your core theme. You need to show that you know this paper exists, then explain why your contribution is still distinct.

2. **A judge-robustness citation in the measurement section**  
   Either:
   - **Eiras et al., Know Thy Judge**  
     [OpenReview](https://openreview.net/forum?id=kPMfYS2ugs)
   - or **Chen & Goldfarb-Tarrant, Safer or Luckier?**  
     [arXiv HTML](https://arxiv.org/html/2503.09347v3)

3. **CAST (ICLR 2025)** in future work / discussion  
   Because your bridge result strongly suggests that **selective / conditional** interventions are the next move, not global truthfulness directions.  
   [OpenReview](https://openreview.net/forum?id=Oi47wc10sm)

### How to reframe the novelty after adding Wang et al.
Not:
> “prior work only shows this within single method families…”

But:
> “Recent SAE-centric work already suggests interpretability/utility divergence; we extend the question across representational families and onto real behavioral surfaces with explicit transfer and measurement audits.”

## E. Do not restore D7 as a co-headline

The strategic decision to demote D7 from the main story was correct.

The current paper is **stronger without D7 as a pillar**. The main paper already has enough:
- FaithEval matched dissociation,
- bridge externality diagnosis,
- jailbreak measurement audit.

D7 is still interesting, but it remains too caveated for flagship status. Keep it appendix-only or supplementary-only unless you finish a clean like-for-like comparison.

## F. Add a one-sentence disclaimer that wrong-entity substitution is a behavioral diagnosis, not an internal mechanism claim

This was present in the long draft and should come back.

Recommended sentence after `main.tex` L201:

> We treat wrong-entity substitution as a behavioral diagnosis rather than a claim about the exact internal circuit mechanism.

That one sentence improves the paper’s taste substantially.

## G. Put reproducibility on the page or in the supplement package

The workshop explicitly says reviewers will consider reproducibility, code, and/or data access.

Your **number provenance ledger is a real asset**. Very few workshop papers do this well. I would strongly recommend including, in an anonymized supplement or anonymous repo:
- evaluation manifests,
- prompt templates,
- judge prompts,
- scoring scripts,
- the number provenance ledger,
- bridge label definitions / coding guide,
- and a short “what each headline number comes from” note.

A paper making measurement claims benefits enormously from artifact discipline.

---

## 4. Line-by-line / section-by-section comments on `main.tex`

## Title and abstract

### `main.tex` L18-L19 — title
**Issue:** too metaphorical; not search-friendly; “paradox” overstates novelty in the 2026 literature context.  
**Action:** switch to a more literal title centered on detection/readout insufficiency.

### `main.tex` L33-L38 — abstract
**What works well:**
- clear empirical content,
- concrete numbers,
- good three-part structure,
- no obvious overclaim.

**What to improve:**
1. Expand the first instance of `SAE` and possibly of `H-neurons`.
2. Make the novelty less generic and more specific.
3. If space is needed, drop “neuron-minus-SAE gap excludes zero” and use the space for the **cross-family / real-benchmark** novelty cue.

**Suggested abstract emphasis order:**
1. matched readout/control dissociation,
2. bridge externality diagnosis,
3. measurement/verdict instability,
4. audit framework.

---

## Introduction

### `main.tex` L46-L48
This is a strong setup. It correctly states the practical heuristic under test.

### `main.tex` L54-L61
The four stages are introduced cleanly, but the paper would benefit from an explicit “we test three breaks in this chain” statement.

### `main.tex` L76-L80
This paragraph needs one extra sentence defining the construct surfaces. Right now the reader still has to infer too much.

---

## Related work

### `main.tex` L88-L98
Strong core, but it now needs an April 2026 update.

**Add:**
- Wang et al. 2026 (interpretability vs utility for SAEs),
- Know Thy Judge / Safer or Luckier for measurement,
- optionally CAST for selective steering as future work.

**Also improve the gap statement.**  
Current gap language is still slightly too close to “we are the first to show divergence.” Better to say you:
- extend divergence claims **across representational families**,
- on **real behavioral surfaces**,
- with **transfer and measurement** as first-class evidence.

---

## Section 3: Localization / readout-to-control

### `main.tex` L103 — section title
Current:
> `Localization Does Not Predict Control`

This is a little too broad relative to what is actually shown. You do **not** show that localization in general fails to predict control; you show that **matched held-out readout quality** is insufficient in this comparison.

**Recommended replacement:**
- `Matched Readout Quality Does Not Guarantee Control`
- or `Good Readouts Do Not Guarantee Control`

That would better match your evidence.

### `main.tex` L109 — “A CETT probe...”
`CETT` is undefined and too niche for a workshop audience.

**Action options:**
- expand it if you really need the acronym, or
- remove the acronym entirely and say something like:
  > “A sparse H-neuron probe following Gao et al. (2025) identified 38 positive-weight neurons...”

I would remove the acronym from the main text.

### `main.tex` L119-L129
This is the paper’s strongest result and should remain the center of gravity.

**What is good:**
- matched readout quality,
- dose-response,
- random controls,
- delta-only SAE null,
- paired slope difference.

**What I would cut if space is needed:**
- Spearman `ρ = 1.0`
- Spearman `ρ = 0.18`

These are not doing much interpretive work relative to the slopes and CIs.

### `main.tex` L139-L142
This caveat is excellent, but it arrives slightly late. The section title and earlier prose are stronger than the caveat. Either:
- soften the section title, or
- move a shorter version of this caveat earlier.

---

## Section 4: Control is surface-local and can externalize

### `main.tex` L162-L169
The substantive point is good, but this sentence is a bit too assertive:

> “this is a portability limit on the metric, not behavioral inactivity.”

That may be true, but it reads as if the paper is adjudicating the endpoint rather than reporting it.

**Safer replacement:**
> “The flat accuracy endpoint coexists with substantial output perturbation, suggesting that behavioral activity and alias-accuracy movement come apart on this surface.”

That keeps the insight while sounding less defensive.

### `main.tex` L173-L182
This section is good, but for taste I would make the hierarchy even clearer:
- **SimpleQA = supporting stress test**
- **bridge benchmark = main generation externality test**

Right now the paragraph on SimpleQA is longer than it needs to be.

### `main.tex` L186-L201
This is very good. The one-shot frozen protocol sentence is valuable and should stay.

**Needed addition:** the “behavioral diagnosis, not circuit claim” disclaimer.

### `main.tex` L207-L223 — substitution examples table
The table is useful, but a few question stubs feel too truncated / colloquial. A skeptical reviewer could misread them.

I would make the question prompts slightly more literal, even if still shortened. For example:
- “Which Danny Boyle film was released in 1996?”
- “Which comic first introduced Superman?”
- etc.

This will make the examples feel more credible and less like hand-picked fragments.

---

## Section 5: Measurement choices changed the conclusion

### `main.tex` L244-L249
The section setup is strong.

### `main.tex` L251-L256
This is the weakest local patch in the current TeX draft.

You say the section is organized around the **H-neuron jailbreak scaling experiment**, but then you immediately bring in:

> “an apparent high-alpha reversal in a gradient-ranked pilot...”

That sentence now feels dangling because D7 is no longer in the main paper. It introduces another intervention family without enough context and slightly muddies the section focus.

**Recommendation:** cut this sentence from the main body and move it to:
- appendix,
- supplement,
- or a footnote at most.

Keep the measurement section entirely centered on the H-neuron jailbreak outputs unless you fully restore D7 context elsewhere.

### `main.tex` L258-L268
This is strong. The binary-vs-graded point is crisp and scientifically useful.

### `main.tex` L270-L277
Good substance, but `CSV-v2` / `CSV-v3` remain internal names unless the reader has your full project context.

At first mention, define them more semantically, e.g.:
- `CSV-v2 graded harmfulness rubric`
- `CSV-v3 taxonomy-based judge`
- `binary harmful/safe judge`

You can still keep the shorthand afterward.

---

## Section 6: Framework / checklist

### `main.tex` L304-L345
This section mostly works. The strongest part is the **minimum audit** table.

### `main.tex` L341-L343
This claim is too literature-wide for the evidence directly presented in the paper:

> “Most published steering results report localization... Few report matched negative controls...”

I agree with the direction, but this sentence should either be:
- softened,
- or explicitly tied to “in our reading of the literature”.

Suggested safer phrasing:
> “In our reading of the recent steering literature, target-surface control is reported more often than matched negative controls, cross-surface tests, or evaluator audits.”

### `main.tex` L346-L365
The recommendations are sensible. If page pressure appears, this is a reasonable place to compress without hurting the paper’s scientific spine.

The most important recommendation is R1. R2-R5 can be compressed stylistically if needed.

---

## Limitations

### `main.tex` L370-L375
Good limitation section.

Two optional additions:
1. explicitly mention the new April 2026 literature context:
   - the paper is not the first to suggest interpretability/utility divergence;
   - its distinctness lies elsewhere.
2. mention that bridge coding is **single-rater and behavioral**, which limits mechanism claims specifically.

---

## 5. Number-provenance and evidence-hierarchy audit

The provenance discipline is strong overall and is one of the paper’s hidden strengths.

### What is already good
- The main headline quantities in the abstract and core sections are mostly traceable.
- The ledger clearly distinguishes historical provenance from canonical current-state audits.
- The strategic assessment already has the right evidence hierarchy.

### What should be improved before submission
If you plan to use the provenance ledger as a reviewer-facing or supplement-facing artifact, update it so it tracks **all main-body headline numbers**, not just most of them.

At minimum, add or verify entries for:
- TruthfulQA flip counts (61 wrong→right, 20 right→wrong)
- the most important evaluator-tie statements (e.g. identical error sets, if you keep that phrasing)
- any main-text monotonicity / random-control summary numbers you keep
- any token-cap figures you keep in the main paper
- all table values shown in the main body

### Important structural note
The provenance file still uses the **older section numbering from the long markdown draft** (e.g. §4.1, §5.2, etc.). If this becomes a supplement, update it so that section references match the actual TeX submission.

### Recommendation
The provenance ledger is strong enough to be turned into an anonymized supplementary artifact. Do that if possible.

---

## 6. Claims to keep, soften, or cut

## Keep as headline-safe
- Held-out readout quality alone is **not sufficient** target-selection evidence in the matched FaithEval comparison.
- H-neurons are a real positive counterexample, but their effects are narrow and surface-local.
- TruthfulQA MC gains do not transfer to open-ended factual generation.
- The bridge benchmark reveals an active degradation pattern dominated by wrong-entity substitution.
- Measurement choices can change the scientific verdict about a steering intervention.

## Soften
- `Localization Does Not Predict Control`  
  → make this about **matched readout quality** rather than localization in general.

- “Wrong-entity substitution” as a **mechanistic** claim  
  → keep it as a **behavioral diagnosis**.

- Literature-wide claims like “few papers do X”  
  → either cite or soften.

- “metric portability limit” phrasing for BioASQ  
  → report the phenomenon more neutrally.

## Cut from main body or move to appendix
- The gradient-ranked pilot anecdote in the measurement section, unless D7 is restored with proper context.
- Any implication that one selector is causally superior in general.
- Any rhetoric suggesting a grand framework contribution detached from the actual case study.

---

## 7. If you can only make five changes, make these five

1. **Change the title** to a direct, thesis-first one.
2. **Add Wang et al. (ICLR 2026)** and reframe novelty accordingly.
3. **Insert one sentence defining the construct surfaces** in the introduction.
4. **Add the bridge disclaimer**: wrong-entity substitution is a behavioral diagnosis, not an internal circuit claim.
5. **Remove the gradient-ranked pilot sentence from the measurement section** unless you restore D7 context elsewhere.

---

## 8. Optional but high-value additions

## A. One explicit “what this paper contributes in 2026” sentence
Near the end of the introduction, add a sentence like:

> Relative to recent SAE-centric and synthetic steering evaluations, this paper contributes a cross-representation case study on real behavioral surfaces, with matched readout comparison, held-out generation externalities, and measurement robustness treated as first-class evidence.

## B. One sentence on selective/conditional future work
Your results naturally motivate **conditional steering**, not stronger global truth vectors.

A good future-work sentence would be:

> A promising next step is to move from global truthfulness directions toward selective interventions that trigger only under answer-risk or context-specific failure conditions, closer in spirit to conditional activation steering than unconditional global shifts.

## C. An anonymized reproducibility note
If allowed in your supplement package, include an anonymized artifact note with:
- code availability,
- prompt manifests,
- judge prompts,
- and provenance files.

---

## 9. Recommended bibliography additions / updates

### Add
- **Wang et al. 2026**, *Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders*  
  [OpenReview PDF](https://openreview.net/pdf/0d6cc4dd34c2b81d084e03c84c4f4171ef27dfc7.pdf)

- **Eiras et al. 2025**, *Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges*  
  [OpenReview](https://openreview.net/forum?id=kPMfYS2ugs)

- **Lee et al. 2025**, *Programming Refusal with Conditional Activation Steering*  
  [OpenReview](https://openreview.net/forum?id=Oi47wc10sm)

### Verify / update venue metadata
- **Arad et al. 2025** appears to have a final EMNLP 2025 version rather than only an arXiv entry.  
  [ACL Anthology PDF](https://aclanthology.org/2025.emnlp-main.519.pdf)

- **Wu et al. 2025 AxBench** final venue metadata should be ICML 2025 spotlight poster if your bib currently lists it differently.  
  [OpenReview](https://openreview.net/forum?id=K2CckZjNy0)

- **Opiełka et al. 2026** final venue metadata should reflect ICLR 2026.  
  [OpenReview](https://openreview.net/forum?id=LmLmhb6GEL)

---

## 10. Final verdict

The draft is substantially better than the longer source in one crucial respect: it is much closer to a **paper** and much farther from a **research archive**.

The remaining work is mostly not experimental. It is about:
- tightening the claim to what is truly earned,
- updating the paper to the April 2026 literature,
- restoring a small amount of missing protective nuance,
- and removing one or two residual traces of the older, broader storyline.

If you make the revisions above, I would judge the paper as a **credible, well-positioned long workshop submission** with a clear identity:

> a rigorous mechanistic-interpretability audit paper about when internal predictive signals fail to become reliable steering handles.

That is a real contribution, and it is a better one than trying to sell this as a new control method or a universal theory of truthfulness steering.

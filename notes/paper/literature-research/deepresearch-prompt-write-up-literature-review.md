## Dynamic Deep Research Prompt

**Objective:**
Produce a live-search, citation-rich **claim-boundary audit**, **novelty assessment**, and **related-work map** for a mechanistic interpretability / representation-engineering paper tentatively framed as:

*Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT*

This is **not** a generic literature review. It is a research-grade attempt to determine:

1. what is already established background,
2. what is only weakly or partially established,
3. what remains genuinely novel in the paper’s **exact** empirical and conceptual form,
4. what wording the paper can safely use without overclaiming,
5. and which citations are essential for the flagship paper versus a companion measurement note.

Your task is to help the authors **not fool themselves**. Do not maximize novelty. Maximize correctness, precision, and research taste.

**Researcher Persona (for the Research AI):**
You are a skeptical mechanistic-interpretability researcher with strong taste in claim hygiene. You care about exact experimental objects, exact intervention operators, exact evaluation surfaces, and exact scope limits. You do not equate “same vocabulary” with “same claim.” You are willing to conclude that many superficially adjacent papers are not true precedents.

---

**I. CONTEXT & DISCOVERY SCOPE:**

### A. The Research Mandate:
* **Origin of Inquiry:** The user is preparing a flagship methods paper, not merely a benchmark paper and not merely an evaluator paper. Among the user-provided documents, treat `2026-04-11-strategic-assessment.md` (revised 2026-04-12) as the **canonical project framing** if the documents disagree. Use the other project documents as historical and evidentiary context, not as overrides.
* **Primary Discovery Goal:** Determine the strongest defensible claim language and the real novelty frontier for a paper whose core thesis is **not** “detectors never work,” but rather:
  * predictive readout quality is **not a reliable heuristic** for choosing intervention targets, and
  * when control works, it may still be narrow, benchmark-local, or surface-local.
* **Key Unknowns to Uncover:**
  * Which literatures already establish the weak background claim that decodability/probe quality does not imply causal use?
  * Which literatures come closest to the stronger claim that good detectors/readouts/features may be **bad steering targets**?
  * Which literatures address the further claim that success on one surface (e.g. MC, one benchmark, one metric) can fail, reverse, or backfire on another?
  * Which literatures show that **measurement choices** can alter the scientific conclusion of steering or safety-intervention results?
  * Has prior work already articulated the integrated framing that **measurement, localization, control, and externality are separable empirical stages**?
  * What is the exact citation architecture needed for the flagship paper versus a companion measurement note?

### B. Canonical Project Framing You Must Respect Unless Live Literature Forces Narrower Wording:
* The paper’s strongest contribution is a **cross-method empirical pattern**, not a new evaluator and not a single flashy intervention result.
* The flagship should be understood through a **four-stage scaffold**:
  1. **Measurement** — can the evaluation be trusted?
  2. **Localization** — where is the feature/readout signal?
  3. **Control** — can intervening there steer behavior?
  4. **Externality** — does that control transfer across tasks/surfaces without harmful side effects or collapse?
* The flagship is best built around **three deep anchor case studies**, not a broad collage:
  1. **Localization → Control break:** SAE features vs H-neurons on FaithEval, where detection/readout quality is matched but steering diverges.
  2. **Control → Externality break:** ITI improves TruthfulQA-style answer selection but fails on open-ended factual generation, including confident wrong-entity substitution on the bridge benchmark.
  3. **Measurement → Conclusion break:** jailbreak evaluation artifacts, judge dependence, truncation, and graded-vs-binary reversals.
* D7-style causal-vs-probe selector evidence and detector-interpretation artifacts (e.g. L1 ranking fragility) are important **supporting** material, but should not automatically be treated as the paper’s co-headliners.

### C. Scope & Temporal Boundaries:
* **Recency Requirement:** Cover foundational older work where necessary, but search aggressively for 2023–present work and especially the newest work available at execution time. Do not rely on internal memory for recent literature.
* **Geographical/Sectoral Focus:** Mechanistic interpretability, probing/representation analysis, causal representation work, activation steering / representation engineering / ITI-style interventions, sparse feature methods, model editing only where relevant to target selection or causal utility, and AI safety evaluation / judge dependence where it changes intervention conclusions.
* **Benchmark-Role Discipline:** Do **not** flatten all benchmarks into one pool.
  * Treat **TruthfulQA MC / answer selection** as a clean control surface.
  * Treat **bridge open-ended factual generation** as the main externality/generation surface.
  * Treat **hard OOD open-ended QA** as a stress test.
  * Treat **FaithEval / FalseQA** as diagnostic surfaces rather than the whole truthfulness story.
* **Future-Work Boundary:** Do not let literature on selective/conditional truthfulness steering, adaptive choosers, or next-step policy ideas become the main novelty claim unless it directly changes the status of the current paper. Those are likely sequel/future-work lanes, not the flagship paper’s core claim.

---

**II. CORE INVESTIGATION & SEARCH VECTORS:**

### A. Primary Research Questions (To be answered via live search):
* **RQ1 — Weak Background Claim:** What is the earliest and strongest lineage showing that probe accuracy, decodability, or accessible representation content does not by itself establish causal relevance or functional use?
* **RQ2 — Exact Steering-Era Claim:** What papers come closest to showing that high-quality detectors/readouts/features are unreliable for choosing intervention targets?
* **RQ3 — Localization → Control:** Are there prior papers where matched or stronger detection quality coexists with sharply worse steering/intervention efficacy on the same or closely matched benchmark?
* **RQ4 — Control → Externality:** What prior work shows that a working intervention on one surface (e.g. multiple choice, short-form labels, one benchmark) fails, backfires, or becomes non-transferable on another surface (e.g. open-ended generation, cross-task transfer, collateral damage)?
* **RQ5 — Measurement → Conclusion:** What prior work shows that evaluator choice, metric design, truncation, parser assumptions, or judge dependence can reverse or compress the scientific conclusion about steering or safety interventions?
* **RQ6 — Integrated Framing:** Has prior literature already articulated the broader methodological point that **measurement, localization, control, and externality are separable empirical stages** that should not be conflated?
* **RQ7 — Positive Counterexamples:** Where do detector-selected or probe-selected targets actually steer well, and what boundary conditions make them work?
* **RQ8 — Claim Hygiene:** Given the literature, what wording is safe, what wording is strong-but-plausible, and what wording is too strong?
* **RQ9 — Related Work Architecture:** Which citations belong in the flagship paper’s Related Work section, and which belong in a companion measurement/evaluator note instead?
* **RQ10 — False Friends:** Which papers look relevant by keywords but are not true precedents for the paper’s actual claim?
* **RQ11 — Reviewer Objections:** What literature would a skeptical reviewer invoke for the following challenges, and how seriously do those objections cut?
  * “This is just old decodability-vs-causality.”
  * “You are mixing apples and oranges across operators/tasks.”
  * “Your positive H-neuron result undercuts your thesis.”
  * “Your selector comparison is caveated / missing controls.”
  * “This is only one model.”

### B. Dynamic Search Vectors (Keywords, boolean queries, and phrasing strategies the AI should use):
Use multiple search families. Search beyond the slogan. Search by **stage break**, not only by vocabulary.

#### 1. Measurement / evaluator-dependence searches
* `"activation steering" evaluation artifact`
* `"jailbreak" evaluation judge dependence`
* `"binary vs graded" harmfulness evaluation language model`
* `truncation artifact safety evaluation llm`
* `disclaimer-heavy outputs evaluation bias`
* `metric choice reverses conclusion activation steering`
* `judge sensitivity intervention evaluation llm`
* `parser failure answer format evaluation`
* `harmful compliance judge disagreement activation steering`
* `safety intervention evaluation artifact language models`

#### 2. Localization / probing / readout fragility searches
* `probe accuracy causal relevance language models`
* `decodability does not imply use`
* `probing limitations selectivity control task`
* `amnesic probing causal language models`
* `representation erasure functional importance`
* `feature accessibility vs causal function`
* `response length confound probing language models`
* `spurious probe features language models`
* `neuron importance stability l1 sparsity interpretability`
* `regularization artifact neuron ranking`

#### 3. Control / steering-target selection searches
* `activation steering target selection`
* `probe-based intervention language models`
* `feature detection vs intervention usefulness`
* `concept detection vs control internal features`
* `predictive feature bad intervention target`
* `representation engineering evaluation steering reliability`
* `sparse autoencoder steering`
* `sae features intervention utility`
* `causal head selection steering`
* `gradient-based selection probe-based selection steering`
* `interpretable feature not best intervention target`

#### 4. Externality / surface-mismatch searches
* `multiple choice vs generation steering language models`
* `truthfulness steering generation failure`
* `steering side effects collateral damage task transfer`
* `activation steering backfires generation`
* `intervention works on benchmark but not open-ended generation`
* `wrong entity substitution generation intervention`
* `steering reliability across prompts tasks models`
* `safety intervention externalities llm`
* `steering benchmark local not transfer`

#### 5. Seed-lineage searches
Search forward/backward from seed lineages such as:
* H-neurons / hallucination-associated neurons
* Inference-Time Intervention (ITI)
* LITO / adaptive intensity truthfulness intervention
* universal truthfulness hyperplane work
* sparse autoencoder feature steering papers
* causal mediation / causal head-selection papers
* classic probing-causality literature

Use these as **entry points**, not as proof that those are the nearest neighbors.

### C. Emerging Hypotheses to Test Against Live Data:
* The weak decodability-vs-causality critique is old and should be treated as background, not novelty.
* The more specific claim that **predictive readout quality is an unreliable heuristic for steering-target choice** is much less directly established.
* The **localization → control** break may be better covered than the **control → externality** break.
* The measurement/evaluator literature may be crucial and currently under-cited in mech-interp novelty claims.
* The paper’s strongest novelty may live partly in the **integrated synthesis across stages and methods**, not in any single isolated effect.
* Strong positive counterexamples likely exist; they should narrow the paper’s claim to **heuristic unreliability**, not impossibility.
* D7-like selector literature may matter, but it may belong to supporting context rather than the headline novelty claim.

---

**III. TOOL INTEGRATION & METHODOLOGY:**

### A. Mandatory Tool Usage:
You **must** use live web search, browsing, PDF retrieval, and citation-chasing tools. Do **not** answer from memory. You must also use the user-provided project documents to understand what the paper is actually claiming. Treat the project documents as internal context for the paper’s evidence and framing; treat live literature search as the source of truth for prior work.

### B. Iterative Search Protocol:
1. Start by decomposing the slogan into at least four claim levels:
   * weak/background claim,
   * core steering-era methodological claim,
   * exact anchor-level claim,
   * integrated four-stage flagship claim.
2. Search each of the **three anchor case-study lineages separately** before doing a broad sweep.
3. Allocate effort **asymmetrically**:
   * ~35%: near-direct steering-target / intervention-utility literature,
   * ~25%: measurement / evaluator-dependence literature,
   * ~20%: control → externality / side-effect / transfer literature,
   * ~20%: foundational probing / decodability / causal-use literature.
   Do **not** spend half the budget on generic probing papers.
4. For every paper that looks close, read beyond the abstract. Inspect the intro, experiments, appendix, and when necessary the code or benchmark documentation.
5. Use backward snowballing, forward snowballing, author-following, and citation graph exploration for all top candidates.
6. Run a separate recency sweep for the most recent year to catch fresh preprints or conference papers.
7. Keep a **false-friend log**: papers that share vocabulary but do not actually test the relevant relationship between readout quality and intervention utility.
8. If a paper seems close, ask explicitly:
   * What exactly is read out or detected?
   * What exactly is intervened on?
   * Is the target-selection heuristic itself under test?
   * Is the intervention operator comparable?
   * Is the evaluation surface multiple choice, open-ended generation, harmfulness refusal, parser-dependent scoring, or something else?
   * Are externalities or collateral damage measured?
   * Does measurement choice alter the conclusion?
9. If initial searches return only broad probing literature, refine toward **target selection**, **intervention utility**, **surface transfer**, and **evaluator dependence**.
10. Distinguish what authors *say* from what their experiments *actually establish*.

### C. Data & Evidence Requirements:
For every paper judged high-relevance, extract:
1. citation
2. venue/year/status (preprint vs accepted)
3. what object is measured/read out
4. what object is intervened on
5. selection heuristic for targets
6. intervention operator
7. evaluation surface(s)
8. whether externalities/collateral damage are measured
9. whether evaluator/measurement issues matter
10. main claim
11. why it matters for this paper
12. anchor relevance
13. overlap strength: weak / moderate / strong / near-direct
14. whether it reduces novelty of the weak claim, the exact claim, the integrated claim, or only supports framing
15. whether it is a positive counterexample, cautionary precedent, methodological neighbor, or false friend

**Important:** Abstract-only judgments are not enough for any paper labeled **strong** or **near-direct**.

---

**IV. SOURCE MATERIAL & VERIFICATION STRATEGY:**

### A. Prioritized Live Sources:
* **Primary (Direct data, live feeds, recent reports):**
  * arXiv PDFs
  * conference proceedings (ICLR, NeurIPS, ICML, ACL, EMNLP, COLM, etc.)
  * official paper pages / author PDFs
  * Semantic Scholar / OpenAlex / Google Scholar / citation graph tools
  * official repositories or benchmark docs when method details or evaluation surfaces are unclear
* **Secondary (News, analysis, journals):**
  * high-quality surveys for orientation only
  * blog posts or discussions only if they help locate primary sources or clarify methods

### B. Mandatory Source Verification (Overcoming Hallucination/Stale Data):
* Cross-check publication dates, version histories, and venue status.
* Separate truly prior work from contemporaneous or follow-on work.
* Verify whether a paper’s evidence is about:
  * decodability vs causality,
  * detector quality vs steering target quality,
  * steering reliability or side effects,
  * measurement artifacts,
  * or something only superficially adjacent.
* Do not let a paper erase the current paper’s novelty merely because it proves a weaker statement.
* Do not let abstract-level similarity substitute for method-level or claim-level overlap.
* Verify whether the paper evaluates **multiple choice vs open-ended generation**, or **binary vs graded scoring**, because those differences matter here.
* When sources conflict, identify whether the conflict comes from differences in object, operator, model scale, task, or evaluator.

### C. Explicit Exclusions:
Ignore or strongly deprioritize:
* prompt-only steering papers with no internal target-selection question
* generic interpretability papers that never test intervention utility
* model editing or control papers that never touch the relationship between predictive signal and intervention target choice
* papers that mention probes/steering/SAEs/causality but are only vocabulary-near
* surveys used as substitutes for primary-source verification
* internal project documents as evidence of prior art
* stale internal framings that the canonical strategic assessment has already closed (for example, treating evaluator optimization, mixed-source vector search, or chooser work as the paper’s main current novelty)

---

**V. SYNTHESIS & INTERPRETATION OF NEW DATA:**

### A. Handling Conflicting Live Data:
When prior work seems to disagree, report whether the disagreement is actually due to different stages of the scaffold:
* measurement,
* localization,
* control,
* externality.

Do not collapse them into one verdict.

### B. Contextualizing the “Now”:
* Build the literature map around the paper’s **three anchors**.
* Distinguish **foundational background** from **true nearest neighbors**.
* Treat the flagship as a paper about **what goes wrong when researchers conflate measurement, localization, control, and externality**, not as a broad essay on steering.
* Keep D7-style selector work and detector-interpretation artifacts in the right place: potentially important, but not automatically central.
* Explicitly test whether the integrated four-stage framing already exists in the literature or whether it is itself a valuable synthesis contribution.
* Preserve the paper’s core asymmetry:
  * some detector-selected targets do work,
  * but detector quality is still not a reliable heuristic overall.

### C. Future Outlook & Actionable Insights:
Based strictly on the newest data found:
* identify the narrowest claim variants that still look genuinely novel,
* identify which future-work lane is suggested by the literature,
* but do **not** recast the current paper as a future selective-intervention project unless the literature clearly collapses the present novelty claim.

---

**VI. FINAL REPORT SPECIFICATIONS:**

### A. Report Structure:
1. **Executive novelty verdict**
   * weak claim novelty
   * exact steering-target claim novelty
   * integrated four-stage claim novelty
   * what the paper should claim vs should not claim
2. **Claim decomposition**
   * at least 6 distinct claim variants, from conservative to ambitious
3. **Paper-safe evidence hierarchy**
   * which precedents threaten the safe core
   * which only affect the supporting / caveated edge
4. **Four-stage prior-work map**
   * measurement
   * localization
   * control
   * externality
5. **Anchor 1: localization → control**
   * nearest precedents for matched/strong detection with failed steering
6. **Anchor 2: control → externality**
   * nearest precedents for surface-local or non-transferable control
7. **Anchor 3: measurement → conclusion**
   * nearest precedents for evaluator/metric choice changing the scientific result
8. **Positive counterexamples and boundary conditions**
   * papers where detector-selected or probe-selected targets do steer well
9. **False-friend table**
   * papers that sound close but are not true precedents
10. **Reviewer-objection memo**
    * apples-to-oranges objection
    * H-neuron counterexample
    * selector-caveat objection
    * single-model objection
11. **Novelty-safe wording ladder**
    * safe formulations
    * stronger but plausible formulations
    * risky formulations to avoid
12. **Related Work citation packs**
    * must cite
    * should cite
    * optional but useful
    * cite only if making stronger claims
    * separate pack for the flagship paper vs the companion measurement note
13. **Evidence table**
    * 12–25 highest-value papers, each with overlap strength, exact lesson, and novelty impact
14. **Optional paper-ready outputs**
    * one concise Related Work paragraph for the flagship paper
    * one concise Related Work paragraph for the companion measurement note
    * one intro-framing paragraph that accurately positions the novelty

### B. Target Length and Detail:
Substantial and discriminating. Prefer depth over breadth, but breadth must still be real. The output should be detailed enough to directly guide the paper’s intro, novelty statement, claim language, and related-work section.

### C. Mandatory Citation Style (Must include retrieval dates and URLs):
Every nontrivial factual claim about the literature must be cited. Include direct links or identifiers, and include retrieval dates for web-sourced materials. Distinguish clearly between:
* what the source explicitly claims,
* what you infer from its methods/results,
* and how that affects the current paper’s novelty boundary.

---

**VII. CONCLUDING DIRECTIVES FOR THE RESEARCH AI:**
* Execute this research by prioritizing live, up-to-date information. Your internal knowledge is only a starting hypothesis.
* Do not maximize novelty. Maximize correctness.
* Do not confuse weak background precedents with near-direct precedents.
* Do not treat all benchmarks or evaluation surfaces as interchangeable.
* Do not let the broad probing literature drown the narrower question of whether predictive internal features are reliable intervention targets.
* Do not let the slogan “good detectors can be terrible steering targets” do the thinking for you; recover the precise defensible claims underneath it.
* Good taste here means being willing to say:
  * “this paper only weakens the weakest form of the claim,”
  * “this is a useful citation but not a direct precedent,”
  * “this is the real nearest neighbor,”
  * or “the integrated synthesis is more novel than any single component.”
* Keep returning to the central paper question: **When does detection stop being a reliable guide to control, and what does the literature already know about that boundary?**
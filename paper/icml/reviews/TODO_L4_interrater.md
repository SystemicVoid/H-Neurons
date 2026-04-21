# L4: Bridge inter-rater reliability — no-brainer fix

> **Status (2026-04-21): CLOSED.** All deliverables landed; raw agreement 96.5% on 57 discordant cases (Cohen's κ = 0.90, Gwet's AC1 = 0.96), two disagreements resolved under the pre-frozen rule (rule_gap = 0/2), adjudicated R→W wrong-entity share 72.1% [57.3, 83.3]. Final analysis: [`../reports/2026-04-21-bridge-irr-review.md`](../reports/2026-04-21-bridge-irr-review.md). Core commits: `7d9a0e5` (artifacts), `2fa9de2` (paper integration), `06a9ccb` (site), `4187a4a` (review report), `e0bd9bc` (§4.3 tightening).

**Limitation:** Bridge failure-mode coding is single-rater, no inter-rater reliability reported.

**Why it matters:** The ~70% wrong-entity substitution claim is the paper's most vivid qualitative result. A reviewer can dismiss it as subjective without IRR. The added value is not just κ — it is making the bridge result look like a serious behavioral mechanism analysis rather than an anecdotal taxonomy.

**Scope: all 57 discordant cases, not just R→W**

Code **all discordant bridge cases**, not only the 43 right-to-wrong flips. Include the 14 wrong-to-right rescues. This lets you compare damage modes vs rescue modes.

**What to do:**

1. Blind re-code all 57 discordant cases (categories: substitution, evasion/factual denial, answer dilution, formal refusal)
2. Second rater priority:
   - **Best:** Real human second rater (strongest IRR evidence)
   - **Acceptable:** LLM judge as second rater with the same rubric (~$1 API cost) — acceptable as a sensitivity check, but weaker as IRR evidence
   - **Weakest:** Self-blinded re-coding
3. **Predefine an adjudication rule before seeing disagreements**
4. Report:
   - Raw agreement
   - Cohen's κ on the 4-category coding
   - **Gwet's AC1** as a robustness statistic (κ can behave badly under skewed category prevalence)
5. Keep the main claim **qualitative** unless agreement is strong: "wrong-entity substitution is the dominant coded mode," not "exactly 70%"
6. `scripts/analyze_concordance.py` already exists — check if it can be reused

**Minimum deliverable:**

> A second blinded coder labeled the 57 discordant bridge cases. Agreement was X%, Cohen's κ = Y, Gwet's AC1 = Z. The wrong-entity-substitution conclusion remained stable after adjudication: N/M right-to-wrong flips were coded as substitutions.

**Better deliverable (add a table):**

| Transition  |  n | Substitution | Evasion/denial | Dilution | Refusal |
| ----------- | -: | -----------: | -------------: | -------: | ------: |
| right→wrong | 43 |            … |              … |        … |       … |
| wrong→right | 14 |            … |              … |        … |       … |

This table would make the externality section materially more convincing.

**Cost:** ~2 hours + ~$1 API credits. Near-zero compute.

**Paper change:** Update L4 in limitation inventory. Add one sentence + optionally one table to §4.3 reporting κ and AC1. Where it appears: Main text, §4.3 bridge subsection.

**Data location:** Bridge discordant cases are in the TriviaQA bridge test set results.

**Deadline context:** Must complete before May 8 EOD (ICML MI Workshop submission). Run in parallel with L5.

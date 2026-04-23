You are doing a clean-room narrative synthesis over a tightly scoped ground-truth evidence pack for an ICML mechanistic interpretability paper.

This is not a summarization task. It is not a prose-polish task. It is not a benchmark-report task.

Your job is to infer the strongest paper framing that is actually earned by the evidence: the framing that best compresses the results, yields the sharpest non-obvious thesis, feels genuinely mech-interp rather than merely evaluative, and survives hostile scrutiny.

Core stance:
Assume prior framing is contaminated by path dependence. Do not preserve it, search for it, reconstruct it, or defer to it.
Treat filenames, section titles, family IDs, act tags, metric prefixes, and other organizational labels as bookkeeping rather than theory.
A framing earns selection only if it beats plausible alternatives on:
- evidential support
- explanatory compression
- benefits AI safety
- surprise
- mechanistic interest
- reviewer defensibility
- likely literature-facing novelty

Important:
Do not optimize for breadth, flatteringness, or continuity with existing storylines.
If the strongest framing is narrower, stranger, more asymmetric, or less grand than the obvious headline, choose it.

## Scope and evidence discipline
Use only the files in the evidence pack unless a listed file directly references a missing dependency that is essential for interpretation.

Do not let the markdown organization dictate the paper’s conceptual frame.

## Working method

### Phase 1 — Ledger-first discovery
Start from `metric_ledger.jsonl`, not the prose.

Extract the strongest empirical facts in plain language, stripped of inherited labels.
Look for:
- asymmetries
- dissociations
- reversals
- transfer failures
- cross-surface inconsistencies
- externalities
- cases where strong internal signal does not translate into useful or safe control
- cases where intervention effects depend heavily on the measurement surface
- cases where the mechanism suggested by one surface is contradicted or refined by another

Prioritize patterns that recur across multiple metrics, surfaces, or intervention families.

### Phase 2 — Rival framing search
Do not lock onto the first plausible headline.

Search for rival framings that explain the evidence from meaningfully different angles.
The search space should include both broader and narrower possibilities, including:
- framings centered on control rather than detection
- framings centered on externalities or transfer structure
- framings centered on measurement-induced illusion
- framings centered on intervention-surface mismatch
- framings centered on a deeper mechanistic dissociation
- at least one tempting framing that sounds good but is not actually well-supported

For each candidate framing, ask:
- What exact empirical pattern does this framing compress?
- What does it explain that rivals do not?
- What evidence is doing most of the work?
- What evidence makes this framing fragile, narrow, or misleading?
- Would this still feel like a mech-interp contribution rather than a generic cautionary note?

### Phase 3 — Selection
Choose the single framing that best balances surprise and survivability.

Penalize framings that are true but generic.
Penalize framings that depend on one evaluator, one benchmark, or one intervention family unless that dependence is itself the story.
Penalize framings that merely restate a well-known methodological cliché.
Reward framings that expose a specific, non-obvious field-level lesson about what mechanistic signals can and cannot buy you.

## Anti-genericity rule
Downgrade any framing that can be paraphrased as:
- “measurement matters”
- “correlation is not causation”
- “steering has tradeoffs”
- “good readouts are not necessarily good steering targets”

Those may be locally true, but they are not sufficient paper framings on their own.
If one of those lines survives, sharpen it into the more specific claim that this evidence uniquely supports.

## Claim hygiene
For every important statement, distinguish among:
- directly evidenced by the pack
- interpretation supported by the pack
- plausible literature-facing novelty hypothesis
- not yet earned

Do not state literature novelty as fact. Literature is out of scope unless explicitly provided.
You may, however, state what kind of literature-facing novelty claim the chosen framing would need to beat if later validated.

## Deliverable
Produce exactly these sections.

## 1. Chosen framing
Provide:
- a title-like framing label
- a one-sentence thesis
- a concise explanation of why this is the strongest paper story
- the specific empirical pattern it compresses
- why it reads as mech-interp rather than merely evaluation or benchmarking
- the most important caveat on this framing

## 2. Candidate framings
Give 6 candidate paper framings.

They must be meaningfully different from one another.
At most one candidate may be close to the obvious “good readouts are unreliable steering targets” line, and if included it must be made more specific and less generic.

For each candidate include:
- framing label
- one-sentence thesis
- what empirical pattern it compresses
- why it is high-signal
- why it could matter for mech-interp specifically
- what evidence most strongly supports it
- what evidence weakens, narrows, or limits it
- likely literature-facing novelty, stated only as a hypothesis
- risk of sounding derivative or generic
- verdict: `strong contender (10)`, `backup (5)`, or `reject (1)` with a 1–10 score

## 3. Why the winner beats the runners-up
Compare the chosen framing against the two closest alternatives.
Focus on:
- explanatory compression
- surprise
- reviewer defensibility
- dependence on fragile assumptions
- whether the framing makes the paper feel like a real contribution instead of a bag of results

## 4. Claim hygiene
List:
- strongest claims fully earned now
- strong claims that are still interpretation-laden
- claims that would require literature validation before being made
- tempting claims that should not appear in the paper

## 5. Hostile self-critique
Attack the chosen framing.

List the 5 strongest objections a skeptical reviewer could raise from this evidence pack alone.
For each objection include:
- why it bites
- whether it is fatal, serious but survivable, or mostly containable
- the most honest containment, concession, or reframing

## Style
- analytical, not diplomatic
- ruthless but fair
- sharp synthesis over exhaustive coverage
- no generic summary first
- no inherited jargon unless the evidence forces it
- no deference to pre-existing narratives
- make a real decision
- do not hedge away the central choice

Final instruction:
I care more about discovering the right paper than preserving any existing one.
If the best framing is narrower, stranger, more conditional, or less flattering than the obvious headline, say so.
Take all the time you need and do your absolute best ! 
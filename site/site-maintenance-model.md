# Site State Analysis

> Current implementation audit for `site/`, replacing the earlier migration plan and session handoff log.
>
> **Updated**: 2026-03-20
> **Scope**: what is live now, what is actually data-driven, what is still brittle, and what the website most needs next.

## Executive Summary

The original scaling problem is mostly solved.
`site/` is no longer a single-page monolith that mixes all narrative, charts, CSS, and JS in one file. The website is now a small multi-page static presentation with shared assets, generated JSON, and clearer separation between weekly memo content, stable results, methods, roadmap, and deep dives.

The main remaining problem is no longer page-splitting. It is drift control.
Some parts of the site already read from canonical JSON exports, while other parts still depend on hand-maintained prose, mixed manual/data sentences, or appendix-style literals. The next phase should therefore be an anti-drift cleanup, not another architecture rewrite.

## Progress Since Last Update

The highest-leverage reliability pass from the previous version of this document has now been done.

Recent completed work:

- `site/data/classifier_summary.json` now includes classifier structure data, not just headline metrics
- `site/assets/charts.js` no longer hardcodes `layerData` or `topNeurons`
- the layer-distribution and top-neuron sections on `results/gemma-3-4b.html` now read from canonical classifier JSON
- the intervention, evaluation-confound, and swing-population sections on `results/gemma-3-4b.html` now bind their peak parse-failure, strict-remap recovery, and frozen-vs-swing counts from canonical intervention JSON
- `data/gemma3_4b/pipeline/neuron_4288_summary.json` now tracks the six-test neuron-4288 verdict as structured site-facing data
- `site/data/classifier_summary.json` now includes `top_neuron_artifact_summary`
- the 4288 verdict and takeaway blocks on `results/gemma-3-4b.html` now bind from canonical JSON instead of frozen HTML literals
- `site/assets/shared.js` now hydrates classifier-structure and distributed-detector bindings for non-chart pages
- the main early-layer and 38-vs-219 detector claims on `story.html` now read from canonical JSON instead of page-local literals
- the intervention trend copy on `story.html` now binds the anti-compliance `&alpha;=0.0` to `&alpha;=3.0` effect from canonical intervention JSON instead of freezing "monotonic rise" prose in HTML
- the `story.html` intervention evidence card now binds the existing standard raw `&alpha;=3.0` point alongside the committed strict-remap point, so that comparison no longer depends on half-editorial prose
- the `story.html` 4288 evidence card now binds the tracked top-neuron takeaway from the artifact payload instead of carrying a page-local summary of the verdict
- `docs/ci_manifest.json` now actually enforces `site/story.html` as a required provenance surface for `anti_compliance_delta_0_to_3`, matching the live story-page comment
- `scripts/audit_ci_coverage.py` now validates the top-neuron verdict payload alongside the classifier-structure payload
- `scripts/audit_ci_coverage.py` now validates the classifier-structure payload
- `data/gemma3_4b/pipeline/classifier_structure_summary.json` is now tracked alongside the other classifier outputs
- `scripts/export_site_data.py` now reads that tracked summary by default instead of loading the checkpoint during normal site export
- a local validator now exists at `uv run python scripts/export_site_data.py --validate-classifier-structure-summary`

That matters because the site no longer tells two separate stories about key quantitative surfaces: one in committed JSON and another in manually maintained page literals.

The main remaining weak point is no longer classifier-structure export reproducibility.
That gap is now closed for normal site work. The remaining risk is now concentrated in the static copy that still wraps the evidence pages, especially the claim-heavy synthesis on `story.html` that still depends on manual framing and incomplete provenance coverage.
Within that story-page risk, the intervention/mechanism slice is now narrower: the headline effect size, the raw-vs-remap `&alpha;=3.0` comparison, and the 4288 takeaway are all wired, and the remaining open decisions are about whether parse-failure framing or other repeated comparisons should stay manual, bind from existing JSON, or wait for a tracked exporter field.

## Current Snapshot

At the time of this review, the site has:

| Area | Current state |
|---|---|
| HTML pages | 7 live pages |
| Shared assets | `assets/shared.css`, `assets/shared.js`, `assets/charts.js` |
| Generated site JSON | 4 files in `site/data/` |
| Charts | 9 `<canvas>` charts |
| Static figures | 8 `<img>` figures |
| Manual-content markers | 23 `data-source="manual"` markers |
| Provenance comments | 18 `<!-- from: ... -->` comments |
| Broken internal refs found in this audit | 0 |

This is a much healthier shape than the old single-page deck. The site has already crossed the important threshold from "one giant hand-edited artifact" to "a small static site with some real structure."

## What Is Actually Live

### Page inventory

| Page | Role | State |
|---|---|---|
| `index.html` | Weekly memo / advisor landing page | Live and clearly narrower than the old monolith |
| `story.html` | Standing paper narrative | Live and useful |
| `results/gemma-3-4b.html` | Quantitative results ledger | Live and currently the main evidence page |
| `methods.html` | Pipeline, prompt framing, reproducibility | Live and fairly stable |
| `extensions.html` | Forward roadmap | Live, but inherently freshness-sensitive |
| `deep-dives/neuron-4288.html` | Artifact appendix for top-neuron story | Live and intentionally static |
| `deep-dives/swing-characterization.html` | Appendix for swing-population analysis | Live and more data-connected than the 4288 page |

### Not live

The following items from the old plan are still not implemented, and that is fine:

- No `archive/` section
- No `backup/` section
- No `results/mistral-24b.html`
- No template/build system

Those were future possibilities in the earlier plan, not missing bugs in the current site.

## What Changed Relative to the Old Plan

The earlier document was written during a migration. Much of that migration is now done, so large parts of the old plan had become stale.

### Already completed

- The single-page presentation has been split into multiple pages.
- Shared CSS has been extracted.
- Shared JS hydration exists.
- JSON-backed site data exists in `site/data/`.
- The main Gemma results have a dedicated page.
- The methods page exists.
- The core-story page exists.
- The extensions page exists.
- The two deep-dive pages exist.

### No longer useful to keep in this file

- Session-by-session handoff prompts
- Completed phase checklists
- Forward-looking instructions that describe work already merged
- Repeated "next slice" notes that belonged to a specific earlier session
- Empty archive/backup planning as if they were immediate priorities

That material belongs in git history, not in the live planning document.

## Current Technical Architecture

### Site model

The site is now a static HTML site with no build step.
That is still a reasonable choice for the current scale.

The implementation is built around:

- `site/assets/shared.css`
  - Shared visual system and page styling
- `site/assets/shared.js`
  - Scroll/progress behaviors
  - Reveal animations
  - Counter effects
  - JSON fetch + text binding for repeated metrics
- `site/assets/charts.js`
  - Chart.js initialization
  - JSON fetch for chart-backed pages
  - Intervention and swing chart rendering

### Data path

The canonical generated site data currently lives in:

- `site/data/classifier_summary.json`
- `site/data/intervention_sweep.json`
- `site/data/pipeline_summary.json`
- `site/data/swing_characterization.json`

The intended write path is already in place:

- `scripts/export_site_data.py`

That is important. It means the project already has the right spine for data-driven rendering, even though not every page element is fully wired yet.

One detail is now worth making explicit:

- `classifier_summary.json` is now schema v2 and includes `selected_h_neuron_structure`
- `data/gemma3_4b/pipeline/classifier_structure_summary.json` is the tracked upstream contract for classifier structure
- the site exporter reads that tracked summary by default
- checkpoint-based recomputation is now an explicit local validation/update step, not a default export dependency

### Nested-page fetch handling

One real implementation detail deserves to stay documented: the site correctly handles nested pages.

`shared.js` resolves JSON paths relative to the script URL, and `charts.js` uses `import.meta.url`.
That is the right fix for pages under `results/` and `deep-dives/`, and it avoids the common "works on root page, breaks in nested path" failure mode.

### HTTP assumptions

The old warning is still valid:

- the site should be served over local HTTP during verification
- `file://` is still the wrong environment for pages that depend on `fetch()`

This is not future planning anymore. It is an operational fact of the current implementation.

## What Is Data-Driven vs Still Hand-Maintained

The site is best understood as a hybrid.
Some surfaces are already tied to generated data; others are still curated editorial artifacts.

### Strongly data-driven already

These parts are in good shape:

- Repeated classifier headline metrics via `data-classifier-bind`
- Repeated classifier structure values on `results/gemma-3-4b.html` via `data-classifier-structure-bind`
- Repeated intervention summary metrics via `data-intervention-summary-bind`
- Results-page intervention narrative facts via `data-intervention-bind`
- Pipeline counts via `data-pipeline-bind`
- Swing characterization metrics via `data-swing-bind`
- Main intervention charts via `site/data/intervention_sweep.json`
- Classifier performance chart via `site/data/classifier_summary.json`
- Classifier layer-distribution and top-neuron charts via `site/data/classifier_summary.json`
- Swing transition / enrichment charts via `site/data/swing_characterization.json`

This is the main success of the current architecture: repeated numbers are no longer copied by hand everywhere.

### Still manual or partly manual

These areas remain drift-prone:

- Narrative paragraphs on `index.html`, `story.html`, `results/gemma-3-4b.html`, and `extensions.html`
- Manual qualitative framing of what counts as the current story
- Manual roadmap prioritization on `extensions.html`
- Static appendix claims on `deep-dives/neuron-4288.html`
- Manual synthesis prose on `results/gemma-3-4b.html` that interprets already-bound evidence

### No longer hardcoded in JS

One previously important gap is now closed:

- `layerData` is no longer hardcoded in `site/assets/charts.js`
- `topNeurons` is no longer hardcoded in `site/assets/charts.js`

That was the cleanest remaining quantitative anti-drift issue in the earlier version of this document, and it is now resolved at the site-consumption level.

### Upstream reliability model

Classifier structure now follows a clearer contract:

- the raw scientific source remains the classifier checkpoint
- the tracked site-facing source is `data/gemma3_4b/pipeline/classifier_structure_summary.json`
- `site/data/classifier_summary.json` is regenerated from that tracked summary
- local checkpoint-based validation remains available when the checkpoint is present

This is a better fit for the repo than making the site exporter depend on an ignored local file.

## Page-by-Page Assessment

### `index.html`

This page now does the right job.
It is no longer trying to be the whole website. It behaves like a meeting memo: agenda, current framing decisions, stable-findings links, and next runs.

What is good:

- Narrower scope
- Clear links outward to the stable pages
- Reuses bound summary metrics instead of repeating hand-edited numerics everywhere

What is still true:

- It is a standing weekly memo page, not a real archived weekly series
- Its framing paragraphs are manual and will keep changing often

### `story.html`

This is now the narrative spine of the site.
It gives the project a place to make a defendable claim without mixing in every chart and every appendix detail.

What is good:

- Clear separation between claim, evidence, threat, falsification, and decision
- Appropriate use of shared bound metrics
- Better rhetorical role than trying to force this material into the landing page

Current limitation:

- The story is still editorially hand-maintained
- That is acceptable, but it means drift risk is conceptual rather than purely numeric

### `results/gemma-3-4b.html`

This is the densest and most important page.
It functions as the results ledger and quantitative appendix for the live Gemma story.

What is good:

- It holds the main charts in one place
- It uses JSON hydration for many repeated quantitative elements
- It is the clearest "stable evidence" destination in the site

What is still fragile:

- It still has the highest concentration of manual narrative in the site
- It is still the page most likely to drift if results change and the prose is not updated with the data exports
- The intervention sections are materially safer than before, but the 4288 verdict block and some concluding synthesis still remain static appendix-style text

If one page deserves the next anti-drift pass first, it is this one.

### `methods.html`

This page is one of the cleanest implementations in the site.
It has a clear role, good use of bindings, and a stable informational contract.

What is good:

- Strong page-role separation
- Metrics mostly come from shared bindings
- The prompt/evaluator mismatch is documented explicitly

This page already looks closer to "finished reference page" than "active workbench."

### `extensions.html`

This page is useful, but it is intentionally more editorial than evidentiary.
It acts like a roadmap and prioritization memo, not a data ledger.

What is good:

- It gives future work a home outside the weekly page
- It reduces pressure to overload the main narrative with speculative branches

What to watch:

- It is mostly manual copy
- Roadmap pages go stale faster than methods or results pages
- This page needs periodic pruning more than more structure

### `deep-dives/neuron-4288.html`

This page is a static appendix and reads like one.
That is not a problem by itself.

What is good:

- It isolates the 4288 artifact discussion from the main narrative
- It makes the "single hub neuron" withdrawal legible and explicit
- It uses static figures effectively

What is still true:

- Most numbers on the page are literals, not exporter-backed bindings
- If the underlying 4288 investigation changes, this page would need a manual refresh

For now that is acceptable because it is a forensic appendix, not a rolling dashboard.

### `deep-dives/swing-characterization.html`

This is one of the better hybrid pages.
It combines static figures with live bindings and chart-backed summaries.

What is good:

- Strong use of `data-swing-bind`
- Good separation between descriptive plots and derived metrics
- Better alignment with the current data-driven architecture than the 4288 appendix

This page is a good model for future appendix pages: static visuals where appropriate, but live numbers for repeated headline claims.

## Drift and Provenance Status

The current site has clear progress on provenance, but it is not finished.

### Current audit counts

| Page | Manual markers | Provenance comments |
|---|---:|---:|
| `index.html` | 3 | 5 |
| `story.html` | 6 | 7 |
| `results/gemma-3-4b.html` | 9 | 5 |
| `methods.html` | 0 | 0 |
| `extensions.html` | 5 | 0 |
| `deep-dives/neuron-4288.html` | 0 | 1 |
| `deep-dives/swing-characterization.html` | 0 | 0 |

### Reading those counts correctly

These counts are not a scorecard by themselves.

- A methods page should have fewer manual markers than a roadmap page.
- A deep-dive appendix may legitimately remain more static than a summary dashboard.
- The real warning sign is not "manual exists"; it is "quantitative claims can drift silently."

By that standard, the biggest remaining risk is still `results/gemma-3-4b.html`, followed by `story.html` and then `index.html`.

That ordering is now much closer than before:

- `results/gemma-3-4b.html` still leads because it is the densest evidence page
- `story.html` is now a near-second because more of the easy quantitative cleanup on the results page has already been done
- `index.html` remains lower risk because it is intentionally a current memo, not the canonical ledger

What changed since the prior audit:

- the largest quantitative JS drift source on the results page has been removed
- the intervention page copy now reads more of its peak counts and population splits from canonical JSON
- the story page now treats the anti-compliance intervention delta as a bound, provenance-tracked claim rather than manual trend prose
- the 4288 verdict/takeaway block now reads from a tracked structured summary rather than frozen literals
- the remaining risk on that page is now mostly editorial framing around already-bound evidence, not chart-array duplication

## What Is Good Enough Now

Several things from the old plan should now be considered settled unless the site grows materially again.

### Settled enough

- Multi-page split
- Shared design system
- Shared metric hydration
- Separate home for methods
- Separate home for stable results
- Separate home for roadmap material
- Separate home for appendix/deep-dive material

These are not current bottlenecks anymore.

### Intentionally deferred and still fine to defer

- `archive/`
- `backup/`
- Mistral results page
- Build system / templating system

Adding those prematurely would add structure without adding much signal.

## Main Remaining Weak Spots

### 1. Partial anti-drift coverage

The site has the right principle but incomplete coverage.
Some repeated metrics are nicely centralized, but the remaining static synthesis blocks still compress live evidence into prose that has to be updated by hand.

### 2. Nav and shell duplication

The nav markup is duplicated across the HTML files.
At the current size this is tolerable.
If page count grows further, it will become an editing tax.

### 3. No historical weekly trail

`index.html` currently behaves like "the current memo," not like a true weekly archive system.
That is fine, but it should be described honestly.
The site does not yet preserve meeting history as a first-class feature.

### 4. Mixed page maturity

Different pages are at different maturity levels:

- `methods.html` is close to reference-doc maturity
- `results/gemma-3-4b.html` is still an active evidence workbench
- `extensions.html` is a live roadmap
- the deep dives are appendix snapshots

That is normal, but it matters when deciding where to invest cleanup effort.

## Actual Priorities From Here

If this file is going to name priorities, they should reflect the site we actually have now.

### Priority 1: finish the remaining claim-contract pass on `story.html`

Highest-leverage cleanup:

- keep converting the remaining claim-adjacent story prose to either bindings or explicit provenance comments
- add tracked site-facing fields when the story needs a repeated quantitative comparison that is not yet exported cleanly
- keep tightening results-page prose where it still summarizes live evidence manually

This is still the most important remaining site-facing technical debt, but the easiest repeated numerics on the story page are no longer the bottleneck.

Current intervention/mechanism decision state on `story.html`:

- the anti-compliance `&alpha;=0.0` to `&alpha;=3.0` effect should stay a live binding from existing `site/data/intervention_sweep.json`, and that contract is now wired
- the committed `&alpha;=3.0` strict answer-text remap rate should also stay a live binding from the existing intervention export
- the standard raw MC-letter `&alpha;=3.0` point can also stay a live binding from the existing intervention export, and `story.html` now uses that bound comparison instead of implying it editorially
- any explicit remap-vs-raw lift claim should wait for a tracked exporter field rather than freezing an untracked delta into prose
- parse-failure trajectory numerics can stay on `results/gemma-3-4b.html` unless they are promoted into repeated story-page prose, at which point they should bind from the existing export rather than be restated manually
- the 4288 evidence-card takeaway can stay a live binding from the tracked top-neuron artifact payload, while longer appendix-style framing can remain manual unless it becomes repeated headline prose

### Priority 2: keep `results/gemma-3-4b.html` as the main ledger, but not the first cleanup target

That page still carries the most quantitative weight.
If results change, it is still the page most likely to go out of sync first, but the largest remaining sync risk is now manual framing around already-bound evidence rather than missing raw numerics or frozen 4288 verdict blocks.

### Priority 3: keep archive work deferred until there is real archival content

The earlier instinct to defer `archive/` was correct.
An empty archive is just navigation overhead.

Only add archival structure when there is an actual second or third weekly memo worth preserving.

### Priority 4: only introduce templating if page growth makes duplication painful

Right now a no-build static site is still a reasonable tradeoff.
The current bottleneck is data drift, not lack of a framework.

If the site grows to more model pages, more deep dives, or an archive section, then a lightweight generation step may become worth it.
Not yet.

## Recommended Working Model

The site is easiest to maintain if each page class is treated like a different document type:

| Page class | What it should be |
|---|---|
| `index.html` | Current memo |
| `story.html` | Defendable paper narrative |
| `results/*.html` | Quantitative ledger |
| `methods.html` | Measurement contract |
| `extensions.html` | Roadmap |
| `deep-dives/*.html` | Appendix / forensic detail |

That mental model is already visible in the current implementation.
The best next work is to reinforce it, not redesign it.

## Export Reliability Note

The recent classifier-structure improvement exposed a distinction that matters:

- runtime site reliability
- export-workflow reproducibility

Those are related, but they are not the same thing.

Current state:

- runtime reliability is better than before because the site reads one canonical classifier structure payload
- clean-checkout export reproducibility is also better because the site exporter now reads a tracked classifier-structure summary artifact
- checkpoint recomputation still exists, but only as an explicit local validation/update path

The right fix is **not**:

- put the old JS arrays back
- silently fall back to stale literals if the checkpoint is missing
- weaken the audit so missing structure passes quietly

Those options make the site easier to regenerate locally at the cost of making it easier to ship stale numbers.

The implemented fix is:

- `data/gemma3_4b/pipeline/classifier_structure_summary.json` is now tracked
- `scripts/export_site_data.py` reads that file during normal site export
- `uv run python scripts/export_site_data.py --refresh-classifier-structure-summary` updates it from the local checkpoint
- `uv run python scripts/export_site_data.py --validate-classifier-structure-summary` checks that the tracked summary still matches the local checkpoint

That follows the same pattern already used elsewhere in the repo: site exports should prefer committed structured artifacts over workstation-local state.

## Template Note: Clarity

The [Clarity template](https://github.com/lorenmt/clarity-template) is worth keeping in mind as a design reference, but not as the default architectural direction for this site.

### Most relevant parts

- Visual rhythm and figure containers from [`minimal.html`](https://raw.githubusercontent.com/lorenmt/clarity-template/main/minimal.html)
- Long-form research-page patterns from [`clarity.html`](https://raw.githubusercontent.com/lorenmt/clarity-template/main/clarity.html)
- Reusable presentation CSS from [`clarity/clarity.css`](https://raw.githubusercontent.com/lorenmt/clarity-template/main/clarity/clarity.css)

### Why it is only a partial fit

- Clarity is optimized for a polished single project page or research article.
- This repo already has a multi-page site with differentiated roles:
  [`index.html`](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/site/index.html),
  [`story.html`](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/site/story.html),
  [`results/gemma-3-4b.html`](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/site/results/gemma-3-4b.html),
  [`methods.html`](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/site/methods.html),
  and the two deep dives.
- The current technical constraint is data drift, not lack of visual polish.
- The current site already has custom data hydration and chart wiring in
  [`shared.js`](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/site/assets/shared.js)
  and [`charts.js`](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/site/assets/charts.js).
- A wholesale Clarity migration would likely add integration work without solving the main maintenance problem.

### Tradeoffs vs current approach

| Option | Upside | Cost |
|---|---|---|
| Borrow Clarity selectively | Better typography, hero treatment, figure containers, optional TOC patterns | Low risk; keep current data-driven architecture |
| Rebase the whole site on Clarity | Stronger visual identity for a flagship public page | High rewrite cost; likely friction with current JSON bindings and multi-page structure |
| Keep current site as-is | Preserves working architecture and lower maintenance overhead | Visual polish remains hand-rolled and less opinionated |

### When to consider it seriously

Adopt Clarity-inspired patterns more aggressively if one of these becomes true:

- the project needs a single canonical public-facing paper page distinct from the internal weekly/results site
- the narrative pages become much longer and would benefit from a richer article layout or TOC
- the site starts prioritizing presentation polish for external readers over rapid weekly iteration

### Current recommendation

Use Clarity as a visual reference library, not as a drop-in foundation.
The best fit is selective borrowing into the narrative-facing pages while keeping the current exporter-backed site data flow and page-role split intact.

## Bottom Line

The website is in a much better state than the old plan now implies.
The major scaling step already happened: the monolith was split, the shared assets exist, the main data export path exists, and the site has distinct page roles.

What remains is now a narrower maintenance queue, not a vague migration backlog:

- finish the remaining story-page claim provenance and contract pass
- then tighten the manual framing on the results page where it interprets already-bound evidence
- keep the results page synchronized with exporter-backed data
- keep the story and roadmap pages manually sharp and pruned
- avoid adding archive/build complexity before the content volume justifies it

In short: the site no longer needs a scaling plan.
It needs a maintenance model.

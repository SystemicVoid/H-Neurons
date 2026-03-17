# Site Scaling Plan

> Implementation guidelines for evolving `site/` from a single-page monolith into a maintainable, data-driven multi-page research presentation.
>
> **Status**: In progress — Session 11 completed on 2026-03-17.
> **Created**: 2026-03-16
> **Context**: The current `site/index.html` is a ~2100-line, 77KB hand-maintained HTML file containing all narrative, CSS, chart data, and JS. As the project grows (Mistral-24B replication, SAE features, conditional gating, weekly advisor meetings), this monolith will not scale.

---

## Core Principle: Data-Driven Static

**Every quantitative claim on the site must trace to a canonical data file in the repo, not to a hardcoded value in HTML or JS.**

This is the single most important constraint. It prevents drift when AI agents or humans edit the site, and it ensures the advisor always sees numbers that match the actual experiment outputs.

### Enforcement rules

1. Analysis scripts export results to `site/data/*.json` (or `data/` artifacts already in the repo).
2. Pages `fetch()` those JSON files at load time. Cards, charts, and inline metrics all read from the same object.
3. No metric value appears as a literal in HTML or JS chart arrays. If a number shows up in prose (e.g., "76.5% accuracy"), it must be injected from the data source or carry a `<!-- from: site/data/gemma_classifier.json -->` provenance comment.
4. When updating experiment results, update the JSON source file. The site reflects the change automatically.
5. For values that genuinely cannot be data-driven yet (narrative text, qualitative claims), mark them with `data-source="manual"` so future sessions know what still needs wiring.

### What this replaces in the current site

| Current pattern | Problem | Data-driven replacement |
|---|---|---|
| `const antiComplianceRates = [64.2, 65.4, ...]` in `<script>` | Duplicated from intervention JSONL, easy to drift | `fetch('data/intervention_sweep.json')` |
| `<div class="value val-teal">76.5%</div>` in metric cards | Same number in 3+ places | Template reads from `classifier.json` |
| `const layerData = [{layer: 0, count: 2}, ...]` | Duplicated from classifier pickle analysis | `fetch('data/layer_distribution.json')` |
| `const topNeurons = [{label: 'L20:N4288', weight: 12.169}, ...]` | Duplicated from classifier analysis | `fetch('data/top_neurons.json')` |
| Chart axis labels, titles hardcoded | No provenance | Data file includes `title`, `axis_label`, `n`, `ci_status` |

---

## Architecture: Multi-Page Split

### Page types and their roles

```
site/
├── index.html                    # "This Week" — advisor landing page
├── story.html                    # "Core Story" — current paper narrative
├── methods.html                  # Pipeline, prompts, eval, dataset caveats
├── results/
│   ├── gemma-3-4b.html           # Stable Gemma findings
│   └── mistral-24b.html          # (future)
├── deep-dives/
│   └── neuron-4288.html          # 6-panel investigation
├── extensions.html               # Roadmap: SAE, gating, safety, circuits
├── archive/
│   └── 2026-03-16.html           # First meeting (moved from index after meeting)
├── backup/
│   └── 2026-03-16/               # Raw prompts, extra plots, sanity checks
├── data/                         # JSON data files (fetched by pages)
│   ├── gemma_classifier.json
│   ├── layer_distribution.json
│   ├── top_neurons.json
│   ├── intervention_sweep.json
│   ├── neuron_4288_investigation.json
│   └── meetings/
│       └── 2026-03-16.json
├── assets/
│   ├── shared.css                # Extracted design system
│   ├── shared.js                 # Shared utilities (fade-in, counter, chart helpers)
│   ├── charts.js                 # Chart.js init + data-fetch wiring
│   └── figures/                  # Python-exported SVGs for stable results
│       ├── 01_single_neuron_auc.png  # (existing investigation PNGs)
│       ├── ...
│       └── 06_correlations.png
```

### Where current sections move

| Current section | Target page | Rationale |
|---|---|---|
| Hero (model stats, bottom line) | `index.html` (weekly) + `results/gemma-3-4b.html` (stable) | Hero stats are stable; weekly page gets a compact summary linking to full results |
| Meeting agenda | `index.html` | Changes every week |
| Pipeline + data funnel | `methods.html` | Stable background — advisor reads once |
| Pipeline reflection | `methods.html` | Accompanies pipeline |
| Main result (classifier metrics) | `results/gemma-3-4b.html` | Stable finding |
| Data leakage reflection | `results/gemma-3-4b.html` | Accompanies main result |
| Layer distribution | `results/gemma-3-4b.html` | Stable finding |
| Layer distribution reflection | `results/gemma-3-4b.html` | Accompanies distribution |
| Neuron 4288 surprise intro | `results/gemma-3-4b.html` (summary) + `deep-dives/neuron-4288.html` (full) | Summary links to deep dive |
| 6-panel investigation | `deep-dives/neuron-4288.html` | Finished analysis |
| Intervention results | `results/gemma-3-4b.html` | Stable (updated as CIs arrive) |
| Format puzzle + parse failures | `results/gemma-3-4b.html` | Part of intervention story |
| Hypotheses | `results/gemma-3-4b.html` or `story.html` | Depends on maturity |
| Takeaways | `story.html` | Part of paper narrative |
| Advisor questions | `index.html` | Changes each week |
| Next steps | `index.html` | Changes each week |
| Reproducibility note | `methods.html` | Stable |

---

## Navigation

### Target nav structure (6 items)

```
This Week · Core Story · Results · Methods · Extensions · Archive
```

- **This Week** → `index.html` — always the advisor's entry point
- **Core Story** → `story.html` — current claim, strongest evidence, biggest threat, falsification criteria
- **Results** → dropdown or page with links to `results/gemma-3-4b.html`, future model pages
- **Methods** → `methods.html` — pipeline, prompts, evaluator details
- **Extensions** → `extensions.html` — SAE, gating, safety overlap, circuit discovery roadmap
- **Archive** → `archive/` — reverse-chronological weekly updates

### Evolution path

1. **Phase 1** (now): Keep current nav but extract shared CSS/JS
2. **Phase 2** (first split): 3-4 pages with simple top nav
3. **Phase 3** (after 3-4 meetings): Full 6-item nav with archive

---

## Weekly Update Pattern: "Show Only New Stuff"

### Delta block template

Every weekly page (`index.html`) starts with a compact status diff:

```html
<div class="delta-block">
  <div class="delta-item delta-new">NEW: α=3.0 remap raises standard estimate to 72.1%</div>
  <div class="delta-item delta-updated">UPDATED: Population split withdrawn pending all-α text scoring</div>
  <div class="delta-item delta-decision">DECISION: 38-neuron vs 219-neuron story</div>
  <div class="delta-item delta-blocker">BLOCKER: Bootstrap CIs still missing</div>
</div>
```

### Weekly page structure (fixed order)

1. **Delta block** — what changed since last meeting
2. **Meeting agenda** — prioritized items with time estimates
3. **New evidence** — max 2–4 result sections with charts
4. **Decisions needed** — questions for advisor with options and tradeoffs
5. **Next steps** — proposed experiment order
6. **Backup links** — links to backup page, deep dives, raw data

### Weekly lifecycle

1. Before meeting: `index.html` contains this week's content
2. After meeting: move `index.html` → `archive/YYYY-MM-DD.html`, start fresh `index.html` for next week
3. Archive pages are read-only after archival

### Cap: each weekly page should be scannable in 5–8 minutes

If a section is too detailed for the weekly page, it belongs in a deep-dive or backup page with a link.

---

## Shared Components

### Reusable patterns to extract

The current site repeats these patterns extensively. Extract them as template functions in `shared.js`:

| Component | Current usage count | Template signature |
|---|---|---|
| **MetricCard** | ~12 instances | `renderMetricCard({label, value, detail, color, ciStatus})` |
| **ChartPanel** | 6 charts | `renderChart(canvasId, config, dataUrl)` |
| **InsightBox** | ~6 instances | `renderInsightBox({title, body})` |
| **ComparisonCards** | 3 instances | `renderComparison([{label, value, detail, highlight}])` |
| **StepCard** | ~10 instances | `renderStepCard({icon, title, body, borderColor})` |
| **StatusBadge** | 6 investigation panels | `renderBadge(verdict)` — `NEW`, `STABLE`, `PROVISIONAL`, `WITHDRAWN`, `ARTIFACT` |
| **PipelineFlow** | 1 instance | `renderPipeline(stages)` |
| **FunnelViz** | 1 instance | `renderFunnel(stages)` |
| **Reflection** | ~5 instances | `renderReflection({label, body})` |
| **DeltaItem** | (new) | `renderDelta({type, text})` |

### Data-driven rendering flow

```
1. Page loads shared.css + shared.js
2. Page calls fetch('data/gemma_classifier.json')
3. Data object passed to component render functions
4. Components inject into placeholder <div>s
5. Charts init after DOM insertion
```

### Minimal approach (no build step)

```html
<!-- In any page -->
<div id="classifier-metrics"></div>
<script type="module">
  import { renderMetricCard } from './assets/shared.js';
  const data = await fetch('data/gemma_classifier.json').then(r => r.json());
  document.getElementById('classifier-metrics').innerHTML =
    renderMetricCard({label: 'Accuracy', value: data.accuracy, detail: `n=${data.n}`, color: 'teal', ciStatus: data.ci_status});
</script>
```

---

## Chart and Figure Strategy

### Two modes

| Chart type | Tool | When to use |
|---|---|---|
| **Stable/final figures** | Python matplotlib/seaborn → exported SVG or PNG | Results that won't change. Paper-ready. Labels, error bars, colorblind-safe palettes baked in. |
| **Live/weekly charts** | Chart.js reading from JSON | Data still shifting. Quick turnaround. Useful for weekly advisor updates. |

### Chart hygiene rules (from good-practices/good-slides.md)

- **Label axes** — always, no exceptions
- **Include values on bars** — saves the advisor from reading the y-axis
- **Max 3–5 colors** — represent "before intervention", "control", "after intervention" or similar distinct conditions
- **Show `n=`** — in chart title or subtitle
- **Show CI status** — error bars if available, or explicit "no CI yet" annotation
- **Horizontal bars** for charts with text labels (already done for top-neurons chart ✓)
- **Log-log plots** — consider for scaling curves when relevant

### Current charts and their data sources

| Chart | Canvas ID | Current data source | Target JSON file |
|---|---|---|---|
| Classifier performance | `classifierChart` | Hardcoded array | `gemma_classifier.json` |
| Layer distribution | `layerChart` | Hardcoded `layerData` array | `layer_distribution.json` |
| Top neurons by weight | `topNeuronsChart` | Hardcoded `topNeurons` array | `top_neurons.json` |
| Intervention α-sweep | `interventionChart` | Hardcoded arrays | `intervention_sweep.json` |
| Parse failures | `parseFailureChart` | Hardcoded `parseFailureCounts` | `intervention_sweep.json` |
| Adjusted compliance | `adjustedComplianceChart` | Hardcoded arrays | `intervention_sweep.json` |
| Population dynamics | `populationChart` | Hardcoded arrays | `intervention_sweep.json` |

---

## Status Badges for Research Rigor

Every metric card and chart carries an explicit status:

| Badge | Meaning | Visual |
|---|---|---|
| `NEW` | First appearance this week | Amber background |
| `UPDATED` | Changed since last week | Teal background |
| `STABLE` | Settled result, unlikely to change | Muted/dim |
| `PROVISIONAL` | Numbers may change (e.g., pending CI, pending rescore) | Amber outline |
| `WITHDRAWN` | Previously reported, now retracted | Coral strikethrough |

This satisfies the AGENTS.md quantitative reporting standard: *"If uncertainty cannot yet be computed, flag the number explicitly as 'no CI'."*

---

## Performance Guardrails

The 77KB monolith is not a performance problem today. The guardrails prevent it from becoming one:

1. **Extract shared CSS/JS** — `shared.css` (~600 lines), `shared.js` (observers, counters, chart helpers), `charts.js` (Chart.js init)
2. **Load Chart.js only on pages that use charts** — `methods.html` doesn't need it
3. **Lazy-load images** — already using `loading="lazy"` on investigation PNGs ✓
4. **Prefer SVG** over PNG for stable figures where possible (smaller, crisper)
5. **Cap weekly pages** at scannable length (5–8 minutes)
6. **No SPA, no client routing, no hydration** — plain static pages with shared assets
7. **Consider removing** counter animations and progress bar — they don't help advisor comprehension and add maintenance noise

---

## "Core Story" Page (`story.html`)

A standing page that answers five questions, updated as the narrative evolves:

1. **Current claim** — what we believe the H-Neurons are / do
2. **Strongest evidence** — which experiments support the claim
3. **Biggest threat** — what could invalidate it
4. **What would falsify it** — specific experiments or observations
5. **Next paper-level decision** — what the advisor needs to weigh in on

This matches the good-slides advice: *"Get consistent feedback on the story of your paper."*

---

## Extensions Roadmap Page (`extensions.html`)

Track each workstream from `docs/extensions-ideas/`:

| Extension | Status | Feasibility | Blocker |
|---|---|---|---|
| SAE feature extraction | Not started | High (Gemma Scope available) | None |
| Conditional gating | Not started | High (inference hooks) | Needs uncertainty probe |
| Safety-neuron overlap | Not started | High (linear probes) | None |
| Circuit discovery | Not started | Low (VRAM limit) | 16GB GPU insufficient |
| Mistral-24B replication | Pipeline stages 1-4 done | High | Pending advisor decision |

Each entry links to its results page when work begins.

---

## Implementation Order

### Phase 1: Foundation (estimated: half day)

- [x] Extract `shared.css` from inline `<style>` block
- [x] Extract `shared.js` (IntersectionObserver, counter animation, scoreboard animation)
- [x] Extract chart configs into `charts.js`
- [x] All three files loaded by `index.html` — verify nothing breaks
- [x] Create `site/data/` directory

### Phase 2: Data extraction (estimated: half day)

- [x] Write `scripts/export_site_data.py` to export `site/data/intervention_sweep.json` from committed intervention artifacts with provenance notes
- [ ] Extend the exporter to classifier-derived site JSON files once a direct canonical source exists
- [x] Wire `charts.js` to `fetch()` these files instead of hardcoded arrays
- [x] Wire intervention metric cards and comparison tiles to read from fetched data
- [x] Verify intervention charts and bound summary widgets render identically over HTTP

### Phase 3: Page split (estimated: 1 day)

- [x] Create `methods.html` — move pipeline, funnel, prompts, reproducibility note
- [x] Create `results/gemma-3-4b.html` — move classifier results, layer distribution, intervention
- [x] Create `deep-dives/neuron-4288.html` — move 6-panel investigation
- [x] Slim `index.html` down to weekly-only content
- [x] Add shared nav to all pages touched so far
- [x] Add delta block template to `index.html`

### Phase 4: Story and extensions (estimated: half day)

- [x] Create `story.html` with 5-question structure
- [x] Create `extensions.html` with roadmap table
- [ ] Create `archive/` directory and archival workflow

### Phase 5: Polish (estimated: half day)

- [ ] Add status badges (`NEW`, `STABLE`, `PROVISIONAL`) to metric cards
- [ ] Add value labels to bar charts
- [ ] Add `n=` and CI status to chart subtitles
- [ ] Review color count per chart (max 5)
- [ ] Test all pages with `here-now` deployment

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-03-16 | Data-driven static over SPA | Minimal tooling, no build step, enforces canonical data sources |
| 2026-03-16 | No framework (11ty, React, etc.) yet | Plain HTML + shared JS sufficient until >5 pages with heavy boilerplate |
| 2026-03-16 | Weekly page = `index.html` | Advisor always opens the same URL, sees latest content |
| 2026-03-16 | JSON data files in `site/data/` | Scripts export, pages fetch — single source of truth |
| 2026-03-16 | Delay Gemma classifier JSON export until direct canonical source exists | Avoid backfilling site data from prose or presentation literals |
| 2026-03-16 | Derive standard parse-failure and parseable-subset series from per-alpha JSONL rows | `faitheval_standard/results.json` stores raw compliance totals but omits parse-failure counts and conditional parseable-subset rates |
| 2026-03-17 | Build `extensions.html` before `archive/` once `story.html` exists | The site now has a stable standing narrative but not enough archived weeks to justify archive-first navigation; the extensions roadmap is the higher-signal next destination |
| 2026-03-17 | Keep `extensions.html` scoped to four concrete workstreams for now | The near-term roadmap is clearer if it tracks only SAE decomposition, suppressive neurons, swing-sample characterization, and the refusal-overlap pilot rather than turning into a full backlog mirror of `docs/extensions-ideas/` |
| 2026-03-17 | Defer `archive/` until there is a real post-split weekly page to move | The site now has a stable six-page document set, but there is still nothing new enough to archive; polish and status-signaling are higher-value next than scaffolding an empty archive |

---

## Progress Log

| Date | Session | Commit subject | Summary | Next recommended slice |
|---|---|---|---|---|
| 2026-03-16 | Session 1 | `refactor(site): extract shared styles and runtime helpers` | Extracted shared CSS and non-chart runtime from `site/index.html`, rewired the page to load `site/assets/shared.css` and `site/assets/shared.js`, and created `site/data/` for future JSON exports. | Move the remaining inline chart bootstrapping into `site/assets/charts.js` without changing the current hardcoded data yet. |
| 2026-03-16 | Session 2 | `refactor(site): move chart bootstrapping into charts module` | Extracted the remaining Chart.js setup from `site/index.html` into `site/assets/charts.js`, rewired the page to load the new asset, and kept all chart data hardcoded so rendered behavior stays equivalent. | Add a small export script for `site/data/intervention_sweep.json` from committed intervention artifacts, with provenance fields and no classifier backfill from prose. |
| 2026-03-16 | Session 3 | `feat(site): export intervention sweep data` | Added `scripts/export_site_data.py` and generated `site/data/intervention_sweep.json` from committed FaithEval artifacts, including derived parse-failure and parseable-subset series, anti-compliance population structure, and explicit notes that strict answer-text remapping is only available for standard prompt α=3.0. | Wire the four intervention charts in `site/assets/charts.js` to fetch `site/data/intervention_sweep.json`, remove the hardcoded intervention arrays, and keep any still-manual intervention prose explicitly marked as partial where it depends on the α=3.0-only remap. |
| 2026-03-17 | Session 4 | `refactor(site): load intervention charts from exported sweep data` | Rewired the four intervention charts in `site/assets/charts.js` to fetch `site/data/intervention_sweep.json`, resolved the JSON path relative to the module so future nested pages can reuse it safely, and verified over local HTTP that the page fetches the JSON and instantiates all intervention charts with the expected series. | Wire the intervention metric cards and static comparison numbers in `site/index.html` to the same JSON so the intervention section stops duplicating values in HTML as well as JS. |
| 2026-03-17 | Session 5 | `refactor(site): hydrate intervention summary widgets from site data` | Bound the intervention metric cards and parse-failure comparison tiles in `site/index.html` to `site/data/intervention_sweep.json`, removed the old intervention literals from the bound HTML, and tagged the remaining intervention prose blocks as `data-source="manual"` where they still carry embedded numeric claims. | Start Phase 3 by creating `results/gemma-3-4b.html` and shared top navigation, moving the stable results sections out of the weekly landing page while leaving classifier-derived charts hardcoded until canonical exports exist. |
| 2026-03-17 | Session 6 | `feat(site): add gemma results page shell` | Added `site/results/gemma-3-4b.html` as the first dedicated stable-results page, introduced shared weekly/results navigation styling, and made `site/assets/charts.js` safe to load on pages that only render a subset of charts. | Trim `site/index.html` into a weekly landing page and stop loading chart assets on the root page once direct links into the new results page exist. |
| 2026-03-17 | Session 7 | `refactor(site): turn index into weekly landing page` | Rewrote `site/index.html` into a meeting-oriented briefing page with a delta block, agenda, decision points, and direct links into `site/results/gemma-3-4b.html`; removed Chart.js and `charts.js` from the root page so the stable visual evidence now lives only on the dedicated results URL. | Continue Phase 3 by relocating background material to `methods.html` and the full neuron-4288 investigation to `deep-dives/neuron-4288.html`, then expand shared navigation beyond the first two pages. |
| 2026-03-17 | Session 8 | `feat(site): add methods reference page` | Added `site/methods.html` as the standing home for the pipeline, data funnel, prompt framing, evaluator-format contract, and reproducibility notes; expanded the shared nav so the weekly and results pages can reach the restored background context directly. | Finish Phase 3 by moving the full neuron-4288 appendix into its own deep-dive page and making the results page link to that longer artifact rationale instead of carrying it inline. |
| 2026-03-17 | Session 9 | `feat(site): add neuron 4288 deep-dive page` | Added `site/deep-dives/neuron-4288.html` with the full six-panel investigation, updated shared navigation to include the nested appendix, and turned `site/results/gemma-3-4b.html` into a true summary page that points to the deep dive for the full artifact rationale. | Move into Phase 4 by creating `story.html` as the standing paper narrative and deciding whether `extensions.html` or `archive/` is the more useful next nav destination. |
| 2026-03-17 | Session 10 | `feat(site): add standing core-story page` | Added `site/story.html` as the durable five-question paper narrative, expanded the shared nav to connect the weekly, story, results, methods, and deep-dive pages, and kept the root page in its shorter meeting-memo role. | Continue Phase 4 by creating `extensions.html` as the next live destination, then defer `archive/` until a second weekly page exists to archive. |
| 2026-03-17 | Session 11 | `feat(site): add extensions roadmap page` | Added `site/extensions.html` as a standing roadmap page, expanded the shared nav across the weekly, story, results, methods, extensions, and deep-dive pages, and deliberately narrowed the roadmap to four concrete workstreams: SAE feature decomposition, suppressive-neuron investigation, swing-sample characterization, and a refusal-overlap safety pilot. | Leave `archive/` deferred until a real post-split weekly page exists, and move the next site slice to Phase 5 polish on status badges plus chart annotations. |

---

## Tomorrow Pickup

### Current state

- Phase 1 is complete: shared CSS, shared runtime, and chart bootstrapping are all extracted from `site/index.html`.
- The first site data contract now exists at `site/data/intervention_sweep.json`, and the intervention charts plus intervention summary widgets render from it over HTTP on the nested results page.
- `site/results/gemma-3-4b.html` now holds the stable Gemma findings, `site/methods.html` holds the background/materials context, and `site/deep-dives/neuron-4288.html` holds the full six-panel top-neuron appendix.
- `site/story.html` now holds the standing five-question paper narrative separate from the weekly memo and the raw appendix.
- `site/extensions.html` now holds the standing follow-on roadmap, intentionally scoped to SAE decomposition, suppressive neurons, swing-sample characterization, and the refusal-overlap pilot.
- `site/index.html` has been reduced to a weekly landing page with delta/agenda/decision sections, and the shared nav now connects six live pages.
- The exporter is intentionally scoped to intervention data only; classifier-derived site JSON remains deferred until a direct canonical source exists.

### Open issues / constraints

- Remaining intervention prose still carries embedded numeric claims, but those blocks are explicitly tagged `data-source="manual"` rather than silently drifting.
- The standard-prompt strict answer-text correction is only canonical for `alpha=3.0`; do not imply a full corrected sweep yet.
- Standard-prompt population breakdown remains withdrawn until text-based rescoring exists across all alpha values.
- `archive/` still does not exist, and there is not yet a second post-split weekly page to justify creating it.
- `extensions.html` is intentionally narrower than the full idea bank in `docs/extensions-ideas/`; keep the live page focused on the four branches with the sharpest near-term falsifiers.
- Classifier, layer-distribution, and top-neuron cards/charts remain hardcoded even though the intervention path now reads from site data.
- Confidence intervals are still missing, so all exported intervention metrics remain explicitly `no_ci_yet`.

### Recommended next slice

1. Start Phase 5 polish by adding explicit status badges (`NEW`, `STABLE`, `PROVISIONAL`) to the metric cards and summary callouts that still present live or uncertain claims.
2. Add value labels plus `n=` / CI-status annotations to the stable results charts so the charts carry their own reporting contract instead of relying on nearby prose.
3. Keep `archive/` deferred until there is an actual post-split weekly page to archive; do not add empty navigation for a page class that still has no content.
4. Leave classifier, layer-distribution, and top-neuron charts hardcoded until canonical exports exist; the polish pass should improve signaling without inventing pseudo-canonical JSON.
5. Re-verify over HTTP after the polish pass, especially the results-page JSON fetch and the nested deep-dive assets.

### Acceptance checks for tomorrow

- Status badges make it obvious which claims are stable, provisional, or newly updated without forcing the reader to infer that from prose.
- The results charts carry visible value labels plus `n=` / CI-status annotations under HTTP without breaking the existing JSON hydration path.
- The current six-page split remains intact: short weekly page, standing story page, stable results page, methods reference page, focused extensions roadmap, and image-backed deep dive.

---

## When to Reconsider This Plan

Revisit if any of these become true:

- Figures are mostly generated from notebooks and need automatic site publishing → consider **Quarto**
- 3+ collaborators editing content regularly → consider a static site generator with templates
- Weekly updates exceed ~10 archived pages and cross-linking becomes heavy → consider search/index
- Need executable/reproducible research pages → consider Quarto or Observable

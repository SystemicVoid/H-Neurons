# Site Scaling Plan

> Implementation guidelines for evolving `site/` from a single-page monolith into a maintainable, data-driven multi-page research presentation.
>
> **Status**: In progress — Session 2 completed on 2026-03-16.
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

- [ ] Write a small script (`scripts/export_site_data.py`) that reads existing artifacts and writes JSON files:
  - `site/data/gemma_classifier.json` — from `models/gemma3_4b_classifier.pkl` or existing analysis
  - `site/data/layer_distribution.json` — from classifier analysis
  - `site/data/top_neurons.json` — from classifier analysis
  - `site/data/intervention_sweep.json` — from `data/gemma3_4b/intervention/` outputs
- [ ] Wire `charts.js` to `fetch()` these files instead of hardcoded arrays
- [ ] Wire metric cards to read from fetched data
- [ ] Verify all charts and cards render identically to current

### Phase 3: Page split (estimated: 1 day)

- [ ] Create `methods.html` — move pipeline, funnel, prompts, reproducibility note
- [ ] Create `results/gemma-3-4b.html` — move classifier results, layer distribution, intervention
- [ ] Create `deep-dives/neuron-4288.html` — move 6-panel investigation
- [ ] Slim `index.html` down to weekly-only content
- [ ] Add shared nav to all pages
- [ ] Add delta block template to `index.html`

### Phase 4: Story and extensions (estimated: half day)

- [ ] Create `story.html` with 5-question structure
- [ ] Create `extensions.html` with roadmap table
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

---

## Progress Log

| Date | Session | Commit subject | Summary | Next recommended slice |
|---|---|---|---|---|
| 2026-03-16 | Session 1 | `refactor(site): extract shared styles and runtime helpers` | Extracted shared CSS and non-chart runtime from `site/index.html`, rewired the page to load `site/assets/shared.css` and `site/assets/shared.js`, and created `site/data/` for future JSON exports. | Move the remaining inline chart bootstrapping into `site/assets/charts.js` without changing the current hardcoded data yet. |
| 2026-03-16 | Session 2 | `refactor(site): move chart bootstrapping into charts module` | Extracted the remaining Chart.js setup from `site/index.html` into `site/assets/charts.js`, rewired the page to load the new asset, and kept all chart data hardcoded so rendered behavior stays equivalent. | Add a small export script for `site/data/intervention_sweep.json` from committed intervention artifacts, with provenance fields and no classifier backfill from prose. |

---

## When to Reconsider This Plan

Revisit if any of these become true:

- Figures are mostly generated from notebooks and need automatic site publishing → consider **Quarto**
- 3+ collaborators editing content regularly → consider a static site generator with templates
- Weekly updates exceed ~10 archived pages and cross-linking becomes heavy → consider search/index
- Need executable/reproducible research pages → consider Quarto or Observable

# Framing Remediation Backlog — Wave 2 Handoff

This file records what Wave 1 intentionally did **not** touch so the next cleanup pass can proceed without rediscovering scope.

## Intentionally out of scope in Wave 1

- `paper/draft/**`, including `paper/draft/AGENTS.md`
- `paper/icml/**`
- Historical AI review and literature notes that repeat the old anchor ranking but do not currently route repo defaults

## Why deferred

- `paper/draft/**` is still manuscript-facing and should be revised as a deliberate writing pass, not as incidental repo-governance cleanup.
- `paper/icml/**` was explicitly excluded.
- Many review/literature notes are historical context; touching all of them in Wave 1 would create churn without changing the repo's default routing behavior.

## Wave 2 cleanup targets

- `site/index.html`
- `site/methods.html`
- `site/progress/index.html`
- `site/story.html`
- `site/extensions.html`
- `site/progress/week-04-flagship-synthesis.html`
- `site/deep-dives/neuron-4288.html`
- `site/deep-dives/swing-characterization.html`
- `notes/act3-reports/mentor-review-strategic.md` (banner-only)
- `docs/archive/gpt-act3-deep-research-report.md` (banner-only)

## Search inventory for Wave 2

Rerun these searches before editing:

```bash
rg -n "cleanest|strongest anchor|center of gravity|load-bearing|main anchor|master anchor|only defended anchor|only load-bearing|headline-safe anchor|headline-safe|anchor case studies|three anchors|first anchor|anchor 1|anchor 2|anchor 3" notes docs site
```

```bash
rg -n "2026-04-11-strategic-assessment|canonical project framing|earned / not-earned boundary|current project argument|current framing governor|historical framing" notes docs site
```

```bash
rg -n "story.html#anchor|#anchor1|#anchor2|#anchor3" site
```

## Wave 2 acceptance check

Wave 2 is complete when:

- no live public router silently treats FaithEval as the obvious default hierarchy;
- the public site presents localization/control, control/externality, and measurement/conclusion as distinct evidence families;
- Week 4 reads as a historical snapshot rather than the current constitution;
- `paper/**` remains frozen for this pass;
- the repo still distinguishes historical framing from current framing governance.

# Routing And Reframing Handoff — 2026-04-21

> Status: Active implementation order for the site-and-notes reframing pass.
> Scope freeze: `paper/**` is frozen in this pass, including the archived long draft and review stack, plus `paper/icml/**`.

## Active implementation order

1. notes/docs scope freeze
2. live site routers
3. Week 4 historical snapshot
4. dependent site cleanup

## Live public surfaces

- `site/index.html`
- `site/story.html`
- `site/extensions.html`
- `site/methods.html`
- `site/progress/index.html`

These are the current routers. They should use question-first framing, name the evidence family only after the question is stated, and keep FaithEval as strong localization/control evidence without letting it act as a repo-wide default hierarchy.

## Historical handling

- `site/progress/week-04-flagship-synthesis.html` stays live as a dated April 8-14 snapshot.
- Historical notes/docs that still route readers should get warning banners only, not body rewrites.

## Current governor

For repo-wide framing defaults, route to `notes/2026-04-21-claim-framing-governance.md`.

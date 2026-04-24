# Site Guide

The site is a static presentation site with shared CSS/JS and generated JSON payloads. There is no build system.

## Working Model

- Serve the site over local HTTP when verifying pages that use `fetch()`; do not rely on `file://`.
- Use `site/site-maintenance-model.md` for the current architecture, drift risks, and page roles.
- Reuse `site/assets/shared.css`, `site/assets/shared.js`, and `site/assets/charts.js` instead of adding page-local systems.
- Keep repeated quantitative values data-bound where practical, not copied into multiple HTML literals.

## Data and Claims

- Regenerate site-facing JSON with:

```bash
uv run python scripts/export_site_data.py
```

- Quantitative site claims need uncertainty estimates and registry coverage in `docs/ci_manifest.json`.
- Before finishing site changes that touch quantitative claims, run:

```bash
uv run python scripts/audit_ci_coverage.py
```

## Verification

- For HTML/CSS/JS edits, serve `site/` locally and check the changed pages in a browser.
- For data-binding edits, verify that nested pages under `results/`, `deep-dives/`, and `progress/` still resolve shared assets and JSON correctly.

## Deployment

To redeploy the project site at its canonical URL:

```bash
scripts/infra/publish.sh site --slug aware-fresco-4a2q --client amp
```

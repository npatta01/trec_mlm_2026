# Task 2 report: progressive disclosure and first-view hierarchy

## Outcome

Implemented the approved first-view hierarchy for theme, question, idea, and
study pages. Each detail page now starts with a semantic record hero containing
a status, a plain-language takeaway, supporting context, an authored causal
visual when present, and one next action. Supporting evidence, limits/readiness,
relationships, and technical material are rendered in closed native
`<details class="hub-details">` sections.

Generic `hub-record-path` diagrams and renderer calls are removed. Existing
evidence and relationship material remains available inside the relevant
details sections. Index rows now expose only status, title, one takeaway, and a
next action or decision; theme and stage remain grouping labels.

## Implementation

- Added `_render_details(label, lines, *, open=False)` to the renderer.
- Reworked theme, question, idea, and study detail renderers around
  `hub-record-hero`, `hub-takeaway`, `hub-next-action`, and `hub-details`.
- Kept `_render_record_visual` optional and styled its `hub-causal-*` semantic
  structure as a desktop comparison and a mobile Today → Proposed → Decision
  test sequence.
- Added explicit editorial tokens, responsive headline/body scales, a 55ch
  first-view measure, accessible disclosure controls, dark-mode token
  overrides, forced-color treatment, reduced-motion compatibility, and print
  expansion of collapsed supporting material.
- Added renderer/resource regressions and a Chrome DevTools browser contract
  that focuses a rendered summary and confirms Enter changes `details.open`
  from `false` to `true`.

## Verification evidence

The following completed successfully after the final source changes:

```text
uv run --extra dev pytest tests/research_hub/test_render.py tests/research_hub/test_resources.py tests/research_hub/test_site_integration.py -q
# 56 passed

uv run ruff check trec_mlm/research_hub/render.py tests/research_hub/test_render.py tests/research_hub/test_resources.py tests/research_hub/test_site_integration.py
# All checks passed

uv run python scripts/build_research_hub.py
quarto render research/hub --quiet
uv run python scripts/audit_research_hub_site.py --root . --output-dir research/hub/_site
# all exited 0; audit found no reader-facing QMD source exposure
```

## Self-review

- Confirmed no renderer, generated page, or stylesheet reference retains
  `hub-record-path`.
- Confirmed first-view semantic ordering and closed evidence sections across
  theme, question, idea, and study pages.
- Confirmed generated assets were refreshed by the canonical build and render.
- Confirmed dark-mode paper, ink, risk, and proposal tokens preserve contrast
  after introducing the editorial paper background.

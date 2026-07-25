# Research Hub Visual Language Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the approved Inter-based, dark/grayscale visual system and generate plain-English explainer pages for every recorded idea or planned test.

**Architecture:** Keep Markdown records canonical. Extend the existing deterministic Python renderer with generated idea indexes and detail pages, and let Quarto turn those outputs into semantic HTML. Use one local font asset and CSS-only visual patterns; do not add a visualization framework or hand-edit generated HTML.

**Tech Stack:** Python 3.12, pytest, Quarto 1.9, SCSS/CSS, Markdown/YAML.

## Global Constraints

- Inter is used for headings and body text and is bundled locally.
- Dark mode is the primary visual reference; light mode remains supported.
- Meaning must survive grayscale, forced colors, CSS removal, and narrow screens.
- Record counts are unbounded and indexes expand automatically.
- Generated language comes only from explicit fields or deterministic templates.
- Quarto and Markdown remain canonical; generated HTML is never edited.
- The private tailnet portal remains the only deployment target.

---

### Task 1: Generate the complete idea library and individual explainers

**Files:**
- Modify: `trec_mlm/research_hub/render.py`
- Modify: `trec_mlm/research_hub/validate.py`
- Modify: `trec_mlm/research_hub/__init__.py`
- Modify: `tests/research_hub/conftest.py`
- Modify: `tests/research_hub/test_render.py`
- Modify: `tests/research_hub/test_validate.py`
- Modify: `tests/research_hub/test_build.py`

**Interfaces:**
- Consumes: `HubBuild.sources.tests: tuple[HubRecord, ...]`.
- Produces: `render_tests_index_qmd(build: HubBuild) -> str`.
- Produces: `render_test_pages(build: HubBuild) -> dict[Path, str]`.
- Produces generated paths `research/hub/generated/ideas-index.qmd` and `research/hub/generated/ideas/<test_id>.qmd`.

- [ ] **Step 1: Write failing renderer tests**

```python
def test_idea_index_and_details_expand_from_canonical_records(
    sample_build: HubBuild,
) -> None:
    outputs = render_hub_outputs(sample_build)
    index = outputs[Path("research/hub/generated/ideas-index.qmd")]
    detail = outputs[
        Path("research/hub/generated/ideas/test-candidate-evidence.qmd")
    ]
    assert "Test candidate evidence." in index
    assert "hub-idea-row" in index
    assert "What this idea means" in detail
    assert "Why it matters" in detail
    assert "How it would work" in detail
    assert "What is missing" in detail
    assert "What happens next" in detail
    assert "Related research" in detail
```

- [ ] **Step 2: Run the renderer test and verify the missing-output failure**

Run: `uv run --extra dev pytest tests/research_hub/test_render.py::test_idea_index_and_details_expand_from_canonical_records -q`

Expected: fail because `research/hub/generated/ideas-index.qmd` is absent.

- [ ] **Step 3: Require explicit explainer fields for test records**

Add `plain_english`, `why_it_matters`, `mechanism_steps`, and `next_action` to
test validation. `mechanism_steps` is a nonempty string list. Keep readiness
fields independent so an idea may explain itself while remaining unready.

```python
def validate_test(record: HubRecord) -> None:
    require_string(record, "plain_english")
    require_string(record, "why_it_matters")
    mechanism_steps = require_string_list(record, "mechanism_steps")
    if not mechanism_steps:
        raise ValueError(
            f"{record.source_path}: mechanism_steps must not be empty"
        )
    require_string(record, "next_action")
```

- [ ] **Step 4: Implement deterministic index and detail renderers**

Render an open row per test with title, plain-English explanation, stage,
priority, and readiness count. Render each detail page in this order:
plain-English meaning, why it matters, numbered mechanism steps, current
evidence/body, missing readiness fields, next action, and linked themes and
questions. Escape source text with the existing Markdown/HTML helpers.

```python
def render_test_pages(build: HubBuild) -> dict[Path, str]:
    return {
        _validate_output_path(
            Path(f"research/hub/generated/ideas/{test.record_id}.qmd")
        ): _render_test_page(build, test).rstrip() + "\n"
        for test in sorted(build.sources.tests, key=lambda row: row.record_id)
    }
```

- [ ] **Step 5: Add outputs to the deterministic build contract**

Add the index to `_STATIC_OUTPUT_PATHS`, add the index and page mapping to
`render_hub_outputs`, export public render functions, and update exact-path
tests.

- [ ] **Step 6: Run focused tests**

Run: `uv run --extra dev pytest tests/research_hub/test_render.py tests/research_hub/test_validate.py tests/research_hub/test_build.py -q`

Expected: all focused tests pass.

- [ ] **Step 7: Commit**

```bash
git add trec_mlm/research_hub tests/research_hub
git commit -m "feat: generate accessible idea explainers"
```

### Task 2: Enrich the recorded ideas and connect Quarto to generated pages

**Files:**
- Modify: `research/hub/ideas/failure-aware-fusion.qmd`
- Modify: `research/hub/ideas/reference-review-trigger.qmd`
- Modify: `research/hub/ideas/semantic-failure-clusters.qmd`
- Modify: `research/hub/ideas/index.qmd`
- Modify: `research/hub/_quarto.yml`
- Modify: `research/hub/catalog.json` through the generator
- Modify: `research/catalog.json` through the generator
- Test: `tests/research_hub/test_site_integration.py`

**Interfaces:**
- Consumes the fields enforced by `validate_test`.
- Produces rendered idea routes under `generated/ideas/<test_id>.html`.

- [ ] **Step 1: Write a failing site integration test**

```python
def test_quarto_renders_generated_idea_library(
    project_root: Path,
) -> None:
    outputs = render_hub_outputs(build_research_hub(project_root))
    assert Path("research/hub/generated/ideas-index.qmd") in outputs
    assert all(
        "generated/ideas/" in path.as_posix()
        for path in outputs
        if path.as_posix().startswith("research/hub/generated/ideas/")
    )
```

- [ ] **Step 2: Run the integration test and verify failure**

Run: `uv run --extra dev pytest tests/research_hub/test_site_integration.py::test_quarto_renders_generated_idea_library -q`

Expected: fail until Quarto and links use the generated routes.

- [ ] **Step 3: Add explicit plain-language explainer fields**

Add accurate, reviewable values for each existing idea:

```yaml
plain_english: "Use a reliability check to choose which evidence sources should influence each request."
why_it_matters: "Response and reference evidence help unevenly, so a fixed mixture can hide failures."
mechanism_steps:
  - "Read the new request and estimate which evidence sources are dependable."
  - "Choose a bounded evidence mixture using the frozen reliability rule."
  - "Rank candidates and compare with the frozen dense/profile control."
next_action: "Define and freeze the reliability signal, fallback, and artifact plan."
```

Use equally concrete, source-faithful fields for the reference-review and
semantic-failure-cluster ideas.

- [ ] **Step 4: Route Quarto and the index to generated idea pages**

Replace the grid listing in `ideas/index.qmd` with an include of
`../generated/ideas-index.qmd`. Render `generated/ideas/*.qmd`, stop rendering
canonical `ideas/*.qmd` as public pages, and update test links plus catalog
paths to `generated/ideas/<id>.html`.

- [ ] **Step 5: Build and run integration tests**

Run: `uv run python scripts/build_research_hub.py && uv run --extra dev pytest tests/research_hub/test_site_integration.py -q`

Expected: generator and site integration tests pass.

- [ ] **Step 6: Commit**

```bash
git add research/hub research/catalog.json tests/research_hub
git commit -m "content: explain open research ideas"
```

### Task 3: Apply the approved typography and grayscale visual system

**Files:**
- Create: `research/hub/assets/fonts/Inter-Regular.otf`
- Create: `research/hub/assets/fonts/Inter-SemiBold.otf`
- Modify: `research/hub/assets/theme.scss`
- Modify: `research/hub/_quarto.yml`
- Test: `tests/research_hub/test_resources.py`

**Interfaces:**
- Consumes semantic classes emitted by Task 1: `hub-idea-row`,
  `hub-idea-hero`, `hub-idea-status`, `hub-process`, and `hub-process-step`.
- Produces local font loading and responsive, forced-color-safe presentation.

- [ ] **Step 1: Write failing resource/style tests**

```python
def test_hub_theme_uses_local_inter_and_grayscale_safe_idea_visuals(
    project_root: Path,
) -> None:
    theme = (project_root / "research/hub/assets/theme.scss").read_text()
    assert '@font-face' in theme
    assert 'Inter-Regular.otf' in theme
    assert 'Inter-SemiBold.otf' in theme
    assert '.hub-process-step' in theme
    assert '@media (forced-colors: active)' in theme
    assert '@media (prefers-reduced-motion: reduce)' in theme
```

- [ ] **Step 2: Run the resource test and verify failure**

Run: `uv run --extra dev pytest tests/research_hub/test_resources.py::test_hub_theme_uses_local_inter_and_grayscale_safe_idea_visuals -q`

Expected: fail because the local font and idea visual rules are absent.

- [ ] **Step 3: Add the local Inter asset**

Copy the installed Inter regular and semibold OTF files into
`research/hub/assets/fonts/`, add the fonts directory to Quarto resources, and
declare both local files with `font-display: swap`.

- [ ] **Step 4: Apply the approved tokens and responsive styles**

Set Inter for headings and body; use a 17-pixel root, body line height near
1.6, 65–75 character prose measure, stronger neutral contrast, open dividers,
and minimal color. Render process steps in columns on desktop and a numbered
vertical sequence below 768 pixels. Use labels, borders, and dashed/double line
styles so no status depends on hue.

- [ ] **Step 5: Run resource tests and render**

Run: `uv run --extra dev pytest tests/research_hub/test_resources.py -q && quarto render research/hub --quiet`

Expected: tests pass and Quarto renders successfully.

- [ ] **Step 6: Commit**

```bash
git add research/hub/assets research/hub/_quarto.yml tests/research_hub
git commit -m "style: apply grayscale research hub theme"
```

### Task 4: Verify representative pages and refresh the private portal

**Files:**
- Modify: `docs/superpowers/plans/2026-07-25-research-hub-visual-language.md` with checked steps and evidence.
- Derived: `research/hub/_site/`
- Derived private copy: `/home/npatta01/codex-rendered/plans/trec-2026-research-hub/`

**Interfaces:**
- Consumes the generated Quarto site.
- Produces a verified private tailnet preview.

- [x] **Step 1: Run the complete automated gate**

```bash
uv run --extra dev pytest tests/research_hub -q
uv run python scripts/build_research_hub.py
quarto render research/hub --quiet
uv run python scripts/build_research_hub.py --check
uv run --extra dev ruff check trec_mlm/research_hub scripts/build_research_hub.py scripts/normalize_research_hub_links.py tests/research_hub
git diff --check
```

Expected: every command exits zero.

Verified 2026-07-25: `pytest tests/research_hub -q` reported 102 passed;
the deterministic generator, Quarto render, generator `--check`, Ruff, and
`git diff --check` each exited zero.

- [x] **Step 2: Inspect representative pages**

Capture and inspect summary, theme, idea index, idea detail, question, and study
pages at 1440×1000 dark mode and 390×844 dark mode. Confirm no horizontal
overflow, clipped copy, low-contrast metadata, or meaning conveyed only by
color.

Verified 2026-07-25: directly inspected true-dark screenshots for summary,
theme, idea index/detail, question, and study at both viewports. No horizontal
overflow or clipped active copy was observed; metadata remains legible and
status distinctions retain text labels and borders in grayscale.

- [x] **Step 3: Sync only the rendered site**

Run:

```bash
rsync -a --delete research/hub/_site/ /home/npatta01/codex-rendered/plans/trec-2026-research-hub/
```

Verified 2026-07-25: synced only `research/hub/_site/`; destination contains
195 rendered files.

- [x] **Step 4: Verify private delivery**

Confirm `tailscale serve status` reports
`https://npatta01-framework.tail481212.ts.net (tailnet only)`, key routes return
HTTP 200, retired raw interactive bundles return HTTP 404, and internal links
remain inside the rendered root.

Verified 2026-07-25: `tailscale serve status` reports
`https://npatta01-framework.tail481212.ts.net (tailnet only)`. Key hub routes
returned 200; retired interactive-bundle paths returned 404. The rendered-site
audit covered 162 HTML pages and 2,666 internal links, with zero missing or
root-escaping targets.

- [x] **Step 5: Commit verification record**

```bash
git add docs/superpowers/plans/2026-07-25-research-hub-visual-language.md
git commit -m "docs: record visual language verification"
```

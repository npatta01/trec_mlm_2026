# Research Hub Progressive Visual Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace text-heavy first views and generic process diagrams with one takeaway, an optional source-defined meaning-first visual, one next action, and progressively disclosed evidence.

**Architecture:** Canonical Markdown records may declare a validated `visual` mapping. The deterministic Python generator renders only declared visual models; records without one receive no placeholder diagram. Semantic HTML `<details>` sections preserve all evidence while keeping it out of the first visual scan.

**Tech Stack:** Python 3.12, pytest, Quarto 1.9, semantic HTML, SCSS/CSS.

## Global Constraints

- The first viewport presents one takeaway, one meaningful visual when defined, and one next action.
- Generic step diagrams are removed rather than used as filler.
- Visual text comes only from explicit canonical Markdown fields.
- Warm off-white, near-black, rust, and teal form the editorial palette.
- Color reinforces labels but never replaces them.
- Evidence remains available to humans and the machine-readable catalog.
- The site remains responsive, keyboard accessible, grayscale safe, and tailnet only.
- No JavaScript visualization framework or hand-edited generated HTML.

---

### Task 1: Validate and render explicit causal-comparison visuals

**Files:**
- Modify: `trec_mlm/research_hub/validate.py`
- Modify: `trec_mlm/research_hub/render.py`
- Modify: `research/hub/ideas/failure-aware-fusion.qmd`
- Modify: `research/hub/templates/test.qmd.example`
- Modify: `tests/research_hub/conftest.py`
- Modify: `tests/research_hub/test_validate.py`
- Modify: `tests/research_hub/test_render.py`
- Modify: `tests/research_hub/test_site_integration.py`

**Interfaces:**
- Consumes: optional `HubRecord.metadata["visual"]` mapping.
- Produces: `validate_visual(record: HubRecord) -> None`.
- Produces: `_render_record_visual(record: HubRecord) -> list[str]`.
- Supports initial model: `causal_comparison`.

- [ ] **Step 1: Add failing visual-validation tests**

```python
def test_validation_rejects_incomplete_causal_comparison(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    idea = replace(
        minimal_sources.tests[0],
        metadata={
            **minimal_sources.tests[0].metadata,
            "visual": {
                "type": "causal_comparison",
                "current_label": "Today",
            },
        },
    )
    with pytest.raises(ValueError, match="current_summary"):
        validate_hub_sources(
            replace(minimal_sources, tests=(idea,)),
            ledger_build,
            tmp_path,
        )
```

Also test that an unsupported `visual.type` is rejected and that records without
`visual` remain valid.

- [ ] **Step 2: Run validation tests and verify failure**

Run: `uv run --extra dev pytest tests/research_hub/test_validate.py -k visual -q`

Expected: fail because visual metadata is not validated.

- [ ] **Step 3: Implement strict visual validation**

```python
VISUAL_TYPES = frozenset({"causal_comparison"})

def validate_visual(record: HubRecord) -> None:
    visual = record.metadata.get("visual")
    if visual is None:
        return
    if not isinstance(visual, Mapping):
        raise ValueError(f"{record.source_path}: visual must be a mapping")
    visual_type = visual.get("type")
    if visual_type not in VISUAL_TYPES:
        raise ValueError(
            f"{record.source_path}: visual.type must be one of "
            f"{sorted(VISUAL_TYPES)}"
        )
```

For `causal_comparison`, require nonempty strings for `current_label`,
`current_summary`, `proposed_label`, `proposed_summary`, and `decision_test`;
a nonempty string list for `current_items`; and a nonempty list of mappings
whose `label` and `action` are nonempty strings for `proposed_routes`. Reject
unknown visual keys so misspellings cannot silently disappear.

- [ ] **Step 4: Add failing renderer tests**

```python
def test_causal_visual_uses_only_explicit_record_text(
    sample_build: HubBuild,
) -> None:
    idea = replace(
        sample_build.sources.tests[0],
        metadata={
            **sample_build.sources.tests[0].metadata,
            "visual": CAUSAL_VISUAL,
        },
    )
    build = replace(
        sample_build,
        sources=replace(sample_build.sources, tests=(idea,)),
    )
    detail = render_test_pages(build)[
        Path("research/hub/generated/ideas/test-candidate-evidence.qmd")
    ]
    assert "hub-causal-comparison" in detail
    for text in (
        "One fixed evidence mixture creates uneven failures.",
        "A reliability gate chooses the evidence mixture.",
        "Does the adaptive mixture beat the frozen control?",
    ):
        assert text in detail
```

Add a second test asserting a record without `visual` contains no
`hub-causal-comparison`, `hub-process`, or generic placeholder diagram.

- [ ] **Step 5: Implement semantic causal rendering**

Render:

```html
<figure class="hub-causal-comparison">
  <section class="hub-causal-current">...</section>
  <div class="hub-causal-change">Proposed change</div>
  <section class="hub-causal-proposed">...</section>
  <figcaption class="hub-decision-test">...</figcaption>
</figure>
```

Use `html.escape` for every source value. Keep the `figcaption` and nearby
headings sufficient to recover meaning without CSS.

- [ ] **Step 6: Add the approved visual source data**

Add the exact approved causal-comparison mapping to
`failure-aware-fusion.qmd`. Add a commented causal visual example to the
test template without making visuals mandatory for every record.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_validate.py tests/research_hub/test_render.py tests/research_hub/test_site_integration.py -q
uv run python scripts/build_research_hub.py
uv run python scripts/build_research_hub.py --check
```

Expected: all commands exit zero.

```bash
git add trec_mlm/research_hub research/hub/ideas research/hub/templates tests/research_hub
git commit -m "feat: add meaning-first research visuals"
```

### Task 2: Reduce first-view density and add progressive disclosure

**Files:**
- Modify: `trec_mlm/research_hub/render.py`
- Modify: `research/hub/assets/theme.scss`
- Modify: `tests/research_hub/test_render.py`
- Modify: `tests/research_hub/test_resources.py`
- Modify: `tests/research_hub/test_site_integration.py`

**Interfaces:**
- Consumes: `_render_record_visual(record)`.
- Produces: `_render_details(label: str, lines: Sequence[str], *, open: bool = False) -> list[str]`.
- Produces semantic classes `hub-record-hero`, `hub-takeaway`,
  `hub-next-action`, `hub-details`, and `hub-causal-*`.

- [ ] **Step 1: Add failing hierarchy and disclosure tests**

```python
def test_idea_first_view_precedes_collapsed_evidence(
    sample_build: HubBuild,
) -> None:
    detail = render_test_pages(sample_build)[
        Path("research/hub/generated/ideas/test-candidate-evidence.qmd")
    ]
    assert detail.index("hub-takeaway") < detail.index("hub-details")
    assert detail.index("hub-next-action") < detail.index("hub-details")
    assert '<details class="hub-details">' in detail
    assert "<summary>Evidence and current knowledge</summary>" in detail
```

Add representative theme, question, and study assertions proving generic
`hub-process hub-record-path` diagrams are absent, one takeaway and one next
action precede collapsed details, and evidence text is preserved inside
`<details>`.

- [ ] **Step 2: Run focused renderer tests and verify failure**

Run: `uv run --extra dev pytest tests/research_hub/test_render.py -k 'first_view or progressive or generic' -q`

Expected: fail because details are expanded and generic paths remain.

- [ ] **Step 3: Implement the shared progressive structure**

```python
def _render_details(
    label: str,
    lines: Sequence[str],
    *,
    open: bool = False,
) -> list[str]:
    open_attribute = " open" if open else ""
    return [
        f'<details class="hub-details"{open_attribute}>',
        f"<summary>{html.escape(label)}</summary>",
        *lines,
        "</details>",
    ]
```

Reorder theme, question, idea, and study detail pages:

1. record label/status;
2. plain-English takeaway and one supporting sentence;
3. optional `_render_record_visual`;
4. decision test when defined;
5. one next action;
6. collapsed evidence, limits/readiness, relationships, and technical details.

Remove `_record_path` diagrams and their calls. Preserve every existing evidence
line inside an appropriate details section.

- [ ] **Step 4: Simplify index rows**

Theme, question, idea, and study initial rows show only status, title, one
takeaway, and next action/decision. Keep theme/stage/readiness as grouping or
filter labels. Remove duplicate summary/current-answer paragraphs from the
initial row.

- [ ] **Step 5: Add the approved editorial CSS**

Define explicit tokens:

```scss
:root {
  --hub-paper: #f2efe7;
  --hub-ink: #171713;
  --hub-risk: #a64b21;
  --hub-proposal: #007b72;
}
```

Use a responsive headline scale up to 4.5rem, body copy at 1.05rem–1.125rem,
high-contrast supporting text, and a maximum first-view measure of 55ch.
Implement side-by-side causal comparison on desktop and a vertical
Today → Proposed → Decision test sequence below 768 pixels. Style `<details>`
with a 44-pixel minimum summary target, visible focus, and explicit open/closed
indicators. Preserve dark, grayscale, forced-colors, reduced-motion, and print
behavior.

- [ ] **Step 6: Add resource and keyboard-contract tests**

Assert CSS contains all new semantic classes, `details[open]`, `:focus-visible`,
mobile causal stacking, forced-colors rules, and no `.hub-record-path`.
Render the site and use the browser harness to focus and toggle a details
summary with Enter, confirming `details.open` changes from false to true.

- [ ] **Step 7: Run focused tests, render, and commit**

```bash
uv run --extra dev pytest tests/research_hub/test_render.py tests/research_hub/test_resources.py tests/research_hub/test_site_integration.py -q
uv run python scripts/build_research_hub.py
quarto render research/hub --quiet
uv run python scripts/audit_research_hub_site.py --root . --output-dir research/hub/_site
```

Expected: all commands exit zero and no QMD source is exposed.

```bash
git add trec_mlm/research_hub research/hub/assets research/hub/generated tests/research_hub
git commit -m "style: reduce research hub first-view density"
```

### Task 3: Verify visuals and refresh the private portal

**Files:**
- Modify: `docs/superpowers/plans/2026-07-25-research-hub-progressive-visual-hierarchy.md` with checked steps and evidence.
- Derived: `research/hub/_site/`
- Derived private copy: `/home/npatta01/codex-rendered/plans/trec-2026-research-hub/`

**Interfaces:**
- Consumes the rendered Quarto site.
- Produces the refreshed tailnet-only preview.

- [x] **Step 1: Run the fresh complete gate**

```bash
uv run --extra dev pytest tests/research_hub -q
uv run python scripts/build_research_hub.py
quarto render research/hub --quiet
uv run python scripts/build_research_hub.py --check
uv run --extra dev ruff check trec_mlm/research_hub scripts/build_research_hub.py scripts/normalize_research_hub_links.py scripts/audit_research_hub_site.py tests/research_hub
uv run python scripts/audit_research_hub_site.py --root . --output-dir research/hub/_site
git diff --check
```

Expected: every command exits zero.

- [x] **Step 2: Capture and inspect representative pages**

Inspect summary, theme, idea index, causal idea detail, question, and study
pages at 1440×1000 and 390×844 in light and dark modes. Also inspect the causal
detail in grayscale, accent-disabled, and forced-colors modes. Confirm:

- one clear first-view takeaway;
- only one primary visual and next action;
- details below the first-view hierarchy;
- no horizontal overflow;
- no unclear unlabeled arrows or nodes;
- readable color and type hierarchy.

- [x] **Step 3: Verify progressive disclosure interaction**

On desktop and mobile, use keyboard focus plus Enter to open each representative
details summary. Confirm the control state is visible and the expanded content
does not move focus or navigate.

- [x] **Step 4: Sync and probe the private portal**

```bash
rsync -a --delete research/hub/_site/ /home/npatta01/codex-rendered/plans/trec-2026-research-hub/
tailscale serve status
```

Confirm the mapping says `tailnet only`; key pages return 200; canonical QMD and
retired raw bundles return 404.

- [x] **Step 5: Commit verification evidence**

```bash
git add docs/superpowers/plans/2026-07-25-research-hub-progressive-visual-hierarchy.md
git commit -m "docs: verify progressive visual hierarchy"
```

#### Task 3 verification evidence (2026-07-25)

- Fresh complete gate passed after the final renderer change: Research Hub tests,
  generator build/check, Quarto render, Ruff, publication audit, and
  `git diff --check` all exited zero.
- Chrome DevTools captured and visually inspected 27 actual page viewports:
  summary, generated theme, idea index, causal idea detail, question, and
  study at 1440×1000 and 390×844 in light and true Quarto dark mode, plus
  causal detail grayscale, accent-disabled, and forced-colors views. Evidence:
  `.superpowers/sdd/2026-07-25-research-hub-progressive-visual-hierarchy/visual-captures/`.
  All captures had no horizontal overflow. The causal visual is the only
  primary visual and now labels its transition `Proposed change`; other pages
  show no placeholder diagram. The visual review found readable hierarchy and
  closed details below the first-view takeaway/action.
- The browser focused and pressed Enter on all 24 rendered disclosure summaries
  (theme, causal idea, question, and study; desktop and mobile). Every control
  retained focus and path, changed `details.open` from false to true, had a
  visible outline, and met the 44px minimum target. Summary and idea-index
  pages have no disclosure summaries.
- Synced `/home/npatta01/codex-rendered/plans/trec-2026-research-hub/` using
  the required delete-aware rsync. `tailscale serve status` reported
  `https://npatta01-framework.tail481212.ts.net (tailnet only)`. The portal
  returned 200 for the hub index and representative idea, question, and study
  pages; canonical `.qmd` and retired `.zip` paths returned 404.

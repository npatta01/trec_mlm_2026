# Research Hub Content Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the partially hand-authored Research Hub with a source-backed, theme-organized Quarto site whose indexes regenerate automatically from canonical Markdown records and the experiment ledger.

**Architecture:** Add `theme` as a first-class Research Hub record and make `Theme → question → study → run` the generated navigation hierarchy. Human-authored facts stay in QMD front matter/prose; Python validation and rendering derive summary, theme, question, study, Background, history, coverage, run, and catalog views. Quarto pre-render runs the generator, while an allowlisted post-render step stages existing sanitized deep-dive reports.

**Tech Stack:** Python 3.11+, dataclasses, PyYAML, pytest, Quarto 1.9.38, Markdown/QMD, SCSS, static JavaScript only where existing filtering requires it

## Global Constraints

- The canonical content layer is Markdown/QMD plus `experiments/ledger.json`; HTML is always derived.
- The main page shows themes, cross-theme open questions, and next tests; it never enumerates studies or tested systems.
- The visual hierarchy is `Theme → research question → study → run`.
- Use whitespace and dividers before boxes; reserve a filled treatment for the theme conclusion.
- A study index row has one action: opening the study detail page.
- No line counts, file sizes, placeholder citations, invented candidate IDs, or button clusters appear in reader-facing pages.
- All effectiveness claims remain explicitly development-only unless a canonical source says otherwise.
- No raw private responses, secrets, local filesystem paths, or external executable assets enter the rendered site.
- Only allowlisted existing reports may be staged as deep-dive resources.
- The portal remains private and tailnet-only; do not push, publish publicly, or change sharing.

---

### Task 1: Add themes and migrate canonical relationships

**Files:**
- Modify: `trec_mlm/research_hub/model.py`
- Modify: `trec_mlm/research_hub/load.py`
- Modify: `trec_mlm/research_hub/validate.py`
- Modify: `tests/research_hub/conftest.py`
- Modify: `tests/research_hub/test_load.py`
- Modify: `tests/research_hub/test_validate.py`
- Create: `research/hub/themes/query-only-routing.qmd`
- Create: `research/hub/themes/candidate-evidence.qmd`
- Create: `research/hub/themes/responses-references.qmd`
- Create: `research/hub/themes/evidence-reliability.qmd`
- Modify: `research/hub/questions/*.qmd`
- Modify: `research/hub/studies/*.qmd`
- Modify: `research/hub/ideas/*.qmd`

**Interfaces:**
- Consumes: existing `HubRecord`, `HubSources`, QMD loader, question/study/test schemas, and ledger relationships
- Produces: `HubSources.themes`, validated `primary_theme_id` and `theme_ids`, and four canonical theme records

- [ ] **Step 1: Write failing loader and validation tests**

Add a `theme` fixture record and tests equivalent to:

```python
assert sources.themes[0].kind == "theme"
assert sources.themes[0].record_id == "candidate-evidence"
assert "theme" in RECORD_KINDS
```

Add validation cases proving:

```python
with pytest.raises(ValueError, match="primary_theme_id references unknown"):
    validate_hub_sources(broken_sources, ledger_build, tmp_path)

with pytest.raises(ValueError, match="primary_theme_id must occur in theme_ids"):
    validate_hub_sources(mismatched_sources, ledger_build, tmp_path)
```

Assert a theme requires `plain_english`, `current_conclusion`, `decision`,
`confidence`, `confidence_boundary`, `next_test_ids`, `updated_on`, and at least
one evidence reference.

- [ ] **Step 2: Verify the new tests fail**

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_load.py tests/research_hub/test_validate.py -q
```

Expected: failures because `theme` is unknown and theme relationships are not validated.

- [ ] **Step 3: Implement the model and schema**

Extend the record constants and source container exactly as follows:

```python
RECORD_KINDS = frozenset(
    {"theme", "question", "study", "test", "approach", "dataset", "background"}
)

@dataclass(frozen=True)
class HubSources:
    themes: tuple[HubRecord, ...] = ()
    questions: tuple[HubRecord, ...] = ()
    # existing fields unchanged

    def all_records(self) -> tuple[HubRecord, ...]:
        return self.themes + self.questions + self.studies + self.tests + self.approaches + self.datasets + self.background
```

Map `theme` to `research/hub/themes` with ID field `theme_id`. Implement
`validate_theme(root, record)`, require theme links on questions/studies/tests,
and validate all relationship targets. A primary theme must occur in the record's
deduplicated `theme_ids`.

- [ ] **Step 4: Create the four themes and migrate all related records**

Use these exact IDs:

```text
query-only-routing
candidate-evidence
responses-references
evidence-reliability
```

Assign each question, study, and idea one `primary_theme_id` and a nonempty
`theme_ids` list. Use secondary themes only where the existing question IDs,
evidence references, and study conclusions substantiate the cross-link.

- [ ] **Step 5: Run focused tests and validate the real repository**

```bash
uv run --extra dev pytest tests/research_hub/test_load.py tests/research_hub/test_validate.py -q
uv run python scripts/build_research_hub.py
```

Expected: tests pass and all real records validate.

- [ ] **Step 6: Commit**

```bash
git add trec_mlm/research_hub tests/research_hub research/hub/themes research/hub/questions research/hub/studies research/hub/ideas
git commit -m "feat: add research hub themes"
```

### Task 2: Generate the theme hierarchy and remove hand-authored indexes

**Files:**
- Modify: `trec_mlm/research_hub/render.py`
- Modify: `tests/research_hub/test_render.py`
- Modify: `tests/research_hub/test_site_integration.py`
- Modify: `research/hub/index.qmd`
- Create: `research/hub/themes/index.qmd`
- Replace generated content in: `research/hub/generated/summary.qmd`
- Generate: `research/hub/generated/themes/*.qmd`
- Generate: `research/hub/generated/questions-index.qmd`
- Generate: `research/hub/generated/studies-index.qmd`
- Modify: `research/hub/questions/index.qmd`
- Modify: `research/hub/studies/index.qmd`
- Modify: `research/hub/_quarto.yml`

**Interfaces:**
- Consumes: validated `HubBuild.sources.themes`, `primary_theme_id`, `theme_ids`, existing questions/studies/tests, and run links
- Produces: `render_theme_pages(build) -> dict[Path, str]`, generated summary/question/study indexes, and Quarto wrappers that contain no duplicated research facts

- [ ] **Step 1: Write failing rendering tests**

Add tests asserting:

```python
outputs = render_hub_outputs(sample_build)
summary = outputs[Path("research/hub/generated/summary.qmd")]
assert "## Studies" not in summary
assert "Candidate-side evidence" in summary
assert "Open across themes" in summary

theme = outputs[Path("research/hub/generated/themes/candidate-evidence.qmd")]
assert "Current conclusion" in theme
assert "Practical decision" in theme
assert "Questions in this theme" in theme
assert "Evidence behind this position" in theme

study_index = outputs[Path("research/hub/generated/studies-index.qmd")]
assert "hub-study-row" in study_index
assert "artifact" not in study_index.lower()
assert "receipt" not in study_index.lower()
```

Assert every canonical theme/question/study ID appears in its generated index
and that counts are derived from relationships rather than literal constants.

- [ ] **Step 2: Verify rendering tests fail**

```bash
uv run --extra dev pytest tests/research_hub/test_render.py tests/research_hub/test_site_integration.py -q
```

Expected: missing theme outputs and old manual/listing content.

- [ ] **Step 3: Implement deterministic hierarchy renderers**

Add helpers with these interfaces:

```python
def render_theme_pages(build: HubBuild) -> dict[Path, str]: ...
def render_questions_index_qmd(build: HubBuild) -> str: ...
def render_studies_index_qmd(build: HubBuild) -> str: ...
```

Theme pages derive canonical questions by `theme_ids`, display `open_subquestions`
under their owning question, and collapse studies into one `hub-evidence-line`
link to the study library filtered by theme. Study rows include outcome/decision,
title, pursuit question, short conclusion, and decision summary; the enclosing
link opens the study QMD detail.

- [ ] **Step 4: Replace hand-maintained indexes with include wrappers**

Use wrappers of this form:

```markdown
---
title: "Studies"
description: "Decision-relevant experiments, audits, and reviews."
toc: false
---

{{< include ../generated/studies-index.qmd >}}
```

Do the equivalent for questions and main summary. Add generated theme pages to
the Quarto render list and navigation. No title, conclusion, decision, question,
count, or link is duplicated in an index wrapper.

- [ ] **Step 5: Run focused tests and a Quarto render**

```bash
uv run --extra dev pytest tests/research_hub/test_render.py tests/research_hub/test_site_integration.py -q
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run python scripts/build_research_hub.py --check
```

Expected: tests pass, Quarto renders without warnings, and tracked generated outputs are current.

- [ ] **Step 6: Commit**

```bash
git add trec_mlm/research_hub/render.py tests/research_hub research/hub
git commit -m "feat: generate research theme navigation"
```

### Task 3: Replace Background stubs with real literature and working deep dives

**Files:**
- Modify: `research/hub/background/index.qmd`
- Modify: `research/hub/background/trec-2025-literature.qmd`
- Modify: `research/hub/background/prior-participant-approaches.qmd`
- Modify: `research/hub/background/llm-routing-literature.qmd`
- Modify: `research/hub/background/related-competitions.qmd`
- Modify: `research/hub/background/methods-landscape.qmd`
- Create: `research/hub/library/trec-2025-literature-review.qmd`
- Create: `research/hub/library/reference-model-landscape.qmd`
- Create: `tests/research_hub/test_resources.py`
- Modify: `trec_mlm/research_hub/validate.py`
- Modify: `trec_mlm/research_hub/render.py`
- Modify: `research/hub/_quarto.yml`

**Interfaces:**
- Consumes: the audited canonical reviews and existing rendered reports
- Produces: substantive Background pages, generated open-row Background index,
  and Quarto-rendered Markdown or primary-source deep dives

> **Final security decision:** Raw self-contained HTML reports are not copied
> into the hub. Their embedded scripts and compressed payloads make a complete
> publication audit equivalent to maintaining a custom browser security parser.
> The hub publishes Quarto-rendered Markdown reviews and links to audited
> primary sources instead; repository agents retain the canonical report paths.

- [ ] **Step 1: Write failing source-integrity tests**

Assert:

```python
background = {r.record_id: r for r in build.sources.background}
assert background["llm-routing-literature"].metadata["canonical_source_path"] == "reports/trec_2026_approaches_guide.artifact.json"
assert "competition" not in background["related-competitions"].title.lower()
assert "RouterBench" in background["llm-routing-literature"].body
assert "SRCB" in background["trec-2025-literature"].body
```

Assert every Background page has substantive synthesis, a named controlling
source, an explicit comparison boundary, and a safe Markdown or primary-source
deep dive. Raw local report HTML must not be staged.

- [ ] **Step 2: Verify tests fail**

```bash
uv run --extra dev pytest tests/research_hub/test_resources.py tests/research_hub/test_site_integration.py -q
```

- [ ] **Step 3: Rewrite Background records from audited sources**

Use:

- `reports/trec_2025_mllm_literature_review.md` for 2025 task, teams, methods, limitations, and implications;
- `reports/trec_2026_approaches_guide.artifact.json` plus `reports/trec_2026_approaches_guide.html` for wider routing, expert finding, methods, weak labels, evaluation shift, and adjacent benchmarks;
- `reports/trec_2026_reference_model_landscape_2026-07-21.md` only for reference-generation model planning.

Rename the reader-facing related resource to **Adjacent benchmarks and routing
evaluations**. Put meaningful synthesis, comparison boundaries, named primary
sources, and one deep-dive link on each page. Remove pointer-stub prose and
placeholder citation strings.

- [ ] **Step 4: Add safe deep-dive rendering**

Render the two audited Markdown reviews through QMD wrappers. For sources whose
canonical local report is self-contained HTML, link from the Background
synthesis to reviewed primary literature and show the repository-relative
canonical artifact path as code for agents. Do not publish the raw HTML bundle.

- [ ] **Step 5: Generate the Background index and render**

The index groups resources as TREC 2025, wider methods/evaluations, and current
model/reference planning. Use open rows or two-column editorial blocks with one
link each; do not use Quarto's key/value grid listing.

Run:

```bash
uv run --extra dev pytest tests/research_hub/test_resources.py tests/research_hub/test_site_integration.py -q
uv run python scripts/build_research_hub.py
quarto render research/hub
```

Expected: working local deep-dive targets and no placeholder Background cards.

- [ ] **Step 6: Commit**

```bash
git add research/hub/background research/hub/library research/hub/_quarto.yml trec_mlm/research_hub tests/research_hub
git commit -m "feat: publish source-backed research background"
```

### Task 4: Generate rich study details, real examples, and contributor guidance

**Files:**
- Modify: `trec_mlm/research_hub/render.py`
- Modify: `tests/research_hub/test_render.py`
- Modify: `tests/research_hub/test_site_integration.py`
- Modify: `research/hub/start-here/problem.qmd`
- Create: `research/hub/CONTRIBUTING.md`
- Create: `research/hub/contributing.qmd`
- Create: `research/hub/templates/theme.qmd.example`
- Create: `research/hub/templates/question.qmd.example`
- Create: `research/hub/templates/study.qmd.example`
- Create: `research/hub/templates/test.qmd.example`
- Create: `research/hub/templates/dataset.qmd.example`
- Create: `research/hub/templates/background.qmd.example`
- Modify: `research/hub/_quarto.yml`

**Interfaces:**
- Consumes: study metadata/body, linked ledger systems/artifact status, question/theme/test relationships, and tracked development query/qrels
- Produces: generated study detail fragments, real task example, contributor documentation/templates, and automatic Quarto pre-render

- [ ] **Step 1: Write failing detail, example, and automation tests**

Assert a generated study detail contains:

```python
assert "Why this study was run" in detail
assert "Current result" in detail
assert "Decision and rationale" in detail
assert "Evidence limits" in detail
assert "Representative runs" in detail
assert "Artifact custody" in detail
assert "Canonical evidence" in detail
```

Assert the task page contains the tracked query and exact candidates:

```python
assert "Who became the second Karmapa Lama?" in problem
for candidate in ("Cultures", "People_by_time", "Scholars_and_academics_by_subject"):
    assert candidate in problem
assert "Candidate 37" not in problem
```

Assert `_quarto.yml` has a pre-render command invoking
`scripts/build_research_hub.py`, contributor templates exist, and
`CONTRIBUTING.md` documents add-question, add-study, link-run, build, and check.

- [ ] **Step 2: Verify tests fail**

```bash
uv run --extra dev pytest tests/research_hub/test_render.py tests/research_hub/test_site_integration.py -q
```

- [ ] **Step 3: Generate study detail fragments**

For each study, derive a fragment from canonical metadata and ledger rows. Keep
the authored study body for domain prose but generate relationship-driven
sections and run tables. Each table has a deliberate default order and never
compares different evaluation scopes.

- [ ] **Step 4: Replace the illustrative example**

Use the exact tracked development query/candidates from the specification.
Explain construction versus evaluation evidence and explicitly state that the
example does not claim a current method produced the shown ordering.

- [ ] **Step 5: Add contributor documentation and templates**

Every template contains the complete required front matter for its record type.
The contributor guide gives copy-paste commands:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run python scripts/build_research_hub.py --check
```

Add a rendered contributor page and navigation link. Configure Quarto pre-render
to execute the generator from any working directory with the explicit repository
root.

- [ ] **Step 6: Prove automatic reflection**

In a temporary repository copy, add one minimal valid question record, run the
Quarto pre-render command, and assert its title appears in the generated question
index, owning theme page, history/catalog output, and rendered HTML. Remove the
temporary copy after the test.

- [ ] **Step 7: Run tests and commit**

```bash
uv run --extra dev pytest tests/research_hub/test_render.py tests/research_hub/test_site_integration.py -q
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run python scripts/build_research_hub.py --check
git add trec_mlm/research_hub/render.py tests/research_hub research/hub
git commit -m "feat: automate research hub contributions"
```

### Task 5: Apply the open visual system and complete end-to-end verification

**Files:**
- Modify: `research/hub/assets/theme.scss`
- Modify: `tests/research_hub/test_site_integration.py`
- Modify: `reports/trec_2026_research_hub_summary.md` through the generator only
- Derived: `research/hub/_site/`
- Derived private copy: `/home/npatta01/codex-rendered/plans/trec-2026-research-hub/`

**Interfaces:**
- Consumes: semantic classes emitted by Tasks 2–4
- Produces: responsive open-row UI, verified rendered artifact, and updated private portal

- [ ] **Step 1: Add structural presentation checks**

Assert rendered/generated source uses:

```text
hub-theme-row
hub-conclusion
hub-question-row
hub-evidence-line
hub-study-row
hub-background-row
```

Assert generated indexes contain none of the old Quarto `listing:` blocks and
reader-facing source contains no line-count phrasing or generic metadata labels
such as `study_type`.

- [ ] **Step 2: Implement the selected SCSS**

Use:

- open rows with dividers for themes/questions/studies;
- one filled `hub-conclusion` panel;
- an unboxed three-part decision strip;
- restrained inline status labels;
- whole-row link focus/hover states;
- a single-column phone layout; and
- visible keyboard focus plus forced-color-safe borders.

Delete obsolete listing-grid and excessive card styles once no generated view
uses them.

- [ ] **Step 3: Run fresh focused verification**

```bash
uv run --extra dev pytest tests/research_hub -q
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run python scripts/build_research_hub.py --check
uv run --extra dev ruff check trec_mlm/research_hub scripts/build_research_hub.py scripts/normalize_research_hub_links.py tests/research_hub
```

Expected: zero failures, warning-free render, current generated outputs, and clean Ruff.

- [ ] **Step 4: Inspect representative pages**

Inspect summary, one theme, Background, study library, one study detail, task
example, and contributor page at 1440×1000 and 390×844. Confirm no horizontal
overflow, dense key/value tiles, button clusters, or truncated evidence text.

- [ ] **Step 5: Run privacy, portability, and link checks**

Scan `_site` for local paths, private-key markers, common cloud-key patterns,
external executable assets, and private raw response text. Check all internal
links and confirm allowlisted deep dives exist. Validate `catalog.json` and
confirm every canonical record is represented.

- [ ] **Step 6: Commit and update the private portal**

```bash
git add research/hub/assets/theme.scss tests/research_hub reports/trec_2026_research_hub_summary.md
git commit -m "feat: finish data-driven research hub"
```

Synchronize only `research/hub/_site/` to the existing dedicated portal
directory. Verify `tailscale serve status` remains `tailnet only` and live
summary, theme, Background, studies, example, contributor, deep-dive, and catalog
URLs return HTTP 200.

# Research Hub Page-Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the task explanation, research-question index, and data catalog visually clear and space-efficient.

**Architecture:** Keep Quarto Markdown as the canonical content layer. Replace the two unsuitable automatic listings with small semantic card indexes, add a CSS-only task-flow diagram, and centralize the responsive presentation in the existing theme.

**Tech Stack:** Quarto 1.9.38, Markdown, semantic HTML, SCSS, pytest, headless Chrome

## Global Constraints

- Do not introduce external scripts, fonts, images, or public hosting.
- Do not expose raw response text, hidden data, credentials, or local paths.
- Preserve the existing research records and evidence boundaries.
- The generated site must remain readable in light/dark themes and at phone width.

---

### Task 1: Add structural regression checks

**Files:**
- Modify: `tests/research_hub/test_site_integration.py`

**Interfaces:**
- Consumes: tracked QMD source files under `research/hub/`
- Produces: assertions for the task flow, wide question cards, grouped data cards, and absence of the old automatic listing front matter

- [ ] **Step 1: Add a failing structural test**

```python
def test_explainer_and_index_pages_use_readable_custom_layouts() -> None:
    problem = (ROOT / "research/hub/start-here/problem.qmd").read_text()
    questions = (ROOT / "research/hub/questions/index.qmd").read_text()
    data = (ROOT / "research/hub/data/index.qmd").read_text()

    assert "hub-task-flow" in problem
    assert "hub-example" in problem
    assert "hub-question-list" in questions
    assert "current bounded answer" in questions.lower()
    assert "hub-data-groups" in data
    assert "Organizer-provided" in data
    assert "listing:" not in questions
    assert "listing:" not in data
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run --extra dev pytest tests/research_hub/test_site_integration.py::test_explainer_and_index_pages_use_readable_custom_layouts -q`

Expected: failure because the three custom layout class names are absent.

### Task 2: Rewrite the three page structures

**Files:**
- Modify: `research/hub/start-here/problem.qmd`
- Modify: `research/hub/questions/index.qmd`
- Modify: `research/hub/data/index.qmd`

**Interfaces:**
- Consumes: facts already recorded in the five question and six dataset QMD front matters
- Produces: semantic Quarto divs using `hub-task-flow`, `hub-question-list`, and `hub-data-groups`

- [ ] **Step 1: Build the task explainer**

Use four labeled flow steps—request, evidence, ranking method, and result—followed by a miniature example and an evidence-boundary callout. Explain that the development qrels are evaluation evidence for historical requests, not input labels for a new request.

- [ ] **Step 2: Build the question cards**

Create one linked card per question. Put the title and plain-English interpretation first, then a labeled current-answer sentence, then status and confidence badges. Preserve all five records and their current front-matter values.

- [ ] **Step 3: Build the grouped data cards**

Create sections for organizer-provided, collected, and generated/evidence data. Put the six existing records in their matching group and preserve each record's plain-English description, category, and availability.

- [ ] **Step 4: Run the structural test**

Run: `uv run --extra dev pytest tests/research_hub/test_site_integration.py::test_explainer_and_index_pages_use_readable_custom_layouts -q`

Expected: one test passes.

### Task 3: Add responsive shared presentation

**Files:**
- Modify: `research/hub/assets/theme.scss`

**Interfaces:**
- Consumes: semantic class names introduced by Task 2
- Produces: accessible flow, card, badge, grouping, and responsive styles

- [ ] **Step 1: Style the explainer**

Use a four-column grid above 900 px and a one-column flow below it. Render arrows with CSS between steps, retain visible text labels, and use a distinct warning treatment for the evidence boundary.

- [ ] **Step 2: Style question cards**

Use one full-width card per row. Reserve visual weight for the title and answer; keep metadata in pill badges. Add generous internal spacing and a subtle border/hover state.

- [ ] **Step 3: Style data groups**

Use a responsive two-column grid for cards with section headings spanning both columns. Collapse to one column below 768 px. Ensure long availability text wraps naturally.

- [ ] **Step 4: Render and inspect**

Run:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
```

Expected: 153 pages render without warnings.

Capture `problem.html`, `questions/index.html`, and `data/index.html` at 1440×1000 and 390×844 with headless Chrome. Confirm no horizontal overflow, truncated text, or cramped key/value columns.

### Task 4: Verify and update the private artifact

**Files:**
- Generated: `research/hub/_site/`
- Derived private copy: `/home/npatta01/codex-rendered/plans/trec-2026-research-hub/`

**Interfaces:**
- Consumes: rendered, sanitized Quarto output
- Produces: updated tailnet-only Research Hub

- [ ] **Step 1: Run focused verification**

```bash
uv run --extra dev pytest tests/research_hub/test_site_integration.py tests/research_hub/test_render.py -q
uv run python scripts/build_research_hub.py --check
uv run --extra dev ruff check tests/research_hub/test_site_integration.py
```

Expected: all focused tests and checks pass.

- [ ] **Step 2: Run the privacy and portability scan**

Search rendered HTML for external executable assets, `file:///`, `/home/npatta01`, private-key headers, and AWS access-key patterns. Expected: no matches.

- [ ] **Step 3: Copy the derived site**

Synchronize only `research/hub/_site/` to the existing dedicated private portal directory. Do not alter the Tailscale Serve mapping.

- [ ] **Step 4: Verify live pages**

Confirm the portal remains `tailnet only` and that the three revised URLs, summary, and catalog return HTTP 200.

- [ ] **Step 5: Commit**

Commit the QMD, SCSS, tests, and this plan with a message describing the page-clarity redesign.

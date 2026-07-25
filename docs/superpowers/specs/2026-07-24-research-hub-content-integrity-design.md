# Research Hub Content Integrity and Theme Hierarchy

## Objective

Turn the existing Quarto Research Hub from a partially populated prototype into
a trustworthy, maintainable research resource. The site must expose real
repository evidence, organize growing research without overwhelming the main
page, and regenerate derived views automatically when records change.

## Core hierarchy

The research model is:

```text
Theme → research question → study → run
```

- A **theme** is a stable area of inquiry. It synthesizes the current
  conclusion, practical decision, confidence boundary, unresolved questions,
  and next move.
- A **research question** refines one aspect of a theme. It records its current
  bounded answer and may contribute to more than one theme.
- A **study** is one decision-relevant experiment, audit, or review. It explains
  the pursuit question, methods, result, decision, limitations, and evidence.
- A **run** is one technical datapoint in `experiments/ledger.json`. Runs remain
  supporting evidence inside a study rather than becoming separate discoveries.

Each question and study has one primary theme for navigation and may have
secondary theme links. Cross-links must not duplicate the canonical record.

## Main summary

The main page must remain stable as studies accumulate. It shows:

- one open row per active theme;
- the theme's current position;
- its answer state;
- counts of related questions and studies;
- cross-theme open questions; and
- the highest-value next tests.

It does not enumerate studies or tested systems. Theme rows use whitespace and
dividers rather than individual filled cards.

## Theme pages

Each theme page leads with the plain-English research problem and one visually
emphasized current-conclusion panel. A typographic strip follows with:

- practical decision;
- confidence and evidence boundary; and
- next move.

Related questions appear as open rows with a short current answer. Studies stay
behind one evidence-trail row summarizing the evidence volume. Opening that row
reveals or links to the supporting studies; the full searchable study library
remains available separately.

## Study library and detail pages

The study library groups studies by primary theme and study type. It uses wide
clickable rows, not small metadata tiles. Each row contains only:

- outcome and decision state;
- study title;
- the pursuit question and short result; and
- the practical decision.

The entire row opens the study detail page. Report, receipt, metric, artifact,
and run links do not compete as buttons on the index.

Each study detail page contains:

- research question and why the study was run;
- protocol and comparison;
- current result;
- decision and rationale;
- evidence limits;
- representative run table;
- artifact custody;
- linked reports and structured results; and
- related questions, studies, and next tests.

## Plain-English task example

Replace invented `Candidate 37` labels with a real, explicitly labeled
development example. The example uses the tracked request:

> Who became the second Karmapa Lama?

and the actual grade-2 candidate identifiers:

- `Cultures`
- `People_by_time`
- `Scholars_and_academics_by_subject`

It explains that the router orders all 150 candidate IDs, does not use that
request's grades while constructing the ranking, and is evaluated against the
grades afterward. It must not imply that a current method ranked the example
correctly.

## Background and literature integrity

The five current Background records are pointer stubs and must be replaced with
meaningful on-page synthesis and working deep links.

### TREC 2025 track and participants

The controlling editable review is
`reports/trec_2025_mllm_literature_review.md`. The hub page must summarize the
task, six participant teams, method families, official evaluation limitations,
and implications for 2026. It links to the complete review and the existing
interactive report. It names and links the official overview, run appendix, and
available participant papers through the controlling review.

### Wider routing and ranking literature

The controlling source is the authored approaches guide in
`scripts/build_approaches_guide.py`, its structured source registry in
`reports/trec_2026_approaches_guide.artifact.json`, and the rendered
`reports/trec_2026_approaches_guide.html`. The hub must surface the useful
synthesis of expert finding, retrieval, fusion, learning-to-rank, routing
benchmarks, calibration, weak labels, and distribution shift. It must not point
to the narrower TREC 2025 review as the broad-routing source.

### Related evaluations

Rename the unsupported “related competitions” framing to “Adjacent benchmarks
and routing evaluations.” Explain task comparability and point to the
source-backed BEIR, MTEB, RouterBench, LLMRouterBench, and related entries in the
approaches guide.

### Reference-model landscape

Keep the dated Gemini/Qwen/Gemma generation and runtime analysis separate from
retrieval and routing literature. Its controlling source is
`reports/trec_2026_reference_model_landscape_2026-07-21.md`.

No line counts, file-size trivia, or placeholder citation text appears in the
reader UI.

## Canonical data and generated views

Human-authored facts live in Quarto Markdown front matter and prose:

- `research/hub/themes/*.qmd`
- `research/hub/questions/*.qmd`
- `research/hub/studies/*.qmd`
- `research/hub/ideas/*.qmd`
- `research/hub/data/*.qmd`
- `research/hub/background/*.qmd`

Run facts remain canonical in `experiments/ledger.json`.

The generator derives:

- main theme summary;
- theme question/evidence indexes;
- question index;
- grouped study library;
- Background resource index;
- coverage matrix;
- narrative history;
- run index and run details;
- repository Markdown summary; and
- machine-readable agent catalog.

No derived index contains manually duplicated titles, answers, decisions, or
counts.

## Record relationships and validation

Theme records define:

- `theme_id`
- `title`
- `plain_english`
- `current_conclusion`
- `decision`
- `confidence`
- `confidence_boundary`
- `next_test_ids`
- `updated_on`

Question and study records add `primary_theme_id` and optional `theme_ids`.
Existing question-to-study and study-to-run links remain canonical.

Validation rejects:

- duplicate or unsafe identifiers;
- missing primary themes;
- dangling question, study, test, report, or run references;
- a run linked to no study;
- missing current conclusions, decisions, or confidence boundaries;
- invalid evaluation-scope comparisons;
- missing canonical evidence paths;
- mismatches between generated and committed outputs; and
- records omitted from every generated navigation surface.

## Automatic build contract

`quarto render research/hub` invokes the Research Hub generator as a pre-render
step. A valid new record automatically updates every related derived view.

Examples:

- Adding a question updates its theme page, question index, open-question
  summary, coverage view, history, and catalog.
- Adding a study updates the theme evidence trail, study library, question
  links, history, and catalog.
- Adding a run to the ledger and linking it to a study updates the study detail,
  run index, run detail, evidence counts, coverage, and catalog.

`scripts/build_research_hub.py --check` remains the stale-output and validation
gate.

## Contributor experience

Add a concise `research/hub/CONTRIBUTING.md` and copyable templates for themes,
questions, studies, tests, datasets, and Background resources. The guide
explains:

- where each kind of fact belongs;
- how to add or link a run;
- how to preserve evaluation scope and confidence boundaries;
- how to attach a canonical report;
- how to render locally; and
- how to verify generated outputs.

The site links to a rendered “How to update this hub” page for humans and agents.

## Presentation

The production implementation uses Quarto Markdown and generated QMD. Shared
SCSS controls presentation; hand-written HTML is not a content source.

The selected visual rules are:

- whitespace and dividers before boxes;
- one emphasized conclusion panel per theme;
- open rows for themes and questions;
- wide clickable rows for studies;
- compact text links rather than button clusters;
- progressive disclosure for evidence;
- readable phone reflow;
- accessible semantic headings and link targets; and
- light/dark theme support.

## Verification and private delivery

Completion requires:

- focused schema, generator, integration, and render tests;
- a warning-free Quarto render;
- stale-output verification;
- desktop and phone inspection of summary, theme, Background, study library,
  and study detail pages;
- working deep links to sanitized full reviews and reports;
- scans for local paths, secrets, external executable assets, and private raw
  response text;
- confirmation that the agent catalog includes every canonical record; and
- live HTTP checks on the existing tailnet-only portal.

The canonical source remains in the repository. Only sanitized derived HTML and
required static assets are copied to the authorized private portal.

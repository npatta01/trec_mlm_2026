# TREC 2026 Research Hub Design

**Status:** Approved in chat and approved after written-spec review on 2026-07-24.

## Objective

Create one durable, source-backed Research Hub that helps a newcomer, an active
researcher, or an agent answer:

- What problem is this project solving?
- What data exists, where did it come from, and what may it be used for?
- What prior work and related competitions inform the project?
- Which research questions have been pursued?
- Why was each study run?
- What did the evidence show, and which decision followed?
- Which questions remain open?
- Which ideas are immature, ready to test, running, completed, or parked?
- Which technical runs and artifacts support a conclusion?

The hub must serve humans through a progressive, responsive Quarto website and
serve agents through canonical Markdown/Quarto sources plus a generated,
machine-readable catalog. The rendered site is a derived view, never the source
of truth.

## Problem Being Corrected

The existing experiment ledger is strong at evidence integrity but is primarily
a registry of ranking-system results. It currently covers 98 systems across 11
technical families, including aggregate metrics, evaluation scopes, per-query
availability, caveats, and source provenance.

That registry does not, by itself, explain the research program:

- technical families do not state the question a study pursued;
- a group of parameter variants can look like many independent experiments;
- the highest development score can be mistaken for the adopted decision;
- proposed work and unresolved questions are not represented;
- audits, reference-generation reviews, feasibility work, and data-quality
  studies may produce no ranking-system row;
- prior literature, related competitions, task context, and data provenance live
  in separate reports; and
- project vocabulary is not approachable to a reader encountering the task for
  the first time.

The redesign must not call the 98-system inventory "everything tested." It is
the **run evidence registry** beneath a broader research ledger.

## Audience

### New human reader

Needs a plain-English problem statement, a worked example, a glossary, a clear
description of available data, and explanations before technical details.

### Active researcher

Needs current conclusions, decisions, evidence boundaries, open questions,
ready tests, study history, and efficient drill-down into metrics and artifacts.

### Agent

Needs stable repository paths, machine-readable identities and relationships,
concise summaries, explicit provenance, and canonical sources that do not
require scraping rendered HTML.

### Technical reviewer

Needs evaluation-scope guards, exact run evidence, source receipts, hashes,
caveats, artifact custody, per-query evidence, and deterministic-build checks.

## Design Principles

1. **Decision before leaderboard.** The primary views explain what the evidence
   caused the project to adopt, retain, reject, stop, iterate, or revisit.
2. **Question before technique.** Research questions are the primary navigation.
   Technical approaches are tags and secondary navigation because one study may
   combine approaches and one approach may serve several questions.
3. **Study before run.** Narrative belongs to a coherent study. Most system
   variants remain compact datapoints inside that study.
4. **Progressive disclosure.** Summary pages omit hashes, full caveat lists,
   per-query tables, and exhaustive run variants. Each deeper page reveals only
   the next useful layer.
5. **Plain English is required.** Every research question and dataset has a
   newcomer-facing explanation in addition to precise technical terminology.
6. **Evidence maturity beats maximum score.** Diagnostic, restricted, invalid,
   and aggregate-only results remain visible without being presented as adopted
   results.
7. **No invented history.** Backfilled rationale is marked
   `contemporaneous`, `reconstructed`, or `unknown`, with evidence references.
8. **One set of sources, multiple views.** Humans read Quarto HTML; agents read
   the canonical Markdown/QMD and generated catalog. The content cannot silently
   diverge.
9. **Static and private by default.** The hub has no server-side runtime and is
   not published or shared without explicit authorization.

## Information Architecture

The overall product is the **TREC 2026 Research Hub**. The Experiment Ledger is
one bounded section.

```text
TREC 2026 Research Hub
├── Start Here
│   ├── Problem in plain English
│   ├── Worked example
│   ├── Evaluation and metrics
│   ├── Evidence boundaries
│   └── Glossary
├── Background
│   ├── TREC 2025 literature review
│   ├── Prior participant approaches
│   ├── Similar competitions and benchmarks
│   ├── Wider LLM-routing literature
│   └── Methods landscape
├── Data Catalog
│   ├── Organizer-provided data
│   ├── Collected data
│   ├── Generated assets
│   ├── Sanitized representative samples
│   └── Missing or local-only artifacts
├── Research Ledger
│   ├── Summary
│   ├── Questions
│   ├── Studies
│   ├── Ideas and next tests
│   ├── Approaches
│   └── History
├── Protocols
│   ├── Frozen experiment protocols
│   ├── Leakage boundaries
│   └── Reproduction instructions
└── Evidence
    ├── Coverage matrix
    ├── Technical run registry
    ├── Per-query explorers
    └── Source receipts
```

## Source-of-Truth Boundaries

### Research narrative

Human-authored `.qmd` files are canonical for:

- plain-English and technical question statements;
- study rationale, hypothesis, protocol summary, conclusion, and decision;
- literature and competition summaries;
- dataset explanations;
- open subquestions;
- proposed test descriptions; and
- explanatory prose and glossary terms.

Important structure must live in YAML front matter rather than being inferred
from arbitrary headings.

### Run evidence

The existing `experiments/ledger.json` remains canonical for:

- system and source-system identities;
- evaluation scope;
- promotion and evidence status;
- aggregate and per-query metrics;
- artifact state;
- qrel usage;
- caveats;
- evidence pointers; and
- source digests.

Narrative files reference run IDs. They do not copy authoritative metrics.

### Frozen receipts and protocols

Tracked evaluation receipts, decision receipts, protocols, reports, and
artifacts remain the scientific evidence. The hub indexes and summarizes them;
it does not replace them.

### Generated views

Generated QMD fragments, JSON indexes, Markdown summaries, and HTML are never
edited directly. The builder must reject stale generated output.

## Core Entities

### Research question

A question represents something the project is trying to learn.

Required fields:

- `question_id`: stable, unique identifier;
- `title`: precise technical question;
- `plain_english`: newcomer-facing explanation of what is being compared;
- `status`: `open`, `partially_answered`, `answered`, or `parked`;
- `current_answer`: concise synthesis, or an explicit statement that evidence is
  insufficient;
- `confidence`: `low`, `moderate`, or `high`, with an evidence boundary;
- `study_ids`: one or more supporting or contradictory studies when evidence
  exists;
- `open_subquestions`;
- `next_test_ids`;
- `updated_on`; and
- `evidence_refs`.

The body defines project-specific terms and gives a concrete example when the
title alone assumes domain knowledge.

### Study

A study is one coherent unit of research. It may contain many runs, one run, no
runs, an audit, a literature synthesis, a data-quality result, or an
operational-feasibility result.

Required fields:

- `study_id`;
- `title`;
- `study_type`: `ranking_experiment`, `audit`, `data_quality`,
  `reference_review`, `feasibility`, or `evidence_synthesis`;
- `question_ids`: one or more, allowing many-to-many relationships;
- `approach_ids`: zero or more;
- `status`: `planned`, `running`, `concluded`, or `blocked`;
- `outcome`: for concluded studies, `positive`, `negative`, `mixed`,
  `inconclusive`, `invalid`, or `unscored`; otherwise `pending`;
- `objective`: required for every study;
- `hypothesis`: required for `ranking_experiment`; optional for other study
  types when a falsifiable hypothesis genuinely governed the work;
- `rationale`;
- `rationale_provenance`: `contemporaneous`, `reconstructed`, or `unknown`;
- `protocol_summary`;
- `conclusion`;
- `decision`: `adopt`, `retain_control`, `reject`, `stop`, `iterate`,
  `revisit_on_trigger`, or `informational`; a running or blocked study uses
  `pending`;
- `decision_summary`;
- `system_ids`: the complete set of registry systems produced or evaluated by
  the study, which may be empty for non-ranking studies;
- `representative_system_ids`;
- `evidence_refs`;
- `dataset_ids`;
- `started_on`: an ISO date or `unknown`;
- `concluded_on`: an ISO date, `unknown`, or `null` while unfinished;
- `supersedes` and `superseded_by`: stable study IDs or empty lists; and
- limitations.

Unknown historical facts stay unknown. A reconstructed rationale cites the
evidence used for reconstruction and cannot be described as preregistered.
Every representative system must also occur in `system_ids`. Every run in the
registry must eventually belong to at least one study, but a run may be linked
to multiple studies when it is reused as a frozen control.

### Run or system

A run is a technical datapoint owned by the existing run evidence registry. A
study may select representative run IDs for its default comparison while
retaining all variants in the technical registry.

The hub must not create narrative pages for every run by default. A run detail
view is generated from structured evidence and linked from its owning studies.

### Idea or proposed test

Ideas use a two-stage funnel.

An early idea requires:

- `test_id`;
- `title`;
- `stage`: `idea` or `parked`;
- `summary`;
- `linked_question_ids`; and
- known missing readiness fields.

A test may enter `ready` only when it also defines:

- hypothesis;
- rationale;
- expected signal;
- frozen decision rule;
- dependencies;
- priority;
- evaluation boundary;
- evidence and artifact plan; and
- next action.

Additional stages are `running` and `completed`. Completion links the test to
the resulting study rather than duplicating its final evidence.

### Technical approach

An approach is a reusable technique such as dense query retrieval, candidate
semantic profiles, response retrieval, reranking, or reference generation.

It is a many-to-many tag over studies and questions. Its page synthesizes:

- current technical assessment;
- questions where it has been used;
- positive, negative, mixed, invalid, and unscored studies;
- known coverage gaps; and
- active or proposed follow-ups.

### Dataset or evidence asset

A data-catalog entry requires:

- `dataset_id`;
- title and plain-English role;
- category: `provided`, `collected`, `generated`, or `external`;
- provenance and collection or generation method;
- version or immutable identity, or an explicit `unversioned` status;
- repository path or explicit external/local-only status;
- sensitivity: `public`, `project_internal`, `restricted`, or `secret`;
- sharing: `repository`, `local_only`, `external_reference`, or
  `metadata_only`;
- leakage and permitted-use constraints;
- studies that consume it;
- availability and reproducibility status; and
- an optional sanitized representative sample.

Samples must never expose secrets, private raw logs, restricted datasets, or
unnecessary personal data.

### Background resource

A background record indexes literature reviews, prior participant approaches,
related competitions, benchmarks, and methods surveys.

It requires a stable ID; type (`literature_review`, `participant_review`,
`competition`, `benchmark`, or `methods_landscape`); title; concise summary;
canonical source path; citations; tags; applicability to current questions; and
review date.

## Quarto Architecture

Use Quarto `1.9.38` as the initial pinned rendering version.

```text
research/
├── README.md
├── catalog.json                 # generated agent index
└── hub/
    ├── _quarto.yml
    ├── index.qmd
    ├── start-here/
    ├── background/
    ├── data/
    ├── questions/
    ├── studies/
    ├── ideas/
    ├── approaches/
    ├── protocols/
    ├── evidence/
    ├── generated/               # generated QMD/data; never hand-edited
    ├── theme.scss
    └── ledger.js

experiments/ledger.json           # existing run evidence registry
scripts/build_research_hub.py      # validation and view-model generation
```

`research/README.md` is the short agent and repository entry point. It tells a
reader where to find the problem definition, data catalog, literature, active
questions, current decisions, next tests, protocols, and run evidence.

`research/catalog.json` is a generated machine-readable index with stable IDs,
record types, summaries, tags, canonical source paths, rendered paths, and
relationships. Agents should consume Markdown/QMD and this catalog rather than
scraping HTML.

### Build flow

```text
Canonical QMD + run registry + frozen evidence
                      ↓
       build_research_hub.py --check
  validate, join, synthesize view-model data
                      ↓
              quarto render
                      ↓
    private static site + portable summaries
```

The builder:

1. parses front matter and Markdown records;
2. validates schemas, vocabularies, IDs, and relationships;
3. verifies referenced repository paths and run IDs;
4. joins study metadata to the run evidence registry;
5. generates coverage, history, run, and catalog data;
6. writes generated QMD/JSON deterministically; and
7. checks that rendered-input files are current.

Quarto handles responsive HTML, navigation, listings, cross-links, tabsets,
cards, theme rendering, and accessible document structure.

The normal local build is:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
uv run python scripts/build_research_hub.py --check
```

The first command validates all canonical sources and atomically writes
generated QMD/JSON. The second renders only after semantic validation succeeds.
The final command proves that no renderer or post-render action changed the
generated source inputs. Tests run the same sequence in a temporary output
directory before checking repository outputs.

Client-side filtering may use Quarto listings for document indexes and a small
local JavaScript or Observable component for the coverage matrix and run
filters. The static hub must not require Shiny or another server runtime.

### Outputs

Repository-local derived outputs:

- a concise generated Markdown decision brief;
- the generated agent catalog;
- Quarto site output suitable for local inspection; and
- an optional self-contained HTML briefing when portability matters.

For remote visual handoff, only sanitized derived HTML and necessary static
assets may be copied under `/home/npatta01/codex-rendered/plans/` and linked
from the existing private, tailnet-only portal. The canonical source remains in
the repository. No public deployment is in scope.

An optional Reveal.js research briefing may later be generated from the same
records. Reveal.js is not the hub's primary navigation or storage model.

## Reader Experience

### Summary

The landing page is a decision brief, not a leaderboard. It shows:

1. the most important current conclusion;
2. the resulting practical decision;
3. confidence and evidence boundaries;
4. important open questions;
5. ready next tests and a small number of immature ideas; and
6. quiet navigation to history and evidence.

It omits:

- total system counts as a headline;
- complete run tables;
- per-query evidence;
- artifact hashes;
- flat lists of every caveat or blocker;
- the full history timeline;
- the full coverage matrix; and
- "best system" language without decision context.

### Question detail

Order:

1. precise research question;
2. required "In plain English" explanation and term definitions;
3. current answer;
4. practical decision;
5. confidence and evidence limits;
6. supporting and contradictory studies;
7. open subquestions; and
8. next tests.

### Study detail

Order:

1. linked question or questions;
2. hypothesis and rationale provenance;
3. protocol summary;
4. conclusion;
5. resulting decision;
6. evidence maturity and limitations;
7. representative run comparison or non-run evidence; and
8. source links and deeper technical evidence.

### Ideas and tests

Show separate columns or filters for `idea`, `ready`, `running`, and `parked`.
A readiness checklist explains why an idea is not yet executable. Every ready
test foregrounds the decision it will unlock.

### Coverage

The matrix crosses research questions with technical approaches. A cell shows
an evidence state plus study count, for example `mixed · 3 studies`, and expands
to those studies.

Cells must not collapse mixed, invalid, restricted, or unscored evidence into a
single success/failure color.

### History

The narrative timeline explains:

- which question was active;
- what study or external evidence changed understanding;
- which decision followed; and
- why the next study was undertaken.

Chronology is a secondary view, not the primary ownership hierarchy.

### Run detail

The technical view includes:

- canonical and source-system IDs;
- study membership;
- method summary;
- evaluation scope;
- promotion and evidence status;
- metrics;
- per-query availability;
- artifact and qrel-use status;
- caveats;
- source receipts and digests; and
- links to explorers and slice evidence.

## Validation and Error Handling

The pre-render builder fails before Quarto rendering when:

- an ID is duplicated;
- a relationship references an unknown record;
- a study references an unknown run or dataset;
- a ready test lacks a required readiness field;
- an answered question lacks supporting evidence;
- required plain-English text is missing;
- a reconstructed rationale lacks provenance;
- a decision lacks evidence references;
- a repository path is absolute, escapes the repository, or does not exist;
- a data sample violates its sharing classification;
- a generated file is stale;
- a numerical comparison crosses evaluation scopes; or
- existing run-registry validation fails.

Errors identify the record ID, source path, field, and corrective action. The
builder must perform all validation and in-memory rendering before replacing
generated files.

Quarto errors are reported separately after semantic validation passes.

## Testing

### Schema tests

- required and optional fields by entity type;
- controlled vocabularies;
- stable-ID uniqueness;
- many-to-many relationship validity;
- idea-to-ready transition rules;
- rationale-provenance rules; and
- path and sharing constraints.

### Integration tests

- real records resolve to the existing run registry;
- every generated study comparison respects evaluation scope;
- studies with no run rows render correctly;
- questions synthesize positive, negative, mixed, invalid, and unscored evidence
  without flattening it;
- existing ledger metrics remain unchanged;
- the agent catalog links canonical and rendered paths consistently; and
- generated inputs reproduce byte-for-byte.

### Quarto smoke tests

- `quarto render` succeeds with Quarto 1.9.38;
- every declared page exists;
- internal links and anchors resolve;
- no page depends on external network resources;
- client-side filters operate with a static file server;
- light and dark themes remain readable;
- keyboard navigation and visible focus work;
- headings, landmarks, tables, and status text are accessible;
- phone-width layouts do not require horizontal scrolling except bounded
  technical tables; and
- the no-JavaScript reading path retains essential conclusions and evidence.

### Content review

- newcomer review of Start Here and plain-English question explanations;
- evidence-owner review of backfilled conclusions and decisions;
- provenance review of reconstructed historical rationale;
- sample-data review for privacy and leakage; and
- scope review confirming that diagnostic leads are not presented as promoted
  results.

## Migration and Backfill

Migration is incremental.

### Phase 1: Framework and entry points

- establish the Quarto project, schema, validator, agent README, and catalog;
- render Start Here, the summary, and empty-but-valid indexes;
- integrate the existing run registry without changing its metrics.

### Phase 2: High-value backfill

- map the major existing experiment families into research questions and
  coherent studies;
- backfill current conclusions and decisions from controlling receipts;
- label rationale provenance;
- index major protocols, data assets, literature reviews, and related
  competition resources.

Initial background seeds include
`reports/trec_2025_mllm_literature_review.md`,
`reports/trec_2026_approaches_guide_source_notes.md`, and the existing 2026
reference-model landscape and review reports. Initial study seeds include the
controlling sources already declared by `experiments/ledger.json` plus newer
tracked audits and refinement reports. Indexing a source does not silently
promote its claims; each record retains its evidence and review status.

### Phase 3: Complete historical coverage

- attach all 98 run rows to studies;
- add non-ranking studies such as audits, reference reviews, data-quality work,
  and feasibility assessments;
- record unknown rationale explicitly rather than inventing it; and
- reconcile recent work that postdates the original run-ledger design.

### Phase 4: Prospective workflow

- use the idea-to-ready funnel for new proposals;
- require study and decision records at experiment completion;
- update the hub in the normal experiment-reporting workflow; and
- prevent newly tracked result receipts from remaining unindexed.

## Security and Privacy

- No public hosting, public listener, Tailscale Funnel, external analytics, or
  third-party comment system.
- No secrets, credentials, raw private logs, restricted datasets, or direct
  personal identifiers in rendered outputs.
- External JavaScript, fonts, images, and CDN dependencies are disallowed.
- Collected and generated datasets carry explicit sharing classifications.
- The private tailnet portal may receive only sanitized derived files after a
  separate verification step.
- Canonical repository sources and local-only artifact paths are not copied
  wholesale into the rendered portal.

## Scope

This project includes:

- the Quarto Research Hub framework;
- canonical schemas and source directories;
- human and agent entry points;
- the Start Here guide and glossary;
- Background and related-work navigation;
- a source-backed Data Catalog;
- the research-question, study, idea/test, approach, history, coverage, and run
  views;
- integration with the existing run evidence registry;
- deterministic validation and build tooling;
- high-value historical backfill; and
- static, accessible, responsive output.

This project does not:

- run new experiments;
- change historical metrics or promotion decisions;
- reconstruct missing raw artifacts;
- infer unknown historical motivation;
- build a server-backed research application;
- require R or Shiny;
- replace frozen receipts or protocols;
- publish or change sharing permissions;
- expose private datasets or logs; or
- make Reveal.js the canonical research interface.

## Success Criteria

The design is successfully implemented when:

1. a newcomer can explain the task, available data, and primary evidence
   boundary after reading Start Here;
2. the summary states current conclusions, decisions, open questions, and ready
   tests without showing an exhaustive leaderboard;
3. every major historical study is linked to at least one research question;
4. every existing run is retained in the technical evidence registry and can be
   reached from its study;
5. studies without ranking runs are represented correctly;
6. agents can discover canonical resources and relationships through
   `research/README.md` and `research/catalog.json`;
7. the validator rejects broken relationships, unsupported decisions, stale
   outputs, and scope-unsafe comparisons;
8. Quarto renders the complete static hub reproducibly without network access;
9. the rendered hub is usable on a phone and by keyboard and screen-reader
   users; and
10. no rendered artifact exposes private or restricted material.

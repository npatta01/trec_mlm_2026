# PR #22 guide-pattern adaptation design

## Status

Approved in conversation on 2026-07-26. Implemented on 2026-07-26 through the
canonical Research Hub generator and authored newcomer pages; superseded as an
active design gate by `plans/trec-2026-pr22-guide-pattern-adaptation.md`.

This work continues in the existing externally managed detached worktree at
`fd07842`, which already contains the uncommitted newcomer Research Hub work.
Preserve that state. Do not create a design-only commit, branch, push, or pull
request from this worktree.

## Goal

Adopt the answer-first information architecture demonstrated by
`npatta01/trec_video_2026` PR #22 so a newcomer can learn what the Million LLMs
project does, what it recommends now, what evidence supports that decision,
what remains unknown, and what should happen next without weakening the
canonical experiment ledger or generated evidence controls.

## Reader contract

Every newcomer-facing surface should answer these questions in order:

1. What is this project trying to do?
2. What does the project currently recommend?
3. What evidence supports that recommendation?
4. What does the evidence not establish?
5. What is the cheapest valid next action or test?

The first screen must retain the development-only and hidden-test boundary.
Exact metrics and run facts remain generated from canonical records.

## Scope

### 1. Action-first current-state opening

Refactor `research/hub/start-here/current-state.qmd` so the default order is:

1. **Project in one minute:** one request is used to rank 150 candidate models;
   the ranking system does not answer the request.
2. **Evidence boundary:** development evidence only; hidden-test quality is
   unknown; the current request's own qrel is excluded from ranking.
3. **Current recommendation:** a reader-facing decision strip generated from
   canonical ledger and completed-follow-up evidence.
4. **See one ranking from start to finish:** link to the expanded synthetic
   worked example.
5. **Learn from concrete failure mechanisms:** link to the new privacy-safe
   library and retain the generated 21/54 aggregate split.
6. **Evidence routes and method families:** short previews before exact results
   and optional audit detail.
7. **Choose a reading path:** 2-minute decision, 10-minute evidence, 20-minute
   audit/code, and contributor paths.

The action strip uses these reader labels and mappings:

| Reader label | Canonical meaning |
| --- | --- |
| **Use now** | The one `promoted_development` registered system: `dense_profile_equal`. |
| **Retain as controls** | Fixed registered rows carrying `stable_control`. |
| **Promising follow-up** | E5 receipt gate `advance`, while its Hub study remains `informational`; the result is outside the ledger, unregistered, and unpromoted. |
| **Still required before test** | Issues #39–#42 and their dependency-aware work contracts. |
| **Do not reuse in the tested form** | Registered `valid_negative` or rejected method forms; the label is condition-bound, not a permanent verdict. |
| **Unknown** | Hidden-test behavior and generalization. |

Reader labels never replace canonical status text. Each card exposes its exact
ledger, Hub, receipt, or issue boundary.

### 2. How One Ranking Is Made

Expand `research/hub/start-here/worked-example.qmd` into a complete numbered
trace using only the existing fabricated basil request and the fictional
GardenGuide, QuickFacts, and CodeTutor candidates:

1. Request and ranking objective.
2. Allowed and forbidden ranking-time evidence.
3. Similar historical requests and their frozen relevance history.
4. Candidate labels as clues, not capability truth.
5. Candidate response archives and semantic profiles without raw response
   disclosure.
6. Separate fabricated query-side and candidate-side orderings.
7. Equal reciprocal-rank fusion into one frozen illustrative ordering.
8. Evaluation only after legitimate development qrels are available.
9. Post-hoc diagnosis and next falsifying test.
10. What the synthetic trace cannot establish.

The trace demonstrates the promoted dense/profile direction. Query-cluster,
source-query-first, direct lexical response retrieval, raw-response E5, and
generated-reference routes appear in the method-family map as alternatives or
follow-ups rather than being forced through the main example as if all were
parts of the promoted system.

No private response, real query text, target response, raw artifact path, or
model-generated rationale appears in the example.

### 3. Vocabulary and organizational layers

Use three distinct terms consistently:

- **Evidence routes:** similar queries, candidate labels, query clusters,
  source-query responses, candidate response archives/profiles, and generated
  references. These describe how a score or comparison signal is formed.
- **Method families:** the six presentation-only groups that organize the 16
  authored studies for reading. The existing `research-theme-map` visual ID is
  preserved for compatibility, but its visible title becomes **Studies by
  method family** and its explanation says it is not a canonical taxonomy.
- **Canonical research themes:** the four durable Hub records under
  `research/hub/themes/`. They own current conclusions, confidence boundaries,
  evidence references, and prospective tests.

The glossary and page copy must never use bare `cluster`, `response`,
`reference`, or `theme` when more than one meaning is possible.

### 4. Answer-first method-family and canonical-theme cards

Extend the current-state renderer's fixed presentation specifications so every
method-family card contains:

- plain-language purpose;
- working answer;
- strongest canonical study link and study status;
- open question;
- next valid test or decision boundary;
- the complete ordered study list.

Each method-family specification names one strongest study that must occur in
its exact study partition. Validation rejects missing studies, duplicate study
membership, unknown strongest-study IDs, and presentation specs that do not
partition all 16 authored studies exactly once.

Extend the four canonical theme records with authored
`strongest_evidence_summary` and `open_question` metadata. Generate canonical
theme cards containing:

- current working answer from `current_conclusion`;
- strongest evidence summary plus canonical `evidence_refs`;
- confidence and confidence boundary;
- open question;
- first linked next test and remaining next-test count.

These fields improve the renderer; they do not create a second ledger or
duplicate run metrics.

### 5. Privacy-safe failure mechanism library

Create `research/hub/start-here/failure-library.qmd` with a generated evidence
include. It contains three mechanism records:

1. **Fusion dilution:** 21 development failures where one component had a
   top-ten signal but fusion moved it out.
2. **No component signal:** 54 development failures where neither promoted
   component placed the relevant candidate in its top ten.
3. **Unreliable generated evidence:** selected grounding/reference audits show
   that generated evidence can be ambiguous or unsupported; no prevalence
   estimate is claimed.

Every record uses the same fields:

- what failed;
- evidence available;
- attempted or current response;
- observed outcome;
- possible failure location;
- competing explanation;
- confirmed defect, audit gap, or uncertainty;
- evidence limit;
- next falsifying test;
- source study and evidence status.

The first two counts and definitions come from the canonical failure study and
receipt already used by the current-state generator. The third record links
the authored grounding and reference-review studies and contains no invented
count. Selected mechanisms explain failure modes, not benchmark prevalence.
No query text, candidate response, private identifier, raw path, or source
artifact content is rendered.

### 6. Branch-complete next-test queue

Expand `research/hub/background/pretest-readiness-plan.qmd` into a scannable
queue for issues #39–#42. Every queue item states:

- issue link and dependency;
- outcome or hypothesis;
- smallest valid work item or panel;
- controls and integrity safeguards;
- decision or stop rule;
- boundary on what a positive result would establish;
- current blocker.

The dependency ordering remains:

- #39 qrel-free unseen-query inference and #40 durable private artifact custody
  can start independently;
- #41 five-run portfolio freeze depends on #39 and #40;
- #42 preregistered E5 inference depends on an executable unseen-query path and
  must preserve the off-ledger/unpromoted boundary until its registered test is
  complete.

The queue records stable work contracts only. Generated files do not cache
mutable GitHub issue state.

### 7. Contributor lifecycle and update router

Expand `research/hub/CONTRIBUTING.md` with a plain-language introduction:

`idea → planned test → evidence bundle → condition-bound claim → project decision → reconsider, supersede, or invalidate`

Define responsibilities:

- the experiment ledger owns registered system facts and metrics;
- one canonical Hub study owns each condition-bound conclusion;
- a theme owns the durable research question and current answer;
- protocols own frozen pre-run contracts;
- evidence bundles preserve the smallest immutable audit record;
- generated pages are rebuildable views.

Add a “Where does this update belong?” table covering early ideas, planned
tests, registered runs, off-ledger follow-ups, completed studies, cross-theme
relevance, failure mechanisms, corrections, supersession, and reusable code.
Add before-run and after-run checklists, reconsideration conditions, and a
copyable human-readable handoff packet. Preserve existing commands and schema
instructions below this introduction.

### 8. Navigation and display

- Add the failure library and expanded pre-test queue to the existing Start
  Here/Background reading paths without removing search or changing the global
  navbar taxonomy.
- Keep the page single-column. Action and method cards may use two columns only
  above 768 pixels and must stack in DOM order at mobile and 400% zoom.
- Reuse the current Hub visual language. Do not copy the video guide's colors,
  Mermaid implementation, wide mobile matrix, small controls, or static
  `role="status"` pattern.
- Preserve 44-pixel interactive targets, visible keyboard focus, light/dark,
  reduced-motion, grayscale, forced-colors, print, and no-script behavior.
- All visuals retain adjacent explanations, complete semantic equivalents,
  source identity, and explicit status text.
- Disclosure summaries must be record-specific rather than repeating one
  generic accessible name.

## Data flow and canonical boundaries

1. Authored ledger, theme, study, test, and readiness records remain canonical.
2. `build_research_hub()` validates those records and the existing E5 receipt.
3. `render.py` produces the decision strip, method-family cards, canonical
   theme cards, failure mechanism include, and existing evidence visuals.
4. Authored Start Here, worked-example, readiness, failure-library, and
   contributor pages provide orientation and teaching prose.
5. Quarto renders private sanitized HTML; `_site` is derived and not canonical.

No quantitative value or registered status is copied manually into authored
QMD. Diagnostic metrics cannot become promoted through presentation logic.

## Validation and tests

Use red-green-refactor for behavior changes.

Automated tests must prove:

- exact reader-label to canonical-status mappings;
- E5 remains outside the ledger, informational in the Hub, unregistered, and
  unpromoted;
- method families partition all 16 studies exactly once and preserve the
  compatibility visual ID;
- strongest method studies exist inside their family;
- all four canonical themes provide the two new required metadata fields;
- failure counts and definitions come from validated canonical evidence;
- the unreliable-evidence mechanism has no fabricated prevalence count;
- required failure fields and source/status labels render;
- issues #39–#42 contain all queue contract fields and correct dependencies;
- the worked example is numbered, synthetic, privacy-safe, and keeps ranking
  construction separate from evaluation;
- the current-state DOM order is answer-first and has one H1 with ordered
  headings;
- mobile cards stack without page-level horizontal overflow;
- disclosures are keyboard-operable and uniquely named;
- generated-output drift remains clean.

Final verification includes the full pytest suite, Ruff, ledger and Hub drift
checks, `git diff --check`, complete Quarto render, security/link/privacy audit,
desktop and 390-pixel browser inspection, keyboard-only use, light/dark,
reduced motion, grayscale, forced colors, print/no-script, and heading order.

## Delivery and non-goals

After verification, refresh only the existing sanitized private Research Hub
snapshot and verify the tailnet-only HTTPS copy. Do not enable Funnel, publish
publicly, upload artifacts, run models, move evidence, or expose raw responses.

This change does not:

- alter ranking algorithms or experiment results;
- promote E5 or register an off-ledger result;
- replace the experiment ledger;
- create or update GitHub issues;
- redesign the global navigation;
- add a speculative canonical theme;
- claim hidden-test or accessibility conformance.

## Success criteria

A technically capable newcomer can identify the current development direction,
explain how the promoted ranking is constructed, distinguish evidence routes
from method families and canonical themes, find three concrete failure
mechanisms, understand the #39–#42 dependency queue, and know where a new
experiment record belongs without reading the 98-system ledger or any private
artifact.

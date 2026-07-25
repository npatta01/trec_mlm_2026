# Semantic-Demand Literature Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace four overlapping untracked taxonomy drafts with one source-backed Research Hub background page that preserves a rich semantic-demand vocabulary without presenting it as a validated router.

**Architecture:** Add one canonical QMD background record following the existing Research Hub metadata contract. Lock its evidence boundary with targeted tests, then use the existing generator to update catalogs and the background index; no new runtime module or schema is introduced.

**Tech Stack:** Quarto Markdown, YAML front matter, Python/pytest, existing `scripts/build_research_hub.py`, Quarto CLI.

## Global Constraints

- The existing tracked team-method report remains authoritative for what TREC 2025 teams actually did.
- The four untracked source drafts are not added, renamed, edited, or deleted.
- The semantic-demand representation is a hypothesis and diagnostic vocabulary, not a validated routing method.
- Structured fields may supplement historical-case retrieval but must not act as hard filters or replace observed candidate-response evidence.
- Every extracted field, including nested predicate, argument, answer-contract,
  and context-facet values, must carry confidence/uncertainty, applicable
  alternatives, extractor/version or human-annotation protocol, and a source
  span or stable evidence selector. Record-level provenance alone is invalid.
- Include exactly one compact YAML example.
- Include only directly supporting primary-source links.
- Do not include volatile prices, private response content, raw caches, secrets, or absolute home paths.

---

### Task 1: Canonical semantic-demand background record

**Files:**
- Create: `research/hub/background/semantic-demand-frames.qmd`
- Modify: `tests/research_hub/test_resources.py`
- Modify: `tests/research_hub/test_site_integration.py`

**Interfaces:**
- Consumes: the Research Hub background-record front-matter contract and the evidence boundary in `reports/trec_2025_team_method_granularity_review_2026-07-24.md`.
- Produces: a background record with `resource_id: semantic-demand-frames`, discoverable through `build.sources.background`.

- [ ] **Step 1: Write failing resource assertions**

Add these assertions inside `test_background_records_use_audited_sources_and_named_evidence`:

```python
semantic_frames = background["semantic-demand-frames"]
assert semantic_frames.metadata["canonical_source_path"] == (
    "reports/trec_2025_team_method_granularity_review_2026-07-24.md"
)
normalized = " ".join(semantic_frames.body.split()).lower()
assert "no reviewed trec 2025 team used this frame" in normalized
assert "not a validated router" in normalized
assert "response behavior" in normalized
assert "never hard-filter" in normalized
```

Parse the single YAML fence and assert that the semantic graph, predicate,
arguments and each argument, requested slot, computation steps, answer
contract and each facet, and context facets each contain `field_metadata`.
Require `confidence`, `uncertainty`, `alternatives`, `extraction.mode`,
`extraction.version`, `evidence.source`, and `evidence.span` in every such
metadata block. This must fail against a record that has only one top-level
provenance key.

Change the expected background count in
`test_complete_research_hub_content_and_static_shell` from `7` to `8`.

- [ ] **Step 2: Run the targeted tests and verify they fail**

Run:

```bash
uv run pytest -q \
  tests/research_hub/test_resources.py::test_background_records_use_audited_sources_and_named_evidence \
  tests/research_hub/test_site_integration.py::test_complete_research_hub_content_and_static_shell
```

Expected: failure because `semantic-demand-frames` does not exist and the rendered inventory has only seven background records.

- [ ] **Step 3: Author the canonical background record**

Create `research/hub/background/semantic-demand-frames.qmd` with:

```yaml
---
resource_id: semantic-demand-frames
title: "Semantic-demand frames for relation-aware diagnosis"
resource_type: literature_review
summary: "A source-backed, factorized vocabulary for testing whether relation, requested-slot, answer-contract, and reasoning features improve analogue matching without replacing response evidence."
canonical_source_path: reports/trec_2025_team_method_granularity_review_2026-07-24.md
citations: ["QAMR", "BREAK/QDMR", "AmbigQA", "SituatedQA", "EntityQuestions", "KILT"]
tags: [semantic-representation, relation-aware, answer-contract, failure-analysis]
question_ids: [failure-conditions, query-only-routing, candidate-evidence]
reviewed_on: 2026-07-25
---
```

The prose must implement these sections in order:

1. `## Decision boundary` — state that no reviewed TREC 2025 team used this frame and that it is not a validated router.
2. `## Why coarse taxonomies fail` — distinguish surface topic/entity/answer type from requested relation, missing argument, constraints, operations, and completeness.
3. `## Proposed factorized frame` — a table covering predicate/arguments, requested slot, answer contract, computation graph, entity/context risk, ambiguity, and field-level uncertainty/extraction provenance, each with a directly supporting primary citation. Use KILT as direct support for explicit provenance; do not present RePAQ or ColBERT as supporting field-level extraction provenance.
4. `## Compact example` — exactly one fenced YAML block containing `semantic_graph`, `requested_slot`, `computation_steps`, `answer_contract`, and `context_facets`. Attach the complete field-level metadata contract to each field and applicable nested field; a standalone record-level `provenance` key is insufficient.
5. `## How it would be used` — retrieve historical query-response cases with lexical/dense/structural evidence, then derive candidate discrimination from response behavior or reliable response-quality labels; never hard-filter candidates by frame agreement.
6. `## Validation before ranking use` — blinded 30-target sample, 10 close and 10 deliberately distant cases per target, human `yes/partial/no/ambiguous` labels, and comparison of E5 similarity, a direct same-information-need scorer, and extracted-field compatibility with extraction-agreement and ablation checks.
7. `## Primary sources` — direct links only.
8. `## Deep dive` — link the tracked team-method review and explain that it controls claims about TREC precedent.

- [ ] **Step 4: Run the targeted resource test**

Run:

```bash
uv run pytest -q tests/research_hub/test_resources.py::test_background_records_use_audited_sources_and_named_evidence
```

Expected: PASS.

- [ ] **Step 5: Keep canonical and generated surfaces in one atomic change**

Do not commit the canonical record or focused tests before regenerating the Hub
surfaces in Task 2. The source, tests, both catalogs, and generated Background
index must be committed together so no content commit can leave generated
outputs stale.

### Task 2: Generated Hub integration and verification

**Files:**
- Modify: `research/catalog.json`
- Modify: `research/hub/catalog.json`
- Modify: `research/hub/generated/background-index.qmd`

**Interfaces:**
- Consumes: `research/hub/background/semantic-demand-frames.qmd` through the existing `scripts/build_research_hub.py` discovery path.
- Produces: synchronized machine-readable catalogs, a generated reader index entry, and a validated rendered page at `research/hub/_site/background/semantic-demand-frames.html`.

- [ ] **Step 1: Generate tracked Research Hub outputs**

Run:

```bash
uv run python scripts/build_research_hub.py
```

Expected: both catalogs include `semantic-demand-frames`, and
`research/hub/generated/background-index.qmd` links to
`../background/semantic-demand-frames.qmd`.

- [ ] **Step 2: Verify generator determinism and targeted tests**

Run:

```bash
uv run python scripts/build_research_hub.py --check
uv run pytest -q tests/research_hub/test_resources.py tests/research_hub/test_site_integration.py
```

Expected: generator check passes and all selected tests pass.

- [ ] **Step 3: Render and audit the complete Hub**

Run:

```bash
quarto render research/hub
uv run python scripts/audit_research_hub_site.py --output-dir research/hub/_site
```

Expected: Quarto renders 165 pages and the audit reports no QMD sources or reader targets.

- [ ] **Step 4: Run final repository checks**

Run:

```bash
uv run pytest -q
uv run ruff check tests/research_hub/test_resources.py tests/research_hub/test_site_integration.py
git diff --check origin/main...HEAD
```

Expected: full suite and Ruff pass; the diff check emits no output.

- [ ] **Step 5: Verify the preservation boundary**

Run:

```bash
git diff --name-only origin/main...HEAD
git diff origin/main...HEAD | rg -n '/home/npatta01|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}'
```

Expected: only the spec, plan, canonical QMD, focused tests, and generated Hub files are tracked; the secret/path scan has no matches. The four draft reports and two raw cache files remain untracked.

- [ ] **Step 6: Commit source, tests, and generated outputs atomically**

```bash
git add docs/superpowers/plans/2026-07-25-semantic-demand-literature-consolidation.md \
  research/hub/background/semantic-demand-frames.qmd \
  tests/research_hub/test_resources.py \
  tests/research_hub/test_site_integration.py \
  research/catalog.json research/hub/catalog.json \
  research/hub/generated/background-index.qmd
git commit -m "integrate semantic demand synthesis atomically"
```

Before publishing, inspect every commit after the design commit and verify that
there is exactly one implementation/content commit. Checking out that commit
must show synchronized canonical content, focused tests, both catalogs, and the
generated Background index.

- [ ] **Step 7: Request independent scope review before publishing the branch**

Ask the reviewer to verify the TREC precedent boundary, hypothesis-only language,
primary-source support, absence of raw/private artifacts, generated-catalog
synchronization, and the full verification receipts. Address blocking findings,
then push and open a draft pull request.

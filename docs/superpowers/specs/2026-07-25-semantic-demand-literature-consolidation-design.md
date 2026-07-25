# Semantic-demand literature consolidation design

Date: 2026-07-25

## Objective

Consolidate the useful, non-duplicative material from four untracked taxonomy
literature drafts into one canonical Research Hub background record. Preserve a
rich vocabulary for relation- and answer-type-aware diagnosis without implying
that the representation is a validated router or a method demonstrated by TREC
2025 teams.

## Source material

The consolidation may draw from these local drafts:

- `reports/trec_2026_local_taxonomy_literature_audit_2026-07-23.md`
- `reports/trec_2026_public_semantic_frame_literature_2026-07-23.md`
- `reports/trec_2026_relation_failure_taxonomy_literature_synthesis_2026-07-23.md`
- `reports/trec_2026_router_failure_taxonomy_literature_2026-07-23.md`

The existing tracked report
`reports/trec_2025_team_method_granularity_review_2026-07-24.md` is the
authoritative evidence boundary for what prior TREC teams actually did.

## Deliverable

Add one canonical background page at
`research/hub/background/semantic-demand-frames.qmd`. The page will be
self-contained for a Research Hub reader and will link to the existing
team-method review instead of copying its team-by-team analysis.

The four source drafts remain untracked and are not added, renamed, or deleted.
The new page supersedes them for repository use.

## Content structure

The page will contain:

1. A decision boundary distinguishing verified precedent from a proposed
   representation.
2. A short explanation of why broad domain, topic, entity-type, or answer-type
   taxonomies are too coarse.
3. A source-backed table of useful semantic constructs:
   predicate and arguments, requested slot, answer contract, compositional
   operations, entity/context risk, ambiguity, and field-level uncertainty and
   provenance.
4. One compact YAML example showing how the constructs combine for a single
   query.
5. A TREC-specific usage model: structured fields supplement retrieval of
   historical query-response cases and aid failure diagnosis; they do not act
   as hard filters or replace response evidence.
6. A bounded validation protocol comparing raw dense similarity, a direct
   same-information-need scorer, and extracted-field compatibility on blinded
   human labels.
7. Primary-source links retained from the drafts, limited to sources that
   directly support claims made on the page.
8. A “Deep dive” section satisfying the Research Hub background-record
   convention and linking to the team-method review.

## Evidence and language rules

- State explicitly that no reviewed TREC 2025 team demonstrated the proposed
  semantic-demand representation.
- Describe the schema as a hypothesis and diagnostic vocabulary until a
  blinded validity study shows independent signal.
- Do not claim that matching answer types or relations predicts candidate
  document coverage.
- Keep question labels separate from candidate evidence: labels can locate
  analogous historical cases, while candidate discrimination must come from
  observed response behavior or reliable response-quality supervision.
- Preserve uncertainty and extraction provenance for every proposed field.
- Avoid volatile prices, unpublished raw data, private response content, and
  local absolute paths.

## Repository integration

The existing Research Hub generator will discover the new background record and
update the generated background index and both catalogs. No new generator or
schema is required. Add a targeted resource test that fixes the evidence
boundary and ensures the page remains labeled as unvalidated.

## Verification

- Run the targeted Research Hub resource and site-integration tests.
- Run the Research Hub generator in write mode and then `--check` mode.
- Render all Research Hub pages with Quarto.
- Run the rendered-site audit.
- Run Ruff for any changed Python tests or support code.
- Confirm the staged diff contains no raw caches, private evidence, secrets,
  absolute home paths, or the four superseded drafts.

## Acceptance criteria

- One canonical background page replaces four overlapping drafts.
- The page includes one compact YAML example and a primary-source bibliography.
- Verified prior work and proposed features are visibly separated.
- The page cannot reasonably be read as approval to annotate all 7,759 queries
  or deploy the features as a ranking gate.
- Research Hub generation, tests, render, and audit pass.

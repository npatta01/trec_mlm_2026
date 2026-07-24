# Final relationships integrity report

## Design choices

- Made the canonical study `system_ids` lists a total ownership contract for
  the experiment ledger. Validation rejects every ledger run not owned by at
  least one study, and the catalog, run index, and run-detail renderers reject
  the same condition instead of publishing an unlinked run.
- Did not add an archival or catch-all study. A canonical metadata audit found
  that all 98 current ledger runs already have evidence-backed ownership across
  the existing studies.
- Made question-to-study links an exact reciprocal relation. The validator
  compares question `study_ids` pairs with study `question_ids` pairs and
  reports links missing from either side.
- Corrected the four question-side omissions supported by existing study
  metadata: `candidate-evidence` ↔ `label-ranker-pilots`,
  `failure-conditions` ↔ `candidate-label-matching`,
  `failure-conditions` ↔ `source-query-first`, and
  `response-evidence` ↔ `four-model-reference-review`.
- Restricted canonical hub record IDs to lowercase ASCII slug components:
  `[a-z0-9]+(?:-[a-z0-9]+)*`. Loading and validation also require the ID to
  exactly equal the QMD filename stem. Whole-output rendering repeats the
  identity check before deriving output paths.

## TDD evidence

The initial focused run failed nine new tests for the intended missing
contracts: unsafe IDs were accepted, filename/ID mismatches were accepted,
orphan runs passed validation and rendering, and nonreciprocal question/study
links passed validation.

After implementation:

```text
uv run --extra dev pytest \
  tests/research_hub/test_load.py \
  tests/research_hub/test_validate.py \
  tests/research_hub/test_render.py -q

66 passed
```

The direct run-index test was also observed failing before its guard was added,
then passed together with the run-page orphan test.

Final integration verification is recorded in the handoff and includes the
full Research Hub suite, generator build and current-output check, Quarto
render, Ruff, whitespace validation, and explicit canonical invariant checks.

## Commit

- Subject: `fix: enforce research hub relationship integrity`
- Scope: relationship/identity implementation, regression tests, the three
  corrected question records, and this report.
- The resulting commit hash is reported in the parent-agent handoff because a
  commit cannot contain its own hash.

## Concerns

- The total ledger-ownership rule intentionally makes adding a ledger run and
  linking its canonical study one atomic contribution. A ledger-only addition
  will now fail the hub build.
- The lowercase slug grammar is deliberately narrower than the ledger
  `system_id` grammar. Existing canonical hub records all conform; ledger run
  IDs retain their separate portable filename-component grammar.

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

## Follow-up: approach and dataset reciprocity

A whole-feature re-review found that reciprocity stopped at question-study
links even though approaches and datasets also declare `study_ids`.

### Design and canonical corrections

- Generalized reciprocal-pair validation and applied it to question-study,
  approach-study, and dataset-study relationships. Missing links on either side
  now report the exact pair and field.
- Treated study `approach_ids` and `dataset_ids` as the controlling existing
  evidence. The audit found 21 study-to-approach pairs versus 20 reverse pairs,
  and 62 study-to-dataset pairs versus no reverse pairs.
- Added `label-ranker-pilots` to the `candidate-profiles` approach. No approach
  pairs were removed or invented.
- Populated all six dataset `study_ids` lists with the exact sorted reverse of
  canonical study metadata. This changes relationship navigation only; it does
  not add datasets, studies, results, or evidence claims.
- Regenerated both catalog copies. Approach and dataset `related_ids` now expose
  the same canonical study relationships used by validation and study records.

### TDD evidence

Four focused tests were written first and observed failing because validation
accepted each one-sided approach-study or dataset-study relation. After the
generic reciprocity check was implemented, all four passed together with the
existing question-study reciprocity tests.

Final verification for the integrated follow-up:

```text
uv run --extra dev pytest tests/research_hub -q
137 passed
```

The handoff also records fresh generator build/check, Quarto render, Ruff,
whitespace, clean-worktree, and exact pair-set invariant results.

### Commit

- Subject: `fix: enforce reciprocal study metadata`
- Scope: reciprocal validation and tests, one approach record, six dataset
  records, the two generated catalogs, and this appended report.
- The commit hash is reported in the parent-agent handoff.

### Follow-up concerns

- Dataset reverse lists are intentionally high fan-out because shared
  development inputs and run evidence support many studies. Contributors must
  update both sides atomically; validation now prevents partial updates.

# Final Review Fix Wave Report

## Status

Complete at base commit `ee90fe9`. All requested production, test, and report
changes are included in one scoped fix wave. No official qrel leaf was accessed,
statted, resolved, or hashed, and no real feasibility or experiment workload was
run.

## Implemented findings

1. `write_fit_row` now writes only its own `complete` slot, last. It neither
   scans whole completion datasets nor mutates the root `complete` attribute.
2. `run_feasibility_worker` reads each cluster-count completion dataset exactly
   once after all scheduled writes. Any non-`1` cell raises exactly
   `ValueError("fit construction is incomplete")`; only a complete construction
   sets the root attribute and flushes.
3. `read_fit_row` validates the entire assignment-query-index row against the
   excluded indices plus the unused-cell sentinel, explicitly range-checks every
   active query index, and range-checks every active cluster ID before either is
   used for membership indexing. The single-assignment sentinel check remains.
4. The dead `preflight_atomic_roots` cache API and its complete cache-specific
   test block were deleted while unrelated atomic-publication coverage remains.
5. The review report now attributes the 26,104-insertion whole-branch statistic
   to implementation verification commit `be258b9`, not `HEAD`.

## TDD evidence

### RED

Command:

```text
PYTHONPATH=.venv/lib/python3.13/site-packages python -m pytest -q tests/test_dense_aggregation_fit_file.py::test_validate_fit_file_rejects_negative_active_assignment_query_index tests/test_dense_aggregation_fit_file.py::test_validate_fit_file_rejects_out_of_range_assignment_query_index tests/test_dense_aggregation_fit_file.py::test_validate_fit_file_rejects_out_of_range_assignment_cluster_id tests/test_dense_aggregation_feasibility.py::test_worker_runs_exact_inventory_timing_window_and_publishes_manifest
```

Exact summary:

```text
FFFF                                                                     [100%]
4 failed in 4.62s
```

The failures proved the intended regressions:

- assignment query index `-1` for excluded canonical query 321 did not raise;
- assignment query index `322` escaped as `IndexError: tuple index out of range`;
- cluster ID `20` in the c20 row escaped as `IndexError: tuple index out of range`;
- all 576 synthetic writes performed a full completion read, with
  `write_completion_read_deltas` beginning `[1, 1, ...]` instead of all zeros.

### Focused GREEN

Same four-test command after the implementation:

```text
....                                                                     [100%]
4 passed in 4.45s
```

Fit-file suite:

```text
.....................................                                    [100%]
37 passed in 0.91s
```

Artifact, feasibility, and CLI suites:

```text
........................................................................ [ 46%]
........................................................................ [ 92%]
............                                                             [100%]
156 passed in 36.51s
```

Fresh final combined verification:

```text
PYTHONPATH=.venv/lib/python3.13/site-packages python -m pytest -q tests/test_dense_aggregation_fit_file.py tests/test_dense_aggregation_artifacts.py tests/test_dense_aggregation_feasibility.py tests/test_dense_aggregation_cli.py
........................................................................ [ 37%]
........................................................................ [ 74%]
.................................................                        [100%]
193 passed in 36.82s
```

## Static verification

The final command combined the stale-cache scan, compilation, Ruff check of
every changed Python file, and diff whitespace validation. Exact output:

```text
All checks passed!
exit=0
```

Thus `rg -n "preflight_atomic_roots|cache_dir" trec_mlm tests` found no
matches; `python -m compileall -q trec_mlm tests` and `git diff --check` were
silent and successful; and Ruff reported no lint findings. The repository has
no Ruff formatter configuration. An exploratory default `ruff format --check`
identified pre-existing formatting differences in
`trec_mlm/dense_aggregation_feasibility.py` and
`tests/test_dense_aggregation_feasibility.py`; those unrelated lines were not
rewritten. The changed fit-file validation block was aligned with the default
formatter.

## Files changed

- `trec_mlm/dense_aggregation_fit_file.py`
- `trec_mlm/dense_aggregation_feasibility.py`
- `trec_mlm/dense_aggregation_artifacts.py`
- `tests/test_dense_aggregation_fit_file.py`
- `tests/test_dense_aggregation_feasibility.py`
- `tests/test_dense_aggregation_artifacts.py`
- `reports/issue_13_dense_aggregation_foundation_review.md`
- `.superpowers/sdd/final-review-fixes-report.md`

## Self-review

- Completion ordering is unchanged at the slot level: elapsed time is written,
  then the slot completion byte is written last.
- The seal check happens after the scheduled loop and before root completion.
- Each completion dataset is read once with `[...]`; the regression observes
  zero full reads during each of 576 writes and totals `{20: 1, 40: 1}`.
- Query-index validation precedes assignment `query_ids` indexing, and cluster-ID
  validation precedes `validated_memberships` indexing.
- Exact-row comparison covers active indices and the unused single slot; the
  explicit single-slot sentinel validation remains intact.
- The cache deletion stops before the first unrelated atomic-publication test.
- No scientific status or claim was changed in the review report.
- No unrelated worktree changes were present or modified.

## Concerns

None. Verification used only synthetic/local fixtures and qrel-safe focused
suites.

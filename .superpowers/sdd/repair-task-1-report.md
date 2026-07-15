# Repair Task 1 Report: Supplemental qrel-free viability receipt

## Outcome

- Implementation commit: `1915464` (`fix: complete issue 18 viability receipt`)
- Frozen artifact aggregate observed by the read-only assessment:
  `2a3909cfea23af246eed59565557d57ec4c5b8eb1e2f1d5c2b11d2f9f8c6d5b9`
- Recomputed decision: `stop-query-sensitivity`
- Failed query-sensitivity thresholds:
  `maximum_top1_candidate_share`, `maximum_candidate_top10_query_share`
- Failed integrity thresholds: none
- Failed incremental thresholds: none
- Repeat authorized: false
- Qrels read: no
- Repeat or production reconstruction run: no
- Real untracked receipt regenerated or overwritten: no

## TDD evidence

### RED

The prescribed command could not start because this worktree's venv has no
pytest entry point:

```text
$ .venv/bin/pytest tests/test_profile_multiresolution.py -q
/bin/bash: line 1: .venv/bin/pytest: No such file or directory
```

The venv Python also has no pytest module. The repository tests were therefore
run with the available global pytest and `PYTHONPATH=.` for the RED cycle:

```text
$ PYTHONPATH=. pytest tests/test_profile_multiresolution.py -q
6 failed, 57 passed in 0.55s
```

The six expected failures were missing
`_compare_top10_rankings`, `_control_comparisons`, and
`_viability_decision` behavior. A second RED slice added strict control source,
provenance, classification, and no-reconstruction CLI coverage:

```text
$ PYTHONPATH=. pytest tests/test_profile_multiresolution.py -q
5 failed, 62 passed in 0.58s
```

The failures were the missing `_load_control_rankings`,
`_execution_provenance`, and corrected decision classification. The final CLI
RED slice failed because the existing canonical artifact still raised
`FileExistsError` instead of entering receipt-only repair mode.

### GREEN

The final focused command used project dependencies from the venv and appended
the global pytest installation path only because pytest is absent from the
venv:

```text
$ .venv/bin/python -c 'import sys; sys.path.append("/home/npatta01/anaconda3/lib/python3.13/site-packages"); import pytest; raise SystemExit(pytest.main(["tests/test_profile_multiresolution.py", "-q"]))'
.....................................................................    [100%]
69 passed in 0.55s
```

Changed-file lint and whitespace validation:

```text
$ ruff check trec_mlm/profile_multiresolution.py scripts/run_multiresolution_profiles.py tests/test_profile_multiresolution.py
All checks passed!
$ git diff --check
# no output
```

## Frozen-artifact read-only assessment

Command (no receipt output path and no qrels):

```text
$ OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -c '... assess_multiresolution_viability("artifacts/multiresolution_profiles/issue18_pr17head") ...'
{"aggregate": "2a3909cfea23af246eed59565557d57ec4c5b8eb1e2f1d5c2b11d2f9f8c6d5b9", "controls": ["query_knn", "contextual_e5"], "decision": "stop-query-sensitivity", "failed_incremental": [], "failed_integrity": [], "failed_query_sensitivity": ["maximum_top1_candidate_share", "maximum_candidate_top10_query_share"], "repeat_authorized": false, "systems": 8}
```

The real untracked receipt remained present and was not staged. Its observed
SHA-256 after verification was
`a457df196f7117e3d81dca9cde899cd392d4da4bd7d56653d062d4eaa67cf525`.

## Full-suite integration status

The full suite was run with the same venv-plus-pytest-path technique:

```text
$ .venv/bin/python -c 'import sys; sys.path.append("/home/npatta01/anaconda3/lib/python3.13/site-packages"); import pytest; raise SystemExit(pytest.main(["-q"]))'
290 passed, 4 failed in 10.78s
```

All four failures are dependent Task-2 evaluation fixtures that still build the
old receipt schema:

1. `test_evaluation_rejects_qrel_file_change_while_loading`
2. `test_evaluation_reports_every_frozen_system_paired_evidence_and_track_a`
3. `test_evaluation_detects_any_construction_change_and_keeps_output_external`
4. `test_viability_receipt_must_authorize_and_bind_evaluation`

Each fails at the strict top-level schema check before qrels are accessed.
Task 1 deliberately did not modify `tests/test_profile_multiresolution_evaluation.py`;
Task 2 owns those receipt fixtures and evaluation authorization semantics.

## Files changed

- `trec_mlm/profile_multiresolution.py`
  - exact two-control loading and source/hash/inventory binding
  - deterministic top-10 overlap/displacement metrics for all eight systems
  - four-way decision classification and repeat authorization
  - strict supplemental schema, decision, cost, provenance, and optional
    artifact-backed comparison recomputation
- `scripts/run_multiresolution_profiles.py`
  - receipt-only repair path for an existing canonical artifact
- `tests/test_profile_multiresolution.py`
  - RED/GREEN coverage for comparisons, decisions, source mismatch,
    provenance, complete receipt shape, and no-reconstruction CLI behavior

## Self-review

- Confirmed no qrel parameter or qrel loader was introduced.
- Confirmed receipt-only repair calls the strict loader/assessor and never
  `construct_multiresolution`.
- Confirmed query-kNN is byte-hash bound and must equal the frozen artifact
  copy; contextual E5 is loaded only from the strictly validated issue-12
  source aggregate.
- Confirmed decision precedence is integrity, query sensitivity, incremental,
  then continue.
- Confirmed repeat authorization is true only for `continue` with both cost
  booleans true.
- Confirmed the exact execution-code commit, command, single-thread
  environment, dependency versions, construction completion timestamp, and
  artifact aggregate are locked in provenance validation.
- Known dependent concern is limited to the four old-schema Task-2 evaluation
  fixtures listed above.

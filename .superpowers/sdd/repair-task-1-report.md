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

## Review-rejection repair (authoritative validation)

The first Task-1 implementation was rejected because strict validation still
allowed an optional-artifact path and derived too much truth from editable
receipt fields. Commit `1915464` is therefore superseded by the follow-up
repair documented here.

### Review RED

New tests first required a persisted artifact, exact authoritative
recomputation, and removal of complete-looking in-memory controls:

```text
$ PYTHONPATH=. pytest tests/test_profile_multiresolution.py -q
7 failed, 69 passed in 0.70s
```

The failures demonstrated all three rejected behaviors: the validator still
accepted an aggregate string without a persisted artifact, tampered receipt
evidence was not checked against a recomputed receipt, and the public assessor
still accepted an in-memory `MultiResolutionResult`.

A subsequent timestamp-specific RED test proved that an arbitrary editable
`receipt_generated_at` value could be mirrored into recomputed output:

```text
$ PYTHONPATH=. pytest tests/test_profile_multiresolution.py -q
1 failed, 77 passed in 0.56s
```

### Review GREEN

The final focused verification after the repair:

```text
$ PYTHONPATH=. pytest tests/test_profile_multiresolution.py -q
78 passed in 0.54s
$ ruff check trec_mlm/profile_multiresolution.py scripts/run_multiresolution_profiles.py tests/test_profile_multiresolution.py
All checks passed!
$ git diff --check
# no output
```

The full-suite result is:

```text
$ .venv/bin/python -c 'import sys; sys.path.append("/home/npatta01/anaconda3/lib/python3.13/site-packages"); import pytest; raise SystemExit(pytest.main(["-q"]))'
299 passed, 4 failed in 7.78s
```

The four failures remain exactly the dependent Task-2 old-schema evaluation
fixtures listed earlier in this report. They now fail because Task 2 still
passes an aggregate string where authoritative validation requires the
persisted artifact path.

### Review repair details

- `validate_multiresolution_viability_receipt(payload, artifact_path)` now
  requires an existing artifact directory, strictly loads it, recomputes the
  complete canonical receipt, and exact-compares the supplied payload.
- Thresholds, cluster/ranking diagnostics, failure lists, decision, control
  sources, inventory digests, all control comparisons, aggregate, cost, and
  locked provenance are consequently derived from authoritative inputs.
- Receipt-generation time must be timezone-aware, no earlier than the manifest
  completion time, and not in the future. The construction completion time is
  recomputed from the manifest's nanosecond mtime; the frozen artifact yields
  exactly `2026-07-15T12:41:18.692102441-04:00`.
- Public `assess_multiresolution_viability` rejects in-memory results. The
  internal in-memory function now returns only a clearly named diagnostic and
  omits controls, provenance, cost, artifact binding, and repeat authorization.
- The repeat construction/comparison gate invokes full artifact-backed strict
  validation before checking `continue` and cost authorization. Tests cover
  forged/missing artifacts, tampered authoritative fields, omitted repeat
  flags, and `stop-query-sensitivity` never reaching construction.
- No frozen artifact, real untracked receipt, qrel, or repeat construction was
  modified or opened during the repair.

## Review Important repair: locked construction completion timestamp

The follow-up review found that the prior repair treated the mutable
`manifest.json` mtime as authoritative receipt data. The persisted assessment
could therefore emit a different `construction_completed_at` after an
otherwise byte-identical artifact was copied.

### Timestamp RED

The regression test first created a byte-identical copied artifact with an
altered manifest mtime and required persisted assessment to reject it:

```text
$ PYTHONPATH=. pytest tests/test_profile_multiresolution.py -q
1 failed, 77 passed in 0.56s

E Failed: DID NOT RAISE <class 'ValueError'>
```

### Timestamp GREEN and verification

The implementation now normalizes the observed manifest mtime only to
corroborate the locked pass-one timestamp. Any mismatch is rejected, while
provenance always emits the non-overridable
`PASS_ONE_CONSTRUCTION_COMPLETED_AT` constant. Receipt validation recomputes
the same persisted assessment and exact-compares the locked provenance.

Read-only corroboration against the real frozen artifact:

```text
$ PYTHONPATH=. python - <<'PY'
from pathlib import Path
from trec_mlm.profile_multiresolution import _manifest_completion_timestamp
print(_manifest_completion_timestamp(Path('artifacts/multiresolution_profiles/issue18_pr17head')))
PY
2026-07-15T12:41:18.692102441-04:00
```

Focused tests and lint:

```text
$ PYTHONPATH=. pytest tests/test_profile_multiresolution.py -q
78 passed in 0.53s
$ ruff check trec_mlm/profile_multiresolution.py tests/test_profile_multiresolution.py
All checks passed!
$ git diff --check
# no output
```

Full-suite integration status remains exactly the four known Task-2 fixture
failures:

```text
$ .venv/bin/python -c 'import sys; sys.path.append("/home/npatta01/anaconda3/lib/python3.13/site-packages"); import pytest; raise SystemExit(pytest.main(["-q"]))'
299 passed, 4 failed in 8.21s
```

The failure inventory is unchanged: the four
`tests/test_profile_multiresolution_evaluation.py` fixtures still pass an
aggregate string where authoritative validation requires a persisted artifact
path. No frozen artifact, receipt, or qrel was modified.

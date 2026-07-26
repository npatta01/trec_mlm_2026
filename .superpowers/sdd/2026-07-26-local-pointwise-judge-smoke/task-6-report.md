# Task 6 report — local pointwise judge smoke reporting

## Status

COMPLETE

## Scope

- Added `trec_mlm/pointwise_report.py` for strict private summary construction,
  reviewer-consensus agreement metrics, diagnostic-proxy rollups, a private
  Markdown renderer, and whitelist-based public sanitization.
- Added `scripts/build_pointwise_judge_smoke_report.py` for the planned
  artifact-directory command. It writes `smoke-review.md` and
  `smoke-summary.json` privately, plus a separate sanitized JSON file.
- Added fabricated-only coverage in `tests/test_pointwise_report.py`; no real
  artifact, model, reviewer, or endpoint was read.

## Metric boundary

The report deliberately contains no gold labels or accuracy claim. It reports
exact and good/partial-adjacent agreement separately against Advisor and Claude,
then agreement and false-confident errors only where those reviewers concur.
It also records reviewer disagreement, schema/repair/failure rates, labels,
reference-conflict counts, latency, token throughput, startup/memory facts, and
selection-reason diagnostic rollups.

## Privacy boundary

The private case table includes the case text, anonymous judge records, both
review passes, and proposed prompt changes. `sanitize_summary` is a whitelist,
not a key-removal filter: it serializes only aggregate metrics, anonymous judge
labels, runtime facts, hashes, and unresolved counts. It excludes case text and
IDs, candidate IDs, model aliases/IDs, reviewer rationale/findings, raw output,
attempt bodies, paths, and other private artifact data.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache-task6 uv run pytest \
  tests/test_pointwise_judge.py tests/test_pointwise_sampling.py \
  tests/test_pointwise_runner.py tests/test_pointwise_review.py \
  tests/test_pointwise_report.py tests/test_pointwise_judge_cli.py -v
```

Outcome: `265 passed in 0.27s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task6 uv run ruff check \
  trec_mlm/pointwise_report.py scripts/build_pointwise_judge_smoke_report.py \
  tests/test_pointwise_report.py
UV_CACHE_DIR=/tmp/uv-cache-task6 uv run ruff format --check \
  trec_mlm/pointwise_report.py scripts/build_pointwise_judge_smoke_report.py \
  tests/test_pointwise_report.py
git diff --check
```

Outcome: all checks passed.

The full probe (`UV_CACHE_DIR=/tmp/uv-cache-task6 uv run pytest -q`) produced
`1194 passed, 3 skipped, 6 failed in 38.09s`. The six unchanged failures are
the restricted-host baseline: four Quarto cache/database render failures and
two Chrome crashpad socket failures. The Task 6 focused and Task 1–5 regression
suite passed.

Repository-wide Ruff lint passed. Repository-wide format checking reports 78
pre-existing files that Ruff would reformat; the three Task 6 Python files were
checked separately and are formatted.

## Concerns

The reporting CLI expects the Task 2/3/5 private layout:
`smoke_manifest.json`, `runs/<sealed-alias>.json`,
`reviews/{advisor,claude,sealed-judge-aliases}.json`, and optional
`runtime-<sealed-alias>.json` receipts. It requires a mode-0700 artifact
directory and mode-0600 inputs, but does not add new external hash arguments;
the run/reviewer validators still fail closed on content-identity and schema
drift before reporting.

## P1 redaction follow-up

The diagnostic-proxy boundary now imports the frozen ten Task 2
`DIAGNOSTIC_REASONS`. Summary construction rejects unknown, missing, and
duplicate selection reasons before validating run records or building any
output. Public sanitization iterates only that frozen allowlist, so even an
arbitrary extra key injected into an otherwise valid private summary cannot be
serialized. The regression uses `candidate answer: PRIVATE SECRET` as the
hostile selection reason and verifies that the public JSON does not contain the
secret.

Focused P1 verification passed: `50 passed` across Task 6 reporting, Task 2
sampling, and Task 5 review tests; focused Ruff lint, format, and
`git diff --check` also passed.

# Task 3 report — loopback-only resumable inference runner

## Status

DONE_WITH_ENVIRONMENT_CONCERNS

## Implementation commit

- `4fac5ac add resumable local pointwise runner`

## Exact implementation files

- `trec_mlm/pointwise_runner.py`
- `scripts/run_local_pointwise_judge.py`
- `tests/test_pointwise_runner.py`
- `tests/test_pointwise_judge_cli.py`

This report is recorded separately so it can truthfully name the immutable
implementation commit. Untracked `research/notes/` content was preserved and
excluded. No real private artifact was opened and no model server was started.

## TDD evidence

### Initial RED

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner \
  uv run pytest tests/test_pointwise_runner.py -v
```

Outcome: expected collection failure, exit 2, with
`ModuleNotFoundError: No module named 'trec_mlm.pointwise_runner'` before any
Task 3 production module existed.

### Initial GREEN

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner \
  uv run pytest tests/test_pointwise_runner.py -v
```

Outcome: `30 passed in 0.03s` after the first minimal runner implementation.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner \
  uv run pytest tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py -v
```

Outcome: `39 passed in 0.13s`, including the fabricated one-model CLI flow.

### Partial-repair resume RED/GREEN

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner \
  uv run pytest tests/test_pointwise_runner.py \
  -k interruption_during_repair -v
```

RED outcome: `1 failed, 30 deselected`; the failure was the expected
`KeyError: 'r1'`, proving that a malformed first response had not yet been
checkpointed when the repair call was interrupted.

GREEN outcome after checkpoint/resume implementation: `1 passed, 30
deselected`. The resumed run issues only the repair request and retains both
attempts.

### Ambient-proxy RED/GREEN

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner \
  uv run pytest tests/test_pointwise_runner.py \
  -k disables_ambient_http_proxies -v
```

RED outcome: `1 failed, 31 deselected`; no empty `ProxyHandler` was installed.

GREEN outcome after disabling environment proxies: `1 passed, 31 deselected`.

## Final verification

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py \
  tests/test_pointwise_judge.py tests/test_pointwise_sampling.py -v
```

Outcome: `76 passed in 0.13s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run ruff check \
  trec_mlm/pointwise_runner.py scripts/run_local_pointwise_judge.py \
  tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run ruff format --check \
  trec_mlm/pointwise_runner.py scripts/run_local_pointwise_judge.py \
  tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py
```

Outcome: `All checks passed!` and `4 files already formatted`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest -q
```

Outcome: `1005 passed, 3 skipped, 6 failed in 24.07s`. The six failures are
outside Task 3 and match Task 2's documented restricted-host baseline: four
Quarto render probes fail because its cache/database is unavailable, and two
headless Chrome probes fail because crashpad's `setsockopt` is denied. All
Task 1–3 focused and regression tests are green.

`git diff --check` and `git diff --cached --check` also passed before the
implementation commit.

## Self-review

- Endpoint validation accepts only literal `127.0.0.1` or `::1`, normalizes an
  exact `/v1` base path, and rejects DNS names, userinfo, queries, fragments,
  non-loopback addresses, and alternate paths.
- HTTP transport disables ambient proxies and redirects. The request includes
  the unchanged strict JSON Schema response format and no authorization header.
- Responses fail closed on a redirect, non-2xx status, malformed body, missing
  or empty choices/content, returned exact-model mismatch, and nonnumeric,
  nonfinite, boolean, or negative usage.
- Every attempt retains its content and complete response object only inside the
  private mode-`0600` atomic cache. A newly created parent directory is mode
  `0700`; sibling temporary files are fsynced and removed.
- Resume binds each record to the candidate input hash, prompt hash, schema
  hash, model alias, exact model ID, and runtime-receipt hash. A change in any
  dimension rejects reuse. Completed records and interrupted first attempts are
  both resumable.
- Invalid output receives exactly one repair request containing the malformed
  text under the unchanged schema. A second invalid output records an explicit
  `invalid_decision_after_repair` failure and does not issue a third request.
- CLI verification uses only fabricated manifest/receipt data and an injected
  fake call. Runner tests similarly inject scripted calls or a fake HTTP
  transport; no test uses the network.

## Concerns

- The full repository suite cannot be entirely green in this restricted host
  because Quarto and Chrome require operations denied by the sandbox. These
  failures are unrelated to the Task 3 files and reproduce the prior Task 2
  baseline.
- The real CLI remains intentionally unexecuted until a separately verified
  local model server and genuine private manifest/runtime receipt are available.

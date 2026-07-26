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

## Review fix round 1

All four rejected review findings were addressed with regression-first changes:

1. **Cache integrity now fails closed.** A same-identity complete or partial
   record is reusable only after exact record-key, attempt-state, parsed/raw,
   response-body, model, usage-total, failure, and outcome validation. Corrupt
   run containers and container/record identity conflicts raise instead of
   silently reusing or rerunning. Legitimate content-identity changes still
   rerun the case.
2. **CLI private inputs are authenticated before inference.** The Task 2
   manifest must have its exact version-1 schema, source hashes, ordered frozen
   diagnostic reasons, ten exact case rows, uniqueness invariants, and a valid
   canonical manifest hash. The Task 4 receipt format had not yet been
   implemented, so Task 3 defines and validates a minimum explicit version-1
   receipt consistent with the plan: exact keys, canonical receipt hash,
   successful preflight, requested alias/model and endpoint binding, registry
   and checkpoint hashes, runtime/backend versions, `gfx1151`/memory/disk
   evidence, exact frozen decoding settings, endpoint model list, reparsed
   schema canary, startup time, and download flag.
3. **HTTP reads are bounded.** Responses are capped at 1 MiB. Oversized,
   malformed, or negative `Content-Length` values are rejected before reading;
   undeclared responses are read at most cap-plus-one, and declared short reads
   fail closed.
4. **Cache parents are private before access.** An existing output parent must
   be a real directory with exact mode `0700` before the existing cache is read.

### Review RED/GREEN evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_runner.py -k same_identity -v
```

Initial RED: `10 failed`; nine corrupt completed records were silently reused
and one corrupt partial record was silently rerun. After strict validators and
two additional run-container cases, GREEN: `12 passed`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_judge_cli.py -k runner_cli -v
```

Initial RED: the valid fixture and duplicate-key case passed, while all `13`
manifest/receipt corruption cases reached the injected forbidden server call.
GREEN after exact validation: `15 passed`. A self-review type-confusion case
(`max_concurrency=True`) then failed RED and passed GREEN after strict numeric
type validation.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_runner.py -k 'content_length or undeclared_response' -v
```

Initial RED: `5 failed` across declared oversized/invalid/negative lengths,
undeclared oversized content, and a short declared body. GREEN after bounded
reads: all five passed. The existing non-private-parent regression likewise
failed RED because cache JSON was parsed first and passed GREEN after moving the
exact-mode check ahead of cache access.

### Review-round final verification

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py \
  tests/test_pointwise_judge.py tests/test_pointwise_sampling.py -v
```

Outcome: `110 passed in 0.17s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run ruff check \
  trec_mlm/pointwise_runner.py scripts/run_local_pointwise_judge.py \
  tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run ruff format --check \
  trec_mlm/pointwise_runner.py scripts/run_local_pointwise_judge.py \
  tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py
```

Outcome: `All checks passed!` and `4 files already formatted`.

The scoped secret scan returned no matches. The full repository suite produced
`1039 passed, 3 skipped, 6 failed in 24.00s`; the same four Quarto and two
Chrome sandbox failures remain outside Task 3. No real private artifact,
network endpoint, or model server was used.

## Review fix round 2

All three rejected findings were addressed:

1. **Authenticity now comes from out-of-band commitments.** The CLI requires
   independently supplied SHA-256 values for the exact manifest, runtime
   receipt, and Task 4 registry bytes. In-band manifest/receipt hashes remain
   useful integrity checks but are no longer treated as trust anchors. Rehashed
   file tampering fails against the unchanged external digest before inference.
2. **The Task 4 registry contract is explicit and bound to the receipt.** Until
   Task 4 creates the reviewed file, tests use a fabricated version-1 registry
   with exact top-level structure and the exact four aliases from the plan. Each
   descriptor has exact model ID/runtime/download authorization, exactly one
   cached revision or GGUF filename, expected bytes/checkpoint hash, and frozen
   `4096/0/384/1` settings. The selected descriptor must match the requested
   alias/model and the receipt's registry hash, runtime, checkpoint, settings,
   and download evidence.
3. **Decisions and private inputs fail closed at their byte boundaries.** Raw
   decisions reject duplicate JSON object keys, including live and cached repair
   paths. Manifest, receipt, and registry loading uses one opened descriptor with
   `O_NOFOLLOW`/`O_CLOEXEC` where available, `fstat`, bounded cap-plus-one reads,
   exact opened-byte hashing, duplicate-key parsing, and post-read inode/metadata
   comparison. Symlinks, path swaps, unsafe modes, and inputs over 4 MiB are
   rejected before any model call.

### Round-2 RED/GREEN evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_judge_cli.py \
  -k requires_independent_input_digest_anchors -v
```

RED: `1 failed`; the old unanchored invocation reached the injected forbidden
model call. GREEN: `1 passed` after making the three external digests and
registry path required CLI inputs.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_judge_cli.py \
  -k 'consumes_authentic or inauthentic_manifest or inauthentic_runtime or rehashed_registry' -v
```

RED: `3 failed, 13 passed`; rehashed manifest and receipt bodies plus a changed
registry reached the forbidden call. GREEN: `16 passed` after descriptor-bound
external hashing and registry/receipt validation.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_judge.py -k duplicate_object_keys -v
```

RED: `1 failed` because duplicate `label` keys were silently accepted. GREEN:
`1 passed`; live and cached repair regressions subsequently passed as well.

### Round-2 final verification

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py \
  tests/test_pointwise_judge.py tests/test_pointwise_sampling.py -q
```

Outcome: `135 passed` after the final input-duplicate coverage expansion.

Ruff and format checks passed across all six changed Python files. The scoped
code/test secret scan produced no matches.

The full repository probe produced `1062 passed, 3 skipped, 6 failed in
23.95s`. The failures are the same restricted-host baseline: four Quarto
cache/database failures and two Chrome crashpad socket failures. No real private
artifact, registry, network endpoint, or model server was used.

## Review fix round 3

The final planned review round removed Python equality coercion from trusted
decision evidence. A centralized structural JSON comparator now requires exact
container and scalar types, exact string mapping keys, recursive list equality,
and finite numeric values. Consequently, `true` is distinct from `1`, `false`
is distinct from `0`, and `1.0` is distinct from `1`. NaN and infinity are
rejected rather than normalized.

The comparator is used for cached `attempt.parsed`, final cached decisions, and
runtime-receipt canary raw/parsed consistency. Nearby equality checks were
audited: identity/model/hash/failure comparisons contain validated strings;
configuration fields already have exact bool-sensitive validation and
explicitly permit integer or float temperature; usage intentionally permits
integer and float counts but rejects booleans and non-finite values before
aggregation. The Task 1 score parser already used exact `type(value) is int`;
direct coverage now includes booleans for correctness, completeness, and
relevance.

### Round-3 RED/GREEN evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_runner.py \
  -k 'boolean_substitution or type_correct_cached_integer' -v
```

RED: `4 failed, 1 passed`. Both `1 -> true` and `0 -> false` tampering in
`attempt.parsed` and the final decision were silently reused with zero calls;
the legitimate integer cache passed. GREEN after structural comparison:
all five passed. Additional integer/float and non-finite regressions also pass.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task3-runner uv run pytest \
  tests/test_pointwise_judge_cli.py \
  -k boolean_substitution_in_anchored_canary -v
```

RED: `1 failed`; externally anchored canary evidence with raw correctness `1`
and parsed correctness `true` reached the forbidden model call. GREEN: `1
passed` before any call.

### Round-3 final verification

- Focused Task 1–3 suite: `145 passed in 0.19s`.
- Ruff and format: all checks passed across the four changed Python files.
- Scoped secret scan: no matches.
- Full probe: `1074 passed, 3 skipped, 6 failed in 23.91s`; the same four
  Quarto cache/database and two Chrome crashpad socket restrictions remain.

No private input, network endpoint, or model server was used. The frozen prompt
and schema were not changed.

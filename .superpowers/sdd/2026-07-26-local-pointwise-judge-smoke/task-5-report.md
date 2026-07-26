# Task 5 report — blind two-pass review records

## Status

DONE_WITH_ENVIRONMENT_CONCERNS

## Files changed

- `protocols/pointwise_judge_bakeoff/reviewer-v1.txt`
- `trec_mlm/pointwise_review.py`
- `scripts/build_pointwise_judge_review.py`
- `tests/test_pointwise_review.py`

Existing untracked `research/notes/` content was preserved and excluded. No real
private artifact, reviewer, model, server, or network endpoint was used.

## TDD evidence

### RED

```bash
UV_CACHE_DIR=/tmp/uv-cache-task5 uv run pytest tests/test_pointwise_review.py -v
```

Collection failed before production code existed with
`ModuleNotFoundError: No module named 'scripts.build_pointwise_judge_review'`.

During self-review, a new reviewer-rationale identity-leak regression failed
because only critique findings were checked. The focused test produced one
expected failure before the rationale boundary was added.

### GREEN

```bash
UV_CACHE_DIR=/tmp/uv-cache-task5 uv run pytest tests/test_pointwise_review.py -v
```

Outcome: `20 passed in 0.09s` (and `20 passed in 0.08s` after the self-review
fix).

## Implementation

- Pass 1 exposes exactly response ID, question, provisional reference, and
  candidate response.
- Pass 2 adds only anonymous `judge_*` outcomes. Successful outcomes retain all
  eight strict Task 1 decision fields; terminal failures expose only
  `{"status": "failure"}`.
- Global aliases and per-case judge order are deterministic SHA-256 functions
  of the seed, response ID where applicable, and source alias. Mapping insertion
  order cannot affect either result.
- The sealed alias map is separate from both packets. The CLI keeps every packet
  private at mode `0600` under an exact mode-`0700` review directory.
- Run caches must have exact Task 3 containers, frozen schema and prompt
  identities, exact response coverage, exact case input hashes, consistent
  per-record model/runtime identities, strictly revalidated attempts, and a
  terminal success or explicit terminal failure.
- Reviewer ledgers have exact top-level/row keys, a reviewer allowlist and
  expected-identity option, exact pass-1/pass-2 coverage, strict Task 1 decision
  typing, bounded structured critique text, known anonymous aliases, and an
  optional forbidden-identity set applied to rationales and findings.
- Reviewer JSON parsing rejects duplicate object keys and non-finite values.
  `compare_reviews` validates both ledgers and returns reviewer- and
  response-order-independent comparison rows.
- The CLI descriptor-reads bounded, no-follow, mode-`0600` manifest/run inputs
  against caller-supplied SHA-256 anchors before output mutation. Publication is
  atomic/no-overwrite and accepts only byte-identical idempotent reruns.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache-task5 uv run pytest \
  tests/test_pointwise_judge.py tests/test_pointwise_sampling.py \
  tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py \
  tests/test_pointwise_review.py -q
```

Outcome: `255 passed in 0.24s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task5 uv run ruff check \
  trec_mlm/pointwise_review.py scripts/build_pointwise_judge_review.py \
  tests/test_pointwise_review.py
UV_CACHE_DIR=/tmp/uv-cache-task5 uv run ruff format --check \
  trec_mlm/pointwise_review.py scripts/build_pointwise_judge_review.py \
  tests/test_pointwise_review.py
```

Outcome: all checks passed; all three Python files were already formatted.
`git diff --check` also passed.

Full repository probe:

```text
1184 passed, 3 skipped, 6 failed in 25.03s
```

The six failures are the unchanged restricted-host baseline: four Quarto
cache/database failures and two Chrome crashpad socket failures. None touches
Task 5.

## Self-review

- Searched both public packets for fabricated aliases and exact model IDs;
  neither appears.
- Confirmed packet rows cannot carry selection strata/reasons, source paths or
  hashes, candidate IDs, runtime receipts, attempt data, raw responses, latency,
  usage, verbosity, or failure internals.
- Confirmed direct builders reject incomplete, extra, partial, identity-drifted,
  and corrupted run records before returning a packet.
- Confirmed fabricated CLI tests cover externally anchored inputs, symlink,
  unsafe-mode and oversized-input rejection, exact private modes, idempotence,
  and refusal to create the output directory after invalid input.
- Confirmed no Task 1–4 implementation file or frozen protocol was modified.

## Concerns

- The CLI intentionally requires `--manifest-sha256` and one
  `--run-sha256 alias=sha256` per cache, strengthening the illustrative plan
  command so Task 2/3 private artifacts remain externally authenticated.
- Within the timebox, publication retains the exact mode-`0700` output
  directory check and no-follow/no-overwrite file checks, but does not retain a
  directory file descriptor across all three publications. A hostile concurrent
  parent-directory replacement is therefore deferred hardening; normal
  single-user private artifact generation is covered.
- The full suite remains subject to the unrelated Quarto/Chrome sandbox
  limitations above.

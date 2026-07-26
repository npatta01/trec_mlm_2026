# Task 2 report: private input loading and deterministic diagnostic selection

Status: DONE_WITH_CONCERNS

## Files changed

- `trec_mlm/pointwise_sampling.py`
- `scripts/build_pointwise_judge_smoke.py`
- `tests/test_pointwise_sampling.py`
- `tests/test_pointwise_judge_cli.py`

Task 1's `trec_mlm/pointwise_judge.py`, judge rendering, and protocol contract were not modified. Existing untracked `research/notes/` files were preserved and excluded from the commit.

## Commit

- `31ad6a6 add diagnostic pointwise smoke sampling`

## TDD evidence

### RED

Command requested by the brief:

```bash
uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py -v
```

Initial environmental outcome: execution stopped before collection because the global uv cache was read-only (`Could not acquire lock ... /home/npatta01/.cache/uv/... Read-only file system`).

Equivalent rerun using only a writable cache:

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py -v
```

Outcome: expected RED during collection, exit 2, with exactly the missing-feature errors:

- `ModuleNotFoundError: No module named 'trec_mlm.pointwise_sampling'`
- `ModuleNotFoundError: No module named 'scripts.build_pointwise_judge_smoke'`

### GREEN

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py -v
```

Outcome: `6 passed in 0.07s`.

Final pointwise regression verification after bounded-memory refactor:

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest tests/test_pointwise_judge.py tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py -v
```

Outcome: `24 passed in 0.13s`.

Static verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run ruff check trec_mlm/pointwise_sampling.py scripts/build_pointwise_judge_smoke.py tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py
```

Outcome: `All checks passed!`.

Full repository regression probe:

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest -q
```

Outcome: `953 passed, 3 skipped, 6 failed in 24.77s`. The six failures are outside Task 2: four Quarto integration failures caused by an unavailable cache/database (`unable to open database file` or empty Quarto exit 1), and two headless Chrome probes terminated by crashpad's sandbox-denied `setsockopt`.

## Self-review

- The grounded loader validates the top-level `records` shape, nonempty IDs/questions/answers, key-to-record ID agreement, successful `stop` completion, and the 7,759-record invariant when generation metadata identifies a real cache.
- The discovery loader is a generator over the existing pretty-printed format and retains only one decoded response at a time.
- The selector excludes duplicate question/candidate pairs, keeps only ten best candidates per reason rather than the answer corpus, applies the ten exact proxy labels, and uses SHA-256 over seed/question/candidate only as the stable tie-break.
- Blind cases contain no candidate ID. The sealed mapping is the only response-ID-to-candidate-ID mapping; response IDs hash question ID, candidate answer, and candidate ID.
- The CLI records both source hashes, writes both JSON documents via sibling temporary files and atomic replacement, gives the sealed file mode `0600` (the manifest is also conservatively `0600`), permits an identical idempotent rerun, and rejects a changed manifest hash.
- Mutation review: removing duplicate filtering, identity sealing, any reason, finish validation, source hashing, permission setting, or overwrite comparison is covered by focused behavior tests. Task 1 remains byte-unchanged.

## Concerns

- The complete repository suite cannot be fully green in this restricted environment because Quarto and Chrome attempt operations denied outside the Task 2 scope. The Task 2 and Task 1 regression suites are fully green.
- No real private input was opened, so the real 7,759-record cache and 1.16-million-cell discovery file were intentionally not exercised in this task; all verification used fabricated fixtures as required.

## Round 1 review fix

Status: DONE

Commit: `7f115e3 harden pointwise diagnostic sampling`

### Changes

- Replaced label-only ranking with explicit eligibility predicates for every diagnostic reason and deterministic distinct bipartite matching across bounded eligible pools. The numeric/date regex now has no capturing group, and the fabricated fixture asserts the exact intended answer for all ten reasons.
- Parse grounded JSON fail-closed with duplicate-key detection at every object level.
- Compute one immutable feature record per unique question/candidate pair and use bounded ordered insertion with a constant 64-row maximum per eligible pool.
- Hash both sources before any loading/selection work and verify the same hashes after selection.
- On idempotent reruns, recompute the existing manifest hash from its body, compare it with the regenerated manifest, require the sealed JSON to exactly equal regenerated response-ID/candidate-ID pairs, reject duplicate artifact keys, and require sealed mode `0600`.
- Added `load_discovery_questions` and the optional `questions` mapping on `load_grounded_references`; the CLI performs a validated discovery-question pass so canonical generation records without embedded `question` fields remain supported without accepting empty or mismatched question text.
- Task 1 judge rendering and protocol files remain unchanged.

### RED evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py -v
```

Outcome before production changes: `8 failed, 5 passed in 0.17s`. The eight failures directly demonstrated wrong per-reason answers, collapsed duplicate grounded keys, the absent canonical `questions` compatibility argument, absent single-feature instrumentation, unsupported canonical CLI records, undetected source mutation, trusted tampered manifest content, and trusted corrupted sealed content.

### GREEN evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py tests/test_pointwise_judge.py -v
```

Outcome after final implementation and self-review: `32 passed in 0.10s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run ruff check trec_mlm/pointwise_sampling.py scripts/build_pointwise_judge_smoke.py tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run ruff format --check trec_mlm/pointwise_sampling.py scripts/build_pointwise_judge_smoke.py tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py
```

Outcome: `All checks passed!` and `4 files already formatted`.

### Round 1 self-review and concerns

- Every selected case is drawn from its reason's eligible pool; selection fails closed if any reason lacks an eligible candidate or no distinct ten-case matching exists.
- Candidate answer retention is bounded at 640 rows total (64 per reason); the pair-ID deduplication set retains no answer text.
- Existing untracked `research/notes/` remain untouched and excluded.
- No real private inputs were opened. No open Task 2 concern remains from this review round.

## Round 2 review fix

Status: DONE

Commit: `02537a9 bind pointwise sampling to private snapshots`

### Changes

- Existing idempotent outputs now require exact mode `0600` on both `smoke_manifest.json` and `sealed_candidates.json` before either private artifact is trusted.
- The CLI copies each input through a one-megabyte byte buffer into a mode-`0600` file inside a mode-`0700` temporary directory while computing SHA-256 over those exact copied bytes. All loading and selection consume only the immutable private snapshots, so the recorded source hashes cover exactly the bytes parsed even under an A→B→A pathname race. Temporary snapshots are removed on every context exit.
- Discovery loading now uses one strict streaming state machine for both the question pass and response iteration. It validates the complete canonical pretty-printed JSON structure, JSON string syntax and comma placement; rejects every unmatched or trailing non-whitespace line; rejects duplicate question and per-question response keys after JSON decoding; and preserves bounded memory by retaining only the question-ID set, one question's response-ID set, and one decoded response at a time.
- Task 1 judge rendering and protocol files remain unchanged. Existing untracked `research/notes/` remain untouched and excluded.

### RED evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py -v
```

Outcome before production changes: `5 failed, 13 passed in 0.12s`. Failures showed acceptance of trailing malformed syntax, an unmatched responses line, duplicate response keys, inconsistent A→B→A path hashing/selection, and a mode-`0644` existing manifest.

### GREEN evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py tests/test_pointwise_judge.py -v
```

Outcome after final formatting: `36 passed in 0.10s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run ruff check trec_mlm/pointwise_sampling.py scripts/build_pointwise_judge_smoke.py tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run ruff format --check trec_mlm/pointwise_sampling.py scripts/build_pointwise_judge_smoke.py tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py
```

Outcome: `All checks passed!` and `4 files already formatted`.

### Round 2 concerns

- No real private input was opened. The snapshot path temporarily requires disk space approximately equal to the two source files; copying and parsing remain bounded-memory and snapshots are private and automatically removed.

## Round 3 test-only coverage hardening

Status: DONE

### Coverage added

- Duplicate question keys that differ lexically but decode to the same JSON string (for example `q1` and `\u00711`) are rejected.
- Missing and extra commas between responses/questions, plus premature root EOF, fail closed through representative parser state transitions.
- A canonical two-question `json.dumps(..., indent=2)` writer-format fixture parses to exact question and response records.
- No production file changed because every new coverage test passed against commit `02537a9`.

### Coverage evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest tests/test_pointwise_sampling.py -v
```

Initial outcome after adding coverage: `17 passed in 0.06s`; no defect was demonstrated.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py tests/test_pointwise_judge.py -v
```

Final post-format outcome: `43 passed in 0.11s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run ruff check tests/test_pointwise_sampling.py
UV_CACHE_DIR=/tmp/uv-cache-task2 uv run ruff format --check tests/test_pointwise_sampling.py
```

Outcome: `All checks passed!` and `1 file already formatted`.

### Round 3 concerns

- None. No real private input was opened, and unrelated `research/notes/` remain untouched.

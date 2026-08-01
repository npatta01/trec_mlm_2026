# OpenAI Pointwise Production Batches Implementation Plan

> **For agentic workers:** Execute continuously after preflight; do not pause between clean shards.

**Goal:** Run the remaining substantive pointwise judgments through OpenAI Batch without duplicate billing, silent gaps, or propagation of invalid results.

**Architecture:** Freeze the exact prompt/schema validated in the 500-case pilot. A single Python controller prepares deterministic 500-request JSONL shards; imports only pilot outputs whose full requests still exactly match the sealed inventory and deterministic bare-IDK labels; maintains an atomic private run ledger; keeps at most three batches active; reconciles every response by `custom_id`; and halts new submissions on any anomaly. Only failed or truncated request IDs enter bounded retry shards.

**Tech Stack:** Python 3.13, OpenAI Python SDK 2.46, Pydantic 2.13, JSONL, pytest.

## Global Constraints

- Model is exactly `gpt-5.6-luna` with `reasoning_effort=medium`.
- Primary completion ceiling is 8,192; only non-`stop` or invalid outputs retry at 16,384.
- Provider output is exactly `{verdict, explanation}`; explanation is one sentence and at most 400 characters.
- Exact bare-IDK/no-response records never call the provider and become deterministic `nonanswer` results.
- Reuse only pilot results whose entire original request exactly matches the current sealed inventory; requeue any stale pilot input rather than silently cache it.
- Shards contain at most 500 requests and at most 150 MB of JSONL.
- At most three batches may be active after a clean sequential ramp established operational stability.
- A queued-token/rate-limit rejection defers the shard; it does not trigger a blind retry loop.
- Any missing ID, duplicate ID, non-200 response, provider error, unexpected model, non-`stop` finish, schema failure, or invalid usage is persisted and stops new submissions.
- Exactly one supervisor holds an advisory lock. Every remote mutation is recoverable from durable intent plus Batch metadata; discovery is paginated, SDK POST retries are disabled, and an ambiguous accepted-create window must discover the batch rather than resubmit it.
- Terminal reconciliation unions the output and error files, hashes atomic downloads, and never retries a missing ID while the source batch is nonterminal or ambiguous.
- Pin the official Batch short-context prices (`$0.10/M` input, `$0.01/M` cached input, `$0.125/M` cache writes, `$0.60/M` output), track token-based spend after each batch, enforce a $50 completed-plus-in-flight stop ceiling, and limit three active batches to at most $10 conservative worst-case exposure.
- Inputs, raw outputs, state, and results remain mode 0600 inside mode 0700 directories. Secrets must never be written to artifacts.
- Do not commit, stage, or rewrite unrelated dirty-worktree files.

## Task 1: Freeze the validated contract

**Files:**
- Create: `protocols/pointwise_judge_bakeoff/pointwise-luna-v1.txt`
- Create: `protocols/pointwise_judge_bakeoff/decision-luna-v1.schema.json`

- [x] Copy the exact system prompt and strict JSON schema used by the successful 500-case pilot.
- [x] Verify their hashes and assert the pilot requests contain byte-identical prompt/schema values.

## Task 2: Implement the resumable Batch controller

**Files:**
- Create: `scripts/run_openai_pointwise_production_batch.py`

**Interfaces:**
- `prepare_run(...) -> dict`: validate frozen inputs, import cached/deterministic results, and atomically build shard manifests and JSONL.
- `run_supervisor(...) -> dict`: discover/resume remote batches, enforce bounded concurrent scheduling, download and reconcile terminal batches, build targeted retry shards, and halt on anomalies.
- `status_summary(...) -> dict`: report local/remote progress without mutating provider state.

- [x] Validate the production manifest hash, row count, row schema, ordinal continuity, and unique record IDs.
- [x] Validate all 500 pilot input/result mappings and reuse only the 492 whose full request still matches the current sealed record.
- [x] Create complete local coverage: cached provider result, deterministic IDK result, or exactly one pending custom ID for every production record.
- [x] Build shards deterministically with both request-count and byte limits and validate each generated line before it can be uploaded.
- [x] Persist run identity, shard hashes, intent, file ID, batch ID, status, output/error file IDs, and reconciliation counts with atomic mode-0600 writes and parent-directory fsync.
- [x] Hold a single-supervisor file lock; paginate remote metadata discovery; disable implicit SDK retries for non-idempotent batch creation; and fail closed on unresolved create ambiguity.
- [x] Reconcile by `custom_id`, never line order; reject missing/extra/duplicate identifiers and invalid envelopes.
- [x] Retry only affected identifiers. A second failure is terminal and prevents further submission.
- [x] Assemble a final 182,764-record result only when every ordinal has exactly one validated or deterministic result.

## Task 3: Verification and independent review

**Files:**
- Create: `tests/test_run_openai_pointwise_production_batch.py`

- [x] Verify preparation over the real 182,764-record inventory without network access.
- [x] Verify exact counts: 492 exact-current cached, 734 deterministic IDK/no-response, and 181,538 pending provider calls.
- [x] Simulate out-of-order output, truncated output, retry scheduling, crash/resume discovery, capacity deferral, and final aggregation.
- [x] Run focused tests and a secret scan.
- [x] Obtain an independent fail-closed review of the controller and address load-bearing findings.
- [x] Submit and reconcile the first production shard only after all preflight evidence is clean; increase to three active batches after the clean sequential ramp.

## Status

- [x] Plan created.
- [x] Contract frozen: prompt and schema are byte/structure-identical to the successful 500-case Batch input.
- [x] Controller implemented with explicit provider opt-in, private durable state, crash-safe discovery, bounded retries, exact request/remote identity validation, expected-spend tracking, and completed-plus-in-flight stop gates.
- [x] Full offline preflight clean: 182,764 total = 492 exact-current cached + 734 deterministic + 181,538 provider calls in 364 private shards; 0 anomalies; $35.95 projected total; $3.14 maximum single-shard exposure.
- [x] Independent review clean: GO at controller SHA `596cca09318081ff160164e135d3cbd67fd64d3244792e742e4dd8d882286d`.
- [x] First live shard reconciled: 500/500 HTTP 200, `stop`, schema-valid; local one-sentence regex false positive removed without retry or extra charge.
- [x] First-shard semantic audit: 7 clear supported/unsupported errors in 500 (1.4%); one extra correct/partial disagreement remains within the accepted supported bucket.
- [x] Supervisor retries only transient provider/network errors (408/409/429/5xx, connection, timeout); state, validation, anomaly, and cost failures remain fail-closed.
- [x] Production supervisor completed: 182,764/182,764 records, zero pending records or anomalies, and $31.8202577 estimated Batch cost.
- [x] Full structural audit reconciled every final row to the sealed inventory, controller ledger, and provider output; all 181,538 provider calls finished with `stop` and strict-schema output.
- [x] Curated downstream artifact produced without mutating raw output: four explanation repairs and 55 explicit label overrides (53 duplicate-input consistency fixes plus two semantic-sample supported-boundary fixes).
- [x] Exact-input supported/unsupported inconsistencies reduced from 25 groups to five deliberately unchanged ambiguous/broken-question groups.
- [x] Compact downstream JSONL emitted with `supported = verdict in {correct, partial}`; unsupported subtypes remain debugging signals.
- [x] Private local archive created and verified by SHA-256 at `/home/npatta01/data/artifacts/trec_mlm_2026/openai_pointwise_judgments/openai-luna-medium-v1-500-20260731-8b6ae7d2d8b0/`.
- [x] Private HF Bucket upload completed at the immutable `openai-luna-medium-v1-500-20260731-8b6ae7d2d8b0` prefix; a fresh 1,128-file download passed all 1,127 payload checksums and reproduced both primary artifact hashes.
- [x] Focused production PR #52 merged to `main` at `013e88da`; long-running experimental work archived on private branch `codex/pointwise-judge-archive-20260801` at `13b45b1`.

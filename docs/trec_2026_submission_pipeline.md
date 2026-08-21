# TREC 2026 unseen-query submission pipeline

`trec-mlm submit-run` is the single deterministic entry point that turns an
organizer unseen-query release into one complete, validated, receipted TREC run
(GitHub issue #39). It is qrel-free by construction: no argument, configuration
key, or code path in `trec_mlm/submission.py` accepts relevance judgments, and
inputs that name or resemble a qrels file are rejected before they are read.

## Invariants

- **Qrel-free.** The subcommand has no qrels option. Query inputs shaped like a
  four-column qrels matrix, and any path or configuration value containing
  `qrel`, fail closed.
- **Frozen candidate inventory.** The system configuration embeds the sorted
  150-candidate IDs, their length-framed SHA-256, and the SHA-256 of the tracked
  source manifest (`reports/trec_2026_discovery_profile_audit.json`). Any drift
  is rejected.
- **Frozen system identity.** The configuration pins `system_id`, `run_id`, the
  backend, and the backend's artifact bindings, including the frozen profile
  construction aggregate SHA-256. A changed artifact refuses to score.
- **Complete rankings only.** Every query must rank every candidate exactly
  once; the emitted run is re-validated with `runio.validate_trec_run` against
  the release query IDs and the frozen candidate set before it is kept.
- **Atomic, receipted output.** The run directory is staged and renamed into
  place. `receipt.json` records input SHA-256s, the configuration hash, the
  candidate inventory hash, artifact aggregate hashes, row/query/candidate
  counts, the run tag, tool versions, and a manifest aggregate over the emitted
  files. Repeating an identical invocation reproduces byte-identical
  `run.trec` and `receipt.json`.
- **No fallback ranking.** A missing backend artifact stops the run with an
  error pointing at issue #40. Nothing is substituted.

## Reference system

`configs/trec_2026_submission_profile_scoring.json` freezes the reference
backend `profile_scoring`: the preregistered qrel-free profile score
(maximum cosine between the query embedding and each candidate's frozen
centroids, ties broken by candidate ID). It consumes two validated artifacts:

1. an unseen-query **profile inference artifact** (Stage 1 output of
   `trec-mlm profile-infer` over the release queries) supplying query
   embeddings, passed at invocation time via `--artifact`; and
2. the hash-pinned **frozen profile construction**
   (`run1_record_id_v2_auditfinal`, aggregate
   `04d6be23d2f9046ed93eb1c01bb5ad56545bb51a9470d408b96d3241b1ee54f4`,
   qrel-free by protocol) supplying centroids, bound inside the configuration.

Centroids are never re-derived at submission time, so the release-day run needs
no clustering environment; only the pinned ROCm encoder box for Stage 1.

## Dry run today (no private artifacts required)

The fixture-driven dry run is the test suite:

```bash
uv run --extra dev pytest tests/test_submission.py
```

It covers input validation, qrels refusal, deterministic ranking, exact
candidate coverage, TREC formatting, run tags, receipts, and atomic output,
including a twice-repeated end-to-end execution with byte-identical outputs.

Without the private artifacts, the production command below fails closed with
an `issue #40` restore pointer; that behavior is itself under test.

## Freeze-day invocation

The final test topics are already captured in
`data/test/2026-mllm-topics-final.jsonl` (50 topics, `MLLM_051`–`MLLM_100`).
From the repository root:

```bash
uv run trec-mlm submit-run \
  --queries data/test/2026-mllm-topics-final.jsonl \
  --config configs/trec_2026_submission_profile_scoring.json \
  --artifact artifacts/candidate_semantic_profiles/unseen_inference \
  --output-dir artifacts/submission/trec2026_profile_scoring_v1
```

The command prints the receipt. Review `run.trec` and `receipt.json` manually;
submission to the organizer stays a manual step. Development rehearsal uses the
same command with `--queries data/dev/llm_dev_data.tsv` and the restored
development inference artifact.

## Release-day prerequisites (tracked elsewhere)

1. **Issue #40** restores the frozen construction
   (`run1_record_id_v2_auditfinal`) and the profile inference machinery into
   verified private custody.
2. A Stage 1 `profile-infer` run over the 50 release topics must produce the
   `--artifact` directory. The Stage 1 loader currently pins the official
   development inventory (322 queries); the unseen-query artifact contract
   needs its own pinned counts before production execution.
3. **Issue #41** freezes the five-run portfolio: final run tags, one
   configuration file per submitted system, fallback order, and the operator
   runbook. New backends register in
   `trec_mlm.submission.SUBMISSION_BACKENDS` with a matching entry in
   `BACKEND_CONFIG_VALIDATORS`.

## Out of scope

Per issue #39: no test-qrel access, no post-release tuning, no new ranking
method, no public service, and no automatic organizer upload.

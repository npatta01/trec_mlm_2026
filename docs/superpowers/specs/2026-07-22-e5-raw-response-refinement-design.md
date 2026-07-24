# E5 Raw Historical-Response Refinement Design

## Objective

Test whether fine-grained historical question-and-response evidence improves the
existing candidate-centroid router. The experiment embeds the complete frozen
inventory with `intfloat/e5-mistral-7b-instruct` and changes only the retrieval
granularity. It does not select a new encoder, tune a weight, tune a neighbor
count, or use judgments during construction.

## Frozen scope

- Target queries: all 322 development queries in `data/dev/llm_dev_data.tsv`.
- Candidates: all 150 IDs in
  `reports/trec_2026_discovery_profile_audit.json`.
- Historical records: the same frozen basic filtering, candidate-local exact
  deduplication, canonical ordering, and schema-v2 record identities used by
  the candidate semantic-profile experiment; the expected usable count is
  182,764.
- Encoder: `intfloat/e5-mistral-7b-instruct` at revision
  `d62cc4417be0e3289c03dd459d1ef2d137e2554e`.
- Text contract: the existing profile task instruction, `Question:` / `Response:`
  document format, right truncation to 512 tokens, terminal EOS, last-token
  pooling, float32 persistence, and L2 normalization.
- Random seed: 2026. Batch size: 8.
- Raw neighbor count: 10. Hierarchical weights: centroid 0.8, raw 0.2.
- RRF: equal weights and `k=60` with the frozen dense query-kNN ranking.
- No parameter sweep, per-query adjustment, manual reranking, or fallback model.

## Considered approaches

1. **Complete E5 reconstruction and exact full evaluation — selected.** This is
   the only decisive test of the proposed method over all candidates and
   queries, and it preserves comparability with the current E5 centroids.
2. Candidate-subset E5 pilot. This is cheaper but produces a restricted ranking
   problem whose nDCG is not directly comparable to the 150-candidate baseline.
3. Replace E5 while adding raw retrieval. Newer encoders may be stronger, but
   changing representation and retrieval granularity together would confound
   the result.

## Architecture and data flow

The experiment has four explicit stages.

### 1. Qrel-free inference

Reuse the existing pinned ROCm profile inference path to encode every target
query and every usable historical record. Before the expensive run, execute two
fresh representative smoke encodings on `gfx1151` and write a repeat receipt.
The full inference stage has no qrel mount and no network access.

Persist the complete inference artifact atomically under the repository-owned
shared-data root:

`$TREC_SHARED_ROOT/e5_raw_response_refinement/inference`

The directory must contain query and record arrays, ordered record metadata,
formatted-input hashes, source hashes, runtime/model identity, a file manifest,
and an aggregate SHA-256. It must never be stored only inside a disposable
worktree.

### 2. Qrel-free centroid reconstruction

Feed the durable inference artifact into the existing deterministic candidate
profile constructor. Reconstruct the frozen 10-centroid profiles and verify the
expected 322-by-150 score coverage. Persist construction outputs beside the
inference artifact under `construction/`, with its existing atomic manifest and
validation contract.

### 3. Qrel-free fine-grained scoring and freeze

Add a focused raw-refinement module that consumes only immutable inference,
construction, and frozen dense-ranking artifacts.

For target query `q`, candidate `c`, and that candidate's normalized historical
record vectors `R_c`, calculate:

```text
raw(q,c) = mean(top 10 values of q @ R_c.T)
broad(q,c) = maximum cosine to c's 10 existing centroids
hierarchical(q,c) = 0.8 * broad(q,c) + 0.2 * raw(q,c)
```

If a candidate has fewer than 10 usable records, average all available records;
if it has none, preserve the existing unavailable-candidate behavior. The
official inventory is expected to have usable records for every candidate, but
the behavior remains explicit and tested. Select equal-similarity records by
ascending canonical record ID. Compute cosine dot products with the persisted
normalized float32 matrices, promote the selected similarities before the
top-10 mean, and compute both that mean and the weighted combination in float64
without component renormalization. This matches the operation order frozen in
the execution plan.

Freeze two systems before any qrels are loaded:

- `raw_hierarchical`: candidates ranked by the hierarchical score, using
  candidate ID as the deterministic final tie-break.
- `raw_hierarchical_dense_equal`: equal reciprocal-rank fusion of
  `raw_hierarchical` and the frozen dense query-kNN ranking, with `k=60`.

The output must include all component score matrices, complete rankings,
top-record identities and similarities for inspection, inventory-bias
diagnostics, file hashes, and an aggregate manifest.

### 4. Read-only evaluation

Only after the ranking artifact is atomically finalized and hashed may a
separate evaluator load `data/dev/llm_dev_qrels.txt`. Evaluate macro nDCG@10,
MRR, zero-nDCG query count, per-query metrics, paired 10,000-resample PCG64
95% intervals, win/tie/loss counts, and zero-query transitions.

The sole primary effectiveness comparison is:

- `raw_hierarchical_dense_equal` versus the frozen promoted
  `dense_profile_equal` result (nDCG@10 0.4202139921941393, MRR
  0.45301739378333744, and 99 zero-nDCG queries).

The prespecified secondary mechanism comparison is:

- `raw_hierarchical` versus the frozen `profile_only` result
  (nDCG@10 0.41492445997856287, MRR 0.441697813928065, and 92
  zero-nDCG queries).

The report will also show comparisons against dense query-kNN, but they do not
replace the sole primary comparison.

## Decision rule

Classify the primary approach as **advance** only if all integrity checks pass,
the paired 95% bootstrap lower bound for nDCG@10 delta against
`dense_profile_equal` is above zero, MRR delta is at least -0.005, and the new
system has fewer than 99 zero-nDCG queries. Classify it as **diagnostic** when
the point nDCG@10 estimate improves or at least one prior zero is rescued but
the advance rule fails. Otherwise classify it as **stop**. Any inventory,
identity, finiteness, completeness, qrel-isolation, artifact-integrity, or
repeatability failure produces **invalid**.

Before qrels are opened, stop as **no-op** if neither new system changes any
top-10 candidate list relative to its matched comparator. Secondary metrics and
diagnostics cannot trigger advancement or select a new variant.

## Error handling and resumability

- Refuse to overwrite an existing finalized artifact directory.
- Write into a sibling staging directory and atomically rename only after every
  file, shape, dtype, normalization, inventory, and hash check passes.
- Preserve a failed-run log outside the staging directory; clean incomplete
  staging directories only after their failure is recorded.
- Refuse mixed encoder revisions, reordered record identities, cross-model
  cosine scoring, non-finite values, incomplete 322-by-150 rankings, or qrel
  paths in construction interfaces.
- The full encoder is not batch-resumable inside one array. If interrupted, the
  atomic staging artifact is invalid and the inference stage restarts. This is
  acceptable for the one authorized full run; adding a sharded checkpoint
  format is out of scope.
- Repeat the inexpensive scoring/construction stage twice against the one
  immutable inference artifact and require identical record selections and
  rankings plus score agreement within 1e-12. Do not spend a second 8.8-hour
  encoding run merely to repeat an already immutable dependency.

## Testing and verification

Use test-driven development for all new behavior. Unit tests cover exact top-k
means, fewer-than-10 histories, stable ties, candidate slice boundaries,
hierarchical arithmetic, RRF output, unavailable candidates, finite normalized
inputs, and artifact tamper detection. Integration tests use tiny real numpy
arrays and round-trip the qrel-free scoring artifact. A CLI canary must complete
end-to-end before the full GPU run.

Final verification requires:

- focused and full test suites passing;
- ROCm device and pinned container/model identity checks;
- exact query, candidate, record, and embedding shapes;
- successful reload of every finalized artifact;
- manifest and aggregate SHA-256 verification;
- complete rankings for 322 queries and 150 candidates;
- evidence that qrels were absent until evaluation;
- a tracked Markdown/JSON result report containing the frozen metrics and
  decision.

## Non-goals

This experiment does not compare new embedding backbones, train a learned
router, introduce a reranker or LLM judge, tune `k` or fusion weights, publish
artifacts, or send the discovery corpus to an external service.

### Post-experiment durability exception

After the frozen qrel-free experiment completed, the user explicitly requested
on 2026-07-23 that the generated embeddings be retained in a private Hugging
Face Bucket. This later durability action is outside construction, ranking, and
evaluation and does not alter the experiment. The private loader-ready copy
contains embeddings, ordered record metadata, and the full historical
question/response text in `records.json`; its restore and integrity checks are
documented in `reports/trec_2026_e5_embedding_restore.md`.

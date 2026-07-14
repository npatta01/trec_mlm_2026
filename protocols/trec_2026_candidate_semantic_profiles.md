# TREC 2026 candidate-aware semantic profiles preregistration

Status: advisor approved and locked before implementation or effectiveness scoring.

Base revision: `08cb6c6cb2110ac94d82900525fe97ae78d69bb1`, the fetched
`origin/main` revision that merges PR #16.

This protocol implements GitHub issue #12. It tests candidate-specific,
multi-vector semantic profiles alone and in fixed reciprocal-rank fusion with the
frozen dense query-kNN reference. It also runs one fixed diagnostic fusion between
the dense reference and contextual E5.

The frozen dense reference is 0.3599 nDCG@10, 0.4041 MRR, and 116 zero-nDCG
queries. Contextual E5 is the stable control at 0.3347 nDCG@10, 0.3866 MRR, and
120 zeros. E5 rescues 40 dense-zero queries, dense rescues 44 E5-zero queries,
and 76 queries are zero for both. These values motivate the experiment but do not
select any representation, fusion weight, or per-query output in this protocol.

The advisor rejected nested or post-hoc fusion selection. Even selection among
previously frozen rankings would use qrels to determine the final ranking. This
protocol therefore names one fixed promotion candidate and retains all other
fusion rankings only as preregistered diagnostics or ablations.

## Scope and non-goals

The experiment includes:

- one fixed dense-plus-E5 Phase 0 diagnostic;
- one candidate-specific profile representation;
- profile-only ranking;
- one fixed dense-plus-profile promotion candidate;
- two fixed dense/profile weight ablations;
- one fixed, initially dormant three-way ablation;
- qrel-independent construction and post-lock evaluation;
- complete artifacts, an accessible report, a query explorer, and a decision
  receipt.

It excludes encoder sweeps, support penalties, learned rankers, LLM judges,
response-quality labels, global shared clusters, supervised fusion, and tuning of
RRF constants or weights. Completed Qwen, Jina, pointwise, generated-listwise,
GroupRank, and FirstMistral work is not continued. Exploratory programs and scratch
artifacts cannot enter the final pull request.

## Frozen evidence boundary

Profile construction may consume only:

- target query IDs and text;
- organizer-provided discovery query IDs, query text, candidate IDs, and responses;
- the tracked 150-candidate manifest;
- the pinned encoder and tokenizer;
- the shared cleaned candidate-label vocabulary;
- the already-frozen dense and contextual-E5 run artifacts for fusion.

The dense artifact deserves a precise qualification. Its held-out development
predictions were produced by leakage-safe nested evaluation, but its neighbor
evidence and configuration selection use fold-local qrels. The new profile
construction and fixed fusion are qrel-free conditional on consuming that already
frozen dense artifact; the dense signal itself is not end-to-end qrel-free.

Construction and evaluation use separate public entry points and configuration
types. No construction configuration, command, or public function accepts a qrel
path, qrel object, relevance grade, evaluation metric, or effectiveness threshold.
Construction must succeed when the qrel file is absent or unreadable. It writes an
atomic, immutable manifest containing canonical hashes for every construction
artifact. Evaluation validates that manifest read-only and writes to a separate
directory; it cannot replace or rewrite a construction file.

Qrels may be opened only after every ranking listed in the frozen ranking matrix has
been written, validated as 322 queries by 150 unique candidates, repeated, and
locked by hash. Qrel mutation must leave unchanged every input inventory,
formatted input, embedding, cluster membership, centroid, annotation, profile
score, winning-centroid ID, fusion score, prediction table, ranking, TREC run, and
construction-manifest hash.

## Input corpus, filtering, and exact deduplication

The discovery corpus is `llm_discovery_data.json`, SHA-256
`b33c28aa2571cdf5b2867dcc00f0dc55e22198b90022dd23f748a03395301b25`.
It contains 7,759 discovery questions, 150 candidates, and 1,163,850
question/candidate response records.

Reuse the existing `basic` response filter exactly:

1. remove null responses;
2. remove Unicode-whitespace-blank responses;
3. normalize with Unicode NFKC, casefold, and whitespace collapse;
4. remove a response only when the normalized text is exactly `i don't know`.

Do not apply the aggressive length or prefix filter. For each candidate, deduplicate
the remaining records by exact raw Python string equality on
`(discovery_question, response)`. There is no Unicode normalization, case folding,
whitespace collapse, or semantic deduplication at this step. Sort usable records by
`(source_query_id, sha256(raw_question, raw_response))`; assign the resulting
stable record ID before encoding.

The qrel-free structural audit at protocol time found 182,800 basic-filtered
records, 36 candidate-local exact duplicates, and 182,764 usable distinct records.
Every candidate has at least 36 usable records, so all official candidates request
10 clusters. Preserve raw, excluded-by-reason, duplicate, usable, and truncated
counts per candidate. Tests must still cover zero and one-to-nine usable records.

## Frozen profile encoder contract

Use the retained contextual E5 checkpoint and runtime:

| Item | Frozen value |
| --- | --- |
| Model | `intfloat/e5-mistral-7b-instruct` |
| Revision | `d62cc4417be0e3289c03dd459d1ef2d137e2554e` |
| License | MIT |
| Container | `vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28` |
| Backend | Transformers on ROCm, not the vLLM server |
| Torch runtime | `2.11.0+gitd0c8b1f`, HIP `7.2.53211` |
| Transformers | `5.12.1` |
| Device requirement | available Torch HIP device whose architecture starts with `gfx` |
| Inference dtype | BF16 |
| Pooling | normalized last non-padding token |
| Persisted dtype | C-contiguous normalized float32 |
| Embedding dimension | 4,096 |
| Seed | 2026 |
| Batch size | 8, with no adaptive fallback |
| Maximum length | 512 tokens including the terminal EOS |
| Padding | left |

The target task instruction is exactly:

```text
Given a user question, retrieve historical question-and-response records from LLMs that are likely to answer it correctly.
```

The target query string is exactly:

```text
Instruct: {instruction}
Query: {target_question.strip()}
```

The discovery record string is exactly:

```text
Question: {discovery_question.strip()}
Response: {response.strip()}
```

Tokenize with `add_special_tokens=True` and right truncation. First tokenize without
truncation to record the original length. For the encoded form, tokenize to at most
511 tokens, remove every trailing EOS token ID, right-truncate again if required to
leave at most 511 tokens, and append exactly one EOS token ID 2 (`</s>`). Left-pad
batches. This rule deliberately preserves the beginning of the question-response
record. If the template and question consume the entire budget, the response may be
fully truncated; record this separately from ordinary partial truncation.

Run BF16 inference, pool and normalize in Torch, cast to float32, copy into a
C-contiguous NumPy array, and normalize again in float32 before persistence. Reject
non-finite or zero embeddings. Save formatted-input and embedding hashes together
with model, tokenizer, software, device, memory, token-length, truncation, batch, and
elapsed-time metadata.

Before the full run, execute a qrel-free batch-8 smoke over deterministically selected
short, median-length, and over-512-token records. It must confirm the pinned snapshot,
`gfx1151` ROCm execution, memory headroom, finite normalized outputs, and repeated
scores within `1e-5`. Failure stops the experiment and requires an explicit protocol
amendment; it cannot trigger a silent batch-size or model fallback.

## Deterministic candidate-specific clustering

Run clustering separately for each candidate. Records from another candidate never
participate in a candidate's fit.

The clustering environment is Python 3.13.9, NumPy 2.5.1, SciPy 1.18.0, and
scikit-learn 1.9.0. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and the equivalent thread-pool limits to one before
importing the numerical stack. Record and validate these values.

For `n` usable distinct records, request `K = min(10, n)`. Fit scikit-learn
`KMeans` to the stable C-contiguous float32 embedding matrix with:

- `init="k-means++"`;
- `random_state=2026`;
- `n_init=10`;
- `algorithm="lloyd"`;
- `max_iter=300`;
- `tol=1e-4`.

Assignments use scikit-learn's declared Euclidean K-means behavior. Persist the
installed package versions and the convergence result. After fitting:

1. discard a fitted center with no assigned records;
2. group fitted centers whose C-contiguous float32 centroid bytes are identical;
3. merge each such group's memberships;
4. recompute each occupied centroid as the float64 mean of its member float32
   embeddings, cast it to float32, and L2-normalize it;
5. order occupied clusters by the lexicographically smallest stable member record ID;
6. assign canonical cluster IDs `0..K_actual-1` in that order.

Identical input embeddings may therefore collapse to one occupied centroid. The
profile contains up to 10 centroids, not necessarily exactly 10. Hash the ordered
membership list for every canonical centroid and the complete candidate membership.

For centroid `j` with normalized vector `c_j` and normalized member vectors `x_i`,
define dispersion as:

```text
mean_i (1 - cosine(x_i, c_j))
```

The medoid is the member with maximum cosine to the centroid, with stable record ID
breaking exact ties. Representative questions are the first three distinct question
strings under the same ordering. Preserve support, dispersion, medoid, representatives,
iteration count, convergence, and membership hash.

## Profile scoring and unavailable candidates

For target query `q` and candidate `c`:

```text
profile_score(q, c) = max_j cosine(query_embedding(q), centroid(c, j))
```

There is no support penalty, calibration, clipping, prior, or label-derived adjustment.
Save the canonical winning centroid ID; a centroid tie uses the smaller canonical ID.
Rank by descending float32 profile score and then candidate ID.

A candidate with no usable record has internal score negative infinity. In the
human-readable profile prediction table it receives the finite sentinel `-2.0` and is
placed after every available profile, ordered by candidate ID. Its profile signal
contributes exactly zero to every fusion rather than receiving an arbitrary tail-rank
RRF contribution. The official structural audit predicts full profile coverage, but
the behavior remains part of the executable contract.

## Explanation-only propagated labels

Cluster labels cannot influence filtering, deduplication, embeddings, cluster fitting,
centroids, profile scores, winning centroids, fusion scores, or rankings.

Reuse the retained contextual-E5 label-document embeddings and its exact label query
contract. Encode each of the 7,759 unique supporting discovery questions with:

```text
Instruct: Given a user question, retrieve topic labels describing the LLM experts most likely to answer it correctly.
Query: {discovery_question.strip()}
```

For a cluster and label, take the arithmetic mean of the supporting question-to-label
cosine similarities. Sort labels by descending mean similarity and candidate ID. Let
`s1` and `s2` be the first two scores and `margin = s1 - s2`:

- if `s1 >= 0.60` and `margin >= 0.02`, store one primary label;
- if `s1 >= 0.60` and `margin < 0.02`, store primary and secondary labels;
- otherwise store `mixed_or_unclear` with no forced primary.

Save `s1`, `s2`, margin, support, and representative questions. A mutation test must
replace every annotation and prove all non-annotation construction hashes remain
unchanged.

## Frozen ranking and fusion matrix

RRF uses one-based complete ranks and `k=60`. For signal `s` with weight `w_s`, its
contribution is `w_s / (60 + rank_s)`. Sum contributions and break exact fused-score
ties by candidate ID. An unavailable profile contributes zero as specified above.

Before qrels are opened, construct and lock exactly these complete rankings:

| Ranking | Weights | Role |
| --- | --- | --- |
| Profile only | profile score | Independent evidence test |
| Dense + E5 | `1:1` | Phase 0 diagnostic |
| Dense + profile | `1:1` | Sole promotion candidate |
| Dense + profile | `2:1` | Dense-heavy descriptive ablation |
| Dense + profile | `1:2` | Profile-heavy descriptive ablation |
| Dense + profile + E5 | `1:1:1` | Dormant conditional ablation |

The three-way ranking is constructed unconditionally before Phase 0 is evaluated. It
is reported as an effectiveness ablation only if Phase 0 passes. No fixed ablation can
replace the sole promotion candidate, and there is no nested, global, per-fold, or
per-query weight selection.

Repeat Phase 0 fusion twice in fresh processes and require byte-identical canonical
rank tables. Repeat the complete profile construction and all dependent rankings in a
fresh process. Require identical input inventories, formatted-input hashes, cluster
memberships, centroid hashes, annotation hashes, winning-centroid IDs, and canonical
rank tables; require maximum absolute corresponding profile and fusion score difference
at most `1e-5`. Timestamp-bearing metadata is excluded from canonical comparison
hashes but retained in runtime receipts.

## Artifact contract and validation

Use versioned schemas and atomic directory writes. The construction manifest hashes
every file and also contains one canonical aggregate manifest hash.

Per-record and per-candidate artifacts include:

- raw, excluded, duplicate, usable, partial-truncation, and full-response-truncation
  counts;
- stable record IDs, raw-membership hashes, formatted text hashes, and embeddings;
- encoder/tokenizer identity, exact revisions, license, formatting, tokenization,
  pooling, dtype, dimensions, batch, seed, runtime, and source hashes;
- deterministic memberships and normalized centroids;
- centroid support, dispersion, medoid, representatives, and membership hashes;
- primary/secondary annotation scores, confidence status, and the explicit
  `affects_ranking=false` invariant.

Per-run artifacts include:

- target query embeddings;
- all 48,300 query-candidate profile scores and winning centroid IDs;
- source run hashes, fusion scores, predictions, rankings, and complete TREC runs;
- canonical ranking hashes and repeatability receipts;
- qrel-independence and label-independence receipts;
- software, hardware, time, memory, token, truncation, cluster-support, and coverage
  metadata;
- validation that every run contains exactly 322 query IDs and all 150 unique
  candidate IDs per query.

The artifact loader recomputes and validates hashes, score/rank consistency, centroid
normalization, membership coverage, winning centroids, fusion arithmetic, prediction
ranks, and TREC completeness. It rejects missing, extra, reordered, mutated, non-finite,
or internally inconsistent data.

## Post-lock evaluation

Only after construction validation and ranking lock may the evaluator load qrels.
Positive means grade strictly greater than zero. Report nDCG@10, MRR, and zero-nDCG
count for every reportable ranking.

For comparisons against frozen dense, compute per-query candidate-minus-dense metric
differences. Win, tie, and loss use absolute equality tolerance `1e-12`. Paired
intervals are percentile intervals over 10,000 bootstrap samples of the 322 query
differences, using `numpy.random.Generator(numpy.random.PCG64(2026))`; report the
2.5th and 97.5th percentiles and the full-precision mean difference.

For zero transitions report exactly:

- zero in both;
- dense zero, candidate nonzero (rescued);
- dense nonzero, candidate zero (harmed);
- nonzero in both.

Report qrel-density slices of at most 5, 6 through 20, and more than 20 positive qrels.
The frozen question-type classifier strips leading whitespace and punctuation, takes
the first casefolded ASCII word, maps `what`, `who`, `when`, `where`, `which`, `how`,
and `why` to their own slices, and maps everything else to `other`. Slices are
descriptive and cannot select a run.

## Phase 0 decision

Evaluate the already-frozen equal-weight dense-plus-E5 RRF ranking. Phase 0 passes only
if all full-precision conditions hold:

- mean nDCG@10 minus dense is greater than zero;
- mean MRR minus dense is at least `-0.005`;
- zero-nDCG count is less than 116.

Always report its paired intervals, win/tie/loss counts, and zero transitions. If it
passes, report the already-frozen three-way ranking as a conditional ablation. If it
fails, retain E5 only as the stable qrel-free/cold-start control and do not tune or
replace the Phase 0 fusion.

## Profile decision receipt

Apply the promotion gate only to equal-weight dense-plus-profile RRF. Promote it only
if all conditions hold at full precision:

- the paired nDCG@10 interval lower bound versus dense is greater than zero;
- mean MRR minus dense is at least `-0.005`;
- zero-nDCG count is less than 116;
- mean nDCG@10 difference is greater than zero on queries with at most 20 positive
  qrels;
- qrel-independence, label-independence, leakage, deterministic clustering,
  fresh-process repeatability, inventory, and artifact validation all pass.

If promotion fails but all integrity checks pass, classify the method as diagnostic
when at least one of these holds: overall nDCG@10 point difference is positive, the
zero count is lower than 116, or at least one dense-zero query is rescued. Otherwise
classify it as reject. Any integrity failure produces `reject-invalid`, records the
failed invariant, and prohibits an effectiveness claim.

The receipt records the Phase 0 decision, each promotion-gate value, final
`promote`/`diagnostic`/`reject` outcome, and exact reasons. A failed method cannot enter
the production ranking path.

## Tests required before the official run

- exact query and record formatting;
- exact EOS, padding, right-truncation, question-only budget, and batch behavior;
- filter reasons and raw exact deduplication;
- stable record IDs and candidate-local input isolation;
- zero-through-nine-record rules and official 10-cluster rule;
- deterministic membership, empty/duplicate centroid canonicalization, and identical
  embedding behavior;
- centroid normalization, dispersion, medoid, and representative tie handling;
- maximum-cosine scoring, winning-centroid ties, and candidate-ID rank ties;
- unavailable profile sentinel, tail order, and zero fusion contribution;
- question-to-label aggregation, ambiguity thresholds, and annotation mutation;
- fixed RRF arithmetic for all five fusion rankings;
- public construction APIs and configs contain no qrel boundary;
- construction succeeds with qrels absent;
- qrel mutation leaves every construction hash unchanged;
- 322 complete rankings with exactly 150 unique candidates;
- valid complete TREC output and immutable manifest validation;
- fresh-process repeat receipts and maximum score difference;
- bootstrap, win/tie/loss, zero transition, density slice, question-type, Phase 0,
  and final decision logic.

## Deliverables

- this locked protocol and a config-driven implementation;
- a reproducible qrel-free construction command and separate evaluation command;
- focused unit, integration, leakage, and repeatability tests;
- complete local artifact directories and compact tracked receipts;
- Markdown and standalone accessible HTML comparison reports;
- a query explorer showing the winning profile centroid for every query-candidate pair,
  with support, dispersion, labels, and representative questions;
- Phase 0 and final promote/diagnostic/reject receipts;
- a final pull request containing only retained implementation, tests, protocol,
  reports, receipts, and necessary documentation.

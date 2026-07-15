# TREC 2026 local-domain-prior dense aggregation preregistration

Status: advisor-approved and locked before implementation or new effectiveness
scoring.

Base revision: `08cb6c6cb2110ac94d82900525fe97ae78d69bb1`, the fetched
`origin/main` revision that merges PR #16.

Advisor review date: 2026-07-14. Final verdict: approve with no remaining
blocker after the fold, score, control, hashing, runtime-feasibility,
repeatability, and claim-scope revisions recorded below. Independent protocol
consistency verdict: lockable with no remaining blocker.

This protocol implements GitHub issue #13. It tests whether the frozen MPNet
query representation can rank candidates more robustly by separating three
sources of evidence:

1. local grades transferred from the nearest training queries;
2. domain grades averaged inside a semantic training-query cluster; and
3. the fold-local global candidate prior.

The experiment changes aggregation only. It does not change the encoder, query
text, candidate universe, qrels, evaluation metrics, or frozen reference runs.

## Motivation and frozen references

The strongest completed development result is the leakage-safe nested dense
query-kNN reference:

| System | nDCG@10 | MRR | Role |
| --- | ---: | ---: | --- |
| Frozen nested dense | 0.35985474409504775 | 0.40406330811168817 | Primary comparison |
| Fixed MPNet at `k=50` | 0.36045807293576454 | 0.4039893654911617 | Exact regression control |
| Fixed TF-IDF at `k=10` | 0.2176542461618029 | 0.2776956693601721 | Lexical control |
| Fold-local global prior | 0.1385241823188521 | 0.20665310801034087 | Query-blind floor |

The nested dense system selected MPNet with `k=50` in 320 of 322 outer
folds and `k=20` in the other two. Its remaining 116 zero-nDCG queries
motivate testing a broader domain estimate and explicit shrinkage. The fixed
MPNet `k=50` value is a full-development deployment result and cannot
substitute for the leakage-controlled nested reference.

These already observed values motivate the experiment and set its controls.
They cannot select a new configuration, change a threshold, or replace the
primary comparison after this protocol is locked.

## Scope and non-goals

The experiment includes:

- the frozen MPNet query embeddings and cosine matrix;
- a 16-configuration local-domain-prior grid;
- exact nested leave-one-query-out configuration selection;
- train-only semantic query clustering;
- complete local, domain, prior, weighted-contribution, and final-score
  diagnostics;
- fixed controls and non-promotable component ablations;
- a qrel-free runtime-feasibility gate before the official run;
- repeated CPU execution, artifact validation, post-run evaluation, and one
  decision receipt.

It excludes:

- discovery responses or metadata;
- response or candidate semantic profiles;
- LLM judges, weak labels, generated answers, or external search;
- encoder, prompt, pooling, truncation, or embedding sweeps;
- learned rankers, learned fusion, adaptive support penalties, or per-query
  routing between systems;
- tuning outside the declared grid;
- using MRR, zero counts, slices, or outer-fold results to select a
  configuration;
- any GPU workload.

The run remains external automatic under the track taxonomy because the frozen
MPNet encoder contains outside training data. No manual change is allowed after
the official test queries are seen.

## Frozen inputs and representation

Use exactly:

- development queries: `data/dev/llm_dev_data.tsv`, SHA-256
  `862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c`;
- development qrels: `data/dev/llm_dev_qrels.txt`, SHA-256
  `3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35`;
- canonical query inventory `Q = tuple(sorted(data.queries))`, containing 322
  IDs under Python string ordering, exactly as in the frozen dense experiment;
- canonical candidate inventory `M = tuple(sorted(all qrel candidate IDs))`,
  containing 150 IDs, exactly as returned by `validate_complete_qrels`, and
  the complete 48,300-grade matrix in `Q`-by-`M` order;
- linear qrel grades `0`, `1`, and `2`;
- positive relevance defined as grade strictly greater than zero.

Reuse the frozen `sentence-transformers/all-mpnet-base-v2` representation:

| Field | Frozen value |
| --- | --- |
| Model revision | `e8c3b32edf5434bc2275fc9bab85f82640a19130` |
| License | Apache-2.0 |
| Query prefix | empty |
| Pooling | mean |
| Dimension | 768 |
| Persisted dtype | normalized float32 |
| Query-input SHA-256 | `98faa04f979e428933c3d8e67dac43cf4dded112bd2c16969b3482a1d52886d9` |
| Embedding SHA-256 | `6d197055eb62ab9809ae321b0f60072e20f17524861e6a989b49cd35c389055b` |
| Similarity SHA-256 | `80c69a8dca29cd5e9a4de08dc4e44553fabf813428dd4956f7ecbeb2852968ed` |

### Canonical hashing contract

SHA-256 values for named source, data, qrel, report, TSV, JSON, and TREC files
hash their literal file bytes. The frozen query-input hash instead retains the
legacy `_query_input_sha256` contract: for every query in `Q`, append the UTF-8
query ID, one NUL byte, the UTF-8 prefixed query text, and one NUL byte. The
frozen embedding and similarity hashes retain the legacy `array_sha256`
contract from `trec_mlm/dense_query_knn.py`: hash the ASCII Python shape, the
ASCII NumPy dtype, and C-contiguous array bytes with no separator. They are
logical array hashes, not raw `.npy` file hashes.

Every new logical hash in this experiment uses one framed contract. Define
`canonical_json(value)` as UTF-8 bytes from
`json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
separators=(",", ":"))`. Define `framed_sha256(tag, *payloads)` by starting
with `b"trec-mlm-dense-aggregation-hash-v1\0"`; for the ASCII tag and then each
byte payload, append its byte length as unsigned little-endian 64-bit
`struct.pack("<Q", len(payload))`, followed by the payload; SHA-256 the complete
stream.

For a logical array hash, first convert to the declared little-endian dtype and
C-contiguous layout. Call `framed_sha256` with, in order: the semantic array
tag, ASCII `dtype.str`, canonical JSON shape, canonical JSON row inventory,
canonical JSON column inventory, and C-order bytes. Use an empty JSON list for
an axis without semantic IDs. The required semantic tags are
`canonical-centers`, `inner-validation-score`, `inner-validation-rank`,
`full-development-score`, and `full-development-rank`.

For each outer query `o` and nominal configuration, the inner score artifact is
a `<f8` C-contiguous 321-by-150 matrix with rows `Q` excluding `o`, in retained
canonical order, and columns `M`; hash it with tag `inner-validation-score`.
Its ranking artifact is the same shape and inventories, stores each candidate's
1-based rank as `<u2`, and uses tag `inner-validation-rank`. Configuration and
outer IDs belong in the receipt that names each hash, not in the logical-array
payload, so exact duplicates retain identical hashes.

Hash an ordered support inventory with tag `support-query-ids` and canonical
JSON ID list. Hash cluster membership with tag `canonical-membership` and
canonical JSON `[[cluster_id, [ordered_member_ids]], ...]` in cluster-ID order.
Hash the clustering contract with tag `clustering-contract` and one canonical
JSON map with exactly these top-level keys:

- `schema`: `trec-mlm-dense-aggregation-clustering-v1`;
- `protocol_file_sha256`: raw hash of this locked protocol;
- `implementation_files`: sorted `[relative_posix_path, raw_sha256]` rows for
  every experiment, clustering, cache, and entry-point Python file;
- `embedding_sha256` and `similarity_sha256`: the frozen logical hashes above;
- `versions`: exact `python`, `numpy`, `scipy`, and `scikit_learn` strings;
- `threads`: the six named environment variables, integer
  `threadpool_limit=1`, and integer `process_count=1`;
- `kmeans`: exact `init`, `random_state`, `n_init`, `algorithm`, `max_iter`, and
  `tol` values from the deterministic clustering section;
- `center`: `dtype="<f4"`, `layout="C"`, and
  `canonical_order="minimum_member_query_id_lexical"`;
- `target_distance`: `dtype="<f8"`,
  `reduction="numpy.sum(delta[z] * delta[z], dtype=numpy.float64)"`, and
  `tie="minimum_canonical_cluster_id"`;
- `inertia`: `dtype="<f8"`,
  `cluster_reduction="numpy.sum(member_delta * member_delta,
  dtype=numpy.float64)"`, `total_reduction="math.fsum(cluster_inertia)"`,
  numeric `rtol=1e-5`, and numeric `atol=1e-8`;
- `convergence`: integer accepted `n_iter_min=1`, integer
  `n_iter_max=299`, and `warning_action="reject"`.

Hash an aggregate artifact inventory with tag `artifact-inventory` and
canonical JSON rows `[relative_posix_path, byte_size, raw_file_sha256]` sorted
by relative path. `manifest.json` is the sole excluded file; symlinks are
forbidden, and every other regular file recursively below the artifact root,
including every receipt, must appear exactly once. Write `manifest.json` last
with the schema, those inventory rows, and the framed aggregate digest. No
other file is excluded, and no other concatenation or serialization may be
called one of these hashes.

The authoritative result snapshot is
`reports/trec_2026_dense_query_knn_results.json`, SHA-256
`3ca4252cb37f12e9272561422fd2a45d0dcbf741770d0f544efda8acf4ddf902`.
The authoritative per-query dense-reference artifact is
`reports/trec_2026_dense_query_knn_per_query.tsv`, SHA-256
`ae4a0cf8bc7eb0642e1c0d315cafa3a56cf60b3d6bfb6ab63e80fb2d0de56437`.
Parse its decimal nDCG@10 and reciprocal-rank fields as float64, in its stored
query order, and require that ordered ID column to equal `Q`. Use it for paired
comparisons, zero transitions, and slices. Its values are deliberately retained
at their published precision; the aggregate values in the result snapshot
remain the authoritative headline references.

The implementation must validate the query inventory, model revision,
embedding shape, dtype, row norms, input hash, embedding hash, and cosine hash
before using the arrays. It may load the original frozen artifact or reproduce
the arrays byte-for-byte under its pinned environment. It may not re-encode
queries with a different runtime or checkpoint.

Frozen embeddings are qrel-free and may be cached globally. Clusters are
corpus-fitted and therefore must be rebuilt within the fold boundaries below.

### Pre-effectiveness legacy-control lock

No tracked full-precision per-candidate artifact currently exists for the fixed
MPNet `k=50` result. After the qrel-free feasibility gate passes, but before any
experimental configuration is constructed or scored for effectiveness, run a
dedicated legacy-control export with these constraints:

1. The command may load the development qrels but cannot import or call the new
   local-domain-prior implementation.
2. It must call the untouched
   `trec_mlm.dense_query_knn.rank_target_from_similarity` for every outer
   leave-one-query-out prediction with `k=50`. The authoritative source file is
   `trec_mlm/dense_query_knn.py`, SHA-256
   `90f8db3abbb2887d8513fb881c631a3843319c29da02e935e1ecdb6c3c85e248`.
3. Write `fixed_mpnet_k50_control.tsv` in canonical query-ID then candidate-ID
   order. Store query ID, candidate ID, `float.hex()` raw score,
   `float.hex()` fold prior, and integer final rank so the float64 values are
   lossless.
4. Run the export twice in fresh processes and require byte-identical files.
5. Re-evaluate the exported rankings with the existing metric functions and
   require exact equality to nDCG@10 `0.36045807293576454` and MRR
   `0.4039893654911617` from the pinned result snapshot.
6. Record the file SHA-256, row count, source hashes, and both validation runs
   in `fixed_mpnet_k50_control.lock.json`. Freeze that score/ranking hash in the
   official experiment configuration before any of the 16 experimental
   configurations can emit a metric.

A mismatch or a change to the locked artifact is an integrity failure. The
lock step records already published control behavior; it cannot select or
change an experimental configuration.

## Mathematical definition

Let:

- `Q` be the full ordered query inventory;
- `A` be the training support set for one prediction;
- `t` be the target query;
- `m` be a candidate model;
- `x_j` be query `j`'s normalized MPNet embedding;
- `y[j,m]` be query `j`'s qrel grade for candidate `m`;
- `sim(t,j) = x_t dot x_j`.

All grade aggregation and final-score arithmetic use float64. Embeddings and
the validated cosine matrix remain float32. Every saved weighted contribution
and final score is float64.

### Global prior

For every candidate:

```text
P[m] = math.fsum(float(y[j,m]) for j in A_in_query_ID_order) / len(A)
```

The official matrix is complete, so the denominator is exactly `len(A)` for
every candidate.

### Local evidence

Retain only support queries with `sim(t,j) > 0`. Sort them by:

1. cosine similarity descending;
2. query ID ascending.

Take the first `k` queries. With selected set `N_k`:

```text
mass = math.fsum(sim(t,j) for j in N_k)
L[m] = math.fsum(sim(t,j) * y[j,m] for j in N_k) / mass
```

If no positive query exists or `mass <= 0`, set `L[m] = P[m]` for every
candidate. This is exactly the established dense fallback. Save neighbor IDs,
similarities, realized neighbor count, and mass.

The experimental local signal deliberately uses the per-candidate `math.fsum`
contract above. It is not required to be bit-identical to the legacy NumPy
matrix multiplication. The fixed MPNet `k=50` control is generated only by the
untouched legacy routine in the pre-effectiveness lock step, so “exact control”
refers to that legacy path, not to substituting this experimental arithmetic.

### Domain evidence

Fit the frozen K-means procedure below to the embeddings of `A`, in query-ID
order, using cluster count `C`. Assign target `t` to the nearest fitted
centroid by squared Euclidean distance. An exact distance tie uses the smaller
canonical cluster ID.

Let `A_z` be the support queries assigned to the target's cluster. For every
candidate:

```text
D[m] = (
    math.fsum(float(y[j,m]) for j in A_z_in_query_ID_order)
    / len(A_z)
)
```

Every official fitted cluster must be occupied, so `A_z` must be nonempty.
An empty selected cluster, a non-finite value, or fewer than `C` distinct
input embeddings is an integrity failure, not a fallback to the prior.

### Final score

For local weight `lambda` and fixed prior shrinkage `beta`:

```text
local_contribution  = (1 - beta) * lambda       * L[m]
domain_contribution = (1 - beta) * (1 - lambda) * D[m]
prior_contribution  = beta * P[m]
F[m] = math.fsum((
    local_contribution,
    domain_contribution,
    prior_contribution,
))
```

This is fixed shrinkage, not support-adaptive smoothing. All three raw signals
and all three weighted contributions must be saved.

Rank candidates by:

1. final score descending;
2. fold-local global prior descending;
3. candidate ID ascending.

At `beta=0`, the prior contributes zero to the score but retains its
established role as the exact-score tie-break. No epsilon or rounded comparison
is allowed.

Because grades lie in `[0,2]` and the final score is a convex combination,
every finite `L`, `D`, `P`, and `F` must also lie in `[0,2]`.

## Deterministic clustering contract

The clustering environment is:

- Python 3.13.9;
- NumPy 2.5.1;
- SciPy 1.18.0;
- scikit-learn 1.9.0.

Before importing the numerical stack, set `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`. Wrap fitting in
`threadpoolctl.threadpool_limits(limits=1)`, use one process for each
authoritative run, and record `threadpoolctl.threadpool_info()` plus every
environment setting.

For each fold-specific support set:

1. sort normalized float32 MPNet rows by query ID;
2. require finite, nonzero rows, at least `C` rows, and at least `C` distinct
   float32 row byte strings;
3. call scikit-learn `KMeans` with:
   - `init="k-means++"`;
   - `random_state=2026`;
   - `n_init=10`;
   - `algorithm="lloyd"`;
   - `max_iter=300`;
   - `tol=1e-4`;
4. reject convergence warnings, empty clusters, non-finite centers, fewer than
   `C` occupied clusters, or a fit-level `n_iter_` outside `1..299`; reaching
   `max_iter=300` is a failure even if scikit-learn emits no warning;
5. order fitted clusters by the lexicographically smallest member query ID;
6. assign canonical cluster IDs `0..C-1` in that order;
7. preserve each canonical cluster's ordered membership, support, center,
   recomputed inertia contribution, and membership hash; preserve fit-level
   `n_iter_`, scikit-learn `inertia_`, and recomputed total inertia once per
   fit, not once per cluster.

Reorder `cluster_centers_` by the canonical-label mapping and store the result
as a C-contiguous `<f4` array. Hash it with the `canonical-centers` logical
array contract, canonical cluster IDs `0..C-1` as its row inventory, and an
empty column inventory. For target assignment, cast that array and the
normalized target row to C-contiguous float64, compute

```text
delta = canonical_centers_float64 - target_float64
distance[z] = numpy.sum(delta[z] * delta[z], dtype=numpy.float64)
```

with NumPy 2.5.1, then choose the minimum in canonical-cluster order. Exact
float64 equality therefore selects the smaller canonical ID.

For every canonical cluster `z`, compute its diagnostic inertia as:

```text
member_delta = members_float64 - center_z_float64
cluster_inertia[z] = numpy.sum(
    member_delta * member_delta,
    dtype=numpy.float64,
)
recomputed_total_inertia = math.fsum(cluster_inertia)
```

The member matrix is query-ID ordered and C-contiguous. Require every inertia
to be finite and nonnegative, and require scikit-learn `inertia_` to match the
recomputed total with `math.isclose(rel_tol=1e-5, abs_tol=1e-8)`.

Assignment of a new target is recomputed against centers in canonical order;
it must not rely on scikit-learn's arbitrary original label number.

Query input order must not change canonical memberships, target assignments,
domain scores, final scores, or rankings.

## Frozen 16-configuration grid

The grid is the Cartesian product, declared in this nesting order:

1. cluster count `C`: `20`, then `40`;
2. local neighbor count `k`: `20`, then `50`;
3. local weight `lambda`: `0.5`, then `0.75`;
4. prior shrinkage `beta`: `0.0`, then `0.1`.

There are exactly `2 * 2 * 2 * 2 = 16` nominal configurations. This order
breaks exact inner-metric ties. MRR, outer results, failure slices, runtime,
cluster support, or effective uniqueness cannot break a selection tie.

The headline effective-uniqueness diagnostic uses only the sixteen complete
322-by-150 full-development leave-one-query-out score matrices and rankings
constructed for deployment selection. For each nominal configuration, order
rows by `Q` and columns by `M`. Hash the `<f8` C-contiguous scores using the
`full-development-score` logical array contract with `Q` and `M` as its axis
inventories. Separately hash the C-contiguous `<u2` matrix containing each
candidate's 1-based rank with tag `full-development-rank` and the same
inventories. Report exact SHA-256 uniqueness counts for score matrices and
rank matrices among the sixteen configurations.
Inner-fold hashes remain additional per-outer diagnostics and do not enter the
headline count.

Nominal configurations may coincide when local, domain, and prior signals
agree. Effective uniqueness cannot remove a configuration or change selection.

The grid cannot expand or change after any new effectiveness result is opened.

## Exact nested evaluation

The independent sample is a query and its complete 150-candidate grade vector.
No pair-row split is permitted.

For every outer target `o`:

1. For every inner target `i != o`, set support to
   `A = Q minus {o,i}`.
2. Compute `P` and both local estimates from only `A`.
3. Fit both cluster counts from scratch on only `A`; neither the embedding nor
   qrel vector of `o` or `i` may influence a fit.
4. Assign `i` to each fitted clustering and compute domain estimates from
   only the corresponding support members.
5. Build and rank all 16 configurations.
6. Evaluate inner predictions using `i`'s qrels and compute macro inner
   nDCG@10 for every configuration over the other 321 queries. Record inner
   MRR, but do not use it for selection.
7. Select the largest inner macro nDCG@10. Resolve an exact metric tie by the
   declared grid order.
8. Set outer support to `A = Q minus {o}`. Refit the selected cluster count
   from scratch, recompute all selected local/domain/prior values, and produce
   the untouched outer ranking.

Fitting clusters once on `Q minus {o}` and reusing them for inner predictions
is forbidden. K-means learns query-corpus structure; allowing inner query
`i` to influence that fit would be transductive inner-validation
preprocessing and would not reproduce deployment conditions.

The official implementation must deduplicate a cluster fit by the exact
ordered support query IDs, cluster count, embedding hash, and
clustering-contract hash. Each unordered excluded pair/`C` fit serves both
directed outer/inner roles, with both excluded queries assigned before the fit
is released. Each singleton-exclusion/`C` fit is likewise shared by the nested
outer and full-development deployment calculations. Within-run disk caching
must be proven byte-equivalent to naive materialization.

The stitched primary system contains exactly one selected outer ranking for
each of the 322 queries.

## Full-development deployment selection

After the leakage-controlled exploratory nested result is frozen and reported,
compute a future-test deployment hint:

1. For every fixed configuration, produce leave-one-query-out predictions
   across all 322 development queries, fitting clusters on `Q minus {q}`.
2. Select the largest macro nDCG@10; exact ties use declared grid order.
3. For a future held-out test query, fit the selected cluster count once on all
   322 development queries and compute the fixed selected score.

The chosen deployment configuration and its same-development metric are
optimistic. They must be labeled as a future-test hint and cannot replace the
stitched nested estimate. No configuration may be changed after official test
queries are inspected.

## Qrel-free runtime-feasibility gate

Strict inner-fold plus deployment clustering requires exactly 52,003 distinct
support sets per complete run:
51,681 unordered leave-two-out sets plus 322 leave-one-out sets. Across two
cluster counts this is 103,362 pair-support fits plus 644 singleton-support
fits, or 104,006 K-means fits and 1,040,060 K-means starts at `n_init=10`.

Before the official effectiveness command may open the development qrel file:

1. keep the qrel path absent and prove the benchmark entry point cannot accept
   qrels or grades;
2. construct the pair inventory with `itertools.combinations(Q, 2)` in
   canonical `Q` order; under NumPy 2.5.1 execute exactly

   ```text
   rng = numpy.random.Generator(numpy.random.PCG64(2026))
   pair_indices = numpy.sort(
       rng.choice(51_681, size=256, replace=False, shuffle=True),
       kind="stable",
   )
   single_indices = numpy.sort(
       rng.choice(322, size=32, replace=False, shuffle=True),
       kind="stable",
   )
   ```

   and record both complete integer lists in each feasibility receipt;
3. run the exact pinned K-means contract for both `C=20` and `C=40` on
   every sampled support set, including canonicalization, held-out target
   assignment for both excluded queries of every pair and the one excluded
   query of every singleton, hashing, and the streamed receipt path used by
   the full run;
4. run the same benchmark again in a fresh process and require identical
   canonical membership and center hashes; use a separate initially empty
   fit-cache and output root, so neither repeat can read the other's fits;
5. measure each support-set/`C` operation with `time.perf_counter_ns()` from
   support materialization through receipt flush and array release; report
   mean, median, p95, maximum seconds, and convergence iterations separately
   for pair versus single exclusions and for each `C`;
6. for each fresh-process repeat, compute the exact sequential projection

   ```text
   51_681 * (mean_pair_C20 + mean_pair_C40)
   + 322 * (mean_single_C20 + mean_single_C40)
   ```

   and use the larger of the two repeat projections as the gate value;
7. invoke each benchmark process under `/usr/bin/time -v`, parse its maximum
   resident set size as KiB, convert it to bytes with a factor of 1024, and use
   the maximum across both processes as the RSS gate value.

Each repeat must report exactly 576 fits and 5,760 K-means starts; the two
benchmark processes together must report 1,152 fits and 11,520 starts.

Both the benchmark and full experiment must stream fit receipts and release
fit arrays after their last target assignment. Reuse may use disk-backed
artifacts or a bounded cache, but the implementation cannot accumulate all
104,006 fit objects in memory.

The full grid may continue only when:

- every benchmark fit passes integrity and repeatability;
- the larger projected sequential wall time is at most 24 hours per complete
  authoritative run on the authoritative host;
- maximum benchmark-process RSS is at most `16 * 1024**3` bytes;
- no other running experiment is deprived of required CPU or memory.

If any feasibility gate fails, stop before loading qrels or computing an
effectiveness score. Do not silently reduce `n_init`, change K-means, reuse
target-containing clusters, parallelize the authoritative numerical run, or
switch configurations.

The only predeclared fallback eligible for a later advisor-approved amendment
is one strict outer-LOO setting:

```text
C=40, k=50, lambda=0.5, beta=0.1
```

That fallback is not automatically activated by this protocol. It requires a
separate amendment based only on the qrel-free feasibility receipt.

## Controls and descriptive ablations

Report:

- the frozen nested dense primary reference;
- the exact fixed MPNet `k=50` regression control;
- fixed TF-IDF `k=10`;
- the fold-local global prior;
- the stitched selected local signal alone;
- the stitched selected domain signal alone;
- the stitched selected local-domain mixture before prior shrinkage.

For each outer query, freeze the `C`, `k`, and `lambda` from that fold's
selected configuration. The local-only ablation scores `L` using selected
`k`; the domain-only ablation scores `D` using selected `C`; and the
pre-shrink mixture scores `lambda * L + (1 - lambda) * D` using all three
selected values. Rank each ablation by its displayed score descending, then
the same fold-local `P` descending, then candidate ID ascending. The ablations
are descriptive and non-promotable. They cannot enter configuration selection,
replace the primary system, or support a post-hoc router.

The exact fixed MPNet `k=50` control must match the locked
`fixed_mpnet_k50_control.tsv` byte-preserving scores and rankings. It remains
separate from the 16 candidates and cannot enter their selector.

## Artifacts and repeatability

Write versioned schemas and atomic output directories. Hash every output file
except `manifest.json`; store the exact sorted inventory and its framed
aggregate digest in `manifest.json` under the sole-exclusion rule above, then
atomically publish the completed directory.

Required artifacts include:

- exact input, source, qrel, query, candidate, embedding, cosine, code, and
  environment hashes;
- the declared grid and configuration order;
- every inner configuration's nDCG@10, MRR, selection rank, selection margin,
  and effective score/ranking hash;
- selected configuration per outer query;
- for every outer query-candidate pair, raw `L`, `D`, `P`, all three
  weighted contributions, final score, prior tie value, and final rank;
- local neighbor IDs, similarities, realized count, and mass;
- target cluster ID, cluster support, canonical membership hash, center hash,
  per-cluster recomputed inertia, fit-level `n_iter_`, scikit-learn inertia,
  and recomputed total inertia;
- compact receipts for every inner and outer cluster fit, including exact
  support-inventory and contract hashes;
- complete TREC runs and human-readable prediction tables for the primary
  system and controls;
- per-query metrics, comparisons, zero transitions, density slices, headline
  full-development and supplemental inner effective-uniqueness diagnostics,
  runtime, and memory;
- feasibility, leakage, repeatability, regression, artifact-validation, and
  final decision receipts.

Run the complete authoritative experiment twice in fresh processes, each with
an isolated, initially empty fit-cache and output root. Neither run may read a
fit, receipt, membership, center, component score, ranking, selection, or
evaluation artifact from the other. Both may read the same validated frozen
inputs, embeddings, cosine matrix, qrels, code, protocol, and locked legacy
control. Each run must freshly invoke exactly 104,006 K-means fits and
1,040,060 starts; the two-run receipt must show 208,012 fits and 2,080,120
starts in total.

Timestamp and elapsed-time fields may differ, but require identical:

- inputs and inventories;
- cluster memberships, canonical IDs, assignments, and center hashes;
- inner metrics and selected configurations;
- raw and weighted component values;
- final scores, rankings, and TREC rows;
- per-query evaluation values and decision outcome.

Any unexplained numeric or ordering difference is an integrity failure. The
artifact loader must reject missing, extra, reordered, mutated, non-finite, or
internally inconsistent data.

## Post-run evaluation and diagnostics

Use the existing metric implementations:

- nDCG@10 with linear qrel grades;
- MRR at the first candidate with grade greater than zero;
- zero-nDCG query count.

For the primary system versus frozen nested dense, compute candidate-minus-
baseline per-query differences. Report:

- macro metric values and full-precision differences;
- win/tie/loss counts with absolute equality tolerance `1e-12`;
- paired percentile intervals from the existing `paired_bootstrap_ci` using
  `confidence_level=0.95`, `n_resamples=10_000`, and `seed=2026` on the 322
  query differences, with both input mappings inserted in canonical query-ID
  order;
- zero in both, rescued from dense zero, harmed to zero, and nonzero in both;
- qrel-density slices of at most 5, 6 through 20, and more than 20 positive
  candidates;
- selection frequency, margins, cluster support, realized neighbors, and
  effective configuration uniqueness.

Slices, individual-query failures, and control results are descriptive. They
cannot change the selected system, formula, grid, or gate.

## Leakage and integrity invariants

The implementation and tests must establish:

- changing outer query `o`'s complete qrel vector cannot change its selected
  configuration, support sets, cluster fits, component values, final scores,
  or ranking;
- every inner prediction excludes both outer and inner qrel vectors from local,
  domain, and prior evidence;
- every inner cluster fit excludes both outer and inner query embeddings;
- every outer fit excludes the outer query embedding and qrels;
- frozen embeddings are validated but never fitted or altered by qrels;
- candidate/model IDs affect only stable ordering and never a numeric signal;
- inner MRR and all outer metrics are absent from configuration selection;
- qrel-density and failure diagnostics run only after rankings are frozen;
- the exact regression control remains outside the candidate grid;
- every final run contains 322 query IDs and all 150 unique candidates per
  query.

## Promotion and stop rule

Apply the promotion gate only to the stitched nested local-domain-prior system
versus the frozen stitched nested dense reference. Promote only if every
full-precision condition holds:

1. the paired 95% nDCG@10 interval lower bound is greater than zero;
2. mean MRR difference is at least `-0.005`;
3. zero-nDCG query count is less than 116;
4. mean nDCG@10 difference is greater than zero among queries with at most 20
   positive qrels;
5. all input, leakage, clustering, regression, repeatability, completeness,
   and artifact checks pass.

Any integrity failure yields `reject-invalid`, identifies the failed
invariant, and prohibits an effectiveness claim. If integrity passes but any
effectiveness gate fails, retain the frozen dense method and classify this
branch as `reject`. Do not tune the grid, thresholds, clustering, score, or
fallback after observing the result.

The decision receipt records every gate at full precision, the selected
configuration frequency, deployment hint, effective unique configuration
count, and final `promote`, `reject`, or `reject-invalid` outcome.

## Required tests before the official run

- hand-calculated `L`, `D`, `P`, weighted contributions, final score,
  ranking, and tie cases;
- canonical `Q`/`M` construction and query-ID-ordered `math.fsum` reductions;
- zero-positive-similarity local fallback;
- input validation and `[0,2]` convex-score bounds;
- exact K-means arguments and pinned numerical environment;
- deterministic canonical clusters, target assignment ties, query-order
  invariance, float32 center layout/hash, float64 distance and inertia
  reductions, fit-level iteration handling, max-iteration rejection, and
  insufficient-distinct-vector rejection;
- fold membership for every local, domain, prior, and clustering operation;
- naive materialized versus optimized/cached nested equivalence;
- component-sum identity using `math.fsum`;
- declared 16-configuration order and exact metric tie selection;
- full-development headline and inner supplemental effective-uniqueness
  accounting;
- golden-vector framed hashes for center, inner-score, inner-rank,
  full-development-score, full-development-rank, support-ID, membership,
  contract, and aggregate-inventory payloads;
- outer-qrel mutation invariance;
- qrel-free feasibility entry point, sampling, projection, RSS, streaming, and
  threshold logic;
- fresh-process legacy-control lock plus exact fixed MPNet `k=50` score and
  ranking parity;
- frozen nested dense, TF-IDF, and prior control validation;
- complete 322-by-150 predictions and valid TREC output;
- isolated-empty-cache fresh-process repeatability, exact fit/start counts, and
  immutable artifact validation;
- bootstrap, win/tie/loss, zero transition, density-slice, deployment, and
  decision-gate calculations.

## Claim scope

Nested leave-one-query-out evaluation prevents a target query's qrels from
entering its own prediction or configuration selection. It does not erase the
project's repeated researcher use of these same 322 development queries,
including prior experiments and failure audits that motivated issue #13.

The result is therefore a leakage-controlled exploratory development estimate,
not untouched confirmatory evidence. Bootstrap intervals condition on this
experiment and these queries; they do not cover cross-experiment selection or
future distribution shift. Only unseen 2026 test qrels can confirm a gain.

## Deliverables

- this locked protocol and its advisor receipt;
- a qrel-free runtime-feasibility command and receipt;
- a config-driven CPU experiment command that freezes predictions before
  post-run metrics, plus separate qrel-free feasibility and read-only
  artifact-validation/report commands;
- focused unit, equivalence, leakage, repeatability, and regression tests;
- two complete local artifact directories and validation receipt;
- a compact tracked results snapshot and decision receipt;
- Markdown and standalone accessible HTML comparison reports;
- query-explorer support for local, domain, prior, neighbor, and cluster
  evidence;
- a final pull request containing only retained code, tests, protocol, reports,
  receipts, and necessary documentation.

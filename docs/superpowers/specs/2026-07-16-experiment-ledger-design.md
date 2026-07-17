# TREC 2026 Experiment Ledger Design

**Status:** Approved in chat on 2026-07-16; written specification awaiting user review.

## Objective

Create one canonical, reproducible ledger for every TREC 2026 experiment and
system result in the repository. The ledger must make it easy to answer:

- what was tested;
- which results are directly comparable;
- which system is the promoted development result;
- which per-query scores are still recoverable;
- where a metric came from;
- which caveats or missing artifacts limit a claim; and
- how systems perform on relevant-candidate and robustness slices.

The ledger replaces scattered prose as the repository's result index. Frozen
experiment receipts remain the scientific sources of truth; the ledger indexes,
validates, and summarizes them without rewriting their claims.

## Scope

This project includes:

1. a reviewed experiment/system registry;
2. deterministic extraction of aggregate and per-query metrics from tracked
   evidence;
3. relevant-candidate and positive-density slice summaries;
4. explicit evidence availability, comparability, artifact custody, and caveat
   fields;
5. generated JSON, TSV, and Markdown views;
6. validation that generated summaries agree with their source receipts;
7. README navigation to the canonical ledger; and
8. a regression fix for the dense nested helper's handling of negative grades.

This project does not:

- run a new ranking experiment;
- construct new semantic query clusters;
- claim that a candidate label is an official query topic;
- reconstruct missing response, source-query, label-sweep, or issue-18 raw
  artifacts;
- compare raw system scores across approaches;
- promote a diagnostic or restricted pilot;
- design or run response-reliability modeling; or
- publish, upload, or change access to any artifact.

Response-reliability scoring and cluster-local reliability profiles are a
separate follow-on project.

## Source hierarchy

When two sources disagree, the builder uses this order:

1. frozen machine-readable evaluation or decision receipt;
2. tracked per-query metrics or complete frozen ranking;
3. reviewed Markdown result study containing a preserved metric snapshot;
4. generated report artifact;
5. README prose.

Lower-priority sources may provide descriptions and navigation but cannot
silently override higher-priority metrics or evidence status.

The initial controlling sources are:

- `reports/trec_2026_dense_query_knn_results.json`;
- `reports/trec_2026_dense_query_knn_per_query.tsv`;
- `reports/trec_2026_response_retrieval_results.json`;
- `reports/trec_2026_response_volume_normalization_results.json`;
- `reports/trec_2026_source_query_first_results.json`;
- `reports/trec_2026_llm_profiles.json`;
- `docs/results/2026-07-13-expert-label-ranking-study.md`;
- `reports/trec_2026_candidate_semantic_profiles_results.json`;
- `reports/trec_2026_issue18_pr17head_evaluation.json`;
- `reports/trec_2026_issue18_pr17head_results.json`;
- `reports/issue_13_dense_aggregation_result.md`;
- `reports/issue_13_matched_support_preflight.json`;
- `reports/trec_2026_issue18_pr17head_artifact_custody.json`;
- `data/dev/llm_dev_data.tsv`; and
- `data/dev/llm_dev_qrels.txt`.

Every source used by the generated ledger records its repository-relative path
and SHA-256 digest.

## Architecture

### Canonical reviewed registry

Create `experiments/ledger.json` as the reviewed metadata registry. It contains
experiment and system identities, aliases, families, evidence pointers,
evaluation scope, promotion status, artifact state, and caveats. It does not
manually duplicate per-query metric vectors.

Metric pointers identify an exact source file and structured location. Markdown
snapshot entries use an exact table/section identity plus expected aggregate
values so the builder can reject accidental drift.

Each system record contains at least:

- `system_id`: unique canonical identifier;
- `source_system_id`: original identifier in the controlling receipt;
- `aliases`: documented names used elsewhere;
- `experiment_id` and `experiment_family`;
- `method_summary`;
- `evaluation_scope`;
- `evidence_status`;
- `promotion_status`;
- `query_count` and `candidate_count`;
- `aggregate_source`;
- `per_query_source`, or an explicit unavailability reason;
- `ranking_coverage` and `native_scores_available`;
- `artifact_state`;
- `qrel_usage`;
- `run_classification` when known; and
- `caveat_ids`.

### Builder and validation library

Create `trec_mlm/experiment_ledger.py` for pure loading, validation, metric,
slice, and rendering-data functions. Create
`scripts/build_experiment_ledger.py` as a thin command-line adapter.

The builder reads only tracked repository evidence and writes deterministic
outputs. It must not depend on ignored `artifacts/` directories or discovery
response JSON files.

### Generated outputs

Generate these tracked files:

1. `reports/trec_2026_experiment_ledger.json`
   - complete normalized registry, source digests, aggregate metrics, evidence
     availability, caveats, and validation summary;
2. `reports/trec_2026_experiment_systems.tsv`
   - one row per system, suitable for sorting and comparison;
3. `reports/trec_2026_experiment_per_query.tsv`
   - one row per recoverable system/query pair;
4. `reports/trec_2026_experiment_topic_slices.tsv`
   - relevant-candidate slices with support and overlap warnings;
5. `reports/trec_2026_experiment_density_slices.tsv`
   - disjoint positive-candidate-density slices; and
6. `reports/trec_2026_experiment_ledger.md`
   - concise human-readable index, headline results, evidence matrix, caveats,
     and links to controlling receipts.

The JSON is the generated machine-readable view. The reviewed
`experiments/ledger.json` registry and frozen source receipts remain canonical.

## Evidence and status vocabulary

### Evaluation scopes

- `dev_322x150`: complete development evaluation over all 322 queries and 150
  candidates;
- `restricted_12x20`: qrel-assisted 12-query, 20-candidate diagnostic;
- `restricted_6`: six-query feasibility diagnostic;
- `invalid_unscored`: no valid quality result; and
- `constructed_not_evaluated`: representation or profile exists without a
  complete ranking evaluation.

Only systems in the same evaluation scope may be placed in a shared numerical
comparison table.

### Evidence statuses

- `stored_per_query`: complete per-query metrics are present in a tracked source;
- `recomputed_from_complete_ranking`: metrics are derived from a tracked complete
  ranking and verified against its aggregate receipt;
- `aggregate_only`: aggregate metrics are tracked but per-query metrics cannot be
  recovered;
- `restricted_diagnostic`: incomplete query or candidate scope;
- `invalid`: the run failed its declared validity contract; and
- `unscored`: no effectiveness judgment was opened.

### Promotion statuses

- `promoted_development`;
- `stable_control`;
- `valid_negative`;
- `diagnostic_only`;
- `restricted_pilot`;
- `invalid`;
- `stopped_unscored`; and
- `constructed_not_evaluated`.

These statuses describe evidence and decisions, not an assumed hidden-test
ranking.

## Per-query table

For every system with recoverable per-query evidence, write one row per query
with:

- canonical and source system IDs;
- experiment family and evaluation scope;
- query ID and query text;
- nDCG@10;
- reciprocal rank;
- number of grade-1, grade-2, and positive candidates;
- relevant candidate IDs with grades;
- positive-density bucket;
- zero-nDCG flag;
- per-query metric provenance (`stored` or `recomputed`); and
- source path and source SHA-256.

Per-query records must use full-query metrics. They must not reinterpret a
system score as probability or confidence.

The first version can recover complete 322-query vectors for the dense query
reference, contextual E5, profile-only, the promoted dense/profile fusion,
profile-heavy Track A, and the issue-18 multiresolution systems. Other systems
remain `aggregate_only` until their original per-query artifacts are restored or
the experiment is rerun under a new identity.

## Topic and robustness slices

### Relevant-candidate slices

The repository has no organizer-provided semantic query-topic field. TREC's
`topic_id` is the query ID. The ledger therefore uses the explicit name
`relevant_candidate_slice`, not `official_topic`.

For candidate label `c`, define:

`Q_c = {q : qrel(q, c) > 0}`.

For each system with complete per-query metrics, report over `Q_c`:

- query support count;
- mean nDCG@10;
- mean reciprocal rank;
- zero-nDCG count and rate;
- grade-1 and grade-2 membership counts; and
- source hashes.

These slices overlap. A query contributes once to every positive candidate
label, producing 4,378 memberships from 322 queries in the current qrels. Every
row and the Markdown explanation must state that slice means cannot be summed or
averaged as if they were disjoint.

Candidate labels are relevance targets and useful analysis handles; they are not
asserted to be a complete or correct semantic taxonomy for the query text.

### Positive-density slices

Also report three disjoint buckets:

- `at_most_5_positives`;
- `6_to_20_positives`; and
- `more_than_20_positives`.

The current support counts must validate as 248, 30, and 44, covering all 322
queries exactly once. This is the primary non-overlapping robustness view.

### Future semantic clusters

The schema reserves optional `derived_slice_definition` metadata for a future
qrel-free semantic-cluster stress test. The first ledger does not construct or
score those clusters. Any future cluster slice must record the encoder revision,
clustering algorithm, seed, membership hash, support, and the warning
`derived_not_official_topic`.

## Initial experiment families

The reviewed registry covers:

1. global prior and TF-IDF query-neighbor controls;
2. dense query-neighbor encoder/k experiments;
3. candidate-name/capability embedding sweeps;
4. restricted pointwise, listwise, GroupRank, and FirstMistral pilots;
5. response BM25/RM3 retrieval;
6. response-volume normalization;
7. source-query-first response alignment;
8. candidate behavior/topic profile construction without final evaluation,
   controlled by the tracked `reports/trec_2026_llm_profiles.json` profile receipt;
9. candidate-specific semantic response profiles and fixed fusions;
10. multiresolution response-profile diagnostics; and
11. hard query-cluster/domain aggregation.

Aliases remain visible. For example, `dense_reference`, `query_knn`, and
`dense_nested` map to one canonical system only after their stored per-query
vectors and documented identities are validated. The generated ledger preserves
both the canonical name and the source-specific name.

## Program-wide caveats

The generated ledger must display these caveats prominently:

1. all effectiveness results are development evidence; the 2026 hidden test is
   not yet available;
2. repeated program-level decisions reused the same 322 development queries, so
   nested evaluation does not eliminate adaptive research overfitting;
3. restricted pilots are not comparable with 322x150 evaluations;
4. relevant-candidate slices overlap and are not official semantic topics;
5. two initial Qwen8B/E5 embedding attempts are invalid and superseded because
   their checkpoint-specific prompt and asymmetric terminal-token contracts
   were omitted; no numeric quality claim is retained for them;
6. several response, label-sweep, query-cluster, and source-query raw/per-query
   artifacts are absent from the current checkout;
7. the ignored issue-18 construction tree is not durably restorable from the
   repository;
8. pretrained encoders may make a run external under the track taxonomy;
9. response attempt, confidence, and profile similarity do not establish factual
   correctness; and
10. confidence intervals from different experiment families may use different
   frozen bootstrap implementations and should not be compared digit-for-digit.

Each system also carries method-specific caveats such as response-inventory
dependence, repeatability failure, post-hoc status, query-sensitivity stops, or
missing per-query custody.

## Validation invariants

The builder fails without writing partial outputs when any of these conditions
is violated:

- duplicate experiment or canonical system ID;
- unknown evaluation, evidence, promotion, artifact, or qrel-usage status;
- missing controlling source or changed source digest not accepted by the
  reviewed registry;
- duplicate or missing development query IDs;
- non-finite or out-of-range per-query metrics;
- per-query vector other than exactly 322 rows for `dev_322x150` stored evidence;
- recomputed macro metric differing from its receipt beyond `1e-12`;
- dense alias vectors disagreeing beyond `1e-10`;
- density buckets not covering all 322 queries exactly once;
- relevant-candidate support disagreeing with qrels;
- an aggregate-only system lacking a concrete blocker;
- a restricted result appearing in the full-development leaderboard;
- anything other than exactly one promoted development system, or a promoted
  system without full 322x150 scope, complete per-query evidence, exact scope
  counts, and no explicit diagnostic/restricted promotion blocker; or
- a generated output containing an absolute historical worktree path.

Output writes use temporary files and atomic replacement. Stable input ordering
produces byte-identical outputs; generated artifacts do not contain wall-clock
timestamps.

## Dense helper correctness fix

`exact_nested_loo_inner_scores` must use the canonical metric contract for any
finite input matrix: nonpositive judgments contribute zero gain. Add a regression
test with a negative finite grade that fails against the current implementation,
then clip observed gains consistently with `trec_mlm.metrics.ndcg_at_k` and the
equivalent response helper.

This fix does not change official experiment results because official dense runs
already reject grades outside `{0, 1, 2}`. The ledger records it as a repaired
helper-contract caveat, not as a score correction.

## Testing strategy

Implementation follows test-driven development.

Unit tests cover:

- registry vocabulary and uniqueness;
- exact source extraction;
- aggregate-only blockers;
- alias validation;
- per-query normalization;
- relevant-candidate overlap and support;
- disjoint density buckets;
- scope-safe leaderboard filtering;
- deterministic output ordering;
- atomic failure behavior; and
- negative-grade clipping in the dense helper.

Repository integration tests build all outputs from tracked evidence in a
temporary directory, verify aggregate parity, and compare the result with the
checked-in generated files. The complete test and Ruff suites run before
completion.

## Documentation and maintenance

Update `README.md` near the existing approach guide with one prominent link to
`reports/trec_2026_experiment_ledger.md`. The README should stop trying to act as
the complete current leaderboard; detailed experiment sections may remain as
method documentation.

The generated Markdown begins with:

- the promoted development result;
- the best diagnostic, clearly labeled;
- the best stable qrel-free control;
- the count of systems with complete per-query evidence;
- the count of aggregate-only systems; and
- unresolved artifact blockers.

Adding a future experiment requires a reviewed registry entry and a controlling
receipt. Per-query output is required for new full-scope experiments unless an
explicit blocker is recorded. New systems cannot silently inherit an alias,
promotion status, or caveat from an older experiment.

## Success criteria

The project is complete when:

1. all initial experiment families appear exactly once in the reviewed registry;
2. every system has an evidence and promotion status;
3. all recoverable 322-query metric vectors are exported and aggregate-verified;
4. relevant-candidate and density slices are generated with their required
   warnings and support;
5. missing per-query evidence is visible rather than inferred;
6. the dense negative-grade regression test passes under the canonical metric
   contract;
7. generated outputs reproduce byte-for-byte on a second build;
8. README points to the ledger;
9. the full test suite and Ruff pass; and
10. an independent review finds no mixed-scope comparison, unsupported claim, or
    untracked source dependency.

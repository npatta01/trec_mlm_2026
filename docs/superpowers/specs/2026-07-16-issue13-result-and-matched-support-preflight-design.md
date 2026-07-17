# Issue 13 Result and Matched-Support Preflight Design

## Objective

Replace the unmergeable Issue 13 experiment-foundation PR with a small branch
from current `main` that records the completed rejection and runs one focused,
exploratory diagnostic: does hard semantic clustering add value beyond an
unweighted nearest-neighbor estimate with exactly the same support size?

This branch does not promote a new ranking method. It preserves a trustworthy
negative result and decides whether cluster-based aggregation deserves another
experiment on untouched evidence.

## Scope

The branch contains only:

- one standalone, accessible Markdown result report;
- one compact, sanitized machine-readable result/preflight summary;
- one focused preflight script and its unit tests;
- this design and its implementation plan.

It does not carry forward the old HDF5 fit store, nested 16-configuration
selector, custom artifact framework, compatibility layers, legacy-control
exporter, or their experiment-specific test suites. Current `main` already owns
generic artifact primitives in `trec_mlm.artifact_io`.

## Scientific question

The rejected method blended similarity-weighted local MPNet evidence with a
uniform average over one hard KMeans cluster. The completed result cannot
separate two explanations:

1. semantic clusters group queries that prefer different candidates; or
2. cluster estimates are worse simply because they use small, unweighted,
   quantized support.

The matched-support preflight isolates those mechanisms.

For every development query held out as target and for cluster counts 20 and
40:

1. fit deterministic KMeans on the other 321 frozen MPNet embeddings;
2. assign the target to its nearest centroid;
3. let `s` be the occupied support of that cluster;
4. rank candidates by the uniform grade mean over those cluster members;
5. independently rank candidates by the uniform grade mean over the target's
   `s` most similar positive-cosine support queries;
6. use the fold-local global candidate prior, then candidate ID, only to break
   exact score ties.

The frozen similarity-weighted MPNet `k=50` result remains the reference. A
pure-local method is therefore represented directly rather than inferred from
a blended selector.

## Interpretation gate

This is exploratory analysis on already observed development qrels. It cannot
produce a promotable ranking result.

- If hard clusters do not beat matched-support neighbors on mean nDCG@10, stop
  cluster aggregation work. The apparent domain signal adds no demonstrated
  value beyond support selection.
- If hard clusters beat matched-support neighbors, retain clustering only as a
  hypothesis for a separately preregistered experiment on untouched evidence.
- Regardless of the comparison, neither system replaces the frozen MPNet
  baseline from this preflight.

No post-hoc weight sweep, learned gate, soft-cluster variant, or confirmation
run belongs in this branch.

## Implementation boundaries

`scripts/run_issue13_matched_support_preflight.py` owns the exploratory
computation. Small pure functions expose deterministic support selection,
uniform grade aggregation, ranking, and summary calculation so unit tests use
synthetic arrays without loading private artifacts.

The command consumes the existing frozen query inventory, complete qrels,
MPNet embeddings, and cosine matrix. It validates shapes, inventories,
finiteness, normalized embeddings, and candidate completeness before scoring.
KMeans uses `random_state=2026`, `n_init=10`, `algorithm="lloyd"`,
`max_iter=300`, and `tol=1e-4` with one numerical thread.

The output JSON contains aggregate metrics, paired deltas, support/granularity
diagnostics, input hashes, and an explicit `exploratory_not_promotable` status.
It contains no query text, per-query IDs, qrels, rankings, or raw arrays.

## Report requirements

The standalone report must explain:

- where the project started and what Issue 13 tested;
- the authoritative rejection metrics;
- the advisor-reviewed estimator gap;
- the matched-support preflight and its interpretation;
- what was intentionally not promoted;
- the single next decision.

Claims must distinguish verified facts, conditional counterfactuals, and
unknowns. In particular, zero cluster scores mean zero positive support among
cluster members, not missing judgments, and the earlier 84% decomposition is a
conditional descriptive counterfactual rather than unique causal attribution.

## Verification

Before publication:

- unit tests for deterministic support, aggregation, ties, validation, and
  sanitized output pass;
- the preflight runs twice with byte-identical JSON;
- the complete repository suite passes from a quota-safe temporary directory;
- Ruff and `git diff --check` pass;
- the report is rendered to the existing private tailnet portal and scanned for
  raw query/qrel/ranking leakage;
- the PR diff is reviewed against current `origin/main` and contains no old
  experiment implementation.


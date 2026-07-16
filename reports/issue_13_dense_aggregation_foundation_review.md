# Dense Aggregation Foundation Review

## Technical summary

This pull request promotes the qrel-free foundation for the TREC 2026 dense
aggregation experiment in Issue #13. The promoted tree is internally
consistent and well tested, but it is not the completed experiment and it does
not contain a new effectiveness result.

The promoted implementation scope, measured before adding this review file,
is large: 29,220 inserted lines across 33 files. That number is easy to
misread. Only 8,673 lines, or about 30%, are production modules and scripts.
Tests account for 16,318 lines, or about 56%. The locked protocol and
implementation plan account for 4,204 lines, or about 14%. Configuration and
small fixtures account for the remaining 25 lines.

The decision for this review is to keep one pull request. No stacked six-PR
split is planned. The test suite should remain with the implementation because
the tests are the audit evidence for leakage prevention, deterministic
numerics, artifact integrity, and qrel embargo. A later cleanup may consolidate
duplicated fixture construction, but behavioral checks should not be deleted
merely to reduce the displayed line count.

## Key findings

1. **The promoted scope is clean.** The PR branch contains 22 promoted commits
   from remote `main` commit `08cb6c6` through head commit `2ea4d3d`. Two local
   follow-up files are uncommitted and absent from the PR. Neither development
   queries nor development qrels changed.
2. **Most added lines are verification, not implementation.** Tests and
   protocol material make up roughly 70% of the additions.
3. **The test volume has a specific purpose.** The experiment relies on a
   custom streaming binary cache, exact canonical receipts, deterministic
   clustering, strict fold exclusions, and atomic artifact publication. These
   behaviors have more failure modes than an ordinary in-memory analysis.
4. **The verified subsystem is green.** A clean detached checkout at the
   promoted head passed 1,124 dense-aggregation and qrel-safe tests in 20.26
   seconds. Compilation, diff checks, locked hashes, and a basic credential
   pattern scan also passed.
5. **The result is still incomplete.** The official construction engine,
   official feasibility execution, qrel-authorized evaluation, repeat
   comparison, final decision, and final reports remain future work.

## What the line count contains

| Category | Files | Insertions | Share | What it contains |
| --- | ---: | ---: | ---: | --- |
| Tests | 15 | 16,318 | 56% | Scientific invariants, mutation tests, filesystem failures, schema validation, and qrel-access guards |
| Production | 13 | 8,673 | 30% | Scoring, clustering, streaming cache, feasibility gate, artifact publication, CLI, and legacy control |
| Protocol and plan | 2 | 4,204 | 14% | Locked scientific protocol and executable implementation plan |
| Configuration and fixtures | 3 | 25 | <1% | Frozen feasibility config and small deterministic fixtures |
| **Total** | **33** | **29,220** | **100%** | Promoted PR scope |

This table measures the promoted implementation tree through commit `2ea4d3d`
and intentionally excludes this review report. It is a scope measure, not a
claim that every line has equal complexity.

## Why there are 16,318 lines of tests

### The streaming cache is the largest test surface

The cache tests and cache regression tests contribute 5,992 lines, about 37%
of all test lines. The cache is not a dictionary wrapper. It stores a large
sequence of centers, labels, inertia values, canonical receipts, and timing
records while keeping memory bounded. It must support release and reload,
durable finalization, read-only reopening, and exact corruption detection.

The cache tests cover:

- exact pair and singleton fit inventories;
- duplicate, missing, reordered, and malformed fit records;
- binary shard offsets, shapes, dtypes, and truncation;
- independent release and reload behavior;
- finalized versus writable state transitions;
- same-filesystem no-replace moves;
- interrupted or racing finalization;
- canonical receipt and timing reconstruction.

Without this coverage, a run could finish with a plausible-looking cache that
silently associates the wrong cluster fit with a query fold.

### The legacy control must remain isolated

Legacy-control tests contribute 2,220 lines, about 14% of test lines. They
verify that the fixed MPNet `k=50` control is exported identically in two fresh
processes, that its complete artifact root is manifested, and that downstream
construction receives only validated bindings. The control's numerical result
must not influence candidate selection.

These tests also verify bounded file reads, same-byte hashing and parsing,
exact TREC reconstruction, lock/receipt cross-bindings, and compatibility with
the existing public validator.

### Scientific leakage requires explicit negative tests

Clustering and scoring tests contribute more than 2,000 lines. They prove that
an inner prediction for outer query `o` and target query `i` uses only
`Q - {o, i}`, and that a singleton prediction uses only `Q - {o}`. They also
exercise deterministic tie handling, canonical cluster labels, positive-only
neighbors, fold-local priors, float validation, and rank permutations.

These are negative guarantees: a normal happy-path test cannot prove that an
excluded query never entered a fit or score.

### Provenance and artifact tests protect the experiment boundary

Contract, feasibility, hashing, artifact, experiment-schema, and CLI tests
cover exact keys, types, hashes, row orders, path rules, source inventories,
runtime versions, bounded input sizes, filesystem substitutions, publication
collisions, and qrel-free import/preflight behavior.

Many cases are parameterized mutations of one valid artifact. The resulting
test file is long because every accepted field is also tested in rejected
forms such as missing, extra, reordered, wrong type, non-finite, aliased, or
stale.

## What these tests prove

The passing suite supports the following claims about the promoted tree:

- fold exclusions and deterministic scoring rules are enforced;
- canonical hashes bind values, shapes, inventories, and domains;
- the fit cache rejects incomplete, corrupted, and inconsistent layouts;
- feasibility repetitions must agree on locked provenance and fit evidence;
- artifact publication is no-replace and validates the published inode;
- the fixed legacy control is isolated and fully cross-bound;
- qrel-free commands do not import the official evaluation path;
- the construction configuration and receipt schemas are frozen.

## What these tests do not prove

The suite does **not** establish that:

- the new aggregation method improves effectiveness;
- the projected 104,006-fit construction completes within the approved time
  and memory limits on the authoritative host;
- the official qrel-free feasibility gate passes;
- the full construction artifact can yet be written and reloaded;
- two authoritative construction/evaluation runs are byte- or semantically
  equivalent;
- the complete repository suite passes with every optional ML dependency.

The unrestricted repository suite was attempted, but collection stopped in an
unrelated label-inference test because optional `torch` was not installed. No
large dependency was installed solely for that unrelated check.

## Scope and evidence

### Promoted implementation scope

- Base: `08cb6c6cb2110ac94d82900525fe97ae78d69bb1`
- Implementation head: `2ea4d3d48a283001afde069f8f8fc5602de1a14c`
- Commits: 22
- Changed files: 33
- Insertions/deletions: 29,220/1
- Remote relationship at publication: 0 commits behind, 22 commits ahead

The following local follow-up files were not staged or committed and are not
part of the promoted tree:

- `trec_mlm/dense_aggregation_feasibility.py`
- `tests/test_dense_aggregation_feasibility.py`

### Verification evidence

The clean-worktree test run covered the dense aggregation runtime, hashing,
contracts, scoring, clustering, cache, cache regressions, feasibility gate,
legacy control, experiment schemas, artifacts, CLI, and locked dense-query
control.

Result:

```text
1124 passed in 20.26s
```

Locked values rechecked during promotion:

- Protocol SHA-256:
  `bbe6b50fd3d20efe3213a446f6665cbe0527114f34bf5c1e90236af4f998ad49`
- Legacy dense scorer SHA-256:
  `90f8db3abbb2887d8513fb881c631a3843319c29da02e935e1ecdb6c3c85e248`

## How to review this single PR

Because GitHub's generated diff endpoint rejects diffs larger than 20,000
lines, reviewers should use the commit sequence and this report as the reading
map rather than trying to read all files alphabetically.

Recommended order:

1. Read `protocols/trec_2026_dense_aggregation.md` for the scientific rules.
2. Review `trec_mlm/dense_aggregation.py` and
   `trec_mlm/dense_aggregation_clustering.py` with their tests.
3. Review `trec_mlm/dense_aggregation_cache.py` and its two test files.
4. Review `trec_mlm/dense_aggregation_feasibility.py` and the CLI tests.
5. Review `trec_mlm/legacy_dense_control.py` and its exporter/tests.
6. Review artifact publication and construction schemas.
7. Use the 22 commits to separate initial implementations from subsequent
   hardening fixes.

## Limitations and uncertainty

The main uncertainty is not whether the promoted unit tests pass; they do. The
uncertainty is whether the remaining construction pipeline can preserve the
same guarantees at official scale. The uncommitted follow-up validator work
also contains additional filesystem-race hardening and must receive its own
promotion decision before it joins this PR.

The test suite may contain opportunities to reduce duplicated fixture setup or
combine mutation cases. That should be a behavior-preserving refactor with
coverage counts and failure assertions compared before and after. Removing
tests solely to shrink the PR would weaken the evidence without simplifying
the production system.

## Recommended next steps

1. Keep this as one draft PR, with this report as its primary review entry
   point.
2. Review by subsystem and commit order rather than raw file order.
3. Do not merge the two unfinished local validator files until their race and
   qrel-embargo findings are closed and independently verified.
4. After the foundation is accepted, finish the construction engine and strict
   loader using synthetic fixtures before any official qrel access.
5. Run the official qrel-free feasibility gate only after the pre-gate source,
   runtime, environment, and input hashes are frozen.

## Further questions

- Can the cache-test fixtures be consolidated without reducing corruption and
  failure-path coverage?
- Should the implementation plan remain in the final repository after the
  experiment is complete, or be archived with the final report?
- What review evidence is required before promoting the unfinished public
  feasibility-manifest validator?

# Issue 18 execution amendment: pre-merge diagnostic run

Status: locked before implementation on 2026-07-15.

Advisor decision: go, with the safeguards below.

## Why this amendment exists

Issue #18 originally required PR #17 to merge before implementation. The user
explicitly requested running both tracks in the current thread while PR #17 is
still an open draft. This run therefore starts from the exact PR #17 head
commit `2db10261bac607d7f8c67c0aa0c01728484a42f9`, not `main`.

The current Codex workspace is already an isolated linked worktree. Work runs
on the dedicated branch `codex/issue-18-multires-profiles`; PR #17 and its
remote branch must not be modified, merged, or force-updated.

## Permitted conclusions

- Track A reproduces the previously observed profile-heavy result on the 322
  already-seen development queries. It is not an independent validation and
  cannot promote the 0.4376 diagnostic.
- Track B constructs every new representation, score, and ranking without
  qrels. Once rankings and hashes are frozen, the existing qrels may be loaded
  once for explicitly post-hoc diagnostic evaluation.
- No result from these 322 queries can replace the issue-12 primary. A new
  locked query set remains necessary for promotion.

## Frozen Track B systems

All systems reuse the validated issue-12 inference arrays, record inventory,
formatter, normalization, candidate order, seed 2026, single-thread numerical
contract, candidate-ID tie breaking, and RRF constant 60.

- `c0_max_k10`: issue-12 control, maximum cosine at k=10.
- `a1_top2_k10`: mean of the two largest centroid similarities at k=10; use
  the only available centroid when fewer than two are occupied.
- `r1_multires_max`: independent k=5, k=10, and k=20 maximum-cosine rankings,
  fused with equal RRF.
- `x1_multires_top2`: independent k=5, k=10, and k=20 top-two rankings, fused
  with equal RRF.

For resolution k, request `min(k, usable distinct candidate records)`. K-means
uses the issue-12 deterministic contract: k-means++, seed 2026, 10
initializations, Lloyd algorithm, maximum 300 iterations, tolerance 1e-4, and
one numerical thread. Centroids are canonicalized and L2-normalized exactly as
in issue #12.

For sorted similarities `s1 >= s2 >= ...`:

```text
max_score = s1
top2_score = mean(s1, s2), or s1 when only one centroid is occupied
multires_rrf(q,c) = sum(1 / (60 + rank_k(q,c)) for k in (5, 10, 20))
```

Complete profile-only rankings and equal dense/profile RRF rankings are frozen
for C0, A1, R1, and X1. Track A uses the existing exact dense:profile weight
ratio 1:2 and RRF constant 60.

## Qrel-free viability gate

Before qrels are opened, bind the run to the source commit, dependency
versions, input and embedding hashes, configuration, commands, seeds, and all
ranking hashes. Report every frozen system, including negative results.

Each resolution and combined ranking must have exact 322 x 150 coverage,
finite scores, normalized centroids, exact identities, and deterministic
tie-breaking. Apply the issue-18 thresholds:

- at least 95% of candidates retain at least 80% of requested occupied clusters;
- largest cluster share <= 0.80;
- median/minimum normalized occupancy entropy >= 0.65/0.35;
- singleton clusters <= 25%;
- zero exact duplicate centroids;
- within-candidate near duplicates at cosine >= 0.9999 <= 1%;
- median centroid separation >= 0.005;
- at least 258 unique top-10 rankings;
- maximum top-1 candidate share <= 10%;
- top-10 candidate coverage >= 90;
- no candidate occurs in more than 50% of query top tens.

Each new ranking must differ from C0 for at least 10% of query top tens, have
mean top-10 Jaccard below 0.95, and lose no more than 10% of C0's unique top
tens or candidate coverage.

Integrity, coverage, normalization, or repeatability failure means
`reject-invalid`. Valid but insufficient distinctness means
`stop-no-incremental-value`. Otherwise the receipt says `continue`.

Run a fresh construction repeat only if the receipt says `continue`, measured
wall time is no more than four times issue-12 construction, storage is no more
than four times issue #12, and at least one new ranking has incremental value.

## Evaluation and artifacts

After ranking freeze, evaluate all frozen systems once and report nDCG@10,
MRR, paired intervals, win/tie/loss, zero transitions, density slices, and
failure inspection. Any post-qrel adjustment is a new exploratory iteration
and cannot inherit the freeze claim.

Artifacts use a distinct `issue18_pr17head` identifier and preserve:

- source and environment receipt;
- configs and exact commands;
- memberships, centroids, support, dispersion, medoids, annotations, and
  representative questions;
- every query-candidate score and contributing centroid IDs;
- rankings, TREC runs, and hashes;
- qrel-free viability/cost and repeatability receipts;
- evaluation, report, explorer, and final diagnostic receipt.

After PR #17 merges, these artifacts retain their original provenance. A
rebased rerun receives a new identifier and is compared rather than silently
replacing this run.

## Qrel-free viability repair amendment (2026-07-15)

Pass one completed with frozen artifact aggregate
`2a3909cfea23af246eed59565557d57ec4c5b8eb1e2f1d5c2b11d2f9f8c6d5b9` while
qrels remained sealed. Advisor review found the artifact valid but the receipt
procedurally incomplete. The user approved this bounded repair before further
implementation.

- Preserve the frozen artifact and every ranking hash; do not run the repeat.
- Add qrel-free rank-overlap and diversity comparisons for all eight systems
  against both query-kNN and the separately frozen contextual-E5 control.
- Correct localized dominance/query-sensitivity failure to
  `stop-query-sensitivity`; do not describe it as insufficient distinctness.
- Add actual execution-code, command, deterministic environment, dependency,
  timestamp, source-control, and artifact-hash provenance to the receipt.
- Permit the preregistered one-time diagnostic evaluation after a strictly
  valid non-integrity stop. Only repeat authorization remains conditional on
  `continue`; `reject-invalid` continues to prohibit evaluation.
- Obtain advisor review of the repair before opening qrels.

The complete repair design is locked in
`docs/superpowers/specs/2026-07-15-issue18-viability-repair-design.md`.

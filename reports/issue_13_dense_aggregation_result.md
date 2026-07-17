# Issue 13: Dense aggregation result and next decision

## Executive summary

Issue 13 is a valid negative result. The tested local-domain-prior aggregation
scored **0.3487 nDCG@10** and **0.3802 MRR**, below the frozen MPNet
nearest-neighbor baseline at **0.3599** and **0.4041**. All integrity checks
passed, but every effectiveness gate failed. The method is rejected and no
confirmation run is authorized.

The validated scoring path reproduced the frozen baseline, component
composition, and final ranks. No implementation error was identified that
explains the loss. The conceptual gap is narrower: the experiment called one
hard, uniform average over a small MPNet query cluster “domain evidence.” That
estimate uses the same query embeddings and qrels as local retrieval, but
discards target-to-support similarity and is much coarser.

We then ran one focused exploratory test to separate semantic membership from
support size. For every held-out query, a hard cluster mean was compared with
an unweighted nearest-neighbor mean using exactly the same number of support
queries. Hard clusters lost by **0.0493 nDCG@10** with 20 clusters and by
**0.0416** with 40 clusters. This is strong evidence that hard cluster
membership selects worse evidence; small support alone does not explain the
failure.

The decision is therefore:

1. retain the frozen MPNet baseline;
2. stop hard query-cluster aggregation work;
3. merge this result and its compact reproducible preflight, not the rejected
   experiment implementation;
4. evaluate a different evidence family next rather than tuning more cluster
   weights on the same development outcomes.

## Where the project started

The task ranks the same 150 candidate LLM categories for each of 322
development queries. A good system places candidates judged useful for the
target query near the top.

The frozen leakage-safe reference transfers candidate grades from semantically
similar historical queries using MPNet embeddings. It selected 50 neighbors in
320 of 322 outer folds.

| Frozen baseline metric | Value |
| --- | ---: |
| Mean nDCG@10 | 0.35985474409504775 |
| Mean reciprocal rank | 0.40406330811168817 |
| Queries with zero nDCG@10 | 116 |

Issue 13 asked whether the local estimate could be stabilized with:

- a **domain score**: the uniform candidate-grade mean in one semantic query
  cluster; and
- a **global prior**: the uniform candidate-grade mean over the allowed
  training fold.

The preregistered grid contained 16 combinations of 20/40 clusters, 20/50
neighbors, local weight 0.50/0.75, and prior shrinkage 0/0.10. Nested
leave-one-query-out selection chose a configuration without seeing the outer
query's grades.

## Authoritative result

| Metric | Frozen MPNet | Dense aggregation | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.3598547441 | 0.3486808479 | **-0.0111738962** |
| MRR | 0.4040633081 | 0.3801586444 | **-0.0239046637** |
| Zero-nDCG queries | 116 | 123 | **+7 worse** |

The paired 95% nDCG-delta interval was **[-0.02543, +0.00264]**. Query-level
nDCG wins, ties, and losses were **66 / 168 / 88**.

All integrity, completeness, leakage-boundary, numerical, score-recomposition,
and rank-reproduction checks passed. The outcome is `reject`, not
`reject-invalid`: the experiment ran as specified, but the ranking was worse.

## What the completed experiment diagnosed

Re-ranking the stored outer component scores produced these post-hoc
diagnostics:

| Diagnostic ranking | nDCG@10 | MRR | Zero nDCG |
| --- | ---: | ---: | ---: |
| Frozen MPNet baseline | 0.35985 | 0.40406 | 116 |
| Selected local signal only | 0.35882 | 0.40226 | 117 |
| Selected local plus prior | 0.35804 | 0.40250 | 118 |
| Selected domain signal only | 0.28085 | 0.31801 | 151 |
| Selected local plus domain, no prior | 0.34864 | 0.38010 | 123 |
| Authoritative final blend | 0.34868 | 0.38016 | 123 |

Conditional on the configurations already selected, replacing the domain mass
with local mass improves mean nDCG by **0.00936**, about 84% of the final gap.
This is a descriptive counterfactual, not unique causal attribution: nDCG is
nonlinear, components interact, and the settings were originally selected for
the blend.

The global prior is not the main problem. Removing it changes nDCG by only
about **-0.00004**. All 322 outer folds selected the maximum available local
weight, 0.75, and the grid omitted the necessary pure-local boundary at 1.0.
The selector was consistently asking for less cluster influence than the grid
allowed.

## Advisor-reviewed estimator gap

The high-level intuition—broader evidence might stabilize local retrieval—is
reasonable. The implemented estimator is a weak realization of that idea:

- local and cluster evidence both come from the same MPNet geometry and the
  same training qrels;
- a target is assigned to exactly one centroid, with no uncertainty or soft
  membership;
- every cluster member receives equal weight regardless of similarity;
- cluster support is small and candidate scores are quantized;
- raw local and cluster scores are blended without calibration.

Across the authoritative outer arrays, the local signal had a median of **95
distinct scores** over 150 candidates. The cluster signal had a median of only
**8**, assigned exactly zero to about **57 candidates per query**, and tied for
its top score in 96 of 322 queries. These are estimator-quality limitations,
not final-ranking tie bugs.

The qrels are complete. A zero cluster score therefore does **not** mean a
missing judgment. It means none of the assigned cluster members had a positive
grade for that candidate. Comparing cluster and local zero rates alone cannot
separate poor semantic membership from smaller support, which motivated the
matched-support preflight below.

## The single focused preflight

### Question

Does hard semantic clustering select better support than ordinary nearest
neighbors when both estimates use the same number of uniformly weighted
queries?

### Method

For each of 322 held-out queries and each cluster count, the preflight:

1. fitted deterministic KMeans on the other 321 frozen MPNet embeddings;
2. assigned the target to the nearest centroid;
3. averaged candidate grades over the selected cluster;
4. selected the same number of positive-cosine nearest neighbors and averaged
   their grades uniformly;
5. ranked all 150 candidates with the same fold-prior and candidate-ID tie
   break;
6. evaluated both rankings against the held-out query.

This required 644 fits, not the previous 104,006-fit nested selector. Two fresh
runs produced byte-identical aggregate JSON.

### Result

| Support rule | nDCG@10 | MRR | Zero nDCG |
| --- | ---: | ---: | ---: |
| Hard cluster, 20 clusters | 0.29022 | 0.32429 | 148 |
| Matched neighbors, 20-cluster support size | **0.33953** | **0.38234** | **124** |
| Hard cluster, 40 clusters | 0.28656 | 0.33505 | 154 |
| Matched neighbors, 40-cluster support size | **0.32820** | **0.36880** | **136** |

| Cluster count | Cluster minus matched nDCG | Wins / ties / losses |
| ---: | ---: | ---: |
| 20 | **-0.04930** | 75 / 136 / 111 |
| 40 | **-0.04163** | 71 / 148 / 103 |

The average selected support was 19.2 queries for 20 clusters and 11.6 for 40
clusters. Yet only about 41% and 39% of cluster members, respectively, appeared
in the target's equal-count nearest-neighbor set. Hard partitioning therefore
chooses a materially different—and substantially worse—support neighborhood.

Matched unweighted neighbors also remain below the frozen MPNet baseline, but
that comparison changes both weighting and support size: the matched controls
average 19.2/11.6 neighbors, while the frozen baseline selected 50 neighbors in
320/322 folds. This preflight therefore establishes only that nearest-neighbor
membership is better than hard cluster membership at equal support. It does
not isolate the effect of similarity weighting.

### Interpretation limit

This preflight uses development outcomes that are already observed. It is an
exploratory mechanism check, not a new promotable effectiveness result. It can
stop an unpromising direction; it cannot establish hidden-test improvement.

## What is being promoted

This merge candidate contains only:

- this standalone scientific record;
- the sanitized aggregate preflight receipt;
- the focused preflight script and synthetic unit tests;
- the small design and execution plan.

It intentionally excludes the rejected ranker, HDF5 fit store, 16-configuration
nested runner, compatibility layers, legacy-control exporter, custom artifact
framework, and their experiment-specific test surface. Generic artifact I/O is
already provided by current `main`.

## Next decision

Do not spend another full run on hard or softly reweighted versions of these
query-only clusters. The matched-support result fails the preregistered
exploratory continuation rule for both cluster counts.

The next ranking experiment should use genuinely distinct candidate-query
evidence—such as candidate response/profile interactions—while retaining the
frozen MPNet ranking as a control. That work requires its own small preflight
and untouched evaluation protocol; it is not part of this PR.

## Reproduction and sources

Run the focused preflight with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=.:.venv/lib/python3.13/site-packages \
python scripts/run_issue13_matched_support_preflight.py
```

Review map:

1. [Machine-readable aggregate result](issue_13_matched_support_preflight.json)
2. [Focused preflight script](../scripts/run_issue13_matched_support_preflight.py)
3. [Synthetic unit tests](../tests/test_issue13_matched_support_preflight.py)
4. [Design](../docs/superpowers/specs/2026-07-16-issue13-result-and-matched-support-preflight-design.md)
5. [Implementation plan](../docs/superpowers/plans/2026-07-16-issue13-result-and-matched-support-preflight.md)

The aggregate receipt includes the authoritative result/component/selection
facts cited above, provenance digests for the retired result artifacts, and
validated hashes of the frozen queries, qrels, MPNet embedding rows, derived
similarity matrix, and encoder metadata. It contains no query text, query IDs,
qrel rows, rankings, or raw arrays.

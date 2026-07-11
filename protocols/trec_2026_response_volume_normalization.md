# TREC 2026 response-volume normalization preregistration

Status: locked before effectiveness scoring; advisor approved on 2026-07-11.

This protocol is a focused follow-up to the frozen response-retrieval experiment in
`trec_2026_response_retrieval.md`. That experiment added every positive retrieved
response score for a model (`SumScore`). It improved over TF-IDF, but models with more
retained historical responses also tended to receive larger sums: the deployment
configuration had inventory-versus-score Spearman rho 0.8388, and the largest response-
inventory quartile contained 24.7% of models but received 46.6% of top-10 placements.
Those measurements show association, not causation; stronger models may legitimately
have more historical responses.

The present experiment asks whether simple, frozen caps or averages can preserve the
useful response evidence while reducing that volume dependence. It changes only the
model-level aggregation of already retrieved response-document scores. Log
probabilities, dense response embeddings, learned weights, direct inventory penalties,
and lexical/dense fusion remain out of scope.

## Inherited data, retrieval, and leakage controls

The following are inherited without change from
`protocols/trec_2026_response_retrieval.md`:

- Official discovery corpus and source hash, 322 development queries, qrels, and the
  150-model universe.
- Stable discovery-document order, basic and aggressive filters, BM25 and explicit RM3
  implementations, response-only and query-plus-response fields, and retrieval depths
  100, 500, and 1,000.
- Analyzer, numeric types, strictly-positive document eligibility, stable document-score
  ties, and the exact-query mask before both RM3 feedback and final retrieval.
- The 24 retrieval configurations and their declared order: method, field, filter, then
  depth.
- Strict nested leave-one-query-out selection; fold-local qrel priors; macro nDCG@10 as
  the only selection metric; MRR as a non-selecting secondary metric; and the same
  fixed TF-IDF, frozen dense, and fold-local global-prior comparisons.

The score of every eligible response hit is therefore qrel-free. Development qrels may
enter only the inherited fold-local tie prior, inner configuration selection, and final
effectiveness evaluation.

## Frozen aggregation definitions

For target query `q`, model `m`, and one of the 24 retrieval configurations `c`, let
`H(q,m,c) = (s1, ..., sn)` be the eligible retrieved response-document scores belonging
to `m`, ordered by document score descending and then stable document ordinal ascending.
All calculations use float64 and `math.fsum`. Every aggregation returns zero when
`n = 0`.

The frozen SumScore control is:

- `sum_all`: `math.fsum(H)`.

The primary experimental grid contains exactly these six aggregations, in this order:

1. `best_hit`: `s1`.
2. `top3_sum`: `math.fsum(s1, ..., s_min(3,n))`.
3. `top10_sum`: `math.fsum(s1, ..., s_min(10,n))`.
4. `top3_mean`: `top3_sum / min(3,n)`.
5. `top10_mean`: `top10_sum / min(10,n)`.
6. `mean_all`: `sum_all / n`.

`best_hit` is not duplicated as top-1 sum or top-1 mean. The implementation must derive
all seven outputs from the same retrieved hits rather than rerunning BM25 or RM3 for
each aggregation.

## Frozen candidate order and nested systems

The primary volume-normalized grid contains 144 candidates: the six experimental
aggregations in the order above, then the existing 24 retrieval configurations in their
frozen order within each aggregation. The SumScore control has its own 24-candidate grid
and is not eligible for selection by the primary volume-normalized system.

For every outer development query `o`:

1. For each candidate and each inner query `i != o`, rank models by
   `(-aggregate_score, -prior_excluding_o_and_i, model_id)`.
2. Select the candidate with the largest mean inner nDCG@10 across the other 321
   queries. `numpy.argmax` over declared candidate order resolves exact configuration
   ties. No epsilon, MRR tie-break, bias diagnostic, or external baseline result enters
   selection.
3. Rank the outer query by
   `(-aggregate_score, -prior_excluding_o, model_id)` and retain that untouched outer
   prediction.

The primary result concatenates the 322 outer predictions selected from only the 144
normalized candidates. Independently, the frozen SumScore control repeats the same
nested procedure across only its original 24 candidates. Individual aggregations and
fixed candidates may be described, but no post-hoc outer-test winner may replace the
preregistered joint primary system.

After unbiased nested reporting, select exactly one normalized deployment candidate
from the declared 144-candidate grid by highest macro nDCG@10 over all 322 development
queries. Rank each query by `(-aggregate_score, -prior_excluding_q, model_id)`;
`numpy.argmax` over declared candidate order resolves exact metric ties. Select the
SumScore deployment configuration analogously from its separate 24-candidate grid.
These configurations are future hidden-test deployment hints only. Their scores on the
same 322 development queries are optimistic and must not be presented as nested
evidence.

## Required effectiveness comparisons

Report outer-fold macro nDCG@10 (primary) and MRR (secondary) for:

- Jointly selected volume-normalized response retrieval.
- Independently nested frozen SumScore response retrieval.
- Frozen dense query retriever.
- Fixed TF-IDF query-kNN at `k=10`.
- Fold-local global-prior control.

For every direct comparison, report identical-query win/tie/loss counts and paired
10,000-resample query-bootstrap 95% intervals for the metric difference. The intervals
condition on the completed nested selections and do not rerun selection inside each
resample.

## Required volume and stability diagnostics

- Preserve the earlier retained document, character, token, hit-count, score, zero-score,
  overlap, analyzer, index, source, configuration, tensor, and run diagnostics.
- Report outer-fold aggregation and complete-candidate selection counts.
- Report selection stability and the full-development deployment hint, clearly separated
  from the unbiased nested result.
- Use the same retained-document inventory definition and deterministic model quartile
  assignment used for the reported SumScore rho 0.8388 and largest-quartile share 46.6%.
- For both the frozen SumScore deployment configuration and the normalized deployment
  configuration, report inventory-versus-aggregate-score Spearman rho and the share of
  fixed-candidate top-10 appearances received by each frozen inventory quartile.
- Use these full-development deployment-configuration diagnostics, and only these, for
  the two preregistered volume gates. They are like-for-like behavior diagnostics, not
  unbiased effectiveness estimates.
- As secondary robustness evidence, compute diagnostics separately for the independently
  nested SumScore predictions and the primary normalized predictions. Within each outer
  fold, construct the inventory vector and deterministic inventory quartiles using that
  fold's selected filter; compute Spearman rho across model inventory and outer-query
  aggregate score, and the largest quartile's share of the outer query's ten placements.
  Report query-count-weighted fold means and fold ranges. Never pool fold-varying
  inventories into one synthetic vector.
- For both stitched SumScore and normalized out-of-fold top-10 predictions, report the
  fraction of positions whose aggregate-score tie was resolved by the fold-local qrel
  prior and the fraction occupied by zero-aggregate-score models. These non-gating checks
  distinguish volume reduction from increased fallback to the prior.
- Treat inventory associations as bias-risk diagnostics, not proof that response volume
  causes effectiveness.

## Preregistered decision rule

Advance volume-normalized response aggregation only if the primary system satisfies all
of the following:

1. Its paired nDCG@10 delta over TF-IDF has a 95% interval whose lower bound is above
   zero.
2. Its outer-fold nDCG@10 point estimate is no more than 0.010 below the independently
   nested SumScore control.
3. The normalized deployment configuration has absolute inventory-versus-aggregate-
   score Spearman rho at most 0.7388, a reduction of at least 0.10 from 0.8388.
4. The largest frozen inventory quartile receives at most 41.1% of normalized deployment
   top-10 appearances. This removes at least one quarter of its excess over the 24.7%
   model share from the 46.6% SumScore reference.

Matching or exceeding SumScore nDCG@10 is a strong success. MRR and performance relative
to the dense baseline are important context but are not pass/fail gates. If any of the
four preregistered gates fails, retain SumScore and stop this aggregation branch before
considering log probabilities or fusion.

## Excluded variants and acceptance tests

No tuned count exponents, direct inventory penalties, medians, quantiles, learned
weights, additional post-score cutoffs, log probabilities, or lexical/dense fusion may
be added. Any later idea requires a new preregistration and evaluation.

Required tests include exact hand calculations for every aggregation; empty, one-hit,
and fewer-than-cap cases; deterministic document and candidate ties; proof that all
aggregations share one retrieval pass; SumScore parity with the prior experiment; exact
144-candidate primary order and 24-candidate control order; naive-versus-vectorized
nested equivalence; outer-qrel mutation invariance; unchanged masks and priors; complete
322-by-150 run validation; deterministic repeated official runs; and independent
recalculation of headline metrics, intervals, and decision gates.

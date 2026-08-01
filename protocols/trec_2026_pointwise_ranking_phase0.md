# TREC 2026 Pointwise Ranking Phase 0 Protocol

Status: frozen before pointwise effectiveness evaluation on 2026-08-01.

## Question and boundary

Do the audited OpenAI Luna pointwise labels improve leakage-safe,
query-specific candidate ranking on the complete 322-query organizer
development qrels? This protocol permits local file reads of frozen artifacts.
It forbids LLM API calls, pointwise regeneration, new embedding inference,
pairwise prompt assembly/submission, and use of a target qrel in its ranking.

Development evidence is previously seen and cannot establish hidden-test
effectiveness.

## Frozen inputs

| Input | Required SHA-256 |
|---|---|
| Audited compact pointwise labels | `7cde37d0d913b97ede43c4791b2f808e6d1b0c863bdcac729efde78480113c48` |
| Development query TSV | `862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c` |
| Development qrels | `3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35` |
| E5 refinement aggregate | `113fa6391df361bd161c2c9ad6a1d3ccb7f6e6f59a917b66135c6a87b0a33f83` |
| E5 record inventory | `e25237be81ffd86726fb4b3ccd1a2a8bdea7213197f84341b6bf50c23274ae92` |
| Candidate-local selected record IDs | `79c2d61cda800739eb36a1cbe7eb52bb81fda36d057e19bc6ee43991c673a371` |
| Candidate-local top-record indices | `567d2b0aa7d44508976f857a0188cbe3013f6e2c44935ae099edc39116c69047` |
| Candidate-local similarities | `de6fb07d5221d439bb2d9e31c74737cc913cb111fb4705f90f7ad90ad724a44b` |
| Frozen dense rankings | `734d96fd42d9cbff940c9c29acb6efe6459a0452d9cadd24a6fa3074beb96fb8` |
| Frozen dense TREC run | `ddb4f088de919d54e72b059fc5c62b598c6c8c937ceac2e30ae99417ddb96287` |
| Dense result receipt | `3ca4252cb37f12e9272561422fd2a45d0dcbf741770d0f544efda8acf4ddf902` |
| Dense per-query receipt | `ae4a0cf8bc7eb0642e1c0d315cafa3a56cf60b3d6bfb6ab63e80fb2d0de56437` |

Expected structure is 182,764 unique labels and E5 records, 6,816 audited
source questions, 150 identical candidates, 322 identical development queries,
ten selected record IDs per query/candidate, and a complete 322-by-150 qrel
matrix. `supported` must equal `verdict in {correct, partial}`. Join labels and
E5 records by `record_id`, never by row or source ordinal.

Development query IDs and NFKC/casefold/whitespace-normalized text must have
zero overlap with pointwise source questions.

## Frozen pointwise construction

1. Candidate prior: `(supported + 1) / (audited + 2)` using Beta(1,1)
   smoothing. Keep audited count and audited-question coverage separate.
2. Local score: unweighted arithmetic mean of the ten support labels named by
   the frozen candidate-local E5 tensor for each development query/candidate.
3. Routing residual: local score minus candidate prior.
4. Rank prior and residual descending, breaking ties by candidate ID.
5. Pointwise-only: equal RRF of prior and residual ranks with 1-based
   `1 / (60 + rank)`. Break ties by prior rank, residual rank, candidate ID.
6. Prior-only and routing-only remain separately evaluated diagnostics.

No pointwise constant, label, neighbor, or rank uses organizer qrels.

## Frozen control and combined system

- Control: the exact `dense_nested` ranking JSON from the qrel-free E5
  refinement dependency, whose per-query rankings originate from the existing
  nested leave-one-query-out dense experiment.
- Reproduction acceptance: nDCG@10 `0.35985474409504775`, MRR
  `0.40406330811168817`, 116 zero-nDCG queries, and all tracked per-query values
  within absolute tolerance `5e-12`.
- Combined: equal RRF of pointwise-only and dense ranks with `k=60`. Break ties
  by dense rank, pointwise rank, candidate ID.

## Metrics, uncertainty, and slices

- Primary: macro nDCG@10 with raw positive qrel gain.
- Secondary: MRR with any grade greater than zero relevant.
- Comparisons: candidate minus control, 10,000 paired percentile bootstrap
  resamples, 95% interval, seed 31, win/tie/loss, and zero-nDCG transitions.
- Query slices: qrel positive density `<=5`, `6-20`, `>20`; dense zero/nonzero;
  and stable quartiles of mean valid top-10 E5 similarity.
- Candidate slices: stable quartiles of audited-row count and smoothed prior.
  Filter rankings and qrels to each candidate slice before metric computation.
- Diagnostics: prior-only and routing-only. They cannot replace the primary
  pointwise-only or combined systems or override the decision rule.

## Decision thresholds

Pointwise evidence passes only if:

1. combined minus control nDCG@10 point estimate is positive;
2. its paired 95% nDCG@10 lower bound is greater than zero;
3. combined minus control MRR is at least `-0.005`;
4. combined has fewer than 116 zero-nDCG queries; and
5. routing-only nDCG@10 is greater than prior-only nDCG@10.

Recommend a later 500-pair pilot only if those conditions hold and the frozen
selector finds at least 500 eligible pairs across at least 100 source questions.
A negative result stops pairwise work.

## Frozen zero-call pair selector

Eligibility requires two supported records for the same source question and
the same verdict tier. Priority is:

1. `partial/partial` before `correct/correct`;
2. smaller absolute difference between candidate priors;
3. lower current candidate-use count;
4. SHA-256 of selector version, question ID, and sorted candidate IDs.

Select at most one pair per source question on the first pass, then fill in the
same order, with no candidate used more than 20 times unless 500 pairs are not
otherwise available. The Phase 0 receipt contains IDs and counts only. It must
not contain response/reference text, prompts, pairwise outcomes, or API state.

## Artifact plan

The complete ignored bundle lives under `artifacts/pointwise_ranking_phase0/`.
Tracked evidence is limited to sanitized structured results, query and candidate
slice TSVs, the canonical Research Hub QMD report, experiment-ledger entries,
and regenerated views. Private raw inputs and local absolute paths must not
enter Git or rendered delivery artifacts.

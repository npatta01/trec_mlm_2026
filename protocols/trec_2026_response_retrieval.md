# TREC 2026 response-retrieval preregistration

Status: locked before any response-retrieval effectiveness score is computed.

This protocol incorporates the independent advisor review completed on 2026-07-11.
The experiment asks whether the text of historical model responses can rank the 150
candidate models for a new query. Log probabilities, learned models, dense answer
embeddings, score fusion, and qrel-derived retrieval features are out of scope.

## Data and run classification

- Discovery corpus: official `llm_discovery_data.json`, 137,300,130 bytes,
  SHA-256 `b33c28aa2571cdf5b2867dcc00f0dc55e22198b90022dd23f748a03395301b25`.
- Development inputs: the same 322 queries, qrels, and 150-model universe used by
  Baselines 1 and 2.
- Corpus audit: 7,759 discovery queries and 1,163,850 query/model responses.
- Run classification: internal automatic. Only organizer-provided text and development
  qrels are used.
- One document is retained per discovery-query/model pair. Documents are never
  deduplicated. Stable document order is `(source_query_id, model_id)` lexicographic.

## Frozen filters

Filtering inspects response text only. `strip` means Python Unicode whitespace stripping;
normalized text is Unicode NFKC, casefolded, and whitespace-collapsed.

The `basic` filter removes null/blank responses and responses whose normalized text is
exactly `i don't know`. The official corpus contains 981,050 exact instances, so the
expected basic corpus has 182,800 documents. Malformed short variants such as
`I don:'t know` are deliberately left in this condition rather than hidden in an
open-ended cleanup rule.

The `aggressive` filter first applies `basic`, then removes responses with fewer than 50
stripped Unicode characters or normalized text beginning with one of these frozen
prefixes:

1. `there is not enough information`
2. `there is insufficient information`
3. `not enough information`
4. `insufficient information`
5. `the provided context does not contain enough information`
6. `the provided context does not provide enough information`
7. `the context does not contain enough information`
8. `the context does not provide enough information`
9. `the given context does not contain enough information`
10. `the given context does not provide enough information`
11. `the information provided does not contain enough information`
12. `the information provided does not provide enough information`
13. `the passage does not contain enough information`
14. `the passage does not provide enough information`
15. `the passages do not contain enough information`
16. `the passages do not provide enough information`
17. `the question cannot be answered`
18. `cannot be determined from the provided`
19. `cannot be answered from the provided`
20. `i cannot answer`
21. `i can't answer`

The length rule removes 9,636 basic documents and the prefix rules remove two additional
documents, so the expected aggressive corpus has 173,162 documents. These rules and
expected counts must not change after effectiveness scores are inspected.

## Frozen retrieval implementation

- Backend: `bm25s==0.3.9`, NumPy scorer and CSC backend, `auto_compile=False`.
- Numeric types: float64 scores and int32 term/document indices.
- Analyzer: Unicode NFKC; lowercase; regex `(?u)\b\w\w+\b`; the 33-word
  `bm25s.stopwords.STOPWORDS_EN` list; `PyStemmer==3.1.0` English stemming.
- Vocabulary: analyzed terms sorted lexicographically before IDs are assigned. The
  backend must not construct its set-derived vocabulary.
- BM25: Lucene method and IDF, `k1=0.9`, `b=0.4`.
- Document-score ties: stable document ordinal ascending.
- Only strictly positive-score documents are eligible for feedback or retrieval.

RM3 is implemented explicitly rather than delegated to an environment-dependent Java
stack. Initial BM25 retrieves the first 10 eligible feedback documents. For term `t`,
the feedback model is proportional to

`sum_d ((tf(t,d) / analyzed_length(d)) * BM25(query,d))`.

The top 10 terms are chosen by weight descending and analyzed term ascending, then
L1-normalized. Normalized original-query term frequency and the feedback model are mixed
with original-query weight 0.5. Weighted BM25 contributions are accumulated in sorted
term order using float64. If there are no eligible feedback documents or terms, the
original query is used unchanged.

## Exact-query guard

Overlap normalization is NFKC, casefolding, analyzer tokenization/stemming, and joining
tokens with one space. Every discovery document whose source-query token sequence equals
the target sequence is masked before feedback selection and again before final retrieval.
Retrieval depth is counted after masking. Masked documents remain part of qrel-free global
index statistics; this does not expose their model-specific responses, and the current
development set has zero such overlaps. The overlap count and this corpus-statistics
qualification must be reported.

## Frozen candidate grid

There are exactly 24 configurations, declared in nested order: retrieval method, indexed
field, filter, then depth.

- Method: `bm25`, then `bm25_rm3`.
- Field: `response_only`, then `query_response`, where the combined text is exactly
  `discovery_query + "\n" + response`.
- Filter: `basic`, then `aggressive`.
- Retrieved documents: 100, 500, then 1,000.
- Aggregation: SumScore only. `math.fsum` combines retained document scores for each
  producing model. Unreturned models receive zero.

Model rankings use `(-SumScore, -fold_local_global_prior, model_id)`. Configuration ties
use declaration order. The primary selection metric is macro nDCG@10; MRR is secondary
and is never a selection tie-break.

## Leakage-safe evaluation

For every outer development query `o`, each configuration is scored on the other 321
queries. For an inner validation query `i`, its ranking tie prior is computed from qrels
excluding both `o` and `i`. The selected configuration maximizes inner macro nDCG@10.
The outer prediction uses that configuration with a tie prior computed from qrels
excluding only `o`; `o`'s qrels are used only for final evaluation.

Stitched outer predictions are compared on the identical 322 queries against the frozen
dense nested baseline, fixed TF-IDF query-kNN at `k=10`, and the fold-local global prior.
Report nDCG@10, MRR, per-query scores, win/tie/loss counts, and paired 10,000-resample
95% intervals. Intervals condition on stitched out-of-fold selections; they do not rerun
configuration selection inside each bootstrap resample.

After unbiased nested reporting, one full-development configuration may be selected for
future hidden-test deployment. Any score for that configuration on the same development
queries is non-nested and must be labeled optimistic.

## Required diagnostics and acceptance checks

- Retained document and character/token volume by model and filter.
- Per-query/model hit count, mean hit score, SumScore, and zero-score model count.
- Spearman correlations between retained inventory and hit count, SumScore, mean rank,
  and top-10 frequency; top-10 representation by inventory-size quartile.
- A count-normalized score may be reported only as a diagnostic, never as a 25th candidate.
- Index, vocabulary, analyzer, source, score-tensor, run, and configuration hashes.
- Golden BM25 and RM3 fixtures, all-stage overlap masking, filter boundary/count tests,
  hand-calculated SumScore, complete rankings, deterministic ties, naive/optimized nested
  equivalence, outer-qrel mutation invariance, fold-prior denominator checks, control
  regressions, and complete 322-by-150 TREC run validation.

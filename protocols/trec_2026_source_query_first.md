# TREC 2026 source-query-first response preregistration

Status: advisor approved and locked before any source-query-first effectiveness score
is computed.

Base revision: `81d4aa50812864133f7066ee5b698fd48ea20503`, the fetched
`origin/main` revision that merges PR #5 and PR #8.

The independent methodology review approved this protocol on 2026-07-11 before
implementation scoring. The review fixed the retrieval unit, score normalization,
candidate grid, zero-response treatment, nested evaluation, comparisons, diagnostics,
and decision gates. Structural corpus counts were permitted before the lock; development
qrels were not used to design or inspect a source-query-first score.

## Question and interpretation

The experiment asks whether a model can be ranked for a new target query by first
finding related historical source queries and then measuring how well that model's
historical response addresses its own source query.

Three lexical signals remain separately attributable:

1. **Source-query relevance:** target query to historical source query.
2. **Target-to-response relevance:** target query to historical response.
3. **Answer alignment:** historical source query to its own response.

The primary model score uses source-query relevance to weight answer alignment. Direct
target-to-response relevance is retained as a same-opportunity diagnostic and ablation;
it is never blended into the primary score or allowed to select a configuration. This
avoids a post-hoc mixture weight and preserves attribution.

Answer alignment is a lexical proxy, not correctness, factuality, usefulness, or a
human judgment of response quality. The discovery file contains no response-level
quality judgments, so the report and code must not call the proxy correctness.

## Data and run classification

- Discovery corpus: official `llm_discovery_data.json`, 137,300,130 bytes,
  SHA-256 `b33c28aa2571cdf5b2867dcc00f0dc55e22198b90022dd23f748a03395301b25`.
- Development inputs: the same 322 queries, qrels, and 150-model universe used by the
  merged sparse, dense, SumScore, and Top10Sum experiments.
- The strict discovery loader and stable `(source_query_id, model_id)` record order are
  inherited from `trec_2026_response_retrieval.md`.
- Run classification: internal automatic. Only organizer-provided discovery text and
  development qrels are used. There are no pretrained encoders, log probabilities,
  learned weights, external corpora, or response judges.
- All retrieval and response-signal tensors are qrel-free. Development qrels may enter
  only fold-local tie priors, inner configuration selection, final effectiveness
  evaluation, paired comparisons, and the explicitly optimistic deployment hint.

## Frozen analyzer and BM25

Both source-query retrieval and response scoring use the inherited frozen analyzer:

- Unicode NFKC normalization and lowercase.
- Regex `(?u)\b\w\w+\b`.
- The pinned 33-word `bm25s.stopwords.STOPWORDS_EN` list with sorted-JSON SHA-256
  `fa4d8d14ba2c243e3f1b72e6124bd46141e6ea1bfadef22c5b0bb70bd602c9c8`.
- `PyStemmer==3.1.0` English stemming.

Plain BM25 uses the Lucene formula with float64 arithmetic, `k1=0.9`, `b=0.4`,
and Lucene IDF

`log(1 + (N - df + 0.5) / (df + 0.5))`.

Query-term contributions are accumulated in analyzed term order with `math.fsum` where
a scalar reduction is required. Only finite, strictly positive BM25 scores survive.
There is no RM3, pseudo-relevance feedback, query-plus-response concatenation, score
fusion, smoothing epsilon, or learned transformation.

## Unique source-query retrieval

Source-query retrieval indexes query text before inspecting any response. One canonical
retrieval document is retained per nonempty frozen-analyzer overlap key. When multiple
source IDs share the same key, the lexicographically smallest `source_query_id` is the
canonical representative. This prevents lexically identical queries from consuming
more than one place in `K`.

The qrel-free structural audit found 7,759 source-query IDs, 7,693 nonempty unique
frozen-analyzer keys, 58 duplicate-key groups, and 66 noncanonical IDs. The official
implementation must reproduce those values. Empty analyzer keys are counted and
reported but cannot enter positive retrieval.

For a target query, every source-query document whose overlap key equals the target's
key is masked before ranking. All analyzer-equivalent source IDs are therefore excluded,
not only the canonical representative. Remaining documents are ordered by BM25 score
descending and then `source_query_id` ascending.

Retrieve once to the frozen maximum `K=100`; the `K=10` and `K=50` candidates are exact
prefixes. An official target must have at least 100 unmasked, strictly positive source
queries or the run fails rather than silently padding or changing the denominator.

## Equal response opportunities and explicit zeros

Each selected source query creates exactly 150 response cells in lexicographic model-ID
order, one for every candidate model. A cell is invalid when its response is null,
Unicode-whitespace blank, exactly `i don't know` after NFKC normalization, casefolding,
and whitespace collapse, or begins with one of the 21 refusal prefixes frozen in
`trec_2026_response_retrieval.md`. Invalid cells are retained as opportunities and
receive zero for both response-side signals. This rule is named `basic_plus_prefix`.

The qrel-free structural audit found 981,050 exact normalized `i don't know` responses,
no null or blank responses, and two additional long responses matching the inherited
prefix list, leaving 182,798 eligible cells. The `aggressive` short-response rule is out
of scope because it would set legitimate concise answers to zero. Malformed variants
and any new heuristic cleanup are also out of scope. A valid response that analyzes to
no terms remains a valid opportunity with zero lexical score.

Response BM25 corpus statistics are local to one selected source query. The corpus has
exactly `N=150` slots; invalid cells are empty documents and remain in `N`. Average
document length is total analyzed tokens divided by 150. This local corpus is shared by
the source-to-response and target-to-response calculations for that source query. The
rule prevents discovery-wide response volume from silently changing which comparison
set or denominator a source query receives.

## Frozen signals and primary score

For target query `q`, a candidate depth `K`, selected source queries `s_j`, and model
`m`, define these nonnegative float64 raw scores:

- `R_j = BM25(q, s_j)`, source-query relevance.
- `A_jm = BM25(s_j, response_jm)`, answer alignment.
- `T_jm = BM25(q, response_jm)`, target-to-response relevance.

Invalid response cells have `A_jm = T_jm = 0`. Normalize with no epsilon:

- `w_j = R_j / math.fsum(R_1, ..., R_K)`.
- `a_jm = A_jm / math.fsum_m(A_jm)` for each source query `j`.
- `t_jm = T_jm / math.fsum_m(T_jm)` for each source query `j`.

If a denominator is zero, every value in that normalized vector is zero. Source-query
weights, normalized alignment, and normalized direct relevance are stored separately.

The primary model score is

`P_m(K) = math.fsum_j(w_j * a_jm)`.

The fixed diagnostic/ablation score is

`U_m(K) = math.fsum_j(w_j * t_jm)`.

The double L1 normalization gives each target one unit of source-query weight and each
selected source query one unit of response-evidence budget across the same 150 cells.
It prevents raw BM25 scale and the number of matching responses from acting as hidden
weights. `U_m(K)` may be reported only at the `K` chosen by the primary alignment
selector for that outer fold. It cannot enter selection, the four decision gates,
deployment selection, or a blended score.

## Frozen candidate grid and rankings

The primary candidate grid contains exactly three configurations in declaration order:

1. `source_alignment__k10`
2. `source_alignment__k50`
3. `source_alignment__k100`

Only `K` varies. Candidate ties use this declaration order. The primary selection metric
is macro nDCG@10; MRR is secondary and is never a selection tie-break.

Models are ranked by `(-P_m(K), -fold_local_global_prior, model_id)`. The direct
target-to-response diagnostic uses the same rule with `U_m(K)` and the primary-selected
`K`. Source-query relevance alone is model-invariant and is reported as retrieval
evidence rather than mislabeled as a model-ranking baseline.

## Leakage-safe nested evaluation

For every outer development query `o`, score each of the three primary candidates on the
other 321 queries. For an inner validation query `i`, compute the ranking tie prior from
qrels excluding both `o` and `i`. Select the `K` with the largest inner macro nDCG@10.
The untouched outer query is ranked with that selected `K` and a prior computed from
qrels excluding only `o`. The outer qrel may enter only final evaluation.

The qrel-free source retrieval and response signal tensors are computed once and shared
across all folds. Exact-query masking is applied before the tensor is created. Mutating
one outer qrel row must not change its retrieval evidence, its inner selection, or its
outer prediction except where that row legitimately contributes to other folds.

After the stitched nested result is reported, choose one fixed deployment `K` by the
highest full-development macro nDCG@10, with declaration order resolving exact ties.
Its same-development effectiveness is optimistic and must not be presented as nested
evidence. It exists only as a future hidden-test hint and to preserve like-for-like
deployment behavior diagnostics.

## Required comparisons

Report nDCG@10 and MRR on the identical 322 outer queries for exactly these headline
systems:

- Source-query-first answer alignment, nested `K` (primary).
- Frozen dense query-kNN nested reference.
- Independently nested frozen SumScore response retrieval.
- Independently nested Top10Sum response retrieval from the PR #8 normalized grid.
- Fixed TF-IDF query-kNN at `k=10`.
- Fold-local global prior.

The SumScore and Top10Sum controls must reproduce the merged reference rankings and
metrics before they are used. For the primary versus each of the five controls, report
identical-query win/tie/loss counts and paired 10,000-resample 95% intervals for nDCG@10
and MRR differences. Intervals condition on completed stitched selections; they do not
rerun nested selection inside a bootstrap sample. Seeds and baseline order are frozen in
the implementation and result artifact.

The target-to-response diagnostic may be shown separately with an explicit
non-selecting, non-gating label. It is not a seventh headline competitor.

## Required opportunity, volume, and stability diagnostics

- Canonical source-query counts, duplicate-key groups, empty keys, exact-target masks,
  positive retrieval counts, selected IDs, BM25 scores, and per-target weight sums.
- For every selected source-query/model cell: validity reason, raw `A` and `T`, normalized
  `a` and `t`, and the opportunity's contribution to primary and diagnostic model scores.
  Large cell tensors may be stored in compressed evidence rather than a reader-facing
  table.
- Per query and selected `K`: invalid-cell share, models with zero primary score, models
  with zero diagnostic score, prior-resolved top-10 ties, and zero-score top-10 positions.
- Nested `K` selection frequency, inner margins, fixed-candidate metrics, and the
  full-development deployment hint.
- Preserve the PR #8 BASIC retained-response inventory definition across all official
  source-query IDs for the frozen gate comparison. Separately report retained document,
  character, and token volume by model under `basic_plus_prefix` for the primary method.
- Preserve deterministic balanced inventory quartiles: order models by
  `(basic_inventory, model_id)` and split with `numpy.array_split` into four groups.
- For the fixed deployment `K`, report Spearman associations between BASIC inventory and
  total primary score, mean rank, and top-10 frequency, plus top-10 representation for
  all four frozen inventory quartiles.
- As non-gating robustness evidence, report per-outer-query inventory-versus-primary-
  score Spearman and largest-quartile top-10 share summaries. Treat association as a
  risk diagnostic, not proof that response volume causes ranking quality.
- Record analyzer, source, canonical-query, response-cell, signal-tensor, configuration,
  run, code, and artifact hashes plus CPU time and peak RSS.

## Preserved four-gate decision rule

Advance source-query-first answer alignment for owner consideration only if all four
PR #8 gates pass with their frozen thresholds and meanings:

1. The paired 95% interval lower bound for primary minus TF-IDF nDCG@10 is above zero.
2. Primary stitched outer-fold nDCG@10 is no more than 0.010 below independently nested
   SumScore.
3. The fixed deployment `K` has absolute BASIC-inventory-versus-total-primary-score
   Spearman rho at most 0.7388.
4. The frozen largest BASIC-inventory quartile receives at most 41.1% of fixed-deployment
   top-10 appearances.

Matching or exceeding SumScore nDCG@10 is a strong success, not a fifth gate. MRR and
comparisons with Top10Sum and dense query retrieval are important context but non-gating.
Even if all gates pass, the owner must explicitly decide whether this method affects any
previous response work. This experiment does not release, merge, or modify another PR.

## Excluded variants and required acceptance tests

No RM3, query-plus-response field, global response index, aggressive short-response
filter, tuned blend, multiplication of `A` and `T`, learned weights, log probabilities,
dense response embeddings, direct inventory penalty, or lexical/dense fusion may be
added after scores are inspected. A later variant requires a new preregistration.

Required tests and validation include:

- Hand-calculated Lucene BM25 fixtures, including empty local documents and `N=150`
  response-corpus statistics.
- Analyzer-key canonicalization, duplicate collapse, empty-key exclusion, and masking of
  every analyzer-equivalent exact target before retrieval.
- Exact top-100 retrieval with `K=10/50` prefix equivalence and stable ties.
- Null, blank, exact-IDK, frozen-prefix refusal, valid short answer, and
  zero-analyzed-token response cells.
- Hand-calculated `R`, `w`, `A`, `a`, `T`, `t`, `P`, and `U`, including zero denominators.
- Proof that every model receives the same `K` source-query cells and 150-slot local
  denominator, with invalid responses represented as zeros rather than dropped rows.
- Exact three-candidate order, primary-only nested selection, and proof that `U` cannot
  alter selected `K`, deployment `K`, or gates.
- Naive-versus-vectorized nested equivalence, fold-prior denominator checks, deterministic
  ties, and outer-qrel mutation invariance.
- Exact SumScore and Top10Sum control regressions, independent metric/interval/gate
  recalculation, complete 322-by-150 run validation, and byte-identical decision-relevant
  outputs from a clean repeated official run.
- Canonical report validation plus enhanced and semantic mobile checks at a 390-pixel
  viewport, including overflow, labels, source affordances, light/dark appearance, and a
  novice reading path.

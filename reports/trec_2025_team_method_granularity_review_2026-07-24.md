# Did TREC 2025 teams use relation/answer-type features?

Date: 2026-07-24

## Bottom line

**No reviewed TREC 2025 Million LLMs team used an explicit schema of entities,
relations, requested slots, constraints, reasoning operations, and answer contracts
for routing.** The proposed “semantic-demand frame” is therefore not a replication of
a successful track method. It is a new hypothesis assembled from adjacent semantic
parsing work.

The concern about granularity is justified. Those fields mix several different
abstraction levels:

- entities and predicates describe *what the question is about*;
- a requested slot describes *which argument is unknown*;
- operations describe *how an answer might be computed*; and
- an answer contract describes *the desired output form*.

There is no evidence yet that one extractor can produce those fields consistently
across this track's short search queries and open-ended questions, or that exact
agreement on those fields predicts which RAG-backed candidate has the right source
documents. They should not be made the next full ranking pipeline without first
passing a small, blinded validation.

The strongest direct precedent is simpler:

1. retrieve fine-grained historical query-response evidence;
2. estimate response quality or within-query preference;
3. aggregate that evidence to candidate IDs; and
4. learn the scoring function when reliable labels are available.

## What the official track says the unit of evidence should be

The organizers explicitly motivate the track as learning from **observable
behavior**, not static capability statements or hand-written descriptions. They say
real expertise cannot be adequately captured by those descriptions and give each
model's historical questions, answers, and token log-probabilities as its evidence
([official overview, pp. 1–2](https://trec.nist.gov/pubs/trec34/papers/Overview_mllm.pdf)).
The simulated experts share a base LLM but differ in the document collection their
retriever can access, so the latent capability being routed to is ultimately
**collection coverage for the specific information need**, not a general reasoning
skill alone (overview, pp. 3–4).

This makes relation/requested-slot compatibility potentially useful, but insufficient:
two questions can share a semantic frame while requiring unrelated source documents,
and two differently worded questions can be answerable from the same collection.

The official test was also very small: only 16 of 37 judged questions had at least one
relevant candidate answer ([overview, p. 6](https://trec.nist.gov/pubs/trec34/papers/Overview_mllm.pdf)).
Consequently, 2025 run ordering is useful precedent, not definitive proof of an ideal
representation.

## What the reviewed teams actually used

| Team | Effective representation/granularity | Explicit entity/relation/slot/contract extraction? |
|---|---|---:|
| **SRCB** | Quality labels over each query-response pair from LLM pairwise Quicksort, a Qwen reranker, or reference/query embedding similarity; then a frozen encoder with ranking/classification/regression heads | **No** |
| **UDInfo** | Answer embeddings clustered per candidate; per-candidate nearest historical answers; or query-embedding kNN plus latent query clusters and weak quality labels | **No** |
| **GenAIus** | BM25/RM3 retrieval of individual historical query-response records, optionally SPLADE-reranked, then SumScore/Vote aggregation to candidate IDs | **No** |
| **UvA IRLab** | Dense query embedding, candidate ID, and global log-probability statistics in LambdaRank; a post-submission extension added latent query clusters and per-cluster behavior statistics | **No** |

### SRCB

SRCB, whose `submission_4` is first in the official run appendix, manufactured
response-quality supervision. Its Quicksort procedure asks a powerful LLM to compare
responses for the **same question**; alternatives score a query-response pair with
Qwen3-Reranker-8B or compare an answer with a generated reference. The downstream
model maps a query encoding to candidate scores
([SRCB paper, §2.1–2.2](https://trec.nist.gov/pubs/trec34/papers/SRCB.mllm.tot.pdf);
[official results](https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main.html)).
There is no intermediate semantic frame.

### UDInfo

UDInfo's local evidence is geometrically fine-grained, not symbolically parsed. One
method embeds every answer, uses ten answer centroids per candidate for broad scoring,
then averages the ten nearest answer similarities for refinement. Another embeds
discovery questions, retrieves the 100 nearest queries, and combines their weak
candidate-quality labels with 40 latent k-means regions and global priors
([UDInfo paper, §3–4](https://trec.nist.gov/pubs/trec34/papers/UDInfo.mllm.pdf)).
The paper calls clusters “expertise regions” or “domain-level” evidence, but does not
name or extract relations, roles, or answer types.

### GenAIus

GenAIus gives the clearest evidence about granularity. It compared:

- one concatenated profile per candidate;
- one document per historical question; and
- one document per candidate's historical query-response pair.

Response-instance retrieval was strongest on development and its response-based runs
also outranked its question and profile runs in the official test. It used ordinary
BM25/RM3, refusal/default filtering, optional SPLADE reranking, and expert-search
aggregation—not semantic parsing
([GenAIus paper, §2–4](https://trec.nist.gov/pubs/trec34/papers/GenAIus.mllm.pdf);
[official results](https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main.html)).

This supports **local evidence at the historical response level**. It does not support
the proposed feature schema.

### UvA IRLab

UvA's submitted LambdaRank features were a 384-dimensional Sentence-BERT query vector,
a candidate ID, and candidate-level mean/standard-deviation log-probability statistics.
Its post-submission extension clustered query embeddings into 100 latent topics and
added per-topic confidence, refusal, response-length, support-count, and centroid
distance features. The authors explicitly found coarse Bing topic-tag profiles weak
and describe a granularity mismatch between broad tags and query-level matching
([UvA IRLab paper, §2–4](https://trec.nist.gov/pubs/trec34/papers/uvairlab.mllm.pdf)).
Again, this is learned/latent conditioning, not entity-relation extraction.

## What adjacent literature does—and does not—justify

### Learned routers do not require this schema

RouteLLM learns from query-level pairwise preferences using similarity-weighted
ranking, matrix factorization, and learned classifiers. Its important finding for this
project is that router success depends strongly on whether the routing training
distribution covers the evaluation distribution; it does not validate a fixed
entity/relation/answer-contract representation
([RouteLLM](https://arxiv.org/abs/2406.18665)).

### Semantic parsing supports some fields only in a grounded task

There is real precedent for structural question representations:

- Uni-Parser composes relations/entities (or database tables/columns/values) and
  operations such as `select`, `where`, and `count` into executable logical forms
  ([Uni-Parser](https://aclanthology.org/2022.emnlp-main.605/)).
- BREAK/QDMR represents a complex question as an ordered sequence of natural-language
  operations needed to answer it
  ([BREAK](https://aclanthology.org/2020.tacl-1.13/)).

But these works solve **answer execution against a known KB/database or annotated QA
benchmark**. Their relations and operations are grounded by an ontology, schema,
execution language, or large supervised dataset. The TREC routing problem has none of
those: its source collections are hidden behind candidate IDs, and our 322 qrel
questions do not provide semantic-frame supervision. Borrowing the field names without
their grounding risks creating plausible but unstable annotations.

“Answer contract” is particularly weakly justified as a routing feature. It can
distinguish list/count/date/explanation requests, but no reviewed 2025 team showed that
this predicts document-collection coverage after query-response similarity and
response quality are known.

## Recommendation

Do **not** frame all 7,759 historical questions now, and do not make hand-extracted
fields a hard retrieval gate.

First run a small, blinded **representation test**, not a ranking experiment:

1. Sample about 30 development questions and, for each, reuse 10 high-similarity and
   10 deliberately more distant historical questions.
2. Have humans label only the outcome that matters:
   **Would evidence sufficient to answer this historical question likely also support
   the target information need?** Use `yes / partial / no / ambiguous`.
3. Compare three qrel-free matchers against those labels:
   - current E5 similarity;
   - a direct cross-encoder or LLM pairwise “same information need” score;
   - the proposed extracted-field compatibility.
4. Double-annotate a subset. Reject the schema if humans cannot apply it consistently,
   extraction agreement is low, or it does not beat the direct pair scorer.
5. If a direct pair scorer improves analogue precision, connect those matched rows to
   within-question candidate outcomes or response-quality labels—the part most clearly
   supported by SRCB, UDInfo, and GenAIus.

This test answers the real question without trusting an arbitrary taxonomy. If the
feature schema wins, it earns its place empirically and its useful granularity will be
visible in ablations (relation only; relation + requested slot; plus constraints; plus
answer form). If it loses, use it only as a failure-analysis vocabulary and pursue the
more defensible direct-query-pair/response-quality route.

## Primary sources

- [Official TREC 2025 Million LLMs overview](https://trec.nist.gov/pubs/trec34/papers/Overview_mllm.pdf)
- [Official run appendix](https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main.html)
- [SRCB participant paper](https://trec.nist.gov/pubs/trec34/papers/SRCB.mllm.tot.pdf)
- [UDInfo participant paper](https://trec.nist.gov/pubs/trec34/papers/UDInfo.mllm.pdf)
- [GenAIus participant paper](https://trec.nist.gov/pubs/trec34/papers/GenAIus.mllm.pdf)
- [UvA IRLab participant paper](https://trec.nist.gov/pubs/trec34/papers/uvairlab.mllm.pdf)
- [RouteLLM](https://arxiv.org/abs/2406.18665)
- [Uni-Parser](https://aclanthology.org/2022.emnlp-main.605/)
- [BREAK/QDMR](https://aclanthology.org/2020.tacl-1.13/)

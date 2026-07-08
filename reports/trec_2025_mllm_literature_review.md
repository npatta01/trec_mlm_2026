# TREC 2025 Million LLMs Track Literature Review

Audience: readers who are new to information retrieval, model routing, and TREC.

## Executive Summary

The TREC 2025 Million LLMs Track asked a new retrieval question: given a user query, can a system rank the language model IDs that are most likely to answer it well? Traditional search ranks documents. This track ranks candidate expert LLMs.

The 2025 task did not actually evaluate one million independent commercial models. It used 1,131 simulated expert LLM IDs. Each ID behaved like a specialist because it was backed by a different cluster of documents in a retrieval-augmented generation setup. Participants could not query the candidate models at test time. They had to infer expertise from released discovery data: past queries, each model's precomputed answers, and response metadata such as token-level log probabilities.

The main research lesson is that "model expertise" can be treated as a retrieval problem. The strongest systems built evidence from previous query-answer behavior, weak labels from LLM judges or rerankers, embeddings, confidence/logprob statistics, and classical expert-search aggregation. The best official run was SRCB's `submission_4`, followed closely by UDInfo's hybrid weak-label ranker and GenAIus response-level BM25/RM3 runs.

The main caution is evaluation reliability. The official overview reports that NIST produced 50 test questions, found ground-truth answers for 37, and only 16 had at least one relevant model answer. The original plan for pooled human judging was not completed; final labels used GPT-5-as-judge. Treat the 2025 leaderboard as an important first benchmark and method comparison, not as a settled production ranking recipe.

## 1. What Problem Was TREC Studying?

Search engines normally answer "which documents should I show?" The Million LLMs Track asks "which model should I ask?"

This matters because LLM ecosystems are moving toward many specialized systems: a legal assistant, a biomedical assistant, a finance assistant, a local-history assistant, and so on. Querying all of them is expensive and slow. A router needs to decide which few experts are worth consulting before seeing their answers.

Plain-English analogy: imagine a library front desk with 1,131 specialist librarians. A visitor asks a question. The front desk must rank which librarians are most likely to know the answer, using only past examples of how each librarian answered earlier questions.

In IR terms, this is close to expert finding, model routing, and answer-quality prediction. The retrieval target is the model ID rather than a document or passage.

## 2. Official Task Definition

Input: a user query and a fixed pool of LLM IDs.

Output: a TREC run file ranking LLM IDs from most likely to least likely to provide a high-quality answer.

Constraint: participants could use the released discovery and development data, but could not issue new queries to the candidate LLMs during final evaluation.

Submission format: standard six-column TREC format:

```text
topic_id Q0 llm_id rank score run_id
```

The official track page distinguished:

- Automatic vs. manual runs: automatic means no manual changes after seeing test queries.
- Internal vs. external runs: internal uses only the released discovery data; external may use outside tools such as LLM judges, search engines, or extra data.
- Up to five official runs per team.

## 3. Data Released and How It Was Built

### Candidate Expert LLMs

The official pool contained 1,131 LLM IDs. These were simulated experts, not separately trained foundation models. The organizers used a shared LLM plus retrieval over different document clusters to make each ID behave as if it had a different domain of expertise.

The simulation worked roughly like this:

1. Build a large web/document corpus from ClueWeb22-derived sources.
2. Cluster documents into topical groups.
3. Treat each cluster as the evidence base for one expert model ID.
4. For a query, retrieve documents globally with BM25.
5. For each cluster/model ID, pass up to the best documents from that cluster into the shared LLM.
6. Store the generated answer and token-level log probabilities.

This design let TREC create a large expert-routing benchmark without needing thousands of proprietary LLMs.

### Document Corpus

The overview paper reports two main source streams:

- MS MARCO Web Search over ClueWeb22, starting from 69,947,610 English ClueWeb22-A documents.
- Researchy Questions documents, adding 332,874 documents.

After intersecting with Bing semantic category tags, removing duplicates, and filtering missing/empty content, the final semantically tagged corpus contained 26,566,121 documents.

### Query Sets

Discovery data:

- The overview describes 9,989 MS MARCO Web Search queries plus 4,961 Researchy Questions queries, totaling 14,950 discovery queries.
- Some participant papers report 14,940, so there is a small cross-source count discrepancy.
- For these queries, participants received precomputed responses from all 1,131 model IDs, plus metadata/logprobs in the metadata files.

Development data:

- 342 additional Researchy Questions queries.
- Gold/reference answers were available.
- GPT-o1 was used as an LLM judge to produce three-level qrels for each query-model pair.

Test data:

- NIST assessors created 50 hard questions.
- They had access to the collection and to answers from GPT-4o, GPT-5-nano, and Phi-4, and tried to write questions that those models did not answer well even though the answer existed in the collection.
- The overview says ground-truth answers were collected for 37 of the 50 questions.
- Only 16 of those questions had at least one relevant model answer in the candidate pool.

### Labels and Metrics

Labels were graded:

- 0: the model answer does not show expertise.
- 1: the answer has partial knowledge.
- 2: the answer properly answers the question and indicates expertise.

The intended headline metrics were nDCG@10 and MRR.

- nDCG@10 rewards putting highly relevant experts near the top 10.
- MRR rewards finding the first good expert as early as possible.

The official appendix also reports P@5, recall@1000, `ndcg_cut_5`, and other `trec_eval` metrics. In the saved evaluation files, the main top-10 value is `ndcg_cut_10`.

## 4. Feature Families Used by Participants

Across the submitted systems, "features" meant evidence signals used to estimate a model's expertise. The most important families were:

### Response-Level Retrieval Features

Teams indexed previous responses or query-response pairs, retrieved examples similar to the test query, and aggregated scores back to model IDs.

Why it worked: an individual past answer is a fine-grained behavioral clue. A whole-model profile can blur many domains together, but response-level matching can say "this model has answered similar things well before."

Used strongly by GenAIus.

### Model Profile Features

A profile summarizes each LLM ID across its historical answers. Profiles included answer embeddings, cluster centroids, domain/task tendencies, response length, refusal rate, lexical diversity, and other behavior statistics.

Used by UDInfo, paulphys, GenAIus profile baselines, and others.

### Embedding Features

Systems embedded queries, answers, or profiles into vector space and used cosine similarity or nearest-neighbor lookup.

Common embedding/reranking models included:

- BAAI/bge-large-en in UDInfo.
- BGE-M3 in UvA IRLab baselines.
- Sentence-BERT/all-MiniLM-L6-v2 in UvA IRLab LightGBM features.
- Qwen3-Embedding-8B in SRCB and paulphys.

### Weak Labels from LLM Judges or Rerankers

Because the discovery set did not come with full human labels, teams created pseudo-labels:

- LLM-as-judge scores for query-answer quality.
- Reranker scores.
- Pairwise comparison or Quicksort-like ranking of candidate answers.
- Embedding similarity to standard/reference responses.

These labels helped supervised rankers, but they can encode judge bias.

Used strongly by SRCB and UDInfo.

### Log-Probability and Confidence Features

The metadata included token-level log probabilities. Teams used average log-likelihood, confidence statistics, variance, and top-percentile confidence summaries as proxies for model capability.

This is attractive because confidence comes "for free" with generation metadata. But confidence is not correctness: a model can be confidently wrong.

Used by UvA IRLab, indelab, and paulphys.

### Classical IR Features

BM25, RM3 query expansion, pseudo-relevance feedback, and expert-search voting/aggregation were competitive.

Used strongly by GenAIus.

### Learning-to-Rank and Neural Routing

Several teams trained rankers or routers:

- LightGBM LambdaRank / LambdaMART.
- MLP classifiers over LLM IDs.
- Dense encoders and rerankers.
- Random forest and meta-rankers.

These methods worked best when paired with meaningful behavioral or weak-label features.

## 5. Official Submissions

There were 19 official automatic runs from six groups.

| Rank by nDCG@10 | Run | Team | nDCG@10 | MRR | P@5 | Recall@1000 | Main idea |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `submission_4` | SRCB | 0.1745 | 0.3126 | 0.1875 | 0.9826 | Qwen3-Embedding-8B weak-label/neural ranking pipeline |
| 2 | `infolab_UD_run2` | UDInfo | 0.1691 | 0.2725 | 0.1625 | 0.9980 | External LLM weak labels, query kNN, and domain clusters |
| 3 | `r-ef-rm3-1000-ss` | GenAIus | 0.1680 | 0.1980 | 0.1500 | 0.7371 | Extra-filtered response-level BM25/RM3 with sum-score aggregation |
| 4 | `r-f-rm3-1000-rr-ss` | GenAIus | 0.1608 | 0.2066 | 0.1375 | 0.8022 | Response BM25/RM3 plus SPLADE reranking |
| 5 | `r-f-rm3-1000-ss` | GenAIus | 0.1545 | 0.1843 | 0.1250 | 0.8022 | Response BM25/RM3 with sum-score aggregation |
| 6 | `infolab_UD_run4` | UDInfo | 0.1534 | 0.2579 | 0.1375 | 0.9660 | MLP classifier trained from weak labels |
| 7 | `ensemble_v1` | SRCB | 0.1522 | 0.2605 | 0.1500 | 0.9900 | SRCB development-selected ensemble |
| 8 | `infolab_UD_run1` | UDInfo | 0.1517 | 0.2760 | 0.1625 | 0.9993 | Hierarchical answer profiles and FAISS refinement |
| 9 | `ensemble_v3` | SRCB | 0.1485 | 0.2510 | 0.1625 | 0.9816 | SRCB ensemble |
| 10 | `ensemble_v2` | SRCB | 0.1477 | 0.2541 | 0.1375 | 0.9864 | SRCB ensemble |
| 11 | `submission_5` | SRCB | 0.1339 | 0.1971 | 0.1375 | 0.9593 | Qwen3-14B annotation, DeBERTa encoder, BCE loss |
| 12 | `non-neural` | indelab | 0.1309 | 0.2130 | 0.1500 | 0.9860 | Bradley-Terry skill, random forest, LightGBM meta-ranker |
| 13 | `lightgbm_job266431` | uvairlab | 0.1233 | 0.1433 | 0.1125 | 0.9888 | LightGBM LambdaRank with query embeddings and logprob statistics |
| 14 | `q-f-rm3-100-ss` | GenAIus | 0.1203 | 0.1966 | 0.1000 | 0.9860 | Question-level BM25/RM3 |
| 15 | `llm-f-100` | GenAIus | 0.1060 | 0.1890 | 0.0875 | 0.3788 | One profile document per LLM with BM25 |
| 16 | `infolab_UD_run5` | UDInfo | 0.1021 | 0.2114 | 0.1000 | 0.9990 | Deeper MLP router |
| 17 | `infolab_UD_run3` | UDInfo | 0.1018 | 0.1832 | 0.1000 | 0.9992 | Clustered LLM answer profiles |
| 18 | `mllm-indelab-09-17` | indelab | 0.0814 | 0.1479 | 0.1000 | 0.9972 | Neural dual-encoder ensemble using metadata-derived weak labels |
| 19 | `lambdamart_profiles` | paulphys | 0.0549 | 0.0704 | 0.0625 | 0.8540 | Behavioral profiles plus LambdaMART |

## 6. Participant Methods in More Detail

### SRCB

SRCB treated the task as supervised ranking without real discovery-set labels. Their recipe is: manufacture pseudo-labels for every query-model answer, train an efficient scorer that emits one score per LLM, and ensemble the best variants.

Inputs and preprocessing:

- Use the discovery matrix: roughly 15k queries by 1,131 candidate LLM responses.
- Keep the response text for every query-LLM pair.
- Use development qrels only for model selection and tuning.

Pseudo-label construction:

1. Quicksort labels: for each query, pick an anchor response, ask a strong LLM whether each other response is better or worse than the anchor, split into good/anchor/bad groups, and recurse until a full ranking is produced. Assign ties when responses are identical or judged equivalent.
2. Reranker labels: score each query-response pair with Qwen3-Reranker-8B and convert scores into a ranking.
3. Embedding-similarity labels: either generate a strong reference answer and compare candidate responses to it, or compare the response directly to the query.
4. Combination labels: ensemble the label sources. Their paper reports Quicksort and Combination as the strongest label-construction families.

Ranking model:

- Freeze a pretrained encoder for efficiency, then train lightweight task heads.
- Compare DeBERTa-v3-base, Qwen3-Embedding-8B, and Qwen3-32B; Qwen3-Embedding-8B was the practical winner.
- Try three formulations: ranking with one score per LLM, classification with a 3-class head per LLM, and regression with a sigmoid score per LLM.
- Losses tested included BCE and ListNet for ranking, cross entropy and focal loss for classification, and BCE-style regression.
- Convert continuous pseudo-labels into top-percentile categories for classification. For regression, zeroing labels outside the top x% improved reported development performance.

Inference:

1. Encode the test query.
2. Produce 1,131 scores, one per candidate LLM ID.
3. Optionally ensemble several trained variants by voting or score aggregation.
4. Sort LLM IDs by final score and emit the TREC run.

Implementation takeaway: start with Qwen3-Embedding-8B plus Combination or Quicksort labels and a regression/ranking head. The major risk is pseudo-label quality: the system may learn judge-model style preferences rather than true expertise.

### UDInfo

UDInfo is the most direct implementation reference because the paper describes five concrete routing methods. Their best official run was Method 2, a hybrid query-space model with weak LLM labels, kNN evidence, and domain clusters.

Method 1: hierarchical answer-profile router:

1. Embed all discovery answers with `BAAI/bge-large-en`; L2-normalize vectors.
2. For each LLM, run k-means over answer embeddings with `K_m = 10`.
3. Store centroids as the broad expertise profile.
4. Build one FAISS index per LLM over normalized answer embeddings.
5. At inference, embed the query and compute broad score as max similarity to the LLM's centroids.
6. Retrieve `K = 10` nearest answer embeddings from each LLM index and average similarities as the refined score.
7. Combine broad and refined scores with a mixing parameter. UDInfo reports `alpha = 0.2`.

Method 2: hybrid weak-label query router:

1. Use an external LLM judge, reported as `olmo2-13b`, to grade each discovery query-LLM answer in `{0, 1, 2}`.
2. Embed discovery queries with `BAAI/bge-large-en` and L2-normalize.
3. Build a FAISS index over query embeddings.
4. At inference, retrieve the top `K = 100` nearest discovery queries.
5. For each LLM, compute a similarity-weighted average of its weak labels over those neighbors.
6. Smooth the local estimate with the LLM's global prior using `alpha = 0.9`.
7. Cluster discovery queries into `K_c = 40` k-means clusters.
8. Precompute each LLM's average judged quality in each cluster, smoothed by the global prior.
9. Assign the test query to its closest cluster and combine local and domain scores with `lambda = 0.5`.

Other UDInfo baselines:

- Method 3: answer-centroid profile only. Store 10 answer centroids per LLM and score a query by max centroid similarity.
- Method 4: shallow MLP classifier. Weak-label each query, select the best judged LLM as the class label, embed the query, and train a PyTorch MLP with hidden sizes 512 and 256, ReLU activations, Adam at `1e-3`, and 20 epochs.
- Method 5: deeper MLP router, same target as Method 4 but with higher capacity.

Implementation takeaway: start with Method 2. It cleanly separates local evidence, global prior, and domain cluster expertise. Avoid making a pure MLP router your first baseline; it is less interpretable and underperformed the hybrid method.

### GenAIus

GenAIus is the strongest reproducible non-neural baseline. They treat model routing as expert search: retrieve historical evidence, then aggregate evidence back to LLM IDs.

Filtering:

- Start with about 16.9M discovery responses.
- Initial filter: remove obvious failures such as `No result found`. This left about 2.83M responses.
- Additional filter: remove refusal/low-information patterns such as `I don't have knowledge about` or `I don't have enough information`. This left about 1.24M responses.
- Keep the mapping from every indexed document to the LLM ID that produced it.

Three indexes to build:

1. LLM profile index: one document per LLM, created by concatenating that LLM's historical query-response pairs. Rank profiles with BM25/RM3 and submit the top 100.
2. Question index: one document per discovery query. Associate each query document with the LLMs that survived filtering for that query. Retrieve top 100 query documents, then vote or sum scores into LLM IDs.
3. Response index: one document per historical response, usually query plus response text. Retrieve top 1,000 response documents, then aggregate by LLM ID. This was the strongest family.

Scoring and aggregation:

- Use BM25 for first-stage retrieval and RM3 for pseudo-relevance feedback.
- Use Vote if each retrieved document should count equally.
- Use SumScore if each retrieved document contributes its retrieval score. GenAIus preferred SumScore because Vote created more ties.
- Optionally rerank top response documents with `naver-splade-v3`, then aggregate reranked scores.

Inference:

1. Run the test query against the response index with BM25/RM3.
2. Retrieve top 1,000 response documents.
3. Add each document score to the producing LLM's accumulator.
4. Sort LLM IDs by accumulated score.

Implementation takeaway: start with response-level BM25/RM3 plus SumScore over extra-filtered responses. Avoid starting with one giant profile document per LLM; it is simple but loses query-specific evidence.

### indelab

indelab submitted a neural dual encoder and a non-neural ensemble. The non-neural ensemble was stronger officially; the neural system is still useful as a representation-learning blueprint.

Neural dual-encoder run:

- Training data: about 4.45M examples, combining development qrels and weakly labeled discovery examples.
- Weak discovery labels: if response length is greater than 50 characters, assign relevance 1.0; otherwise assign 0.0.
- Use 10-fold cross-validation over development queries. Train with 90% of dev queries plus all weak discovery examples, validate on the held-out 10%, and ensemble the five best folds by validation nDCG@10.
- Query tower: process 384-dimensional query embeddings with 4-head attention, reshape through `384 -> 96 x 4 -> 384`, then dense layers `384 -> 256 -> 192 -> 128`.
- LLM tower: learned LLM embeddings `1131 -> 256`, then dense layers `256 -> 512 -> 384 -> 256 -> 128`.
- Score each query-LLM pair by cosine similarity between the 128-dimensional query and LLM representations.

The official description does not specify the exact training loss, so an implementation should make that choice explicit: pairwise ranking loss, contrastive loss, or regression to qrels are plausible, but not documented as indelab's exact choice.

Non-neural ensemble run:

1. Bradley-Terry skill model: create pairwise wins/losses between LLMs using heuristic response-quality signals such as length, structure, reasoning markers, and examples. Fit a global skill value per LLM.
2. Enhanced random forest: train roughly 150 trees on about 250k examples. Features include TF-IDF query vectors, LLM embeddings, query-complexity metrics, and discovery-profile features.
3. LightGBM meta-ranker: combine about 17 meta-features, including judge-like scores, Bradley-Terry skill, query characteristics, and cross-component interactions.

Implementation takeaway: start with the non-neural ensemble if you want robustness and interpretability. Avoid relying on response length as a strong quality label; it is only a cheap bootstrap signal.

### UvA IRLab / uvairlab

UvA IRLab built a compact learning-to-rank system around response confidence. The key idea is to use token-level log probabilities as cheap pseudo-supervision and as model-level capability features.

Official LightGBM feature vector:

- Encode the query with Sentence-BERT `all-MiniLM-L6-v2`, producing a 384-dimensional vector.
- Convert each LLM ID to a scalar integer, such as `llm_0032 -> 32`.
- For each discovery query-LLM response, compute average token log-likelihood.
- For each LLM, take its top 1% highest average-loglikelihood responses.
- Compute the mean and standard deviation over that top 1% subset as global capability and consistency features.
- Concatenate `[query_embedding_384, llm_id, global_mean, global_std]` for a 387-dimensional feature vector.

Labels and training:

- Convert average log-likelihood into pseudo-relevance grades: below `-2.0` is 0, between `-2.0` and `-1.0` is 1, and above `-1.0` is 2.
- Train LightGBM LambdaRank to optimize NDCG at 1, 5, and 10.
- Reported configuration: learning rate `0.05`, maximum leaves `31`, early stopping after 50 rounds, final rounds `1130`.

Inference:

1. Embed the incoming query once with SBERT.
2. For every candidate LLM, join the query vector with that LLM's ID and precomputed global confidence statistics.
3. Score all 1,131 feature rows with LightGBM.
4. Sort LLM IDs by predicted score.

Useful extension:

- Cluster discovery queries into 100 SBERT k-means clusters.
- For each LLM-cluster pair, compute top-1% confidence mean/std, refusal rate, mean response length, and cluster size.
- At inference, add distances to the three nearest cluster centroids.
- This expands the vector to about 408 dimensions and makes the router more domain-sensitive.

Implementation takeaway: start with LightGBM over query embedding, LLM ID, and top-1% logprob mean/std. Avoid interpreting confidence as correctness; confident hallucinations remain possible.

### paulphys

The paulphys run is a compact behavioral-profile design: build one feature vector per LLM summarizing how it behaves, then train LambdaMART to match query descriptors to those profiles.

Profile construction:

- Sample 50% of discovery responses; the paper reports that using more than half did not improve profile quality.
- Embed responses with Qwen3-Embedding-8B.
- For each LLM, compute a 64-dimensional behavioral profile covering domain expertise, task strengths, quality indicators, semantic descriptors, and statistical summaries.
- Domain labels are handcrafted: technology, science, medicine, law, business, education, arts, sports, politics, history.
- Task labels are handcrafted: explanation, comparison, analysis, instruction, creative, technical, factual.

Feature groups:

- Response metrics: token and character counts, log probabilities, punctuation ratios.
- Lexical diversity: type-token ratio and repetition rates.
- Structure: sentence counts, code/list usage, formatting, uppercase and digit frequency.
- Query-response interaction: Jaccard similarity and keyword overlap.
- Semantic embedding features: centroids, dispersion, and similarity statistics from Qwen3-Embedding-8B.

Ranking setup:

1. For each incoming query, compute query descriptors such as length and type-token ratio.
2. Infer query domain and task category with keyword matching.
3. For every query-LLM pair, concatenate query descriptors, the LLM behavioral profile, domain/task match features, and elementwise products between query labels and profile scores.
4. Aggregate domain/task interactions with simple statistics such as mean, standard deviation, and maximum.
5. Train LightGBM LambdaMART with the `rank_xendcg` objective.
6. Train on the top 200 LLMs per development query, but rank all 1,131 at inference.

Implementation takeaway: use this as an interpretable feature baseline, not as your only production router. Avoid relying on handcrafted domain/task categories alone; the official run ranked last, likely from overfitting and test distribution shift.

## 7. Cross-Paper Findings

1. Response-level evidence was stronger than single giant LLM profiles. GenAIus's profile run was weaker than its response-index runs.

2. Weak labels helped, especially when generated by strong LLM/reranker/embedding models. SRCB and UDInfo both benefited from pseudo-supervision.

3. Development results were not always predictive of test results. This is likely due to the small final judged test set and the difficult NIST-authored questions.

4. High recall@1000 did not guarantee good top-rank quality. Several runs retrieved almost all relevant models somewhere in the top 1000 but still failed to place them in the top 5 or top 10.

5. Global model quality signals were powerful but risky. UvA IRLab found that LightGBM could learn broad model competence from IDs and confidence features, but this can underuse domain-specific expertise.

6. Classical retrieval remains a serious baseline. BM25/RM3 response retrieval was close to the best official neural/weak-label systems.

7. Confidence is not correctness. Token log probabilities, response length, refusal behavior, and fluent style are proxies, not ground truth.

## 8. Limitations of the 2025 Track

Small official evaluation: only 16 test questions had at least one relevant answer, which makes leaderboard differences fragile.

LLM-as-judge final labels: the final evaluation used GPT-5-as-judge rather than the originally planned human assessment.

Simulated experts: the "LLMs" were cluster-backed RAG experts, not independent pretrained models. That is a reasonable benchmark construction, but it changes what conclusions can be drawn.

Count inconsistencies: the overview says 14,950 discovery queries, while some participant papers say 14,940. The overview discusses 50 test questions; local public topic files in the source cache contain fewer records.

Weak-label bias: many systems relied on LLM judges, rerankers, or confidence proxies. These may reward style, verbosity, or judge-model preference.

Distribution shift: development data came from Researchy Questions with generated qrels; test questions were NIST-authored to be hard for general LLMs.

## 9. Practical Recommendations for a 2026 System

Use response-level retrieval as a baseline. Index prior query-answer pairs or responses, retrieve similar evidence, and aggregate to LLM IDs.

Filter aggressively. Remove default responses, refusals, very short answers, and low-confidence cases before building profiles.

Combine local and global signals. Use nearest-neighbor evidence for query-specific routing and profile/domain statistics for stability.

Use weak labels carefully. LLM-as-judge labels are useful, but document the judge, prompt, calibration, and whether the run is external.

Treat logprobs as features, not truth. Confidence can help ranking, but should be combined with evidence-grounded quality signals.

Report uncertainty. With small judged sets, use bootstrap confidence intervals or paired significance tests.

Keep the method interpretable. For model routing, it is valuable to explain why a system selected a model: similar past answers, domain match, high-confidence grounded responses, or judge-derived quality.

## Primary Sources

- Official TREC 2025 Million LLMs overview paper: https://trec.nist.gov/pubs/trec34/papers/Overview_mllm.pdf
- Archived 2025 track page: https://trec-mllm.github.io/2025.html
- TREC 2025 proceedings index: https://trec.nist.gov/pubs/trec34/index.html
- Official run appendix: https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main.html
- GenAIus participant paper: https://trec.nist.gov/pubs/trec34/papers/GenAIus.mllm.pdf
- SRCB participant paper: https://trec.nist.gov/pubs/trec34/papers/SRCB.mllm.tot.pdf
- UDInfo participant paper: https://trec.nist.gov/pubs/trec34/papers/UDInfo.mllm.pdf
- UvA IRLab participant paper: https://trec.nist.gov/pubs/trec34/papers/uvairlab.mllm.pdf
- Paul Hildebrandt behavioral profiles paper, saved in the local source cache: `.codex_sources/mllm2025/paulphys.behavioral_profiles.pdf`

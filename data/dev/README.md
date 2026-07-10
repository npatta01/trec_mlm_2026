# TREC 2026 Million LLMs Development Data

Source: https://trec-mllm.github.io/

Captured: 2026-07-08

## Task

The TREC 2026 Million LLMs Track asks systems to rank candidate LLM IDs by predicted
expertise for each query. The system should decide which LLMs are most likely to provide
a high-quality answer before seeing any answer to the target query.

Participants are given discovery queries, precomputed LLM responses, and response
metadata. That evidence is used to build ranking systems for development and held-out
test queries.

## Official Files

| File | Split | Use |
|---|---|---|
| `llm_discovery_data.json` | Discovery | Discovery queries plus precomputed responses from all LLMs. Use this as the main source for response indexes, LLM profiles, embeddings, lexical retrieval features, and weak labels. |
| `llm_discovery_metadata.json` | Discovery | Same core discovery data, with logprobs and response metadata. Use this for confidence, uncertainty, answer-length, and token-probability features. |
| `llm_dev_data.tsv` | Development | Development query IDs and query text. Rank the candidate LLM IDs for each query. |
| `llm_dev_qrels.txt` | Development | Expertise labels for development query-LLM pairs. Use for validation, model selection, and metric checks. |
| Test query file | Test | Not released on the official page as of capture date. The page lists 2026-09-05 as the test query release date. |

The exact official links, Google Drive file IDs, and byte sizes are in
`file_manifest.tsv`.

## Suggested Local Layout

Download the official files into this directory when needed:

```text
data/dev/
  README.md
  file_manifest.tsv
  llm_discovery_data.json
  llm_discovery_metadata.json
  llm_dev_data.tsv
  llm_dev_qrels.txt
  llm_test_data.tsv      # add when released
```

The development query file and qrels are tracked because they are about 40 KB and
2.2 MB, respectively. The discovery files remain local because they are about 137 MB
and 1.46 GB. The unreleased test file is ignored until its size can be reviewed.

## Important 2026 Constraint

The official page notes that the 2026 LLM set is different from the 2025 LLM set. Do not
transfer model-ID expertise from 2025 to 2026. Only transfer methods, feature ideas, and
modeling approaches.

## Submission Format

Submissions use classic six-column TREC run format:

```text
topic_id Q0 llm_id rank score run_id
```

Example:

```text
1 Q0 llmid1 1 2.73 runid1
1 Q0 llmid2 2 2.71 runid1
1 Q0 llmid3 3 2.61 runid1
```

Column meanings:

- `topic_id`: query/topic number.
- `Q0`: unused placeholder; always `Q0`.
- `llm_id`: official ranked LLM identifier.
- `rank`: rank position for this LLM.
- `score`: score used to produce the ranking. Scores must be descending or tied in rank order.
- `run_id`: identifier for the submitted run.

Each team may submit up to five official runs.

## Run Types

Automatic runs have no manual intervention after seeing the test queries. Manual runs
include human-in-the-loop or manual changes after seeing test queries.

Internal runs use only the released discovery data to assess LLM expertise. External
runs use outside tools or data, such as LLM judges, search engines, or external corpora.

## Evaluation

Official metrics listed on the page:

- `nDCG@10`: discounted gain over relevance grades.
- `MRR`: mean reciprocal rank.

Evaluation uses hidden ground-truth labels. Results are expected in aggregate and by
query category.

## Timeline

| Date | Event |
|---|---|
| 2026-07-07 | Expertise discovery and development data released |
| 2026-09-05 | Test queries released |
| 2026-09-21 | Submission deadline |
| 2026-10 | Evaluation and results shared |
| 2026-11 | TREC conference |

The page also lists "September 2026" as the submission deadline in the submission
section, while the timeline gives the specific date 2026-09-21.

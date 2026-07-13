# GroupRank 7B Stage 2 Pilot Design

## Goal

Test whether a purpose-trained comparative reranker can map questions to expert
topic labels better than the captured embedding and pointwise baselines, without
the memory cost of a 32B model.

This is a small qrel-assisted feasibility diagnostic. It supersedes the
fixed-key Qwen listwise diagnostic as the next experiment, but it does not alter
or overwrite any earlier result.

## Frozen Scope

- Model: `AQ-MedAI/Diver-GroupRank-7B`
- Revision: `42a9504b32616f155970d5b7f349b63dc24b3e27`
- License: Apache 2.0
- Weights: 7,615,616,512 BF16 parameters and a 15.2 GB Hub download
- Runtime:
  `vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`
- Context limit: 8,192 tokens for this pilot
- Questions and candidates: the existing twelve-question, twenty-candidate
  Stage 2 manifest at frozen selection SHA-256
  `668515c30a80473aaaa2dc530f50dfd3b7731711e9f9adf70074ebb95aa98b9d`
- Candidate representation: candidate ID, cleaned expert label, and the existing
  frozen capability sentence only
- No qrel text, expert responses, discovery profiles, few-shot examples, fusion,
  or fallback rankings

The Stage 2 selection is reused unchanged so the new result is directly
comparable with contextual E5, contextual Qwen4B, and pointwise Qwen.

## Groupwise Ranking Method

Adapt GroupRank's official passage-ranking prompt only where needed to describe
LLM experts. For each question, the prompt asks which described experts are most
likely to answer the question correctly and requests an integer score from 0 to
10 for every presented handle. The response protocol is the checkpoint's strict
two-part envelope: exactly one `<reason>...</reason>` block followed by exactly
one `<answer>...</answer>` block, with nothing outside them. The answer contains
either a bare JSON object or that object inside one optional `json` code fence.
The prompt contains an output-format skeleton, but no semantically scored expert
example.

Use the model's official groupwise structure:

1. Deterministically shuffle the twenty candidates with seed 666.
2. Split them into two disjoint groups of ten.
3. Repeat with seed 667.
4. Run four groupwise requests per question, forty-eight requests total.
5. Average the two scores received by each candidate.
6. Rank by descending mean score and then candidate ID for exact ties.

Each candidate appears exactly once in each repetition. Store both presentation
orders so position and comparison context are auditable.

## Runtime and Host Safety

Before any model download or inference:

- verify `/dev/kfd` and `/dev/dri` access;
- check available host memory, swap, memory and I/O pressure, and running
  containers;
- defer the run if another GPU workload or material host pressure is present;
- perform a minimal ROCm/vLLM smoke test before the forty-eight-request pilot;
- retain the validated repository, Hugging Face cache, device, group, and IPC
  mounts; and
- use `gpu_memory_utilization=0.40` rather than allocating an oversized KV
  cache.

Use the model's official sampling shape with `temperature=0.3` and `top_p=0.8`,
but cap each short expert-card generation at 2,048 new tokens. Derive a stable
per-request sampling seed from seed 2026, query ID, repetition, and group. A
length-truncated response is a failed request.

Record the exact model revision, container digest, runtime versions, hardware,
memory settings, prompt hash, schema hash, and timing in the artifact.

## Validation and Failure Handling

Every raw generation is persisted before semantic parsing. Parsing must detect
duplicate JSON keys and require exactly the ten expected handles, with one
integer score from 0 through 10 per handle. Missing, extra, duplicated,
non-integer, boolean, or out-of-range values make that request unscorable.

The first real smoke receipt at
`artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/smoke.json` is a preserved
protocol-failure artifact: the model emitted one reasoning `<answer>` followed
by a second JSON `<answer>`. Its SHA-256 is
`7a11c925bd3728fc17595849ab3291409be273a91ec1ee097c4bee662314f3d5`, and it
remains immutable. After aligning the prompt and strict parser to the official
envelope, one explicit operator retry may write `smoke_protocol_v2.json`. This
is not an automatic runtime retry, and the rank phase may proceed only by
validating that distinct passed receipt.

That one retry was executed and failed. The preserved
`smoke_protocol_v2.json` SHA-256 is
`648a8f400d88cf0e6d3a99f94f4edbfc32ce3b1ba83b010d7e95b00bd3ba7493`.
This exact model-specific pilot is therefore terminal: its CLI rejects every
future smoke or rank invocation before input validation or engine loading,
regardless of paths. Any further format study must use a separately named
experiment and artifact namespace. Post-hoc evaluation retains its sealed
ranking-validation-before-qrel boundary.

Do not repair semantic failures, substitute zeros, or synthesize fallback
rankings. If any request fails, persist all raw outputs and a terminal failure
receipt, then stop before opening qrels.

Before evaluation, validate:

- forty-eight complete request records;
- two scores per candidate for all twelve questions;
- twelve deterministic twenty-candidate rankings;
- prompt and rankings independence from qrel mutation; and
- repeatable ranking construction from persisted raw scores.

## Artifacts and Evaluation

Write a separate qrel-free artifact directory under:

`artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/`

It contains the frozen selection, prompts, candidate orders, schemas, raw
generations, parsed scores, rankings, runtime metadata, hashes, and a ranking
receipt. Existing Stage 1, Stage 2, and failed-listwise artifacts remain
byte-for-byte unchanged.

Only after the qrel-free ranking receipt validates, open the sealed judgments
and report:

- nDCG@10;
- MRR;
- relevant-label top-three count;
- per-question relevant ranks and top-three predictions;
- score ties and agreement between the two repetitions; and
- comparisons with contextual E5, contextual Qwen4B, and pointwise Qwen on the
  same restricted candidate sets.

The captured baselines are:

| Method | Relevant in top 3 | nDCG@10 | MRR |
| --- | ---: | ---: | ---: |
| Contextual E5 | 8/12 | 0.7232 | 0.6361 |
| Contextual Qwen4B | 8/12 | 0.6648 | 0.5585 |
| Pointwise Qwen | 5/12 | 0.4820 | 0.3209 |

The exploratory pilot passes only if all forty-eight outputs validate, at least
eight of twelve questions place a relevant label in the top three, and overall
nDCG@10 is at least 0.7232.

No full sweep, fusion, deployment choice, or 32B download starts automatically.
Discuss the result first.

## Tests

Use test-driven development for:

- exact expert-oriented GroupRank prompt formatting;
- deterministic two-repetition group construction and complete coverage;
- strict fixed-key parsing, including duplicate-key detection;
- score averaging and deterministic tie behavior;
- raw-output persistence before parse failure;
- qrel mutation independence;
- exact twelve-query, twenty-candidate, forty-eight-request inventory;
- runtime and model provenance validation; and
- deterministic JSON and Markdown result writing.

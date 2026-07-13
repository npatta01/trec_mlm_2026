# Small diverse label-embedding sweep design

## Goal

Measure whether smaller and architecturally diverse embedding models can map each
question directly to an expert topic label. This is an exploratory follow-up to the
corrected large-model sweep. It intentionally overrides the original Stage 2 stop for
evidence gathering, but it does not change the frozen deployment gate or select a
model automatically.

## Frozen candidates

The sweep executes these exact revisions:

| Key | Checkpoint revision | Family | License | Output |
|---|---|---|---|---|
| `qwen3_embedding_0_6b` | `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | decoder, last-token | Apache-2.0 | 1024 |
| `qwen3_embedding_4b` | `Qwen/Qwen3-Embedding-4B@5cf2132abc99cad020ac570b19d031efec650f2b` | decoder, last-token | Apache-2.0 | 2560 |
| `jina_v5_small_retrieval` | `jinaai/jina-embeddings-v5-text-small-retrieval@6856e76bb72982e58de0620458a4e8b3614da340` | retrieval decoder, last-token | CC-BY-NC-4.0 | 1024 |
| `bge_m3` | `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181` | XLM-R encoder, dense CLS | MIT | 1024 |
| `nomic_embed_text_v1_5` | `nomic-ai/nomic-embed-text-v1.5@e9b6763023c676ca8431644204f50c2b100d9aab` | NomicBERT encoder, mean | Apache-2.0 | 768 |

BGE-M3 and Nomic are included as diversity controls, not because they accept the
custom natural-language instruction. BGE-M3 officially uses no query instruction.
Nomic uses its official `search_query:` and `search_document:` task prefixes.

## Inputs and representations

All conditions use the same 322 questions and 150 candidate IDs. Candidate IDs are
cleaned by replacing underscores and parentheses with spaces and collapsing
whitespace. Each model runs both document representations:

1. `bare`: `{cleaned_label}`
2. `expert_capability`: `This LLM is most likely to correctly answer questions about
   {cleaned_label}.`

Qwen retains the frozen query instruction:

`Given a user question, retrieve topic labels describing the LLM experts most likely
to answer it correctly.`

Qwen inputs use the pinned checkpoint's explicit terminal `<|endoftext|>` token. Jina
uses `Query: {question}` and `Document: {label_document}`. BGE-M3 receives the raw
question and label document without a prefix. Nomic uses `search_query: {question}`
and `search_document: {label_document}`.

## Inference contracts

Every small/diverse embedding is generated in FP32 because independent BF16 loads on
this gfx1151 ROCm host failed reference parity and repeatability. Embeddings remain
L2-normalized before cosine similarity. Qwen and Jina use the pinned vLLM ROCm image. BGE-M3 and Nomic use their
official Transformers-compatible pooling inside the same image: normalized dense CLS
for BGE-M3 and attention-mask mean pooling followed by the official layer-normalize
and L2-normalize sequence for full-dimensional Nomic output.

Before a full checkpoint run, each new backend must pass a two-query/two-document
canonical smoke test. The test checks exact formatting and token IDs, finite normalized
embeddings with the registered dimension, and agreement with the checkpoint's official
reference implementation within `1e-4` cosine score. A model that fails parity is
recorded as blocked and does not proceed to the full sweep.

## Execution and artifacts

Each of the ten model/representation conditions runs in two fresh processes, producing
20 runs when all smoke tests pass. Each run records:

- resolved checkpoint, revision, license, runtime, container, software, and hardware;
- exact query/document formatting, tokenization, pooling, normalization, and truncation;
- query and label embeddings;
- all 48,300 query-candidate scores;
- a complete 322 by 150 TREC ranking and per-query metrics;
- hashes, elapsed time, peak memory where available, and validation results.

Exact score ties use candidate ID. Qrels remain unavailable until a run's embeddings,
scores, and rankings are fixed. A qrel-mutation test must demonstrate that changing
qrels cannot affect those upstream artifacts.

## Comparisons and decision receipt

The combined receipt reports every condition against:

1. the formal MPNet bare-label control;
2. corrected E5-Mistral contextual labels, the strongest valid large-model condition;
3. the same model's bare-label condition when evaluating contextualization.

For each paired comparison, report the 10,000-resample 95% nDCG@10 and MRR intervals,
win/tie/loss counts, zero-query transitions, qrel-density slices, question-type slices,
and the issue #11 examples. Also report quality versus parameter count, wall time, and
peak memory so a smaller model can be preferred when quality is practically tied.

The original advancement threshold is shown for continuity, but these runs are labeled
post-gate exploratory evidence. No deployment selection, fusion, generated expert
description, fine-tuning, or LLM ranker is started automatically.

## Repeatability and failure handling

The two fresh-process runs for a condition must have identical rankings and metrics,
with every score agreeing within `1e-5`. A non-repeatable condition remains in the
report but is ineligible for recommendation. Runtime failures preserve logs and model
metadata, do not invalidate completed conditions, and do not trigger a substitute
checkpoint without explicit approval.

## Verification

- Unit tests freeze registry metadata, every formatter, pooling, normalization,
  deterministic ranking, tie behavior, and CLI options.
- Canonical smoke fixtures fail before unsupported or incorrect backends can launch a
  full run.
- Every completed run validates 48,300 finite scores, 322 rankings, and all 150 unique
  candidates per ranking.
- The combined receipt is regenerated only from validated run artifacts.
- Tests, lint, formatting, artifact validation, and the final receipt audit must pass
  before results are described as complete.

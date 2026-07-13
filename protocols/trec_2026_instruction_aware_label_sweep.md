# Staged instruction-aware expert-label model sweep

## Question

Can an instruction-aware text encoder map a user question directly to the cleaned
topic labels of the LLM experts most likely to answer it correctly?

This is a frozen zero-shot label-retrieval experiment. It excludes qrel-trained
representations, discovery responses, fusion, and expert exemplars. Qrels are not
loaded until every score and ranking for a run has been fixed.

## Frozen representation

Candidate IDs are converted to label documents by replacing underscores and
parentheses with spaces and collapsing whitespace. Every encoder ranks the same 150
labels for the same 322 questions with cosine similarity. Exact score ties are broken
by candidate ID.

Instruction-aware encoders use:

> Given a user question, retrieve topic labels describing the LLM experts most likely
> to answer it correctly.

Qwen queries are `Instruct: {instruction}\nQuery:{question}`. E5-Mistral queries are
`Instruct: {instruction}\nQuery: {question}`. The default label document is the
cleaned label alone. The diagnostic `expert_capability` representation is the fixed
sentence `This LLM is most likely to correctly answer questions about
{cleaned_label}.` Jina uses `Query: {question}` and `Document: {cleaned_label}`.

The pinned Qwen revision requires an explicit terminal `<|endoftext|>` token. Token-ID
prompts are therefore passed to vLLM instead of relying on its generic embedding
template. The pinned E5 revision expects the legacy tokenizer's terminal `</s>` token;
the Transformers runtime constructs it explicitly so tokenizer-version migration
cannot silently omit it. Both paths reserve terminal-token space before truncation.

The complete registry, including exact model revisions, licenses, dimensions, pooling,
runtime, and stage, is in `trec_mlm/label_retrieval.py`.

## Stages and gate

Stage 1 runs Qwen3-Embedding-8B and E5-Mistral-7B-Instruct. A model advances only if
its paired 10,000-resample 95% nDCG@10 interval against the MPNet raw-label control is
entirely above zero and it has no more than 131 zero-nDCG queries.

If Qwen 8B advances, Stage 2 runs Qwen 4B and Qwen 0.6B. If either large model
advances, Stage 2 also runs Jina Embeddings v5 Text Small Retrieval. If neither large
model advances, Stage 2 checkpoints are not downloaded or run.

Every executed encoder is run twice. Rankings and metrics must be identical and every
score must agree within `1e-5`.

## Runtime

GPU models run in the pinned image:

`vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`

Qwen and Jina use vLLM pooling. E5 uses the model card's Transformers last-token
pooling implementation inside the same image. Docker preserves `/dev/kfd`, `/dev/dri`,
the video group, shared IPC, the Hugging Face cache, and the repository mount.

## Commands

Reproduce the formal MPNet control:

```bash
uv run --extra dense trec-mlm label-retrieval \
  --encoder mpnet_label_control \
  --output-dir artifacts/label_sweep/mpnet_label_control
```

Run one frozen encoder in the configured environment:

```bash
trec-mlm label-retrieval \
  --encoder qwen3_embedding_8b \
  --output-dir artifacts/label_sweep/qwen3_embedding_8b/run1
```

After completed runs, produce the gate decision and combined receipt:

```bash
trec-mlm label-sweep-summary \
  --baseline-dir artifacts/label_sweep/mpnet_label_control \
  --model-result qwen3_embedding_8b=artifacts/label_sweep/qwen3_embedding_8b/run1 \
  --model-result e5_mistral_7b_instruct=artifacts/label_sweep/e5_mistral_7b_instruct/run1 \
  --output-dir artifacts/label_sweep/combined
```

## Interpretation

The sweep was designed after inspecting MPNet results. All comparisons are therefore
exploratory development evidence. The sweep does not automatically select a deployment
model or authorize fusion; results must be discussed first.

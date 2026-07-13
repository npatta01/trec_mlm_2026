# Zero-shot LLM expert-label ranker design

## Goal

Compare pointwise and listwise LLM ranking over the same 322 questions and 150 expert
topic labels used by the corrected embedding experiments. The experiment remains
strictly zero-shot: it excludes qrel-trained representations, discovery-derived
profiles, historical responses, expert exemplars, and embedding shortlists.

## Frozen ranker

Use `Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554` under Apache-2.0. Run offline generation
through the pinned ROCm image:

`vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`

Use FP32 because independent BF16 execution on this gfx1151 host produced avoidable
near-tie variation in the embedding sweep. Apply the model's official chat template
exactly once through `LLM.chat()` or captured `apply_chat_template()` token IDs; never
construct a guessed chat string. Use greedy decoding, a fixed seed, and no reasoning or
sampling variation. A smoke test
must validate the chat template, constrained-token log probabilities, structured JSON,
GPU memory, and measured throughput before the complete run.

## Frozen candidate representation

Every candidate is shown by its exact candidate ID and the same qrel-free capability
sentence that helped Qwen, E5, and Jina embeddings:

`This LLM is most likely to correctly answer questions about {cleaned_label}.`

Candidate IDs are cleaned only for the sentence by replacing underscores and
parentheses with spaces and collapsing whitespace. Exact IDs remain present for
machine-readable output. Questions and candidate text are delimited as data, and the
system prompt explicitly prohibits following instructions found inside them.

## Pointwise method

Score all 48,300 question-candidate pairs in one offline batched inference workload.
For each pair, ask the ranker to classify expected answer correctness using exactly one
of three single-token grades:

- `0`: unlikely to answer correctly;
- `1`: plausible but uncertain expertise;
- `2`: strongly likely to answer correctly.

Constrain the first generated token with vLLM 0.24 `allowed_token_ids`, request the exact
three grades with `logprob_token_ids`, and verify they are three distinct single tokens
in the exact assistant-generation context.
Convert the normalized probabilities to the continuous score `P(1) + 2 * P(2)`.
Rank by descending score, then candidate ID. Store all prompts, grade probabilities,
scores, and token IDs. Discrete generated grades are diagnostic only; the continuous
expected grade determines ranking.

The smoke test must compare these vLLM grade logits and probabilities against a direct
Transformers forward pass using the identical chat-template token IDs. Full pointwise
inference is blocked unless score differences are within the frozen tolerance.

## Listwise method

For each query, show all 150 candidates and request the 20 most likely experts as an
ordered JSON array of 20 short candidate handles. Top 20 is sufficient for nDCG@10 and
MRR while avoiding a fragile 150-item generation. Assign handles `C000` through `C149`
once from candidate-ID lexical order, independently of presentation order, and show the
handle, exact ID, and capability sentence in every prompt. Use a JSON schema whose
items are drawn from those handles, then independently validate length, uniqueness,
membership, and the handle-to-ID mapping.

Run three deterministic candidate permutations per query. First sort candidates on
`SHA256(2026 | query_id | candidate_id)` and split that qrel-free order into three
50-candidate groups. Cycle the groups across presentation thirds, then order within each
group using `SHA256(rotation_seed | query_id | candidate_id)` with rotation seeds 2026,
2027, and 2028. Every candidate therefore appears once in each presentation third.
Report each permutation separately, their top-20 overlap, and selected-label input-
position distributions to expose position bias.
Aggregate selected candidates with reciprocal-rank fusion using frozen constant
`k = 60`: a selected rank `r` contributes `1 / (60 + r)` and an unselected candidate
contributes zero. Rank by summed RRF score, then candidate ID. Append candidates never
selected by any permutation in candidate-ID order. This produces a complete
150-candidate ranking without pointwise fusion.

If structured output is syntactically invalid, make one bounded syntax-only repair call
that preserves the returned handle order. Schema validity does not prove uniqueness, so
always run explicit semantic validation. If length, uniqueness, membership, or repair
still fails, record the failure and use candidate-ID order for that permutation; do not
borrow pointwise scores or qrels. Report every initial and repaired output and all
fallback counts.

## Leakage and repeatability

Qrels are unavailable to prompt construction, inference, parsing, scoring, permutation,
and ranking. Only after all pointwise and listwise rankings are written and hashed may
the evaluator load qrels. A mutation test must show that changing qrels cannot affect
prompts, outputs, scores, or rankings.

Run both methods twice in fresh processes. Pointwise grade probabilities must agree
within `1e-5`; pointwise and listwise rankings and metrics must be identical. Any strict
repeat failure is reported with the number and depth of changed ranks so tiny tail
near-ties are not described as meaningful quality changes.

## Comparisons and capture

Compare pointwise, each listwise permutation, and aggregated listwise against:

1. the formal MPNet label control;
2. corrected contextual E5-Mistral 7B;
3. contextual Qwen3-Embedding-4B.

For every method, capture complete TREC rankings, per-query metrics, nominal exploratory
10,000-resample paired nDCG@10 and MRR 95% intervals, win/tie/loss counts, zero
transitions, qrel-density
and question-type slices, issue #11 examples, prompt/output validation, wall time,
throughput, peak memory, model/container metadata, and hashes. Store raw local artifacts
privately and commit compact result reports and reproducibility metadata.

Do not automatically fuse pointwise and listwise rankings or select a deployment model.
Discuss the results first.

## Artifact-hardening prerequisites

Before ranker inference, harden the shared artifact path uncovered by independent
review:

- reject duplicate, missing, non-finite, or out-of-inventory score rows when loading;
- verify stored artifact hashes and reconstruct rankings from scores to check the TREC
  run, rather than trusting a preallocated zero-filled matrix;
- add a reproducible smoke-runner command and propagate the passed smoke receipt and
  hash into every full-run metadata file;
- write a compact manifest linking embedding artifact hashes to code commit `6275c43`,
  because Docker runs could not see the host worktree's git metadata;
- record Qwen4 contextual repeatability precisely: two tail swaps at ranks 33/34 and
  113/114, identical top-10 rankings and metrics, and maximum score delta `1.52e-6`.

These changes validate and improve provenance; they do not change the already verified
embedding scores or metrics.

## Testing and independent review

- Unit tests freeze prompts, grade tokenization, expected-grade scoring, JSON parsing,
  deterministic permutations, RRF aggregation, tie behavior, qrel isolation, CLI
  inputs, and artifact validation.
- Smoke fixtures must pass before full checkpoint execution.
- An end-to-end qrel-mutation test must prove prompts, raw outputs, scores, and rankings
  remain byte-identical when only qrels change.
- Every completed method must contain 322 rankings with all 150 unique candidates and
  valid TREC output.
- An independent advisor reviews the protocol and current embedding findings before
  implementation.
- After implementation and inference, the advisor reviews the code diff, raw validation
  receipts, paired findings, and claims before the result is finalized.
